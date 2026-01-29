import math
import os
import sys
import copy

import torch

import util.misc as misc
import util.lr_sched as lr_sched
from util.crop import center_crop_arr


def train_one_epoch(model, model_without_ddp, data_loader, optimizer, device, epoch, log_writer=None, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (x, _labels) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # per iteration (instead of per epoch) lr scheduler
        lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        # normalize image to [-1, 1]
        x = x.to(device, non_blocking=True).to(torch.float32).div_(255)
        x = x * 2.0 - 1.0

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            loss = model(x)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()

        model_without_ddp.update_ema()

        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)

        if log_writer is not None:
            # Use epoch_1000x as the x-axis in TensorBoard to calibrate curves.
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            if data_iter_step % args.log_freq == 0:
                log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
                log_writer.add_scalar('lr', lr, epoch_1000x)

@torch.no_grad()
def evaluate_denoise(model_without_ddp, args, epoch, batch_size=64, log_writer=None):
    """
    Evaluate denoising quality on real images by: x -> z_t -> x_hat, and report pixel metrics.
    This is not a perceptual benchmark; it is intended to sanity-check denoising behavior.
    """
    import math
    from torchvision import datasets, transforms

    model_without_ddp.eval()
    world_size = misc.get_world_size()
    local_rank = misc.get_rank()

    split = getattr(args, "denoise_split", "val")
    split_dir = os.path.join(args.data_path, split)
    if not os.path.isdir(split_dir):
        split = "train"
        split_dir = os.path.join(args.data_path, split)

    transform_eval = transforms.Compose([
        transforms.Lambda(lambda img: center_crop_arr(img, args.img_size)),
        transforms.PILToTensor(),
    ])

    dataset_eval = datasets.ImageFolder(split_dir, transform=transform_eval)
    sampler_eval = torch.utils.data.DistributedSampler(
        dataset_eval, num_replicas=world_size, rank=local_rank, shuffle=False, drop_last=False
    )
    data_loader_eval = torch.utils.data.DataLoader(
        dataset_eval,
        sampler=sampler_eval,
        batch_size=batch_size,
        num_workers=getattr(args, "num_workers", 4),
        pin_memory=True,
        drop_last=False,
    )

    # switch to ema params, hard-coded to be the first one
    model_state_dict = copy.deepcopy(model_without_ddp.state_dict())
    ema_state_dict = copy.deepcopy(model_without_ddp.state_dict())
    for i, (name, _value) in enumerate(model_without_ddp.named_parameters()):
        assert name in ema_state_dict
        ema_state_dict[name] = model_without_ddp.ema_params1[i]
    model_without_ddp.load_state_dict(ema_state_dict)

    denoise_t = float(getattr(args, "denoise_t", 0.5))
    denoise_t = max(0.0, min(1.0, denoise_t))
    denoise_mode = getattr(args, "denoise_mode", "ode")
    denoise_steps = getattr(args, "denoise_steps", None)
    if denoise_steps is None:
        denoise_steps = int(getattr(args, "num_sampling_steps", 50))
    else:
        denoise_steps = int(denoise_steps)
    denoise_steps = max(1, denoise_steps)

    target_total = int(getattr(args, "denoise_num_images", 2048))
    target_per_rank = int(math.ceil(target_total / world_size))

    device = torch.device("cuda")
    mse_sum = 0.0
    l1_sum = 0.0
    psnr_sum = 0.0
    n = 0

    for x_uint8, _labels in data_loader_eval:
        if n >= target_per_rank:
            break

        x = x_uint8.to(device, non_blocking=True).to(torch.float32).div_(255.0)
        x = x * 2.0 - 1.0

        bsz = x.size(0)
        if n + bsz > target_per_rank:
            x = x[: target_per_rank - n]
            bsz = x.size(0)

        t = torch.full((bsz, 1, 1, 1), denoise_t, device=device)
        e = torch.randn_like(x) * model_without_ddp.noise_scale
        z = t * x + (1.0 - t) * e

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            if denoise_mode == "single":
                x_hat = model_without_ddp.predict_x(z, t)
            elif denoise_mode == "ode":
                x_hat = model_without_ddp.denoise(z, t_start=denoise_t, t_end=1.0, steps=denoise_steps, method=model_without_ddp.method)
            else:
                raise ValueError(f"Unknown denoise_mode: {denoise_mode}")

        # Metrics in [0, 1] space.
        x01 = (x + 1.0) * 0.5
        xh01 = (x_hat + 1.0) * 0.5
        mse = (xh01 - x01).pow(2).mean(dim=(1, 2, 3))
        l1 = (xh01 - x01).abs().mean(dim=(1, 2, 3))
        psnr = (-10.0 * torch.log10(mse.clamp_min(1e-10))).to(torch.float32)

        mse_sum += mse.sum().item()
        l1_sum += l1.sum().item()
        psnr_sum += psnr.sum().item()
        n += bsz

    # all-reduce sums
    if misc.is_dist_avail_and_initialized():
        t = torch.tensor([mse_sum, l1_sum, psnr_sum, float(n)], device=device)
        torch.distributed.all_reduce(t)
        mse_sum, l1_sum, psnr_sum, n = t.tolist()

    mse_mean = mse_sum / max(1.0, n)
    l1_mean = l1_sum / max(1.0, n)
    psnr_mean = psnr_sum / max(1.0, n)

    if misc.is_main_process():
        print(f"[denoise-eval] split={split} mode={denoise_mode} t={denoise_t} steps={denoise_steps} n={int(n)}")
        print(f"[denoise-eval] MSE={mse_mean:.6e} L1={l1_mean:.6e} PSNR={psnr_mean:.4f}")

    if log_writer is not None and misc.is_main_process():
        tag = f"denoise_t{denoise_t:g}_{denoise_mode}"
        log_writer.add_scalar(f"{tag}/mse", mse_mean, epoch)
        log_writer.add_scalar(f"{tag}/l1", l1_mean, epoch)
        log_writer.add_scalar(f"{tag}/psnr", psnr_mean, epoch)

    # back to no ema
    model_without_ddp.load_state_dict(model_state_dict)
    if misc.is_dist_avail_and_initialized():
        torch.distributed.barrier()
