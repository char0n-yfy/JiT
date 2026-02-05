import math
import os
import sys
import copy

import torch
import torchvision.utils as vutils

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

        # Avoid synchronizing every iteration (hurts throughput). Only sync on
        # print iterations so the logged timings/memory are more meaningful.
        if torch.cuda.is_available() and (data_iter_step % print_freq == 0 or data_iter_step == len(data_loader) - 1):
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

    # Prefer ImageFolder (expects root/<class_name>/*.jpg). If the split directory
    # contains images directly (e.g., ImageNet val unpacked without class folders),
    # fall back to a flat image dataset that recursively scans for image files.
    try:
        dataset_eval = datasets.ImageFolder(split_dir, transform=transform_eval)
    except FileNotFoundError as e:
        from torch.utils.data import Dataset
        from torchvision.datasets.folder import IMG_EXTENSIONS, default_loader, has_file_allowed_extension

        paths: list[str] = []
        for root, _dirs, files in os.walk(split_dir):
            for fname in files:
                if has_file_allowed_extension(fname, IMG_EXTENSIONS):
                    paths.append(os.path.join(root, fname))
        paths.sort()

        if len(paths) == 0:
            raise

        class FlatImageDataset(Dataset):
            def __init__(self, image_paths: list[str], transform=None):
                self.image_paths = image_paths
                self.transform = transform

            def __len__(self) -> int:
                return len(self.image_paths)

            def __getitem__(self, index: int):
                img = default_loader(self.image_paths[index])
                if self.transform is not None:
                    img = self.transform(img)
                return img, 0

        dataset_eval = FlatImageDataset(paths, transform=transform_eval)
        if misc.is_main_process():
            print(f"[denoise-eval] Warning: failed to build ImageFolder from {split_dir} ({e}); using flat image dataset (n={len(dataset_eval)}).")
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

    denoise_t_list = getattr(args, "denoise_t_list", None)
    if denoise_t_list is None:
        denoise_t_list = [float(getattr(args, "denoise_t", 0.5))]
    elif isinstance(denoise_t_list, (float, int)):
        denoise_t_list = [float(denoise_t_list)]
    else:
        denoise_t_list = [float(t) for t in denoise_t_list]

    denoise_t_list = [max(0.0, min(1.0, float(t))) for t in denoise_t_list]
    if len(denoise_t_list) == 0:
        denoise_t_list = [float(getattr(args, "denoise_t", 0.5))]
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
    num_t = len(denoise_t_list)
    mse_sum = [0.0] * num_t
    l1_sum = [0.0] * num_t
    psnr_sum = [0.0] * num_t
    n = 0

    wrote_vis = False

    for x_uint8, _labels in data_loader_eval:
        if n >= target_per_rank:
            break

        x = x_uint8.to(device, non_blocking=True).to(torch.float32).div_(255.0)
        x = x * 2.0 - 1.0

        bsz = x.size(0)
        if n + bsz > target_per_rank:
            x = x[: target_per_rank - n]
            bsz = x.size(0)

        e = torch.randn_like(x) * model_without_ddp.noise_scale
        x01 = (x + 1.0) * 0.5

        make_vis = (not wrote_vis) and misc.is_main_process() and bool(getattr(args, "output_dir", ""))
        if make_vis:
            vis_n = min(4, bsz)
            x_vis = x01[:vis_n].clamp(0, 1)
            vis_dir = os.path.join(args.output_dir, "vis")
            os.makedirs(vis_dir, exist_ok=True)
            vis_blocks = []

        for ti, denoise_t in enumerate(denoise_t_list):
            t = torch.full((bsz, 1, 1, 1), float(denoise_t), device=device)
            z = t * x + (1.0 - t) * e

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                if denoise_mode == "single":
                    x_hat = model_without_ddp.predict_x(z, t)
                elif denoise_mode == "ode":
                    x_hat = model_without_ddp.denoise(
                        z,
                        t_start=float(denoise_t),
                        t_end=1.0,
                        steps=denoise_steps,
                        method=model_without_ddp.method,
                    )
                else:
                    raise ValueError(f"Unknown denoise_mode: {denoise_mode}")

            # Metrics in [0, 1] space.
            xh01 = (x_hat + 1.0) * 0.5
            mse = (xh01 - x01).pow(2).mean(dim=(1, 2, 3))
            l1 = (xh01 - x01).abs().mean(dim=(1, 2, 3))
            psnr = (-10.0 * torch.log10(mse.clamp_min(1e-10))).to(torch.float32)

            mse_sum[ti] += mse.sum().item()
            l1_sum[ti] += l1.sum().item()
            psnr_sum[ti] += psnr.sum().item()

            if make_vis:
                z01 = ((z[:vis_n] + 1.0) * 0.5).clamp(0, 1)
                xh_vis = xh01[:vis_n].clamp(0, 1)

                grid_noisy = vutils.make_grid(z01, nrow=vis_n)
                grid_denoise = vutils.make_grid(xh_vis, nrow=vis_n)
                grid_clean = vutils.make_grid(x_vis, nrow=vis_n)

                # 垂直拼接：噪声/去噪/干净
                grid = torch.cat([grid_noisy, grid_denoise, grid_clean], dim=1)
                vis_blocks.append(grid)

                tag = f"denoise_examples_t{denoise_t:g}_{denoise_mode}"
                if log_writer is not None:
                    log_writer.add_image(tag, grid, epoch)
                vutils.save_image(grid, os.path.join(vis_dir, f"epoch_{epoch:04d}_t{denoise_t:g}.png"))

        n += bsz

        if make_vis:
            grid_multi = torch.cat(vis_blocks, dim=1)
            tag_multi = f"denoise_examples_multi_{denoise_mode}"
            if log_writer is not None:
                log_writer.add_image(tag_multi, grid_multi, epoch)
            vutils.save_image(grid_multi, os.path.join(vis_dir, f"epoch_{epoch:04d}_multi.png"))
            wrote_vis = True

    # all-reduce sums
    if misc.is_dist_avail_and_initialized():
        reduce_t = torch.tensor([*mse_sum, *l1_sum, *psnr_sum, float(n)], device=device)
        torch.distributed.all_reduce(reduce_t)
        reduce_list = reduce_t.tolist()
        mse_sum = reduce_list[0:num_t]
        l1_sum = reduce_list[num_t:2 * num_t]
        psnr_sum = reduce_list[2 * num_t:3 * num_t]
        n = reduce_list[3 * num_t]

    denom = max(1.0, n)
    mse_mean = [s / denom for s in mse_sum]
    l1_mean = [s / denom for s in l1_sum]
    psnr_mean = [s / denom for s in psnr_sum]

    if misc.is_main_process():
        if num_t == 1:
            t0 = denoise_t_list[0]
            print(f"[denoise-eval] split={split} mode={denoise_mode} t={t0:g} steps={denoise_steps} n={int(n)}")
            print(f"[denoise-eval] MSE={mse_mean[0]:.6e} L1={l1_mean[0]:.6e} PSNR={psnr_mean[0]:.4f}")
        else:
            print(f"[denoise-eval] split={split} mode={denoise_mode} steps={denoise_steps} n={int(n)} t_list={denoise_t_list}")
            for t_val, mse_m, l1_m, psnr_m in zip(denoise_t_list, mse_mean, l1_mean, psnr_mean):
                print(f"[denoise-eval] t={t_val:g} MSE={mse_m:.6e} L1={l1_m:.6e} PSNR={psnr_m:.4f}")

    if log_writer is not None and misc.is_main_process():
        for t_val, mse_m, l1_m, psnr_m in zip(denoise_t_list, mse_mean, l1_mean, psnr_mean):
            tag = f"denoise_t{t_val:g}_{denoise_mode}"
            log_writer.add_scalar(f"{tag}/mse", mse_m, epoch)
            log_writer.add_scalar(f"{tag}/l1", l1_m, epoch)
            log_writer.add_scalar(f"{tag}/psnr", psnr_m, epoch)

    # back to no ema
    model_without_ddp.load_state_dict(model_state_dict)
    if misc.is_dist_avail_and_initialized():
        torch.distributed.barrier()
