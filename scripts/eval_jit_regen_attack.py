#!/usr/bin/env python3
"""Evaluate JiT regeneration attack against multiple watermark schemes.

Pipeline per sample:
1) embed watermark on clean image
2) inject noise into watermarked image at time t
3) purify back to t=1 using JiT ODE denoiser
4) decode watermark + compute quality metrics

Outputs:
- per-image CSV
- grouped summary CSV
- full JSON report
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import sys
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
# Disable torch.compile for portable evaluation runs (CPU / limited toolchains).
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
import torch
from PIL import Image
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from denoiser import Denoiser
from utils.metrics import compute_ber, compute_lpips, compute_psnr, compute_ssim
from utils.preprocess import square_image
from utils.watermarkers import (
    DwtDctSvdWatermarker,
    HiDDeNWatermarker,
    RivaGANWatermarker,
    SSLLatentWatermarker,
    StegaStampWatermarker,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("JiT regeneration attack evaluation")

    parser.add_argument("--config", type=str, default="configs/jit_uncond_imagenet256.yaml")
    parser.add_argument("--ckpt_dir", type=str, default="output_dir/jit_uncond_imagenet256/checkpoints")
    parser.add_argument("--checkpoints", type=str, nargs="+", default=None,
                        help="Optional explicit checkpoint paths. Defaults to all *.pth in --ckpt_dir")
    parser.add_argument("--test_dir", type=str, default="test")
    parser.add_argument("--output_dir", type=str, default="output_dir/jit_regen_attack_eval")

    parser.add_argument("--algorithms", type=str, nargs="+",
                        default=["stega", "dwt", "rivagan", "ssl", "hidden"],
                        choices=["stega", "dwt", "rivagan", "ssl", "hidden"])
    parser.add_argument("--noise_t_list", type=float, nargs="+", default=[0.2, 0.4, 0.6, 0.8])
    parser.add_argument("--denoise_steps", type=int, default=None,
                        help="Override ODE steps; default uses config num_sampling_steps")
    parser.add_argument("--sampling_method", type=str, default=None, choices=[None, "euler", "heun"])
    parser.add_argument("--square_mode", type=str, default="fit", choices=["fit", "pad"])
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--save_images", action="store_true")
    parser.add_argument("--log_every", type=int, default=10,
                        help="Print progress every N eval jobs.")

    parser.add_argument("--no_lpips", action="store_true", help="Disable LPIPS metric")
    parser.add_argument("--autocast", action="store_true", help="Use bf16 autocast on CUDA denoising")
    parser.add_argument("--device", type=str, default=None, help="e.g. cuda, cpu, cuda:0")
    parser.add_argument("--use_ema1", action="store_true")
    parser.add_argument("--no_use_ema1", dest="use_ema1", action="store_false")
    parser.set_defaults(use_ema1=True)

    parser.add_argument("--stega_encoder_path", type=str,
                        default="stegastamp_pkg/weights/encoder_best_loss_0.005250_step_66185.pth")
    parser.add_argument("--stega_decoder_path", type=str,
                        default="stegastamp_pkg/weights/decoder_best_loss_0.005250_step_66185.pth")

    parser.add_argument("--dwt_payload_bits", type=int, default=32)

    parser.add_argument("--rivagan_payload_bits", type=int, default=32)
    parser.add_argument("--rivagan_threshold", type=float, default=0.52)

    parser.add_argument("--ssl_payload_bits", type=int, default=30)
    parser.add_argument("--ssl_model_path", type=str, default="ssl_watermarking-main/models/dino_r50_plus.pth")
    parser.add_argument("--ssl_normlayer_path", type=str, default="ssl_watermarking-main/normalayer/out2048_coco_orig.pth")
    parser.add_argument("--ssl_carrier_path", type=str, default="ssl_watermarking-main/seed/ssl_carrier_seed2025.pt")
    parser.add_argument("--ssl_epochs", type=int, default=10)
    parser.add_argument("--ssl_target_psnr", type=float, default=40.0)
    parser.add_argument("--ssl_lambda_w", type=float, default=5e4)
    parser.add_argument("--ssl_lambda_i", type=float, default=1.0)
    parser.add_argument("--ssl_lr", type=float, default=1e-2)

    parser.add_argument("--hidden_payload_bits", type=int, default=30)
    parser.add_argument("--hidden_checkpoint_path", type=str,
                        default="HiDDeN-master/experiments/jpeg-compression/checkpoints/epoch-300.pyt")

    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be YAML mapping, got {type(data).__name__}")
    return data


def build_model_args(cfg: dict[str, Any]) -> SimpleNamespace:
    defaults = {
        "model": "JiT-B/16",
        "img_size": 256,
        "attn_dropout": 0.0,
        "proj_dropout": 0.0,
        "gated_attn": False,
        "P_mean": -0.8,
        "P_std": 0.8,
        "t_eps": 5e-2,
        "noise_scale": 1.0,
        "ema_decay1": 0.9999,
        "ema_decay2": 0.9996,
        "sampling_method": "heun",
        "num_sampling_steps": 50,
    }
    merged = {k: cfg.get(k, v) for k, v in defaults.items()}
    return SimpleNamespace(**merged)


def resolve_checkpoints(ckpt_dir: Path, explicit: list[str] | None) -> list[Path]:
    if explicit:
        paths = [Path(p) for p in explicit]
    else:
        paths = sorted(ckpt_dir.glob("*.pth"))
    paths = [p for p in paths if p.is_file()]
    if not paths:
        raise FileNotFoundError("No checkpoints found to evaluate.")
    return paths


def load_torch_checkpoint_compat(path: Path, map_location: str = "cpu") -> Any:
    """Load checkpoints across torch versions with a safe-first fallback path."""
    try:
        return torch.load(str(path), map_location=map_location)
    except pickle.UnpicklingError as err:
        msg = str(err)
        if "Weights only load failed" not in msg:
            raise

        # First, try to keep weights-only behavior by allowlisting argparse.Namespace.
        safe_globals_ctx = getattr(torch.serialization, "safe_globals", None)
        if safe_globals_ctx is not None:
            try:
                with safe_globals_ctx([argparse.Namespace]):
                    return torch.load(str(path), map_location=map_location)
            except Exception:
                pass

        # Final fallback for trusted local checkpoints that include non-tensor metadata.
        print(f"[warn] weights-only load failed for {path.name}; retrying with weights_only=False")
        try:
            return torch.load(str(path), map_location=map_location, weights_only=False)
        except TypeError:
            # Older torch may not accept weights_only kwarg.
            return torch.load(str(path), map_location=map_location)


def build_watermarkers(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    wm: dict[str, Any] = {}
    algos = set(args.algorithms)

    if "stega" in algos:
        wm["stega"] = StegaStampWatermarker(
            encoder_path=root / args.stega_encoder_path,
            decoder_path=root / args.stega_decoder_path,
            payload_bits=100,
        )
    if "dwt" in algos:
        wm["dwt"] = DwtDctSvdWatermarker(payload_bits=args.dwt_payload_bits)
    if "rivagan" in algos:
        wm["rivagan"] = RivaGANWatermarker(
            payload_bits=args.rivagan_payload_bits,
            threshold=args.rivagan_threshold,
        )
    if "ssl" in algos:
        wm["ssl"] = SSLLatentWatermarker(
            payload_bits=args.ssl_payload_bits,
            model_path=root / args.ssl_model_path,
            normlayer_path=root / args.ssl_normlayer_path,
            carrier_path=root / args.ssl_carrier_path,
            epochs=args.ssl_epochs,
            target_psnr=args.ssl_target_psnr,
            lambda_w=args.ssl_lambda_w,
            lambda_i=args.ssl_lambda_i,
            lr=args.ssl_lr,
        )
    if "hidden" in algos:
        wm["hidden"] = HiDDeNWatermarker(
            payload_bits=args.hidden_payload_bits,
            checkpoint_path=root / args.hidden_checkpoint_path,
        )
    return wm


def pil_to_bchw01(img: Image.Image, device: torch.device) -> torch.Tensor:
    return transforms.ToTensor()(img).unsqueeze(0).to(device)


def tensor01_to_pil(x: torch.Tensor) -> Image.Image:
    return transforms.ToPILImage()(x.detach().cpu().clamp(0.0, 1.0).squeeze(0))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def choose_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    metric_keys = [
        "bitacc_before", "ber_before", "bitacc_after", "ber_after", "bitacc_drop",
        "psnr_to_clean", "ssim_to_clean", "lpips_to_clean", "mse_to_clean", "l1_to_clean",
        "psnr_to_wm", "ssim_to_wm", "lpips_to_wm", "mse_to_wm", "l1_to_wm",
        "elapsed_sec",
    ]
    grouped: dict[tuple[str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        grouped[(r["checkpoint"], r["algorithm"], float(r["noise_t"]))].append(r)

    out: list[dict[str, Any]] = []
    for (ckpt, algo, noise_t), items in grouped.items():
        rec: dict[str, Any] = {
            "checkpoint": ckpt,
            "algorithm": algo,
            "noise_t": noise_t,
            "n": len(items),
        }
        for key in metric_keys:
            vals = [float(it[key]) for it in items if isinstance(it[key], (int, float)) and not math.isnan(float(it[key]))]
            rec[f"{key}_mean"] = float(np.mean(vals)) if vals else float("nan")
            rec[f"{key}_std"] = float(np.std(vals)) if vals else float("nan")
        out.append(rec)
    out.sort(key=lambda x: (x["checkpoint"], x["algorithm"], x["noise_t"]))
    return out


def main() -> None:
    args = parse_args()
    root = REPO_ROOT

    cfg = load_yaml(root / args.config)
    model_args = build_model_args(cfg)

    ckpts = resolve_checkpoints(root / args.ckpt_dir, args.checkpoints)
    noise_t_list = [float(max(1e-6, min(0.999, t))) for t in args.noise_t_list]

    device = choose_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    print(f"[info] using device={device}")

    watermarkers = build_watermarkers(args, root)
    if not watermarkers:
        raise RuntimeError("No watermark algorithms selected.")

    test_dir = root / args.test_dir
    image_paths = sorted([p for p in test_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}])
    if args.max_images is not None:
        image_paths = image_paths[: args.max_images]
    if not image_paths:
        raise RuntimeError(f"No images found under {test_dir}")

    run_dir = root / args.output_dir / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.save_images:
        (run_dir / "attacked_images").mkdir(parents=True, exist_ok=True)
        (run_dir / "watermarked_inputs").mkdir(parents=True, exist_ok=True)

    base_entries: list[dict[str, Any]] = []
    print(f"[prep] embedding watermarks on {len(image_paths)} images with algorithms={list(watermarkers.keys())}")
    for img_path in image_paths:
        clean = Image.open(img_path).convert("RGB")
        clean_eval = square_image(clean, size=int(model_args.img_size), mode=args.square_mode)
        clean_tensor_cpu = transforms.ToTensor()(clean_eval).unsqueeze(0)

        for algo, wm in watermarkers.items():
            secret = rng.integers(0, 2, size=wm.payload_bits, dtype=np.uint8)
            wm_native = wm.encode(clean_eval, secret)
            wm_eval = square_image(wm_native, size=int(model_args.img_size), mode=args.square_mode)
            wm_tensor_cpu = transforms.ToTensor()(wm_eval).unsqueeze(0)

            pred_before = wm.decode(wm_eval)[: wm.payload_bits]
            ber_before = compute_ber(secret, pred_before)
            bitacc_before = 1.0 - ber_before

            base_entries.append(
                {
                    "image_name": img_path.name,
                    "image_stem": img_path.stem,
                    "algorithm": algo,
                    "secret_bits": secret,
                    "clean_pil": clean_eval,
                    "wm_pil": wm_eval,
                    "clean_tensor_cpu": clean_tensor_cpu,
                    "wm_tensor_cpu": wm_tensor_cpu,
                    "bitacc_before": bitacc_before,
                    "ber_before": ber_before,
                }
            )

            if args.save_images:
                out_dir = run_dir / "watermarked_inputs" / algo
                out_dir.mkdir(parents=True, exist_ok=True)
                wm_eval.save(out_dir / f"{img_path.stem}.png")

    lpips_enabled = not args.no_lpips
    lpips_warned = False

    total_jobs = len(ckpts) * len(noise_t_list) * len(base_entries)
    job_idx = 0
    global_start = time.time()

    per_image_rows: list[dict[str, Any]] = []
    for ckpt_path in ckpts:
        print(f"[ckpt] loading {ckpt_path}")
        model = Denoiser(model_args).to(device)
        checkpoint = load_torch_checkpoint_compat(ckpt_path, map_location="cpu")

        if args.use_ema1 and isinstance(checkpoint, dict) and "model_ema1" in checkpoint:
            state_dict = checkpoint["model_ema1"]
            used_key = "model_ema1"
        elif isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"]
            used_key = "model"
        else:
            state_dict = checkpoint
            used_key = "raw_state_dict"

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(f"[warn] state mismatch for {ckpt_path.name}: missing={len(missing)} unexpected={len(unexpected)}")

        model.eval()
        steps = int(args.denoise_steps if args.denoise_steps is not None else model.steps)
        method = args.sampling_method if args.sampling_method is not None else model.method
        print(f"[ckpt] using state={used_key}, steps={steps}, method={method}")

        for t_noise in noise_t_list:
            print(f"[eval] checkpoint={ckpt_path.name} t={t_noise:.3f}")
            for item in base_entries:
                job_idx += 1
                clean01 = item["clean_tensor_cpu"].to(device)
                wm01 = item["wm_tensor_cpu"].to(device)
                secret = item["secret_bits"]
                wm_algo = watermarkers[item["algorithm"]]

                start = time.time()
                x_w = wm01 * 2.0 - 1.0
                t_tensor = torch.full((x_w.size(0), 1, 1, 1), t_noise, device=device, dtype=x_w.dtype)
                noise = torch.randn_like(x_w) * model.noise_scale
                z_t = t_tensor * x_w + (1.0 - t_tensor) * noise

                amp_ctx = (
                    torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(args.autocast and device.type == "cuda"))
                    if device.type == "cuda"
                    else nullcontext()
                )
                with torch.no_grad(), amp_ctx:
                    x_hat = model.denoise(z_t, t_start=t_noise, t_end=1.0, steps=steps, method=method)
                x_hat01 = ((x_hat + 1.0) * 0.5).clamp(0.0, 1.0)
                attacked_pil = tensor01_to_pil(x_hat01)

                pred_after = wm_algo.decode(attacked_pil)[: wm_algo.payload_bits]
                ber_after = compute_ber(secret, pred_after)
                bitacc_after = 1.0 - ber_after

                mse_to_clean = float(torch.mean((x_hat01 - clean01) ** 2).item())
                l1_to_clean = float(torch.mean(torch.abs(x_hat01 - clean01)).item())
                mse_to_wm = float(torch.mean((x_hat01 - wm01) ** 2).item())
                l1_to_wm = float(torch.mean(torch.abs(x_hat01 - wm01)).item())

                psnr_to_clean = float(compute_psnr(x_hat01, clean01).item())
                ssim_to_clean = float(compute_ssim(x_hat01, clean01).item())
                psnr_to_wm = float(compute_psnr(x_hat01, wm01).item())
                ssim_to_wm = float(compute_ssim(x_hat01, wm01).item())

                lpips_to_clean = float("nan")
                lpips_to_wm = float("nan")
                if lpips_enabled:
                    try:
                        lpips_to_clean = float(compute_lpips(x_hat01, clean01).item())
                        lpips_to_wm = float(compute_lpips(x_hat01, wm01).item())
                    except Exception as e:
                        lpips_enabled = False
                        if not lpips_warned:
                            print(f"[warn] LPIPS disabled due to runtime error: {e}")
                            lpips_warned = True

                elapsed_sec = float(time.time() - start)
                row = {
                    "checkpoint": ckpt_path.name,
                    "checkpoint_path": str(ckpt_path),
                    "algorithm": item["algorithm"],
                    "image_name": item["image_name"],
                    "noise_t": float(t_noise),
                    "bitacc_before": float(item["bitacc_before"]),
                    "ber_before": float(item["ber_before"]),
                    "bitacc_after": float(bitacc_after),
                    "ber_after": float(ber_after),
                    "bitacc_drop": float(item["bitacc_before"] - bitacc_after),
                    "psnr_to_clean": psnr_to_clean,
                    "ssim_to_clean": ssim_to_clean,
                    "lpips_to_clean": lpips_to_clean,
                    "mse_to_clean": mse_to_clean,
                    "l1_to_clean": l1_to_clean,
                    "psnr_to_wm": psnr_to_wm,
                    "ssim_to_wm": ssim_to_wm,
                    "lpips_to_wm": lpips_to_wm,
                    "mse_to_wm": mse_to_wm,
                    "l1_to_wm": l1_to_wm,
                    "elapsed_sec": elapsed_sec,
                }
                per_image_rows.append(row)

                if (job_idx == 1) or (job_idx % max(1, args.log_every) == 0) or (job_idx == total_jobs):
                    elapsed = time.time() - global_start
                    avg = elapsed / job_idx
                    remain = max(total_jobs - job_idx, 0) * avg
                    print(
                        f"[progress] {job_idx}/{total_jobs} "
                        f"avg={avg:.2f}s/job eta={remain/60.0:.1f}m "
                        f"algo={item['algorithm']} img={item['image_name']}"
                    )

                if args.save_images:
                    out_dir = run_dir / "attacked_images" / ckpt_path.stem / item["algorithm"] / f"t_{t_noise:.2f}"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    attacked_pil.save(out_dir / f"{item['image_stem']}.png")

        del model
        del checkpoint
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary_rows = aggregate_rows(per_image_rows)

    per_image_csv = run_dir / "per_image_metrics.csv"
    summary_csv = run_dir / "summary_metrics.csv"
    report_json = run_dir / "report.json"

    write_csv(per_image_csv, per_image_rows)
    write_csv(summary_csv, summary_rows)

    report = {
        "args": vars(args),
        "resolved": {
            "device": str(device),
            "config_path": str(root / args.config),
            "checkpoint_paths": [str(p) for p in ckpts],
            "noise_t_list": noise_t_list,
            "num_images": len(image_paths),
            "num_algorithms": len(watermarkers),
            "model_args": vars(model_args),
        },
        "summary": summary_rows,
    }
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"[done] per-image metrics: {per_image_csv}")
    print(f"[done] summary metrics : {summary_csv}")
    print(f"[done] report json     : {report_json}")


if __name__ == "__main__":
    main()
