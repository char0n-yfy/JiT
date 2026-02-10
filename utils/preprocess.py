#!/usr/bin/env python3
"""Shared preprocessing helpers for watermark attacks.

Standardizes:
- Square alignment with aspect‑ratio preservation (fit/pad).
- Downsampling to a common attack resolution (default 256×256).
- Optional upsampling back to a target evaluation/native resolution.

All tensors are in [0, 1] with shape [B, 3, H, W].
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision import transforms


_TO_TENSOR = transforms.ToTensor()


def square_image(
    img: Image.Image,
    size: int = 400,
    mode: str = "fit",
) -> Image.Image:
    """Resize to square while preserving aspect ratio.

    mode:
      - 'fit': center crop after proportional resize (like test_attack_on_images).
      - 'pad': proportional resize then pad to target size.
    """
    mode = (mode or "fit").lower()
    resample = Image.Resampling.BICUBIC
    if mode == "pad":
        return ImageOps.pad(img, (size, size), method=resample)
    return ImageOps.fit(img, (size, size), method=resample)


def pil_to_tensor(
    img: Image.Image,
    device: torch.device | str,
) -> torch.Tensor:
    """Convert PIL image to BCHW float tensor in [0, 1]."""
    t = _TO_TENSOR(img).unsqueeze(0)
    return t.to(device)


def resize_bicubic(
    x: torch.Tensor,
    size: Tuple[int, int],
) -> torch.Tensor:
    """High‑quality bicubic resize with antialias when available."""
    try:
        return F.interpolate(x, size=size, mode="bicubic", align_corners=False, antialias=True)
    except TypeError:
        # Older PyTorch without antialias flag
        return F.interpolate(x, size=size, mode="bicubic", align_corners=False)


def prepare_attack_tensors(
    clean_pil: Image.Image,
    watermarked_pil: Image.Image,
    device: torch.device | str,
    attack_size: int = 256,
    eval_size: int = 256,
) -> Dict[str, torch.Tensor]:
    """Prepare attack-size tensors and eval-size tensors from PIL images.

    Returns dict with:
      - 'clean_eval'       : [1,3,eval,eval]   clean image
      - 'wm_eval'          : [1,3,eval,eval]   watermarked image
      - 'clean_attack'     : [1,3,attack,attack] clean downsampled
      - 'wm_attack'        : [1,3,attack,attack] watermarked downsampled
    """
    dev = torch.device(device)
    clean_eval = pil_to_tensor(clean_pil, dev)
    wm_eval = pil_to_tensor(watermarked_pil, dev)

    clean_attack = resize_bicubic(clean_eval, size=(attack_size, attack_size))
    wm_attack = resize_bicubic(wm_eval, size=(attack_size, attack_size))

    return {
        "clean_eval": clean_eval,
        "wm_eval": wm_eval,
        "clean_attack": clean_attack,
        "wm_attack": wm_attack,
    }


__all__ = [
    "square_image",
    "pil_to_tensor",
    "resize_bicubic",
    "prepare_attack_tensors",
]
