#!/usr/bin/env python3
"""Top-level package for shared watermark utilities.

This package provides:
  - ``utils.metrics``    : PSNR/SSIM/LPIPS/BER helpers (no extra deps like pytorch_msssim).
  - ``utils.preprocess`` : 400→256 preprocessing and resize helpers.

Having this file ensures that ``import utils.metrics`` resolves to this
package instead of the single-file ``utils.py`` inside WatermarkAttacker-main.
"""

from .metrics import (  # noqa: F401
    compute_psnr,
    compute_ssim,
    compute_lpips,
    compute_ber,
)
from .preprocess import (  # noqa: F401
    square_image,
    pil_to_tensor,
    resize_bicubic,
    prepare_attack_tensors,
)

__all__ = [
    "compute_psnr",
    "compute_ssim",
    "compute_lpips",
    "compute_ber",
    "square_image",
    "pil_to_tensor",
    "resize_bicubic",
    "prepare_attack_tensors",
]
