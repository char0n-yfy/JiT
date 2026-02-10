#!/usr/bin/env python3
"""Utility functions for image/audio watermark evaluation metrics.

Provides lightweight helpers to compute PSNR, SSIM, LPIPS, and BER that can be
imported from other scripts. All functions assume inputs are torch.Tensor in
range [0, 1] and shape [B,C,H,W] unless stated otherwise.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

_HAS_TV_SSIM = False
try:
    from torchvision.metrics import structural_similarity_index_measure as _tv_ssim

    _HAS_TV_SSIM = True
except Exception:
    _HAS_TV_SSIM = False


def compute_psnr(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Peak Signal-to-Noise Ratio (dB) between two images."""
    mse = F.mse_loss(a, b, reduction="mean").clamp_min(eps)
    return 10.0 * torch.log10(1.0 / mse)


def compute_ssim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Structural Similarity Index. Uses torchvision if available, else a simple fallback."""
    if _HAS_TV_SSIM:
        # Returns a scalar tensor
        return _tv_ssim(a, b, data_range=1.0)

    # Fallback: crude luminance-contrast-structure approximation on batch
    mu_a = a.mean(dim=[1, 2, 3], keepdim=True)
    mu_b = b.mean(dim=[1, 2, 3], keepdim=True)
    sigma_a = ((a - mu_a) ** 2).mean(dim=[1, 2, 3], keepdim=True)
    sigma_b = ((b - mu_b) ** 2).mean(dim=[1, 2, 3], keepdim=True)
    sigma_ab = ((a - mu_a) * (b - mu_b)).mean(dim=[1, 2, 3], keepdim=True)
    C1 = 0.01**2
    C2 = 0.03**2
    ssim = ((2 * mu_a * mu_b + C1) * (2 * sigma_ab + C2)) / (
        (mu_a**2 + mu_b**2 + C1) * (sigma_a + sigma_b + C2)
    )
    return ssim.mean()  # batch average


def compute_lpips(a: torch.Tensor, b: torch.Tensor, net: str = "alex") -> torch.Tensor:
    """LPIPS distance. Requires `lpips` package. Returns mean over batch.

    To avoid repeatedly constructing the LPIPS network (which is expensive),
    we keep a small cache keyed by (device, net) and reuse the same model.
    """
    try:
        import lpips  # type: ignore
    except Exception as e:  # pragma: no cover - optional dep
        raise ImportError("lpips package is required for compute_lpips") from e

    if a.shape[1] != 3 or b.shape[1] != 3:
        raise ValueError("LPIPS expects 3-channel RGB tensors.")

    device = a.device
    key = (str(device), net)
    global _LPIPS_CACHE  # type: ignore[name-defined]
    try:
        cache = _LPIPS_CACHE  # type: ignore[name-defined]
    except NameError:
        cache = {}
        _LPIPS_CACHE = cache  # type: ignore[assignment]

    loss_fn = cache.get(key)
    if loss_fn is None:
        # lpips 内部目前仍使用 torchvision 的 `pretrained=True` 接口，
        # 会触发关于 `pretrained`/`weights` 的弃用警告，这里在构建
        # 模型时临时屏蔽这些噪声 warning，不影响实际行为。
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The parameter 'pretrained' is deprecated since 0.13",
                category=UserWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message="Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13",
                category=UserWarning,
            )
            loss_fn = lpips.LPIPS(net=net).to(device)
        loss_fn.eval()
        cache[key] = loss_fn

    with torch.no_grad():
        dist = loss_fn(a, b)  # shape [B,1,1,1] or [B,1,1]
    return dist.mean()


def compute_ber(bits_ref: np.ndarray, bits_pred: np.ndarray) -> float:
    """Bit Error Rate between two {0,1} vectors."""
    ref = np.asarray(bits_ref).astype(np.uint8).flatten()
    pred = np.asarray(bits_pred).astype(np.uint8).flatten()
    if ref.shape != pred.shape:
        raise ValueError(f"Bit shapes differ: {ref.shape} vs {pred.shape}")
    return float(np.mean(np.abs(ref.astype(np.float32) - pred.astype(np.float32))))


def compute_tpr_at_fpr(
    y_true: Sequence[int] | np.ndarray | torch.Tensor,
    y_score: Sequence[float] | np.ndarray | torch.Tensor,
    target_fpr: float = 0.01,
) -> float:
    """Compute TPR at a given FPR (default 1%).

    Args:
        y_true: Binary ground-truth labels (0 for negative, 1 for positive).
        y_score: Continuous scores (larger should mean more likely positive).
        target_fpr: Desired false-positive rate in [0, 1].

    Returns:
        TPR (recall) at FPR ~= target_fpr, obtained by interpolating the ROC
        curve. Raises ValueError if there are no positive or no negative
        examples.
    """
    if isinstance(y_true, torch.Tensor):
        y_true_np = y_true.detach().cpu().numpy()
    else:
        y_true_np = np.asarray(y_true)

    if isinstance(y_score, torch.Tensor):
        y_score_np = y_score.detach().cpu().numpy()
    else:
        y_score_np = np.asarray(y_score, dtype=np.float64)

    y_true_np = y_true_np.astype(np.int64).flatten()
    y_score_np = y_score_np.astype(np.float64).flatten()

    if y_true_np.shape != y_score_np.shape:
        raise ValueError(f"Shapes differ: labels {y_true_np.shape}, scores {y_score_np.shape}")

    n_pos = int(np.sum(y_true_np == 1))
    n_neg = int(np.sum(y_true_np == 0))
    if n_pos == 0 or n_neg == 0:
        raise ValueError(f"Need at least one positive and one negative (got {n_pos} pos, {n_neg} neg).")

    # Sort by score descending
    order = np.argsort(-y_score_np)
    y_true_sorted = y_true_np[order]

    tp = np.cumsum(y_true_sorted == 1).astype(np.float64)
    fp = np.cumsum(y_true_sorted == 0).astype(np.float64)

    tpr = tp / n_pos
    fpr = fp / n_neg

    # Ensure FPR is monotonically increasing (it should be by construction).
    # Interpolate TPR at the requested FPR.
    target_fpr_clamped = float(np.clip(target_fpr, 0.0, 1.0))
    # np.interp expects ascending x; fpr is ascending.
    tpr_at_target = float(np.interp(target_fpr_clamped, fpr, tpr))
    return tpr_at_target


def choose_tau_for_fpr(
    bitacc_clean: Sequence[float] | np.ndarray | torch.Tensor,
    bitacc_before: Sequence[float] | np.ndarray | torch.Tensor,
    target_fpr: float = 0.01,
) -> float:
    """Select a detection threshold tau s.t. FPR(clean) ~= target_fpr.

    Implements the“clean vs before” ROC方法：将 clean 作为负样本、before
    作为正样本，按分数从大到小扫描，一旦累计 FPR 达到目标值，就取当前
    分数作为 tau。
    """
    y_true = np.array([0] * len(bitacc_clean) + [1] * len(bitacc_before))
    y_score = np.concatenate([np.asarray(bitacc_clean, dtype=np.float64),
                              np.asarray(bitacc_before, dtype=np.float64)])

    order = np.argsort(-y_score)
    scores_sorted = y_score[order]
    labels_sorted = y_true[order]

    n_neg = int(np.sum(labels_sorted == 0))
    if n_neg == 0:
        raise ValueError("choose_tau_for_fpr: no negative samples provided.")

    fp = 0
    tau = scores_sorted[-1]  # fallback to the lowest score
    target_fpr_clamped = float(np.clip(target_fpr, 0.0, 1.0))

    for s, y in zip(scores_sorted, labels_sorted):
        if y == 0:
            fp += 1
        fpr = fp / n_neg
        if fpr >= target_fpr_clamped:
            tau = s
            break
    return float(tau)

__all__ = [
    "compute_psnr",
    "compute_ssim",
    "compute_lpips",
    "compute_ber",
    "compute_tpr_at_fpr",
    "choose_tau_for_fpr",
]
