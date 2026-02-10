#!/usr/bin/env python3
"""Unified watermarker interface skeleton for multiple algorithms.

This module defines a common API that you can use in large‑scale experiments
to hide/restore messages with different watermarking schemes:

  - DwtDctSvdWatermarker     (DWT-DCT-SVD / Invisible Watermark)
  - RivaGANWatermarker       (RivaGAN / Invisible Watermark)
  - SSLLatentWatermarker     (ssl_watermarking latent-space watermark)
  - StegaStampWatermarker    (StegaStamp encoder/decoder)
  - HiDDeNWatermarker        (HiDDeN encoder/decoder)

StegaStamp / DWT-DCT-SVD / RivaGAN / SSL-Latent / HiDDeN are integrated
behind one consistent interface for embedding and decoding.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import random
import subprocess
import sys

import numpy as np
from PIL import Image
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F_vision
import importlib.util


@dataclass
class BaseWatermarker(ABC):
    """Abstract base class for all watermark schemes.

    The goal is to expose a consistent encode/decode API regardless of the
    underlying implementation.
    """

    name: str
    # Number of payload bits exposed through this interface (logical capacity).
    payload_bits: int
    # Native resolution for this scheme (H, W). None means "no strong assumption".
    native_size: Optional[Tuple[int, int]] = None

    @abstractmethod
    def encode(self, image: Image.Image, message_bits: np.ndarray) -> Image.Image:
        """Embed `message_bits` into `image` and return a watermarked image."""

    @abstractmethod
    def decode(self, image: Image.Image) -> np.ndarray:
        """Decode message bits from `image` and return a 1D np.ndarray of {0,1}."""


# ---------------------------------------------------------------------------
# StegaStamp
# ---------------------------------------------------------------------------

from .preprocess import square_image  # type: ignore  # local package import


class StegaStampWatermarker(BaseWatermarker):
    """StegaStamp implementation backed by `stegastamp_pkg.api.StegaStamp`.

    This wrapper:
      - Resizes input images to 400x400 square (fit/pad) before embedding.
      - Accepts an arbitrary payload length via `payload_bits`, but internally
        maps to the 100-bit payload expected by the original StegaStamp model
        by padding or truncating.
    """

    def __init__(
        self,
        encoder_path: Path,
        decoder_path: Path,
        payload_bits: int = 100,
        device: Optional[str] = None,
        square_mode: str = "fit",
    ) -> None:
        from stegastamp_pkg.api import StegaStamp  # local project dependency

        if payload_bits != 100:
            raise ValueError(
                f"StegaStampWatermarker currently assumes a 100‑bit payload; "
                f"got payload_bits={payload_bits}."
            )

        dev = device or ("cuda" if _has_cuda() else "cpu")
        self._steg = StegaStamp(str(encoder_path), str(decoder_path), device=dev)
        super().__init__(name="StegaStamp", payload_bits=payload_bits, native_size=(400, 400))
        self.square_mode = square_mode

    def _normalize_bits(self, message_bits: np.ndarray) -> np.ndarray:
        """Validate and normalize input bits to exactly 100 bits."""
        bits = np.asarray(message_bits, dtype=np.uint8).flatten()
        if bits.size != self.payload_bits:
            raise ValueError(
                f"StegaStampWatermarker expects exactly {self.payload_bits} bits; "
                f"got {bits.size}."
            )
        return bits

    def encode(self, image: Image.Image, message_bits: np.ndarray) -> Image.Image:
        bits_100 = self._normalize_bits(message_bits)
        img_rgb = image.convert("RGB")
        if img_rgb.size != self.native_size:
            img_rgb = square_image(img_rgb, size=self.native_size[0], mode=self.square_mode)
        encoded = self._steg.embed(img_rgb, secret=bits_100, return_residual=False)
        return encoded

    def decode(self, image: Image.Image) -> np.ndarray:
        # We request bits from the StegaStamp API (if available).
        result = self._steg.decode(image.convert("RGB"), return_bits=True)
        if isinstance(result, tuple) and len(result) == 2:
            _, bits = result
        elif isinstance(result, np.ndarray):
            bits = result
        else:
            # Model returned a string secret; fall back to empty bit array.
            bits = np.zeros(self.payload_bits, dtype=np.uint8)
        bits = np.asarray(bits, dtype=np.uint8).flatten()
        # In practice StegaStamp should always return 100 bits; if not, clamp.
        if bits.size != self.payload_bits:
            out = np.zeros(self.payload_bits, dtype=np.uint8)
            out[: min(self.payload_bits, bits.size)] = bits[: self.payload_bits]
            return out
        return bits


def _has_cuda() -> bool:
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


def _load_ssl_module(name: str, path: Path):
    """Load a module from ssl_watermarking-main under a unique name."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# DwtDctSvd (Invisible Watermark)
# ---------------------------------------------------------------------------


class DwtDctSvdWatermarker(BaseWatermarker):
    """Wrapper of `imwatermark` frequency watermarking using `dwtDctSvd`.

    Notes:
      - This is the standard non-differentiable implementation used by
        `invisible-watermark` (OpenCV + pywt backend).
      - API keeps `encode/decode` signatures consistent with other watermarkers.
      - `decode_probs_tensor` is intentionally unsupported because imwatermark
        does not expose differentiable logits/probabilities.
    """

    def __init__(
        self,
        payload_bits: int = 32,
        image_size: Tuple[int, int] = (256, 256),
        square_mode: str = "fit",
        method: str = "dwtDctSvd",
        scales: Optional[list[int]] = None,
    ) -> None:
        if payload_bits <= 0:
            raise ValueError("payload_bits must be positive.")
        if method != "dwtDctSvd":
            raise ValueError(f"Only 'dwtDctSvd' is supported in this wrapper; got method={method!r}.")

        super().__init__(name="DwtDctSvd", payload_bits=payload_bits, native_size=image_size)
        self.square_mode = square_mode
        self.method = method
        self.scales = list(scales) if scales is not None else [0, 36, 0]
        self.differentiable = False

        self._encoder_cls, self._decoder_cls = self._load_imwatermark_backend()

    @staticmethod
    def _check_imwatermark_runtime() -> None:
        """Verify imwatermark runtime in a subprocess to avoid hard-crashing current process."""
        check_cmd = (
            "from imwatermark import WatermarkEncoder, WatermarkDecoder; "
            "import cv2, pywt; "
            "print('ok')"
        )
        proc = subprocess.run(
            [sys.executable, "-c", check_cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if proc.returncode != 0:
            stderr = (proc.stderr or "").strip()
            raise RuntimeError(
                "imwatermark runtime is unavailable. Please ensure `imwatermark`, `opencv-python`, "
                "and `pywavelets` are correctly installed in the current environment. "
                f"Details: {stderr[:500]}"
            )

    @classmethod
    def _load_imwatermark_backend(cls):
        cls._check_imwatermark_runtime()
        from imwatermark import WatermarkEncoder, WatermarkDecoder  # type: ignore
        return WatermarkEncoder, WatermarkDecoder

    @staticmethod
    def _to_cv2_bgr(img: Image.Image):
        import cv2  # type: ignore

        arr = np.array(img.convert("RGB"), dtype=np.uint8)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    @staticmethod
    def _from_cv2_bgr(arr: np.ndarray) -> Image.Image:
        import cv2  # type: ignore

        rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))

    def _normalize_bits(self, message_bits: np.ndarray) -> np.ndarray:
        bits = np.asarray(message_bits, dtype=np.uint8).flatten()
        if bits.size != self.payload_bits:
            raise ValueError(
                f"{self.name} expects exactly {self.payload_bits} bits; got {bits.size}."
            )
        return bits

    def encode(self, image: Image.Image, message_bits: np.ndarray) -> Image.Image:
        bits = self._normalize_bits(message_bits)

        img_rgb = image.convert("RGB")
        if self.native_size is not None and img_rgb.size != self.native_size:
            img_rgb = square_image(img_rgb, size=self.native_size[0], mode=self.square_mode)

        img_bgr = self._to_cv2_bgr(img_rgb)
        encoder = self._encoder_cls()
        encoder.set_watermark("bits", bits.tolist())
        encoded_bgr = encoder.encode(img_bgr, method=self.method, scales=self.scales)
        return self._from_cv2_bgr(encoded_bgr)

    def decode(self, image: Image.Image) -> np.ndarray:
        img_rgb = image.convert("RGB")
        if self.native_size is not None and img_rgb.size != self.native_size:
            img_rgb = square_image(img_rgb, size=self.native_size[0], mode=self.square_mode)

        img_bgr = self._to_cv2_bgr(img_rgb)
        decoder = self._decoder_cls("bits", self.payload_bits)
        bits = decoder.decode(img_bgr, method=self.method, scales=self.scales)
        bits = np.asarray(bits, dtype=np.uint8).flatten()
        out = np.zeros(self.payload_bits, dtype=np.uint8)
        out[: min(self.payload_bits, bits.size)] = bits[: self.payload_bits]
        return out

    def decode_probs_tensor(self, x: torch.Tensor, device: Optional[torch.device | str] = None) -> torch.Tensor:
        raise NotImplementedError(
            "imwatermark `dwtDctSvd` backend is non-differentiable and does not provide "
            "decode probabilities for gradient-based attacks."
        )


class RivaGANWatermarker(BaseWatermarker):
    """Wrapper of `imwatermark` RivaGAN method.

    Notes:
      - Uses `invisible-watermark` ONNX runtime backend (`method='rivaGan'`).
      - Current upstream only supports 32-bit payloads.
      - This method is non-differentiable.
    """

    def __init__(
        self,
        payload_bits: int = 32,
        image_size: Tuple[int, int] = (256, 256),
        square_mode: str = "fit",
        threshold: float = 0.52,
    ) -> None:
        if payload_bits != 32:
            raise ValueError(
                "RivaGANWatermarker currently supports only 32-bit payloads "
                f"(got payload_bits={payload_bits})."
            )
        super().__init__(name="RivaGAN", payload_bits=payload_bits, native_size=image_size)
        self.square_mode = square_mode
        self.threshold = float(threshold)
        self.differentiable = False

        self._encoder_cls, self._decoder_cls = self._load_rivagan_backend()

    @staticmethod
    def _check_rivagan_runtime() -> None:
        check_cmd = (
            "from imwatermark import WatermarkEncoder, WatermarkDecoder; "
            "import cv2, onnxruntime; "
            "WatermarkEncoder.loadModel(); WatermarkDecoder.loadModel(); "
            "print('ok')"
        )
        proc = subprocess.run(
            [sys.executable, "-c", check_cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if proc.returncode != 0:
            stderr = (proc.stderr or "").strip()
            raise RuntimeError(
                "RivaGAN backend is unavailable. Please ensure `imwatermark`, `opencv-python`, "
                "and `onnxruntime` are correctly installed in the current environment. "
                f"Details: {stderr[:500]}"
            )

    @classmethod
    def _load_rivagan_backend(cls):
        cls._check_rivagan_runtime()
        from imwatermark import WatermarkEncoder, WatermarkDecoder  # type: ignore

        WatermarkEncoder.loadModel()
        WatermarkDecoder.loadModel()
        return WatermarkEncoder, WatermarkDecoder

    @staticmethod
    def _to_cv2_bgr(img: Image.Image):
        import cv2  # type: ignore

        arr = np.array(img.convert("RGB"), dtype=np.uint8)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    @staticmethod
    def _from_cv2_bgr(arr: np.ndarray) -> Image.Image:
        import cv2  # type: ignore

        rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))

    def _normalize_bits(self, message_bits: np.ndarray) -> np.ndarray:
        bits = np.asarray(message_bits, dtype=np.uint8).flatten()
        if bits.size != self.payload_bits:
            raise ValueError(
                f"{self.name} expects exactly {self.payload_bits} bits; got {bits.size}."
            )
        return bits

    def encode(self, image: Image.Image, message_bits: np.ndarray) -> Image.Image:
        bits = self._normalize_bits(message_bits)

        img_rgb = image.convert("RGB")
        if self.native_size is not None and img_rgb.size != self.native_size:
            img_rgb = square_image(img_rgb, size=self.native_size[0], mode=self.square_mode)

        img_bgr = self._to_cv2_bgr(img_rgb)
        encoder = self._encoder_cls()
        encoder.set_watermark("bits", bits.tolist())
        encoded_bgr = encoder.encode(img_bgr, method="rivaGan")
        return self._from_cv2_bgr(encoded_bgr)

    def decode(self, image: Image.Image) -> np.ndarray:
        img_rgb = image.convert("RGB")
        if self.native_size is not None and img_rgb.size != self.native_size:
            img_rgb = square_image(img_rgb, size=self.native_size[0], mode=self.square_mode)

        img_bgr = self._to_cv2_bgr(img_rgb)
        decoder = self._decoder_cls("bits", self.payload_bits)
        bits = decoder.decode(img_bgr, method="rivaGan", threshold=self.threshold)
        bits = np.asarray(bits, dtype=np.uint8).flatten()
        out = np.zeros(self.payload_bits, dtype=np.uint8)
        out[: min(self.payload_bits, bits.size)] = bits[: self.payload_bits]
        return out

    def decode_probs_tensor(self, x: torch.Tensor, device: Optional[torch.device | str] = None) -> torch.Tensor:
        raise NotImplementedError(
            "imwatermark `rivaGan` backend is non-differentiable and does not provide "
            "decode probabilities for gradient-based attacks."
        )




# ---------------------------------------------------------------------------
# SSL latent watermark (ssl_watermarking)
# ---------------------------------------------------------------------------


class SSLLatentWatermarker(BaseWatermarker):
    """Skeleton for ssl_watermarking latent-space watermark.

    Typical backend:
      - Backbone: ResNet-50 DINO (dino_r50_plus.pth).
      - Norm layer: PCA whitening layer (e.g., out2048_coco_orig.pth).

    TODO:
      - Wrap calls to encode.py/decode.py or refactor core logic into a
        callable class so you can embed/decode from Python directly.
    """

    def __init__(
        self,
        payload_bits: int = 30,
        image_size: Tuple[int, int] = (128, 128),
        model_name: str = "resnet50",
        model_path: Optional[Path] = None,
        normlayer_path: Optional[Path] = None,
        epochs: int = 50,
        target_psnr: float = 40.0,
        lambda_w: float = 5e4,
        lambda_i: float = 1.0,
        lr: float = 1e-2,
        carrier_path: Optional[Path] = None,
        carrier_seed: Optional[int] = None,
        save_carrier: bool = False,
    ) -> None:
        """Construct an SSL latent-space watermarker.

        Args:
            payload_bits: Number of bits to embed (default 30).
            image_size: Input/output resolution (default 128×128).
            model_name: Backbone name for torchvision/timm (default resnet50).
            model_path: Path to the pretrained DINO backbone weights.
            normlayer_path: Path to the PCA whitening layer weights.
            epochs: Number of optimisation steps for embedding.
            target_psnr: PSNR budget for image distortion.
            lambda_w: Weight on watermark loss.
            lambda_i: Weight on image distortion loss.
            lr: Learning rate for image optimisation.
        """
        if payload_bits <= 0:
            raise ValueError("payload_bits must be positive.")
        super().__init__(name="SSL-Latent", payload_bits=payload_bits, native_size=image_size)

        # Resolve default paths relative to repo root if not provided.
        repo_root = Path(__file__).resolve().parents[1]
        ssl_root = repo_root / "ssl_watermarking-main"
        if model_path is None:
            model_path = ssl_root / "models" / "dino_r50_plus.pth"
        if normlayer_path is None:
            # For 128×128 inputs, the resized whitening is typically used.
            # Fall back to COCO whitening if the resized one is unavailable.
            resized_candidate = ssl_root / "normalayer" / "out2048_coco_resized.pth"
            if resized_candidate.exists():
                normlayer_path = resized_candidate
            else:
                normlayer_path = ssl_root / "normalayer" / "out2048_coco_orig.pth"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.img_size = image_size
        self.epochs = int(epochs)
        self.target_psnr = float(target_psnr)
        self.lambda_w = float(lambda_w)
        self.lambda_i = float(lambda_i)
        self.lr = float(lr)
        self._carrier_path = Path(carrier_path).expanduser() if carrier_path else None
        self._carrier_seed = carrier_seed
        self._save_carrier = bool(save_carrier)

        # Allow legacy numpy scalar pickles in ssl_watermarking checkpoints when
        # running with newer PyTorch (weights_only=True by default).
        try:
            from torch.serialization import add_safe_globals  # type: ignore
            import numpy as _np

            add_safe_globals([_np.core.multiarray.scalar])  # type: ignore[arg-type]
        except Exception:
            pass

        # ----------------- Load ssl_watermarking backbone + normlayer -----------------
        utils_mod = _load_ssl_module("sslw_utils", ssl_root / "utils.py")
        self._ssl_utils = utils_mod
        backbone = utils_mod.build_backbone(str(model_path), model_name)
        normlayer = utils_mod.load_normalization_layer(str(normlayer_path))
        self.model = utils_mod.NormLayerWrapper(backbone, normlayer).to(self.device)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

        # Determine feature dimension and generate/load carriers for payload bits.
        if self._carrier_seed is not None:
            np.random.seed(self._carrier_seed)
            torch.manual_seed(self._carrier_seed)
            random.seed(self._carrier_seed)

        carrier_loaded = False
        if self._carrier_path and self._carrier_path.exists():
            try:
                carrier_tensor = torch.load(self._carrier_path, map_location=self.device)
                if isinstance(carrier_tensor, torch.Tensor):
                    carrier_loaded = True
                    self.carrier = carrier_tensor.to(self.device)
                else:
                    raise ValueError("carrier file did not contain a torch.Tensor")
            except Exception:
                carrier_loaded = False

        if not carrier_loaded:
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 224, 224, device=self.device)
                D = self.model(dummy).size(-1)
            self.carrier = utils_mod.generate_carriers(payload_bits, D).to(self.device)
            if self._carrier_path and self._save_carrier:
                try:
                    self._carrier_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(self.carrier.cpu(), self._carrier_path)
                except Exception:
                    pass

        # ImageNet normalization (same as ssl_watermarking.utils_img.default_transform)
        self._mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(3, 1, 1)
        self._to_tensor = T.ToTensor()

    def _preprocess_pil(self, image: Image.Image) -> torch.Tensor:
        img = image.convert("RGB")
        if self.native_size is not None and img.size != self.native_size:
            # Preserve aspect ratio via square crop/pad to 128×128.
            img = square_image(img, size=self.native_size[0], mode="fit")
        t = self._to_tensor(img).to(self.device)
        t = (t - self._mean) / self._std
        return t.unsqueeze(0)  # [1,3,H,W]

    def _postprocess_tensor(self, t: torch.Tensor) -> Image.Image:
        t = t.detach().cpu().squeeze(0)
        t = t * self._std.cpu() + self._mean.cpu()
        t = t.clamp(0.0, 1.0)
        return T.ToPILImage()(t)

    def encode(self, image: Image.Image, message_bits: np.ndarray) -> Image.Image:
        bits = np.asarray(message_bits, dtype=np.uint8).flatten()
        if bits.size != self.payload_bits:
            raise ValueError(
                f"{self.name} expects exactly {self.payload_bits} bits; got {bits.size}."
            )
        msg = torch.from_numpy(bits).to(self.device).bool().view(1, -1)  # [1,K]
        msg_signs = 2 * msg.float() - 1.0  # [1,K] in {‑1,+1}

        x0 = self._preprocess_pil(image)  # [1,3,H,W]

        # Ensure gradients are enabled even if caller has globally disabled them
        # (e.g., torch.set_grad_enabled(False) in evaluation scripts).
        with torch.enable_grad():
            x = x0.clone().detach().requires_grad_(True)
            optimizer = torch.optim.Adam([x], lr=self.lr)

            margin = 5.0
            for _ in range(self.epochs):
                ft = self.model(x)  # [1,D]
                dot = ft @ self.carrier.T  # [1,K]

                loss_w = torch.clamp(margin - dot * msg_signs, min=0).mean()
                loss_i = torch.mean((x - x0) ** 2)
                loss = self.lambda_w * loss_w + self.lambda_i * loss_i

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    # Simple clamp in normalized space to avoid divergence.
                    x.clamp_(-3.0, 3.0)

        return self._postprocess_tensor(x)

    def decode(self, image: Image.Image) -> np.ndarray:
        x = self._preprocess_pil(image)  # [1,3,H,W]
        with torch.no_grad():
            ft = self.model(x)  # [1,D]
            dot = ft @ self.carrier.T  # [1,K]
            msg = (torch.sign(dot).squeeze(0) > 0).cpu().numpy().astype(np.uint8)
        return msg.flatten()[: self.payload_bits]


# ---------------------------------------------------------------------------
# HiDDeN
# ---------------------------------------------------------------------------


class HiDDeNWatermarker(BaseWatermarker):
    """HiDDeN watermarker backed by `HiDDeN-master` codebase.

    This wrapper:
      - Uses the default HiDDeN architecture (128×128, 30‑bit message).
      - Loads a trained Encoder/Decoder from a checkpoint (by default the
        `combined-noise--epoch-400.pyt` experiment).
      - Applies only the Identity noise layer during encode (external attack
        code in this repo is responsible for additional noise/attacks).
    """

    def __init__(
        self,
        payload_bits: int = 30,
        image_size: Tuple[int, int] = (128, 128),
        checkpoint_path: Optional[Path] = None,
        device: Optional[str] = None,
    ) -> None:
        if payload_bits <= 0:
            raise ValueError("payload_bits must be positive.")

        super().__init__(name="HiDDeN", payload_bits=payload_bits, native_size=image_size)

        # Resolve HiDDeN repo paths.
        repo_root = Path(__file__).resolve().parents[1]
        hidden_root = repo_root / "HiDDeN-master"
        if not hidden_root.exists():
            raise FileNotFoundError(
                f"HiDDeN-master directory not found at expected location: {hidden_root}"
            )

        # Select device.
        self.device = torch.device(device or ("cuda" if _has_cuda() else "cpu"))
        self.img_size = image_size

        # Lazily import HiDDeN modules with their expected names.
        # We avoid importing HiDDeN's utils.py to prevent clashes with this
        # repo's `utils` package, and we also make sure that HiDDeN's `model`
        # directory is treated as a package even if a different top-level
        # `model` module exists in the environment.
        import sys
        import types

        hidden_root_str = str(hidden_root)
        if hidden_root_str not in sys.path:
            sys.path.insert(0, hidden_root_str)

        # Ensure that `import model.*` resolves to HiDDeN-master/model.
        model_pkg_path = hidden_root / "model"
        existing_model = sys.modules.get("model")
        if existing_model is None or not hasattr(existing_model, "__path__"):
            pkg = types.ModuleType("model")
            pkg.__path__ = [str(model_pkg_path)]
            sys.modules["model"] = pkg
        else:
            paths = list(getattr(existing_model, "__path__", []))
            if str(model_pkg_path) not in paths:
                paths.append(str(model_pkg_path))
                existing_model.__path__ = paths

        from options import HiDDenConfiguration  # type: ignore
        from model.hidden import Hidden  # type: ignore
        from noise_layers.noiser import Noiser  # type: ignore

        H, W = image_size

        # Mirror the default configuration used in HiDDeN-master/main.py.
        self._config = HiDDenConfiguration(
            H=H,
            W=W,
            message_length=payload_bits,
            encoder_blocks=4,
            encoder_channels=64,
            decoder_blocks=7,
            decoder_channels=64,
            use_discriminator=True,
            use_vgg=False,
            discriminator_blocks=3,
            discriminator_channels=64,
            decoder_loss=1.0,
            encoder_loss=0.7,
            adversarial_loss=1e-3,
            enable_fp16=False,
        )

        # Only Identity noise for this unified interface; attacks are handled
        # elsewhere in the experiment pipeline.
        noise_config = []
        self._noiser = Noiser(noise_config, self.device)

        # Construct HiDDeN model and load encoder/decoder weights.
        self._hidden = Hidden(self._config, self.device, self._noiser, tb_logger=None)

        if checkpoint_path is None:
            # 默认改为 JPEG 压缩噪声训练的高精度权重
            checkpoint_path = (
                hidden_root
                / "experiments"
                / "jpeg-compression"
                / "checkpoints"
                / "epoch-300.pyt"
            )
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"HiDDeN checkpoint not found at: {checkpoint_path}")

        checkpoint = torch.load(str(checkpoint_path), map_location=self.device)

        # Checkpoint format follows HiDDeN-master/utils.save_checkpoint.
        if isinstance(checkpoint, dict) and "enc-dec-model" in checkpoint:
            state_dict = checkpoint["enc-dec-model"]
        else:
            # Fallback: assume checkpoint is directly the encoder/decoder state_dict.
            state_dict = checkpoint

        self._hidden.encoder_decoder.load_state_dict(state_dict)
        self._hidden.encoder_decoder.eval()

        # Simple image <-> tensor transforms (RGB, [-1, 1] range).
        self._to_tensor = T.ToTensor()

    # --------------------------- internal helpers ---------------------------

    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Convert PIL.Image to normalized tensor [1,3,H,W] in [-1,1]."""
        img = image.convert("RGB")
        if self.native_size is not None and img.size != self.native_size:
            img = square_image(img, size=self.native_size[0], mode="fit")
        t = self._to_tensor(img).to(self.device)  # [3,H,W] in [0,1]
        t = t * 2.0 - 1.0  # [-1,1]
        return t.unsqueeze(0)  # [1,3,H,W]

    def _tensor_to_image(self, t: torch.Tensor) -> Image.Image:
        """Convert normalized tensor [1,3,H,W] in [-1,1] back to PIL.Image."""
        t = t.detach().cpu().squeeze(0)
        t = (t + 1.0) / 2.0
        t = t.clamp(0.0, 1.0)
        return T.ToPILImage()(t)

    def _normalize_bits(self, message_bits: np.ndarray) -> np.ndarray:
        bits = np.asarray(message_bits, dtype=np.uint8).flatten()
        if bits.size != self.payload_bits:
            raise ValueError(
                f"{self.name} expects exactly {self.payload_bits} bits; got {bits.size}."
            )
        return bits

    # ------------------------------ API methods -----------------------------

    def encode(self, image: Image.Image, message_bits: np.ndarray) -> Image.Image:
        """Embed `message_bits` into `image` using HiDDeN encoder."""
        bits = self._normalize_bits(message_bits)
        msg = torch.from_numpy(bits.astype(np.float32)).to(self.device).view(1, -1)

        x = self._preprocess_image(image)
        with torch.no_grad():
            encoded_image, _, _ = self._hidden.encoder_decoder(x, msg)
        return self._tensor_to_image(encoded_image)

    def decode(self, image: Image.Image) -> np.ndarray:
        """Recover message bits from a (possibly attacked) watermarked image."""
        x = self._preprocess_image(image)
        with torch.no_grad():
            decoded = self._hidden.encoder_decoder.decoder(x)
        bits = decoded.detach().cpu().numpy().round().clip(0, 1).astype(np.uint8)
        return bits.flatten()[: self.payload_bits]


__all__ = [
    "BaseWatermarker",
    "StegaStampWatermarker",
    "DwtDctSvdWatermarker",
    "RivaGANWatermarker",
    "SSLLatentWatermarker",
    "HiDDeNWatermarker",
]
