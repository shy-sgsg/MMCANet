"""Small, auditable single-stage joint denoise+super-resolution model."""

from __future__ import annotations

import torch
from torch import nn


def _load_swinir() -> type[nn.Module]:
    """Load the vendored BasicSR SwinIR implementation without global installs.

    The repository keeps the upstream SwinIR source under ``BasicSR/`` and its
    accompanying license.  We add that directory to ``sys.path`` only at the
    import boundary so the formal baseline is auditable and does not depend on
    a separately installed BasicSR package.
    """

    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    basicsr_root = root / "BasicSR"
    if str(basicsr_root) not in sys.path:
        sys.path.insert(0, str(basicsr_root))
    from basicsr.archs.swinir_arch import SwinIR  # type: ignore

    return SwinIR


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class JointESPCN(nn.Module):
    """ESPCN-style one-pass LR-SAR -> HR-SAR restoration baseline.

    The network has no optical input and performs denoising and x2
    reconstruction in one mapping.  It is deliberately compact and is used
    as a protocol-adapted task-matched baseline, not as an exact reproduction
    of any one published implementation.
    """

    def __init__(self, *, scale: int = 2, width: int = 48, blocks: int = 4) -> None:
        super().__init__()
        if scale != 2:
            raise ValueError("The current fixed protocol supports scale=2 only")
        if width < 8 or blocks < 1:
            raise ValueError("width must be >=8 and blocks >=1")
        self.scale = int(scale)
        self.head = nn.Sequential(nn.Conv2d(1, width, 5, padding=2), nn.ReLU(inplace=True))
        self.body = nn.Sequential(*[ResidualBlock(width) for _ in range(blocks)])
        self.fuse = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, 1 * scale * scale, 3, padding=1),
        )
        self.pixel_shuffle = nn.PixelShuffle(scale)
        self.output = nn.Sigmoid()

    def forward(self, lr_sar: torch.Tensor) -> torch.Tensor:
        if lr_sar.ndim != 4 or lr_sar.shape[1] != 1:
            raise ValueError(f"Expected LR SAR [B,1,H,W], got {tuple(lr_sar.shape)}")
        if not torch.is_floating_point(lr_sar) or not torch.isfinite(lr_sar).all():
            raise ValueError("LR SAR must be finite floating-point")
        features = self.head(lr_sar)
        features = self.body(features) + features
        output = self.output(self.pixel_shuffle(self.fuse(features)))
        expected = (lr_sar.shape[-2] * self.scale, lr_sar.shape[-1] * self.scale)
        if output.shape[-2:] != expected:
            raise RuntimeError(f"Unexpected JointESPCN output size: {tuple(output.shape)}")
        return output


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


class SwinIRJoint(nn.Module):
    """Single-stage SwinIR adaptation for noisy LR SAR to HR pseudo-target.

    This uses the vendored upstream SwinIR architecture, but deliberately
    changes the interface to one SAR channel and trains from scratch under the
    fixed 1600/200/200 protocol.  It is therefore a protocol-adapted baseline,
    not a claim of reproducing a published SAR-specific SwinIR checkpoint.
    """

    def __init__(
        self,
        *,
        scale: int = 2,
        img_size: int = 128,
        embed_dim: int = 60,
        depths: tuple[int, ...] = (2, 2, 2, 2),
        num_heads: tuple[int, ...] = (4, 4, 4, 4),
        window_size: int = 8,
        mlp_ratio: float = 2.0,
        upsampler: str = "pixelshuffledirect",
        allow_variable_input: bool = False,
    ) -> None:
        super().__init__()
        if scale != 2:
            raise ValueError("The fixed protocol currently supports scale=2 only")
        if img_size <= 0 or img_size % window_size != 0:
            raise ValueError("img_size must be a positive multiple of window_size")
        if len(depths) != len(num_heads):
            raise ValueError("depths and num_heads must have the same length")
        if embed_dim % num_heads[0] != 0:
            raise ValueError("embed_dim must be divisible by each attention head count")
        SwinIR = _load_swinir()
        img_range = 1.0
        self.config = {
            "name": "SwinIRJoint",
            "scale": int(scale),
            "img_size": int(img_size),
            "in_chans": 1,
            "out_chans": 1,
            "embed_dim": int(embed_dim),
            "depths": [int(item) for item in depths],
            "num_heads": [int(item) for item in num_heads],
            "window_size": int(window_size),
            "mlp_ratio": float(mlp_ratio),
            "upsampler": upsampler,
            "img_range": img_range,
            "output_activation": "sigmoid",
            "external_pretrained_weights": False,
            "upstream_architecture": "BasicSR/basicsr/archs/swinir_arch.py",
            "allow_variable_input": bool(allow_variable_input),
        }
        self.allow_variable_input = bool(allow_variable_input)
        self.window_size = int(window_size)
        self.core = SwinIR(
            img_size=img_size,
            patch_size=1,
            in_chans=1,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            upscale=scale,
            img_range=img_range,
            upsampler=upsampler,
            resi_connection="1conv",
        )
        self.output = nn.Sigmoid()

    def forward(self, lr_sar: torch.Tensor) -> torch.Tensor:
        if lr_sar.ndim != 4 or lr_sar.shape[1] != 1:
            raise ValueError(f"Expected LR SAR [B,1,H,W], got {tuple(lr_sar.shape)}")
        if not torch.is_floating_point(lr_sar) or not torch.isfinite(lr_sar).all():
            raise ValueError("LR SAR must be finite floating-point")
        expected_input = (128, 128)
        if self.allow_variable_input:
            height, width = (int(value) for value in lr_sar.shape[-2:])
            if height <= 0 or width <= 0 or height % self.window_size != 0 or width % self.window_size != 0:
                raise ValueError(
                    "Variable SwinIRJoint inputs must have positive spatial dimensions "
                    f"divisible by window_size={self.window_size}, got {tuple(lr_sar.shape[-2:])}"
                )
        elif tuple(lr_sar.shape[-2:]) != expected_input:
            raise ValueError(f"SwinIRJoint currently requires {expected_input}, got {tuple(lr_sar.shape[-2:])}")
        output = self.output(self.core(lr_sar))
        expected = (lr_sar.shape[-2] * 2, lr_sar.shape[-1] * 2)
        if output.shape[1] != 1 or tuple(output.shape[-2:]) != expected:
            raise RuntimeError(f"Unexpected SwinIRJoint output size: {tuple(output.shape)}")
        return output
