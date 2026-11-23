"""Matched, scale-aware C2 exploratory restoration controls.

Both arms have identical modules and parameter counts. The SAR-only arm
supplies zeros for RGB; the guided arm can use matched, permuted, or missing
HR optical. Output is a normalized noisy SAR pseudo-target representation.
"""
from __future__ import annotations

import torch
from torch import nn

from JointJDSR.model import ResidualBlock


class C2MatchedJoint(nn.Module):
    def __init__(self, *, scale: int, optical_enabled: bool, width: int = 32,
                 blocks: int = 4, hr_width: int = 16) -> None:
        super().__init__()
        if scale not in (2, 4):
            raise ValueError('C2 scale must be 2 or 4')
        if width < 8 or blocks < 1 or hr_width < 8:
            raise ValueError('invalid feature dimensions')
        self.scale = scale
        self.optical_enabled = optical_enabled
        self.head = nn.Sequential(nn.Conv2d(1, width, 5, padding=2), nn.ReLU())
        self.body = nn.Sequential(*(ResidualBlock(width) for _ in range(blocks)))
        self.upsample = nn.Sequential(nn.Conv2d(width, hr_width * scale * scale, 3, padding=1),
                                      nn.PixelShuffle(scale), nn.ReLU())
        self.output = nn.Sequential(nn.Conv2d(hr_width + 3, hr_width, 3, padding=1),
                                    nn.ReLU(), nn.Conv2d(hr_width, 1, 3, padding=1), nn.Sigmoid())

    def forward(self, sar_lr: torch.Tensor, optical_hr: torch.Tensor | None = None) -> torch.Tensor:
        if sar_lr.ndim != 4 or sar_lr.shape[1] != 1 or not torch.is_floating_point(sar_lr):
            raise ValueError('SAR must be floating point [B,1,H,W]')
        if not torch.isfinite(sar_lr).all():
            raise ValueError('SAR contains nonfinite values')
        b, _, h, w = sar_lr.shape
        expected = (b, 3, h * self.scale, w * self.scale)
        if not self.optical_enabled and optical_hr is not None:
            raise ValueError('SAR-only arm must not receive optical data')
        if optical_hr is None:
            optical_hr = sar_lr.new_zeros(expected)
        elif (tuple(optical_hr.shape) != expected or optical_hr.device != sar_lr.device
              or optical_hr.dtype != sar_lr.dtype or not torch.isfinite(optical_hr).all()):
            raise ValueError('optical shape, device, dtype or finiteness mismatch')
        sar_features = self.head(sar_lr)
        sar_features = self.body(sar_features) + sar_features
        hr_features = self.upsample(sar_features)
        result = self.output(torch.cat((hr_features, optical_hr), dim=1))
        if tuple(result.shape) != (b, 1, h * self.scale, w * self.scale):
            raise RuntimeError('unexpected output shape')
        return result
