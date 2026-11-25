"""Canonical SAR normalization and synthetic degradation operators.

This module is the only authoritative implementation used by MMCANet and the
baselines.  The operators are explicit about the representation domain: the
local PNG data do not carry enough metadata to infer whether pixel values are
calibrated amplitude, intensity, power, or display values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple, Union

import torch
import torch.nn.functional as F


NormalizationMode = Literal["linear", "log", "log_clip", "log_percentile"]
SpeckleDomain = Literal["intensity", "amplitude", "representation"]
NegativePolicy = Literal["clamp", "error", "preserve"]


@dataclass(frozen=True)
class NormalizationStats:
    """Parameters needed to inspect or approximately invert preprocessing."""

    mode: str
    log_scale: float
    low: float
    high: float
    input_min: float
    input_max: float
    clipped_to_unit: bool


def _validate_single_sar(sar: torch.Tensor) -> None:
    if sar.ndim != 3 or sar.shape[0] != 1:
        raise ValueError(f"Expected SAR tensor shape [1, H, W], got {tuple(sar.shape)}")
    if not torch.is_floating_point(sar):
        raise TypeError(f"Expected floating-point SAR tensor, got {sar.dtype}")
    if not torch.isfinite(sar).all():
        raise ValueError("SAR tensor contains NaN or Inf values")


def preprocess_sar_tensor(
    sar: torch.Tensor,
    norm_mode: NormalizationMode = "log",
    log_scale: float = 20.0,
    clip_low: float = 0.1,
    clip_high: float = 99.9,
    input_clamp_max: Optional[float] = 1.0,
    clip_to_unit: bool = True,
    eps: float = 1e-6,
    return_stats: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, NormalizationStats]]:
    """Normalize a single non-negative SAR representation.

    ``log`` is an invertible log1p compression for inputs already scaled to
    [0, 1]. ``log_clip``/``log_percentile`` additionally apply per-image
    percentile normalization and are therefore representation-changing legacy
    modes.  They must not be described as radiometric calibration.
    """

    _validate_single_sar(sar)
    if eps <= 0:
        raise ValueError("eps must be positive")
    if log_scale <= 0:
        raise ValueError("log_scale must be positive")

    input_min = float(sar.min().item())
    input_max = float(sar.max().item())
    if input_clamp_max is None:
        x = torch.clamp_min(sar, 0.0)
    else:
        if input_clamp_max <= 0:
            raise ValueError("input_clamp_max must be positive when provided")
        x = torch.clamp(sar, 0.0, float(input_clamp_max))
    mode = "log_clip" if norm_mode == "log_percentile" else norm_mode

    if mode == "linear":
        low = 0.0
        high = 1.0
        out = x
    elif mode in {"log", "log_clip"}:
        scale_tensor = x.new_tensor(float(log_scale))
        logged = torch.log1p(scale_tensor * x) / torch.log1p(scale_tensor)
        if mode == "log":
            low = 0.0
            high = 1.0
            out = logged
        else:
            if not (0.0 <= clip_low < clip_high <= 100.0):
                raise ValueError(
                    f"Expected 0 <= clip_low < clip_high <= 100, got {clip_low}, {clip_high}"
                )
            flat = logged.reshape(-1)
            lo = torch.quantile(flat, float(clip_low) / 100.0)
            hi = torch.quantile(flat, float(clip_high) / 100.0)
            if float((hi - lo).abs().item()) <= eps:
                raise ValueError("Percentile normalization has a near-zero dynamic range")
            low = float(lo.item())
            high = float(hi.item())
            out = (logged - lo) / (hi - lo)
    else:
        raise ValueError(f"Unsupported SAR normalization mode: {norm_mode}")

    if clip_to_unit:
        out = torch.clamp(out, 0.0, 1.0)
    if not torch.isfinite(out).all():
        raise ValueError("SAR normalization produced NaN or Inf values")

    stats = NormalizationStats(
        mode=mode,
        log_scale=float(log_scale),
        low=low,
        high=high,
        input_min=input_min,
        input_max=input_max,
        clipped_to_unit=bool(clip_to_unit),
    )
    return (out, stats) if return_stats else out


def inverse_preprocess_sar_tensor(
    normalized: torch.Tensor,
    stats: NormalizationStats,
) -> torch.Tensor:
    """Invert linear/log preprocessing and reconstruct percentile modes.

    Percentile modes are only exactly invertible for values that were not
    clipped to [0, 1]; clipped tails cannot be recovered and are reconstructed
    at the stored percentile boundaries.
    """

    _validate_single_sar(normalized)
    if stats.mode == "linear":
        return normalized
    if stats.mode not in {"log", "log_clip"}:
        raise ValueError(f"Unsupported normalization stats mode: {stats.mode}")

    logged = normalized
    if stats.mode == "log_clip":
        logged = normalized * (stats.high - stats.low) + stats.low
    scale = normalized.new_tensor(stats.log_scale)
    restored = torch.expm1(logged * torch.log1p(scale)) / scale
    return torch.clamp(restored, 0.0, 1.0)


def frequency_truncation_degrade(
    sar: torch.Tensor,
    window_h: int,
    window_w: int,
    return_diagnostics: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:
    """Apply an ideal centered rectangular low-pass mask in the 2-D FFT.

    The inverse FFT uses its real component.  Taking ``abs(ifft)`` is avoided
    because it folds negative ringing and introduces an undocumented nonlinear
    rectification.  Even-sized windows can create a small imaginary residual at
    one boundary frequency; its relative energy is reported for audit.
    """

    _validate_single_sar(sar)
    h, w = sar.shape[-2:]
    if window_h <= 0 or window_w <= 0:
        raise ValueError("Frequency window dimensions must be positive")
    requested_h = min(int(window_h), h)
    requested_w = min(int(window_w), w)

    # A strict centered sub-window of an even-sized DFT has one unmatched
    # boundary frequency.  Use the largest odd window not exceeding the
    # request (unless the full axis is retained) so the mask is conjugate
    # symmetric and the inverse of real input remains real.
    keep_h = requested_h if requested_h == h or requested_h % 2 == 1 else requested_h - 1
    keep_w = requested_w if requested_w == w or requested_w % 2 == 1 else requested_w - 1

    cy, cx = h // 2, w // 2
    top = cy - keep_h // 2
    left = cx - keep_w // 2
    bottom = top + keep_h
    right = left + keep_w

    spectrum = torch.fft.fftshift(torch.fft.fft2(sar.squeeze(0)))
    mask = torch.zeros((h, w), dtype=spectrum.dtype, device=sar.device)
    mask[top:bottom, left:right] = 1.0
    filtered = torch.fft.ifft2(torch.fft.ifftshift(spectrum * mask))

    real = filtered.real
    imag_energy = torch.sum(filtered.imag.square())
    real_energy = torch.sum(real.square())
    imag_ratio = float((imag_energy / (real_energy + 1e-20)).item())
    output = real.unsqueeze(0).to(dtype=sar.dtype)
    diagnostics = {
        "input_energy": float(torch.sum(sar.square()).item()),
        "output_energy": float(torch.sum(output.square()).item()),
        "imag_to_real_energy": imag_ratio,
        "requested_keep_h": float(requested_h),
        "requested_keep_w": float(requested_w),
        "keep_h": float(keep_h),
        "keep_w": float(keep_w),
    }
    return (output, diagnostics) if return_diagnostics else output


def anisotropic_gaussian_blur(
    sar: torch.Tensor,
    kernel_size: int = 9,
    sigma_azimuth: float = 2.0,
    sigma_range: float = 0.8,
    angle_deg: float = 0.0,
) -> torch.Tensor:
    """Apply an anisotropic Gaussian in image coordinates to one SAR channel.

    ``sigma_azimuth`` is applied along the rotated x axis and
    ``sigma_range`` along the rotated y axis.  Convolution uses zero padding,
    so enabling this optional operator attenuates values near image borders;
    the physical range/azimuth meaning of x/y depends on the input product's
    orientation.
    """

    _validate_single_sar(sar)
    if kernel_size < 3:
        raise ValueError("kernel_size must be at least 3")
    if sigma_azimuth <= 0 or sigma_range <= 0:
        raise ValueError("Gaussian sigmas must be positive")
    k = int(kernel_size)
    if k % 2 == 0:
        k += 1
    radius = k // 2

    theta = math.radians(float(angle_deg))
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    coords = torch.arange(-radius, radius + 1, device=sar.device, dtype=sar.dtype)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    x_rot = xx * cos_t + yy * sin_t
    y_rot = -xx * sin_t + yy * cos_t
    kernel = torch.exp(
        -0.5 * ((x_rot / float(sigma_azimuth)) ** 2 + (y_rot / float(sigma_range)) ** 2)
    )
    kernel = kernel / kernel.sum()
    blurred = F.conv2d(sar.unsqueeze(0), kernel.view(1, 1, k, k), padding=radius)
    return blurred.squeeze(0)


def sample_gamma_speckle_multiplier(
    shape: torch.Size,
    reference: torch.Tensor,
    looks: float,
    domain: SpeckleDomain = "representation",
) -> torch.Tensor:
    """Sample a unit-mean Gamma multiplier (or its amplitude square root)."""

    if looks <= 0:
        raise ValueError(f"looks must be positive, got {looks}")
    if domain not in {"intensity", "amplitude", "representation"}:
        raise ValueError(f"Unsupported speckle domain: {domain}")
    look_count = reference.new_tensor(float(looks))
    gamma_dist = torch.distributions.Gamma(concentration=look_count, rate=look_count)
    multiplier = gamma_dist.sample(shape)
    if domain == "amplitude":
        multiplier = torch.sqrt(multiplier)
    return multiplier


def gamma_speckle_noise(
    sar: torch.Tensor,
    looks: float = 1.0,
    domain: SpeckleDomain = "representation",
) -> torch.Tensor:
    """Inject multiplicative speckle under an explicit domain assumption.

    For intensity, the multiplier has mean 1 and variance 1/L.  For amplitude,
    the square root of that multiplier is used.  ``representation`` retains the
    historical direct multiplier but deliberately makes no physical-domain
    claim.
    """

    _validate_single_sar(sar)
    multiplier = sample_gamma_speckle_multiplier(sar.shape, sar, looks, domain)
    return sar * multiplier


def downsample_sar(sar: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """Downsample one SAR channel with explicitly configured bilinear sampling."""

    _validate_single_sar(sar)
    if target_h <= 0 or target_w <= 0:
        raise ValueError("Target size must be positive")
    return F.interpolate(
        sar.unsqueeze(0),
        size=(int(target_h), int(target_w)),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)


def build_sar_lr_from_physical_degradation(
    sar_mr: torch.Tensor,
    input_downsample_size: int = 64,
    downsample_factor: int = 2,
    use_freq_truncation: bool = True,
    use_anisotropic_blur: bool = False,
    use_speckle: bool = True,
    freq_window_size: int = 0,
    blur_kernel_size: int = 9,
    blur_sigma_azimuth: float = 2.0,
    blur_sigma_range: float = 0.8,
    blur_angle_deg: float = 0.0,
    speckle_looks: float = 1.0,
    speckle_domain: SpeckleDomain = "representation",
    negative_policy: NegativePolicy = "clamp",
    clamp_output: bool = False,
    return_steps: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
    """Build an LR sample using one audited serial synthetic degradation.

    Order: FFT low-pass -> optional blur -> downsample -> non-negative policy ->
    multiplicative speckle -> optional [0, 1] clamp.

    This operator is not a sensor model by itself.  The default
    ``representation`` speckle domain is a controlled synthetic multiplier,
    not calibrated intensity/amplitude noise and not a physical ENL setting.
    Applying speckle to an already-speckled pseudo-target is a composite-noise
    assumption that must remain explicit in any experiment claim.
    """

    _validate_single_sar(sar_mr)
    if negative_policy not in {"clamp", "error", "preserve"}:
        raise ValueError(f"Unsupported negative policy: {negative_policy}")
    h, w = sar_mr.shape[-2:]
    if input_downsample_size > 0:
        target_h = min(int(input_downsample_size), h)
        target_w = min(int(input_downsample_size), w)
    else:
        if downsample_factor <= 0:
            raise ValueError("downsample_factor must be positive")
        target_h = max(1, h // int(downsample_factor))
        target_w = max(1, w // int(downsample_factor))
    keep_h = min(int(freq_window_size) if freq_window_size > 0 else target_h, h)
    keep_w = min(int(freq_window_size) if freq_window_size > 0 else target_w, w)

    steps: Dict[str, torch.Tensor] = {"input": sar_mr.clone()}
    sar_lr = sar_mr.clone()
    if use_freq_truncation:
        sar_lr = frequency_truncation_degrade(sar_lr, keep_h, keep_w)
        steps["frequency_truncation"] = sar_lr.clone()
    if use_anisotropic_blur:
        sar_lr = anisotropic_gaussian_blur(
            sar_lr,
            kernel_size=blur_kernel_size,
            sigma_azimuth=blur_sigma_azimuth,
            sigma_range=blur_sigma_range,
            angle_deg=blur_angle_deg,
        )
        steps["anisotropic_blur"] = sar_lr.clone()
    if (target_h, target_w) != (h, w):
        sar_lr = downsample_sar(sar_lr, target_h, target_w)
        steps["downsample"] = sar_lr.clone()

    if torch.any(sar_lr < 0):
        if negative_policy == "error":
            raise ValueError("Degradation produced negative values before multiplicative speckle")
        if negative_policy == "clamp":
            sar_lr = torch.clamp_min(sar_lr, 0.0)
            steps["pre_speckle_nonnegative"] = sar_lr.clone()
    # This is the single canonical snapshot used by the formal A2 cascade
    # protocol.  It is deliberately taken after the configured negative-value
    # policy and immediately before Gamma speckle.  Keeping this key explicit
    # avoids consumers guessing whether ``downsample`` already includes the
    # non-negative clamp for a particular input.
    steps["pre_speckle"] = sar_lr.clone()
    if use_speckle:
        if negative_policy == "preserve" and speckle_domain in {"intensity", "amplitude"}:
            raise ValueError("Negative values are invalid for intensity/amplitude speckle")
        sar_lr = gamma_speckle_noise(sar_lr, looks=speckle_looks, domain=speckle_domain)
        steps["speckle"] = sar_lr.clone()
    if clamp_output:
        sar_lr = torch.clamp(sar_lr, 0.0, 1.0)
        steps["output_clamp"] = sar_lr.clone()
    if not torch.isfinite(sar_lr).all():
        raise ValueError("Degradation produced NaN or Inf values")

    return (sar_lr, steps) if return_steps else sar_lr


__all__ = [
    "NormalizationStats",
    "anisotropic_gaussian_blur",
    "build_sar_lr_from_physical_degradation",
    "downsample_sar",
    "frequency_truncation_degrade",
    "gamma_speckle_noise",
    "inverse_preprocess_sar_tensor",
    "preprocess_sar_tensor",
    "sample_gamma_speckle_multiplier",
]
