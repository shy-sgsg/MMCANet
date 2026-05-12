import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def frequency_truncation_degrade(sar: torch.Tensor, window_h: int, window_w: int) -> torch.Tensor:
    """Keep only the low-frequency center in the 2D spectrum."""
    if sar.ndim != 3 or sar.shape[0] != 1:
        raise ValueError(f"Expected SAR tensor shape [1, H, W], got {tuple(sar.shape)}")

    h, w = sar.shape[-2:]
    keep_h = max(1, min(int(window_h), h))
    keep_w = max(1, min(int(window_w), w))

    cy, cx = h // 2, w // 2
    top = max(0, cy - keep_h // 2)
    left = max(0, cx - keep_w // 2)
    bottom = min(h, top + keep_h)
    right = min(w, left + keep_w)

    sar_fft = torch.fft.fftshift(torch.fft.fft2(sar.squeeze(0)))
    mask = torch.zeros((h, w), dtype=sar.dtype, device=sar.device)
    mask[top:bottom, left:right] = 1.0
    sar_fft = sar_fft * mask

    sar_ifft = torch.fft.ifft2(torch.fft.ifftshift(sar_fft))
    return torch.abs(sar_ifft).unsqueeze(0)


def anisotropic_gaussian_blur(
    sar: torch.Tensor,
    kernel_size: int = 9,
    sigma_azimuth: float = 2.0,
    sigma_range: float = 0.8,
    angle_deg: float = 0.0,
) -> torch.Tensor:
    """Approximate defocus in amplitude domain with anisotropic Gaussian blur."""
    if sar.ndim != 3 or sar.shape[0] != 1:
        raise ValueError(f"Expected SAR tensor shape [1, H, W], got {tuple(sar.shape)}")

    k = max(3, int(kernel_size))
    if k % 2 == 0:
        k += 1
    radius = k // 2

    sigma_x = max(float(sigma_azimuth), 1e-3)
    sigma_y = max(float(sigma_range), 1e-3)
    theta = math.radians(float(angle_deg))
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    coords = torch.arange(-radius, radius + 1, device=sar.device, dtype=sar.dtype)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")

    x_rot = xx * cos_t + yy * sin_t
    y_rot = -xx * sin_t + yy * cos_t

    kernel = torch.exp(-0.5 * ((x_rot / sigma_x) ** 2 + (y_rot / sigma_y) ** 2))
    kernel = kernel / (kernel.sum() + 1e-12)
    kernel = kernel.view(1, 1, k, k)

    blurred = F.conv2d(sar.unsqueeze(0), kernel, padding=radius)
    return blurred.squeeze(0)


def gamma_speckle_noise(sar: torch.Tensor, looks: float = 1.0) -> torch.Tensor:
    """Inject multiplicative Gamma speckle with mean 1 and variance 1/L."""
    if sar.ndim != 3 or sar.shape[0] != 1:
        raise ValueError(f"Expected SAR tensor shape [1, H, W], got {tuple(sar.shape)}")

    look_count = max(float(looks), 1e-3)
    gamma_dist = torch.distributions.Gamma(
        concentration=sar.new_tensor(look_count),
        rate=sar.new_tensor(look_count),
    )
    noise = gamma_dist.sample(sar.shape)
    return sar * noise


def downsample_sar(sar: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """Downsample SAR amplitude to the target spatial size."""
    if sar.ndim != 3 or sar.shape[0] != 1:
        raise ValueError(f"Expected SAR tensor shape [1, H, W], got {tuple(sar.shape)}")
    if target_h <= 0 or target_w <= 0:
        raise ValueError("Target size must be positive.")
    sar_lr = F.interpolate(
        sar.unsqueeze(0),
        size=(int(target_h), int(target_w)),
        mode="bilinear",
        align_corners=False,
    )
    return sar_lr.squeeze(0)


def build_sar_lr_from_physical_degradation(
    sar_mr: torch.Tensor,
    input_downsample_size: int = 64,
    use_freq_truncation: bool = True,
    use_anisotropic_blur: bool = False,
    use_speckle: bool = True,
    freq_window_size: int = 0,
    blur_kernel_size: int = 9,
    blur_sigma_azimuth: float = 2.0,
    blur_sigma_range: float = 0.8,
    blur_angle_deg: float = 0.0,
    speckle_looks: float = 1.0,
    clamp_output: bool = True,
) -> torch.Tensor:
    """Serial physical pipeline: frequency truncation -> blur -> downsample -> multiplicative speckle."""
    h, w = sar_mr.shape[-2:]
    if freq_window_size > 0:
        keep_h = min(int(freq_window_size), h)
        keep_w = min(int(freq_window_size), w)
    else:
        keep_h = min(max(1, int(input_downsample_size)), h)
        keep_w = min(max(1, int(input_downsample_size)), w)

    sar_lr = sar_mr.clone()

    if use_freq_truncation:
        sar_lr = frequency_truncation_degrade(sar_lr, window_h=keep_h, window_w=keep_w)
    if use_anisotropic_blur:
        sar_lr = anisotropic_gaussian_blur(
            sar_lr,
            kernel_size=blur_kernel_size,
            sigma_azimuth=blur_sigma_azimuth,
            sigma_range=blur_sigma_range,
            angle_deg=blur_angle_deg,
        )

    if input_downsample_size > 0 and (h != input_downsample_size or w != input_downsample_size):
        sar_lr = downsample_sar(sar_lr, target_h=input_downsample_size, target_w=input_downsample_size)

    if use_speckle:
        sar_lr = gamma_speckle_noise(sar_lr, looks=speckle_looks)

    if clamp_output:
        sar_lr = torch.clamp(sar_lr, 0.0, 1.0)
    return sar_lr


def normalize_sar_log_clip(
    sar: torch.Tensor,
    log_scale: float = 20.0,
    clip_low: float = 1.0,
    clip_high: float = 99.0,
    clamp_max: Optional[float] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Normalize SAR dynamic range using log1p + percentile clipping to [0, 1]."""
    if clamp_max is None:
        sar = torch.clamp(sar, min=0.0)
    else:
        sar = torch.clamp(sar, min=0.0, max=float(clamp_max))

    scale = max(float(log_scale), eps)
    sar = torch.log1p(scale * sar)

    q_low = max(0.0, min(float(clip_low), 100.0)) / 100.0
    q_high = max(0.0, min(float(clip_high), 100.0)) / 100.0
    if q_high <= q_low:
        q_low, q_high = 0.01, 0.99

    flat = sar.reshape(-1)
    lo = torch.quantile(flat, q_low)
    hi = torch.quantile(flat, q_high)

    sar = torch.clamp(sar, min=lo, max=hi)
    return (sar - lo) / (hi - lo + eps)


# def to_3ch(gray: torch.Tensor) -> torch.Tensor:
#     """Repeat a [1, H, W] tensor to [3, H, W]."""
#     if gray.ndim != 3 or gray.shape[0] != 1:
#         raise ValueError(f"Expected gray tensor shape [1, H, W], got {tuple(gray.shape)}")
#     return gray.repeat(3, 1, 1)

def to_3ch(gray: torch.Tensor) -> torch.Tensor:
    # 如果输入是 [B, 1, H, W]
    if gray.ndim == 4:
        return gray.repeat(1, 3, 1, 1)
    # 如果输入是 [1, H, W] 或 [H, W]
    elif gray.ndim == 3:
        return gray.repeat(3, 1, 1)
    elif gray.ndim == 2:
        return gray.unsqueeze(0).repeat(3, 1, 1)
    else:
        raise ValueError(f"Unexpected tensor shape {tuple(gray.shape)}")


def to_1ch(rgb: torch.Tensor) -> torch.Tensor:
    """Average a [3, H, W] tensor to [1, H, W]."""
    if rgb.ndim != 3 or rgb.shape[0] != 3:
        raise ValueError(f"Expected rgb tensor shape [3, H, W], got {tuple(rgb.shape)}")
    return rgb.mean(dim=0, keepdim=True)
