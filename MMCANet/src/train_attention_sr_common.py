import math
import os
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models import VGG16_Weights, vgg16


def preprocess_sar_tensor(
    sar: torch.Tensor,
    norm_mode: str = "log_clip",
    log_scale: float = 20.0,
    clip_low: float = 0.1,
    clip_high: float = 99.9,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Normalize SAR dynamic range with soft log+percentile (preserves extreme values).
    
    Changed from hard clip at 1%-99% to soft 0.1%-99.9% to preserve sharp scatterers.
    """
    sar = torch.clamp(sar, 0.0, 1.0)

    if norm_mode == "linear":
        return sar
    if norm_mode != "log_clip":
        raise ValueError(f"Unsupported SAR normalization mode: {norm_mode}")

    # Soft log transform: converts multiplicative speckle to additive
    scale = max(float(log_scale), eps)
    sar = torch.log1p(scale * sar) / torch.log1p(sar.new_tensor(scale))

    # Very soft percentile clipping (0.1%-99.9%) to preserve extreme scatterers
    q_low = max(0.0, min(float(clip_low), 100.0)) / 100.0
    q_high = max(0.0, min(float(clip_high), 100.0)) / 100.0
    if q_high <= q_low:
        q_low, q_high = 0.001, 0.999

    flat = sar.reshape(-1)
    lo = torch.quantile(flat, q_low)
    hi = torch.quantile(flat, q_high)

    # Normalize but allow values to exceed 1.0 (rare extremes are preserved)
    sar = (sar - lo) / (hi - lo + eps)
    return sar


def build_sar_lr_from_mr(sar_mr: torch.Tensor, input_downsample_size: int = 64) -> torch.Tensor:
    h, w = sar_mr.shape[-2:]
    sar_lr = F.interpolate(
        sar_mr.unsqueeze(0),
        size=(input_downsample_size, input_downsample_size),
        mode="bilinear",
        align_corners=False,
    )
    sar_lr = F.interpolate(
        sar_lr,
        size=(h, w),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    return sar_lr


def frequency_truncation_degrade(sar: torch.Tensor, window_h: int, window_w: int) -> torch.Tensor:
    """Step 1: degrade resolution by keeping only low-frequency center in 2D spectrum."""
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
    """Step 2: approximate defocus in amplitude domain with anisotropic Gaussian blur."""
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
    """Step 3: inject multiplicative Gamma speckle with mean 1 and variance 1/L."""
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

    if input_downsample_size > 0:
        target_h = min(int(input_downsample_size), h)
        target_w = min(int(input_downsample_size), w)
    else:
        factor = max(1, int(downsample_factor))
        target_h = max(1, h // factor)
        target_w = max(1, w // factor)

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
    if target_h != h or target_w != w:
        sar_lr = downsample_sar(sar_lr, target_h=target_h, target_w=target_w)
    if use_speckle:
        sar_lr = gamma_speckle_noise(sar_lr, looks=speckle_looks)

    if clamp_output:
        sar_lr = torch.clamp(sar_lr, 0.0, 1.0)
    return sar_lr


class SAROptDataset(Dataset):
    def __init__(
        self,
        sar_dir: str,
        opt_dir: str,
        input_downsample_size: int = 0,
        downsample_factor: int = 2,
        augment: bool = True,
        sar_norm_mode: str = "log_clip",
        sar_log_scale: float = 20.0,
        sar_clip_low: float = 1.0,
        sar_clip_high: float = 99.0,
        lr_degradation_mode: str = "legacy",
        degrade_freq_truncation: bool = True,
        degrade_anisotropic_blur: bool = False,
        degrade_speckle: bool = True,
        degrade_freq_window_size: int = 0,
        degrade_blur_kernel_size: int = 9,
        degrade_blur_sigma_azimuth: float = 2.0,
        degrade_blur_sigma_range: float = 0.8,
        degrade_blur_angle_deg: float = 0.0,
        degrade_speckle_looks: float = 1.0,
        degrade_clamp_output: bool = True,
    ):
        self.sar_dir = sar_dir
        self.opt_dir = opt_dir
        self.input_downsample_size = input_downsample_size
        self.downsample_factor = downsample_factor
        self.augment = augment
        self.sar_norm_mode = sar_norm_mode
        self.sar_log_scale = sar_log_scale
        self.sar_clip_low = sar_clip_low
        self.sar_clip_high = sar_clip_high
        self.lr_degradation_mode = lr_degradation_mode
        self.degrade_freq_truncation = degrade_freq_truncation
        self.degrade_anisotropic_blur = degrade_anisotropic_blur
        self.degrade_speckle = degrade_speckle
        self.degrade_freq_window_size = degrade_freq_window_size
        self.degrade_blur_kernel_size = degrade_blur_kernel_size
        self.degrade_blur_sigma_azimuth = degrade_blur_sigma_azimuth
        self.degrade_blur_sigma_range = degrade_blur_sigma_range
        self.degrade_blur_angle_deg = degrade_blur_angle_deg
        self.degrade_speckle_looks = degrade_speckle_looks
        self.degrade_clamp_output = degrade_clamp_output

        if self.lr_degradation_mode not in {"legacy", "physical"}:
            raise ValueError(
                f"Unsupported lr_degradation_mode: {self.lr_degradation_mode}. "
                "Expected one of {'legacy', 'physical'}."
            )

        sar_names = set(os.listdir(sar_dir))
        opt_names = set(os.listdir(opt_dir))
        self.image_names = sorted(list(sar_names.intersection(opt_names)))
        if not self.image_names:
            raise RuntimeError("No matched SAR/OPT image pairs found.")

        self.to_tensor = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, idx: int):
        img_name = self.image_names[idx]
        sar_path = os.path.join(self.sar_dir, img_name)
        opt_path = os.path.join(self.opt_dir, img_name)

        sar_mr = self.to_tensor(Image.open(sar_path).convert("L"))
        sar_mr = preprocess_sar_tensor(
            sar=sar_mr,
            norm_mode=self.sar_norm_mode,
            log_scale=self.sar_log_scale,
            clip_low=self.sar_clip_low,
            clip_high=self.sar_clip_high,
        )
        opt_hr = self.to_tensor(Image.open(opt_path).convert("RGB"))

        if self.augment:
            # Keep modality alignment by applying the same random flip to both streams.
            if torch.rand(1).item() < 0.5:
                sar_mr = torch.flip(sar_mr, dims=[2])
                opt_hr = torch.flip(opt_hr, dims=[2])
            if torch.rand(1).item() < 0.5:
                sar_mr = torch.flip(sar_mr, dims=[1])
                opt_hr = torch.flip(opt_hr, dims=[1])

        if self.lr_degradation_mode == "physical":
            sar_lr = build_sar_lr_from_physical_degradation(
                sar_mr=sar_mr,
                input_downsample_size=self.input_downsample_size,
                downsample_factor=self.downsample_factor,
                use_freq_truncation=self.degrade_freq_truncation,
                use_anisotropic_blur=self.degrade_anisotropic_blur,
                use_speckle=self.degrade_speckle,
                freq_window_size=self.degrade_freq_window_size,
                blur_kernel_size=self.degrade_blur_kernel_size,
                blur_sigma_azimuth=self.degrade_blur_sigma_azimuth,
                blur_sigma_range=self.degrade_blur_sigma_range,
                blur_angle_deg=self.degrade_blur_angle_deg,
                speckle_looks=self.degrade_speckle_looks,
                clamp_output=self.degrade_clamp_output,
            )
        else:
            if self.input_downsample_size > 0:
                target_size = self.input_downsample_size
            else:
                factor = max(1, int(self.downsample_factor))
                target_size = max(1, sar_mr.shape[-1] // factor)
            sar_lr = build_sar_lr_from_mr(sar_mr=sar_mr, input_downsample_size=target_size)

        return sar_lr, opt_hr, sar_mr


class SpatialCrossAttention(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        reduced_channels = max(in_channels // 8, 1)
        self.query_conv = nn.Conv2d(in_channels, reduced_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, reduced_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, sar_feat: torch.Tensor, opt_feat: torch.Tensor) -> torch.Tensor:
        b, c, h, w = sar_feat.size()
        n = h * w

        proj_query = self.query_conv(sar_feat).view(b, -1, n).permute(0, 2, 1)
        proj_key = self.key_conv(opt_feat).view(b, -1, n)
        proj_value = self.value_conv(opt_feat).view(b, -1, n)

        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(b, c, h, w)
        return sar_feat + self.gamma * out


class ShallowGuideFusion(nn.Module):
    """Memory-friendly top-level fusion using channel gating (no NxN attention map)."""

    def __init__(self, in_channels: int):
        super().__init__()
        hidden = max(in_channels // 4, 1)
        self.opt_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, in_channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, sar_feat: torch.Tensor, opt_feat: torch.Tensor) -> torch.Tensor:
        guided_opt = self.opt_proj(opt_feat)
        channel_gate = self.gate(opt_feat)
        return sar_feat + self.gamma * (guided_opt * channel_gate)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpsampleConv(nn.Module):
    """Checkerboard-free upsampling: bilinear interpolation followed by convolution."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Encoder(nn.Module):
    def __init__(self, in_channels: int, base_channels: int = 32):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.enc4 = ConvBlock(base_channels * 4, base_channels * 8)
        self.bottleneck = ConvBlock(base_channels * 8, base_channels * 16)

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        x4 = self.enc4(self.pool(x3))
        xb = self.bottleneck(self.pool(x4))
        return [x1, x2, x3, x4], xb


class TwoStreamFrequencyGuidedUNet(nn.Module):
    def __init__(self, base_channels: int = 32, upscale: int = 2):
        super().__init__()
        self.upscale = max(1, int(upscale))
        self.sar_encoder = Encoder(in_channels=1, base_channels=base_channels)
        self.opt_encoder = Encoder(in_channels=3, base_channels=base_channels)

        self.deep_cross_attn = SpatialCrossAttention(in_channels=base_channels * 16)

        self.up4 = UpsampleConv(base_channels * 16, base_channels * 8)
        self.dec4 = ConvBlock(base_channels * 16, base_channels * 8)

        self.up3 = UpsampleConv(base_channels * 8, base_channels * 4)
        self.dec3 = ConvBlock(base_channels * 8, base_channels * 4)

        self.up2 = UpsampleConv(base_channels * 4, base_channels * 2)
        self.dec2 = ConvBlock(base_channels * 4, base_channels * 2)

        self.up1 = UpsampleConv(base_channels * 2, base_channels)
        self.dec1 = ConvBlock(base_channels * 2, base_channels)

        self.shallow_fusion = ShallowGuideFusion(in_channels=base_channels)
        if self.upscale == 1:
            self.sr_upsample = nn.Identity()
        else:
            self.sr_upsample = nn.Sequential(
                nn.Conv2d(base_channels, base_channels * self.upscale * self.upscale, kernel_size=3, padding=1),
                nn.PixelShuffle(self.upscale),
                nn.ReLU(inplace=True),
            )
        self.out_conv = nn.Conv2d(base_channels, 1, kernel_size=1)

    def forward(self, sar_input: torch.Tensor, opt_input: torch.Tensor) -> torch.Tensor:
        sar_skips, sar_bottleneck = self.sar_encoder(sar_input)
        opt_skips, opt_bottleneck = self.opt_encoder(opt_input)

        fused_bottleneck = self.deep_cross_attn(sar_bottleneck, opt_bottleneck)

        d4 = self.up4(fused_bottleneck)
        d4 = self.dec4(torch.cat([d4, sar_skips[3]], dim=1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, sar_skips[2]], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, sar_skips[1]], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, sar_skips[0]], dim=1))

        d1 = self.shallow_fusion(d1, opt_skips[0])
        d1 = self.sr_upsample(d1)
        return torch.sigmoid(self.out_conv(d1))


class VGGFeatureLoss(nn.Module):
    def __init__(self, layer_ids=(13, 20), use_imagenet_weights: bool = True, input_size: int = 128):
        super().__init__()

        weights = None
        if use_imagenet_weights:
            try:
                weights = VGG16_Weights.IMAGENET1K_V1
            except Exception:
                weights = None

        try:
            vgg_model = vgg16(weights=weights)
        except Exception as exc:
            print(f"[WARN] Failed to load VGG pretrained weights: {exc}")
            print("[WARN] Falling back to random-initialized VGG for feature loss.")
            vgg_model = vgg16(weights=None)

        self.features = vgg_model.features.eval()
        for param in self.features.parameters():
            param.requires_grad = False

        self.layer_ids = set(layer_ids)
        self.max_layer = max(layer_ids)
        self.input_size = input_size
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.input_size and pred.shape[-1] != self.input_size:
            pred = F.interpolate(pred, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
            target = F.interpolate(target, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)

        pred_rgb = pred.repeat(1, 3, 1, 1)
        target_rgb = target.repeat(1, 3, 1, 1)

        pred_norm = (pred_rgb - self.mean) / self.std
        target_norm = (target_rgb - self.mean) / self.std

        loss = pred_norm.new_tensor(0.0)
        x = pred_norm
        y = target_norm
        for idx, layer in enumerate(self.features):
            x = layer(x)
            y = layer(y)
            if idx in self.layer_ids:
                loss = loss + F.l1_loss(x, y)
            if idx >= self.max_layer:
                break
        return loss


def gradient_magnitude(x: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    kernel_x = x.new_tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
    kernel_y = x.new_tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3)

    grad_x = F.conv2d(x, kernel_x, padding=1)
    grad_y = F.conv2d(x, kernel_y, padding=1)
    return torch.sqrt(grad_x * grad_x + grad_y * grad_y + epsilon)


def normalize_map(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x_min = x.amin(dim=(2, 3), keepdim=True)
    x_max = x.amax(dim=(2, 3), keepdim=True)
    return (x - x_min) / (x_max - x_min + eps)


def low_frequency_l1_loss(sar_out: torch.Tensor, sar_mr: torch.Tensor, down_factor: int = 4, eps: float = 1e-3) -> torch.Tensor:
    """Low-frequency loss using Charbonnier (smooth L1) for consistency with main loss."""
    if down_factor <= 1:
        return charbonnier_loss(sar_out, sar_mr, eps=eps)

    sar_out_low = F.avg_pool2d(sar_out, kernel_size=down_factor, stride=down_factor)
    sar_mr_low = F.avg_pool2d(sar_mr, kernel_size=down_factor, stride=down_factor)
    return charbonnier_loss(sar_out_low, sar_mr_low, eps=eps)


def high_frequency_extraction(x: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    """Extract high-frequency components via Laplacian or LoG.
    
    Used to isolate optical edges for injection into SAR features.
    Removes low-frequency color/shadow artifacts before fusion.
    """
    # Simple Laplacian kernel for high-pass filtering
    laplacian = x.new_tensor(
        [[0.0, -1.0, 0.0],
         [-1.0, 4.0, -1.0],
         [0.0, -1.0, 0.0]],
        dtype=x.dtype
    ).view(1, 1, 3, 3)
    
    # Apply to each channel independently
    b, c, h, w = x.shape
    x_flat = x.view(b * c, 1, h, w)
    hf = F.conv2d(x_flat, laplacian, padding=1)
    hf = hf.view(b, c, h, w)
    
    # Normalize to prevent extreme gradients
    hf_std = hf.std(dim=(2, 3), keepdim=True) + 1e-6
    hf = hf / (hf_std * 2.0 + 1e-6)
    return hf


def noise_signal_decorrelation_loss(signal: torch.Tensor, noisy_observation: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    noise = noisy_observation - signal

    signal_center = signal - signal.mean(dim=(2, 3), keepdim=True)
    noise_center = noise - noise.mean(dim=(2, 3), keepdim=True)

    cov = (signal_center * noise_center).mean(dim=(2, 3))
    signal_std = torch.sqrt((signal_center * signal_center).mean(dim=(2, 3)) + eps)
    noise_std = torch.sqrt((noise_center * noise_center).mean(dim=(2, 3)) + eps)

    corr = cov / (signal_std * noise_std + eps)
    return corr.abs().mean()


def total_variation_loss(x: torch.Tensor) -> torch.Tensor:
    grad_x = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
    grad_y = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
    return grad_x.mean() + grad_y.mean()


def charbonnier_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Charbonnier loss (smooth L1) for smoother gradients and sharper edges.
    
    Transitions from L1 (for large errors, robust to outliers) to L2 (for small errors,
    smooth convergence). Eliminates blockiness and oscillation near convergence.
    """
    diff = pred - target
    return torch.mean(torch.sqrt(diff * diff + eps * eps))
