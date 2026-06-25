import argparse
import math
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms


def _add_repo_paths():
    root = Path(__file__).resolve().parents[1]
    basicsr_path = root / "BasicSR"
    deepdespeckling_path = root / "deepdespeckling"
    mmcanet_src = root / "MMCANet" / "src"
    if str(basicsr_path) not in os.sys.path:
        os.sys.path.insert(0, str(basicsr_path))
    if str(deepdespeckling_path) not in os.sys.path:
        os.sys.path.insert(0, str(deepdespeckling_path))
    if str(mmcanet_src) not in os.sys.path:
        os.sys.path.insert(0, str(mmcanet_src))


_add_repo_paths()

from basicsr.metrics.psnr_ssim import calculate_ssim_pt  # noqa: E402
from deepdespeckling.sar2sar.sar2sar_denoiser import Sar2SarDenoiser  # noqa: E402
from deepdespeckling.utils.utils import denormalize_sar_image  # noqa: E402
from sar_degrade import build_sar_lr_from_physical_degradation, normalize_sar_log_clip  # noqa: E402
from train_attention_sr_common import high_frequency_extraction, preprocess_sar_tensor  # noqa: E402


class SFTLayer(nn.Module):
    def __init__(self, feat_ch: int, cond_ch: int):
        super().__init__()
        self.scale = nn.Conv2d(cond_ch, feat_ch, kernel_size=3, padding=1)
        self.shift = nn.Conv2d(cond_ch, feat_ch, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale = self.scale(cond)
        shift = self.shift(cond)
        return x * scale + shift


class SFTResBlock(nn.Module):
    def __init__(self, feat_ch: int, cond_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(feat_ch, feat_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(feat_ch, feat_ch, kernel_size=3, padding=1)
        self.sft1 = SFTLayer(feat_ch, cond_ch)
        self.sft2 = SFTLayer(feat_ch, cond_ch)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.sft1(self.conv1(x), cond), inplace=True)
        out = self.sft2(self.conv2(out), cond)
        return x + out


class SFTSRNet(nn.Module):
    def __init__(self, base_channels: int = 32, num_blocks: int = 6, scale: int = 2):
        super().__init__()
        self.scale = max(1, int(scale))
        self.sar_head = nn.Conv2d(1, base_channels, kernel_size=3, padding=1)
        self.opt_head = nn.Sequential(
            nn.Conv2d(6, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
        )

        self.blocks = nn.ModuleList([SFTResBlock(base_channels, base_channels) for _ in range(num_blocks)])

        if self.scale & (self.scale - 1):
            raise ValueError("scale must be a power-of-two for PixelShuffle upsampling.")
        up_layers = []
        num_ups = int(math.log2(self.scale))
        for i in range(num_ups):
            up_layers.extend(
                [
                    nn.Conv2d(base_channels, base_channels * 4, kernel_size=3, padding=1),
                    nn.PixelShuffle(2),
                ]
            )
            # Add ReLU after all but the last PixelShuffle to avoid checkerboard artifacts
            if i < num_ups - 1:
                up_layers.append(nn.ReLU(inplace=True))
        self.upsampler = nn.Sequential(*up_layers) if up_layers else nn.Identity()
        self.out_conv = nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)

    def forward(self, sar_lr: torch.Tensor, opt_hr: torch.Tensor) -> torch.Tensor:
        if opt_hr.shape[-2:] != sar_lr.shape[-2:]:
            opt_hr = F.interpolate(opt_hr, size=sar_lr.shape[-2:], mode="bilinear", align_corners=False)
        opt_hf = high_frequency_extraction(opt_hr)
        opt_cond = torch.cat([opt_hr, opt_hf], dim=1)
        x = self.sar_head(sar_lr)
        cond = self.opt_head(opt_cond)
        for blk in self.blocks:
            x = blk(x, cond)
        x = self.upsampler(x)
        out = self.out_conv(x)
        # Use clamp instead of sigmoid to preserve dynamic range
        return torch.clamp(out, min=0.0, max=1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SFT-conditioned early-fusion SR baseline")
    parser.add_argument("--sar-dir", default="data/sar_mini", help="Directory with SAR images")
    parser.add_argument("--opt-dir", default="data/opt_mini", help="Directory with optical images")
    parser.add_argument("--mode", choices=["train", "infer"], default="train")
    parser.add_argument("--weights", default="weights/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth", help="Path to model weights for inference or resume")
    parser.add_argument("--sar2sar-weights", default="/home/shy/AIR/SAR+AI/project/JDSR/deepdespeckling/deepdespeckling/sar2sar/saved_model/sar2sar.pth", help="Path to SAR2SAR weights")

    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--num-blocks", type=int, default=6)

    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lambda-ssim", type=float, default=0.1)

    parser.add_argument("--input-downsample-size", type=int, default=128)
    parser.add_argument("--downsample-factor", type=int, default=2)
    parser.add_argument("--lr-degradation-mode", choices=["legacy", "physical"], default="physical")
    parser.add_argument("--degrade-freq-truncation", action="store_true", default=True)
    parser.add_argument("--degrade-anisotropic-blur", action="store_true", default=False)
    parser.add_argument("--degrade-speckle", action="store_true", default=True)
    parser.add_argument("--degrade-freq-window-size", type=int, default=64)
    parser.add_argument("--degrade-blur-kernel-size", type=int, default=9)
    parser.add_argument("--degrade-blur-sigma-azimuth", type=float, default=2.0)
    parser.add_argument("--degrade-blur-sigma-range", type=float, default=0.8)
    parser.add_argument("--degrade-blur-angle-deg", type=float, default=0.0)
    parser.add_argument("--degrade-speckle-looks", type=float, default=2.0)
    parser.add_argument("--degrade-clamp-output", action="store_true", default=True)
    parser.add_argument("--sar-norm-mode", choices=["linear", "log_clip"], default="log_clip")
    parser.add_argument("--sar-log-scale", type=float, default=20.0)
    parser.add_argument("--sar-clip-low", type=float, default=0.1)
    parser.add_argument("--sar-clip-high", type=float, default=99.9)

    parser.add_argument("--sar2sar-input-gain", type=float, default=1.0)
    parser.add_argument("--sar2sar-patch-size", type=int, default=128)
    parser.add_argument("--sar2sar-stride-size", type=int, default=64)
    parser.add_argument("--sr-log-scale", type=float, default=20.0)
    parser.add_argument("--sr-clip-low", type=float, default=0.1)
    parser.add_argument("--sr-clip-high", type=float, default=99.9)

    parser.add_argument("--save-dir", default="./checkpoints/sft_baseline", help="Directory to save model checkpoints")
    parser.add_argument("--save-every", type=int, default=10)

    parser.add_argument("--out-dir", default="./outputs/sft_baseline_infer", help="Directory to save inference outputs")
    return parser.parse_args()


def resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def robust_load_state_dict(model: nn.Module, ckpt_path: str, device: torch.device | None = None, prefer_strict: bool = True):
    """Load a checkpoint flexibly.

    Handles checkpoints saved as {"state_dict": ...}, {"params": ...}, nested dicts,
    or a plain state_dict. Tries strict load first, then falls back to strict=False.
    Prints missing/unexpected keys to help the user.
    """
    if not ckpt_path or not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    raw = torch.load(ckpt_path, map_location="cpu")

    state = None
    if isinstance(raw, dict):
        # common wrapper keys
        for key in ("state_dict", "model", "params", "net", "network", "model_state_dict"):
            if key in raw and isinstance(raw[key], dict):
                state = raw[key]
                break

        # if not found, try to find a dict value that looks like a state_dict
        if state is None:
            for v in raw.values():
                if isinstance(v, dict) and any(isinstance(t, torch.Tensor) for t in v.values()):
                    state = v
                    break

        # if still not found, raw itself may already be a flat state_dict (name->tensor)
        if state is None:
            if all(isinstance(v, torch.Tensor) for v in raw.values()):
                state = raw
            else:
                # heuristics: check if keys look like module.param (contain dots)
                keys = list(raw.keys())[:10]
                if keys and all(isinstance(k, str) and ('.' in k or k.endswith('_weight') or k.endswith('_bias')) for k in keys):
                    state = raw

    if state is None:
        raise RuntimeError(f"Could not locate a state-dict in checkpoint {ckpt_path}. Top-level keys: {list(raw.keys()) if isinstance(raw, dict) else type(raw)}")

    # move tensors to device if requested
    if device is not None:
        for k, v in list(state.items()):
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    # try strict load first, then fallback to non-strict
    try:
        res = model.load_state_dict(state, strict=prefer_strict)
    except RuntimeError as e:
        print(f"Strict load failed: {e}. Retrying with strict=False.")
        res = model.load_state_dict(state, strict=False)

    # load_state_dict returns a NamedTuple in newer torch; print results
    try:
        missing = getattr(res, "missing_keys", None)
        unexpected = getattr(res, "unexpected_keys", None)
    except Exception:
        missing = None
        unexpected = None

    print(f"Loaded checkpoint '{ckpt_path}'. Missing keys: {missing}; Unexpected keys: {unexpected}")
    return res


def build_dataloader(args: argparse.Namespace, train: bool) -> DataLoader:
    dataset = SarOptDataset(
        sar_dir=args.sar_dir,
        opt_dir=args.opt_dir,
        input_downsample_size=args.input_downsample_size,
        downsample_factor=args.downsample_factor,
        augment=train,
        sar_log_scale=args.sar_log_scale,
        sar_clip_low=args.sar_clip_low,
        sar_clip_high=args.sar_clip_high,
        lr_degradation_mode=args.lr_degradation_mode,
        degrade_freq_truncation=args.degrade_freq_truncation,
        degrade_anisotropic_blur=args.degrade_anisotropic_blur,
        degrade_speckle=args.degrade_speckle,
        degrade_freq_window_size=args.degrade_freq_window_size,
        degrade_blur_kernel_size=args.degrade_blur_kernel_size,
        degrade_blur_sigma_azimuth=args.degrade_blur_sigma_azimuth,
        degrade_blur_sigma_range=args.degrade_blur_sigma_range,
        degrade_blur_angle_deg=args.degrade_blur_angle_deg,
        degrade_speckle_looks=args.degrade_speckle_looks,
        degrade_clamp_output=args.degrade_clamp_output,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=train,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def sar2sar_denoise_batch(
    sar_lr: torch.Tensor,
    sar2sar: Sar2SarDenoiser,
    args: argparse.Namespace,
) -> torch.Tensor:
    denoised = []
    with torch.no_grad():
        for idx in range(sar_lr.size(0)):
            sar_np = sar_lr[idx].squeeze(0).cpu().numpy()
            sar_np = denormalize_sar_image(sar_np)
            sar_np = sar_np * float(args.sar2sar_input_gain)
            despeckled = sar2sar.denoise_image(
                sar_np, patch_size=args.sar2sar_patch_size, stride_size=args.sar2sar_stride_size
            )
            sar_dn = torch.from_numpy(despeckled["denoised"]).float().unsqueeze(0)
            sar_dn_norm = normalize_sar_log_clip(
                sar_dn,
                log_scale=args.sr_log_scale,
                clip_low=args.sr_clip_low,
                clip_high=args.sr_clip_high,
            )
            # Ensure output is in valid range [0, 1]
            sar_dn_norm = torch.clamp(sar_dn_norm, min=0.0, max=1.0)
            denoised.append(sar_dn_norm)
    return torch.stack(denoised, dim=0)


class SarOptDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        sar_dir: str,
        opt_dir: str,
        input_downsample_size: int,
        downsample_factor: int,
        augment: bool,
        sar_log_scale: float,
        sar_clip_low: float,
        sar_clip_high: float,
        lr_degradation_mode: str,
        degrade_freq_truncation: bool,
        degrade_anisotropic_blur: bool,
        degrade_speckle: bool,
        degrade_freq_window_size: int,
        degrade_blur_kernel_size: int,
        degrade_blur_sigma_azimuth: float,
        degrade_blur_sigma_range: float,
        degrade_blur_angle_deg: float,
        degrade_speckle_looks: float,
        degrade_clamp_output: bool,
    ):
        self.sar_dir = sar_dir
        self.opt_dir = opt_dir
        self.input_downsample_size = input_downsample_size
        self.downsample_factor = downsample_factor
        self.augment = augment
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
            raise ValueError("lr_degradation_mode must be 'legacy' or 'physical'.")

        sar_names = {p.name for p in Path(sar_dir).iterdir() if p.is_file()}
        opt_names = {p.name for p in Path(opt_dir).iterdir() if p.is_file()}
        self.image_names = sorted(sar_names.intersection(opt_names))
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
            norm_mode="log_clip",
            log_scale=self.sar_log_scale,
            clip_low=self.sar_clip_low,
            clip_high=self.sar_clip_high,
        )
        opt_hr = self.to_tensor(Image.open(opt_path).convert("RGB"))

        if self.augment:
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
            sar_lr = F.interpolate(
                sar_mr.unsqueeze(0),
                size=(self.input_downsample_size, self.input_downsample_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        return sar_lr, opt_hr, sar_mr


def train(args: argparse.Namespace) -> None:
    device = resolve_device()
    model = SFTSRNet(base_channels=args.base_channels, num_blocks=args.num_blocks, scale=args.scale).to(device)
    # 实例化原本的去噪器
    sar2sar = Sar2SarDenoiser()
    
    # 强行将你特定的 sar2sar.pth 加载到它内部的模型中
    if hasattr(sar2sar, 'model') and args.sar2sar_weights:
        try:
            robust_load_state_dict(sar2sar.model, args.sar2sar_weights, device=device, prefer_strict=False)
            print(f"Successfully loaded SAR2SAR weights from {args.sar2sar_weights}")
        except Exception as e:
            print(f"Warning: failed to load SAR2SAR weights from {args.sar2sar_weights}: {e}")

    if args.weights and os.path.isfile(args.weights):
        try:
            robust_load_state_dict(model, args.weights, device=device, prefer_strict=False)
            print(f"Loaded SFT weights from {args.weights}")
        except Exception as e:
            print(f"Warning: failed to load SFT weights from {args.weights}: {e}")

    loader = build_dataloader(args, train=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for sar_lr, opt_hr, sar_hr in loader:
            sar_lr = sar2sar_denoise_batch(sar_lr, sar2sar, args).to(device)
            opt_hr = opt_hr.to(device)
            sar_hr = sar_hr.to(device)

            pred = model(sar_lr, opt_hr)
            loss_l1 = F.l1_loss(pred, sar_hr)
            ssim = calculate_ssim_pt(pred, sar_hr, crop_border=0, test_y_channel=False)
            loss_ssim = 1.0 - ssim.mean()
            loss = loss_l1 + args.lambda_ssim * loss_ssim

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += float(loss.item())

        avg_loss = total_loss / max(1, len(loader))
        print(f"Epoch {epoch}: loss={avg_loss:.6f}")

        if epoch % args.save_every == 0:
            save_path = os.path.join(args.save_dir, f"sft_sr_epoch_{epoch}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Saved {save_path}")


def load_gray(path: str) -> torch.Tensor:
    to_tensor = transforms.ToTensor()
    return to_tensor(Image.open(path).convert("L"))


def save_gray_image(path: str, gray: np.ndarray, use_16bit: bool) -> None:
    gray = np.clip(gray, 0.0, 1.0)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if use_16bit:
        img = (gray * 65535.0).round().astype(np.uint16)
        Image.fromarray(img, mode="I;16").save(path)
    else:
        img = (gray * 255.0).round().astype(np.uint8)
        Image.fromarray(img, mode="L").save(path)


def save_triptych(path: str, original: np.ndarray, denoised: np.ndarray, sr: np.ndarray) -> None:
    original = np.clip(original, 0.0, 1.0)
    denoised = np.clip(denoised, 0.0, 1.0)
    sr = np.clip(sr, 0.0, 1.0)

    sr_h, sr_w = sr.shape
    orig_img = Image.fromarray((original * 255.0).round().astype(np.uint8), mode="L")
    den_img = Image.fromarray((denoised * 255.0).round().astype(np.uint8), mode="L")
    sr_img = Image.fromarray((sr * 255.0).round().astype(np.uint8), mode="L")

    orig_img = orig_img.resize((sr_w, sr_h), resample=Image.BILINEAR)
    den_img = den_img.resize((sr_w, sr_h), resample=Image.BILINEAR)

    triptych = Image.new("L", (sr_w * 3, sr_h))
    triptych.paste(orig_img, (0, 0))
    triptych.paste(den_img, (sr_w, 0))
    triptych.paste(sr_img, (sr_w * 2, 0))

    os.makedirs(os.path.dirname(path), exist_ok=True)
    triptych.save(path)


def save_quad(path: str, original: np.ndarray, denoised: np.ndarray, opt_rgb: np.ndarray, sr: np.ndarray) -> None:
    """Save a 4-panel comparison: original SAR (L), denoised SAR (L), optical (RGB), SR SAR (L).

    All inputs expected in range [0,1]. opt_rgb should be HxWx3.
    """
    original = np.clip(original, 0.0, 1.0)
    denoised = np.clip(denoised, 0.0, 1.0)
    sr = np.clip(sr, 0.0, 1.0)
    opt_rgb = np.clip(opt_rgb, 0.0, 1.0)

    sr_h, sr_w = sr.shape
    # Resize gray panels to SR size
    orig_img = Image.fromarray((original * 255.0).round().astype(np.uint8), mode="L").resize((sr_w, sr_h), resample=Image.BILINEAR)
    den_img = Image.fromarray((denoised * 255.0).round().astype(np.uint8), mode="L").resize((sr_w, sr_h), resample=Image.BILINEAR)
    sr_img = Image.fromarray((sr * 255.0).round().astype(np.uint8), mode="L")

    # Optical: expect HxWx3
    opt_h, opt_w, _ = opt_rgb.shape
    opt_img = Image.fromarray((opt_rgb * 255.0).round().astype(np.uint8), mode="RGB").resize((sr_w, sr_h), resample=Image.BILINEAR)

    quad = Image.new("RGB", (sr_w * 4, sr_h))
    # paste grayscale as RGB by converting
    quad.paste(orig_img.convert("RGB"), (0, 0))
    quad.paste(den_img.convert("RGB"), (sr_w, 0))
    quad.paste(opt_img, (sr_w * 2, 0))
    quad.paste(sr_img.convert("RGB"), (sr_w * 3, 0))

    os.makedirs(os.path.dirname(path), exist_ok=True)
    quad.save(path)

def infer(args: argparse.Namespace) -> None:
    device = resolve_device()
    model = SFTSRNet(base_channels=args.base_channels, num_blocks=args.num_blocks, scale=args.scale).to(device)
    sar2sar = Sar2SarDenoiser()

    if not args.weights:
        raise FileNotFoundError("--weights is required for inference")
    
    # Use robust loader to accept different checkpoint formats
    try:
        print(f"\n=== Loading SFT model weights from {args.weights} ===")
        robust_load_state_dict(model, args.weights, device=device, prefer_strict=False)
        print("SFT model weights loaded successfully.\n")
    except Exception as e:
        raise RuntimeError(f"Failed to load SFT model weights from {args.weights}: {e}")
    
    # Attempt to load SAR2SAR weights into the internal denoiser if provided
    if hasattr(sar2sar, 'model') and args.sar2sar_weights:
        try:
            print(f"\n=== Loading SAR2SAR weights from {args.sar2sar_weights} ===")
            robust_load_state_dict(sar2sar.model, args.sar2sar_weights, device=device, prefer_strict=False)
            print("SAR2SAR weights loaded successfully.\n")
        except Exception as e:
            print(f"Warning: failed to load SAR2SAR weights from {args.sar2sar_weights}: {e}\n")
    model.eval()

    # --- 修改点 1: 确保所有子目录都存在 ---
    out_root = Path(args.out_dir)
    sub_dirs = ["original", "denoised", "sr", "triptych", "quad"]
    for sd in sub_dirs:
        (out_root / sd).mkdir(parents=True, exist_ok=True)

    names = sorted(set(p.name for p in Path(args.sar_dir).iterdir() if p.is_file()))
    
    for name in names:
        # 加载原始 SAR 图像
        sar_path = str(Path(args.sar_dir) / name)
        opt_path = str(Path(args.opt_dir) / name)
        
        # 预检查文件是否存在
        if not os.path.exists(opt_path):
            print(f"Skip {name}: Optical image not found.")
            continue

        orig_tensor = load_gray(sar_path)  # Tensor [1, H, W]
        opt = transforms.ToTensor()(Image.open(opt_path).convert("RGB")).unsqueeze(0).to(device)
        
        # --- 修改点 2: 规范化 SAR 处理流程 ---
        # sar2sar_denoise_batch 通常返回 tensor，确保它在正确的 device 上
        sar_denoised = sar2sar_denoise_batch(orig_tensor.unsqueeze(0), sar2sar, args).to(device)
        
        # 调试: 打印形状和统计信息
        print(f"\n[{name}] Inference debugging:")
        print(f"  sar_denoised: shape={sar_denoised.shape}, min={sar_denoised.min():.4f}, max={sar_denoised.max():.4f}, mean={sar_denoised.mean():.4f}")
        print(f"  opt: shape={opt.shape}, min={opt.min():.4f}, max={opt.max():.4f}, mean={opt.mean():.4f}")
        
        with torch.no_grad():
            pred = model(sar_denoised, opt)
        
        # 调试: 打印输出统计
        print(f"  pred: shape={pred.shape}, min={pred.min():.4f}, max={pred.max():.4f}, mean={pred.mean():.4f}")
        
        # 处理输出数据
        out = pred.squeeze().cpu().numpy()
        out = np.clip(out, 0.0, 1.0)
        
        # Min-max normalization to full [0, 1] range
        out_min, out_max = out.min(), out.max()
        if out_max > out_min + 1e-6:
            out = (out - out_min) / (out_max - out_min)
        out = np.clip(out, 0.0, 1.0)
        print(f"  clipped output: min={out.min():.4f}, max={out.max():.4f}, mean={out.mean():.4f}")
        
        # 保存 SR 结果 (16-bit)
        sr_img_16 = (out * 65535.0).round().astype(np.uint16)
        # 使用 Path 拼接路径更安全
        Image.fromarray(sr_img_16, mode="I;16").save(out_root / "sr" / name)

        # 保存中间结果用于对比 (8-bit)
        orig_np = orig_tensor.squeeze().numpy()
        den_np = sar_denoised.squeeze().cpu().numpy()

        # optical to HxWx3 numpy in [0,1]
        opt_np = opt.squeeze(0).permute(1, 2, 0).cpu().numpy()

        save_gray_image(str(out_root / "original" / name), orig_np, use_16bit=False)
        save_gray_image(str(out_root / "denoised" / name), den_np, use_16bit=False)
        save_triptych(str(out_root / "triptych" / name), orig_np, den_np, out)
        save_quad(str(out_root / "quad" / name), orig_np, den_np, opt_np, out)
        
        print(f"Processed and Saved: {name}")

def main() -> None:
    args = parse_args()
    if args.mode == "train":
        train(args)
    else:
        infer(args)


if __name__ == "__main__":
    main()
