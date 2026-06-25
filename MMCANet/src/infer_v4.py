import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

from train_attention_sr_common import (
    TwoStreamFrequencyGuidedUNet,
    anisotropic_gaussian_blur,
    build_sar_lr_from_physical_degradation,
    downsample_sar,
    frequency_truncation_degrade,
    gamma_speckle_noise,
    preprocess_sar_tensor,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference for MMCANet v4")
    parser.add_argument("--sar-dir", default="../data/sar", help="Directory with SAR images")
    parser.add_argument("--opt-dir", default="../data/opt", help="Directory with optical images")
    parser.add_argument("--weights", required=True, help="Path to model weights")
    parser.add_argument("--out-dir", default="./outputs/v4", help="Output directory")
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--upscale", type=int, default=2)

    parser.add_argument("--use-physical", action="store_true", default=False)
    parser.add_argument("--input-downsample-size", type=int, default=128)
    parser.add_argument("--downsample-factor", type=int, default=2)
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

    parser.add_argument("--debug-physical", action="store_true", default=False)
    parser.add_argument("--debug-max-images", type=int, default=2)

    parser.add_argument("--sar-norm-mode", choices=["linear", "log_clip"], default="log_clip")
    parser.add_argument("--sar-log-scale", type=float, default=20.0)
    parser.add_argument("--sar-clip-low", type=float, default=0.1)
    parser.add_argument("--sar-clip-high", type=float, default=99.9)
    parser.add_argument("--stretch-low", type=float, default=2.0)
    parser.add_argument("--stretch-high", type=float, default=98.0)
    return parser.parse_args()


def list_pairs(sar_dir: str, opt_dir: str):
    sar_names = {p.name for p in Path(sar_dir).iterdir() if p.is_file()}
    opt_names = {p.name for p in Path(opt_dir).iterdir() if p.is_file()}
    names = sorted(sar_names.intersection(opt_names))
    if not names:
        raise RuntimeError("No matched SAR/OPT image pairs found.")
    return names


def save_gray_image(path: str, gray: np.ndarray, use_16bit: bool = False) -> None:
    gray = np.clip(gray, 0.0, 1.0)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if use_16bit:
        img = (gray * 65535.0).round().astype(np.uint16)
        Image.fromarray(img, mode="I;16").save(path)
    else:
        img = (gray * 255.0).round().astype(np.uint8)
        Image.fromarray(img, mode="L").save(path)


def save_rgb_image(path: str, rgb: np.ndarray) -> None:
    rgb = np.clip(rgb, 0.0, 1.0)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = (rgb * 255.0).round().astype(np.uint8)
    Image.fromarray(img, mode="RGB").save(path)


def save_triptych(path: str, left: np.ndarray, mid: np.ndarray, right: np.ndarray, target_size=None) -> None:
    left = np.clip(left, 0.0, 1.0)
    mid = np.clip(mid, 0.0, 1.0)
    right = np.clip(right, 0.0, 1.0)

    if target_size is None:
        h, w = right.shape[:2]
    else:
        h, w = target_size
    if left.ndim == 2:
        left_img = Image.fromarray((left * 255.0).round().astype(np.uint8), mode="L").convert("RGB")
    else:
        left_img = Image.fromarray((left * 255.0).round().astype(np.uint8), mode="RGB")
    if mid.ndim == 2:
        mid_img = Image.fromarray((mid * 255.0).round().astype(np.uint8), mode="L").convert("RGB")
    else:
        mid_img = Image.fromarray((mid * 255.0).round().astype(np.uint8), mode="RGB")
    if right.ndim == 2:
        right_img = Image.fromarray((right * 255.0).round().astype(np.uint8), mode="L").convert("RGB")
    else:
        right_img = Image.fromarray((right * 255.0).round().astype(np.uint8), mode="RGB")

    left_img = left_img.resize((w, h), resample=Image.BILINEAR)
    mid_img = mid_img.resize((w, h), resample=Image.BILINEAR)
    right_img = right_img.resize((w, h), resample=Image.BILINEAR)

    canvas = Image.new("RGB", (w * 3, h))
    canvas.paste(left_img, (0, 0))
    canvas.paste(mid_img, (w, 0))
    canvas.paste(right_img, (w * 2, 0))

    os.makedirs(os.path.dirname(path), exist_ok=True)
    canvas.save(path)


def save_quad(path: str, a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> None:
    a = np.clip(a, 0.0, 1.0)
    b = np.clip(b, 0.0, 1.0)
    c = np.clip(c, 0.0, 1.0)
    d = np.clip(d, 0.0, 1.0)

    h, w = c.shape[:2]
    def to_rgb(x):
        if x.ndim == 2:
            return Image.fromarray((x * 255.0).round().astype(np.uint8), mode="L").convert("RGB")
        return Image.fromarray((x * 255.0).round().astype(np.uint8), mode="RGB")

    img_a = to_rgb(a).resize((w, h), resample=Image.BILINEAR)
    img_b = to_rgb(b).resize((w, h), resample=Image.BILINEAR)
    img_c = to_rgb(c)
    img_d = to_rgb(d).resize((w, h), resample=Image.BILINEAR)

    canvas = Image.new("RGB", (w * 2, h * 2))
    canvas.paste(img_a, (0, 0))
    canvas.paste(img_b, (w, 0))
    canvas.paste(img_c, (0, h))
    canvas.paste(img_d, (w, h))

    os.makedirs(os.path.dirname(path), exist_ok=True)
    canvas.save(path)


def percentile_stretch(gray: np.ndarray, low: float = 2.0, high: float = 98.0, eps: float = 1e-6) -> np.ndarray:
    low = float(np.clip(low, 0.0, 100.0))
    high = float(np.clip(high, 0.0, 100.0))
    if high <= low:
        low, high = 2.0, 98.0
    lo = np.quantile(gray, low / 100.0)
    hi = np.quantile(gray, high / 100.0)
    gray = np.clip(gray, lo, hi)
    return (gray - lo) / (hi - lo + eps)


def tensor_stats(x: torch.Tensor) -> str:
    x_detached = x.detach()
    return (
        f"shape={tuple(x_detached.shape)} min={x_detached.min().item():.5f} "
        f"max={x_detached.max().item():.5f} mean={x_detached.mean().item():.5f} "
        f"std={x_detached.std().item():.5f}"
    )


def build_sar_lr_debug(
    sar_mr: torch.Tensor,
    input_downsample_size: int,
    downsample_factor: int,
    use_freq_truncation: bool,
    use_anisotropic_blur: bool,
    use_speckle: bool,
    freq_window_size: int,
    blur_kernel_size: int,
    blur_sigma_azimuth: float,
    blur_sigma_range: float,
    blur_angle_deg: float,
    speckle_looks: float,
    clamp_output: bool,
):
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

    steps = [("input", sar_mr.clone())]
    sar_lr = sar_mr.clone()

    if use_freq_truncation:
        sar_lr = frequency_truncation_degrade(sar_lr, window_h=keep_h, window_w=keep_w)
        steps.append(("freq_trunc", sar_lr))
    if use_anisotropic_blur:
        sar_lr = anisotropic_gaussian_blur(
            sar_lr,
            kernel_size=blur_kernel_size,
            sigma_azimuth=blur_sigma_azimuth,
            sigma_range=blur_sigma_range,
            angle_deg=blur_angle_deg,
        )
        steps.append(("blur", sar_lr))
    if target_h != h or target_w != w:
        sar_lr = downsample_sar(sar_lr, target_h=target_h, target_w=target_w)
        steps.append(("downsample", sar_lr))
    if use_speckle:
        sar_lr = gamma_speckle_noise(sar_lr, looks=speckle_looks)
        steps.append(("speckle", sar_lr))

    if clamp_output:
        sar_lr = torch.clamp(sar_lr, 0.0, 1.0)
        steps.append(("clamp", sar_lr))
    return sar_lr, steps


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TwoStreamFrequencyGuidedUNet(base_channels=args.base_channels, upscale=args.upscale).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    names = list_pairs(args.sar_dir, args.opt_dir)

    out_root = Path(args.out_dir)
    out_orig = out_root / "original"
    out_deg = out_root / "degraded"
    out_out = out_root / "output"
    out_opt = out_root / "optical"
    out_triptych = out_root / "triptych"
    out_quad = out_root / "quad"
    out_debug = out_root / "debug_physical"

    to_tensor = transforms.ToTensor()

    debug_budget = max(0, int(args.debug_max_images))
    for name in names:
        sar_path = Path(args.sar_dir) / name
        opt_path = Path(args.opt_dir) / name

        sar_mr = to_tensor(Image.open(sar_path).convert("L"))
        sar_mr = preprocess_sar_tensor(
            sar=sar_mr,
            norm_mode=args.sar_norm_mode,
            log_scale=args.sar_log_scale,
            clip_low=args.sar_clip_low,
            clip_high=args.sar_clip_high,
        )
        opt_hr = to_tensor(Image.open(opt_path).convert("RGB"))

        if args.use_physical:
            if args.debug_physical and debug_budget > 0:
                sar_lr, steps = build_sar_lr_debug(
                    sar_mr=sar_mr,
                    input_downsample_size=args.input_downsample_size,
                    downsample_factor=args.downsample_factor,
                    use_freq_truncation=args.degrade_freq_truncation,
                    use_anisotropic_blur=args.degrade_anisotropic_blur,
                    use_speckle=args.degrade_speckle,
                    freq_window_size=args.degrade_freq_window_size,
                    blur_kernel_size=args.degrade_blur_kernel_size,
                    blur_sigma_azimuth=args.degrade_blur_sigma_azimuth,
                    blur_sigma_range=args.degrade_blur_sigma_range,
                    blur_angle_deg=args.degrade_blur_angle_deg,
                    speckle_looks=args.degrade_speckle_looks,
                    clamp_output=args.degrade_clamp_output,
                )
                for step_name, step_tensor in steps:
                    print(f"[DEBUG] {name} | {step_name}: {tensor_stats(step_tensor)}")
                    step_np = step_tensor.squeeze(0).cpu().numpy()
                    save_gray_image(str(out_debug / step_name / name), step_np, use_16bit=False)
                debug_budget -= 1
            else:
                sar_lr = build_sar_lr_from_physical_degradation(
                    sar_mr=sar_mr,
                    input_downsample_size=args.input_downsample_size,
                    downsample_factor=args.downsample_factor,
                    use_freq_truncation=args.degrade_freq_truncation,
                    use_anisotropic_blur=args.degrade_anisotropic_blur,
                    use_speckle=args.degrade_speckle,
                    freq_window_size=args.degrade_freq_window_size,
                    blur_kernel_size=args.degrade_blur_kernel_size,
                    blur_sigma_azimuth=args.degrade_blur_sigma_azimuth,
                    blur_sigma_range=args.degrade_blur_sigma_range,
                    blur_angle_deg=args.degrade_blur_angle_deg,
                    speckle_looks=args.degrade_speckle_looks,
                    clamp_output=args.degrade_clamp_output,
                )
        else:
            sar_lr = sar_mr

        sar_in = sar_lr
        opt_in = opt_hr
        if args.use_physical and opt_in.shape[-2:] != sar_in.shape[-2:]:
            opt_in = F.interpolate(
                opt_in.unsqueeze(0),
                size=sar_in.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        with torch.no_grad():
            sar_out = model(sar_in.unsqueeze(0).to(device), opt_in.unsqueeze(0).to(device))
        sar_out = sar_out.squeeze(0).squeeze(0).cpu().numpy()

        sar_mr_np = sar_mr.squeeze(0).numpy()
        opt_np = opt_hr.permute(1, 2, 0).numpy()

        sar_mr_np = percentile_stretch(sar_mr_np, low=args.stretch_low, high=args.stretch_high)
        sar_out = percentile_stretch(sar_out, low=args.stretch_low, high=args.stretch_high)

        save_gray_image(str(out_orig / name), sar_mr_np, use_16bit=False)
        save_rgb_image(str(out_opt / name), opt_np)
        save_gray_image(str(out_out / name), sar_out, use_16bit=False)

        if args.use_physical:
            sar_lr_np = sar_lr.squeeze(0).numpy()
            sar_lr_np = percentile_stretch(sar_lr_np, low=args.stretch_low, high=args.stretch_high)
            save_gray_image(str(out_deg / name), sar_lr_np, use_16bit=False)
            save_quad(
                str(out_quad / name),
                sar_mr_np,
                sar_lr_np,
                sar_out,
                opt_np,
            )
        else:
            out_h, out_w = sar_out.shape
            save_triptych(
                str(out_triptych / name),
                sar_mr_np,
                sar_out,
                opt_np,
                target_size=(out_h, out_w),
            )

        print(f"Saved {name}")


if __name__ == "__main__":
    main()
