import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
import torch
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms

from sar_degrade import (
    build_sar_lr_from_physical_degradation,
    normalize_sar_log_clip,
    to_1ch,
    to_3ch,
)

torch.serialization.add_safe_globals([np.core.multiarray.scalar])


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

from basicsr.archs.swinir_arch import SwinIR  # noqa: E402
from basicsr.metrics.psnr_ssim import calculate_ssim_pt  # noqa: E402
from deepdespeckling.sar2sar.sar2sar_denoiser import Sar2SarDenoiser  # noqa: E402
from deepdespeckling.utils.utils import denormalize_sar_image  # noqa: E402
from train_attention_sr_common import preprocess_sar_tensor  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cascade SAR2SAR denoising + SwinIR super-resolution")
    parser.add_argument("--sar-dir", default="../data/sar", help="Directory with SAR images")
    parser.add_argument("--opt-dir", default="../data/opt", help="Directory with optical images (for pairing)")
    parser.add_argument("--output-dir", default="./outputs/cascade", help="Output directory")
    parser.add_argument("--swinir-weights", required=True, help="Path to SwinIR weights (.pth)")
    parser.add_argument("--mode", choices=["infer", "finetune"], default="infer")
    parser.add_argument("--scale", type=int, default=4, help="SR upscale factor")
    parser.add_argument("--patch-size", type=int, default=64, help="SwinIR img_size")
    parser.add_argument("--window-size", type=int, default=8, help="SwinIR window size")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Device preference")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of images (0 = all)")
    parser.add_argument("--save-intermediate", action="store_true", help="Save LR and denoised intermediates")

    parser.add_argument("--input-downsample-size", type=int, default=64, help="Frequency truncation window size")
    parser.add_argument("--degrade-freq-truncation", action="store_true", default=True)
    parser.add_argument("--degrade-anisotropic-blur", action="store_true", default=False)
    parser.add_argument("--degrade-speckle", action="store_true", default=True)
    parser.add_argument("--degrade-freq-window-size", type=int, default=0)
    parser.add_argument("--degrade-blur-kernel-size", type=int, default=9)
    parser.add_argument("--degrade-blur-sigma-azimuth", type=float, default=2.0)
    parser.add_argument("--degrade-blur-sigma-range", type=float, default=0.8)
    parser.add_argument("--degrade-blur-angle-deg", type=float, default=0.0)
    parser.add_argument("--degrade-speckle-looks", type=float, default=1.0)
    parser.add_argument("--degrade-clamp-output", action="store_true", default=True)

    parser.add_argument("--sar2sar-input-gain", type=float, default=1.0, help="Scale noisy SAR before SAR2SAR")
    parser.add_argument("--sar2sar-weights", default="", help="Path to SAR2SAR weights (.pth)")
    parser.add_argument("--sar2sar-patch-size", type=int, default=256)
    parser.add_argument("--sar2sar-stride-size", type=int, default=254)
    parser.add_argument("--sr-log-scale", type=float, default=20.0)
    parser.add_argument("--sr-clip-low", type=float, default=1.0)
    parser.add_argument("--sr-clip-high", type=float, default=99.0)
    parser.add_argument("--output-16bit", action="store_true", default=True, help="Save SR output as 16-bit PNG")

    parser.add_argument("--finetune-epochs", type=int, default=10)
    parser.add_argument("--finetune-learning-rate", type=float, default=1e-6)
    parser.add_argument("--finetune-batch-size", type=int, default=1)
    parser.add_argument("--finetune-num-workers", type=int, default=2)
    parser.add_argument("--finetune-lambda-ssim", type=float, default=0.0)
    parser.add_argument("--finetune-save-dir", default="./checkpoints")
    parser.add_argument("--finetune-save-every", type=int, default=5)
    parser.add_argument("--finetune-resume", default="")
    parser.add_argument("--finetune-amp", action="store_true", default=False)
    parser.add_argument("--finetune-grad-accum-steps", type=int, default=4)
    parser.add_argument("--finetune-freeze-transformer", action="store_true", default=True)
    parser.add_argument("--finetune-debug-dir", default="./outputs/finetune_debug")
    parser.add_argument("--finetune-debug-max-batches", type=int, default=2)
    parser.add_argument("--finetune-debug", action="store_true", default=True)
    parser.add_argument("--finetune-stop-on-nan", action="store_true", default=False)
    return parser.parse_args()


def resolve_device(pref: str) -> torch.device:
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def list_pairs(sar_dir: str, opt_dir: str) -> List[str]:
    sar_names = {p.name for p in Path(sar_dir).iterdir() if p.is_file()}
    opt_names = {p.name for p in Path(opt_dir).iterdir() if p.is_file()}
    names = sorted(sar_names.intersection(opt_names))
    if not names:
        raise RuntimeError("No matched SAR/OPT image pairs found.")
    return names


def load_gray_image(path: str) -> torch.Tensor:
    to_tensor = transforms.ToTensor()
    return to_tensor(Image.open(path).convert("L"))


class SarOptDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        sar_dir: str,
        opt_dir: str,
        input_downsample_size: int,
        sar_log_scale: float,
        sar_clip_low: float,
        sar_clip_high: float,
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
        self.sar_log_scale = sar_log_scale
        self.sar_clip_low = sar_clip_low
        self.sar_clip_high = sar_clip_high
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

        sar_lr = build_sar_lr_from_physical_degradation(
            sar_mr=sar_mr,
            input_downsample_size=self.input_downsample_size,
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
        return sar_lr, opt_hr, sar_mr


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


def tensor_stats(tag: str, x: torch.Tensor) -> None:
    x_cpu = x.detach().float().cpu()
    finite = torch.isfinite(x_cpu)
    if not finite.any():
        print(f"{tag}: all values are non-finite")
        return
    vals = x_cpu[finite]
    print(
        f"{tag}: min={vals.min().item():.6f} max={vals.max().item():.6f} mean={vals.mean().item():.6f}"
    )


def is_finite_tensor(x: torch.Tensor) -> bool:
    return torch.isfinite(x).all().item()


def build_swinir(scale: int, patch_size: int, window_size: int, weights: str, device: torch.device) -> SwinIR:
    model = SwinIR(
        upscale=scale,
        in_chans=3,
        img_size=patch_size,
        window_size=window_size,
        img_range=1.0,
        depths=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler="pixelshuffle",
        resi_connection="1conv",
    )
    loadnet = torch.load(weights, map_location="cpu")
    if isinstance(loadnet, dict) and "params_ema" in loadnet:
        state_dict = loadnet["params_ema"]
    elif isinstance(loadnet, dict) and "params" in loadnet:
        state_dict = loadnet["params"]
    elif isinstance(loadnet, dict):
        state_dict = loadnet
    else:
        raise ValueError("Unsupported SwinIR checkpoint format.")
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model.to(device)


def normalize_sar2sar_weights(weights_path: str, output_dir: str) -> str:
    if not weights_path:
        return weights_path
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"SAR2SAR weights not found: {weights_path}")

    ckpt = torch.load(weights_path, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return weights_path

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        raise ValueError("Unsupported SAR2SAR checkpoint format.")

    os.makedirs(output_dir, exist_ok=True)
    base = os.path.basename(weights_path)
    name, ext = os.path.splitext(base)
    normalized_path = os.path.join(output_dir, f"{name}_sar2sar_wrapped{ext}")
    torch.save({"model_state_dict": state_dict}, normalized_path)
    return normalized_path


def run_swinir(model: SwinIR, img: torch.Tensor, scale: int, window_size: int) -> torch.Tensor:
    if img.ndim != 4:
        raise ValueError("Expected input tensor [B, C, H, W]")
    mod_pad_h = (window_size - img.size(2) % window_size) % window_size
    mod_pad_w = (window_size - img.size(3) % window_size) % window_size
    if mod_pad_h > 0 or mod_pad_w > 0:
        img = torch.nn.functional.pad(img, (0, mod_pad_w, 0, mod_pad_h), "reflect")

    with torch.no_grad():
        output = model(img)

    _, _, h, w = output.size()
    if mod_pad_h > 0 or mod_pad_w > 0:
        output = output[:, :, 0 : h - mod_pad_h * scale, 0 : w - mod_pad_w * scale]
    return output


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
            sar_dn_norm = torch.clamp(sar_dn_norm, 0.0, 1.0)
            denoised.append(sar_dn_norm)
    return torch.stack(denoised, dim=0)


def build_finetune_loader(args: argparse.Namespace) -> torch.utils.data.DataLoader:
    dataset = SarOptDataset(
        sar_dir=args.sar_dir,
        opt_dir=args.opt_dir,
        input_downsample_size=args.input_downsample_size,
        sar_log_scale=args.sr_log_scale,
        sar_clip_low=args.sr_clip_low,
        sar_clip_high=args.sr_clip_high,
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
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=args.finetune_batch_size,
        shuffle=True,
        num_workers=args.finetune_num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def finetune_swinir(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    sar2sar = Sar2SarDenoiser()
    if args.sar2sar_weights:
        sar2sar.weights_path = normalize_sar2sar_weights(args.sar2sar_weights, args.finetune_save_dir)
    model = build_swinir(
        scale=args.scale,
        patch_size=args.patch_size,
        window_size=args.window_size,
        weights=args.swinir_weights,
        device=device,
    )
    model.train()

    if args.finetune_resume and os.path.isfile(args.finetune_resume):
        model.load_state_dict(torch.load(args.finetune_resume, map_location="cpu"))

    if args.finetune_freeze_transformer:
        for name, param in model.named_parameters():
            if "layers" in name:
                param.requires_grad = False

    loader = build_finetune_loader(args)
    opt = torch.optim.Adam(model.parameters(), lr=args.finetune_learning_rate)
    use_amp = args.finetune_amp and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    os.makedirs(args.finetune_save_dir, exist_ok=True)
    if args.finetune_debug:
        os.makedirs(args.finetune_debug_dir, exist_ok=True)

    for epoch in range(1, args.finetune_epochs + 1):
        total_loss = 0.0
        opt.zero_grad(set_to_none=True)
        accum_steps = max(1, int(args.finetune_grad_accum_steps))
        for step, (sar_lr, _opt_hr, sar_hr) in enumerate(loader, start=1):
            if args.finetune_debug and step <= args.finetune_debug_max_batches:
                tensor_stats("sar_lr_raw", sar_lr)
                tensor_stats("sar_hr", sar_hr)

            sar_lr = sar2sar_denoise_batch(sar_lr, sar2sar, args).to(device)
            sar_hr = sar_hr.to(device)

            if args.finetune_debug and step <= args.finetune_debug_max_batches:
                tensor_stats("sar_dn_norm", sar_lr)
                debug_out = sar_lr[0].squeeze(0).detach().cpu().numpy()
                save_gray_image(
                    os.path.join(args.finetune_debug_dir, f"denoised_step_{step}.png"),
                    debug_out,
                    use_16bit=False,
                )

            sr_input = to_3ch(sar_lr).to(device)
            with autocast(enabled=use_amp):
                pred = model(sr_input)

            if pred.shape[1] == 3:
                pred = pred.mean(dim=1, keepdim=True)
            pred = torch.clamp(pred, 0.0, 1.0)

            pred_fp32 = pred.float()
            sar_hr_fp32 = sar_hr.float()
            loss_l1 = torch.nn.functional.l1_loss(pred_fp32, sar_hr_fp32)

            loss = loss_l1
            if args.finetune_lambda_ssim > 0:
                ssim = calculate_ssim_pt(pred_fp32, sar_hr_fp32, crop_border=0, test_y_channel=False)
                loss_ssim = 1.0 - ssim.mean()
                loss = loss + args.finetune_lambda_ssim * loss_ssim

            if not torch.isfinite(loss):
                print("[WARN] Non-finite loss detected. Skipping step.")
                tensor_stats("pred", pred)
                tensor_stats("sar_hr", sar_hr)
                print(f"loss_l1={loss_l1.item()} use_amp={use_amp}")
                if not is_finite_tensor(pred):
                    print("pred contains non-finite values")
                if not is_finite_tensor(sar_hr):
                    print("sar_hr contains non-finite values")
                if args.finetune_stop_on_nan:
                    raise RuntimeError("Non-finite loss encountered.")
                continue

            loss = loss / accum_steps

            scaler.scale(loss).backward()

            if step % accum_steps == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

            total_loss += float(loss.item()) * accum_steps

        if step % accum_steps != 0:
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

        avg_loss = total_loss / max(1, len(loader))
        print(f"Epoch {epoch}: loss={avg_loss:.6f}")

        if epoch % args.finetune_save_every == 0:
            save_path = os.path.join(args.finetune_save_dir, f"swinir_sar_epoch_{epoch}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Saved {save_path}")


def run_infer(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)

    names = list_pairs(args.sar_dir, args.opt_dir)
    if args.limit > 0:
        names = names[: args.limit]

    sar2sar = Sar2SarDenoiser()
    if args.sar2sar_weights:
        sar2sar.weights_path = normalize_sar2sar_weights(args.sar2sar_weights, args.output_dir)
    swinir = build_swinir(
        scale=args.scale,
        patch_size=args.patch_size,
        window_size=args.window_size,
        weights=args.swinir_weights,
        device=device,
    )

    out_root = Path(args.output_dir)
    out_orig = out_root / "original"
    out_dn = out_root / "denoised"
    out_sr = out_root / "sr"
    out_triptych = out_root / "triptych"

    for name in names:
        sar_path = Path(args.sar_dir) / name
        sar_hr = load_gray_image(str(sar_path))

        sar_dn_norm = sar2sar_denoise_batch(sar_hr.unsqueeze(0), sar2sar, args).squeeze(0)
        sr_input = to_3ch(sar_dn_norm).unsqueeze(0).to(device)
        sr_out = run_swinir(swinir, sr_input, args.scale, args.window_size)

        sr_gray = to_1ch(sr_out.squeeze(0)).cpu().numpy().squeeze(0)

        save_gray_image(str(out_orig / name), sar_hr.squeeze(0).numpy(), use_16bit=False)
        save_gray_image(str(out_dn / name), sar_dn_norm.squeeze(0).numpy(), use_16bit=False)
        save_gray_image(str(out_sr / name), sr_gray, use_16bit=args.output_16bit)
        save_triptych(
            str(out_triptych / name),
            sar_hr.squeeze(0).numpy(),
            sar_dn_norm.squeeze(0).numpy(),
            sr_gray,
        )

        print(f"Processed {name}")


def main() -> None:
    args = parse_args()
    if args.mode == "finetune":
        finetune_swinir(args)
    else:
        run_infer(args)


if __name__ == "__main__":
    main()
