import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from train_attention_sr_common import (
    TwoStreamFrequencyGuidedUNet,
    build_sar_lr_from_mr,
    build_sar_lr_from_physical_degradation,
    preprocess_sar_tensor,
)


def load_image_and_preprocess(
    sar_path: str,
    opt_path: str,
    input_downsample_size: int = 64,
    lr_degradation_mode: str = "physical",
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
    sar_norm_mode: str = "log_clip",
    sar_log_scale: float = 20.0,
    sar_clip_low: float = 1.0,
    sar_clip_high: float = 99.0,
):
    to_tensor = transforms.ToTensor()

    sar_mr = to_tensor(Image.open(sar_path).convert("L"))
    sar_mr = preprocess_sar_tensor(
        sar=sar_mr,
        norm_mode=sar_norm_mode,
        log_scale=sar_log_scale,
        clip_low=sar_clip_low,
        clip_high=sar_clip_high,
    )
    opt_hr = to_tensor(Image.open(opt_path).convert("RGB"))

    if lr_degradation_mode == "physical":
        sar_lr = build_sar_lr_from_physical_degradation(
            sar_mr=sar_mr,
            input_downsample_size=input_downsample_size,
            use_freq_truncation=degrade_freq_truncation,
            use_anisotropic_blur=degrade_anisotropic_blur,
            use_speckle=degrade_speckle,
            freq_window_size=degrade_freq_window_size,
            blur_kernel_size=degrade_blur_kernel_size,
            blur_sigma_azimuth=degrade_blur_sigma_azimuth,
            blur_sigma_range=degrade_blur_sigma_range,
            blur_angle_deg=degrade_blur_angle_deg,
            speckle_looks=degrade_speckle_looks,
            clamp_output=degrade_clamp_output,
        )
    elif lr_degradation_mode == "legacy":
        sar_lr = build_sar_lr_from_mr(sar_mr=sar_mr, input_downsample_size=input_downsample_size)
    elif lr_degradation_mode == "identity":
        sar_lr = sar_mr.clone()
    else:
        raise ValueError(
            f"Unsupported lr_degradation_mode: {lr_degradation_mode}. "
            "Expected one of {'legacy', 'physical', 'identity'}."
        )

    return sar_lr, opt_hr, sar_mr


def parse_args():
    parser = argparse.ArgumentParser(description="Inference for NGF+LNCC guided SAR super-resolution model")
    parser.add_argument("--weight", default="twostream_crossattn_ngf_lncc_0426_epoch_50.pth", help="Path to model checkpoint")
    parser.add_argument("--sar", default="../data/sar/9813.png", help="Path to SAR image")
    parser.add_argument("--opt", default="../data/opt/9813.png", help="Path to optical image")
    parser.add_argument("--out", default="./recon_attention_sr_v3_ngf.png", help="Output SAR reconstruction path")
    parser.add_argument("--figure", default="./compare_attention_sr_v3_ngf.png", help="Output comparison figure path")
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--input-downsample-size", type=int, default=64)
    parser.add_argument(
        "--lr-degradation-mode",
        choices=["legacy", "physical", "identity"],
        default="physical",
        help=(
            "legacy: bilinear down/up; "
            "physical: frequency truncation -> optional anisotropic blur -> multiplicative speckle; "
            "identity: no degradation, use SAR input directly"
        ),
    )
    parser.add_argument(
        "--degrade-freq-truncation",
        dest="degrade_freq_truncation",
        action="store_true",
        help="Enable frequency truncation step in physical degradation mode",
    )
    parser.add_argument(
        "--no-degrade-freq-truncation",
        dest="degrade_freq_truncation",
        action="store_false",
        help="Disable frequency truncation step in physical degradation mode",
    )
    parser.set_defaults(degrade_freq_truncation=True)
    parser.add_argument(
        "--degrade-anisotropic-blur",
        dest="degrade_anisotropic_blur",
        action="store_true",
        help="Enable anisotropic blur approximation step in physical degradation mode",
    )
    parser.add_argument(
        "--no-degrade-anisotropic-blur",
        dest="degrade_anisotropic_blur",
        action="store_false",
        help="Disable anisotropic blur approximation step in physical degradation mode",
    )
    parser.set_defaults(degrade_anisotropic_blur=False)
    parser.add_argument(
        "--degrade-speckle",
        dest="degrade_speckle",
        action="store_true",
        help="Enable multiplicative Gamma speckle injection step in physical degradation mode",
    )
    parser.add_argument(
        "--no-degrade-speckle",
        dest="degrade_speckle",
        action="store_false",
        help="Disable multiplicative Gamma speckle injection step in physical degradation mode",
    )
    parser.set_defaults(degrade_speckle=True)
    parser.add_argument(
        "--degrade-freq-window-size",
        type=int,
        default=0,
        help="Frequency truncation central window size in pixels; <=0 uses input-downsample-size",
    )
    parser.add_argument("--degrade-blur-kernel-size", type=int, default=9)
    parser.add_argument(
        "--degrade-blur-sigma-azimuth",
        type=float,
        default=2.0,
        help="Anisotropic blur sigma along azimuth direction",
    )
    parser.add_argument(
        "--degrade-blur-sigma-range",
        type=float,
        default=0.8,
        help="Anisotropic blur sigma along range direction",
    )
    parser.add_argument("--degrade-blur-angle-deg", type=float, default=0.0)
    parser.add_argument(
        "--degrade-speckle-looks",
        type=float,
        default=1.0,
        help="Equivalent number of looks L used by Gamma speckle noise",
    )
    parser.add_argument(
        "--degrade-clamp-output",
        dest="degrade_clamp_output",
        action="store_true",
        help="Clamp degraded SAR input back to [0, 1]",
    )
    parser.add_argument(
        "--no-degrade-clamp-output",
        dest="degrade_clamp_output",
        action="store_false",
        help="Do not clamp degraded SAR input",
    )
    parser.set_defaults(degrade_clamp_output=True)
    parser.add_argument("--sar-norm-mode", choices=["linear", "log_clip"], default="log_clip")
    parser.add_argument("--sar-log-scale", type=float, default=20.0)
    parser.add_argument("--sar-clip-low", type=float, default=1.0)
    parser.add_argument("--sar-clip-high", type=float, default=99.0)
    parser.add_argument("--show", action="store_true", help="Display the figure window")
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.figure) or ".", exist_ok=True)

    if not os.path.isfile(args.weight):
        raise FileNotFoundError(f"Checkpoint not found: {args.weight}")
    if not os.path.isfile(args.sar):
        raise FileNotFoundError(f"SAR image not found: {args.sar}")
    if not os.path.isfile(args.opt):
        raise FileNotFoundError(f"Optical image not found: {args.opt}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TwoStreamFrequencyGuidedUNet(base_channels=args.base_channels).to(device)
    state_dict = torch.load(args.weight, map_location=device)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict)
    model.eval()

    sar_lr, opt_hr, sar_mr = load_image_and_preprocess(
        sar_path=args.sar,
        opt_path=args.opt,
        input_downsample_size=args.input_downsample_size,
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
        sar_norm_mode=args.sar_norm_mode,
        sar_log_scale=args.sar_log_scale,
        sar_clip_low=args.sar_clip_low,
        sar_clip_high=args.sar_clip_high,
    )

    if args.lr_degradation_mode == "physical":
        freq_window = args.degrade_freq_window_size if args.degrade_freq_window_size > 0 else args.input_downsample_size
        print(
            "Inference LR physical degradation (serial): "
            f"freq_trunc={args.degrade_freq_truncation} [window={freq_window}], "
            f"aniso_blur={args.degrade_anisotropic_blur} "
            f"[kernel={args.degrade_blur_kernel_size}, sigma_azimuth={args.degrade_blur_sigma_azimuth}, "
            f"sigma_range={args.degrade_blur_sigma_range}, angle={args.degrade_blur_angle_deg}], "
            f"speckle={args.degrade_speckle} [looks={args.degrade_speckle_looks}], "
            f"clamp={args.degrade_clamp_output}"
        )
    elif args.lr_degradation_mode == "identity":
        print("Inference LR degradation mode: identity (no degradation, SAR input is used directly)")
    else:
        print("Inference LR degradation mode: legacy bilinear downsample + upsample")

    sar_input = sar_lr.unsqueeze(0).to(device)
    opt_input = opt_hr.unsqueeze(0).to(device)

    with torch.no_grad():
        sar_out = model(sar_input, opt_input)

    out_img = np.clip(sar_out.squeeze().cpu().numpy(), 0.0, 1.0)
    out_img_uint8 = (out_img * 255.0).round().astype(np.uint8)
    Image.fromarray(out_img_uint8, mode="L").save(args.out)
    print(f"Saved reconstructed SAR: {args.out}")

    sar_lr_img = sar_lr.squeeze().numpy()
    sar_mr_img = sar_mr.squeeze().numpy()
    opt_hr_img = opt_hr.permute(1, 2, 0).numpy()

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(sar_lr_img, cmap="gray")
    axes[0].set_title("Input: LR SAR")
    axes[0].axis("off")

    axes[1].imshow(opt_hr_img)
    axes[1].set_title("Input: HR Optical")
    axes[1].axis("off")

    axes[2].imshow(out_img, cmap="gray")
    axes[2].set_title("Output: SAR SR v3 (NGF+LNCC)")
    axes[2].axis("off")

    axes[3].imshow(sar_mr_img, cmap="gray")
    axes[3].set_title("Reference: MR SAR")
    axes[3].axis("off")

    plt.tight_layout()
    plt.savefig(args.figure, dpi=150)
    print(f"Saved comparison figure: {args.figure}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
