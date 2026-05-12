import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from train_attention_sr_common import TwoStreamFrequencyGuidedUNet, build_sar_lr_from_mr, preprocess_sar_tensor


def load_image_and_preprocess(
    sar_path: str,
    opt_path: str,
    input_downsample_size: int = 64,
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

    sar_lr = build_sar_lr_from_mr(sar_mr=sar_mr, input_downsample_size=input_downsample_size)

    return sar_lr, opt_hr, sar_mr


def parse_args():
    parser = argparse.ArgumentParser(description="Inference for improved two-stream SAR super-resolution model")
    parser.add_argument("--weight", default="./twostream_crossattn_freqdecouple_final.pth", help="Path to model checkpoint")
    parser.add_argument("--sar", default="../data/sar/8078.png", help="Path to SAR image")
    parser.add_argument("--opt", default="../data/opt/8078.png", help="Path to optical image")
    parser.add_argument("--out", default="./recon_attention_sr_v2.png", help="Output SAR reconstruction path")
    parser.add_argument("--figure", default="./compare_attention_sr_v2.png", help="Output comparison figure path")
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--input-downsample-size", type=int, default=64)
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
        sar_norm_mode=args.sar_norm_mode,
        sar_log_scale=args.sar_log_scale,
        sar_clip_low=args.sar_clip_low,
        sar_clip_high=args.sar_clip_high,
    )

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
    axes[2].set_title("Output: SAR SR v2")
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
