import argparse
import os

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from train_attention_sr_common import (
    SAROptDataset,
    TwoStreamFrequencyGuidedUNet,
    VGGFeatureLoss,
    gradient_magnitude,
    low_frequency_l1_loss,
    normalize_map,
    total_variation_loss,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train improved two-stream SAR super-resolution model")
    parser.add_argument("--sar-dir", default="../data/sar_mini", help="Directory with SAR images")
    parser.add_argument("--opt-dir", default="../data/opt_mini", help="Directory with optical images")
    parser.add_argument(
        "--phase",
        choices=["phase1", "phase2", "phase3"],
        default="phase2",
        help="phase1: low; phase2: low+structure+tv; phase3: phase2+small VGG",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--input-downsample-size", type=int, default=64)
    parser.add_argument("--lowfreq-down-factor", type=int, default=4)
    parser.add_argument("--lambda-struct", type=float, default=0.1, help="Structure loss weight (edge in v2)")
    parser.add_argument("--lambda-vgg", type=float, default=0.02, help="Small VGG weight used only in phase3")
    parser.add_argument("--lambda-tv", type=float, default=1e-4)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--save-prefix", default="twostream_crossattn_freqdecouple")
    parser.add_argument("--save-dir", default=".")
    parser.add_argument("--vgg-input-size", type=int, default=128, help="Resolution used for VGG perceptual loss")
    parser.add_argument("--no-vgg-imagenet", action="store_true", help="Use random VGG instead of ImageNet pretrained")
    parser.add_argument("--no-augment", action="store_true", help="Disable paired random flips")
    parser.add_argument("--sar-norm-mode", choices=["linear", "log_clip"], default="log_clip")
    parser.add_argument("--sar-log-scale", type=float, default=20.0)
    parser.add_argument("--sar-clip-low", type=float, default=1.0)
    parser.add_argument("--sar-clip-high", type=float, default=99.0)
    return parser.parse_args()


def phase_weights(args):
    if args.phase == "phase1":
        return 0.0, 0.0, 0.0
    if args.phase == "phase2":
        return args.lambda_struct, args.lambda_tv, 0.0
    return args.lambda_struct, args.lambda_tv, args.lambda_vgg


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    dataset = SAROptDataset(
        sar_dir=args.sar_dir,
        opt_dir=args.opt_dir,
        input_downsample_size=args.input_downsample_size,
        augment=not args.no_augment,
        sar_norm_mode=args.sar_norm_mode,
        sar_log_scale=args.sar_log_scale,
        sar_clip_low=args.sar_clip_low,
        sar_clip_high=args.sar_clip_high,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    print(f"Loaded dataset with {len(dataset)} paired samples.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TwoStreamFrequencyGuidedUNet(base_channels=args.base_channels).to(device)

    use_vgg = args.phase == "phase3" and args.lambda_vgg > 0.0
    vgg_loss_fn = None
    if use_vgg:
        vgg_loss_fn = VGGFeatureLoss(
            use_imagenet_weights=not args.no_vgg_imagenet,
            input_size=args.vgg_input_size,
        ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    print(f"Model running on device: {device}")
    print(
        f"Phase: {args.phase} | SAR norm: {args.sar_norm_mode} "
        f"(log_scale={args.sar_log_scale}, clip=[{args.sar_clip_low}, {args.sar_clip_high}])"
    )

    for epoch in range(args.epochs):
        model.train()

        epoch_total = 0.0
        epoch_low = 0.0
        epoch_struct = 0.0
        epoch_vgg = 0.0
        epoch_tv = 0.0

        w_struct, w_tv, w_vgg = phase_weights(args)

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for sar_lr, opt_hr, sar_mr in pbar:
            sar_lr = sar_lr.to(device)
            opt_hr = opt_hr.to(device)
            sar_mr = sar_mr.to(device)

            optimizer.zero_grad()

            sar_out = model(sar_lr, opt_hr)

            loss_low = low_frequency_l1_loss(
                sar_out=sar_out,
                sar_mr=sar_mr,
                down_factor=args.lowfreq_down_factor,
            )

            opt_gray = opt_hr.mean(dim=1, keepdim=True)
            pred_edge = normalize_map(gradient_magnitude(sar_out))
            opt_edge = normalize_map(gradient_magnitude(opt_gray))

            loss_struct = F.l1_loss(pred_edge, opt_edge)
            loss_tv = total_variation_loss(sar_out)
            loss_vgg = sar_out.new_tensor(0.0)
            if vgg_loss_fn is not None:
                loss_vgg = vgg_loss_fn(pred_edge, opt_edge)

            struct_term = w_struct * loss_struct
            vgg_term = w_vgg * loss_vgg
            tv_term = w_tv * loss_tv
            total_loss = loss_low + struct_term + tv_term + vgg_term

            total_loss.backward()
            optimizer.step()

            epoch_total += total_loss.item()
            epoch_low += loss_low.item()
            epoch_struct += loss_struct.item()
            epoch_vgg += loss_vgg.item()
            epoch_tv += loss_tv.item()

            pbar.set_postfix(
                {
                    "Total": f"{total_loss.item():.4f}",
                    "Low": f"{loss_low.item():.4f}",
                    "StructW": f"{struct_term.item():.4f}",
                    "TVW": f"{tv_term.item():.4f}",
                    "VGGW": f"{vgg_term.item():.4f}",
                }
            )

        n_batches = len(dataloader)
        avg_total = epoch_total / n_batches
        avg_low = epoch_low / n_batches
        avg_struct = epoch_struct / n_batches
        avg_vgg = epoch_vgg / n_batches
        avg_tv = epoch_tv / n_batches
        deep_gamma = model.deep_cross_attn.gamma.detach().item()
        shallow_gamma = model.shallow_fusion.gamma.detach().item()

        print(
            f"Epoch [{epoch + 1}/{args.epochs}] | "
            f"Total: {avg_total:.5f}, Low: {avg_low:.5f}, Struct: {avg_struct:.5f}, "
            f"TV: {avg_tv:.5f}, VGG: {avg_vgg:.5f}, "
            f"w_struct: {w_struct:.3f}, w_tv: {w_tv:.6f}, w_vgg: {w_vgg:.3f}, "
            f"gamma_deep: {deep_gamma:.5f}, gamma_shallow: {shallow_gamma:.5f}"
        )

        if (epoch + 1) % args.save_every == 0:
            ckpt_path = os.path.join(args.save_dir, f"{args.save_prefix}_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    final_path = os.path.join(args.save_dir, f"{args.save_prefix}_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"Training complete. Final checkpoint saved to: {final_path}")


if __name__ == "__main__":
    main()
