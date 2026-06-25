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
    low_frequency_l1_loss,
    total_variation_loss,
)


def sobel_gradients(x: torch.Tensor):
    kernel_x = x.new_tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
    kernel_y = x.new_tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3)
    grad_x = F.conv2d(x, kernel_x, padding=1)
    grad_y = F.conv2d(x, kernel_y, padding=1)
    return grad_x, grad_y


def normalize_map(x: torch.Tensor, eps: float = 1e-6):
    x_min = x.amin(dim=(2, 3), keepdim=True)
    x_max = x.amax(dim=(2, 3), keepdim=True)
    return (x - x_min) / (x_max - x_min + eps)


def edge_support_mask(x: torch.Tensor, threshold: float, sharpness: float, eps: float = 1e-6):
    """Return a soft mask that keeps structure guidance only where edge support is strong."""
    _, _, magnitude = normalized_gradient_field(x, eps=eps)
    magnitude_norm = normalize_map(magnitude, eps=eps)
    return torch.sigmoid(sharpness * (magnitude_norm - threshold))


def normalized_gradient_field(x: torch.Tensor, eps: float = 1e-6):
    grad_x, grad_y = sobel_gradients(x)
    magnitude = torch.sqrt(grad_x * grad_x + grad_y * grad_y + eps)
    ngf_x = grad_x / magnitude
    ngf_y = grad_y / magnitude
    return ngf_x, ngf_y, magnitude


def ngf_alignment_loss(
    sar_out: torch.Tensor,
    opt_gray: torch.Tensor,
    mask_threshold: float = 0.1,
    mask_sharpness: float = 20.0,
    support_mask: torch.Tensor | None = None,
    eps: float = 1e-6,
):
    sar_nx, sar_ny, _ = normalized_gradient_field(sar_out, eps=eps)
    opt_nx, opt_ny, opt_mag = normalized_gradient_field(opt_gray, eps=eps)

    dot = torch.clamp(sar_nx * opt_nx + sar_ny * opt_ny, min=-1.0, max=1.0)

    opt_mag_norm = normalize_map(opt_mag)
    edge_mask = torch.sigmoid(mask_sharpness * (opt_mag_norm - mask_threshold))
    if support_mask is not None:
        edge_mask = edge_mask * support_mask

    loss_map = edge_mask * (1.0 - torch.abs(dot))
    return loss_map.sum() / (edge_mask.sum() + eps)


def artifact_suppression_loss(
    sar_out: torch.Tensor,
    support_mask: torch.Tensor,
    eps: float = 1e-6,
):
    """Penalize high-frequency hallucinations outside SAR-supported edge regions."""
    _, _, sar_mag = normalized_gradient_field(sar_out, eps=eps)
    sar_mag_norm = normalize_map(sar_mag, eps=eps)

    smooth_region = 1.0 - support_mask
    loss_map = smooth_region * sar_mag_norm
    return loss_map.sum() / (smooth_region.sum() + eps)


def local_lncc_loss(a: torch.Tensor, b: torch.Tensor, window_size: int = 9, eps: float = 1e-6):
    pad = window_size // 2
    mean_a = F.avg_pool2d(a, kernel_size=window_size, stride=1, padding=pad)
    mean_b = F.avg_pool2d(b, kernel_size=window_size, stride=1, padding=pad)

    mean_aa = F.avg_pool2d(a * a, kernel_size=window_size, stride=1, padding=pad)
    mean_bb = F.avg_pool2d(b * b, kernel_size=window_size, stride=1, padding=pad)
    mean_ab = F.avg_pool2d(a * b, kernel_size=window_size, stride=1, padding=pad)

    var_a = torch.clamp(mean_aa - mean_a * mean_a, min=0.0)
    var_b = torch.clamp(mean_bb - mean_b * mean_b, min=0.0)
    cov_ab = mean_ab - mean_a * mean_b

    lncc = cov_ab / torch.sqrt(var_a * var_b + eps)
    return 1.0 - lncc.mean()


def parse_args():
    parser = argparse.ArgumentParser(description="Train SR model with NGF + LNCC cross-modal guidance")
    parser.add_argument("--sar-dir", default="../data/sar", help="Directory with SAR images")
    parser.add_argument("--opt-dir", default="../data/opt", help="Directory with optical images")
    parser.add_argument(
        "--phase",
        choices=["phase1", "phase2", "phase3"],
        default="phase2",
        help="phase1: low; phase2: low+structure+tv; phase3: phase2+small VGG",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--lr-scheduler",
        choices=["none", "multistep", "cosine", "plateau"],
        default="multistep",
        help="Learning rate decay strategy",
    )
    parser.add_argument(
        "--lr-milestones",
        type=str,
        default="25,40",
        help="Comma-separated milestone epochs used by MultiStepLR",
    )
    parser.add_argument("--lr-gamma", type=float, default=0.5, help="LR decay factor")
    parser.add_argument("--lr-min", type=float, default=1e-6, help="Minimum LR for cosine/plateau schedulers")
    parser.add_argument("--lr-patience", type=int, default=5, help="Patience epochs used by ReduceLROnPlateau")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--input-downsample-size", type=int, default=0)
    parser.add_argument("--downsample-factor", type=int, default=2)
    parser.add_argument(
        "--lr-degradation-mode",
        choices=["legacy", "physical"],
        default="physical",
        help="legacy: bilinear down/up; physical: frequency truncation -> optional anisotropic blur -> multiplicative speckle",
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
    parser.add_argument("--lowfreq-down-factor", type=int, default=4)
    parser.add_argument("--lambda-struct", type=float, default=0.15, help="Structure loss weight in phase2/3")
    parser.add_argument("--struct-lncc-ratio", type=float, default=0.25, help="LNCC ratio inside structure composite")
    parser.add_argument("--lambda-vgg", type=float, default=0.02, help="Small VGG weight used only in phase3")
    parser.add_argument("--lambda-tv", type=float, default=1e-4)
    parser.add_argument("--lambda-artifact-suppress", type=float, default=0.05, help="Penalty for hallucinated edges outside SAR-supported regions")
    parser.add_argument("--ngf-mask-threshold", type=float, default=0.1)
    parser.add_argument("--ngf-mask-sharpness", type=float, default=20.00)
    parser.add_argument("--sar-edge-support-threshold", type=float, default=0.08, help="Edge-support threshold derived from SAR magnitude")
    parser.add_argument("--sar-edge-support-sharpness", type=float, default=16.0, help="Sharpness for the SAR support mask")
    parser.add_argument("--lncc-window", type=int, default=9)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--save-prefix", default="twostream_crossattn_ngf_lncc")
    parser.add_argument("--save-dir", default="./checkpoints", help="Directory to save model checkpoints")
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


def parse_milestones(milestone_text: str, max_epoch: int):
    milestones = []
    for token in milestone_text.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            milestone = int(token)
        except ValueError:
            continue
        if 1 <= milestone <= max_epoch:
            milestones.append(milestone)
    return sorted(set(milestones))


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    dataset = SAROptDataset(
        sar_dir=args.sar_dir,
        opt_dir=args.opt_dir,
        input_downsample_size=args.input_downsample_size,
        downsample_factor=args.downsample_factor,
        augment=not args.no_augment,
        sar_norm_mode=args.sar_norm_mode,
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
    scheduler = None
    scheduler_desc = "none"
    if args.lr_scheduler == "multistep":
        milestones = parse_milestones(args.lr_milestones, args.epochs)
        if milestones:
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=milestones,
                gamma=args.lr_gamma,
            )
            scheduler_desc = f"MultiStepLR(milestones={milestones}, gamma={args.lr_gamma})"
        else:
            print("[WARN] MultiStepLR selected but no valid milestones were provided; scheduler disabled.")
    elif args.lr_scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, args.epochs),
            eta_min=args.lr_min,
        )
        scheduler_desc = f"CosineAnnealingLR(T_max={max(1, args.epochs)}, eta_min={args.lr_min})"
    elif args.lr_scheduler == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.lr_gamma,
            patience=max(1, args.lr_patience),
            min_lr=args.lr_min,
        )
        scheduler_desc = (
            f"ReduceLROnPlateau(factor={args.lr_gamma}, patience={max(1, args.lr_patience)}, "
            f"min_lr={args.lr_min})"
        )

    print(f"Model running on device: {device}")
    print(
        f"Phase: {args.phase} | SAR norm: {args.sar_norm_mode} "
        f"(log_scale={args.sar_log_scale}, clip=[{args.sar_clip_low}, {args.sar_clip_high}])"
    )
    if args.lr_degradation_mode == "physical":
        freq_window = args.degrade_freq_window_size if args.degrade_freq_window_size > 0 else args.input_downsample_size
        print(
            "LR physical degradation (serial): "
            f"freq_trunc={args.degrade_freq_truncation} [window={freq_window}], "
            f"aniso_blur={args.degrade_anisotropic_blur} "
            f"[kernel={args.degrade_blur_kernel_size}, sigma_azimuth={args.degrade_blur_sigma_azimuth}, "
            f"sigma_range={args.degrade_blur_sigma_range}, angle={args.degrade_blur_angle_deg}], "
            f"speckle={args.degrade_speckle} [looks={args.degrade_speckle_looks}], "
            f"clamp={args.degrade_clamp_output}"
        )
    else:
        print("LR degradation mode: legacy bilinear downsample + upsample")
    print(f"LR setup: init={args.learning_rate}, scheduler={scheduler_desc}")

    for epoch in range(args.epochs):
        model.train()

        epoch_total = 0.0
        epoch_low = 0.0
        epoch_struct = 0.0
        epoch_ngf = 0.0
        epoch_lncc = 0.0
        epoch_tv = 0.0
        epoch_vgg = 0.0
        epoch_artifact = 0.0

        w_struct, w_tv, w_vgg = phase_weights(args)
        lncc_ratio = max(0.0, min(1.0, args.struct_lncc_ratio))

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
            sar_support_mask = edge_support_mask(
                sar_mr,
                threshold=args.sar_edge_support_threshold,
                sharpness=args.sar_edge_support_sharpness,
            )
            _, _, sar_mag = normalized_gradient_field(sar_out)
            _, _, opt_mag = normalized_gradient_field(opt_gray)

            loss_ngf = ngf_alignment_loss(
                sar_out=sar_out,
                opt_gray=opt_gray,
                mask_threshold=args.ngf_mask_threshold,
                mask_sharpness=args.ngf_mask_sharpness,
                support_mask=sar_support_mask,
            )
            loss_lncc = local_lncc_loss(
                a=normalize_map(sar_mag),
                b=normalize_map(opt_mag),
                window_size=args.lncc_window,
            )
            loss_tv = total_variation_loss(sar_out)
            loss_artifact = artifact_suppression_loss(sar_out, support_mask=sar_support_mask)
            loss_struct = (1.0 - lncc_ratio) * loss_ngf + lncc_ratio * loss_lncc

            pred_edge = normalize_map(sar_mag)
            opt_edge = normalize_map(opt_mag)
            loss_vgg = sar_out.new_tensor(0.0)
            if vgg_loss_fn is not None:
                loss_vgg = vgg_loss_fn(pred_edge, opt_edge)

            struct_term = w_struct * loss_struct
            tv_term = w_tv * loss_tv
            vgg_term = w_vgg * loss_vgg
            artifact_term = args.lambda_artifact_suppress * loss_artifact
            total_loss = loss_low + struct_term + tv_term + vgg_term + artifact_term

            total_loss.backward()
            optimizer.step()

            epoch_total += total_loss.item()
            epoch_low += loss_low.item()
            epoch_struct += loss_struct.item()
            epoch_ngf += loss_ngf.item()
            epoch_lncc += loss_lncc.item()
            epoch_tv += loss_tv.item()
            epoch_vgg += loss_vgg.item()
            epoch_artifact += loss_artifact.item()

            pbar.set_postfix(
                {
                    "Total": f"{total_loss.item():.4f}",
                    "Low": f"{loss_low.item():.4f}",
                    "StructW": f"{struct_term.item():.4f}",
                    "ArtW": f"{artifact_term.item():.4f}",
                    "TVW": f"{tv_term.item():.4f}",
                    "VGGW": f"{vgg_term.item():.4f}",
                }
            )

        n_batches = len(dataloader)
        avg_total = epoch_total / n_batches
        avg_low = epoch_low / n_batches
        avg_struct = epoch_struct / n_batches
        avg_ngf = epoch_ngf / n_batches
        avg_lncc = epoch_lncc / n_batches
        avg_tv = epoch_tv / n_batches
        avg_vgg = epoch_vgg / n_batches
        avg_artifact = epoch_artifact / n_batches

        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_total)
            else:
                scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        deep_gamma = model.deep_cross_attn.gamma.detach().item()
        shallow_gamma = model.shallow_fusion.gamma.detach().item()

        print(
            f"Epoch [{epoch + 1}/{args.epochs}] | "
            f"Total: {avg_total:.5f}, Low: {avg_low:.5f}, Struct: {avg_struct:.5f}, "
            f"NGF: {avg_ngf:.5f}, LNCC: {avg_lncc:.5f}, TV: {avg_tv:.5f}, VGG: {avg_vgg:.5f}, "
            f"Artifact: {avg_artifact:.5f}, "
            f"w_struct: {w_struct:.3f}, w_tv: {w_tv:.6f}, w_vgg: {w_vgg:.3f}, lncc_ratio: {lncc_ratio:.2f}, "
            f"w_artifact: {args.lambda_artifact_suppress:.3f}, "
            f"lr: {current_lr:.7f}, gamma_deep: {deep_gamma:.5f}, gamma_shallow: {shallow_gamma:.5f}"
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
