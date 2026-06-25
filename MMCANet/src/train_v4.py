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
	low_frequency_l1_loss,
	charbonnier_loss,
	high_frequency_extraction,
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


def normalized_gradient_field(x: torch.Tensor, eps: float = 1e-6):
	grad_x, grad_y = sobel_gradients(x)
	magnitude = torch.sqrt(grad_x * grad_x + grad_y * grad_y + eps)
	return grad_x, grad_y, magnitude


def optical_guided_tv_loss(
	sar_out: torch.Tensor,
	opt_gray: torch.Tensor,
	alpha: float = 5.0,
	eps: float = 1e-6,
):
	_, _, opt_mag = normalized_gradient_field(opt_gray, eps=eps)
	opt_mag_norm = normalize_map(opt_mag, eps=eps)

	sar_dx = torch.abs(sar_out[:, :, :, :-1] - sar_out[:, :, :, 1:])
	sar_dy = torch.abs(sar_out[:, :, :-1, :] - sar_out[:, :, 1:, :])

	weight_x = torch.exp(-alpha * opt_mag_norm[:, :, :, :-1])
	weight_y = torch.exp(-alpha * opt_mag_norm[:, :, :-1, :])

	loss_tv_x = torch.mean(weight_x * sar_dx)
	loss_tv_y = torch.mean(weight_y * sar_dy)
	return loss_tv_x + loss_tv_y


def local_mean_ratio_loss(
	sar_out: torch.Tensor,
	sar_gt: torch.Tensor,
	window_size: int = 5,
	eps: float = 1e-6,
):
	"""Penalize local mean ratio deviation to discourage checkerboard hacks."""
	ratio = sar_gt / (sar_out + eps)
	pad = window_size // 2
	local_mean = F.avg_pool2d(ratio, kernel_size=window_size, stride=1, padding=pad)
	return torch.mean(torch.abs(local_mean - 1.0))


def parse_args():
	parser = argparse.ArgumentParser(description="Train MMCANet v4 with optical-guided TV loss")
	parser.add_argument("--sar-dir", default="data/sar", help="Directory with SAR images")
	parser.add_argument("--opt-dir", default="data/opt", help="Directory with optical images")
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
	parser.add_argument("--upscale", type=int, default=2)

	parser.add_argument("--input-downsample-size", type=int, default=128)
	parser.add_argument("--downsample-factor", type=int, default=2)
	parser.add_argument(
		"--lr-degradation-mode",
		choices=["legacy", "physical"],
		default="physical",
		help="legacy: bilinear down; physical: frequency truncation -> blur -> downsample -> speckle",
	)
	parser.add_argument("--degrade-freq-truncation", action="store_true", default=True)
	parser.add_argument("--degrade-anisotropic-blur", action="store_true", default=False)
	parser.add_argument("--degrade-speckle", action="store_true", default=True)
	parser.add_argument("--no-degrade-speckle", dest="degrade_speckle", action="store_false")
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
	parser.add_argument("--no-augment", action="store_true", help="Disable paired random flips")

	parser.add_argument("--lambda-rec", type=float, default=0.5)
	parser.add_argument("--lambda-lowfreq", type=float, default=1.0)
	parser.add_argument("--lowfreq-down-factor", type=int, default=4)
	parser.add_argument("--lambda-guided-tv", type=float, default=0.001)
	parser.add_argument("--guided-tv-alpha", type=float, default=5.0)
	parser.add_argument("--lambda-ratio", type=float, default=0.2)
	parser.add_argument("--ratio-window", type=int, default=5)

	parser.add_argument("--save-every", type=int, default=10)
	parser.add_argument("--save-prefix", default="mmcanet_0519_v2", help="Prefix for saved checkpoint filenames")
	parser.add_argument("--save-dir", default="./checkpoints/0519", help="Directory to save model checkpoints")
	return parser.parse_args()


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
	model = TwoStreamFrequencyGuidedUNet(base_channels=args.base_channels, upscale=args.upscale).to(device)

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
		f"LR setup: init={args.learning_rate}, scheduler={scheduler_desc}, "
		f"lambda_rec={args.lambda_rec}, lambda_lowfreq={args.lambda_lowfreq}, "
		f"lambda_guided_tv={args.lambda_guided_tv}"
	)

	for epoch in range(args.epochs):
		model.train()
		epoch_total = 0.0
		epoch_rec = 0.0
		epoch_low = 0.0
		epoch_tv = 0.0
		epoch_ratio = 0.0

		pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}")
		for sar_lr, opt_hr, sar_mr in pbar:
			sar_lr = sar_lr.to(device)
			opt_hr = opt_hr.to(device)
			sar_mr = sar_mr.to(device)

			optimizer.zero_grad()

			opt_lr = opt_hr
			if opt_lr.shape[-2:] != sar_lr.shape[-2:]:
				opt_lr = F.interpolate(
					opt_lr,
					size=sar_lr.shape[-2:],
					mode="bilinear",
					align_corners=False,
				)

			sar_out = model(sar_lr, opt_lr)
			if sar_out.shape[-2:] != sar_mr.shape[-2:]:
				raise ValueError(
					"Model output size does not match SAR ground truth. "
					"Check --upscale and degradation settings."
				)

			loss_rec = charbonnier_loss(sar_out, sar_mr)
			loss_low = low_frequency_l1_loss(
				sar_out=sar_out,
				sar_mr=sar_mr,
				down_factor=args.lowfreq_down_factor,
			)
			opt_gray = opt_hr.mean(dim=1, keepdim=True)
			# Extract high-frequency components to avoid color/shadow pollution
			opt_hf = high_frequency_extraction(opt_gray)
			loss_tv = optical_guided_tv_loss(
				sar_out=sar_out,
				opt_gray=opt_hf,
				alpha=args.guided_tv_alpha,
			)
			loss_ratio = local_mean_ratio_loss(
				sar_out=sar_out,
				sar_gt=sar_mr,
				window_size=args.ratio_window,
			)

			total_loss = (
				args.lambda_rec * loss_rec
				+ args.lambda_lowfreq * loss_low
				+ args.lambda_guided_tv * loss_tv
				+ args.lambda_ratio * loss_ratio
			)

			total_loss.backward()
			optimizer.step()

			epoch_total += total_loss.item()
			epoch_rec += loss_rec.item()
			epoch_low += loss_low.item()
			epoch_tv += loss_tv.item()
			epoch_ratio += loss_ratio.item()

			pbar.set_postfix(
				{
					"Total": f"{total_loss.item():.4f}",
					"Rec": f"{loss_rec.item():.4f}",
					"Low": f"{loss_low.item():.4f}",
					"gTV": f"{loss_tv.item():.6f}",
					"Ratio": f"{loss_ratio.item():.6f}",
				}
			)

		n_batches = len(dataloader)
		avg_total = epoch_total / n_batches
		avg_rec = epoch_rec / n_batches
		avg_low = epoch_low / n_batches
		avg_tv = epoch_tv / n_batches
		avg_ratio = epoch_ratio / n_batches

		if scheduler is not None:
			if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
				scheduler.step(avg_total)
			else:
				scheduler.step()

		current_lr = optimizer.param_groups[0]["lr"]
		print(
			f"Epoch [{epoch + 1}/{args.epochs}] | "
			f"Total: {avg_total:.5f}, Rec: {avg_rec:.5f}, Low: {avg_low:.5f}, gTV: {avg_tv:.6f}, Ratio: {avg_ratio:.6f}, "
			f"lr: {current_lr:.7f}"
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
