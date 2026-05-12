import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SAR metrics for prediction/GT pairs")
    parser.add_argument("--pred-dir", required=True, help="Directory with predicted images")
    parser.add_argument("--gt-dir", required=True, help="Directory with ground-truth images")
    parser.add_argument(
        "--metrics",
        default="psnr,ssim,rmse,enl,epi,tbr",
        help="Comma-separated metrics: psnr,ssim,rmse,enl,epi,tbr",
    )
    parser.add_argument("--crop-border", type=int, default=0, help="Crop border before computing PSNR/SSIM")
    parser.add_argument("--output-csv", default="./metrics.csv", help="Output CSV path")

    parser.add_argument("--enl-mask-dir", default="", help="Directory with ENL background masks")
    parser.add_argument("--enl-flat-percentile", type=float, default=20.0, help="Auto ENL flat-region percentile")

    parser.add_argument("--epi-edge-percentile", type=float, default=90.0, help="Edge percentile for EPI mask")

    parser.add_argument("--tbr-target-mask-dir", default="", help="Directory with TBR target masks")
    parser.add_argument("--tbr-background-mask-dir", default="", help="Directory with TBR background masks")
    parser.add_argument("--tbr-target-percentile", type=float, default=99.5, help="Auto TBR target percentile")
    parser.add_argument("--tbr-background-percentile", type=float, default=20.0, help="Auto TBR background percentile")
    parser.add_argument("--tbr-mask-source", choices=["gt", "pred"], default="gt")

    return parser.parse_args()


def list_pairs(pred_dir: str, gt_dir: str) -> List[str]:
    pred_names = {p.name for p in Path(pred_dir).iterdir() if p.is_file()}
    gt_names = {p.name for p in Path(gt_dir).iterdir() if p.is_file()}
    names = sorted(pred_names.intersection(gt_names))
    if not names:
        raise RuntimeError("No matched prediction/GT pairs found.")
    return names


def load_gray(path: str) -> np.ndarray:
    img = Image.open(path)
    arr = np.array(img)
    if arr.ndim == 3:
        arr = np.array(img.convert("L"))
    arr = arr.astype(np.float32)
    if arr.dtype == np.uint16 or arr.max() > 255:
        arr = arr / 65535.0
    else:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def load_mask(path: str, shape: Tuple[int, int]) -> np.ndarray:
    mask = np.array(Image.open(path).convert("L"))
    if mask.shape != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask > 0


def sobel_magnitude(x: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(x, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def compute_enl(img: np.ndarray, mask: np.ndarray, eps: float = 1e-6) -> float:
    vals = img[mask]
    if vals.size == 0:
        return float("nan")
    mean = float(np.mean(vals))
    var = float(np.var(vals))
    return (mean * mean) / (var + eps)


def compute_epi(pred: np.ndarray, gt: np.ndarray, edge_mask: np.ndarray, eps: float = 1e-6) -> float:
    pred_g = sobel_magnitude(pred)
    gt_g = sobel_magnitude(gt)
    vals_pred = pred_g[edge_mask]
    vals_gt = gt_g[edge_mask]
    if vals_gt.size == 0:
        return float("nan")
    return float(np.mean(vals_pred) / (np.mean(vals_gt) + eps))


def compute_tbr(img: np.ndarray, target_mask: np.ndarray, background_mask: np.ndarray, eps: float = 1e-6) -> float:
    tgt = img[target_mask]
    bg = img[background_mask]
    if tgt.size == 0 or bg.size == 0:
        return float("nan")
    return float(np.mean(tgt) / (np.mean(bg) + eps))


def auto_flat_mask(gt: np.ndarray, percentile: float) -> np.ndarray:
    grad = sobel_magnitude(gt)
    thresh = np.percentile(grad, percentile)
    return grad <= thresh


def auto_edge_mask(gt: np.ndarray, percentile: float) -> np.ndarray:
    grad = sobel_magnitude(gt)
    thresh = np.percentile(grad, percentile)
    return grad >= thresh


def auto_target_background_masks(img: np.ndarray, target_p: float, background_p: float) -> Tuple[np.ndarray, np.ndarray]:
    target_thresh = np.percentile(img, target_p)
    background_thresh = np.percentile(img, background_p)
    target_mask = img >= target_thresh
    background_mask = img <= background_thresh
    return target_mask, background_mask


def main() -> None:
    args = parse_args()
    metrics = [m.strip().lower() for m in args.metrics.split(",") if m.strip()]

    names = list_pairs(args.pred_dir, args.gt_dir)

    rows: List[Dict[str, float]] = []
    for name in names:
        pred = load_gray(str(Path(args.pred_dir) / name))
        gt = load_gray(str(Path(args.gt_dir) / name))

        if pred.shape != gt.shape:
            raise ValueError(f"Shape mismatch for {name}: pred {pred.shape}, gt {gt.shape}")

        row: Dict[str, float] = {"name": name}

        if "psnr" in metrics:
            pred_255 = (pred * 255.0).round()
            gt_255 = (gt * 255.0).round()
            psnr = calculate_psnr(pred_255[..., None], gt_255[..., None], args.crop_border, input_order="HWC")
            row["psnr"] = float(psnr)

        if "ssim" in metrics:
            pred_255 = (pred * 255.0).round()
            gt_255 = (gt * 255.0).round()
            ssim = calculate_ssim(pred_255[..., None], gt_255[..., None], args.crop_border, input_order="HWC")
            row["ssim"] = float(ssim)

        if "rmse" in metrics:
            rmse = float(np.sqrt(np.mean((pred - gt) ** 2)))
            row["rmse"] = rmse

        if "enl" in metrics:
            if args.enl_mask_dir:
                mask = load_mask(str(Path(args.enl_mask_dir) / name), pred.shape)
            else:
                mask = auto_flat_mask(gt, args.enl_flat_percentile)
            row["enl"] = compute_enl(pred, mask)

        if "epi" in metrics:
            edge_mask = auto_edge_mask(gt, args.epi_edge_percentile)
            row["epi"] = compute_epi(pred, gt, edge_mask)

        if "tbr" in metrics:
            if args.tbr_target_mask_dir and args.tbr_background_mask_dir:
                tgt_mask = load_mask(str(Path(args.tbr_target_mask_dir) / name), pred.shape)
                bg_mask = load_mask(str(Path(args.tbr_background_mask_dir) / name), pred.shape)
            else:
                source = gt if args.tbr_mask_source == "gt" else pred
                tgt_mask, bg_mask = auto_target_background_masks(
                    source, args.tbr_target_percentile, args.tbr_background_percentile
                )
            row["tbr_pred"] = compute_tbr(pred, tgt_mask, bg_mask)
            row["tbr_gt"] = compute_tbr(gt, tgt_mask, bg_mask)

        rows.append(row)

    if not rows:
        raise RuntimeError("No rows computed.")

    keys = sorted({k for row in rows for k in row.keys()})
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    summary: Dict[str, float] = {}
    for k in keys:
        if k == "name":
            continue
        vals = [row[k] for row in rows if k in row and np.isfinite(row[k])]
        if vals:
            summary[k] = float(np.mean(vals))

    print("Summary:")
    for k in sorted(summary.keys()):
        print(f"  {k}: {summary[k]:.6f}")


if __name__ == "__main__":
    main()
