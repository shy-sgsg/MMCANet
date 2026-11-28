"""表示域 Task B 的固定指标与观测重投影误差。"""
from __future__ import annotations

import math

import numpy as np

from MMCANetV2.phase_c3a_metrics import edge_metrics, sobel_magnitude, ssim


METRIC_NAMES = (
    "psnr_db",
    "ssim",
    "epi_corr",
    "edge_strength_ratio",
    "bright_mean_ratio",
    "bright_p95_ratio",
    "background_mean_ratio",
    "reprojection_mae",
)


def _image(value: np.ndarray, name: str) -> np.ndarray:
    image = np.asarray(value, dtype=np.float64)
    if image.ndim == 3 and image.shape[0] == 1:
        image = image[0]
    if image.ndim != 2 or not np.isfinite(image).all():
        raise ValueError(f"{name} 必须为有限二维数组")
    return image


def _order_mask(values: np.ndarray, *, count: int, largest: bool) -> np.ndarray:
    flat = values.reshape(-1)
    count = min(max(int(count), 1), flat.size)
    if largest:
        selected = np.argpartition(flat, flat.size - count)[flat.size - count :]
    else:
        selected = np.argpartition(flat, count - 1)[:count]
    mask = np.zeros(flat.size, dtype=bool)
    mask[selected] = True
    return mask.reshape(values.shape)


def _edge_metrics_with_optional_epi(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    *,
    allow_undefined_epi: bool,
) -> dict[str, float | None]:
    if not allow_undefined_epi:
        return edge_metrics(prediction, target, mask)
    pred_edge = sobel_magnitude(prediction)[mask]
    target_edge = sobel_magnitude(target)[mask]
    target_mean = float(target_edge.mean())
    if not math.isfinite(target_mean) or abs(target_mean) <= 1e-12:
        raise ValueError("目标梯度强度为零或过小，无法计算有效比值")
    strength_ratio = float(pred_edge.mean() / target_mean)
    pred_centered = pred_edge - pred_edge.mean()
    target_centered = target_edge - target_edge.mean()
    denominator = math.sqrt(
        float(np.sum(pred_centered**2) * np.sum(target_centered**2))
    )
    if not math.isfinite(denominator) or abs(denominator) <= 1e-12:
        return {"epi_corr": None, "edge_strength_ratio": strength_ratio}
    correlation = float(np.sum(pred_centered * target_centered) / denominator)
    if not math.isfinite(correlation) or not math.isfinite(strength_ratio):
        raise FloatingPointError("边缘指标结果不是有限值")
    return {"epi_corr": correlation, "edge_strength_ratio": strength_ratio}


def compute_taskb_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    low_resolution_input: np.ndarray,
    *,
    allow_undefined_epi: bool = False,
) -> dict[str, float | None]:
    """计算固定 range=1 指标；high/background mask 只由目标排序定义。"""
    pred = _image(prediction, "预测")
    truth = _image(target, "参考")
    lr = _image(low_resolution_input, "LR 输入")
    if pred.shape != truth.shape:
        raise ValueError("预测和参考尺寸不一致")
    if pred.shape[0] % 2 or pred.shape[1] % 2:
        raise ValueError("HR 高宽必须为偶数")
    if lr.shape != (pred.shape[0] // 2, pred.shape[1] // 2):
        raise ValueError("LR 输入必须是 HR 的二分之一尺寸")
    if pred.min() < 0 or pred.max() > 1 or truth.min() < 0 or truth.max() > 1:
        raise ValueError("预测与参考必须位于固定 [0,1] 指标域")
    if lr.min() < 0:
        raise ValueError("LR 输入必须非负")

    mse = float(np.mean(np.square(pred - truth), dtype=np.float64))
    psnr = math.inf if mse == 0 else 10.0 * math.log10(1.0 / mse)
    interior = np.zeros(truth.shape, dtype=bool)
    interior[1:-1, 1:-1] = True
    edges = _edge_metrics_with_optional_epi(
        pred, truth, interior, allow_undefined_epi=allow_undefined_epi
    )

    pixel_count = truth.size
    bright_mask = _order_mask(truth, count=math.ceil(pixel_count * 0.01), largest=True)
    background_mask = _order_mask(truth, count=math.ceil(pixel_count * 0.50), largest=False)
    bright_truth = truth[bright_mask]
    bright_pred = pred[bright_mask]
    background_truth_mean = float(truth[background_mask].mean())
    if background_truth_mean <= 1e-12:
        raise ValueError("背景参考均值为零，无法计算背景比值")
    if np.any(bright_truth <= 1e-12):
        raise ValueError("亮回波参考含零值，无法计算有效比值")

    reprojected = pred.reshape(pred.shape[0] // 2, 2, pred.shape[1] // 2, 2).mean(
        axis=(1, 3), dtype=np.float64
    )
    result = {
        "psnr_db": float(psnr),
        "ssim": ssim(pred, truth, data_range=1.0),
        **edges,
        "bright_mean_ratio": float(bright_pred.mean() / bright_truth.mean()),
        "bright_p95_ratio": float(
            np.percentile(bright_pred, 95, method="linear")
            / np.percentile(bright_truth, 95, method="linear")
        ),
        "background_mean_ratio": float(
            pred[background_mask].mean() / background_truth_mean
        ),
        "reprojection_mae": float(np.mean(np.abs(reprojected - lr), dtype=np.float64)),
    }
    invalid = []
    for name, value in result.items():
        if value is None and name == "epi_corr" and allow_undefined_epi:
            continue
        if value is None or (
            not math.isfinite(value) and not (name == "psnr_db" and math.isinf(value))
        ):
            invalid.append(name)
    if invalid:
        raise FloatingPointError(f"Task B 指标包含非有限值：{invalid}")
    return result
