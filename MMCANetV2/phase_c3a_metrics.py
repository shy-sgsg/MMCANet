"""C3-A 表示域指标的最小、可审计实现。"""
from __future__ import annotations

import math

import cv2
import numpy as np


def ssim(pred: np.ndarray, gt: np.ndarray, data_range: float = 1.0) -> float:
    """使用 11×11、高斯 sigma=1.5 的表示域 SSIM，并裁去 5 像素边界。"""
    prediction = np.asarray(pred, dtype=np.float64)
    target = np.asarray(gt, dtype=np.float64)
    if prediction.ndim != 2 or target.ndim != 2 or prediction.shape != target.shape:
        raise ValueError("SSIM 输入尺寸必须相同且为二维图像")
    if min(prediction.shape) < 11:
        raise ValueError("SSIM 输入图像尺寸不得小于 11×11")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("SSIM 输入含非有限值")
    if not math.isfinite(float(data_range)) or float(data_range) <= 0:
        raise ValueError("SSIM data_range 必须为正的有限数")

    mu_x = cv2.GaussianBlur(prediction, (11, 11), 1.5)
    mu_y = cv2.GaussianBlur(target, (11, 11), 1.5)
    mu_x_sq = mu_x * mu_x
    mu_y_sq = mu_y * mu_y
    mu_xy = mu_x * mu_y
    sigma_x_sq = cv2.GaussianBlur(prediction * prediction, (11, 11), 1.5) - mu_x_sq
    sigma_y_sq = cv2.GaussianBlur(target * target, (11, 11), 1.5) - mu_y_sq
    sigma_xy = cv2.GaussianBlur(prediction * target, (11, 11), 1.5) - mu_xy
    c1 = (0.01 * float(data_range)) ** 2
    c2 = (0.03 * float(data_range)) ** 2
    value = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
    )
    return float(np.mean(value[5:-5, 5:-5]))


def sobel_magnitude(array: np.ndarray) -> np.ndarray:
    value = np.asarray(array, dtype=np.float64)
    if value.ndim != 2 or not np.isfinite(value).all():
        raise ValueError("Sobel 输入必须是有限二维图像")
    gx = cv2.Sobel(value, cv2.CV_64F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT_101)
    gy = cv2.Sobel(value, cv2.CV_64F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT_101)
    return np.sqrt(gx * gx + gy * gy)


def edge_metrics(
    pred: np.ndarray, gt: np.ndarray, mask: np.ndarray, eps: float = 1e-12
) -> dict[str, float]:
    prediction = np.asarray(pred, dtype=np.float64)
    target = np.asarray(gt, dtype=np.float64)
    edge_mask = np.asarray(mask, dtype=bool)
    if prediction.ndim != 2 or prediction.shape != target.shape or edge_mask.shape != target.shape:
        raise ValueError("边缘指标要求预测、目标和掩膜尺寸相同且为二维")
    if not edge_mask.any():
        raise ValueError("边缘指标掩膜为空")
    pred_edge = sobel_magnitude(prediction)[edge_mask]
    gt_edge = sobel_magnitude(target)[edge_mask]
    gt_edge_mean = float(gt_edge.mean())
    if not math.isfinite(gt_edge_mean) or abs(gt_edge_mean) <= eps:
        raise ValueError("目标梯度强度为零或过小，无法计算有效比值")
    strength_ratio = float(pred_edge.mean() / gt_edge_mean)
    pred_centered = pred_edge - pred_edge.mean()
    gt_centered = gt_edge - gt_edge.mean()
    denominator = math.sqrt(float(np.sum(pred_centered**2) * np.sum(gt_centered**2)))
    if not math.isfinite(denominator) or abs(denominator) <= eps:
        raise ValueError("EPI 相关系数分母为零或过小，无法计算有效相关")
    correlation = float(np.sum(pred_centered * gt_centered) / denominator)
    if not math.isfinite(correlation) or not math.isfinite(strength_ratio):
        raise FloatingPointError("边缘指标结果不是有限值")
    return {"epi_corr": correlation, "edge_strength_ratio": strength_ratio}


def _image2d(array: np.ndarray, label: str) -> np.ndarray:
    value = np.asarray(array, dtype=np.float64)
    if value.ndim == 3 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 2 or not np.isfinite(value).all():
        raise ValueError(f"{label} 必须为有限二维图像")
    return value


def build_fixed_masks(
    lr: np.ndarray, target: np.ndarray, scale: int
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """复现 C2 screen 冻结的 edge/high-return/background 掩膜定义。"""
    if scale not in {2, 4}:
        raise ValueError("C2 mask scale 必须为 2 或 4")
    lr_image = _image2d(lr, "LR SAR")
    target_image = _image2d(target, "HR pseudo-target")
    if target_image.shape != (lr_image.shape[0] * scale, lr_image.shape[1] * scale):
        raise ValueError("C2 LR/HR 掩膜尺寸与声明倍率不匹配")

    edge_magnitude = sobel_magnitude(target_image)
    edge_threshold = float(np.percentile(edge_magnitude, 90.0, method="linear"))
    high_threshold = float(np.percentile(lr_image, 99.5, method="linear"))
    background_threshold = float(np.percentile(lr_image, 20.0, method="linear"))
    edge_mask = edge_magnitude >= edge_threshold
    high_lr = lr_image >= high_threshold
    background_lr = lr_image <= background_threshold
    high_mask = np.repeat(np.repeat(high_lr, scale, axis=0), scale, axis=1)
    background_mask = np.repeat(np.repeat(background_lr, scale, axis=0), scale, axis=1)
    masks = {"edge": edge_mask, "high_return": high_mask, "background": background_mask}
    if any(not mask.any() for mask in masks.values()):
        raise ValueError("C2 固定评价掩膜为空")
    if np.any(high_mask & background_mask):
        raise ValueError("C2 输入高回波与背景掩膜重叠")
    if edge_mask.all():
        raise ValueError("C2 edge mask 覆盖全图")
    thresholds = {"edge": edge_threshold, "high_return": high_threshold, "background": background_threshold}
    return masks, thresholds
