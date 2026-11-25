"""经固定 C2 x2-light 算子计算 SAR 表示域观测能量。"""
from __future__ import annotations

import hashlib
import sys
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEGRADATION_PATH = PROJECT_ROOT / "MMCANet" / "src"
EXPECTED_DEGRADATION_SHA256 = (
    "b7c0e9208a132792824bdde4296d7d3650587a144bf7ab4ec34bb161e0cb7bd1"
)
POSITIVE_EPSILON = 1e-8
GAMMA_LOOKS = 8.0

if str(DEGRADATION_PATH) not in sys.path:
    sys.path.insert(0, str(DEGRADATION_PATH))
from degradation import build_sar_lr_from_physical_degradation  # noqa: E402


@lru_cache(maxsize=1)
def _verify_degradation_hash() -> None:
    path = DEGRADATION_PATH / "degradation.py"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != EXPECTED_DEGRADATION_SHA256:
        raise RuntimeError("固定退化算子 SHA-256 不匹配，拒绝继续")


def apply_representation_operator(
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """返回未截断低分辨率值 z 与 clamp 后均值 a，输入为 [B,1,256,256]。"""
    if target.ndim != 4 or target.shape[1:] != (1, 256, 256):
        raise ValueError("target shape 必须为 [B,1,256,256]")
    if not torch.is_floating_point(target) or not torch.isfinite(target).all():
        raise ValueError("target 必须是有限浮点张量")
    if torch.any((target < 0.0) | (target > 1.0)):
        raise ValueError("target 必须位于 [0,1]")
    _verify_degradation_hash()
    z_values, a_values = [], []
    for sample in target:
        _, steps = build_sar_lr_from_physical_degradation(
            sample,
            input_downsample_size=128,
            downsample_factor=2,
            use_freq_truncation=True,
            freq_window_size=129,
            use_anisotropic_blur=False,
            use_speckle=False,
            speckle_looks=GAMMA_LOOKS,
            speckle_domain="representation",
            negative_policy="clamp",
            clamp_output=False,
            return_steps=True,
        )
        z_values.append(steps["downsample"])
        a_values.append(steps["pre_speckle"])
    return torch.stack(z_values), torch.stack(a_values)


def _energy_from_intermediates(
    z: torch.Tensor,
    a: torch.Tensor,
    observation: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """计算正值 Gamma 负对数似然和零值 clamp 单侧约束。"""
    if z.shape != a.shape or z.shape != observation.shape:
        raise ValueError("z、a 与 observation shape 必须完全一致")
    if not all(torch.isfinite(value).all() for value in (z, a, observation)):
        raise ValueError("z、a、observation 必须全部 finite")
    if torch.any(a < 0.0) or torch.any(observation < 0.0):
        raise ValueError("a 与 observation 必须非负")

    positive = observation > 0.0
    zero = ~positive
    differentiable_zero = (z.sum() + a.sum()) * 0.0
    if positive.any():
        selected_a = a[positive].clamp_min(POSITIVE_EPSILON)
        selected_y = observation[positive]
        positive_nll = GAMMA_LOOKS * (
            selected_a.log() + selected_y / selected_a
        ).mean()
    else:
        positive_nll = differentiable_zero
    if zero.any():
        zero_constraint = F.relu(z[zero]).square().mean()
        zero_violation_rate = (z[zero] > 0.0).to(z.dtype).mean()
    else:
        zero_constraint = differentiable_zero
        zero_violation_rate = differentiable_zero
    return {
        "positive_nll": positive_nll,
        "zero_constraint": zero_constraint,
        "zero_violation_rate": zero_violation_rate,
        "total": positive_nll + zero_constraint,
    }


def observation_energy(
    prediction: torch.Tensor,
    observation: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """对 [B,1,256,256] HR 估计和固定 [B,1,128,128] LR 计算观测能量。"""
    if prediction.ndim != 4 or prediction.shape[1:] != (1, 256, 256):
        raise ValueError("prediction shape 必须为 [B,1,256,256]")
    if observation.shape != (prediction.shape[0], 1, 128, 128):
        raise ValueError("observation shape 必须为 [B,1,128,128]")
    if prediction.device != observation.device or prediction.dtype != observation.dtype:
        raise ValueError("prediction 与 observation 的 device/dtype 必须相同")
    if not torch.isfinite(prediction).all() or not torch.isfinite(observation).all():
        raise ValueError("prediction 与 observation 必须全部 finite")
    if torch.any((prediction < 0.0) | (prediction > 1.0)):
        raise ValueError("prediction 必须位于 [0,1]")
    if torch.any(observation < 0.0):
        raise ValueError("observation 必须非负")
    z, a = apply_representation_operator(prediction)
    return _energy_from_intermediates(z, a, observation)
