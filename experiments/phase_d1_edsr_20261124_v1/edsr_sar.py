"""基于仓库 vendored EDSR 的单通道 SAR ×2 任务适配。"""
from __future__ import annotations

import importlib.util
import math
import sys
import types
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EDSR_SOURCE = PROJECT_ROOT / "BasicSR/basicsr/archs/edsr_arch.py"
REGISTRY_SOURCE = PROJECT_ROOT / "BasicSR/basicsr/utils/registry.py"
_EDSR_CLASS = None


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 vendored 源文件：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_published_edsr_class():
    """从 vendored EDSR 源码加载网络，绕过未完整 vendoring 的 BasicSR 顶层包。"""
    global _EDSR_CLASS
    if _EDSR_CLASS is not None:
        return _EDSR_CLASS
    if not EDSR_SOURCE.is_file() or not REGISTRY_SOURCE.is_file():
        raise FileNotFoundError("仓库 vendored EDSR 或 registry 源文件缺失")

    basicsr_root = PROJECT_ROOT / "BasicSR/basicsr"
    basicsr = types.ModuleType("basicsr")
    basicsr.__path__ = [str(basicsr_root)]
    sys.modules["basicsr"] = basicsr
    utils = types.ModuleType("basicsr.utils")
    utils.__path__ = [str(basicsr_root / "utils")]
    sys.modules["basicsr.utils"] = utils
    registry = _load_module("basicsr.utils.registry", REGISTRY_SOURCE)
    registry.ARCH_REGISTRY = registry.Registry("arch")

    archs = types.ModuleType("basicsr.archs")
    archs.__path__ = [str(basicsr_root / "archs")]
    sys.modules["basicsr.archs"] = archs
    arch_util = types.ModuleType("basicsr.archs.arch_util")

    class ResidualBlockNoBN(nn.Module):
        def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
            super().__init__()
            self.res_scale = res_scale
            self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
            self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
            self.relu = nn.ReLU(inplace=True)
            if not pytorch_init:
                for layer in (self.conv1, self.conv2):
                    nn.init.kaiming_normal_(layer.weight)
                    layer.weight.data.mul_(0.1)
                    nn.init.zeros_(layer.bias)

        def forward(self, value):
            residual = self.conv2(self.relu(self.conv1(value)))
            return value + residual * self.res_scale

    def make_layer(block, count, **kwargs):
        return nn.Sequential(*(block(**kwargs) for _ in range(count)))

    class Upsample(nn.Sequential):
        def __init__(self, scale, num_feat):
            layers = []
            if (scale & (scale - 1)) == 0:
                for _ in range(int(math.log(scale, 2))):
                    layers.extend([nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1), nn.PixelShuffle(2)])
            elif scale == 3:
                layers.extend([nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1), nn.PixelShuffle(3)])
            else:
                raise ValueError(f"EDSR 不支持此倍率：{scale}")
            super().__init__(*layers)

    arch_util.ResidualBlockNoBN = ResidualBlockNoBN
    arch_util.make_layer = make_layer
    arch_util.Upsample = Upsample
    sys.modules["basicsr.archs.arch_util"] = arch_util
    edsr = _load_module("basicsr.archs.edsr_arch", EDSR_SOURCE)
    _EDSR_CLASS = edsr.EDSR
    return _EDSR_CLASS


def build_edsr_sar(edsr_class: type[nn.Module]) -> nn.Module:
    """建立单通道、倍率 ×2、输出限定到 SAR target 域的 EDSR 适配版."""
    class SAROnlyEDSR(nn.Module):
        def __init__(self):
            super().__init__()
            self.core = edsr_class(
                num_in_ch=1,
                num_out_ch=1,
                num_feat=64,
                num_block=16,
                upscale=2,
                res_scale=1,
                img_range=1.0,
                rgb_mean=(0.5, 0.5, 0.5),
            )
            self.core.mean = torch.tensor([0.5]).view(1, 1, 1, 1)
            self.output = nn.Sigmoid()
            self.experiment_config = {
                "身份": "TASK_ADAPTED_PUBLISHED",
                "方法": "EDSR SAR 单通道 x2 任务适配版",
                "上游源码": "BasicSR/basicsr/archs/edsr_arch.py",
                "结构": "num_feat=64, num_block=16, res_scale=1, no-BN, PixelShuffle x2",
                "输入输出通道": "1→1",
                "输入均值与 img_range": "0.5 / 1.0",
                "输出适配": "Sigmoid 映射到 [0,1]；属于 SAR 任务适配",
                "初始化": "随机初始化，不使用外部预训练权重",
                "方法边界": "保留 EDSR 主体结构；不是作者 SAR 权重或原论文训练方案复现",
            }

        def forward(self, sar_lr: torch.Tensor) -> torch.Tensor:
            if sar_lr.ndim != 4 or sar_lr.shape[1] != 1:
                raise ValueError("SAR LR 输入必须为单通道 [B,1,H,W]")
            if not torch.isfinite(sar_lr).all():
                raise ValueError("SAR LR 输入含非有限值")
            prediction = self.output(self.core(sar_lr))
            expected = (sar_lr.shape[0], 1, sar_lr.shape[-2] * 2, sar_lr.shape[-1] * 2)
            if tuple(prediction.shape) != expected:
                raise RuntimeError("EDSR 输出尺寸不符合 x2 SAR 合同")
            return prediction

    return SAROnlyEDSR()
