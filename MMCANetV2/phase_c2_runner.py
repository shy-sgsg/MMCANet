"""C2 训练与 paired screen 的执行前数据合同。"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import math
import sys

import torch
from torch.utils.data import DataLoader

from MMCANetV2.phase_c2_data import (
    C2FixedDataset,
    verify_frozen_c2_source_manifest,
)


def audit_c2_cross_condition_parity(
    condition_roots: Mapping[str, str | Path],
    source_manifest: str | Path,
    *,
    role: str,
) -> dict[str, object]:
    """只读四条件 manifest/files.csv，核验身份与共享源数组哈希。"""
    if role not in {"train", "development"}:
        raise ValueError("C2 运行只接受 train 或 development 角色")
    protocol, protocol_sha256 = verify_frozen_c2_source_manifest(source_manifest)
    frozen_conditions = protocol.get("conditions")
    required_conditions = {"x2_light", "x2_severe", "x4_light", "x4_severe"}
    if not isinstance(frozen_conditions, dict) or set(frozen_conditions) != required_conditions:
        raise ValueError("冻结 C2 协议必须包含四个冻结条件")
    if set(condition_roots) != required_conditions:
        raise ValueError("C2 输入目录必须恰好覆盖四个冻结条件")

    role_spec = protocol.get("roles", {}).get(
        "training" if role == "train" else "development", {}
    )
    if role_spec.get("role") != role:
        raise ValueError("冻结 C2 协议中的角色定义不匹配")
    expected_count = role_spec.get("samples")
    expected_group_count = role_spec.get("source_groups")

    reference_identities: tuple[tuple[str, str, str], ...] | None = None
    reference_payloads: dict[str, tuple[str, str, str, str]] | None = None
    condition_records: dict[str, dict[str, object]] = {}

    for condition_id in sorted(required_conditions):
        condition = frozen_conditions[condition_id]
        scale = condition.get("scale")
        if scale not in {2, 4}:
            raise ValueError(f"冻结条件 {condition_id} 的 scale 无效")
        dataset = C2FixedDataset(
            condition_roots[condition_id],
            source_manifest,
            role=role,
            scale=scale,
            verify_hashes_on_read=False,
        )
        if dataset.manifest.get("condition_id") != condition_id:
            raise ValueError(f"固定输入 manifest 的条件 ID 与 {condition_id} 不一致")
        if len(dataset) != expected_count:
            raise ValueError(f"条件 {condition_id} 的样本数不符合冻结协议")

        identities = tuple(
            (row["sample_id"], row["source_group_id"], row["patch_id"])
            for row in dataset.rows
        )
        if reference_identities is None:
            reference_identities = identities
        elif identities != reference_identities:
            raise ValueError(f"条件 {condition_id} 的 sample identity/group/patch 与其他条件不一致")

        payloads: dict[str, tuple[str, str, str, str]] = {}
        for row in dataset.rows:
            sample = row["sample_id"]
            fields = (
                "target_sha256",
                "optical_sha256",
                "sar_source_sha512",
                "optical_source_sha512",
            )
            values = tuple(row.get(field, "") for field in fields)
            expected_lengths = (64, 64, 128, 128)
            if any(
                len(value) != length
                or any(char not in "0123456789abcdef" for char in value)
                for value, length in zip(values, expected_lengths)
            ):
                raise ValueError(f"条件 {condition_id} 的共享源哈希格式无效：{sample}")
            payloads[sample] = values
        if reference_payloads is None:
            reference_payloads = payloads
        else:
            for sample, values in payloads.items():
                if sample not in reference_payloads:
                    raise ValueError(f"条件 {condition_id} 缺少共同身份：{sample}")
                expected = reference_payloads[sample]
                labels = ("target", "optical", "SAR source", "optical source")
                for label, actual_value, expected_value in zip(labels, values, expected):
                    if actual_value != expected_value:
                        raise ValueError(
                            f"条件 {condition_id} 的 {label} 哈希跨条件不一致：{sample}"
                        )

        group_count = len({row["source_group_id"] for row in dataset.rows})
        if group_count != expected_group_count:
            raise ValueError(f"条件 {condition_id} 的来源组数量不符合冻结协议")
        condition_records[condition_id] = {
            "scale": scale,
            "sample_count": len(dataset),
            "source_group_count": group_count,
            "files_csv_sha256": dataset.manifest["files_csv_sha256"],
        }

    if reference_identities is None or reference_payloads is None:
        raise ValueError("C2 四条件没有可核验样本")
    return {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha256,
        "source_manifest_sha256": protocol["source_manifest_sha256"],
        "role": role,
        "sample_count": len(reference_identities),
        "source_group_count": len({group for _sample, group, _patch in reference_identities}),
        "shared_target_optical_hashes": True,
        "shared_source_hashes": True,
        "conditions": condition_records,
    }


def _load_c2_model_class():
    """从仓库主线加载冻结 C2 模型。"""
    repository_root = Path(__file__).resolve().parents[1]
    model_source = repository_root / "MMCANet" / "src"
    if str(model_source) not in sys.path:
        sys.path.insert(0, str(model_source))
    from MMCANetV2.phase_c2_matched import C2MatchedJoint

    return C2MatchedJoint


def build_c2_initial_state(
    scale: int,
    model_config: Mapping[str, int],
    *,
    seed: int,
) -> dict[str, torch.Tensor]:
    """按固定种子生成单个 scale 的公共初始化状态。"""
    if scale not in {2, 4} or not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("C2 scale 或初始化 seed 无效")
    required = {"width", "blocks", "hr_width"}
    if set(model_config) != required:
        raise ValueError("C2 model_config 必须恰含 width、blocks、hr_width")
    model_class = _load_c2_model_class()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = model_class(
            scale=scale,
            optical_enabled=False,
            width=int(model_config["width"]),
            blocks=int(model_config["blocks"]),
            hr_width=int(model_config["hr_width"]),
        )
        return {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()
        }


def build_c2_model(
    scale: int,
    arm: str,
    model_config: Mapping[str, int],
    initial_state: Mapping[str, torch.Tensor],
):
    """构造指定 arm，并严格载入 scale 内共享的初始化状态。"""
    if arm not in {"sar_only", "optical_guided"}:
        raise ValueError("C2 arm 必须为 sar_only 或 optical_guided")
    if scale not in {2, 4}:
        raise ValueError("C2 scale 必须为 2 或 4")
    model_class = _load_c2_model_class()
    model = model_class(
        scale=scale,
        optical_enabled=(arm == "optical_guided"),
        width=int(model_config["width"]),
        blocks=int(model_config["blocks"]),
        hr_width=int(model_config["hr_width"]),
    )
    model.load_state_dict(dict(initial_state), strict=True)
    return model


def build_c2_dataloader(dataset, training: Mapping[str, object], *, role: str) -> DataLoader:
    """按冻结设置创建可复现的 train 或 development DataLoader。"""
    if role not in {"train", "development"} or getattr(dataset, "role", None) != role:
        raise ValueError("C2 DataLoader 的角色与数据集角色不匹配")
    seed = training.get("seed")
    batch_size = training.get("batch_size")
    num_workers = training.get("num_workers")
    drop_last = training.get("drop_last")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("C2 DataLoader seed 必须为整数")
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("C2 batch_size 必须为正整数")
    if num_workers != 0 or drop_last is not False:
        raise ValueError("C2 冻结 DataLoader 要求 num_workers=0 且 drop_last=false")
    if training.get("augmentation") != "none" or training.get("random_crops") is not False:
        raise ValueError("C2 冻结训练禁止随机增强和随机裁剪")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(role == "train"),
        generator=generator,
        num_workers=0,
        drop_last=False,
    )


def development_group_balanced_mean_psnr(
    rows: list[Mapping[str, object]],
) -> float:
    """先在每个来源组内平均 PSNR，再对来源组等权平均。"""
    grouped: dict[str, list[float]] = {}
    for row in rows:
        group = str(row.get("source_group_id", ""))
        if not group:
            raise ValueError("development PSNR 行缺少 source_group_id")
        value = float(row["psnr"])
        if not math.isfinite(value):
            raise ValueError("development PSNR 必须为有限数值")
        grouped.setdefault(group, []).append(value)
    if not grouped:
        raise ValueError("development PSNR 不能为空")
    return float(sum(sum(values) / len(values) for values in grouped.values()) / len(grouped))


def select_best_c2_epoch(
    rows: list[Mapping[str, object]],
) -> dict[str, object]:
    """按 group-balanced mean PSNR 取最大值，同分时保留更早 epoch。"""
    if not rows:
        raise ValueError("C2 checkpoint 选择缺少 development epoch 记录")
    ordered = sorted(rows, key=lambda row: int(row["epoch"]))
    seen_epochs: set[int] = set()
    best_row: dict[str, object] | None = None
    best_score = -math.inf
    for row in ordered:
        epoch = row.get("epoch")
        if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch <= 0:
            raise ValueError("C2 epoch 必须为正整数")
        if epoch in seen_epochs:
            raise ValueError("C2 epoch 记录重复")
        seen_epochs.add(epoch)
        score = float(row["development_group_balanced_mean_psnr"])
        if not math.isfinite(score):
            raise ValueError("C2 checkpoint 指标必须为有限数值")
        if best_row is None or score > best_score:
            best_row = dict(row)
            best_score = score
    if best_row is None:
        raise ValueError("C2 checkpoint 选择没有有效 epoch")
    return best_row


def _c2_batch_inputs(batch: Mapping[str, object], model, device: torch.device):
    """核对并搬运一个固定 C2 batch。"""
    scale = getattr(model, "scale", None)
    if scale not in {2, 4}:
        raise ValueError("C2 模型必须声明 scale=2 或 scale=4")
    try:
        sar_lr = batch["lr"].to(device)
        optical_hr = batch["optical"].to(device)
        target = batch["target"].to(device)
    except (KeyError, AttributeError) as exc:
        raise ValueError("C2 batch 缺少 lr、optical 或 target 张量") from exc
    if sar_lr.ndim != 4 or sar_lr.shape[1] != 1 or sar_lr.dtype != torch.float32:
        raise ValueError("C2 SAR batch 必须为 float32 [B,1,H,W]")
    batch_size, _, height, width = sar_lr.shape
    if tuple(optical_hr.shape) != (batch_size, 3, height * scale, width * scale):
        raise ValueError("C2 optical batch 的通道数或空间尺寸不匹配")
    if tuple(target.shape) != (batch_size, 1, height * scale, width * scale):
        raise ValueError("C2 target batch 的通道数或空间尺寸不匹配")
    if optical_hr.dtype != torch.float32 or target.dtype != torch.float32:
        raise ValueError("C2 optical/target batch 必须为 float32")
    if not torch.isfinite(sar_lr).all() or not torch.isfinite(optical_hr).all() or not torch.isfinite(target).all():
        raise ValueError("C2 输入 batch 含 NaN 或 Inf")
    if optical_hr.min() < 0 or optical_hr.max() > 1 or target.min() < 0 or target.max() > 1:
        raise ValueError("C2 optical/target batch 超出 [0,1]")
    return sar_lr, optical_hr, target


def _c2_forward(model, sar_lr, optical_hr, target):
    optical_input = optical_hr if getattr(model, "optical_enabled", False) else None
    prediction = model(sar_lr, optical_input)
    if tuple(prediction.shape) != tuple(target.shape):
        raise ValueError("C2 模型输出尺寸与 target 不匹配")
    if not torch.isfinite(prediction).all():
        raise ValueError("C2 模型输出含 NaN 或 Inf")
    if prediction.min() < 0 or prediction.max() > 1:
        raise ValueError("C2 模型输出超出 [0,1]")
    return prediction


def train_c2_epoch(
    model,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
) -> dict[str, float | int]:
    """执行一个无增强、L1、无混合精度的 C2 训练 epoch。"""
    model.train()
    loss_sum = 0.0
    sample_count = 0
    optimizer_steps = 0
    for batch in loader:
        sar_lr, optical_hr, target = _c2_batch_inputs(batch, model, device)
        optimizer.zero_grad(set_to_none=True)
        prediction = _c2_forward(model, sar_lr, optical_hr, target)
        loss = torch.nn.functional.l1_loss(prediction, target)
        if not torch.isfinite(loss):
            raise FloatingPointError("C2 L1 loss 含 NaN 或 Inf")
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise FloatingPointError(f"C2 参数梯度缺失或非有限：{name}")
        optimizer.step()
        for name, parameter in model.named_parameters():
            if not torch.isfinite(parameter).all():
                raise FloatingPointError(f"C2 参数更新后出现非有限值：{name}")
        count = int(target.shape[0])
        loss_sum += float(loss.detach().cpu()) * count
        sample_count += count
        optimizer_steps += 1
    if sample_count == 0:
        raise ValueError("C2 train loader 为空")
    return {
        "train_l1": loss_sum / sample_count,
        "sample_count": sample_count,
        "optimizer_steps": optimizer_steps,
    }


def evaluate_c2_epoch(
    model,
    loader: DataLoader,
    *,
    device: torch.device,
) -> dict[str, object]:
    """计算 development 逐图 PSNR 与等权来源组 checkpoint 指标。"""
    model.eval()
    per_image: list[dict[str, object]] = []
    with torch.inference_mode():
        for batch in loader:
            sar_lr, optical_hr, target = _c2_batch_inputs(batch, model, device)
            prediction = _c2_forward(model, sar_lr, optical_hr, target)
            mse = (prediction.to(torch.float64) - target.to(torch.float64)).square().mean(dim=(1, 2, 3))
            if not torch.isfinite(mse).all() or (mse <= 0).any():
                raise FloatingPointError("C2 development PSNR 的均方误差为零或非有限值")
            psnr_values = 10.0 * torch.log10(1.0 / mse)
            names = list(batch["name"])
            groups = list(batch["source_group_id"])
            patches = list(batch["patch_id"])
            if not (len(names) == len(groups) == len(patches) == len(psnr_values)):
                raise ValueError("C2 development batch 的身份字段数量不匹配")
            for name, group, patch, value in zip(names, groups, patches, psnr_values):
                score = float(value.cpu())
                if not math.isfinite(score):
                    raise FloatingPointError("C2 development PSNR 非有限")
                per_image.append(
                    {
                        "sample_id": str(name),
                        "source_group_id": str(group),
                        "patch_id": str(patch),
                        "psnr": score,
                    }
                )
    return {
        "sample_count": len(per_image),
        "group_count": len({str(row["source_group_id"]) for row in per_image}),
        "development_group_balanced_mean_psnr": development_group_balanced_mean_psnr(per_image),
        "per_image": per_image,
    }
