#!/usr/bin/env python3
"""按冻结补充协议评价 C3-A 的多次生成样本集合。"""
from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
METRIC_ROOT = ROOT / "Early-FusionCascadeJDSR"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(METRIC_ROOT) not in sys.path:
    sys.path.insert(0, str(METRIC_ROOT))

from scripts.cli_zh import ChineseArgumentParser

try:
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from MMCANetV2.phase_c2_data import C2FixedDataset
    from MMCANetV2.phase_c3a_flow import SarTargetOnlyDataset, sample_heun, upsample_sar_condition
    from MMCANetV2.phase_c3a_sample_set import (
        energy_score,
        fair_energy_score_iid,
        fair_variogram_score_iid,
        mean_pixelwise_sample_std,
        pixelwise_mean,
        source_group_balanced_mean,
        variogram_score,
    )
    from scripts import evaluate_c3a_multisample as frozen_v3
    from scripts.run_phase_c3a_sar_flow import (
        INPUT_ROOT,
        PROTOCOL_PATH as TRAIN_PROTOCOL_PATH,
        SOURCE_MANIFEST,
        _cuda_compute_process_records,
        _csv_write,
        _environment,
        _json_write,
        sha256_file,
    )
except ModuleNotFoundError as error:
    if __name__ == "__main__":
        dependency = error.name or "未知依赖"
        print(f"运行错误：缺少依赖 {dependency}，无法执行 C3-A 多次生成评分。", file=sys.stderr)
        raise SystemExit(2) from None
    raise


SUPPLEMENT_PROTOCOL = ROOT / "protocols/mmcanet_phase_c3a/sample_set_evaluation_c1_v2_20261114_v3.json"
EXPECTED_SUPPLEMENT_SHA256 = "ddb0883fac67e86f5b062a9e58304ec9efd8eec9a1010b3af68c53e3fb3aa0e5"
SAMPLE_SET_MODULE = ROOT / "MMCANetV2/phase_c3a_sample_set.py"
OUTPUT_DIR = ROOT / "experiments/diagnostics/c3a_sample_set_evaluation_c1_v2_20261114_v3"

IDENTITY_METRICS = (
    "像素均值PSNR",
    "像素均值SSIM",
    "像素均值EPI",
    "像素均值边缘强度比",
    "平均逐像素样本标准差",
    "EnergyScore插件",
    "EnergyScoreIIDFair",
    "VariogramScore插件",
    "VariogramScoreIIDFair",
)
DRAW_METRICS = (
    "PSNR",
    "SSIM",
    "EPI",
    "边缘强度比",
    "最终裁剪比例",
    "原始输出低于0比例",
    "原始输出高于1比例",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_supplement_protocol() -> tuple[dict[str, Any], str, dict[str, Any], str]:
    raw = SUPPLEMENT_PROTOCOL.read_bytes()
    supplement_hash = hashlib.sha256(raw).hexdigest()
    if supplement_hash != EXPECTED_SUPPLEMENT_SHA256:
        raise ValueError("C3-A 多次生成补充协议 SHA-256 不匹配")
    protocol = json.loads(raw)
    if protocol.get("format") != "c3a_sample_set_evaluation_protocol_v3":
        raise ValueError("C3-A 多次生成补充协议格式不匹配")
    if protocol.get("status") != "prospective_not_yet_used_for_model_evaluation":
        raise ValueError("C3-A 多次生成补充协议状态不匹配")
    bound_module = protocol.get("implementation_binding", {})
    if (
        bound_module.get("scoring_module_path") != str(SAMPLE_SET_MODULE.relative_to(ROOT))
        or bound_module.get("scoring_module_sha256") != sha256_file(SAMPLE_SET_MODULE)
    ):
        raise ValueError("C3-A 多次生成评分实现与协议绑定不匹配")
    data = protocol.get("data", {})
    if (
        data.get("role") != "development"
        or data.get("payload_arrays_read") != ["sar_lr", "noisy_pseudo_target"]
        or data.get("optical_payload_read") is not False
        or data.get("train_payload_read") is not False
        or data.get("confirmation_payload_accessed") is not False
        or data.get("test_set_accessed") is not False
        or data.get("test_set_used") is not False
    ):
        raise ValueError("补充协议的数据角色或载荷访问边界不匹配")

    v3_protocol, v3_hash = frozen_v3._read_protocol()
    supersession = protocol.get("supersession", {})
    training = protocol.get("training_binding", {})
    sampling = protocol.get("sampling", {})
    v3_sampling = v3_protocol.get("sampling", {})
    draws = int(sampling.get("draws_per_identity", 0))
    if (
        supersession.get("c3a_v3_path") != str(frozen_v3.EVALUATION_PROTOCOL.relative_to(ROOT))
        or supersession.get("c3a_v3_sha256") != v3_hash
        or training.get("sha256") != v3_protocol.get("training_protocol", {}).get("sha256")
        or training.get("expected_run") != "experiments/phase_c3a_sar_only_flow_c1_v2_20261105_v1"
        or data.get("source_manifest_path") != v3_protocol.get("data", {}).get("source_manifest_path")
        or data.get("source_manifest_sha256") != v3_protocol.get("data", {}).get("source_manifest_sha256")
        or data.get("fixed_input_root") != v3_protocol.get("data", {}).get("fixed_input_root")
        or data.get("sample_count") != v3_protocol.get("data", {}).get("sample_count")
        or data.get("source_group_count") != v3_protocol.get("data", {}).get("source_group_count")
        or draws < 2
        or sampling.get("draw_indices") != list(range(draws))
        or draws != int(v3_sampling.get("draws_per_identity", -1))
        or int(sampling.get("seed", -1)) != int(v3_sampling.get("seed", -2))
    ):
        raise ValueError("补充协议与冻结 C3-A v3 采样或训练绑定不匹配")
    return protocol, supplement_hash, v3_protocol, v3_hash


def score_identity_samples(
    raw_draws: np.ndarray,
    target: np.ndarray,
    edge_mask: np.ndarray,
    *,
    sample_id: str,
    source_group_id: str,
    patch_id: str,
    expected_draws: int = 4,
    variogram_offsets: Sequence[tuple[int, int]] = ((0, 1), (1, 0), (1, 1), (1, -1), (0, 2), (2, 0)),
    variogram_power: float = 0.5,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """计算一个身份的单次 draw、像素均值和集合分数，不保存预测图。"""
    draws = np.asarray(raw_draws, dtype=np.float64)
    observation = np.asarray(target, dtype=np.float64)
    mask = np.asarray(edge_mask)
    if (
        draws.ndim != 4
        or draws.shape[1] != 1
        or draws.shape[0] != expected_draws
        or observation.shape != draws.shape[1:]
        or mask.shape != draws.shape[-2:]
        or mask.dtype != np.bool_
        or not mask.any()
        or not np.isfinite(draws).all()
        or not np.isfinite(observation).all()
        or np.any((observation < 0.0) | (observation > 1.0))
    ):
        raise ValueError("单身份 draw、target 或固定 edge mask 不符合协议张量合同")
    if any(not isinstance(value, str) or not value.strip() for value in (sample_id, source_group_id, patch_id)):
        raise ValueError("单身份样本、来源组或 patch 标记无效")

    batch = {"name": [sample_id], "source_group_id": [source_group_id], "patch_id": [patch_id]}
    target_tensor = torch.from_numpy(np.ascontiguousarray(observation[None, ...]))
    clipped_draws = np.clip(draws, 0.0, 1.0)
    draw_rows: list[dict[str, Any]] = []
    for draw_index in range(expected_draws):
        raw_tensor = torch.from_numpy(np.ascontiguousarray(draws[draw_index : draw_index + 1]))
        row = frozen_v3._metric_rows(
            raw_tensor,
            raw_tensor,
            target_tensor,
            batch,
            method_name="C3-A SAR-only Flow",
        )[0]
        frozen_v3._attach_clip_range_components([row], raw_tensor)
        row.update(
            frozen_v3._edge_metrics_with_mask(
                clipped_draws[draw_index, 0], observation[0], mask
            )
        )
        row["采样编号"] = draw_index
        draw_rows.append(row)

    mean_image = pixelwise_mean(clipped_draws)
    mean_tensor = torch.from_numpy(np.ascontiguousarray(mean_image[None, ...]))
    mean_row = frozen_v3._metric_rows(
        mean_tensor,
        mean_tensor,
        target_tensor,
        batch,
        method_name="C3-A 像素均值点估计",
    )[0]
    mean_row.update(
        frozen_v3._edge_metrics_with_mask(mean_image[0], observation[0], mask)
    )
    result = {
        "样本 ID": sample_id,
        "来源组 ID": source_group_id,
        "patch ID": patch_id,
        "像素均值PSNR": float(mean_row["PSNR"]),
        "像素均值SSIM": float(mean_row["SSIM"]),
        "像素均值EPI": float(mean_row["EPI"]),
        "像素均值边缘强度比": float(mean_row["边缘强度比"]),
        "平均逐像素样本标准差": mean_pixelwise_sample_std(clipped_draws),
        "EnergyScore插件": energy_score(clipped_draws, observation),
        "EnergyScoreIIDFair": fair_energy_score_iid(clipped_draws, observation),
        "VariogramScore插件": variogram_score(
            clipped_draws,
            observation,
            offsets_yx=variogram_offsets,
            power=variogram_power,
        ),
        "VariogramScoreIIDFair": fair_variogram_score_iid(
            clipped_draws,
            observation,
            offsets_yx=variogram_offsets,
            power=variogram_power,
        ),
    }
    if not np.isfinite(np.asarray([result[key] for key in IDENTITY_METRICS], dtype=np.float64)).all():
        raise FloatingPointError("单身份像素均值或样本集合评分含非有限值")
    return draw_rows, result


def aggregate_source_group_metric(rows: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    if not rows or any(metric not in row for row in rows):
        raise ValueError("来源组等权汇总缺少样本或指标")
    return source_group_balanced_mean(
        sample_ids=[str(row["样本 ID"]) for row in rows],
        source_group_ids=[str(row["来源组 ID"]) for row in rows],
        scores=[float(row[metric]) for row in rows],
    )


def _open_development_dataset() -> SarTargetOnlyDataset:
    fixed = C2FixedDataset(
        INPUT_ROOT / "development",
        SOURCE_MANIFEST,
        role="development",
        scale=2,
        verify_hashes_on_read=False,
    )
    if fixed.manifest.get("condition_id") != "x2_light":
        raise ValueError("C3-A 多次生成评价只接受冻结 x2_light 固定输入")
    return SarTargetOnlyDataset(fixed)


def _source_group_rows(
    identity_rows: list[dict[str, Any]],
    draw_rows: list[dict[str, Any]],
    draw_count: int,
) -> list[dict[str, Any]]:
    groups = sorted({str(row["来源组 ID"]) for row in identity_rows})
    output: list[dict[str, Any]] = []
    for group in groups:
        members = [row for row in identity_rows if row["来源组 ID"] == group]
        row: dict[str, Any] = {"来源组 ID": group, "样本数": len(members)}
        for metric in IDENTITY_METRICS:
            row[metric] = float(np.mean([float(item[metric]) for item in members], dtype=np.float64))
        for draw_index in range(draw_count):
            selected = [item for item in draw_rows if item["来源组 ID"] == group and item["采样编号"] == draw_index]
            if len(selected) != len(members):
                raise ValueError("来源组逐 draw 身份数不匹配")
            for metric in DRAW_METRICS:
                row[f"采样{draw_index}_{metric}"] = float(
                    np.mean([float(item[metric]) for item in selected], dtype=np.float64)
                )
        output.append(row)
    return output


def _build_summary(
    identity_rows: list[dict[str, Any]], draw_rows: list[dict[str, Any]], draw_count: int
) -> dict[str, Any]:
    identity_summary = {
        metric: aggregate_source_group_metric(identity_rows, metric) for metric in IDENTITY_METRICS
    }
    per_draw: dict[str, Any] = {}
    for draw_index in range(draw_count):
        selected = [row for row in draw_rows if row["采样编号"] == draw_index]
        per_draw[str(draw_index)] = {
            metric: aggregate_source_group_metric(selected, metric) for metric in DRAW_METRICS
        }
    return {
        "按来源组等权汇总的身份级对象": identity_summary,
        "按来源组等权汇总的单次draw": per_draw,
        "统计解释": {
            "单次draw": "每个 draw 单独报告；按身份、来源组汇总。",
            "像素均值点估计": "先在同一身份的四个裁剪输出上逐像素求 float64 均值，再与 noisy pseudo-target 配对。",
            "经验集合评分": "插件分数描述有限经验集合；fair 分数仅在 IID draw 假设下估计底层分布评分。",
            "离散度": "平均逐像素样本标准差仅描述四个 draw 的有限样本离散度，不等同校准或不确定性。",
            "推断": "本补充不计算区间、不用于 checkpoint 选择、不改变阶段 gate。",
        },
    }


def _read_training_protocol() -> dict[str, Any]:
    return json.loads(TRAIN_PROTOCOL_PATH.read_text(encoding="utf-8"))


@torch.inference_mode()
def _evaluate_batches(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    seed: int,
    draw_indices: Sequence[int],
    sampler_steps: int,
    noise_sigma: float,
    edge_masks_by_id: dict[str, np.ndarray],
    expected_count: int,
    variogram_offsets: Sequence[tuple[int, int]],
    variogram_power: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
    """在指定设备上流式采样；CPU 合成批次与正式 CUDA 入口共用此路径。"""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    model.eval()
    draw_rows: list[dict[str, Any]] = []
    identity_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for batch_index, batch in enumerate(loader, start=1):
        lr = batch["lr"].to(device, non_blocking=False)
        target = batch["target"].to(device, non_blocking=False)
        condition = upsample_sar_condition(lr, size=(256, 256))
        raw_batches: list[np.ndarray] = []
        for draw_index in draw_indices:
            noise = torch.cat(
                [
                    frozen_v3.draw_noise_for_identity(
                        str(sample_id), draw_index=int(draw_index), seed=seed, device=device
                    )
                    for sample_id in batch["name"]
                ],
                dim=0,
            )
            raw = sample_heun(
                model,
                condition,
                noise,
                steps=sampler_steps,
                noise_sigma=noise_sigma,
            )
            if raw.shape != target.shape or not torch.isfinite(raw).all():
                raise FloatingPointError("C3-A 多次生成输出尺寸不符或含 NaN/Inf")
            raw_batches.append(raw.detach().cpu().numpy().copy())

        stacked_draws = np.stack(raw_batches, axis=0)
        target_batch = target.detach().cpu().numpy()
        names = [str(value) for value in batch["name"]]
        groups = [str(value) for value in batch["source_group_id"]]
        patches = [str(value) for value in batch["patch_id"]]
        masks = frozen_v3._edge_masks_for_batch(batch, edge_masks_by_id)
        for index, sample_id in enumerate(names):
            one_draw_rows, one_identity = score_identity_samples(
                stacked_draws[:, index],
                target_batch[index],
                masks[index],
                sample_id=sample_id,
                source_group_id=groups[index],
                patch_id=patches[index],
                expected_draws=len(draw_indices),
                variogram_offsets=variogram_offsets,
                variogram_power=variogram_power,
            )
            for row in one_draw_rows:
                row["采样编号"] = int(draw_indices[row["采样编号"]])
            draw_rows.extend(one_draw_rows)
            identity_rows.append(one_identity)
        print(
            f"多次生成评分进度：{batch_index}/{len(loader)} 批，"
            f"已完成 {len(identity_rows)}/{expected_count} 个 development 身份。",
            flush=True,
        )

    if len(draw_rows) != expected_count * len(draw_indices) or len(identity_rows) != expected_count:
        raise ValueError("多次生成评分输出身份数与冻结 development 清单不一致")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return draw_rows, identity_rows, time.perf_counter() - started


def _validate_run_location(
    run_dir: Path,
    expected_run: Path,
    *,
    allow_compatible_run_location: bool,
) -> bool:
    if run_dir == expected_run:
        return False
    if not allow_compatible_run_location:
        raise ValueError("训练目录与补充协议绑定的 C3-A 运行不一致")
    try:
        run_dir.relative_to((ROOT / "experiments").resolve())
    except ValueError as error:
        raise ValueError("路径覆盖仅允许使用仓库 experiments/ 下的兼容运行目录") from error
    return True


def _run(
    training_run_dir: Path,
    output_dir: Path,
    *,
    user_authorized_resource_override: bool = False,
    allow_compatible_run_location: bool = False,
) -> int:
    protocol, supplement_hash, v3_protocol, v3_hash = _read_supplement_protocol()
    if output_dir.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖：{output_dir}")
    expected_run = (ROOT / protocol["training_binding"]["expected_run"]).resolve()
    run_dir = training_run_dir.resolve()
    run_location_overridden = _validate_run_location(
        run_dir,
        expected_run,
        allow_compatible_run_location=allow_compatible_run_location,
    )

    device, free_bytes, total_bytes = frozen_v3._check_resources(
        v3_protocol, user_authorized_override=user_authorized_resource_override
    )
    process_query_error = None
    try:
        observed_cuda_processes = _cuda_compute_process_records()
    except (OSError, subprocess.SubprocessError, RuntimeError) as error:
        if not user_authorized_resource_override:
            raise
        observed_cuda_processes = []
        process_query_error = f"{type(error).__name__}: {error}"
    # 当前 v3 JSON 将绑定写在 training_protocol.sha256；复用原校验函数时
    # 仅构造内存适配视图，不改冻结协议文件或旧评价入口。
    validation_protocol = dict(v3_protocol)
    validation_protocol["training_protocol_sha256"] = v3_protocol["training_protocol"]["sha256"]
    run_manifest, checkpoint = frozen_v3._validate_training_run(run_dir, validation_protocol, v3_hash)
    training_protocol = _read_training_protocol()
    input_manifest = json.loads((run_dir / "input_manifest.json").read_text(encoding="utf-8"))
    dataset = _open_development_dataset()
    fixed = run_manifest["fixed_input_records"]["development"]
    if (
        dataset.manifest_sha256 != fixed.get("manifest_sha256")
        or dataset.files_csv_sha256 != fixed.get("files_csv_sha256")
    ):
        raise ValueError("C3-A 训练与补充评价的 development 固定输入哈希不匹配")
    sample_count, source_group_count = frozen_v3._validate_development_identity(dataset, protocol["data"])
    if (
        input_manifest.get("roles", {}).get("development") != fixed
        or dataset.role != "development"
        or sample_count != int(protocol["data"]["sample_count"])
        or source_group_count != int(protocol["data"]["source_group_count"])
    ):
        raise ValueError("development 身份或固定输入角色与补充协议不一致")

    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short", "--untracked-files=normal"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    code_paths = {
        "scripts/evaluate_c3a_sample_set_supplement.py": Path(__file__).resolve(),
        "scripts/evaluate_c3a_multisample.py": Path(frozen_v3.__file__).resolve(),
        "scripts/run_phase_c3a_sar_flow.py": ROOT / "scripts/run_phase_c3a_sar_flow.py",
        "MMCANetV2/phase_c2_data.py": ROOT / "MMCANetV2/phase_c2_data.py",
        "MMCANetV2/phase_c2_runner.py": ROOT / "MMCANetV2/phase_c2_runner.py",
        "MMCANetV2/phase_c2_matched.py": ROOT / "MMCANetV2/phase_c2_matched.py",
        "MMCANetV2/phase_c3a_flow.py": ROOT / "MMCANetV2/phase_c3a_flow.py",
        "MMCANetV2/phase_c3a_metrics.py": ROOT / "MMCANetV2/phase_c3a_metrics.py",
        "MMCANetV2/phase_c3a_sample_set.py": SAMPLE_SET_MODULE,
        "JointJDSR/model.py": ROOT / "JointJDSR/model.py",
        "scripts/training_resume.py": ROOT / "scripts/training_resume.py",
    }
    code_hashes = {name: sha256_file(path) for name, path in code_paths.items()}
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, Any] = {
        "格式": "c3a_sample_set_evaluation_result_v3",
        "运行 ID": output_dir.name,
        "状态": "运行中",
        "开始时间 UTC": _utc_now(),
        "补充协议路径": str(SUPPLEMENT_PROTOCOL.relative_to(ROOT)),
        "补充协议 SHA-256": supplement_hash,
        "C3-A v3 协议 SHA-256": v3_hash,
        "训练运行目录": str(run_dir.relative_to(ROOT)),
        "训练运行清单 SHA-256": sha256_file(run_dir / "manifest.json"),
        "补充协议绑定的预期训练运行目录": str(expected_run.relative_to(ROOT)),
        "训练运行目录有偏离": run_location_overridden,
        "训练协议 SHA-256": run_manifest["protocol_sha256"],
        "最佳 checkpoint SHA-256": sha256_file(run_dir / "checkpoints/best.pt"),
        "输入清单 SHA-256": sha256_file(run_dir / "input_manifest.json"),
        "C1 来源清单 SHA-256": sha256_file(SOURCE_MANIFEST),
        "development 固定输入 manifest SHA-256": dataset.manifest_sha256,
        "development 固定输入 files.csv SHA-256": dataset.files_csv_sha256,
        "代码版本": head,
        "工作区存在未提交修改": bool(status.strip()),
        "相关代码 SHA-256": code_hashes,
        "development 样本数": sample_count,
        "development 来源组数": source_group_count,
        "每身份 draw 数": int(protocol["sampling"]["draws_per_identity"]),
        "draw 编号": protocol["sampling"]["draw_indices"],
        "development 载荷已访问": False,
        "train 载荷已访问": False,
        "optical 载荷已读取": False,
        "confirmation 载荷已访问": False,
        "test 载荷已访问": False,
        "test 已使用": False,
        "更新 checkpoint": False,
        "用于选择模型": False,
        "改变阶段 gate": False,
        "写入预测图像": False,
        "GPU 预检空闲字节": free_bytes,
        "GPU 总字节": total_bytes,
        "GPU 资源预检": {
            "冻结资源要求": v3_protocol["resources"],
            "观测到的 CUDA 计算进程": observed_cuda_processes,
            "进程清单读取错误": process_query_error,
            "用户授权覆盖": user_authorized_resource_override,
            "授权日期": "2026-11-23" if user_authorized_resource_override else None,
            "授权说明": (
                "用户明确授权在其评价任务并行运行，并忽略冻结协议的 3 GiB 显存预检；以 CUDA OOM 作为停止边界。"
                if user_authorized_resource_override
                else None
            ),
        },
    }
    try:
        _json_write(output_dir / "manifest.json", manifest)
        (output_dir / "protocol.json").write_bytes(SUPPLEMENT_PROTOCOL.read_bytes())
        (output_dir / "c3a_v3_protocol.json").write_bytes(frozen_v3.EVALUATION_PROTOCOL.read_bytes())
        (output_dir / "input_manifest.json").write_text(
            json.dumps(input_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (output_dir / "command.txt").write_text(
            "C3-A 多次生成评分命令：\n"
            + shlex.join(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--training-run-dir",
                    str(run_dir),
                    "--output-dir",
                    str(output_dir),
                    *( ["--user-authorized-resource-override"] if user_authorized_resource_override else [] ),
                    *( ["--allow-compatible-run-location"] if allow_compatible_run_location else [] ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        (output_dir / "git_state.txt").write_text(
            f"提交：{head}\n运行开始前工作区状态：\n{status}", encoding="utf-8"
        )

        manifest["development 载荷已访问"] = True
        _json_write(output_dir / "manifest.json", manifest)
        dataset.verify_all()
        edge_masks_by_id, edge_manifest_rows = frozen_v3._build_fixed_edge_masks(dataset)
        _csv_write(
            output_dir / "edge_mask_manifest.csv",
            edge_manifest_rows,
            ["样本 ID", "来源组 ID", "patch ID", "edge 阈值", "edge 像素数", "edge 覆盖率", "edge mask SHA-256"],
        )

        batch_size = int(v3_protocol["resources"]["batch_size"])
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
        model = frozen_v3._build_flow_model(training_protocol["method"]["model"]).to(device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()
        frozen_v3._set_evaluation_determinism(v3_protocol["determinism"])
        torch.cuda.reset_peak_memory_stats(device)
        sampling = protocol["sampling"]
        draw_indices = [int(value) for value in sampling["draw_indices"]]
        score_config = protocol["objects"]["sample_set_distribution"]["spatial_variogram_score"]
        offsets = tuple(tuple(int(value) for value in pair) for pair in score_config["offsets_yx"])
        variogram_power = float(score_config["power"])
        draw_rows, identity_rows, runtime = _evaluate_batches(
            model,
            loader,
            device=device,
            seed=int(sampling["seed"]),
            draw_indices=draw_indices,
            sampler_steps=int(v3_protocol["sampling"]["sampler_steps"]),
            noise_sigma=float(v3_protocol["sampling"]["noise_sigma"]),
            edge_masks_by_id=edge_masks_by_id,
            expected_count=sample_count,
            variogram_offsets=offsets,
            variogram_power=variogram_power,
        )
        group_rows = _source_group_rows(identity_rows, draw_rows, len(draw_indices))
        summary = _build_summary(identity_rows, draw_rows, len(draw_indices))
        _csv_write(
            output_dir / "single_draw_metrics.csv",
            draw_rows,
            ["样本 ID", "来源组 ID", "patch ID", "采样编号", *DRAW_METRICS],
        )
        _csv_write(
            output_dir / "identity_object_metrics.csv",
            identity_rows,
            ["样本 ID", "来源组 ID", "patch ID", *IDENTITY_METRICS],
        )
        group_fields = ["来源组 ID", "样本数", *IDENTITY_METRICS]
        group_fields.extend(f"采样{draw_index}_{metric}" for draw_index in draw_indices for metric in DRAW_METRICS)
        _csv_write(output_dir / "source_group_metrics.csv", group_rows, group_fields)
        _json_write(output_dir / "summary.json", summary)

        environment = _environment(device)
        environment_zh = {
            "Python 版本": environment.get("python"),
            "Python 解释器": environment.get("python_executable"),
            "PyTorch 版本": environment.get("pytorch"),
            "CUDA 运行时": environment.get("cuda_runtime"),
            "cuDNN 版本": environment.get("cudnn"),
            "驱动版本": environment.get("driver"),
            "GPU 设备": environment.get("device"),
            "GPU 总显存字节": environment.get("device_total_memory_bytes"),
            "GPU 预检空闲显存字节": environment.get("device_free_memory_before_run_bytes"),
            "平台": environment.get("platform"),
            "NumPy 版本": environment.get("numpy"),
            "TorchVision 版本": environment.get("torchvision"),
            "运行耗时秒": runtime,
            "推理峰值分配显存字节": int(torch.cuda.max_memory_allocated(device)),
            "确定性设置": v3_protocol["determinism"],
        }
        _json_write(output_dir / "environment.json", environment_zh)
        (output_dir / "environment.txt").write_text(
            "\n".join(f"{key}：{value}" for key, value in environment_zh.items()) + "\n",
            encoding="utf-8",
        )
        output_hashes = {
            name: sha256_file(output_dir / name)
            for name in (
                "edge_mask_manifest.csv",
                "single_draw_metrics.csv",
                "identity_object_metrics.csv",
                "source_group_metrics.csv",
                "summary.json",
            )
        }
        manifest.update(
            {
                "状态": "完成",
                "结束时间 UTC": _utc_now(),
                "实际生成 draw 数": len(draw_indices),
                "逐 draw 输出行数": len(draw_rows),
                "身份级输出行数": len(identity_rows),
                "来源组输出行数": len(group_rows),
                "评价耗时秒": runtime,
                "推理峰值分配显存字节": int(torch.cuda.max_memory_allocated(device)),
                "结果文件 SHA-256": output_hashes,
            }
        )
        _json_write(output_dir / "manifest.json", manifest)
        print(
            f"多次生成评分完成：{sample_count} 个 development 身份、{source_group_count} 个来源组、"
            f"{len(draw_indices)} 次 draw；结果目录：{output_dir}",
            flush=True,
        )
        return 0
    except Exception as error:
        manifest.update(
            {
                "状态": "失败",
                "结束时间 UTC": _utc_now(),
                "错误类型": type(error).__name__,
                "错误信息": str(error),
            }
        )
        _json_write(output_dir / "manifest.json", manifest)
        raise


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = ChineseArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-run-dir",
        type=Path,
        default=ROOT / "experiments/phase_c3a_sar_only_flow_c1_v2_20261105_v1",
        help="已完整结束的冻结 C3-A 训练运行目录",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="新的评分结果目录；目录已存在时拒绝覆盖",
    )
    parser.add_argument(
        "--user-authorized-resource-override",
        action="store_true",
        help="按用户 2026-11-23 授权忽略并行 CUDA 进程与 3 GiB 显存预检，并记录资源偏离。",
    )
    parser.add_argument(
        "--allow-compatible-run-location",
        action="store_true",
        help="允许使用 experiments/ 下通过冻结 C3-A manifest 与 checkpoint 核验的唯一运行目录，并记录路径偏离。",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    parser = ChineseArgumentParser(description=__doc__)
    args = _parse_args(argv)
    try:
        return _run(
            args.training_run_dir,
            args.output_dir,
            user_authorized_resource_override=args.user_authorized_resource_override,
            allow_compatible_run_location=args.allow_compatible_run_location,
        )
    except Exception as error:
        parser.exit(1, f"运行失败：{error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
