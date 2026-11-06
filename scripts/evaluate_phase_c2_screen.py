#!/usr/bin/env python3
"""使用冻结 C2 checkpoint 在 development 上运行 matched/permuted/missing 配对评价。"""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from MMCANetV2.phase_c2_data import (  # noqa: E402
    C2FixedDataset,
    sha256_file,
    verify_frozen_c2_source_manifest,
)
from MMCANetV2.phase_c2_execution import build_c2_training_specs  # noqa: E402
from MMCANetV2.phase_c2_runner import (  # noqa: E402
    audit_c2_cross_condition_parity,
    build_c2_initial_state,
    build_c2_model,
)
from MMCANetV2.phase_c2_screen import (  # noqa: E402
    build_cross_group_permutation,
    decide_c2_screen,
    evaluate_c2_sample_controls,
    summarize_c2_paired_contrasts,
    build_c2_paired_contrast_rows,
    write_c2_screen_artifacts,
)


class ChineseArgumentParser(argparse.ArgumentParser):
    """将 argparse 面向用户的固定标题改为简体中文。"""

    def format_usage(self) -> str:
        return super().format_usage().replace("usage:", "用法：", 1)

    def format_help(self) -> str:
        return super().format_help().replace("usage:", "用法：", 1).replace(
            "可选参数:", "可选参数：", 1
        )

    def error(self, message: str) -> None:
        if message.startswith("unrecognized arguments: "):
            message = "无法识别参数：" + message.removeprefix("unrecognized arguments: ")
        elif message.startswith("the following arguments are required: "):
            message = "缺少必需参数：" + message.removeprefix("the following arguments are required: ")
        self._print_message(f"{self.prog}: 参数错误：{message}\n", sys.stderr)
        self._print_message(self.format_usage(), sys.stderr)
        self.exit(2)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_nested(path: Path, other: Path) -> bool:
    return path == other or path.is_relative_to(other) or other.is_relative_to(path)


def _write_csv(path: Path, rows: list[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError("拒绝写入空的置换映射")
    fields = list(rows[0])
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()


def _git_facts() -> dict[str, object]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short", "--untracked-files=all"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    return {"commit": head, "dirty": bool(status), "status_entry_count": len(status)}


def _environment_facts() -> dict[str, object]:
    try:
        torchvision_version = importlib.metadata.version("torchvision")
    except importlib.metadata.PackageNotFoundError:
        torchvision_version = None
    driver_version = None
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0 and result.stdout.strip():
        driver_version = result.stdout.splitlines()[0].strip()
    properties = torch.cuda.get_device_properties(0)
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "pytorch": torch.__version__,
        "torchvision": torchvision_version,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "driver": driver_version,
        "platform": platform.platform(),
        "device": properties.name,
        "device_total_memory_bytes": int(properties.total_memory),
        "device_count": torch.cuda.device_count(),
    }


def validate_c2_screen_matrix_manifest(
    matrix: Mapping[str, object], protocol: Mapping[str, object], protocol_sha256: str
) -> list[dict[str, object]]:
    """确认八个冻结训练全部完成且没有 test/confirmation 访问。"""
    specs = build_c2_training_specs(protocol)
    identity = matrix.get("identity")
    runs = matrix.get("runs")
    if (
        matrix.get("format") != "c2_training_matrix_v1"
        or matrix.get("status") != "training_complete"
        or matrix.get("confirmation_payload_accessed") is not False
        or matrix.get("test_set_accessed") is not False
        or matrix.get("test_set_used") is not False
    ):
        raise ValueError("C2 训练矩阵未完成或访问边界标记无效")
    if not isinstance(identity, Mapping) or not isinstance(runs, Mapping):
        raise ValueError("C2 训练矩阵缺少 identity 或 runs")
    if (
        identity.get("protocol_id") != protocol.get("protocol_id")
        or identity.get("protocol_sha256") != protocol_sha256
        or identity.get("source_manifest_sha256") != protocol.get("source_manifest_sha256")
        or identity.get("training") != protocol.get("training")
        or identity.get("model") != protocol.get("model")
        or identity.get("conditions") != protocol.get("conditions")
        or identity.get("run_specs") != specs
    ):
        raise ValueError("C2 训练矩阵的冻结协议、配置或八组矩阵不匹配")
    expected_ids = {str(spec["run_id"]) for spec in specs}
    if set(runs) != expected_ids or any(
        not isinstance(runs[run_id], Mapping) or runs[run_id].get("status") != "complete"
        for run_id in expected_ids
    ):
        raise ValueError("C2 八组训练不完整，拒绝开始 paired screen")
    return specs


def preflight_c2_cuda_inference(protocol: Mapping[str, object]) -> dict[str, object]:
    """在读取固定数组前，以合成单样本检查冻结模型的 CUDA 推理可用性。"""
    if not torch.cuda.is_available():
        raise RuntimeError("C2 paired screen 需要 CUDA；当前不可用")
    device = torch.device("cuda:0")
    free_before, total_bytes = torch.cuda.mem_get_info(device)
    properties = torch.cuda.get_device_properties(device)
    model_config = protocol["model"]
    state = build_c2_initial_state(
        2,
        {
            "width": int(model_config["width"]),
            "blocks": int(model_config["blocks"]),
            "hr_width": int(model_config["hr_width"]),
        },
        seed=int(protocol["training"]["seed"]),
    )
    model = build_c2_model(
        2,
        "optical_guided",
        {
            "width": int(model_config["width"]),
            "blocks": int(model_config["blocks"]),
            "hr_width": int(model_config["hr_width"]),
        },
        state,
    ).to(device)
    model.eval()
    try:
        torch.cuda.reset_peak_memory_stats(device)
        with torch.inference_mode():
            sar = torch.zeros((1, 1, 128, 128), dtype=torch.float32, device=device)
            optical = torch.zeros((1, 3, 256, 256), dtype=torch.float32, device=device)
            output = model(sar, optical)
            if tuple(output.shape) != (1, 1, 256, 256) or not torch.isfinite(output).all():
                raise RuntimeError("C2 CUDA 合成推理输出 shape 或数值无效")
        torch.cuda.synchronize(device)
        free_after, _ = torch.cuda.mem_get_info(device)
        return {
            "status": "passed",
            "device": properties.name,
            "device_total_memory_bytes": int(total_bytes),
            "free_memory_before_bytes": int(free_before),
            "free_memory_after_bytes": int(free_after),
            "peak_allocated_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            "synthetic_input_shapes": {"sar_lr": [1, 1, 128, 128], "optical_hr": [1, 3, 256, 256]},
            "project_data_read": False,
        }
    finally:
        del model, state
        torch.cuda.empty_cache()


def _current_development_input_records(
    input_root: Path, condition_ids: tuple[str, ...]
) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    for condition in condition_ids:
        path = input_root / condition / "development"
        manifest_path = path / "manifest.json"
        files_path = path / "files.csv"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("condition_id") != condition
            or manifest.get("role") != "development"
            or manifest.get("status") != "complete"
            or manifest.get("confirmation_payload_accessed") is not False
            or manifest.get("test_set_accessed") is not False
            or manifest.get("test_set_used") is not False
        ):
            raise ValueError(f"固定输入 manifest 的 development 角色或边界标记无效：{condition}")
        records[f"{condition}/development"] = {
            "root": str(path.resolve()),
            "manifest_sha256": sha256_file(manifest_path),
            "files_csv_sha256": sha256_file(files_path),
        }
    return records


def _load_best_c2_model(
    *,
    matrix_root: Path,
    matrix: Mapping[str, object],
    protocol: Mapping[str, object],
    protocol_sha256: str,
    condition_id: str,
    arm: str,
):
    run_id = f"{condition_id}__{arm}"
    run_dir = matrix_root / "runs" / run_id
    run_manifest_path = run_dir / "manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    run_summary = matrix["runs"][run_id]
    spec = next(item for item in build_c2_training_specs(protocol) if item["run_id"] == run_id)
    context = run_manifest.get("context")
    if (
        run_manifest.get("format") != "c2_training_run_v1"
        or run_manifest.get("status") != "complete"
        or run_manifest.get("confirmation_payload_accessed") is not False
        or run_manifest.get("test_set_accessed") is not False
        or run_manifest.get("test_set_used") is not False
        or not isinstance(context, Mapping)
        or context.get("run_id") != run_id
        or context.get("condition_id") != condition_id
        or context.get("arm") != arm
        or context.get("scale") != spec["scale"]
        or context.get("optical_enabled") != spec["optical_enabled"]
        or context.get("protocol_sha256") != protocol_sha256
        or context.get("source_manifest_sha256") != protocol.get("source_manifest_sha256")
        or run_summary.get("best_epoch") != run_manifest.get("best_epoch")
        or run_summary.get("best_score") != run_manifest.get("best_score")
    ):
        raise ValueError(f"{run_id} 的完整性、协议身份或 best 选择记录无效")
    expected_input_record = matrix["identity"]["input_records"].get(f"{condition_id}/development")
    if context.get("input_records", {}).get("development") != expected_input_record:
        raise ValueError(f"{run_id} 的 development 输入 manifest 与矩阵不一致")

    initial_path = matrix_root / "initialization" / f"scale_{spec['scale']}.pt"
    initial_sha = sha256_file(initial_path)
    if context.get("initial_state_sha256") != initial_sha:
        raise ValueError(f"{run_id} 的公共初始化 checkpoint hash 不一致")
    initial = torch.load(initial_path, map_location="cpu", weights_only=False)
    if (
        initial.get("format") != "c2_shared_initialization_v1"
        or initial.get("protocol_sha256") != protocol_sha256
        or initial.get("scale") != spec["scale"]
        or initial.get("seed") != protocol["training"]["seed"]
    ):
        raise ValueError(f"{run_id} 的公共初始化元数据不匹配")

    best_path = run_dir / "checkpoints" / "best.pt"
    if run_manifest.get("best_checkpoint_sha256") != sha256_file(best_path):
        raise ValueError(f"{run_id} 的 best checkpoint hash 不一致")
    checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("format") != "c2_best_checkpoint_v1"
        or checkpoint.get("context") != dict(context)
        or checkpoint.get("training") != protocol.get("training")
        or checkpoint.get("epoch") != run_manifest.get("best_epoch")
        or checkpoint.get("score") != run_manifest.get("best_score")
    ):
        raise ValueError(f"{run_id} 的 best checkpoint 配置或选择指标不匹配")
    model_config = {
        "width": int(protocol["model"]["width"]),
        "blocks": int(protocol["model"]["blocks"]),
        "hr_width": int(protocol["model"]["hr_width"]),
    }
    model = build_c2_model(
        int(spec["scale"]), arm, model_config, initial["state_dict"]
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if sum(parameter.numel() for parameter in model.parameters()) != context.get("parameter_count"):
        raise ValueError(f"{run_id} 的参数总数与训练记录不一致")
    return model.cuda().eval()


def _prediction_numpy(model, sar: torch.Tensor, optical: torch.Tensor | None) -> np.ndarray:
    prediction = model(sar, optical)
    if tuple(prediction.shape) != (1, 1, sar.shape[-2] * model.scale, sar.shape[-1] * model.scale):
        raise ValueError("C2 推理输出尺寸与冻结 scale 不匹配")
    if not torch.isfinite(prediction).all() or prediction.min() < 0 or prediction.max() > 1:
        raise ValueError("C2 推理输出含非有限值或超出 [0,1]")
    return prediction[0, 0].detach().to(device="cpu", dtype=torch.float32).numpy()


def _evaluate_condition(
    *,
    condition_id: str,
    dataset: C2FixedDataset,
    matrix_root: Path,
    matrix: Mapping[str, object],
    protocol: Mapping[str, object],
    protocol_sha256: str,
    permutation: Mapping[str, str],
    sample_evaluator: Callable[..., list[dict[str, object]]] | None = None,
) -> list[dict[str, object]]:
    scale = int(protocol["conditions"][condition_id]["scale"])
    guided = _load_best_c2_model(
        matrix_root=matrix_root,
        matrix=matrix,
        protocol=protocol,
        protocol_sha256=protocol_sha256,
        condition_id=condition_id,
        arm="optical_guided",
    )
    sar_only = _load_best_c2_model(
        matrix_root=matrix_root,
        matrix=matrix,
        protocol=protocol,
        protocol_sha256=protocol_sha256,
        condition_id=condition_id,
        arm="sar_only",
    )
    guided_count = sum(parameter.numel() for parameter in guided.parameters())
    sar_count = sum(parameter.numel() for parameter in sar_only.parameters())
    if guided_count != sar_count:
        raise ValueError(f"{condition_id} 两个训练 arm 的参数量不相等")
    sample_indices = {row["sample_id"]: index for index, row in enumerate(dataset.rows)}
    if set(sample_indices) != set(permutation):
        raise ValueError(f"{condition_id} 的置换映射与 development 样本集合不一致")

    rows: list[dict[str, object]] = []
    device = torch.device("cuda:0")
    with torch.inference_mode():
        for index, source_row in enumerate(dataset.rows):
            item = dataset[index]
            sample = str(source_row["sample_id"])
            donor_sample = permutation[sample]
            donor_row = dataset.rows[sample_indices[donor_sample]]
            donor_optical = dataset._load_array(
                donor_row, "optical", (3, 256, 256), verify_hash=False
            )
            sar = item["lr"].unsqueeze(0).to(device)
            matched_optical = item["optical"].unsqueeze(0).to(device)
            permuted_optical = torch.from_numpy(np.ascontiguousarray(donor_optical)).unsqueeze(0).to(device)
            predictions = {
                "matched": _prediction_numpy(guided, sar, matched_optical),
                "permuted": _prediction_numpy(guided, sar, permuted_optical),
                "missing": _prediction_numpy(guided, sar, None),
                "sar_only": _prediction_numpy(sar_only, sar, None),
            }
            evaluator = sample_evaluator or evaluate_c2_sample_controls
            rows.extend(
                evaluator(
                    item["lr"].numpy(),
                    item["target"].numpy(),
                    predictions,
                    condition_id=condition_id,
                    sample_id=sample,
                    source_group_id=str(source_row["source_group_id"]),
                    patch_id=str(source_row["patch_id"]),
                    scale=scale,
                )
            )
            del sar, matched_optical, permuted_optical, donor_optical, predictions
    del guided, sar_only
    torch.cuda.empty_cache()
    return rows


def run_phase_c2_screen(
    *,
    source_manifest: str | Path,
    input_root: str | Path,
    matrix_root: str | Path,
    output_root: str | Path,
) -> dict[str, object]:
    source_manifest = Path(source_manifest).resolve()
    input_root = Path(input_root).resolve()
    matrix_root = Path(matrix_root).resolve()
    output_root = Path(output_root).resolve()
    protocol, protocol_sha256 = verify_frozen_c2_source_manifest(source_manifest)
    matrix_path = matrix_root / "manifest.json"
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    specs = validate_c2_screen_matrix_manifest(matrix, protocol, protocol_sha256)
    if not input_root.is_dir():
        raise FileNotFoundError("C2 development 固定输入根目录不存在")
    if output_root.exists():
        raise FileExistsError("拒绝覆盖已存在的 C2 paired screen 输出目录")
    if any(_is_nested(output_root, path) for path in (source_manifest, input_root, matrix_root)):
        raise ValueError("C2 screen 输出目录不得与来源、输入或训练矩阵路径重叠")

    condition_ids = tuple(protocol["conditions"])
    current_inputs = _current_development_input_records(input_root, condition_ids)
    frozen_inputs = matrix["identity"].get("input_records")
    if not isinstance(frozen_inputs, Mapping) or any(
        frozen_inputs.get(key) != record for key, record in current_inputs.items()
    ):
        raise ValueError("当前 development manifest/files.csv 与训练矩阵不一致")
    parity = audit_c2_cross_condition_parity(
        {condition: input_root / condition / "development" for condition in condition_ids},
        source_manifest,
        role="development",
    )
    resource_preflight = preflight_c2_cuda_inference(protocol)

    datasets: dict[str, C2FixedDataset] = {}
    for condition in condition_ids:
        scale = int(protocol["conditions"][condition]["scale"])
        dataset = C2FixedDataset(
            input_root / condition / "development",
            source_manifest,
            role="development",
            scale=scale,
            verify_hashes_on_read=False,
        )
        if len(dataset) != int(protocol["roles"]["development"]["samples"]):
            raise ValueError(f"{condition} 的 development 数量不符合冻结协议")
        dataset.verify_all()
        datasets[condition] = dataset

    reference_ids = [row["sample_id"] for row in datasets[condition_ids[0]].rows]
    reference_groups = [row["source_group_id"] for row in datasets[condition_ids[0]].rows]
    for condition in condition_ids[1:]:
        if [row["sample_id"] for row in datasets[condition].rows] != reference_ids:
            raise ValueError("四条件 development 样本顺序或集合不一致")
        if [row["source_group_id"] for row in datasets[condition].rows] != reference_groups:
            raise ValueError("四条件 development 来源组身份不一致")
    permutation_rows = build_cross_group_permutation(
        [{"role": "development", **row} for row in datasets[condition_ids[0]].rows],
        offset=int(protocol["permutation"]["offset"]),
    )
    permutation = {
        str(row["receiver_sample_id"]): str(row["donor_sample_id"])
        for row in permutation_rows
    }
    if len(permutation) != len(reference_ids) or set(permutation) != set(reference_ids):
        raise ValueError("冻结跨组错配映射不覆盖完整 development 样本")

    output_root.mkdir(parents=True, exist_ok=False)
    execution_manifest: dict[str, object] = {
        "format": "c2_screen_execution_v1",
        "status": "running",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha256,
        "source_manifest_sha256": protocol["source_manifest_sha256"],
        "source_manifest_path": str(source_manifest),
        "input_root": str(input_root),
        "matrix_root": str(matrix_root),
        "matrix_manifest_sha256": sha256_file(matrix_path),
        "current_development_inputs": current_inputs,
        "parity_audit": parity,
        "cuda_preflight": resource_preflight,
        "git": _git_facts(),
        "environment": _environment_facts(),
        "argv": sys.argv,
        "training_run_count": len(specs),
        "development_sample_count": len(reference_ids),
        "development_source_group_count": len(set(reference_groups)),
        "completed_conditions": [],
        "confirmation_payload_accessed": False,
        "test_set_accessed": False,
        "test_set_used": False,
        "started_utc": _utc_now(),
    }
    execution_manifest_path = output_root / "execution_manifest.json"
    execution_manifest_path.write_text(
        json.dumps(execution_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    started = time.perf_counter()
    method_rows: list[dict[str, object]] = []
    try:
        torch.use_deterministic_algorithms(bool(protocol["training"]["deterministic_algorithms"]))
        torch.backends.cudnn.benchmark = bool(protocol["training"]["cudnn_benchmark"])
        torch.cuda.reset_peak_memory_stats(torch.device("cuda:0"))
        for condition in condition_ids:
            condition_rows = _evaluate_condition(
                condition_id=condition,
                dataset=datasets[condition],
                matrix_root=matrix_root,
                matrix=matrix,
                protocol=protocol,
                protocol_sha256=protocol_sha256,
                permutation=permutation,
            )
            method_rows.extend(condition_rows)
            execution_manifest["completed_conditions"].append(condition)
            execution_manifest["method_metric_row_count"] = len(method_rows)
            execution_manifest["updated_utc"] = _utc_now()
            execution_manifest_path.write_text(
                json.dumps(execution_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

        if len(method_rows) != len(condition_ids) * len(reference_ids) * 4:
            raise ValueError("C2 paired screen 逐图输出数与冻结开发集覆盖不一致")
        paired_rows = build_c2_paired_contrast_rows(method_rows, protocol)
        summary_rows, group_rows = summarize_c2_paired_contrasts(paired_rows, protocol)
        decision = decide_c2_screen(summary_rows, protocol)
        write_c2_screen_artifacts(
            output_root,
            protocol=protocol,
            method_rows=method_rows,
            paired_rows=paired_rows,
            summary_rows=summary_rows,
            group_rows=group_rows,
            permutation_rows=permutation_rows,
            decision=decision,
            provenance={
                "matrix_manifest_sha256": execution_manifest["matrix_manifest_sha256"],
                "matrix_root": str(matrix_root),
                "development_input_records": current_inputs,
                "parity_audit": parity,
                "training_run_count": len(specs),
                "cuda_preflight": resource_preflight,
                "synthetic_validation": False,
            },
        )
        execution_manifest.update(
            status="complete",
            method_metric_row_count=len(method_rows),
            paired_contrast_row_count=len(paired_rows),
            decision=decision,
            runtime_seconds=time.perf_counter() - started,
            peak_allocated_memory_bytes=int(torch.cuda.max_memory_allocated(torch.device("cuda:0"))),
            ended_utc=_utc_now(),
        )
        execution_manifest_path.write_text(
            json.dumps(execution_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"C2 development 配对屏幕已完成：{output_root}", flush=True)
        print(f"探索性决策：{decision['status']}；不改变 B1 晋级门槛。", flush=True)
        return execution_manifest
    except Exception as exc:
        execution_manifest.update(
            status="failed",
            failure_type=type(exc).__name__,
            failure_message=str(exc),
            runtime_seconds=time.perf_counter() - started,
            failed_utc=_utc_now(),
        )
        execution_manifest_path.write_text(
            json.dumps(execution_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise


def main() -> None:
    parser = ChineseArgumentParser(description=__doc__, add_help=False)
    parser._optionals.title = "可选参数"
    parser.add_argument("-h", "--help", action="help", help="显示此帮助并退出")
    parser.add_argument("--source-manifest", type=Path, required=True, help="冻结 C1 v2 来源清单")
    parser.add_argument("--input-root", type=Path, required=True, help="四条件固定 development 输入根目录")
    parser.add_argument("--matrix-root", type=Path, required=True, help="已完成的 C2 八组训练矩阵目录")
    parser.add_argument("--output-root", type=Path, required=True, help="全新的唯一 paired screen 输出目录")
    args = parser.parse_args()
    try:
        run_phase_c2_screen(
            source_manifest=args.source_manifest,
            input_root=args.input_root,
            matrix_root=args.matrix_root,
            output_root=args.output_root,
        )
    except Exception as exc:
        if isinstance(exc, FileNotFoundError):
            message = "冻结来源清单、development 固定输入或训练 checkpoint 文件不存在，请核对路径。"
        elif isinstance(exc, PermissionError):
            message = "没有读取 development 输入或写入 screen 目录的权限。"
        elif isinstance(exc, ValueError):
            message = str(exc)
        elif isinstance(exc, torch.cuda.OutOfMemoryError):
            message = "CUDA 显存不足；screen 未改变冻结模型或推理控制。"
        else:
            message = "paired screen 执行失败；请检查运行目录中的执行 manifest。"
        print(f"C2 配对屏幕失败：{message}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
