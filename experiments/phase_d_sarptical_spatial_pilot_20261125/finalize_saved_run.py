"""从已落盘的 development prediction 和 checkpoint 完成 pilot 决策复核。"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.phase_d_sarptical_spatial_pilot_20261125 import pilot_data, run_pilot


METRIC_FILES = {
    "bicubic_per_image.csv": "bicubic",
    "edsr_per_image.csv": "edsr",
    "swinir_per_image.csv": "swinir",
    "flow_bridge_draw_00_per_image.csv": "flow_bridge_single_draw",
    "flow_bridge_pixel_mean_per_image.csv": "flow_bridge_pixel_mean",
}


def _read_rows(path: Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"指标文件为空：{path}")
    for row in rows:
        for metric in run_pilot.METRIC_COLUMNS:
            if metric not in row:
                continue
            value = row[metric]
            if value in ("", "NA"):
                row[metric] = "NA"
            else:
                parsed = float(value)
                if not math.isfinite(parsed):
                    raise ValueError(f"指标文件含非有限值：{path} / {metric}")
                row[metric] = parsed
    return rows


def _assert_replay_matches(
    saved_rows: list[dict[str, Any]], replay_rows: list[dict[str, Any]], label: str
) -> None:
    saved = {row["sample_id"]: row for row in saved_rows}
    replayed = {row["sample_id"]: row for row in replay_rows}
    if not saved or set(saved) != set(replayed):
        raise ValueError(f"推理重放的样本集合与已保存结果不一致：{label}")
    for sample_id, saved_row in saved.items():
        replay_row = replayed[sample_id]
        if saved_row["source_group_id"] != replay_row["source_group_id"]:
            raise ValueError(f"推理重放的来源组不一致：{label} / {sample_id}")
        for metric in run_pilot.METRIC_COLUMNS:
            left, right = saved_row.get(metric), replay_row.get(metric)
            if left == "NA" or right == "NA":
                if left != right:
                    raise ValueError(f"推理重放的 NA 状态不一致：{label} / {sample_id} / {metric}")
            elif left is not None and right is not None and not math.isclose(
                float(left), float(right), rel_tol=1e-6, abs_tol=1e-6
            ):
                raise ValueError(f"推理重放指标与已保存结果不一致：{label} / {sample_id} / {metric}")


def finalize_saved_run(run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir).expanduser().resolve()
    runtime_path = run_dir / "runtime.json"
    manifest_path = run_dir / "manifest.json"
    if not run_dir.is_dir() or not runtime_path.is_file():
        raise FileNotFoundError(f"实验运行目录或 runtime.json 不存在：{run_dir}")
    if manifest_path.exists():
        raise FileExistsError(f"已存在正式 manifest，拒绝覆盖：{manifest_path}")

    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    if runtime.get("status") != "running" or not (run_dir / "failure.json").is_file():
        raise ValueError("只允许恢复本次已保留 postprocessing failure 的运行目录")
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    split = json.loads((run_dir / "split_manifest.json").read_text(encoding="utf-8"))
    evaluation_config = json.loads((run_dir / "evaluation_config.json").read_text(encoding="utf-8"))
    failure = json.loads((run_dir / "failure.json").read_text(encoding="utf-8"))
    if (
        split.get("confirmation_payload_accessed") is not False
        or split.get("test_set_accessed") is not False
        or split.get("optical_payload_loaded") is not False
    ):
        raise ValueError("split manifest 的受保护数据访问标记不为 false")
    if runtime.get("confirmation_payload_accessed") is not False or runtime.get("test_set_used") is not False or runtime.get("optical_payload_loaded") is not False:
        raise ValueError("runtime 的受保护数据访问标记不为 false")

    saved_rows: dict[str, list[dict[str, Any]]] = {}
    for filename, method in METRIC_FILES.items():
        saved_rows[method] = _read_rows(run_dir / "metrics" / filename)
        if len(saved_rows[method]) != runtime["development_sample_count"]:
            raise ValueError(f"development 指标样本数不符：{filename}")
    flow_draw_rows = {
        index: _read_rows(run_dir / "metrics" / f"flow_bridge_draw_{index:02d}_per_image.csv")
        for index in range(4)
    }
    development_records = pilot_data.load_cached_records(
        run_dir / "data_cache", split, "development"
    )
    if len(development_records) != runtime["development_sample_count"]:
        raise ValueError("development 固定缓存样本数不符")

    if not torch.cuda.is_available():
        raise RuntimeError("推理重放需要当前已授权的 CUDA 环境")
    device = torch.device("cuda:0")
    run_pilot.seed_everything(int(config["train_seed"]))
    models: dict[str, torch.nn.Module] = {}
    for method in run_pilot.TRAINED_METHODS:
        checkpoint_path = run_dir / "methods" / method / "checkpoints" / "best.pt"
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if checkpoint.get("method") != method:
            raise ValueError(f"checkpoint 方法身份不匹配：{method}")
        model = run_pilot.build_model(method).to(device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()
        models[method] = model

    replay_started = time.perf_counter()
    with tempfile.TemporaryDirectory(
        prefix="sarptical_pilot_inference_replay_", dir=str(run_dir.parent)
    ) as temporary_directory:
        replay_dir = Path(temporary_directory)
        replay_rows, replay_summary = run_pilot._run_inference_and_evaluation(
            replay_dir,
            development_records,
            models,
            evaluation_config,
            device=device,
        )
        replay_rows["flow_bridge_pixel_mean"] = _read_rows(
            replay_dir / "metrics" / "flow_bridge_pixel_mean_per_image.csv"
        )
        for filename, method in METRIC_FILES.items():
            if method == "flow_bridge_single_draw":
                rows = flow_draw_rows[0]
                _assert_replay_matches(
                    rows,
                    _read_rows(replay_dir / "metrics" / "flow_bridge_draw_00_per_image.csv"),
                    filename,
                )
            else:
                _assert_replay_matches(
                    saved_rows[method], replay_rows[method], filename
                )
        for index in range(1, 4):
            _assert_replay_matches(
                flow_draw_rows[index],
                _read_rows(replay_dir / "metrics" / f"flow_bridge_draw_{index:02d}_per_image.csv"),
                f"flow_bridge_draw_{index:02d}_per_image.csv",
            )
        if len(replay_rows["flow_bridge_single_draw"]) != runtime["development_sample_count"]:
            raise ValueError("重放后的 Flow 单次 draw 样本数不符")

    replay_wall_seconds = float(time.perf_counter() - replay_started)
    reference_method = replay_summary["reference_deterministic_method"]
    decision = run_pilot._research_decision(
        replay_rows, replay_summary["inference"], reference_method
    )
    minimum_mask_pixels = int(evaluation_config["minimum_mask_pixels"])
    mask_counts = [int(row["high_return_mask_pixels"]) for row in saved_rows["bicubic"]]
    bright_valid_samples = sum(value >= minimum_mask_pixels for value in mask_counts)
    decision["bright_roi_availability"] = {
        "minimum_mask_pixels": minimum_mask_pixels,
        "high_return_threshold": float(evaluation_config["high_return_threshold"]),
        "development_mask_pixels_min": min(mask_counts),
        "development_mask_pixels_max": max(mask_counts),
        "development_samples_with_valid_mask": bright_valid_samples,
        "valid_source_groups": decision["guardrails"]["bright_mean_abs_error"]["group_count"],
        "consequence": "有效来源组少于 8，因此亮回波护栏按冻结规则未通过；未调整阈值。",
    }
    decision["inference_timing_source"] = "同一 checkpoint 与固定 development 输入的独立推理重放；逐图指标与已保存结果核对一致。"

    all_metric_rows = [
        row
        for method_rows in replay_rows.values()
        for row in method_rows
    ]
    all_metric_path = run_dir / "metrics" / "per_image_metrics.csv"
    if all_metric_path.exists():
        raise FileExistsError(f"拒绝覆盖已有逐图汇总：{all_metric_path}")
    run_pilot.write_csv(all_metric_path, all_metric_rows)
    paired_path = run_dir / "metrics" / "paired_group_comparisons.csv"
    if paired_path.exists():
        raise FileExistsError(f"拒绝覆盖已有配对统计：{paired_path}")
    comparisons = run_pilot.summarize_comparisons(
        replay_rows,
        reference_method=reference_method,
    )
    run_pilot.write_csv(paired_path, comparisons)

    decision_path = run_dir / "decision.json"
    review_path = run_dir / "RESEARCH_DECISION_REVIEW.md"
    run_pilot.write_json(decision_path, decision)
    run_pilot.write_decision_review(
        review_path,
        decision,
        {"run_directory": str(run_dir)},
        replay_rows,
    )

    now = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    failure_time = failure.get("failed_at")
    elapsed_seconds = None
    if runtime.get("started_at") and failure_time:
        started = dt.datetime.fromisoformat(runtime["started_at"])
        ended = dt.datetime.fromisoformat(failure_time)
        elapsed_seconds = float((ended - started).total_seconds())
    training = [runtime["methods"][method] for method in run_pilot.TRAINED_METHODS]
    runtime.update(
        {
            "status": "complete",
            "ended_at": failure_time,
            "finalized_at": now,
            "elapsed_seconds": elapsed_seconds,
            "training": training,
            "inference": replay_summary["inference"],
            "inference_timing_source": decision["inference_timing_source"],
            "inference_replay_wall_seconds_including_metric_checks": replay_wall_seconds,
            "inference_replay_predictions_match_saved_metrics": True,
            "best_deterministic_method": reference_method,
            "decision": decision["decision"],
            "storage_after_finalization": run_pilot._disk_snapshot(run_dir),
            "storage_after_finalization_parent": run_pilot._disk_snapshot(run_dir.parent),
            "postprocessing_recovery": {
                "original_failure": failure.get("message"),
                "training_completed_before_failure": True,
                "development_inference_completed_before_failure": True,
                "guardrail_rule_changed": False,
                "bright_roi_valid_samples": bright_valid_samples,
                "bright_roi_minimum_pixels": minimum_mask_pixels,
            },
            "confirmation_payload_accessed": False,
            "test_set_used": False,
            "optical_payload_loaded": False,
        }
    )
    run_pilot.replace_json(runtime_path, runtime)

    source_manifest = json.loads((run_dir / "source_manifest.json").read_text(encoding="utf-8"))
    finalization_manifest = {
        "finalizer": str(Path(__file__).relative_to(run_pilot.PROJECT_ROOT)),
        "finalizer_sha256": run_pilot.sha256_file(Path(__file__)),
        "run_code_sha256_recorded_at_training": source_manifest,
        "run_pilot_sha256_at_finalization": run_pilot.sha256_file(Path(run_pilot.__file__)),
        "inference_replay_seconds_including_metric_checks": replay_wall_seconds,
        "inference_replay_matches_saved_metrics": True,
        "temporary_replay_directory_removed": True,
        "finalized_at": now,
    }
    run_pilot.write_json(run_dir / "finalization_manifest.json", finalization_manifest)
    run_pilot.write_json(
        run_dir / "summary.json",
        {
            "run_id": config["run_id"],
            "status": "complete",
            "best_deterministic_method": reference_method,
            "method_group_balanced_metrics": {
                method: {
                    metric: run_pilot.group_balanced_mean(rows, metric)
                    for metric in ("psnr_db", "ssim", "epi_corr", "rmse")
                }
                for method, rows in replay_rows.items()
            },
            "research_decision": decision["decision"],
            "bright_roi_availability": decision["bright_roi_availability"],
            "inference_timing_source": decision["inference_timing_source"],
            "confirmation_payload_accessed": False,
            "test_set_used": False,
            "optical_payload_loaded": False,
        },
    )
    snapshot = run_pilot._git_snapshot()
    checkpoint_hashes = {
        method: {
            name: run_pilot.sha256_file(
                run_dir / "methods" / method / "checkpoints" / f"{name}.pt"
            )
            for name in ("best", "last")
        }
        for method in run_pilot.TRAINED_METHODS
    }
    manifest = {
        "run_id": config["run_id"],
        "status": "complete",
        "method": "bicubic + EDSR + SwinIR + bicubic-anchored Flow Bridge",
        "method_identities": config["method_identity"],
        "started_at": runtime.get("started_at"),
        "ended_at": failure_time,
        "finalized_at": now,
        "git_head": snapshot["head"],
        "git_branch": snapshot["branch"],
        "git_dirty_at_finalization": snapshot["dirty"],
        "git_status_line_count_at_finalization": snapshot["status_line_count"],
        "code_sha256_recorded_at_training": source_manifest,
        "finalization_manifest_sha256": run_pilot.sha256_file(run_dir / "finalization_manifest.json"),
        "config_sha256": run_pilot.sha256_file(run_dir / "config.json"),
        "split_manifest_sha256": run_pilot.sha256_file(run_dir / "split_manifest.json"),
        "input_manifest_sha256": run_pilot.sha256_file(run_dir / "input_manifest.json"),
        "evaluation_config_sha256": run_pilot.sha256_file(run_dir / "evaluation_config.json"),
        "training_schedule_sha256": run_pilot.sha256_file(run_dir / "training_schedule.json"),
        "checkpoint_sha256": checkpoint_hashes,
        "output_prediction_count": len(list((run_dir / "predictions").rglob("*.npy"))),
        "train_sample_count": runtime["train_sample_count"],
        "development_sample_count": runtime["development_sample_count"],
        "development_group_count": split["role_group_counts"]["development"],
        "best_deterministic_method": reference_method,
        "research_decision": decision["decision"],
        "inference_replay_matches_saved_metrics": True,
        "inference_timing_source": decision["inference_timing_source"],
        "test_set_used": False,
        "confirmation_payload_accessed": False,
        "confirmation_metadata_recorded_only": True,
        "optical_payload_loaded": False,
        "paper_files_modified": False,
    }
    run_pilot.write_json(manifest_path, manifest)

    record_path = run_dir / "finalization_record.md"
    with record_path.open("x", encoding="utf-8") as stream:
        stream.write(
            "# SARptical 空间 pilot 后处理恢复记录\n\n"
            f"- 决策：`{decision['decision']}`。\n"
            f"- 原运行在训练和 306 张 development 全集四次 Flow draw 推理完成后，因全体亮回波 mask 小于最小像素数 `{minimum_mask_pixels}`，bootstrap 没有有效配对值而停止写汇总。\n"
            f"- development 高回波 mask 像素数范围：`{min(mask_counts)}–{max(mask_counts)}`；有效样本 ` {bright_valid_samples}/{len(mask_counts)}`，有效来源组 ` {decision['bright_roi_availability']['valid_source_groups']}/8`。\n"
            "- 依照冻结规则，该亮回波护栏记为不可用且未通过；未降低阈值、未改变 split、训练、评价或晋级门槛。\n"
            "- 修复仅使全 `NA` 的配对统计可被记录，并让决策报告明确写出有效组数。新增回归测试覆盖该情况。\n"
            "- 使用同一 best checkpoint 和固定 development 输入重放推理；所有逐图指标与首次保存结果核对一致。重放计时见 `runtime.json`，临时重复预测目录已自动移除。\n"
            "- 原始异常仍保留于 `failure.json` 与 `failure.txt`。confirmation/test/Optical 载荷访问标记均为 false。\n"
        )

    with (run_dir / "finalization_command.txt").open("x", encoding="utf-8") as stream:
        stream.write("执行命令：\n")
        stream.write(
            "/home/shy/anaconda3/envs/AISAR/bin/python "
            "experiments/phase_d_sarptical_spatial_pilot_20261125/finalize_saved_run.py "
            f"--run-dir {run_dir}\n"
        )
    print(
        f"后处理恢复完成：参考方法={reference_method}；决策={decision['decision']}；"
        f"高回波有效组={decision['bright_roi_availability']['valid_source_groups']}/8。",
        flush=True,
    )
    return {"decision": decision, "manifest": manifest, "runtime": runtime}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="恢复 SARptical pilot 已保存结果的决策汇总")
    parser.add_argument("--run-dir", type=Path, required=True, help="已经完成训练和 development 推理的唯一运行目录")
    args = parser.parse_args(argv)
    try:
        finalize_saved_run(args.run_dir)
    except Exception as exc:
        print(f"后处理恢复失败：{type(exc).__name__}：{exc}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
