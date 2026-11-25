"""对已保存的 SAR-only Residual DDPM 结果执行只读失败机理诊断。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_INPUT_ROOT = Path(
    "/tmp/jdsr-c3a-20261123/data/derived/c2_fixed_inputs_c1_v2_20261031_v1"
)
DEFAULT_RUN_DIR = (
    PROJECT_ROOT
    / "experiments/published_baselines_sar_only_20261123_v1"
    / "运行_sar_only_diffusion_20261123_v1"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments/diagnostics/phase_d0_residual_failure_20261124_v1"
)
EXPECTED = {
    "train": {"samples": 1600, "groups": 92},
    "development": {"samples": 200, "groups": 19},
}


def _finite_array(value, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim not in (1, 2) or not np.isfinite(array).all():
        raise ValueError(f"{label} 必须是有限的一维或二维数组")
    return array


def residual_alignment_metrics(
    target_residual: np.ndarray,
    predicted_residual: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict[str, float]:
    """计算残差方向、点估计误差和目标辅助的无约束 oracle。"""
    target = _finite_array(target_residual, "目标残差")
    predicted = _finite_array(predicted_residual, "预测残差")
    if target.shape != predicted.shape:
        raise ValueError("残差 shape 必须一致")
    if mask is None:
        selected_target = target.reshape(-1)
        selected_predicted = predicted.reshape(-1)
    else:
        selected_mask = np.asarray(mask, dtype=bool)
        if selected_mask.shape != target.shape or not selected_mask.any():
            raise ValueError("mask shape 必须匹配残差且至少包含一个像素")
        selected_target = target[selected_mask]
        selected_predicted = predicted[selected_mask]

    dot = float(np.dot(selected_target, selected_predicted))
    target_norm = float(np.linalg.norm(selected_target))
    predicted_norm = float(np.linalg.norm(selected_predicted))
    cosine = dot / (target_norm * predicted_norm) if target_norm and predicted_norm else math.nan
    target_centered = selected_target - selected_target.mean()
    predicted_centered = selected_predicted - selected_predicted.mean()
    centered_denominator = float(np.linalg.norm(target_centered) * np.linalg.norm(predicted_centered))
    pearson = (
        float(np.dot(target_centered, predicted_centered) / centered_denominator)
        if centered_denominator
        else math.nan
    )
    baseline_mse = float(np.mean(selected_target**2))
    prediction_mse = float(np.mean((selected_predicted - selected_target) ** 2))
    predicted_square_sum = float(np.dot(selected_predicted, selected_predicted))
    oracle_alpha = dot / (predicted_square_sum + np.finfo(np.float64).eps)
    oracle_mse = float(np.mean((oracle_alpha * selected_predicted - selected_target) ** 2))
    return {
        "cosine": float(cosine),
        "pearson": float(pearson),
        "dot_product": dot,
        "target_residual_rms": float(np.sqrt(baseline_mse)),
        "predicted_residual_rms": float(np.sqrt(np.mean(selected_predicted**2))),
        "residual_rms_ratio": float(
            np.sqrt(np.mean(selected_predicted**2))
            / (np.sqrt(baseline_mse) + np.finfo(np.float64).eps)
        ),
        "sign_agreement": float(np.mean(np.sign(selected_target) == np.sign(selected_predicted))),
        "baseline_mse": baseline_mse,
        "prediction_mse": prediction_mse,
        "delta_mse_vs_base": prediction_mse - baseline_mse,
        "oracle_alpha": float(oracle_alpha),
        "oracle_mse": oracle_mse,
    }


def frequency_energy_fractions(residual: np.ndarray) -> dict[str, float]:
    """以固定径向频带返回残差频谱能量比例。"""
    image = _finite_array(residual, "残差频谱输入")
    if image.ndim != 2:
        raise ValueError("频谱输入必须是二维图像")
    height, width = image.shape
    fy = np.fft.fftfreq(height)[:, None]
    fx = np.fft.fftfreq(width)[None, :]
    radial = np.sqrt(fy**2 + fx**2)
    power = np.abs(np.fft.fft2(image)) ** 2
    total = float(power.sum())
    if total <= np.finfo(np.float64).eps:
        return {"low_fraction": math.nan, "mid_fraction": math.nan, "high_fraction": math.nan}
    return {
        "low_fraction": float(power[radial < 0.125].sum() / total),
        "mid_fraction": float(power[(radial >= 0.125) & (radial < 0.25)].sum() / total),
        "high_fraction": float(power[radial >= 0.25].sum() / total),
    }


def sample_variability_metrics(draws: np.ndarray) -> dict[str, float]:
    """汇总固定 draw 集合的像素标准差和 draw 两两差异。"""
    images = np.asarray(draws, dtype=np.float64)
    if images.ndim == 4 and images.shape[1] == 1:
        images = images[:, 0]
    if images.ndim != 3 or images.shape[0] < 2 or not np.isfinite(images).all():
        raise ValueError("draws 必须为至少两张有限二维图像")
    pixel_std = images.std(axis=0, ddof=0)
    pairwise = [
        float(np.mean(np.abs(images[left] - images[right])))
        for left in range(len(images))
        for right in range(left + 1, len(images))
    ]
    return {
        "draw_count": int(images.shape[0]),
        "pixel_std_mean": float(pixel_std.mean()),
        "pixel_std_rms": float(np.sqrt(np.mean(pixel_std**2))),
        "draw_pair_mean_absolute_difference": float(np.mean(pairwise)),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_role_manifest(data_root: Path, role: str) -> tuple[list[dict[str, str]], dict[str, object]]:
    if role not in EXPECTED:
        raise ValueError("仅允许读取 train 与 development")
    split_root = data_root / role
    manifest_path = split_root / "manifest.json"
    files_path = split_root / "files.csv"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = EXPECTED[role]
    if (
        manifest.get("format") != "c2_fixed_inputs_v1"
        or manifest.get("status") != "complete"
        or manifest.get("role") != role
        or manifest.get("scale") != 2
        or manifest.get("sample_count") != expected["samples"]
        or manifest.get("test_set_accessed") is not False
        or manifest.get("test_set_used") is not False
        or manifest.get("confirmation_payload_accessed") is not False
    ):
        raise ValueError(f"{role} 固定输入 manifest 不符合允许范围")
    files_hash = _sha256(files_path)
    if files_hash != manifest.get("files_csv_sha256"):
        raise ValueError(f"{role} files.csv 哈希不匹配")

    required = ("sample_id", "source_group_id", "patch_id", "lr_sha256", "target_sha256")
    rows = []
    with files_path.open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        columns = {name: header.index(name) for name in required}
        for source_row in reader:
            rows.append({name: source_row[index] for name, index in columns.items()})
    rows.sort(key=lambda row: row["sample_id"])
    sample_ids = [row["sample_id"] for row in rows]
    groups = {row["source_group_id"] for row in rows}
    if (
        len(sample_ids) != expected["samples"]
        or len(sample_ids) != len(set(sample_ids))
        or len(groups) != expected["groups"]
        or "" in groups
    ):
        raise ValueError(f"{role} 样本或来源组计数与冻结划分不符")
    for modality in ("lr", "target"):
        directory = split_root / modality
        actual_ids = {path.stem for path in directory.glob("*.npy")}
        if actual_ids != set(sample_ids):
            raise ValueError(f"{role}/{modality} 文件集合与 SAR 清单不一致")
    return rows, {
        "role": role,
        "sample_count": len(rows),
        "source_group_count": len(groups),
        "manifest_sha256": _sha256(manifest_path),
        "files_csv_sha256": files_hash,
        "test_set_accessed": False,
        "test_set_used": False,
        "confirmation_payload_accessed": False,
        "optical_payload_loaded": False,
    }


def _load_sar(
    path: Path, expected_hash: str, label: str, *, unit_range: bool
) -> np.ndarray:
    if _sha256(path) != expected_hash:
        raise ValueError(f"{label} 输入哈希不匹配：{path.name}")
    value = np.load(path, allow_pickle=False)
    if value.dtype != np.float32 or value.ndim != 3 or value.shape[0] != 1:
        raise ValueError(f"{label} 必须是 float32 [1,H,W]：{path.name}")
    if not np.isfinite(value).all():
        raise ValueError(f"{label} 数值必须有限：{path.name}")
    if unit_range and (float(value.min()) < 0.0 or float(value.max()) > 1.0):
        raise ValueError(f"{label} 数值必须位于 [0,1]：{path.name}")
    return value[0].astype(np.float64, copy=False)


def _load_prediction_image(path: Path, expected_shape: tuple[int, int], label: str) -> np.ndarray:
    value = np.load(path, allow_pickle=False)
    if value.dtype != np.float32 or value.shape != expected_shape:
        raise ValueError(f"{label} prediction 必须为 float32 {expected_shape}：{path.name}")
    if not np.isfinite(value).all() or float(value.min()) < 0.0 or float(value.max()) > 1.0:
        raise ValueError(f"{label} prediction 必须有限且位于 [0,1]：{path.name}")
    return value.astype(np.float64, copy=False)


def _load_draw_predictions(path: Path, expected_shape: tuple[int, int]) -> np.ndarray:
    value = np.load(path, allow_pickle=False)
    if value.dtype != np.float32 or value.shape != (4, *expected_shape):
        raise ValueError(f"Residual DDPM 必须保存四张 float32 二维 draw：{path.name}")
    if not np.isfinite(value).all() or float(value.min()) < 0.0 or float(value.max()) > 1.0:
        raise ValueError(f"Residual DDPM draw 必须有限且位于 [0,1]：{path.name}")
    return value.astype(np.float64, copy=False)


def _check_prediction_files(directory: Path, sample_ids: set[str], label: str) -> None:
    actual_ids = {path.stem for path in directory.glob("*.npy")}
    if actual_ids != sample_ids:
        raise ValueError(f"{label} 文件集合与 SAR 样本身份不一致")


def _append_residual_distribution(
    rows: list[dict[str, object]], split: str, record: dict[str, str], target: np.ndarray, base: np.ndarray, masks: dict[str, np.ndarray]
) -> None:
    residual = target - base
    row: dict[str, object] = {
        "split": split,
        "sample_id": record["sample_id"],
        "source_group_id": record["source_group_id"],
        "patch_id": record["patch_id"],
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "residual_rms": float(np.sqrt(np.mean(residual**2))),
        "residual_abs_median": float(np.median(np.abs(residual))),
        "residual_abs_p90": float(np.percentile(np.abs(residual), 90.0)),
        "residual_abs_p95": float(np.percentile(np.abs(residual), 95.0)),
        "residual_abs_p99": float(np.percentile(np.abs(residual), 99.0)),
    }
    row.update({f"residual_{name}": value for name, value in frequency_energy_fractions(residual).items()})
    for name, mask in masks.items():
        values = residual[mask]
        row[f"{name}_residual_rms"] = float(np.sqrt(np.mean(values**2)))
        row[f"{name}_residual_mean"] = float(values.mean())
    rows.append(row)


def _append_direction_rows(
    rows: list[dict[str, object]],
    record: dict[str, str],
    target: np.ndarray,
    base: np.ndarray,
    draws: np.ndarray,
    masks: dict[str, np.ndarray],
) -> None:
    target_residual = target - base
    single_predictions = draws[:, 0] if draws.ndim == 4 else draws
    units: list[tuple[str, int | str, np.ndarray]] = [
        ("single_draw", index, image) for index, image in enumerate(single_predictions)
    ]
    units.append(("pixel_mean", "mean", single_predictions.mean(axis=0)))
    variability = sample_variability_metrics(single_predictions)
    for unit, draw_index, prediction in units:
        predicted_residual = prediction - base
        row: dict[str, object] = {
            "sample_id": record["sample_id"],
            "source_group_id": record["source_group_id"],
            "unit": unit,
            "draw_index": draw_index,
        }
        row.update(residual_alignment_metrics(target_residual, predicted_residual))
        row.update({f"predicted_{name}": value for name, value in frequency_energy_fractions(predicted_residual).items()})
        for name, mask in masks.items():
            local = residual_alignment_metrics(target_residual, predicted_residual, mask)
            for metric in ("cosine", "prediction_mse", "baseline_mse", "delta_mse_vs_base", "oracle_alpha"):
                row[f"{name}_{metric}"] = local[metric]
        if unit == "pixel_mean":
            row.update(variability)
            row["single_draw_mean_psnr_db"] = float(
                np.mean([10.0 * np.log10(1.0 / (np.mean((image - target) ** 2) + 1e-12)) for image in single_predictions])
            )
            row["pixel_mean_psnr_db"] = float(
                10.0 * np.log10(1.0 / (np.mean((prediction - target) ** 2) + 1e-12))
            )
        rows.append(row)


def _group_balanced(rows: list[dict[str, object]], grouping: tuple[str, ...]) -> list[dict[str, object]]:
    groups: dict[tuple[str, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row[key]) for key in grouping)].append(row)
    output = []
    for keys, group_rows in sorted(groups.items()):
        summary: dict[str, object] = {key: value for key, value in zip(grouping, keys)}
        summary["sample_count"] = len({row.get("sample_id") for row in group_rows})
        summary["row_count"] = len(group_rows)
        numeric = sorted(
            key
            for key in {name for row in group_rows for name in row}
            if key not in {"sample_id", "source_group_id", "split", "unit", "draw_index"}
            and all(isinstance(row.get(key), (int, float, np.integer, np.floating)) for row in group_rows if key in row)
        )
        for metric in numeric:
            values = [float(row[metric]) for row in group_rows if metric in row and math.isfinite(float(row[metric]))]
            if values:
                summary[f"mean_{metric}"] = float(np.mean(values))
        output.append(summary)
    return output


def _bootstrap_group_means(
    rows: list[dict[str, object]], metric: str, seed: int, replicates: int
) -> dict[str, object]:
    by_group: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = float(row[metric])
        if math.isfinite(value):
            by_group[str(row["source_group_id"])].append(value)
    group_values = np.asarray([np.mean(values) for _, values in sorted(by_group.items())], dtype=np.float64)
    if len(group_values) < 2:
        raise ValueError("来源组 bootstrap 至少需要两个有效组")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(group_values), size=(replicates, len(group_values)))
    estimates = group_values[indices].mean(axis=1)
    return {
        "metric": metric,
        "group_count": int(len(group_values)),
        "group_balanced_mean": float(group_values.mean()),
        "bootstrap_95_ci": [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))],
        "replicates": int(replicates),
        "seed": int(seed),
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"拒绝写入空结果表：{path.name}")
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def run_analysis(input_root: Path, run_dir: Path, output_dir: Path) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖：{output_dir}")
    data_root = input_root / "x2_light" if (input_root / "x2_light").is_dir() else input_root
    if not data_root.is_dir():
        raise FileNotFoundError(f"固定 train/development SAR 输入不存在：{data_root}")
    run_manifest_path = run_dir / "manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    if (
        run_manifest.get("status") != "complete"
        or run_manifest.get("test_set_accessed") is not False
        or run_manifest.get("test_set_used") is not False
        or run_manifest.get("confirmation_payload_accessed") is not False
        or run_manifest.get("optical_payload_loaded") is not False
    ):
        raise ValueError("Residual DDPM run manifest 未满足 train/development SAR-only 边界")

    role_rows: dict[str, list[dict[str, str]]] = {}
    split_summaries = {}
    for role, expected in EXPECTED.items():
        rows, summary = _read_role_manifest(data_root, role)
        recorded = run_manifest.get("inputs", {}).get(role, {})
        for key in ("sample_count", "source_group_count", "manifest_sha256", "files_csv_sha256"):
            if recorded.get(key) != summary.get(key):
                raise ValueError(f"run manifest 与固定 {role} 输入记录不一致：{key}")
        if summary["sample_count"] != expected["samples"] or summary["source_group_count"] != expected["groups"]:
            raise ValueError(f"{role} 固定 split 计数异常")
        role_rows[role] = rows
        split_summaries[role] = summary

    for role, rows in role_rows.items():
        ids = {row["sample_id"] for row in rows}
        _check_prediction_files(run_dir / "swinir_base_predictions" / role, ids, f"SwinIR {role} 预测")
    development_ids = {row["sample_id"] for row in role_rows["development"]}
    residual_prediction_dir = run_dir / "predictions" / "residual"
    _check_prediction_files(residual_prediction_dir, development_ids, "Residual DDPM development 预测")

    from MMCANetV2.phase_c3a_metrics import build_fixed_masks

    residual_rows: list[dict[str, object]] = []
    direction_rows: list[dict[str, object]] = []
    started = time.perf_counter()
    for role, rows in role_rows.items():
        split_root = data_root / role
        for index, record in enumerate(rows, start=1):
            sample_id = record["sample_id"]
            lr = _load_sar(
                split_root / "lr" / f"{sample_id}.npy",
                record["lr_sha256"],
                "LR",
                unit_range=False,
            )
            target = _load_sar(
                split_root / "target" / f"{sample_id}.npy",
                record["target_sha256"],
                "target",
                unit_range=True,
            )
            base_path = run_dir / "swinir_base_predictions" / role / f"{sample_id}.npy"
            base = _load_prediction_image(base_path, target.shape, f"SwinIR {role}")
            if target.shape != base.shape or target.shape != (lr.shape[0] * 2, lr.shape[1] * 2):
                raise ValueError(f"SAR LR、target、SwinIR prediction 尺寸不匹配：{sample_id}")
            masks, _ = build_fixed_masks(lr, target, 2)
            _append_residual_distribution(residual_rows, role, record, target, base, masks)

            if role == "development":
                prediction = _load_draw_predictions(
                    residual_prediction_dir / f"{sample_id}.npy", target.shape
                )
                _append_direction_rows(direction_rows, record, target, base, prediction, masks)
            if index % 200 == 0:
                print(f"已完成 {role} SAR 样本 {index}/{len(rows)}", flush=True)

    group_residual_rows = _group_balanced(residual_rows, ("split", "source_group_id"))
    group_direction_rows = _group_balanced(direction_rows, ("unit", "source_group_id"))
    bootstrap = {}
    for unit in ("single_draw", "pixel_mean"):
        selected = [row for row in direction_rows if row["unit"] == unit]
        if unit == "single_draw":
            # 每幅图先对四个 draw 的指标取均值，再以来源组为重采样单位。
            by_sample: dict[str, list[dict[str, object]]] = defaultdict(list)
            for row in selected:
                by_sample[str(row["sample_id"])].append(row)
            selected = []
            for sample_rows in by_sample.values():
                averaged = dict(sample_rows[0])
                for key in sample_rows[0]:
                    if isinstance(sample_rows[0][key], (int, float, np.integer, np.floating)):
                        values = [float(row[key]) for row in sample_rows]
                        finite_values = [value for value in values if math.isfinite(value)]
                        averaged[key] = float(np.mean(finite_values)) if finite_values else math.nan
                averaged["unit"] = "single_draw_mean_of_four"
                selected.append(averaged)
        for metric in ("cosine", "delta_mse_vs_base", "oracle_alpha"):
            bootstrap[f"{unit}_{metric}"] = _bootstrap_group_means(
                selected, metric, seed=20261124, replicates=10000
            )

    output_dir.mkdir(parents=True, exist_ok=False)
    _write_csv(output_dir / "train_development_residuals_per_image.csv", residual_rows)
    _write_csv(output_dir / "residual_direction_per_image.csv", direction_rows)
    _write_csv(output_dir / "residuals_by_source_group.csv", group_residual_rows)
    _write_csv(output_dir / "direction_by_source_group.csv", group_direction_rows)
    summary: dict[str, object] = {
        "run_id": "phase_d0_residual_failure_20261124_v1",
        "status": "complete",
        "scope": "只读 train/development SAR 残差诊断；不训练、不推理、不选择 checkpoint",
        "input_root": str(data_root),
        "source_run_dir": str(run_dir),
        "source_run_manifest_sha256": _sha256(run_manifest_path),
        "source_run_code_sha256": run_manifest.get("code_sha256"),
        "splits": split_summaries,
        "test_set_accessed": False,
        "test_set_used": False,
        "confirmation_payload_accessed": False,
        "optical_payload_loaded": False,
        "cuda_used": False,
        "bootstrap": bootstrap,
        "elapsed_seconds": time.perf_counter() - started,
    }
    with (output_dir / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    with (output_dir / "command.txt").open("x", encoding="utf-8") as stream:
        stream.write(" ".join(sys.argv) + "\n")
    with (output_dir / "environment.txt").open("x", encoding="utf-8") as stream:
        stream.write(f"Python: {platform.python_version()}\nNumPy: {np.__version__}\nCPU 后处理：是\nCUDA 使用：否\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="只读分析 Residual DDPM 失败机理")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    summary = run_analysis(args.input_root, args.run_dir, args.output_dir)
    print(f"D0 诊断完成：{summary['run_id']}；未访问 confirmation/test/optical，未使用 GPU。")
    print(f"结果目录：{args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
