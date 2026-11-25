"""只读统计固定 train/development SAR LR 中的精确零值。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED = {"train": (1600, 92), "development": (200, 19)}
BOOTSTRAP_SEED = 20261124
BOOTSTRAP_REPLICATES = 10_000


def summarize_array(value: np.ndarray) -> dict[str, float | int | None]:
    array = np.asarray(value)
    if array.dtype != np.float32 or array.ndim != 3 or array.shape[0] != 1:
        raise ValueError("LR 必须为 float32 [1,H,W]")
    if not np.isfinite(array).all():
        raise ValueError("LR 必须全部有限")
    if np.any(array < 0.0):
        raise ValueError("LR 必须非负")
    zeros = array == 0.0
    positive = array[array > 0.0]
    return {
        "pixel_count": int(array.size),
        "zero_pixel_count": int(zeros.sum()),
        "zero_fraction": float(zeros.mean()),
        "positive_pixel_count": int(positive.size),
        "minimum_positive_value": float(positive.min()) if positive.size else None,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_role(input_root: Path, role: str) -> tuple[list[dict], dict]:
    expected_count, expected_groups = EXPECTED[role]
    root = input_root / role
    manifest_path = root / "manifest.json"
    files_path = root / "files.csv"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("format") != "c2_fixed_inputs_v1"
        or manifest.get("status") != "complete"
        or manifest.get("role") != role
        or manifest.get("condition_id") != "x2_light"
        or manifest.get("sample_count") != expected_count
        or manifest.get("test_set_accessed") is not False
        or manifest.get("test_set_used") is not False
        or manifest.get("confirmation_payload_accessed") is not False
    ):
        raise ValueError(f"{role} 输入 manifest 不符合固定范围")
    files_hash = sha256_file(files_path)
    if files_hash != manifest.get("files_csv_sha256"):
        raise ValueError(f"{role} files.csv 哈希不匹配")
    with files_path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"sample_id", "source_group_id", "lr_sha256"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"{role} files.csv 缺少必要字段")
        rows = [
            {name: row[name] for name in required}
            for row in reader
        ]
    rows.sort(key=lambda row: row["sample_id"])
    sample_ids = [row["sample_id"] for row in rows]
    group_ids = {row["source_group_id"] for row in rows}
    if (
        len(rows) != expected_count
        or len(sample_ids) != len(set(sample_ids))
        or len(group_ids) != expected_groups
        or "" in group_ids
    ):
        raise ValueError(f"{role} 样本或来源组计数不符")
    actual_files = {path.stem for path in (root / "lr").glob("*.npy")}
    if actual_files != set(sample_ids):
        raise ValueError(f"{role} LR 文件集合与固定清单不一致")
    metadata = {
        "role": role,
        "sample_count": len(rows),
        "source_group_count": len(group_ids),
        "manifest_sha256": sha256_file(manifest_path),
        "files_csv_sha256": files_hash,
    }
    return rows, metadata


def bootstrap_group_mean(group_values: dict[str, float]) -> list[float]:
    values = np.asarray(list(group_values.values()), dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, len(values), size=(BOOTSTRAP_REPLICATES, len(values)))
    means = values[draws].mean(axis=1)
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def run(input_root: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖：{output_dir}")
    role_rows, split_metadata = {}, {}
    for role in EXPECTED:
        role_rows[role], split_metadata[role] = load_role(input_root, role)
    train_groups = {row["source_group_id"] for row in role_rows["train"]}
    dev_groups = {row["source_group_id"] for row in role_rows["development"]}
    if train_groups & dev_groups:
        raise ValueError("train/development 来源组重叠")

    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    per_image = []
    group_counts: dict[tuple[str, str], list[int]] = {}
    for role, records in role_rows.items():
        for record in records:
            path = input_root / role / "lr" / f"{record['sample_id']}.npy"
            if sha256_file(path) != record["lr_sha256"]:
                raise ValueError(f"LR 文件哈希不匹配：{role}/{path.name}")
            array = np.load(path, allow_pickle=False)
            stats = summarize_array(array)
            per_image.append(
                {
                    "role": role,
                    "sample_id": record["sample_id"],
                    "source_group_id": record["source_group_id"],
                    **stats,
                }
            )
            key = (role, record["source_group_id"])
            counts = group_counts.setdefault(key, [0, 0])
            counts[0] += stats["zero_pixel_count"]
            counts[1] += stats["pixel_count"]

    group_rows = [
        {
            "role": role,
            "source_group_id": group,
            "zero_pixel_count": counts[0],
            "pixel_count": counts[1],
            "zero_fraction": counts[0] / counts[1],
        }
        for (role, group), counts in sorted(group_counts.items())
    ]
    summaries = {}
    for role in EXPECTED:
        images = [row for row in per_image if row["role"] == role]
        groups = [row for row in group_rows if row["role"] == role]
        zero_count = sum(row["zero_pixel_count"] for row in images)
        pixel_count = sum(row["pixel_count"] for row in images)
        image_fractions = np.asarray([row["zero_fraction"] for row in images], dtype=np.float64)
        group_fractions = {row["source_group_id"]: row["zero_fraction"] for row in groups}
        positive_mins = [row["minimum_positive_value"] for row in images if row["minimum_positive_value"] is not None]
        summaries[role] = {
            **split_metadata[role],
            "zero_pixel_count": zero_count,
            "pixel_count": pixel_count,
            "pooled_zero_fraction": zero_count / pixel_count,
            "images_with_any_zero_count": sum(row["zero_pixel_count"] > 0 for row in images),
            "image_zero_fraction_median": float(np.median(image_fractions)),
            "image_zero_fraction_p10": float(np.quantile(image_fractions, 0.10)),
            "image_zero_fraction_p90": float(np.quantile(image_fractions, 0.90)),
            "image_zero_fraction_max": float(image_fractions.max()),
            "source_group_balanced_zero_fraction": float(np.mean(list(group_fractions.values()))),
            "source_group_bootstrap_95_ci": bootstrap_group_mean(group_fractions),
            "source_group_bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "source_group_bootstrap_seed": BOOTSTRAP_SEED,
            "minimum_positive_lr_value": min(positive_mins) if positive_mins else None,
        }

    for name, rows in (("per_image.csv", per_image), ("by_source_group.csv", group_rows)):
        with (output_dir / name).open("x", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    result = {
        "run_id": "phase_d2_zero_support_20261124_v1",
        "status": "complete",
        "input_root": str(input_root),
        "output_dir": str(output_dir),
        "splits": split_metadata,
        "summary": summaries,
        "test_set_accessed": False,
        "test_set_used": False,
        "confirmation_payload_accessed": False,
        "optical_payload_loaded": False,
        "training_performed": False,
        "cuda_used": False,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "python_executable": sys.executable,
        "analysis_code_sha256": sha256_file(Path(__file__).resolve()),
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "执行命令.txt").write_text(
        " ".join([sys.executable, *sys.argv]) + "\n", encoding="utf-8"
    )
    print(
        "精确零值统计完成："
        + "；".join(
            f"{role}={summaries[role]['pooled_zero_fraction']:.6%}（组等权 {summaries[role]['source_group_balanced_zero_fraction']:.6%}）"
            for role in EXPECTED
        ),
        flush=True,
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments/diagnostics/phase_d2_zero_support_20261124_v1",
    )
    args = parser.parse_args()
    run(args.input_root, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
