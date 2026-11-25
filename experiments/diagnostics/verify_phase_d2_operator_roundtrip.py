"""用固定 train/development SAR 数组核对已登记的表示域退化算子。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEGRADATION_PATH = PROJECT_ROOT / "MMCANet" / "src"
if str(DEGRADATION_PATH) not in sys.path:
    sys.path.insert(0, str(DEGRADATION_PATH))
from degradation import build_sar_lr_from_physical_degradation  # noqa: E402

DEFAULT_INPUT_ROOT = PROJECT_ROOT / "data/derived/c2_fixed_inputs_c1_v2_20261031_v1/x2_light"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments/diagnostics/phase_d2_operator_roundtrip_20261124_v1"
EXPECTED = {"train": (1600, 92), "development": (200, 19)}
OPERATOR_HASH = "b7c0e9208a132792824bdde4296d7d3650587a144bf7ab4ec34bb161e0cb7bd1"
MAX_ABS_TOLERANCE = 1e-7


def error_summary(stored: np.ndarray, regenerated: np.ndarray) -> dict[str, float | bool]:
    left = np.asarray(stored)
    right = np.asarray(regenerated)
    if left.shape != right.shape:
        raise ValueError("算子输出与固定 LR 的 shape 不一致")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("算子往返值必须有限")
    difference = right.astype(np.float64) - left.astype(np.float64)
    return {
        "bitwise_equal": bool(np.array_equal(left, right)),
        "max_abs_error": float(np.max(np.abs(difference))),
        "mae": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(difference**2))),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_role(input_root: Path, role: str) -> tuple[list[dict[str, str]], dict[str, object]]:
    if role not in EXPECTED:
        raise ValueError("仅允许读取 train 与 development")
    root = input_root / role
    manifest_path = root / "manifest.json"
    files_path = root / "files.csv"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_samples, expected_groups = EXPECTED[role]
    if (
        manifest.get("format") != "c2_fixed_inputs_v1"
        or manifest.get("status") != "complete"
        or manifest.get("role") != role
        or manifest.get("condition_id") != "x2_light"
        or manifest.get("sample_count") != expected_samples
        or manifest.get("test_set_accessed") is not False
        or manifest.get("test_set_used") is not False
        or manifest.get("confirmation_payload_accessed") is not False
    ):
        raise ValueError(f"{role} manifest 不符合固定输入边界")
    config = manifest["degradation"]
    expected_config = {
        "scale": 2,
        "input_size": 128,
        "target_size": 256,
        "fft_window": 129,
        "blur": False,
        "downsample": "bilinear align_corners=false",
        "negative_policy": "clamp",
        "multiplier": "Gamma(shape=L,rate=L) in normalized representation domain",
        "L": 8.0,
        "upper_clamp": False,
    }
    if any(config.get(key) != value for key, value in expected_config.items()):
        raise ValueError(f"{role} 退化参数与固定核验方案不一致")
    dependency_hash = _sha256(DEGRADATION_PATH / "degradation.py")
    recorded_dependency_hash = manifest.get("dependency_sha256", {}).get(
        "MMCANet/src/degradation.py"
    )
    if dependency_hash != OPERATOR_HASH or recorded_dependency_hash != dependency_hash:
        raise ValueError("当前 degradation.py 与固定输入 manifest 中的算子哈希不一致")
    files_hash = _sha256(files_path)
    if files_hash != manifest.get("files_csv_sha256"):
        raise ValueError(f"{role} files.csv 哈希不匹配")
    required = ("sample_id", "source_group_id", "patch_id", "sample_seed", "lr_sha256", "target_sha256")
    with files_path.open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        columns = {name: header.index(name) for name in required}
        rows = [
            {name: values[index] for name, index in columns.items()}
            for values in reader
        ]
    rows.sort(key=lambda row: row["sample_id"])
    identities = [row["sample_id"] for row in rows]
    groups = {row["source_group_id"] for row in rows}
    if (
        len(rows) != expected_samples
        or len(identities) != len(set(identities))
        or len(groups) != expected_groups
        or "" in groups
    ):
        raise ValueError(f"{role} 样本身份或来源组计数错误")
    for modality in ("lr", "target"):
        actual = {path.stem for path in (root / modality).glob("*.npy")}
        if actual != set(identities):
            raise ValueError(f"{role}/{modality} 文件集合与 SAR 清单不一致")
    return rows, {
        "role": role,
        "sample_count": len(rows),
        "source_group_count": len(groups),
        "manifest_sha256": _sha256(manifest_path),
        "files_csv_sha256": files_hash,
        "generator_sha256_recorded": manifest.get("generator_sha256"),
        "degradation_dependency_sha256_recorded": manifest.get("dependency_sha256", {}).get(
            "MMCANet/src/degradation.py"
        ),
    }


def _read_array(path: Path, expected_hash: str, label: str, *, target: bool) -> np.ndarray:
    if _sha256(path) != expected_hash:
        raise ValueError(f"{label} 文件哈希不匹配：{path.name}")
    value = np.load(path, allow_pickle=False)
    if value.dtype != np.float32 or value.ndim != 3 or value.shape[0] != 1:
        raise ValueError(f"{label} 必须为 float32 [1,H,W]：{path.name}")
    if not np.isfinite(value).all():
        raise ValueError(f"{label} 含非有限值：{path.name}")
    if target and (float(value.min()) < 0.0 or float(value.max()) > 1.0):
        raise ValueError(f"target 超出 [0,1]：{path.name}")
    return value


def run(input_root: Path, output_dir: Path) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖：{output_dir}")
    role_rows: dict[str, list[dict[str, str]]] = {}
    split_metadata = {}
    for role in EXPECTED:
        role_rows[role], split_metadata[role] = _read_role(input_root, role)
    train_groups = {row["source_group_id"] for row in role_rows["train"]}
    dev_groups = {row["source_group_id"] for row in role_rows["development"]}
    if train_groups & dev_groups:
        raise ValueError("train/development 来源组重叠")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 当前不可用；按冻结方案不改用 CPU")
    device = torch.device("cuda:0")
    torch.use_deterministic_algorithms(True)
    output_dir.mkdir(parents=True, exist_ok=False)
    initial_free, total_memory = torch.cuda.mem_get_info(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    manifest: dict[str, object] = {
        "run_id": "phase_d2_operator_roundtrip_20261124_v1",
        "status": "running",
        "input_root": str(input_root),
        "output_dir": str(output_dir),
        "splits": split_metadata,
        "operator_sha256": _sha256(DEGRADATION_PATH / "degradation.py"),
        "analysis_code_sha256": _sha256(Path(__file__).resolve()),
        "max_abs_tolerance": MAX_ABS_TOLERANCE,
        "test_set_accessed": False,
        "test_set_used": False,
        "confirmation_payload_accessed": False,
        "optical_payload_loaded": False,
        "training_performed": False,
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "memory_total_bytes": int(total_memory),
        "memory_free_before_bytes": int(initial_free),
        "peak_allocated_bytes": None,
        "peak_reserved_bytes": None,
        "cuda_oom": False,
        "command": [sys.executable, *sys.argv],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    csv_path = output_dir / "per_sample.csv"
    fieldnames = (
        "role", "sample_id", "source_group_id", "patch_id", "sample_seed",
        "bitwise_equal", "max_abs_error", "mae", "rmse", "within_tolerance",
    )
    rows_completed = 0
    errors = []
    try:
        with csv_path.open("x", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            for role, records in role_rows.items():
                split_root = input_root / role
                for record in records:
                    sample_id = record["sample_id"]
                    target = _read_array(
                        split_root / "target" / f"{sample_id}.npy",
                        record["target_sha256"],
                        "target",
                        target=True,
                    )
                    stored_lr = _read_array(
                        split_root / "lr" / f"{sample_id}.npy",
                        record["lr_sha256"],
                        "LR",
                        target=False,
                    )
                    seed = int(record["sample_seed"])
                    target_tensor = torch.from_numpy(target).to(device)
                    with torch.random.fork_rng(devices=[device.index]):
                        torch.manual_seed(seed)
                        regenerated_tensor = build_sar_lr_from_physical_degradation(
                            target_tensor,
                            input_downsample_size=128,
                            downsample_factor=2,
                            use_freq_truncation=True,
                            freq_window_size=129,
                            use_anisotropic_blur=False,
                            use_speckle=True,
                            speckle_looks=8.0,
                            speckle_domain="representation",
                            negative_policy="clamp",
                            clamp_output=False,
                        )
                    regenerated = regenerated_tensor.cpu().numpy()
                    metrics = error_summary(stored_lr, regenerated)
                    row = {
                        "role": role,
                        "sample_id": sample_id,
                        "source_group_id": record["source_group_id"],
                        "patch_id": record["patch_id"],
                        "sample_seed": seed,
                        **metrics,
                        "within_tolerance": metrics["max_abs_error"] <= MAX_ABS_TOLERANCE,
                    }
                    writer.writerow(row)
                    rows_completed += 1
                    if not row["within_tolerance"]:
                        errors.append(row)
                    if rows_completed % 50 == 0:
                        stream.flush()
                        print(f"已完成 {rows_completed}/1800 张 train/development SAR 核验", flush=True)
        torch.cuda.synchronize(device)
        final_free, _ = torch.cuda.mem_get_info(device)
        manifest.update(
            status="complete" if not errors and rows_completed == 1800 else "not_reproduced",
            rows_completed=rows_completed,
            failed_sample_count=len(errors),
            max_abs_error=max(float(row["max_abs_error"]) for row in rows_from_csv(csv_path)),
            mean_absolute_error=float(np.mean([float(row["mae"]) for row in rows_from_csv(csv_path)])),
            memory_free_after_bytes=int(final_free),
            peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
            peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device)),
            cuda_oom=False,
            elapsed_seconds=time.perf_counter() - started,
        )
    except torch.cuda.OutOfMemoryError as exc:
        manifest.update(
            status="cuda_oom",
            rows_completed=rows_completed,
            cuda_oom=True,
            error=str(exc),
            traceback=traceback.format_exc(),
            peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
            peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device)),
            elapsed_seconds=time.perf_counter() - started,
        )
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise
    except Exception as exc:
        manifest.update(
            status="failed",
            rows_completed=rows_completed,
            error=str(exc),
            traceback=traceback.format_exc(),
            cuda_oom=False,
            elapsed_seconds=time.perf_counter() - started,
        )
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise

    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "decision.txt").write_text(
        "判据：所有 train/development SAR LR 均由固定 target、记录 sample_seed 和记录退化算子复现，且最大绝对误差不超过 1e-7。\n"
        f"结果：{'通过' if manifest['status'] == 'complete' else '未复现'}；超差样本 {manifest['failed_sample_count']}。\n"
        "边界：仅验证 C1 v2 x2_light 归一化表示域退化链，不验证真实传感器物理模型。\n",
        encoding="utf-8",
    )
    print(
        f"算子往返核验{'通过' if manifest['status'] == 'complete' else '未通过'}："
        f"{rows_completed}/1800；最大绝对误差={manifest['max_abs_error']:.9g}；CUDA OOM={manifest['cuda_oom']}。",
        flush=True,
    )
    return manifest


def rows_from_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    result = run(args.input_root, args.output_dir)
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
