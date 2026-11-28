"""冻结 GT25 metadata split，并仅解包 train/development 的合成 Task B 数据。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.phase_e_data_inventory import (  # noqa: E402
    build_pair_inventory,
    build_reference_inventory,
)
from experiments.phase_e_data_protocol import (  # noqa: E402
    allocate_spatial_splits,
    assign_spatial_groups,
)
from experiments.phase_e_taskb_data import (  # noqa: E402
    build_fixed_taskb_pair,
    center_crop_reference,
    read_reference_tiff,
)

SEED = 20261126
FUSION_SHA256 = "3b2f2ea9eb95a2008ee4c6bcf1da1301df8f07d50c83a3f958a9bd011b657847"
DATASET_SHA256 = "6220032dcb9e733dadf43cd764cb69b8f250158a1c2e72c2d6e3567dc92e6cfc"
EXPECTED_ROLE_COUNTS = {"train": 327, "development": 70, "confirmation": 70}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def archive_members(archive: Path) -> tuple[list[str], set[str]]:
    result = subprocess.run(
        ["7z", "l", "-slt", str(archive)],
        check=True,
        capture_output=True,
        text=True,
    )
    members = []
    solid_flags = set()
    for line in result.stdout.splitlines():
        if line.startswith("Path = "):
            member = line[7:]
            if member and member not in {str(archive), archive.name}:
                members.append(member)
        elif line.startswith("Solid = "):
            solid_flags.add(line.partition("=")[2].strip())
    if not members:
        raise ValueError(f"归档未提供成员路径：{archive}")
    if solid_flags != {"-"}:
        raise ValueError(f"归档不是已确认的非 solid 格式，停止选择性解包：{solid_flags}")
    return members, solid_flags


def _derive_sample_seed(sample_id: str) -> int:
    material = f"{SEED}:{sample_id}:representation-gamma".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") & ((1 << 63) - 1)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_split_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = (
        "sample_id",
        "location_id",
        "source_group_id",
        "split_role",
        "orbit",
        "polarization",
        "latitude",
        "longitude",
        "reference_member",
    )
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _git_state() -> dict[str, Any]:
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--short", "--untracked-files=all"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return {"head": head, "dirty": bool(dirty), "status_short": dirty}
    except (OSError, subprocess.CalledProcessError) as error:
        return {"head": None, "dirty": True, "error": str(error)}


def _prepare_role(
    *,
    role: str,
    rows: list[dict[str, Any]],
    fusion_archive: Path,
    extracted_root: Path,
    data_root: Path,
    input_manifest_rows: list[dict[str, Any]],
    existing_source_root: Path | None = None,
) -> None:
    reuse_source = role == "train" and existing_source_root is not None
    source_root = existing_source_root if reuse_source else extracted_root / role
    if not reuse_source:
        source_root.mkdir(parents=True, exist_ok=False)
    reference_dir = data_root / role / "reference"
    lr_dir = data_root / role / "lr"
    reference_dir.mkdir(parents=True, exist_ok=False)
    lr_dir.mkdir(parents=True, exist_ok=False)

    allowed_members = {str(row["reference_member"]) for row in rows}
    if role == "confirmation":
        raise ValueError("不允许为 confirmation 调用影像解包或预处理")
    if any(".." in Path(member).parts for member in allowed_members):
        raise ValueError("归档成员路径包含越界片段")
    if not reuse_source:
        extract = subprocess.run(
            [
                "7z",
                "x",
                "-y",
                f"-o{source_root}",
                str(fusion_archive),
                *sorted(allowed_members),
            ],
            capture_output=True,
            text=True,
        )
        (source_root / "解包日志.txt").write_text(
            "命令：7z 选择性解包 train/development 授权成员\n"
            + extract.stdout
            + extract.stderr,
            encoding="utf-8",
        )
        if extract.returncode:
            raise RuntimeError(f"{role} 参考影像选择性解包失败，日志：{source_root / '解包日志.txt'}")

    expected_paths = {source_root / member for member in allowed_members}
    actual_paths = {path for path in source_root.rglob("*.tif") if path.is_file()}
    if actual_paths != expected_paths:
        raise ValueError(f"{role} 解包文件集合与授权清单不一致")

    for record in rows:
        member = str(record["reference_member"])
        source_path = source_root / member
        reference_raw = read_reference_tiff(source_path)
        if (
            not np.isfinite(reference_raw).all()
            or reference_raw.min() < 0
            or reference_raw.max() > 255
        ):
            raise ValueError(f"GT25 参考超出冻结的 [0,255] 范围：{record['sample_id']}")
        reference, crop_window = center_crop_reference(reference_raw, size=512)
        sample_seed = _derive_sample_seed(str(record["sample_id"]))
        lr, target = build_fixed_taskb_pair(reference, seed=sample_seed)
        sample_id = str(record["sample_id"])
        reference_output = reference_dir / f"{sample_id}.npy"
        lr_output = lr_dir / f"{sample_id}.npy"
        with reference_output.open("xb") as stream:
            np.save(stream, target, allow_pickle=False)
        with lr_output.open("xb") as stream:
            np.save(stream, lr, allow_pickle=False)
        input_manifest_rows.append(
            {
                "sample_id": sample_id,
                "source_group_id": record["source_group_id"],
                "split_role": role,
                "reference_member": member,
                "source_tiff_sha256": sha256_file(source_path),
                "source_tiff_path": str(source_path),
                "reference_npy": str(reference_output),
                "reference_npy_sha256": sha256_file(reference_output),
                "lr_npy": str(lr_output),
                "lr_npy_sha256": sha256_file(lr_output),
                "source_shape": list(reference_raw.shape),
                "crop_window": crop_window,
                "shape": list(reference.shape),
                "source_dtype": str(reference_raw.dtype),
                "source_min": float(reference_raw.min()),
                "source_max": float(reference_raw.max()),
                "source_mean": float(reference.mean(dtype=np.float64)),
                "input_seed": sample_seed,
                "noise_model": "表示域 Gamma(k=1, theta=1)，理论均值 1、方差 1",
                "lr_min": float(lr.min()),
                "lr_max": float(lr.max()),
                "lr_mean": float(lr.mean(dtype=np.float64)),
                "lr_std": float(lr.std(dtype=np.float64)),
            }
        )


def prepare(
    fusion_archive: Path,
    dataset_archive: Path,
    metadata_csv: Path,
    protocol: Path,
    output_dir: Path,
    existing_train_source_root: Path | None = None,
) -> dict[str, Any]:
    for path in (fusion_archive, dataset_archive, metadata_csv, protocol):
        if not path.is_file():
            raise FileNotFoundError(f"必需输入文件不存在：{path}")
    if output_dir.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖：{output_dir}")
    if existing_train_source_root is not None and not existing_train_source_root.is_dir():
        raise FileNotFoundError(f"可复用 train 原始影像目录不存在：{existing_train_source_root}")

    fusion_sha = sha256_file(fusion_archive)
    if fusion_sha != FUSION_SHA256:
        raise ValueError("fusion.rar SHA-256 与官方公开清单不一致")
    dataset_sha = sha256_file(dataset_archive)
    if dataset_sha != DATASET_SHA256:
        raise ValueError("dataset.rar SHA-256 与官方公开清单不一致")

    sar_members, sar_solid = archive_members(dataset_archive)
    fusion_members, fusion_solid = archive_members(fusion_archive)
    with metadata_csv.open(encoding="utf-8-sig", newline="") as stream:
        metadata_rows = list(csv.DictReader(stream))
    if not metadata_rows:
        raise ValueError("官方 metadata CSV 为空")

    direct_pairs = build_pair_inventory(metadata_rows, sar_members, fusion_members)
    reference_inventory = build_reference_inventory(
        metadata_rows, fusion_members, polarization="VV"
    )
    grouped = assign_spatial_groups(reference_inventory["records"], separation_m=8_000.0)
    splits = allocate_spatial_splits(grouped, seed=SEED)
    counts = {role: len(rows) for role, rows in splits.items()}
    if counts != EXPECTED_ROLE_COUNTS:
        raise ValueError(f"metadata-only split 数量与冻结协议不一致：{counts}")
    role_by_group = {}
    for role, rows in splits.items():
        for row in rows:
            previous = role_by_group.setdefault(row["source_group_id"], role)
            if previous != role:
                raise ValueError("来源组跨越数据角色")
    if len({row["sample_id"] for rows in splits.values() for row in rows}) != sum(counts.values()):
        raise ValueError("metadata-only split 含重复 sample_id")
    if len(splits["development"]) < 5:
        raise ValueError("development 来源组少于 5，停止 pilot")

    output_dir.mkdir(parents=True, exist_ok=False)
    disk_free_before = shutil.disk_usage(output_dir.parent).free
    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir()
    all_rows = [row for role in splits for row in splits[role]]
    split_manifest = {
        "format": "phase_e_gt25_synthetic_split_v3",
        "status": "metadata_frozen_before_payload_access",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_path": str(protocol),
        "protocol_sha256": sha256_file(protocol),
        "seed": SEED,
        "separation_m": 8_000.0,
        "fractions": {"train": 0.70, "development": 0.15, "confirmation": 0.15},
        "fusion_archive_sha256": fusion_sha,
        "dataset_archive_sha256": dataset_sha,
        "metadata_csv_sha256": sha256_file(metadata_csv),
        "fusion_archive_solid_flags": sorted(fusion_solid),
        "dataset_archive_solid_flags": sorted(sar_solid),
        "direct_pairing_audit": direct_pairs["audit"],
        "reference_inventory_audit": reference_inventory["audit"],
        "role_counts": counts,
        "role_group_counts": {
            role: len({row["source_group_id"] for row in rows})
            for role, rows in splits.items()
        },
        "confirmation_payload_accessed": False,
        "test_payload_accessed": False,
        "optical_payload_loaded": False,
        "image_payload_read": False,
        "samples": all_rows,
    }
    split_path = metadata_dir / "split_manifest.json"
    _write_json(split_path, split_manifest)
    _write_split_csv(metadata_dir / "split_manifest.csv", all_rows)
    shutil.copy2(protocol, metadata_dir / protocol.name)
    (output_dir / "命令.txt").write_text(
        "数据准备命令：\n" + shlex.join([sys.executable, *sys.argv]) + "\n",
        encoding="utf-8",
    )

    extracted_root = output_dir / "source"
    input_root = output_dir / "fixed_inputs"
    extracted_root.mkdir()
    input_root.mkdir()
    input_manifest_rows = []
    input_manifest = {
        "format": "phase_e_gt25_synthetic_inputs_v3",
        "status": "preparing_train_development_only",
        "split_manifest_sha256": sha256_file(split_path),
        "protocol_sha256": split_manifest["protocol_sha256"],
        "roles_allowed": ["train", "development"],
        "reference_normalization": "GeoTIFF raw values / 255; no per-image stretch",
        "noise_model": "representation-domain Gamma(k=1, theta=1)",
        "downsample_operator": "non-overlapping 2x2 area average, stride 2",
        "confirmation_payload_accessed": False,
        "test_payload_accessed": False,
        "optical_payload_loaded": False,
        "samples": input_manifest_rows,
    }
    input_manifest_path = metadata_dir / "input_manifest.json"
    _write_json(input_manifest_path, input_manifest)

    for role in ("train", "development"):
        _prepare_role(
            role=role,
            rows=splits[role],
            fusion_archive=fusion_archive,
            extracted_root=extracted_root,
            data_root=input_root,
            input_manifest_rows=input_manifest_rows,
            existing_source_root=existing_train_source_root,
        )
        role_label = {"train": "训练集", "development": "开发集", "confirmation": "确认集"}.get(role, role)
        print(f"已准备{role_label}：{len(splits[role])} 张，未访问确认影像。", flush=True)
        input_manifest["status"] = f"prepared_{role}"
        _write_json(input_manifest_path, input_manifest)

    input_manifest["status"] = "complete"
    input_manifest["created_utc"] = datetime.now(timezone.utc).isoformat()
    _write_json(input_manifest_path, input_manifest)
    shutil.copy2(input_manifest_path, output_dir / "input_manifest.json")
    environment = {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "torch": __import__("torch").__version__,
        "cuda_runtime": __import__("torch").version.cuda,
        "disk_free_before_bytes": disk_free_before,
        "disk_free_after_bytes": shutil.disk_usage(output_dir.parent).free,
        "git": _git_state(),
    }
    _write_json(output_dir / "environment.json", environment)
    return {
        "output_dir": str(output_dir),
        "role_counts": counts,
        "input_count": len(input_manifest_rows),
        "confirmation_payload_accessed": False,
        "split_manifest_sha256": sha256_file(split_path),
        "input_manifest_sha256": sha256_file(input_manifest_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="准备 Phase E GT25 表示域合成 Task B 数据")
    parser.add_argument("--fusion-archive", type=Path, required=True)
    parser.add_argument("--dataset-archive", type=Path, required=True)
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--existing-train-source-root", type=Path)
    args = parser.parse_args()
    result = prepare(
        fusion_archive=args.fusion_archive,
        dataset_archive=args.dataset_archive,
        metadata_csv=args.metadata_csv,
        protocol=args.protocol,
        output_dir=args.output_dir,
        existing_train_source_root=args.existing_train_source_root,
    )
    print(
        "数据准备完成："
        f"训练={result['role_counts']['train']}、"
        f"开发={result['role_counts']['development']}，"
        f"确认影像访问={result['confirmation_payload_accessed']}。"
    )
    print(f"输入清单：{result['input_manifest_sha256']}；目录：{result['output_dir']}")


if __name__ == "__main__":
    main()
