"""SARptical 空间 pilot 的只读 SAR 数据入口与固定输入构造。"""
from __future__ import annotations

import hashlib
import math
import re
import zipfile
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.io import loadmat
from torch.nn import functional as F


SPLIT_SEED = 187762
GRID_SIZE_METERS = 250.0
BUFFER_METERS = 75.0
EXPECTED_GRID_GROUPS = 28
EXPECTED_ROLE_GROUP_COUNTS = {"train": 12, "development": 8, "confirmation": 8}
EXPECTED_ROLE_SAMPLE_COUNTS = {"train": 916, "development": 306, "confirmation": 306}
SAR_DIRECTORY = "patch_SAR_OPT_SQUARE"
COORDINATE_MEMBER = "selectedPntXYZ.mat"
SAR_NAME_PATTERN = re.compile(r"^point_(\d+)_ampPatch\.mat$")
PROTECTED_ROLES = {"confirmation", "test", "optical"}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_coordinate_matrix(payload: bytes) -> np.ndarray:
    try:
        values = loadmat(BytesIO(payload)).get("selectedPntXYZ")
    except Exception as exc:
        raise ValueError("selectedPntXYZ.mat 无法解析") from exc
    if (
        not isinstance(values, np.ndarray)
        or values.ndim != 2
        or values.shape[1] != 4
        or values.shape[0] == 0
    ):
        raise ValueError("SARptical 坐标表必须是非空 N×4 数值矩阵")
    if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
        raise ValueError("SARptical 坐标表必须为有限数值")
    ids = values[:, 0]
    if not np.array_equal(ids, np.rint(ids)) or len(np.unique(ids)) != len(ids):
        raise ValueError("SARptical 坐标点 ID 必须是唯一整数")
    return values.astype(np.float64, copy=False)


def build_spatial_split(archive_path: Path, seed: int = SPLIT_SEED) -> dict[str, Any]:
    """只读取 ZIP 成员名和坐标 MAT，构建冻结的 12/8/8 网格分组。"""
    archive_path = Path(archive_path)
    if int(seed) != SPLIT_SEED:
        raise ValueError(f"空间划分 seed 已冻结为 {SPLIT_SEED}")
    if not archive_path.is_file():
        raise FileNotFoundError(f"SARptical 归档不存在：{archive_path}")

    try:
        with zipfile.ZipFile(archive_path, "r") as archive:
            names = archive.namelist()
            coordinate_members = [
                name for name in names if Path(name).name == COORDINATE_MEMBER
            ]
            if len(coordinate_members) != 1:
                raise ValueError("归档内 selectedPntXYZ.mat 必须且只能有一份")
            sar_members: dict[int, str] = {}
            for name in names:
                if not name.endswith("_ampPatch.mat"):
                    continue
                match = SAR_NAME_PATTERN.match(Path(name).name)
                if match is None:
                    raise ValueError(f"SAR 文件名不符合点 ID 约定：{name}")
                point_id = int(match.group(1))
                if point_id in sar_members:
                    raise ValueError(f"SAR 文件点 ID 重复：{point_id}")
                sar_members[point_id] = name
            coordinate_payload = archive.read(coordinate_members[0])
    except zipfile.BadZipFile as exc:
        raise ValueError("SARptical 数据归档损坏") from exc

    coordinates = _read_coordinate_matrix(coordinate_payload)
    coordinate_ids = {int(value) for value in coordinates[:, 0]}
    if coordinate_ids != set(sar_members):
        raise ValueError("SAR 文件名点 ID 与坐标表不完全对应")

    grouped: dict[tuple[int, int], list[tuple[int, float, float, float]]] = defaultdict(list)
    for point_id_value, east_value, north_value, height_value in coordinates:
        point_id = int(point_id_value)
        east, north, height = float(east_value), float(north_value), float(height_value)
        grid_east = math.floor(east / GRID_SIZE_METERS)
        grid_north = math.floor(north / GRID_SIZE_METERS)
        offset_east = east - grid_east * GRID_SIZE_METERS
        offset_north = north - grid_north * GRID_SIZE_METERS
        if (
            BUFFER_METERS <= offset_east <= GRID_SIZE_METERS - BUFFER_METERS
            and BUFFER_METERS <= offset_north <= GRID_SIZE_METERS - BUFFER_METERS
        ):
            grouped[(grid_east, grid_north)].append((point_id, east, north, height))

    ordered_groups = sorted(grouped)
    if len(ordered_groups) != EXPECTED_GRID_GROUPS:
        raise ValueError(f"保守空间筛选后应有 28 个网格组，实际为 {len(ordered_groups)}")
    if sum(len(grouped[key]) for key in ordered_groups) != sum(EXPECTED_ROLE_SAMPLE_COUNTS.values()):
        raise ValueError("保守空间筛选样本数不符合冻结候选 1,528")

    order = np.random.Generator(np.random.PCG64(SPLIT_SEED)).permutation(len(ordered_groups))
    roles_by_group: dict[tuple[int, int], str] = {}
    for rank, index in enumerate(order):
        role = "train" if rank < 12 else "development" if rank < 20 else "confirmation"
        roles_by_group[ordered_groups[int(index)]] = role

    groups: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    for grid_east, grid_north in ordered_groups:
        role = roles_by_group[(grid_east, grid_north)]
        point_rows = sorted(grouped[(grid_east, grid_north)], key=lambda row: row[0])
        sample_ids = [f"point_{row[0]}" for row in point_rows]
        group_id = f"utm250_{grid_east}_{grid_north}"
        groups.append(
            {
                "source_group_id": group_id,
                "grid_east_index": int(grid_east),
                "grid_north_index": int(grid_north),
                "sample_count": len(point_rows),
                "role": role,
                "sample_ids": sample_ids,
            }
        )
        for point_id, east, north, height in point_rows:
            samples.append(
                {
                    "sample_id": f"point_{point_id}",
                    "point_id": int(point_id),
                    "source_group_id": group_id,
                    "role": role,
                    "utm_easting": east,
                    "utm_northing": north,
                    "height": height,
                    "sar_member": sar_members[point_id],
                }
            )

    role_group_counts = {
        role: sum(group["role"] == role for group in groups)
        for role in EXPECTED_ROLE_GROUP_COUNTS
    }
    role_sample_counts = {
        role: sum(sample["role"] == role for sample in samples)
        for role in EXPECTED_ROLE_SAMPLE_COUNTS
    }
    if role_group_counts != EXPECTED_ROLE_GROUP_COUNTS:
        raise ValueError(f"空间角色组数与冻结协议不符：{role_group_counts}")
    if role_sample_counts != EXPECTED_ROLE_SAMPLE_COUNTS:
        raise ValueError(f"空间角色样本数与冻结协议不符：{role_sample_counts}")

    return {
        "schema_version": "sarptical_spatial_split_v1",
        "seed": SPLIT_SEED,
        "bit_generator": "PCG64",
        "numpy_version": np.__version__,
        "grid_size_meters": GRID_SIZE_METERS,
        "buffer_meters": BUFFER_METERS,
        "groups_total": len(groups),
        "samples_total": len(samples),
        "role_group_counts": role_group_counts,
        "role_sample_counts": role_sample_counts,
        "source_archive_sha256": sha256_file(archive_path),
        "coordinate_member": coordinate_members[0],
        "coordinate_member_sha256": sha256_bytes(coordinate_payload),
        "confirmation_payload_accessed": False,
        "test_set_accessed": False,
        "optical_payload_loaded": False,
        "groups": groups,
        "samples": samples,
    }


def parse_sar_mat_payload(payload: bytes, sample_id: str) -> np.ndarray:
    """只接受恰好一个有限 112×112 数值图像矩阵。"""
    try:
        fields = loadmat(BytesIO(payload))
    except Exception as exc:
        raise ValueError(f"{sample_id} 的 SAR MAT 文件无法解析") from exc
    candidates = []
    for key, value in fields.items():
        if key.startswith("__") or not isinstance(value, np.ndarray):
            continue
        if value.shape == (112, 112) and np.issubdtype(value.dtype, np.number):
            candidates.append((key, value))
    if len(candidates) != 1:
        if len(candidates) > 1:
            raise ValueError(f"{sample_id} 的 SAR MAT 矩阵有歧义")
        raise ValueError(f"{sample_id} 的 SAR MAT 必须含唯一 112×112 数值矩阵")
    field_name, array = candidates[0]
    if field_name != "ampCrop":
        raise ValueError(f"{sample_id} 的唯一矩阵字段不是 ampCrop")
    if not np.isfinite(array).all():
        raise ValueError(f"{sample_id} 的 SAR MAT 矩阵含非有限值")
    return np.ascontiguousarray(array, dtype=np.float32)


def _authorized_sample(
    sample_id: str,
    role: str,
    split: dict[str, Any],
) -> dict[str, Any]:
    if role in PROTECTED_ROLES:
        raise ValueError(f"禁止读取 {role} 载荷")
    if role not in {"train", "development"}:
        raise ValueError(f"禁止读取未知数据角色：{role}")
    sample_lookup = {row["sample_id"]: row for row in split.get("samples", [])}
    sample = sample_lookup.get(sample_id)
    if sample is None:
        raise ValueError(f"样本不在冻结 split 中：{sample_id}")
    if sample["role"] != role:
        raise ValueError(f"样本角色不匹配：{sample_id} 属于 {sample['role']}，请求为 {role}")
    member = sample.get("sar_member")
    if not isinstance(member, str) or not member.startswith(f"{SAR_DIRECTORY}/"):
        raise ValueError(f"{sample_id} 的 SAR 成员路径无效")
    return sample


def _load_authorized_sar(
    archive_path: Path,
    sample_id: str,
    role: str,
    split: dict[str, Any],
) -> tuple[np.ndarray, str]:
    sample = _authorized_sample(sample_id, role, split)
    with zipfile.ZipFile(archive_path, "r") as archive:
        payload = archive.read(sample["sar_member"])
    return parse_sar_mat_payload(payload, sample_id), sha256_bytes(payload)


def load_sar_matrix(
    archive_path: Path,
    sample_id: str,
    role: str,
    split: dict[str, Any],
) -> np.ndarray:
    """按 split 中的真实角色授权读取；confirmation/test/Optical 在开 ZIP 前拒绝。"""
    return _load_authorized_sar(archive_path, sample_id, role, split)[0]


def prepare_pair_arrays(
    target_raw: np.ndarray, *, mean: float, std: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按固定 antialias bicubic ×2 构造 LR、标准化目标和 bicubic 锚点。"""
    target = np.asarray(target_raw, dtype=np.float32)
    if target.shape != (112, 112) or not np.isfinite(target).all():
        raise ValueError("SAR target 必须是有限 112×112 矩阵")
    if not math.isfinite(float(mean)) or not math.isfinite(float(std)) or std <= 0:
        raise ValueError("训练标准化统计量必须有限且标准差为正")
    raw_tensor = torch.from_numpy(np.ascontiguousarray(target))[None, None]
    lr_raw = F.interpolate(
        raw_tensor,
        size=(56, 56),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    target_norm = (raw_tensor - float(mean)) / float(std)
    lr_norm = (lr_raw - float(mean)) / float(std)
    anchor = F.interpolate(
        lr_norm,
        size=(112, 112),
        mode="bicubic",
        align_corners=False,
        antialias=False,
    )
    return (
        np.ascontiguousarray(lr_norm[0].numpy(), dtype=np.float32),
        np.ascontiguousarray(target_norm[0].numpy(), dtype=np.float32),
        np.ascontiguousarray(anchor[0].numpy(), dtype=np.float32),
    )


def prepare_fixed_inputs(
    archive_path: Path,
    split: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """读取 train/development SAR 成员，冻结 train 标准化并写入固定缓存。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    raw_records: dict[str, list[tuple[dict[str, Any], np.ndarray, str]]] = {
        "train": [],
        "development": [],
    }
    with zipfile.ZipFile(archive_path, "r") as archive:
        for role in ("train", "development"):
            rows = [sample for sample in split["samples"] if sample["role"] == role]
            for row in rows:
                authorized_row = _authorized_sample(row["sample_id"], role, split)
                try:
                    payload = archive.read(authorized_row["sar_member"])
                except (zipfile.BadZipFile, KeyError) as exc:
                    raise ValueError(
                        f"{role} SAR 成员损坏或不可读：{row['sample_id']} / {authorized_row['sar_member']}"
                    ) from exc
                matrix = parse_sar_mat_payload(payload, row["sample_id"])
                raw_records[role].append((row, matrix, sha256_bytes(payload)))

    if len(raw_records["train"]) != EXPECTED_ROLE_SAMPLE_COUNTS["train"]:
        raise ValueError("train SAR 载荷数量与冻结清单不一致")
    if len(raw_records["development"]) != EXPECTED_ROLE_SAMPLE_COUNTS["development"]:
        raise ValueError("development SAR 载荷数量与冻结清单不一致")

    train_values = np.concatenate(
        [matrix.reshape(-1).astype(np.float64, copy=False) for _, matrix, _ in raw_records["train"]]
    )
    mean = float(train_values.mean(dtype=np.float64))
    std = float(train_values.std(dtype=np.float64))
    if not math.isfinite(mean) or not math.isfinite(std) or std <= 1e-8:
        raise ValueError("train SAR 标准化统计量无效")

    manifest_rows: list[dict[str, Any]] = []
    for role in ("train", "development"):
        role_dir = output_dir / role
        role_dir.mkdir(parents=True, exist_ok=False)
        for row, matrix, source_hash in raw_records[role]:
            lr, target, anchor = prepare_pair_arrays(matrix, mean=mean, std=std)
            cache_path = role_dir / f"{row['sample_id']}.npz"
            with cache_path.open("xb") as stream:
                np.savez(
                    stream,
                    lr=lr,
                    target=target,
                    anchor=anchor,
                    sample_id=np.asarray(row["sample_id"]),
                    source_group_id=np.asarray(row["source_group_id"]),
                )
            manifest_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "source_group_id": row["source_group_id"],
                    "role": role,
                    "sar_member": row["sar_member"],
                    "sar_member_sha256": source_hash,
                    "cache_relative_path": str(cache_path.relative_to(output_dir)),
                    "cache_sha256": sha256_file(cache_path),
                }
            )

    return {
        "schema_version": "sarptical_fixed_input_manifest_v1",
        "normalization": {
            "source_role": "train",
            "mean": mean,
            "std": std,
            "fit_pixel_count": int(train_values.size),
            "target_domain": "SARptical ampCrop dB 表示域，按 train 全局均值/标准差线性标准化",
        },
        "degradation": {
            "scale": 2,
            "operation": "PyTorch bicubic 下采样",
            "antialias": True,
            "align_corners": False,
            "input_shape": [1, 56, 56],
            "target_shape": [1, 112, 112],
            "interpretation": "表示域受控退化；不代表真实传感器模型",
        },
        "sample_count": len(manifest_rows),
        "role_sample_counts": {
            role: sum(row["role"] == role for row in manifest_rows)
            for role in ("train", "development")
        },
        "confirmation_payload_accessed": False,
        "test_set_accessed": False,
        "optical_payload_loaded": False,
        "samples": manifest_rows,
    }


def load_cached_records(input_root: Path, split: dict[str, Any], role: str) -> list[dict[str, Any]]:
    """加载已冻结的 train/development 缓存；不提供 confirmation/test 入口。"""
    if role in PROTECTED_ROLES or role not in {"train", "development"}:
        raise ValueError(f"禁止加载 {role} 缓存")
    role_samples = [row for row in split["samples"] if row["role"] == role]
    records = []
    for role_sample in role_samples:
        sample_id = role_sample["sample_id"]
        path = Path(input_root) / role / f"{sample_id}.npz"
        with np.load(path, allow_pickle=False) as arrays:
            lr = arrays["lr"]
            target = arrays["target"]
            anchor = arrays["anchor"]
            if (
                lr.shape != (1, 56, 56)
                or target.shape != (1, 112, 112)
                or anchor.shape != (1, 112, 112)
                or any(value.dtype != np.float32 for value in (lr, target, anchor))
                or any(not np.isfinite(value).all() for value in (lr, target, anchor))
            ):
                raise ValueError(f"固定输入缓存的尺寸、类型或有限性错误：{sample_id}")
            if str(arrays["sample_id"].item()) != sample_id:
                raise ValueError(f"固定输入缓存中的 sample_id 不匹配：{sample_id}")
            if str(arrays["source_group_id"].item()) != role_sample["source_group_id"]:
                raise ValueError(f"固定输入缓存中的来源组不匹配：{sample_id}")
            records.append(
                {
                    "sample_id": sample_id,
                    "source_group_id": role_sample["source_group_id"],
                    "lr": lr,
                    "target": target,
                    "anchor": anchor,
                }
            )
    return records
