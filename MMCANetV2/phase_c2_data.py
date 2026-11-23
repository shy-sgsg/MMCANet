"""读取并核验与 C1 样本清单绑定的 C2 train/development 固定输入。"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

C2_PROTOCOL_ID = 'mmcanet_phase_c2_screen_20261029_v2'
C2_PROTOCOL_PATH = (Path(__file__).resolve().parents[1]
                    / 'protocols/mmcanet_phase_c2/screen_protocol_20261029_v2.json')
C2_PROTOCOL_SHA256 = 'e7b2eb183968dab3512aa2a5ff65244ebf43764dbdd9baa2d25bfffa0864bd1e'


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_frozen_c2_source_manifest(source_manifest: str | Path) -> tuple[dict, str]:
    """核验仓库固定的 C2 协议及其唯一 C1 来源清单。"""
    try:
        protocol_bytes = C2_PROTOCOL_PATH.read_bytes()
    except OSError as exc:
        raise ValueError('冻结 C2 协议文件不可用') from exc
    protocol_sha256 = hashlib.sha256(protocol_bytes).hexdigest()
    if protocol_sha256 != C2_PROTOCOL_SHA256:
        raise ValueError('冻结 C2 协议哈希不匹配')
    try:
        protocol = json.loads(protocol_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError('冻结 C2 协议文件格式无效') from exc
    if (not isinstance(protocol, dict)
            or protocol.get('protocol_id') != C2_PROTOCOL_ID
            or not isinstance(protocol.get('source_manifest_sha256'), str)):
        raise ValueError('冻结 C2 协议标识无效')
    if sha256_file(Path(source_manifest)) != protocol['source_manifest_sha256']:
        raise ValueError('C1 来源清单与冻结 C2 协议不匹配')
    return protocol, protocol_sha256


def sample_id(group: str, patch_id: str) -> str:
    return hashlib.sha256(f'{group}\0{patch_id}'.encode('utf-8')).hexdigest()


class C2FixedDataset(Dataset):
    """读取固定 SAR LR、HR optical 与 HR pseudo-target 三元组。

    来源 CSV 可保留 confirmation 成员元数据，但只接受 train 或 development
    角色，绝不打开 confirmation 影像。每个载入数组都按冻结的 files.csv 核验。
    """

    def __init__(self, root: str | Path, source_manifest: str | Path, *, role: str, scale: int,
                 verify_hashes_on_read: bool = True, allow_confirmation: bool = False):
        if not isinstance(allow_confirmation, bool):
            raise TypeError('allow_confirmation 必须为布尔值')
        if allow_confirmation and role != 'confirmation':
            raise ValueError('仅 confirmation 角色可启用确认集授权')
        allowed_roles = {'train', 'development'}
        if allow_confirmation:
            allowed_roles.add('confirmation')
        if role not in allowed_roles or scale not in {2, 4}:
            raise ValueError('仅支持经显式授权的 train/development/confirmation 角色及 2/4 倍尺度')
        if not isinstance(verify_hashes_on_read, bool):
            raise TypeError('verify_hashes_on_read 必须为布尔值')
        self.root, self.role, self.scale = Path(root), role, scale
        self.verify_hashes_on_read = verify_hashes_on_read
        self._all_arrays_verified = False
        source_manifest = Path(source_manifest)
        protocol, protocol_sha256 = verify_frozen_c2_source_manifest(source_manifest)
        manifest = json.loads((self.root / 'manifest.json').read_text(encoding='utf-8'))
        if (manifest.get('format') != 'c2_fixed_inputs_v1' or manifest.get('status') != 'complete'
                or manifest.get('role') != role or manifest.get('scale') != scale
                or manifest.get('test_set_accessed') is not False or manifest.get('test_set_used') is not False
                or manifest.get('confirmation_payload_accessed') is not (role == 'confirmation')
                or manifest.get('protocol_id') != protocol['protocol_id']
                or manifest.get('protocol_sha256') != protocol_sha256):
            raise ValueError('C2 固定输入清单无效')
        if manifest.get('source_manifest_sha256') != sha256_file(source_manifest):
            raise ValueError('C1 样本清单哈希不匹配')
        if not isinstance(manifest.get('degradation'), dict) or not manifest.get('condition_id'):
            raise ValueError('缺少固定退化条件定义')
        files = self.root / 'files.csv'
        if manifest.get('files_csv_sha256') != sha256_file(files):
            raise ValueError('C2 files.csv 哈希不匹配')
        with source_manifest.open(encoding='utf-8', newline='') as stream:
            all_source = list(csv.DictReader(stream))
        group_roles = {}
        for row in all_source:
            if row['role'] not in {'train', 'development', 'confirmation'}:
                raise ValueError('C1 来源清单包含未知角色')
            group = row['source_group_id']
            if group in group_roles and group_roles[group] != row['role']:
                raise ValueError('C1 来源场景组跨角色')
            group_roles[group] = row['role']
        source = [r for r in all_source if r['role'] == role]
        with files.open(encoding='utf-8', newline='') as stream:
            rows = list(csv.DictReader(stream))
        source_keys = [(r['source_group_id'], r['patch_id']) for r in source]
        keys = [(r['source_group_id'], r['patch_id']) for r in rows]
        if (not source_keys or len(source_keys) != len(set(source_keys)) or len(keys) != len(set(keys))
                or set(keys) != set(source_keys) or len(rows) != manifest.get('sample_count')):
            raise ValueError('C2 固定输入身份集合与冻结 C1 角色清单不一致')
        for row in rows:
            if row['sample_id'] != sample_id(row['source_group_id'], row['patch_id']):
                raise ValueError('sample ID 未绑定来源场景组和 patch')
            for field in ('lr_sha256', 'target_sha256', 'optical_sha256'):
                value = row.get(field, '')
                if len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                    raise ValueError('数组哈希格式无效')
        self.rows = sorted(rows, key=lambda row: row['sample_id'])
        self.manifest = manifest

    def __len__(self) -> int:
        return len(self.rows)

    def _load_array(self, row: dict, name: str, shape: tuple[int, ...], *, verify_hash: bool) -> np.ndarray:
        path = self.root / name / (row['sample_id'] + '.npy')
        if verify_hash and sha256_file(path) != row[name + '_sha256']:
            raise ValueError(f'{name} 数组哈希不匹配：{row["sample_id"]}')
        array = np.load(path, allow_pickle=False)
        if array.shape != shape or array.dtype != np.float32 or not np.isfinite(array).all():
            raise ValueError(f'{name} 数组的形状、类型或数值无效：{row["sample_id"]}')
        if name in {'target', 'optical'} and (array.min() < 0 or array.max() > 1):
            raise ValueError(f'{name} 数组数值超出 [0,1]')
        return array

    def verify_all(self) -> None:
        """Verify every fixed array once before enabling hash-free epoch reads.

        The fast-read mode is intended only for an immutable, manifest-bound
        input directory. Default datasets continue checking each file on every
        access. A partially failed verification never enables fast reads.
        """
        self._all_arrays_verified = False
        for row in self.rows:
            for name, shape in (('lr', (1, 256 // self.scale, 256 // self.scale)),
                                ('target', (1, 256, 256)), ('optical', (3, 256, 256))):
                self._load_array(row, name, shape, verify_hash=True)
        self._all_arrays_verified = True

    def __getitem__(self, index: int) -> dict:
        if not self.verify_hashes_on_read and not self._all_arrays_verified:
            raise RuntimeError('使用未逐次校验的数据集前必须先调用 verify_all')
        row = self.rows[index]
        arrays = {}
        for name, shape in (('lr', (1, 256 // self.scale, 256 // self.scale)),
                            ('target', (1, 256, 256)), ('optical', (3, 256, 256))):
            array = self._load_array(row, name, shape, verify_hash=self.verify_hashes_on_read)
            arrays[name] = torch.from_numpy(np.ascontiguousarray(array))
        return {'name': row['sample_id'], 'source_group_id': row['source_group_id'],
                'patch_id': row['patch_id'], **arrays}
