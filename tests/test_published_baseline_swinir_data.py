from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np

from experiments.published_baselines_sar_only_20261123_v1.train_swinir_sar import load_pair_arrays


def test_训练脚本直接启动时可加载仓库模块():
    repository = Path(__file__).resolve().parents[1]
    script = repository / "experiments/published_baselines_sar_only_20261123_v1/train_swinir_sar.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_只加载并核验固定SAR输入与伪目标(tmp_path):
    sample_id = "样例"
    root = tmp_path
    (root / "lr").mkdir()
    (root / "target").mkdir()
    lr = np.zeros((1, 128, 128), dtype=np.float32)
    target = np.ones((1, 256, 256), dtype=np.float32) * 0.5
    lr_path = root / "lr" / f"{sample_id}.npy"
    target_path = root / "target" / f"{sample_id}.npy"
    np.save(lr_path, lr, allow_pickle=False)
    np.save(target_path, target, allow_pickle=False)
    row = {
        "sample_id": sample_id,
        "lr_sha256": _sha256(lr_path),
        "target_sha256": _sha256(target_path),
    }

    arrays = load_pair_arrays(root, row, scale=2)

    assert set(arrays) == {"lr", "target"}
    assert arrays["lr"].shape == (1, 128, 128)
    assert arrays["target"].shape == (1, 256, 256)


def test_固定输入哈希不匹配时拒绝读取(tmp_path):
    sample_id = "样例"
    (tmp_path / "lr").mkdir()
    (tmp_path / "target").mkdir()
    np.save(tmp_path / "lr" / f"{sample_id}.npy", np.zeros((1, 128, 128), np.float32), allow_pickle=False)
    np.save(tmp_path / "target" / f"{sample_id}.npy", np.zeros((1, 256, 256), np.float32), allow_pickle=False)
    row = {"sample_id": sample_id, "lr_sha256": "0" * 64, "target_sha256": "0" * 64}

    try:
        load_pair_arrays(tmp_path, row, scale=2)
    except ValueError as error:
        assert "哈希" in str(error)
    else:
        raise AssertionError("固定输入哈希不匹配时必须拒绝")
