import csv
from datetime import datetime
import json

import numpy as np
import pytest

from MMCANetV2.phase_c3a_archive import bright_proxy_metrics, write_c3a_sample_archive


def test_bright_proxy_includes_all_values_tied_at_target_quantile():
    target = np.array([[0.0, 0.4, 0.5, 0.5, 0.5]], dtype=np.float32)

    scores = bright_proxy_metrics(target, target, quantile=0.5)

    assert scores["目标分位阈值"] == pytest.approx(0.5)
    assert scores["目标高值像素数"] == 3
    assert scores["目标高值区域占比"] == pytest.approx(0.6)
    assert scores["高值区域绝对误差"] == pytest.approx(0.0)
    assert scores["高值区域均值比"] == pytest.approx(1.0)
    assert scores["高值区域峰值比"] == pytest.approx(1.0)


def test_sample_archive_binds_arrays_ids_and_source_group_statistics(tmp_path):
    output = tmp_path / "归档"
    targets = np.zeros((2, 1, 4, 4), dtype=np.float32)
    targets[:, :, 0, 0] = 1.0
    flow_raw = np.repeat(targets[:, None], 4, axis=1)
    baseline_raw = targets.copy()
    metadata = {
        "training_protocol_sha256": "a" * 64,
        "evaluation_protocol_sha256": "b" * 64,
        "archive_protocol_sha256": "g" * 64,
        "source_manifest_sha256": "c" * 64,
        "development_input_manifest_sha256": "d" * 64,
        "flow_checkpoint_sha256": "e" * 64,
        "baseline_checkpoint_sha256": "f" * 64,
        "seed": 20261105,
        "sampler_steps": 8,
        "confirmation_payload_accessed": False,
        "test_set_accessed": False,
        "test_set_used": False,
        "optical_payload_read": False,
    }

    manifest = write_c3a_sample_archive(
        output,
        sample_ids=("id-1", "id-2"),
        source_group_ids=("group-a", "group-b"),
        patch_ids=("patch-1", "patch-2"),
        targets=targets,
        flow_raw=flow_raw,
        baseline_raw=baseline_raw,
        metadata=metadata,
        bootstrap_seed=20261106,
        bootstrap_repeats=100,
    )

    with np.load(output / "generated_samples.npz", allow_pickle=False) as archive:
        assert archive["flow_raw"].shape == (2, 4, 1, 4, 4)
        assert archive["flow_clipped"].dtype == np.float32
        assert archive["baseline_clipped"].shape == (2, 1, 4, 4)
        assert archive["sample_ids"].tolist() == ["id-1", "id-2"]
        assert np.array_equal(archive["flow_clipped"], flow_raw)
    assert manifest["sample_count"] == 2
    assert manifest["source_group_count"] == 2
    assert manifest["generated_draw_count"] == 8
    assert manifest["test_set_accessed"] is False
    with (output / "bright_proxy_per_draw.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 10
    with (output / "bright_proxy_source_group_summary.csv").open(encoding="utf-8-sig", newline="") as stream:
        group_rows = list(csv.DictReader(stream))
    assert {row["来源组 ID"] for row in group_rows} == {"group-a", "group-b"}
    assert (output / "bright_proxy_group_bootstrap.json").is_file()
    runtime = json.loads((output / "runtime.json").read_text(encoding="utf-8"))
    assert datetime.fromisoformat(runtime["归档开始 UTC"]).tzinfo is not None
    assert "runtime.json" in manifest["files_sha256"]
    saved_manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert saved_manifest["archive_sha256"] == manifest["archive_sha256"]


def test_sample_archive_refuses_existing_destination(tmp_path):
    output = tmp_path / "已存在"
    output.mkdir()
    with pytest.raises(FileExistsError):
        write_c3a_sample_archive(
            output,
            sample_ids=("id-1",),
            source_group_ids=("group-a",),
            patch_ids=("patch-1",),
            targets=np.zeros((1, 1, 4, 4), dtype=np.float32),
            flow_raw=np.zeros((1, 4, 1, 4, 4), dtype=np.float32),
            baseline_raw=np.zeros((1, 1, 4, 4), dtype=np.float32),
            metadata={},
        )
