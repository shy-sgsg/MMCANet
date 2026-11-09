"""C3-A 多次采样稳定性汇总的合成测试。"""
from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.evaluate_c3a_multisample import (
    EXPECTED_EVALUATION_PROTOCOL_SHA256,
    _build_flow_model,
    _compute_edge_metrics,
    _read_protocol,
    _validate_development_identity,
    _validate_training_run,
    aggregate_multi_sample_metrics,
    draw_noise_for_identity,
    group_bootstrap_summary,
)
from scripts import evaluate_c3a_multisample as evaluator


def _rows(sample_id: str, group_id: str, patch_id: str, psnr_values: list[float]):
    return [
        {
            "样本 ID": sample_id,
            "来源组 ID": group_id,
            "patch ID": patch_id,
            "采样编号": draw,
            "PSNR": value,
            "SSIM": 0.5 + draw * 0.01,
            "EPI": 0.5 + draw * 0.1,
            "边缘强度比": 0.8 + draw * 0.05,
            "最终裁剪比例": 0.20,
            "原始输出低于0比例": 0.05 + draw * 0.01,
            "原始输出高于1比例": 0.15 - draw * 0.01,
            "预测像素标准差均值": 0.1,
        }
        for draw, value in enumerate(psnr_values)
    ]


def test_identity_aggregation_averages_draws_without_pseudoreplication():
    flow_rows = _rows("sample-a", "scene-a", "1", [21.0, 22.0, 23.0, 24.0])
    flow_rows += _rows("sample-b", "scene-b", "2", [19.0, 19.5, 20.0, 20.5])
    baseline_rows = [
        {"样本 ID": "sample-a", "来源组 ID": "scene-a", "patch ID": "1", "PSNR": 20.0, "SSIM": 0.4, "EPI": 0.4, "边缘强度比": 0.7, "最终裁剪比例": 0.05, "原始输出低于0比例": 0.02, "原始输出高于1比例": 0.03},
        {"样本 ID": "sample-b", "来源组 ID": "scene-b", "patch ID": "2", "PSNR": 18.0, "SSIM": 0.3, "EPI": 0.3, "边缘强度比": 0.6, "最终裁剪比例": 0.04, "原始输出低于0比例": 0.01, "原始输出高于1比例": 0.03},
    ]

    summaries = aggregate_multi_sample_metrics(flow_rows, baseline_rows, expected_draws=4)

    first, second = summaries
    assert first["样本 ID"] == "sample-a"
    assert first["Flow 平均 PSNR"] == pytest.approx(22.5)
    assert first["Flow PSNR 样本标准差"] == pytest.approx(np.std([21.0, 22.0, 23.0, 24.0], ddof=1))
    assert first["平均配对 ΔPSNR"] == pytest.approx(2.5)
    assert first["Flow 平均 SSIM"] == pytest.approx(0.515)
    assert first["Flow 平均 EPI"] == pytest.approx(0.65)
    assert first["平均配对 ΔEPI"] == pytest.approx(0.25)
    assert first["Flow EPI 样本标准差"] == pytest.approx(np.std([0.5, 0.6, 0.7, 0.8], ddof=1))
    assert first["Flow 平均边缘强度比"] == pytest.approx(0.875)
    assert first["平均预测像素标准差"] == pytest.approx(0.1)
    assert first["Flow 平均原始输出低于0比例"] == pytest.approx(0.065)
    assert first["Flow 平均原始输出高于1比例"] == pytest.approx(0.135)
    assert first["baseline 原始输出低于0比例"] == pytest.approx(0.02)
    assert first["baseline 原始输出高于1比例"] == pytest.approx(0.03)
    assert second["平均配对 ΔPSNR"] == pytest.approx(1.75)
    assert len(summaries) == 2


def test_raw_output_range_diagnostics_separate_lower_and_upper_clipping():
    helper = getattr(evaluator, "_attach_clip_range_components", None)
    assert callable(helper), "多次采样评价器应提供原始输出上下界越界分解"

    rows = [{"最终裁剪比例": 0.5}]
    raw = torch.tensor([[[[-0.5, 0.0, 1.0, 1.5]]]], dtype=torch.float32)

    helper(rows, raw)

    assert rows[0]["原始输出低于0比例"] == pytest.approx(0.25)
    assert rows[0]["原始输出高于1比例"] == pytest.approx(0.25)
    assert rows[0]["最终裁剪比例"] == pytest.approx(
        rows[0]["原始输出低于0比例"] + rows[0]["原始输出高于1比例"]
    )


def test_identity_aggregation_rejects_inconsistent_clip_components():
    flow_rows = _rows("sample-a", "scene-a", "1", [21.0, 22.0, 23.0, 24.0])
    flow_rows[0]["原始输出高于1比例"] += 0.01
    baseline = [
        {
            "样本 ID": "sample-a",
            "来源组 ID": "scene-a",
            "patch ID": "1",
            "PSNR": 20.0,
            "SSIM": 0.4,
            "EPI": 0.4,
            "边缘强度比": 0.7,
            "最终裁剪比例": 0.05,
            "原始输出低于0比例": 0.02,
            "原始输出高于1比例": 0.03,
        }
    ]

    with pytest.raises(ValueError, match="越界比例之和"):
        aggregate_multi_sample_metrics(flow_rows, baseline, expected_draws=4)


def test_aggregation_rejects_missing_or_duplicate_sampling_draws():
    incomplete = _rows("sample-a", "scene-a", "1", [21.0, 22.0, 23.0])
    baseline = [{"样本 ID": "sample-a", "来源组 ID": "scene-a", "patch ID": "1", "PSNR": 20.0, "SSIM": 0.4}]
    with pytest.raises(ValueError, match="采样次数"):
        aggregate_multi_sample_metrics(incomplete, baseline, expected_draws=4)

    duplicated = _rows("sample-a", "scene-a", "1", [21.0, 22.0, 23.0, 24.0])
    duplicated[-1]["采样编号"] = 2
    with pytest.raises(ValueError, match="重复采样编号"):
        aggregate_multi_sample_metrics(duplicated, baseline, expected_draws=4)


def test_aggregation_requires_exact_baseline_identity_and_group_match():
    flow_rows = _rows("sample-a", "scene-a", "1", [21.0, 22.0, 23.0, 24.0])
    missing_baseline = [{"样本 ID": "other", "来源组 ID": "scene-a", "patch ID": "1", "PSNR": 20.0, "SSIM": 0.4}]
    with pytest.raises(ValueError, match="身份集合"):
        aggregate_multi_sample_metrics(flow_rows, missing_baseline, expected_draws=4)

    wrong_group = [{"样本 ID": "sample-a", "来源组 ID": "scene-b", "patch ID": "1", "PSNR": 20.0, "SSIM": 0.4}]
    with pytest.raises(ValueError, match="身份/来源组/patch"):
        aggregate_multi_sample_metrics(flow_rows, wrong_group, expected_draws=4)


def test_draw_noise_is_reproducible_and_changes_with_draw_index():
    first = draw_noise_for_identity("sample-a", draw_index=0, seed=20261105, device=torch.device("cpu"))
    repeat = draw_noise_for_identity("sample-a", draw_index=0, seed=20261105, device=torch.device("cpu"))
    second = draw_noise_for_identity("sample-a", draw_index=1, seed=20261105, device=torch.device("cpu"))

    assert first.shape == (1, 1, 256, 256)
    assert torch.equal(first, repeat)
    assert not torch.equal(first, second)


def test_protocol_schema_and_model_configuration_match_frozen_training_run():
    protocol, digest = _read_protocol()
    assert digest == EXPECTED_EVALUATION_PROTOCOL_SHA256
    assert protocol["format"] == "c3a_multi_sample_stability_protocol_v3"
    assert protocol["data"]["sample_count"] == 200
    assert protocol["data"]["source_group_count"] == 19
    assert protocol["determinism"]["deterministic_algorithms"] is True
    assert protocol["determinism"]["cudnn_deterministic"] is True
    assert "baseline_metrics.csv" in protocol["outputs"]["files"]
    assert "edge_mask_manifest.csv" in protocol["outputs"]["files"]
    assert "EPI" in protocol["metrics"]["per_draw"]
    assert "原始输出低于0比例" in protocol["metrics"]["per_draw"]
    assert "原始输出高于1比例" in protocol["metrics"]["per_draw"]
    assert protocol["metrics"]["selection_or_gate_change"] is False
    assert "Flow 平均原始输出低于0比例" in protocol["metrics"]["group_bootstrap_metrics"]
    assert "Flow 平均原始输出高于1比例" in protocol["metrics"]["group_bootstrap_metrics"]
    assert protocol["metrics"]["structural"]["edge_mask"]["threshold_percentile"] == 90.0

    import json
    from pathlib import Path

    training_protocol = json.loads(
        Path("protocols/mmcanet_phase_c3a/sar_only_flow_c1_v2_20261105_v1.json").read_text(encoding="utf-8")
    )
    model = _build_flow_model(training_protocol["method"]["model"])
    assert sum(parameter.numel() for parameter in model.parameters()) == 323_617


def test_structural_metrics_reuse_fixed_edge_mask_and_fail_on_constant_prediction():
    rng = np.random.default_rng(31)
    target = rng.uniform(0.0, 1.0, size=(1, 32, 32)).astype(np.float32)
    lr = rng.uniform(0.0, 1.0, size=(1, 16, 16)).astype(np.float32)

    identity = _compute_edge_metrics(target, target, lr)

    assert identity["EPI"] == pytest.approx(1.0)
    assert identity["边缘强度比"] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="为零/过小"):
        _compute_edge_metrics(np.zeros_like(target), target, lr)


def test_group_bootstrap_resamples_source_groups_and_is_reproducible():
    rows = [
        {"来源组 ID": "scene-a", "Δ": 1.0},
        {"来源组 ID": "scene-a", "Δ": 3.0},
        {"来源组 ID": "scene-b", "Δ": 5.0},
    ]

    first = group_bootstrap_summary(rows, metric="Δ", seed=7, repeats=100)
    second = group_bootstrap_summary(rows, metric="Δ", seed=7, repeats=100)

    assert first == second
    assert first["来源组等权均值"] == pytest.approx(3.5)
    assert first["来源组数"] == 2


def test_frozen_v3_group_bootstrap_uses_metrics_config_and_expected_group_count():
    protocol, _ = _read_protocol()
    metric_names = protocol["metrics"]["group_bootstrap_metrics"]
    rows = []
    for group_index in range(protocol["data"]["source_group_count"]):
        row = {"来源组 ID": f"合成来源组-{group_index:02d}"}
        for metric_index, metric in enumerate(metric_names):
            row[metric] = float(group_index + metric_index + 1)
        rows.append(row)

    summarize = getattr(evaluator, "summarize_group_bootstrap_metrics", None)
    assert callable(summarize), "C3-A 冻结 bootstrap 配置汇总器尚未实现"
    summaries = summarize(rows, protocol)

    assert set(summaries) == set(metric_names)
    assert all(summary["来源组数"] == 19 for summary in summaries.values())
    assert all(summary["bootstrap 重复数"] == 10000 for summary in summaries.values())
    assert all(summary["bootstrap seed"] == 20261106 for summary in summaries.values())
    with pytest.raises(ValueError, match="来源组数量与冻结协议不一致"):
        summarize(rows[:-1], protocol)


def test_development_manifest_must_cover_expected_unique_identities_and_groups():
    data_protocol = {"sample_count": 2, "source_group_count": 2}
    dataset = SimpleNamespace(
        rows=[
            {"sample_id": "sample-a", "source_group_id": "scene-a"},
            {"sample_id": "sample-b", "source_group_id": "scene-b"},
        ]
    )

    assert _validate_development_identity(dataset, data_protocol) == (2, 2)
    duplicate_dataset = SimpleNamespace(rows=dataset.rows + [dataset.rows[0]])
    with pytest.raises(ValueError, match="身份数或身份重复"):
        _validate_development_identity(duplicate_dataset, data_protocol)

    wrong_group_protocol = {"sample_count": 2, "source_group_count": 1}
    with pytest.raises(ValueError, match="来源组数"):
        _validate_development_identity(dataset, wrong_group_protocol)


def _make_synthetic_training_run(tmp_path):
    import scripts.evaluate_c3a_multisample as evaluator

    training_protocol = json.loads(evaluator.TRAIN_PROTOCOL.read_text(encoding="utf-8"))
    training_protocol_hash = hashlib.sha256(evaluator.TRAIN_PROTOCOL.read_bytes()).hexdigest()
    source_manifest_hash = training_protocol["split"]["source_manifest_sha256"]
    run_dir = tmp_path / "合成训练运行"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    checkpoint_path = checkpoint_dir / "best.pt"
    resume_binding = {"固定测试": "CPU 合成续训绑定"}
    checkpoint = {
        "format": "mmcanet_c3a_sar_only_flow_checkpoint_v2",
        "protocol_sha256": training_protocol_hash,
        "epoch": 3,
        "best_epoch": 3,
        "best_development_group_balanced_mean_psnr": 0.5,
        "resume_binding": resume_binding,
        "model_state_dict": {"weight": torch.tensor([0.5])},
    }
    torch.save(checkpoint, checkpoint_path)
    fixed_input_records = {
        "train": {
            "root": "合成/train",
            "manifest_sha256": "1" * 64,
            "files_csv_sha256": "2" * 64,
            "sample_count": 1600,
            "source_group_count": 92,
            "payload_arrays_read": ["lr", "target"],
        },
        "development": {
            "root": "合成/development",
            "manifest_sha256": "3" * 64,
            "files_csv_sha256": "4" * 64,
            "sample_count": 200,
            "source_group_count": 19,
            "payload_arrays_read": ["lr", "target"],
        },
    }
    run_manifest = {
        "format": "mmcanet_c3a_sar_only_flow_run_v2",
        "status": "complete",
        "protocol_sha256": training_protocol_hash,
        "c1_source_manifest_sha256": source_manifest_hash,
        "best_checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "best_epoch": 3,
        "best_development_group_balanced_mean_psnr": 0.5,
        "resume_binding": resume_binding,
        "fixed_input_records": fixed_input_records,
        "confirmation_payload_accessed": False,
        "test_set_accessed": False,
        "test_set_used": False,
        "optical_payload_read": False,
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(run_manifest), encoding="utf-8"
    )
    input_manifest = {
        "source_manifest_path": training_protocol["split"]["source_manifest_path"],
        "source_manifest_sha256": source_manifest_hash,
        "resume_binding": resume_binding,
        "roles": fixed_input_records,
        "confirmation_payload_accessed": False,
        "test_set_accessed": False,
        "test_set_used": False,
        "optical_payload_read": False,
    }
    (run_dir / "input_manifest.json").write_text(
        json.dumps(input_manifest), encoding="utf-8"
    )
    evaluation_protocol = {
        "training_protocol_sha256": training_protocol_hash,
        "data": {"sample_count": 200, "source_group_count": 19},
    }
    return run_dir, evaluation_protocol, run_manifest, input_manifest


def test_training_run_validator_accepts_complete_cpu_synthetic_run(tmp_path):
    run_dir, protocol, _, _ = _make_synthetic_training_run(tmp_path)

    manifest, checkpoint = _validate_training_run(run_dir, protocol, "synthetic-eval-hash")

    assert manifest["status"] == "complete"
    assert checkpoint["epoch"] == 3
    assert checkpoint["model_state_dict"]["weight"].item() == pytest.approx(0.5)


def test_training_run_validator_rejects_checkpoint_binding_mismatch(tmp_path):
    run_dir, protocol, _, input_manifest = _make_synthetic_training_run(tmp_path)
    input_manifest["resume_binding"] = {"绑定": "不一致"}
    (run_dir / "input_manifest.json").write_text(json.dumps(input_manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="绑定"):
        _validate_training_run(run_dir, protocol, "synthetic-eval-hash")


@pytest.mark.parametrize(
    "protected_flag",
    [
        "confirmation_payload_accessed",
        "test_set_accessed",
        "test_set_used",
        "optical_payload_read",
    ],
)
def test_training_run_validator_rejects_open_protected_payload_flags(tmp_path, protected_flag):
    run_dir, protocol, run_manifest, _ = _make_synthetic_training_run(tmp_path)
    run_manifest[protected_flag] = True
    (run_dir / "manifest.json").write_text(
        json.dumps(run_manifest), encoding="utf-8"
    )

    with pytest.raises(ValueError, match=protected_flag):
        _validate_training_run(run_dir, protocol, "synthetic-eval-hash")


@pytest.mark.parametrize(
    ("field", "value"),
    [("sample_count", 199), ("source_group_count", 18)],
)
def test_training_run_validator_rejects_development_manifest_mismatch(tmp_path, field, value):
    run_dir, protocol, _, _ = _make_synthetic_training_run(tmp_path)
    protocol["data"][field] = value

    with pytest.raises(ValueError, match="development"):
        _validate_training_run(run_dir, protocol, "synthetic-eval-hash")


def test_training_run_validator_rejects_checkpoint_hash_mismatch(tmp_path):
    run_dir, protocol, run_manifest, _ = _make_synthetic_training_run(tmp_path)
    run_manifest["best_checkpoint_sha256"] = "0" * 64
    (run_dir / "manifest.json").write_text(
        json.dumps(run_manifest), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="checkpoint 哈希"):
        _validate_training_run(run_dir, protocol, "synthetic-eval-hash")


def test_training_run_validator_requires_input_manifest_before_payload_read(tmp_path):
    run_dir, protocol, _, _ = _make_synthetic_training_run(tmp_path)
    (run_dir / "input_manifest.json").unlink()

    with pytest.raises(ValueError, match="input_manifest"):
        _validate_training_run(run_dir, protocol, "synthetic-eval-hash")


def test_training_run_validator_rejects_input_manifest_source_hash_mismatch(tmp_path):
    run_dir, protocol, _, input_manifest = _make_synthetic_training_run(tmp_path)
    input_manifest["source_manifest_sha256"] = "0" * 64
    (run_dir / "input_manifest.json").write_text(
        json.dumps(input_manifest), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="来源清单 SHA-256"):
        _validate_training_run(run_dir, protocol, "synthetic-eval-hash")


def test_training_run_validator_rejects_input_manifest_source_path_mismatch(tmp_path):
    run_dir, protocol, _, input_manifest = _make_synthetic_training_run(tmp_path)
    input_manifest["source_manifest_path"] = "其他来源清单.csv"
    (run_dir / "input_manifest.json").write_text(
        json.dumps(input_manifest), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="来源清单路径"):
        _validate_training_run(run_dir, protocol, "synthetic-eval-hash")


def test_training_run_validator_rejects_input_role_record_mismatch(tmp_path):
    run_dir, protocol, _, input_manifest = _make_synthetic_training_run(tmp_path)
    input_manifest["roles"]["development"]["files_csv_sha256"] = "0" * 64
    (run_dir / "input_manifest.json").write_text(
        json.dumps(input_manifest), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="roles.*训练 manifest"):
        _validate_training_run(run_dir, protocol, "synthetic-eval-hash")


@pytest.mark.parametrize(
    "protected_flag",
    [
        "confirmation_payload_accessed",
        "test_set_accessed",
        "test_set_used",
        "optical_payload_read",
    ],
)
def test_training_run_validator_rejects_open_input_manifest_flags(tmp_path, protected_flag):
    run_dir, protocol, _, input_manifest = _make_synthetic_training_run(tmp_path)
    input_manifest[protected_flag] = True
    (run_dir / "input_manifest.json").write_text(
        json.dumps(input_manifest), encoding="utf-8"
    )

    with pytest.raises(ValueError, match=protected_flag):
        _validate_training_run(run_dir, protocol, "synthetic-eval-hash")


def test_training_run_validator_rejects_non_train_development_input_roles(tmp_path):
    run_dir, protocol, _, input_manifest = _make_synthetic_training_run(tmp_path)
    input_manifest["roles"]["confirmation"] = {
        "sample_count": 200,
        "source_group_count": 20,
        "payload_arrays_read": [],
    }
    (run_dir / "input_manifest.json").write_text(
        json.dumps(input_manifest), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="只能包含 train/development"):
        _validate_training_run(run_dir, protocol, "synthetic-eval-hash")


def test_training_run_validator_checks_train_fixed_input_counts(tmp_path):
    run_dir, protocol, run_manifest, input_manifest = _make_synthetic_training_run(tmp_path)
    for manifest in (run_manifest, input_manifest):
        manifest["fixed_input_records" if "fixed_input_records" in manifest else "roles"]["train"]["sample_count"] = 1599
    (run_dir / "manifest.json").write_text(json.dumps(run_manifest), encoding="utf-8")
    (run_dir / "input_manifest.json").write_text(json.dumps(input_manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="train.*样本数"):
        _validate_training_run(run_dir, protocol, "synthetic-eval-hash")


def test_training_run_validator_rejects_optical_array_in_input_manifest(tmp_path):
    run_dir, protocol, run_manifest, input_manifest = _make_synthetic_training_run(tmp_path)
    run_manifest["fixed_input_records"]["development"]["payload_arrays_read"] = [
        "lr",
        "optical",
        "target",
    ]
    input_manifest["roles"] = run_manifest["fixed_input_records"]
    (run_dir / "manifest.json").write_text(json.dumps(run_manifest), encoding="utf-8")
    (run_dir / "input_manifest.json").write_text(
        json.dumps(input_manifest), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="必须且只能包含 lr/target"):
        _validate_training_run(run_dir, protocol, "synthetic-eval-hash")
