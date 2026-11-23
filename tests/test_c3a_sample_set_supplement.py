import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

import scripts.evaluate_c3a_sample_set_supplement as supplement
from MMCANetV2.phase_c3a_sample_set import (
    energy_score,
    fair_energy_score_iid,
    fair_variogram_score_iid,
    pixelwise_mean,
    variogram_score,
)


def test_protocol_hashes_and_frozen_v3_bindings_are_valid():
    supplement_protocol, supplement_hash, v3_protocol, v3_hash = supplement._read_supplement_protocol()

    assert supplement_hash == supplement.EXPECTED_SUPPLEMENT_SHA256
    assert v3_hash == supplement.frozen_v3.EXPECTED_EVALUATION_PROTOCOL_SHA256
    assert supplement_protocol["implementation_binding"]["scoring_module_sha256"] == supplement.sha256_file(
        supplement.SAMPLE_SET_MODULE
    )
    assert supplement_protocol["training_binding"]["sha256"] == v3_protocol["training_protocol"]["sha256"]


def test_identity_scoring_separates_draw_mean_spread_and_both_distribution_estimands():
    rng = np.random.default_rng(731)
    target = rng.uniform(0.2, 0.8, size=(1, 256, 256)).astype(np.float32)
    perturbations = rng.normal(0.0, 0.08, size=(4, 1, 256, 256)).astype(np.float32)
    raw_draws = target[None, ...] + perturbations
    raw_draws[0, 0, 0, 0] = -0.1
    raw_draws[1, 0, 0, 1] = 1.1
    edge_mask = np.ones((256, 256), dtype=bool)

    draw_rows, identity_row = supplement.score_identity_samples(
        raw_draws,
        target,
        edge_mask,
        sample_id="合成样本甲",
        source_group_id="合成场景甲",
        patch_id="合成块甲",
    )

    clipped = np.clip(raw_draws, 0.0, 1.0)
    mean_image = pixelwise_mean(clipped)
    offsets = ((0, 1), (1, 0), (1, 1), (1, -1), (0, 2), (2, 0))
    assert len(draw_rows) == 4
    assert [row["采样编号"] for row in draw_rows] == [0, 1, 2, 3]
    assert draw_rows[0]["原始输出低于0比例"] > 0
    assert draw_rows[1]["原始输出高于1比例"] > 0
    assert identity_row["像素均值PSNR"] == pytest.approx(
        10.0 * np.log10(1.0 / np.mean((mean_image - target.astype(np.float64)) ** 2)),
        rel=1e-10,
    )
    assert identity_row["EnergyScore插件"] == pytest.approx(energy_score(clipped, target))
    assert identity_row["EnergyScoreIIDFair"] == pytest.approx(fair_energy_score_iid(clipped, target))
    assert identity_row["VariogramScore插件"] == pytest.approx(
        variogram_score(clipped, target, offsets_yx=offsets, power=0.5)
    )
    assert identity_row["VariogramScoreIIDFair"] == pytest.approx(
        fair_variogram_score_iid(clipped, target, offsets_yx=offsets, power=0.5)
    )
    assert identity_row["平均逐像素样本标准差"] > 0


def test_source_group_summary_weights_groups_equally():
    rows = [
        {"样本 ID": "甲", "来源组 ID": "组甲", "指标": 0.0},
        {"样本 ID": "乙", "来源组 ID": "组甲", "指标": 4.0},
        {"样本 ID": "丙", "来源组 ID": "组乙", "指标": 10.0},
    ]

    result = supplement.aggregate_source_group_metric(rows, "指标")

    assert result["来源组等权均值"] == pytest.approx(6.0)
    assert result["各来源组均值"] == {"组乙": 10.0, "组甲": 2.0}


def test_export_summaries_preserve_draws_and_weight_source_groups_equally():
    members = (("甲一", "组甲", 0.0), ("甲二", "组甲", 4.0), ("乙一", "组乙", 10.0))
    identity_rows = []
    draw_rows = []
    for sample_id, group_id, value in members:
        identity_rows.append(
            {
                "样本 ID": sample_id,
                "来源组 ID": group_id,
                **{metric: value for metric in supplement.IDENTITY_METRICS},
            }
        )
        for draw_index in range(2):
            draw_rows.append(
                {
                    "样本 ID": sample_id,
                    "来源组 ID": group_id,
                    "采样编号": draw_index,
                    **{metric: value + draw_index for metric in supplement.DRAW_METRICS},
                }
            )

    group_rows = supplement._source_group_rows(identity_rows, draw_rows, 2)
    summary = supplement._build_summary(identity_rows, draw_rows, 2)

    assert len(group_rows) == 2
    assert group_rows[0]["来源组 ID"] == "组乙"
    assert group_rows[1]["来源组 ID"] == "组甲"
    assert group_rows[1]["像素均值PSNR"] == pytest.approx(2.0)
    assert group_rows[1]["采样1_PSNR"] == pytest.approx(3.0)
    assert summary["按来源组等权汇总的身份级对象"]["像素均值PSNR"]["来源组等权均值"] == pytest.approx(6.0)
    assert summary["按来源组等权汇总的单次draw"]["1"]["PSNR"]["来源组等权均值"] == pytest.approx(7.0)


def test_synthetic_cpu_batch_loop_generates_reproducible_draw_rows_and_identity_scores():
    class ZeroVelocity(torch.nn.Module):
        def forward(self, state, _condition, _time):
            return torch.zeros_like(state)

    rng = np.random.default_rng(991)
    sample = {
        "name": "合成样本甲",
        "source_group_id": "合成场景甲",
        "patch_id": "合成块甲",
        "lr": torch.from_numpy(rng.uniform(0.1, 1.0, size=(1, 128, 128)).astype(np.float32)),
        "target": torch.from_numpy(rng.uniform(0.1, 0.9, size=(1, 256, 256)).astype(np.float32)),
    }
    loader = DataLoader([sample], batch_size=1, shuffle=False, num_workers=0)
    edge_masks = {"合成样本甲": np.ones((256, 256), dtype=bool)}
    options = {
        "device": torch.device("cpu"),
        "seed": 20261105,
        "draw_indices": [0, 1, 2, 3],
        "sampler_steps": 8,
        "noise_sigma": 0.1,
        "edge_masks_by_id": edge_masks,
        "expected_count": 1,
        "variogram_offsets": ((0, 1), (1, 0), (1, 1), (1, -1), (0, 2), (2, 0)),
        "variogram_power": 0.5,
    }
    first_model = ZeroVelocity()
    first_draws, first_identities, first_runtime = supplement._evaluate_batches(
        first_model, loader, **options
    )
    second_draws, second_identities, second_runtime = supplement._evaluate_batches(
        ZeroVelocity(), loader, **options
    )

    assert not first_model.training
    assert len(first_draws) == 4
    assert len(first_identities) == 1
    assert [row["采样编号"] for row in first_draws] == [0, 1, 2, 3]
    assert first_identities[0]["平均逐像素样本标准差"] > 0
    assert first_draws == second_draws
    assert first_identities == second_identities
    assert first_runtime >= 0 and second_runtime >= 0


def test_missing_training_manifest_fails_before_development_payload_or_output_creation(
    tmp_path, monkeypatch
):
    output_dir = tmp_path / "新结果"
    training_run = tmp_path / "experiments/phase_c3a_sar_only_flow_c1_v2_20261105_v1"
    training_run.mkdir(parents=True)
    training_hash = supplement.sha256_file(supplement.TRAIN_PROTOCOL_PATH)
    monkeypatch.setattr(supplement, "ROOT", tmp_path)
    monkeypatch.setattr(
        supplement,
        "_read_supplement_protocol",
        lambda: (
            {"training_binding": {"expected_run": "experiments/phase_c3a_sar_only_flow_c1_v2_20261105_v1"}},
            "补充协议摘要",
            {"training_protocol": {"sha256": training_hash}},
            "v3 协议摘要",
        ),
    )
    observed = {}

    def fake_check_resources(_protocol, *, user_authorized_override=False):
        observed["user_authorized_override"] = user_authorized_override
        return torch.device("cpu"), 0, 0

    monkeypatch.setattr(supplement.frozen_v3, "_check_resources", fake_check_resources)

    def fail_if_data_accessed(*_args, **_kwargs):
        raise AssertionError("训练 manifest 校验前读取了 development 固定输入")

    monkeypatch.setattr(supplement, "_open_development_dataset", fail_if_data_accessed)

    with pytest.raises(FileNotFoundError):
        supplement._run(training_run, output_dir, user_authorized_resource_override=True)

    assert observed["user_authorized_override"] is True
    assert not output_dir.exists()


def test_supplement_run_location_override_requires_flag_and_stays_under_experiments(tmp_path, monkeypatch):
    monkeypatch.setattr(supplement, "ROOT", tmp_path)
    expected = (tmp_path / "experiments/expected").resolve()
    alternate = (tmp_path / "experiments/attempt2").resolve()
    outside = (tmp_path / "outside/attempt2").resolve()

    assert supplement._validate_run_location(expected, expected, allow_compatible_run_location=False) is False
    with pytest.raises(ValueError, match="补充协议绑定"):
        supplement._validate_run_location(alternate, expected, allow_compatible_run_location=False)
    assert supplement._validate_run_location(alternate, expected, allow_compatible_run_location=True) is True
    with pytest.raises(ValueError, match="experiments"):
        supplement._validate_run_location(outside, expected, allow_compatible_run_location=True)


def test_supplement_cli_exposes_explicit_resource_and_run_location_overrides():
    args = supplement._parse_args(
        ["--user-authorized-resource-override", "--allow-compatible-run-location"]
    )

    assert args.user_authorized_resource_override is True
    assert args.allow_compatible_run_location is True
