from __future__ import annotations

import hashlib
import json
import random

import numpy as np
import pytest
import torch
from torch import nn

from experiments import phase_e_train_taskb_pilot as pilot


def _write_checkpoint(path, epoch: int) -> str:
    payload = {
        "format": "phase_e_taskb_checkpoint_v1",
        "method": "centered_residual_conditional_ddpm",
        "epoch": epoch,
        "seed": pilot.SEED,
    }
    torch.save(payload, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_interrupted_run(tmp_path):
    stage_dir = tmp_path / "candidate_components" / "centered_residual_diffusion"
    checkpoint_dir = stage_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    best_checkpoint = checkpoint_dir / "epoch_003.pt"
    last_checkpoint = checkpoint_dir / "epoch_004.pt"
    best_sha = _write_checkpoint(best_checkpoint, 3)
    last_sha = _write_checkpoint(last_checkpoint, 4)
    manifest = {
        "method": "中心残差条件 DDPM epsilon-prediction",
        "status": "interrupted",
        "seed": pilot.SEED,
        "epochs": 15,
        "patch_lr": 64,
        "diffusion_steps": 1000,
        "beta_schedule": "linear(1e-4,0.02)",
        "sampling": {"method": "DDIM", "steps": 20, "eta": 0},
        "ema_decay": 0.999,
        "input_manifest_sha256": "固定输入哈希",
        "confirmation_payload_accessed": False,
        "test_payload_accessed": False,
        "optical_payload_loaded": False,
        "last_epoch": 4,
        "last_checkpoint": str(last_checkpoint),
        "last_checkpoint_sha256": last_sha,
        "best_epoch": 3,
        "best_score": 5.453242691364306,
        "best_checkpoint": str(best_checkpoint),
        "best_checkpoint_sha256": best_sha,
    }
    (stage_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )
    return tmp_path, manifest


def test_resume_stage_accepts_matching_interrupted_run_and_returns_last_and_best(tmp_path):
    source_run, _ = _make_interrupted_run(tmp_path)

    state = pilot._validate_resume_stage(
        source_run, input_manifest_sha256="固定输入哈希", patch_size=64
    )

    assert state["resume_epoch"] == 4
    assert state["resume_checkpoint"].name == "epoch_004.pt"
    assert state["best_epoch"] == 3
    assert state["best_checkpoint"].name == "epoch_003.pt"
    assert state["best_score"] == pytest.approx(5.453242691364306)


def test_resume_stage_accepts_best_checkpoint_from_explicit_parent_run(tmp_path):
    source_run, manifest = _make_interrupted_run(tmp_path / "resume")
    artifact_run = tmp_path / "artifact_parent"
    parent_checkpoint_dir = artifact_run / "candidate_components" / "centered_residual_diffusion" / "checkpoints"
    parent_checkpoint_dir.mkdir(parents=True)
    parent_best = parent_checkpoint_dir / "epoch_003.pt"
    parent_sha = _write_checkpoint(parent_best, 3)
    manifest["best_checkpoint"] = str(parent_best)
    manifest["best_checkpoint_sha256"] = parent_sha
    stage_manifest = (
        source_run / "candidate_components" / "centered_residual_diffusion" / "manifest.json"
    )
    stage_manifest.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

    state = pilot._validate_resume_stage(
        source_run,
        input_manifest_sha256="固定输入哈希",
        patch_size=64,
        allowed_checkpoint_roots=(artifact_run,),
    )

    assert state["best_checkpoint"] == parent_best.resolve()


def test_resume_stage_rejects_best_checkpoint_outside_explicit_parent_roots(tmp_path):
    source_run, manifest = _make_interrupted_run(tmp_path / "resume")
    unrelated_run = tmp_path / "unrelated"
    checkpoint_dir = unrelated_run / "candidate_components" / "centered_residual_diffusion" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    unrelated_checkpoint = checkpoint_dir / "epoch_003.pt"
    unrelated_sha = _write_checkpoint(unrelated_checkpoint, 3)
    manifest["best_checkpoint"] = str(unrelated_checkpoint)
    manifest["best_checkpoint_sha256"] = unrelated_sha
    stage_manifest = (
        source_run / "candidate_components" / "centered_residual_diffusion" / "manifest.json"
    )
    stage_manifest.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="不在允许的运行目录"):
        pilot._validate_resume_stage(
            source_run,
            input_manifest_sha256="固定输入哈希",
            patch_size=64,
            allowed_checkpoint_roots=(tmp_path / "another_parent",),
        )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda manifest: manifest.update(input_manifest_sha256="别的哈希"), "输入清单哈希"),
        (lambda manifest: manifest.update(patch_lr=48), "patch"),
        (lambda manifest: manifest.update(status="running"), "中断"),
        (lambda manifest: manifest.update(last_epoch=3), "轮次"),
    ],
)
def test_resume_stage_rejects_protocol_or_checkpoint_mismatch(tmp_path, change, message):
    source_run, manifest = _make_interrupted_run(tmp_path)
    change(manifest)
    stage_manifest = (
        source_run / "candidate_components" / "centered_residual_diffusion" / "manifest.json"
    )
    stage_manifest.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

    with pytest.raises((ValueError, RuntimeError), match=message):
        pilot._validate_resume_stage(
            source_run, input_manifest_sha256="固定输入哈希", patch_size=64
        )


def test_resume_stage_rejects_checkpoint_hash_mismatch(tmp_path):
    source_run, manifest = _make_interrupted_run(tmp_path)
    manifest["last_checkpoint_sha256"] = "0" * 64
    stage_manifest = (
        source_run / "candidate_components" / "centered_residual_diffusion" / "manifest.json"
    )
    stage_manifest.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

    with pytest.raises((ValueError, RuntimeError), match="SHA256"):
        pilot._validate_resume_stage(
            source_run, input_manifest_sha256="固定输入哈希", patch_size=64
        )


def test_diffusion_training_state_restores_model_optimizer_ema_and_cpu_rng():
    torch.manual_seed(17)
    source_model = nn.Linear(2, 1)
    source_ema = pilot._ExponentialMovingAverage(source_model, decay=0.9)
    source_optimizer = torch.optim.AdamW(source_model.parameters(), lr=0.01)
    loss = source_model(torch.ones((1, 2))).sum()
    loss.backward()
    source_optimizer.step()
    source_ema.update(source_model)

    random.seed(101)
    np.random.seed(101)
    torch.manual_seed(101)
    saved_rng = {
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": [],
    }
    expected_random = (random.random(), np.random.random(), torch.rand(1))

    payload = {
        "format": "phase_e_taskb_checkpoint_v1",
        "method": "centered_residual_conditional_ddpm",
        "epoch": 4,
        "seed": pilot.SEED,
        "model_state_dict": source_model.state_dict(),
        "ema_state_dict": source_ema.model.state_dict(),
        "optimizer_state_dict": source_optimizer.state_dict(),
        **saved_rng,
    }
    restored_model = nn.Linear(2, 1)
    restored_ema = pilot._ExponentialMovingAverage(restored_model, decay=0.9)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=0.01)
    with torch.no_grad():
        restored_model.weight.zero_()
        restored_ema.model.weight.zero_()

    pilot._load_diffusion_training_state(
        payload, restored_model, restored_optimizer, restored_ema,
        expected_epoch=4, device=torch.device("cpu")
    )

    torch.testing.assert_close(restored_model.weight, source_model.weight)
    torch.testing.assert_close(restored_model.bias, source_model.bias)
    torch.testing.assert_close(restored_ema.model.weight, source_ema.model.weight)
    assert restored_optimizer.state_dict()["state"].keys() == source_optimizer.state_dict()["state"].keys()
    assert (random.random(), np.random.random(), torch.rand(1)) == pytest.approx(expected_random)


def test_diffusion_training_state_rejects_wrong_epoch():
    model = nn.Linear(2, 1)
    ema = pilot._ExponentialMovingAverage(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    payload = {
        "format": "phase_e_taskb_checkpoint_v1",
        "method": "centered_residual_conditional_ddpm",
        "epoch": 3,
        "seed": pilot.SEED,
    }

    with pytest.raises(ValueError, match="轮次"):
        pilot._load_diffusion_training_state(
            payload, model, optimizer, ema, expected_epoch=4, device=torch.device("cpu")
        )


def test_group_bootstrap_keeps_undefined_epi_missing_and_reports_support():
    def metric_row(sample_id, group_id, epi_corr, psnr):
        return {
            "sample_id": sample_id,
            "source_group_id": group_id,
            "psnr_db": psnr,
            "ssim": 0.5,
            "epi_corr": epi_corr,
            "edge_strength_ratio": 0.8,
            "bright_mean_ratio": 0.9,
            "bright_p95_ratio": 0.9,
            "background_mean_ratio": 1.0,
            "reprojection_mae": 0.1,
        }

    rows = [
        metric_row("scene_a", "group_a", None, 10.0),
        metric_row("scene_b", "group_b", 0.6, 12.0),
    ]
    reference = [
        metric_row("scene_a", "group_a", 0.4, 20.0),
        metric_row("scene_b", "group_b", 0.4, 20.0),
    ]

    paired, summary = pilot._group_bootstrap(
        rows, reference, method="DDIM_单次", reference="EDSR"
    )

    epi_row = next(row for row in paired if row["sample_id"] == "scene_a")
    epi_summary = next(row for row in summary if row["metric"] == "epi_corr")
    assert epi_row["epi_corr_method"] is None
    assert epi_row["delta_epi_corr"] is None
    assert epi_summary["valid_sample_count"] == 1
    assert epi_summary["missing_sample_count"] == 1
    assert epi_summary["source_group_count"] == 1
    assert epi_summary["group_balanced_delta"] == pytest.approx(0.2)


def test_final_decision_gate_fails_when_candidate_epi_is_incomplete(tmp_path, monkeypatch):
    methods = (
        "Bicubic", "EDSR", "SwinIR", "均值校正点估计", "DDIM_单次",
        "DDIM_四次抽样像素均值", "DDIM_draw_0", "DDIM_draw_1",
        "DDIM_draw_2", "DDIM_draw_3",
    )
    records = [
        {"sample_id": "scene_a", "source_group_id": "group_a"},
        {"sample_id": "scene_b", "source_group_id": "group_b"},
    ]
    metric_values = {
        "Bicubic": (18.0, 0.70, 0.75, 1.1, 1.1, 0.2),
        "EDSR": (20.0, 0.80, 0.80, 1.0, 1.0, 0.15),
        "SwinIR": (21.0, 0.82, 0.82, 1.0, 1.0, 0.14),
        "均值校正点估计": (22.0, 0.85, 0.84, 1.0, 1.0, 0.12),
        "DDIM_单次": (22.0, 0.85, None, 1.0, 1.0, 0.12),
        "DDIM_四次抽样像素均值": (22.0, 0.85, 0.84, 1.0, 1.0, 0.12),
        "DDIM_draw_0": (22.0, 0.85, 0.84, 1.0, 1.0, 0.12),
        "DDIM_draw_1": (22.0, 0.85, 0.84, 1.0, 1.0, 0.12),
        "DDIM_draw_2": (22.0, 0.85, 0.84, 1.0, 1.0, 0.12),
        "DDIM_draw_3": (22.0, 0.85, 0.84, 1.0, 1.0, 0.12),
    }

    def fake_eval(method, predictions, selected_records, output_root):
        psnr, ssim, epi, bright, background, reprojection = metric_values[method]
        return [
            {
                "method": method,
                "sample_id": record["sample_id"],
                "source_group_id": record["source_group_id"],
                "psnr_db": psnr,
                "ssim": ssim,
                "epi_corr": epi,
                "edge_strength_ratio": 0.9,
                "bright_mean_ratio": bright,
                "bright_p95_ratio": bright,
                "background_mean_ratio": background,
                "reprojection_mae": reprojection,
            }
            for record in selected_records
        ]

    monkeypatch.setattr(pilot, "_eval_and_save", fake_eval)
    summary = pilot._write_final_results(
        tmp_path,
        records,
        {method: {} for method in methods},
    )

    assert summary["candidate_epi_valid_sample_count"] == 0
    assert summary["candidate_epi_missing_sample_count"] == 2
    assert summary["candidate_epi_complete"] is False
    assert summary["predeclared_continue_gate_met"] is False


def test_prediction_loader_rejects_missing_and_extra_samples(tmp_path):
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    np.save(predictions / "scene_a.npy", np.full((512, 512), 0.25, dtype=np.float32))
    np.save(predictions / "scene_extra.npy", np.full((512, 512), 0.5, dtype=np.float32))

    with pytest.raises(ValueError, match="文件集合"):
        pilot._load_prediction_map(predictions, ["scene_a", "scene_b"])


def test_prediction_loader_reads_exact_finite_float32_sample_set(tmp_path):
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    expected = np.full((512, 512), 0.25, dtype=np.float32)
    np.save(predictions / "scene_a.npy", expected)

    loaded = pilot._load_prediction_map(predictions, ["scene_a"])

    assert list(loaded) == ["scene_a"]
    np.testing.assert_array_equal(loaded["scene_a"], expected)


def test_prediction_loader_accepts_fixed_channel_first_training_mean(tmp_path):
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    expected = np.full((1, 512, 512), 0.25, dtype=np.float32)
    np.save(predictions / "scene_a.npy", expected)

    loaded = pilot._load_prediction_map(
        predictions, ["scene_a"], expected_shape=(1, 512, 512)
    )

    np.testing.assert_array_equal(loaded["scene_a"], expected)
