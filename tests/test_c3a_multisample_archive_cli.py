from pathlib import Path
import subprocess
import sys

import pytest

from scripts import evaluate_c3a_multisample as evaluator


def test_archive_directory_is_an_opt_in_cli_argument(tmp_path):
    run_dir = tmp_path / "C3-A"
    archive_dir = tmp_path / "样本归档"

    args = evaluator._parse_args(
        ["--training-run-dir", str(run_dir), "--archive-directory", str(archive_dir)]
    )

    assert args.training_run_dir == run_dir
    assert args.archive_directory == archive_dir


def test_archive_directory_defaults_to_disabled():
    args = evaluator._parse_args([])

    assert args.archive_directory is None


def test_multisample_resource_override_ignores_active_cuda_processes_and_low_memory(monkeypatch):
    monkeypatch.setattr(evaluator.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(evaluator, "_active_cuda_processes", lambda: [589855])
    monkeypatch.setattr(evaluator.torch.cuda, "mem_get_info", lambda _device: (2 * 1024**3, 4 * 1024**3))

    device, free_bytes, total_bytes = evaluator._check_resources(
        {"resources": {"minimum_free_memory_gib": 3}},
        user_authorized_override=True,
    )

    assert device == evaluator.torch.device("cuda:0")
    assert free_bytes == 2 * 1024**3
    assert total_bytes == 4 * 1024**3


def test_multisample_resource_gates_remain_strict_without_override(monkeypatch):
    monkeypatch.setattr(evaluator.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(evaluator, "_active_cuda_processes", lambda: [589855])

    with pytest.raises(RuntimeError, match="其他 CUDA 计算进程"):
        evaluator._check_resources({"resources": {"minimum_free_memory_gib": 3}})


def test_multisample_resource_override_is_explicit_cli_argument():
    args = evaluator._parse_args(["--user-authorized-resource-override"])

    assert args.user_authorized_resource_override is True


def test_training_protocol_hash_reader_accepts_frozen_v3_nested_field():
    assert evaluator._expected_training_protocol_hash(
        {"training_protocol": {"sha256": "frozen-training-hash"}}
    ) == "frozen-training-hash"


def test_multisample_evaluator_wraps_fixed_development_dataset_before_sar_only_view(monkeypatch):
    observed = {}

    class FixedDataset:
        def __init__(self, root, source_manifest, *, role, scale, verify_hashes_on_read):
            observed.update(
                root=root,
                source_manifest=source_manifest,
                role=role,
                scale=scale,
                verify_hashes_on_read=verify_hashes_on_read,
            )
            self.role = role
            self.scale = scale
            self.manifest = {"condition_id": "x2_light"}

    class SarOnlyDataset:
        def __init__(self, fixed_dataset):
            observed["wrapped_role"] = fixed_dataset.role

    monkeypatch.setattr(evaluator, "C2FixedDataset", FixedDataset)
    monkeypatch.setattr(evaluator, "SarTargetOnlyDataset", SarOnlyDataset)

    evaluator._open_development_dataset()

    assert observed["role"] == "development"
    assert observed["scale"] == 2
    assert observed["verify_hashes_on_read"] is False
    assert observed["wrapped_role"] == "development"


def test_multisample_result_directory_can_be_unique_and_must_stay_under_experiments():
    requested = "experiments/diagnostics/c3a_multisample_retry_20261123"

    assert evaluator._resolve_output_path(requested) == (evaluator.ROOT / requested).resolve()

    with pytest.raises(ValueError, match="experiments"):
        evaluator._resolve_output_path("/tmp/c3a_multisample")


def test_multi_sample_returns_raw_draw_archive_when_capture_is_requested(monkeypatch):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Dataset
    from MMCANetV2.phase_c3a_metrics import sobel_magnitude

    class OneDevelopmentIdentity(Dataset):
        def __len__(self):
            return 1

        def __getitem__(self, index):
            del index
            target = torch.zeros((1, 256, 256), dtype=torch.float32)
            target[:, 48:192, 48:192] = 0.6
            target[:, 96:160, 96:160] = 1.0
            return {
                "name": "identity-1",
                "source_group_id": "group-1",
                "patch_id": "patch-1",
                "lr": torch.zeros((1, 128, 128), dtype=torch.float32),
                "target": target,
            }

    class SmallVelocity(torch.nn.Module):
        def forward(self, state, condition, time):
            return 0.01 * state + 0.05 * condition + time[:, None, None, None]

    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args, **kwargs: None)
    target = OneDevelopmentIdentity()[0]["target"].numpy()[0]
    edge_mask = sobel_magnitude(target) > 0
    result = evaluator.evaluate_multi_sample(
        SmallVelocity(),
        DataLoader(OneDevelopmentIdentity(), batch_size=1, shuffle=False),
        device=torch.device("cpu"),
        seed=20261105,
        draws=2,
        sampler_steps=1,
        noise_sigma=0.1,
        edge_masks_by_id={"identity-1": edge_mask},
        capture_predictions=True,
    )

    rows, _runtime, capture = result
    assert len(rows) == 2
    assert capture["flow_raw"].shape == (1, 2, 1, 256, 256)
    assert capture["sample_ids"] == ["identity-1"]
    assert capture["source_group_ids"] == ["group-1"]
    assert np.isfinite(capture["flow_raw"]).all()


def test_standalone_evaluation_help_uses_chinese_and_exits_cleanly():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / "scripts/evaluate_c3a_multisample.py"), "--help"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "用法：" in result.stdout
    assert "--archive-directory" in result.stdout
