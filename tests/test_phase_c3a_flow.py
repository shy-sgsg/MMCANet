import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from MMCANetV2.phase_c3a_flow import (
    SarConditionalFlow,
    SarTargetOnlyDataset,
    make_flow_training_pair,
    sample_heun,
    upsample_sar_condition,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_sar_target_only_dataset_never_needs_optical_payload(tmp_path):
    (tmp_path / "lr").mkdir()
    (tmp_path / "target").mkdir()
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    (tmp_path / "files.csv").write_text("sample_id\n", encoding="utf-8")
    sample_id = "sample-a"
    lr_path = tmp_path / "lr" / f"{sample_id}.npy"
    target_path = tmp_path / "target" / f"{sample_id}.npy"
    np.save(lr_path, np.full((1, 128, 128), 2.0, dtype=np.float32), allow_pickle=False)
    np.save(target_path, np.full((1, 256, 256), 0.5, dtype=np.float32), allow_pickle=False)
    row = {
        "sample_id": sample_id,
        "source_group_id": "scene-a",
        "patch_id": "1",
        "lr_sha256": _sha256(lr_path),
        "target_sha256": _sha256(target_path),
    }
    fixed = SimpleNamespace(
        role="train",
        scale=2,
        root=tmp_path,
        rows=[row],
        manifest={"condition_id": "x2_light"},
    )
    dataset = SarTargetOnlyDataset(fixed)
    dataset.verify_all()
    sample = dataset[0]
    assert set(sample) == {"name", "source_group_id", "patch_id", "lr", "target"}
    assert sample["lr"].shape == (1, 128, 128)
    assert sample["lr"].max().item() == 2.0
    assert sample["target"].shape == (1, 256, 256)
    assert not (tmp_path / "optical").exists()


def test_flow_pair_uses_registered_linear_interpolation_without_clipping():
    target = torch.full((2, 1, 4, 4), 0.8)
    condition = torch.zeros_like(target)
    noise = torch.ones_like(target)
    time = torch.tensor([0.0, 1.0])
    state, velocity = make_flow_training_pair(
        target, condition, time, noise, noise_sigma=0.1
    )
    assert torch.allclose(state[0], torch.full_like(state[0], 0.1))
    assert torch.allclose(state[1], target[1])
    assert torch.allclose(velocity, torch.full_like(velocity, 0.7))


def test_flow_pair_rejects_mixed_float_dtypes():
    target = torch.full((1, 1, 4, 4), 0.8, dtype=torch.float32)
    condition = torch.zeros_like(target, dtype=torch.float64)
    noise = torch.ones_like(target)

    with pytest.raises(ValueError, match="dtype"):
        make_flow_training_pair(
            target, condition, torch.tensor([0.5]), noise, noise_sigma=0.1
        )


@pytest.mark.parametrize(
    "time",
    [
        torch.tensor([1.2]),
        torch.tensor([float("nan")]),
        torch.tensor([1], dtype=torch.int64),
    ],
)
def test_flow_pair_rejects_invalid_time(time):
    target = torch.full((1, 1, 4, 4), 0.8)
    condition = torch.zeros_like(target)
    noise = torch.ones_like(target)

    with pytest.raises(ValueError, match="time"):
        make_flow_training_pair(target, condition, time, noise, noise_sigma=0.1)


def test_sar_conditional_flow_performs_finite_cpu_optimizer_update():
    model = SarConditionalFlow(base_channels=8, time_dimension=16)
    target = torch.rand(2, 1, 32, 36)
    condition = torch.rand_like(target)
    time = torch.tensor([0.2, 0.8])
    noise = torch.randn_like(target)
    state, target_velocity = make_flow_training_pair(
        target, condition, time, noise, noise_sigma=0.1
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    before = [parameter.detach().clone() for parameter in model.parameters()]

    optimizer.zero_grad(set_to_none=True)
    result = model(state, condition, time)
    assert result.shape == state.shape
    assert torch.isfinite(result).all()
    loss = torch.nn.functional.mse_loss(result, target_velocity)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    optimizer.step()
    assert any(
        not torch.equal(previous, current.detach())
        for previous, current in zip(before, model.parameters())
    )


def test_heun_sampler_does_not_clip_intermediate_or_final_state():
    class ConstantVelocity(nn.Module):
        def forward(self, state, condition, time):
            return torch.ones_like(state)

    condition = torch.zeros(1, 1, 8, 8)
    noise = torch.ones_like(condition)
    result = sample_heun(
        ConstantVelocity(), condition, noise, steps=4, noise_sigma=0.1
    )
    assert torch.allclose(result, torch.full_like(result, 1.1))


def test_heun_sampler_rejects_mixed_condition_and_noise_dtypes():
    class ConstantVelocity(nn.Module):
        def forward(self, state, condition, time):
            return torch.ones_like(state)

    condition = torch.zeros(1, 1, 4, 4, dtype=torch.float32)
    noise = torch.ones(1, 1, 4, 4, dtype=torch.float64)

    with pytest.raises(ValueError, match="dtype"):
        sample_heun(ConstantVelocity(), condition, noise, steps=2, noise_sigma=0.1)


def test_heun_sampler_rejects_broadcastable_wrong_velocity_shape():
    class BroadcastVelocity(nn.Module):
        def forward(self, state, condition, time):
            return torch.ones(state.shape[0], 1, 1, 1, dtype=state.dtype)

    condition = torch.zeros(1, 1, 4, 4)
    noise = torch.ones_like(condition)

    with pytest.raises(ValueError, match="速度场.*shape"):
        sample_heun(BroadcastVelocity(), condition, noise, steps=2, noise_sigma=0.1)


def test_heun_sampler_rejects_velocity_dtype_mismatch():
    class DoubleVelocity(nn.Module):
        def forward(self, state, condition, time):
            return torch.ones_like(state, dtype=torch.float64)

    condition = torch.zeros(1, 1, 4, 4)
    noise = torch.ones_like(condition)

    with pytest.raises(ValueError, match="速度场.*dtype"):
        sample_heun(DoubleVelocity(), condition, noise, steps=2, noise_sigma=0.1)


def test_heun_sampler_rejects_velocity_device_mismatch():
    class MetaVelocity(nn.Module):
        def forward(self, state, condition, time):
            return torch.ones_like(state, device="meta")

    condition = torch.zeros(1, 1, 4, 4)
    noise = torch.ones_like(condition)

    with pytest.raises(ValueError, match="速度场.*device"):
        sample_heun(MetaVelocity(), condition, noise, steps=2, noise_sigma=0.1)


def test_sar_condition_uses_bilinear_resize_with_requested_size():
    lr = torch.zeros(1, 1, 8, 8)
    condition = upsample_sar_condition(lr, size=(16, 16))
    assert condition.shape == (1, 1, 16, 16)


def test_c3a_runner_uses_the_project_ssim_implementation():
    from scripts.run_phase_c3a_sar_flow import project_ssim as c3a_ssim

    assert c3a_ssim.__module__ == "eval_metrics"
