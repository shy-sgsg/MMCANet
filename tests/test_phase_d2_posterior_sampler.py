import numpy as np
import pytest
import torch
from torch import nn

from experiments.phase_d2_observation_aware_20261124_v1.sample_posterior import (
    ddim_posterior_sample,
    evaluate_development,
)


class ZeroEpsilonModel(nn.Module):
    def forward(self, noisy, condition, timesteps):
        return noisy * 0.0


def test_posterior_sampler_is_reproducible_finite_and_bounded():
    model = ZeroEpsilonModel().eval()
    observation = torch.full((1, 1, 128, 128), 0.2)
    alpha_bar = torch.linspace(0.99, 0.02, 8).cumprod(dim=0)
    timesteps = torch.tensor([7, 3, 0])

    first = ddim_posterior_sample(
        model, observation, alpha_bar, timesteps, seed=123, return_diagnostics=True
    )
    second = ddim_posterior_sample(
        model, observation, alpha_bar, timesteps, seed=123, return_diagnostics=True
    )

    first_image, diagnostics = first
    second_image, _ = second
    assert first_image.shape == (1, 1, 256, 256)
    assert torch.isfinite(first_image).all()
    assert first_image.min().item() >= 0.0
    assert first_image.max().item() <= 1.0
    assert torch.equal(first_image, second_image)
    assert diagnostics["mean_guidance_gradient_norm"] >= 0.0


def test_posterior_sampler_rejects_invalid_guidance_or_timesteps():
    model = ZeroEpsilonModel()
    observation = torch.ones((1, 1, 128, 128))
    alpha_bar = torch.linspace(0.99, 0.10, 8)
    with pytest.raises(ValueError, match="guidance"):
        ddim_posterior_sample(model, observation, alpha_bar, torch.tensor([7, 0]), 3, -1.0)
    with pytest.raises(ValueError, match="descending"):
        ddim_posterior_sample(model, observation, alpha_bar, torch.tensor([0, 7]), 3)


def test_development_evaluation_keeps_draws_pixel_mean_and_paired_baseline_separate():
    rng = np.random.default_rng(10)
    target = rng.uniform(0.1, 0.9, size=(1, 256, 256)).astype(np.float32)
    lr = rng.uniform(0.05, 1.5, size=(1, 128, 128)).astype(np.float32)
    record = {"sample_id": "sample-a", "source_group_id": "group-a", "lr": lr, "target": target}
    predictions = {"sample-a": np.stack([target[0] + offset for offset in (0.003, 0.01, -0.01, 0.02)])}
    predictions["sample-a"] = np.clip(predictions["sample-a"], 0.0, 1.0).astype(np.float32)
    baseline = {"sample-a": np.clip(target[0] - 0.02, 0.0, 1.0).astype(np.float32)}
    observations = {"sample-a": {"draws": [{"positive_nll": 1.0, "zero_constraint": 0.0, "zero_violation_rate": 0.0}] * 4,
                                   "pixel_mean": {"positive_nll": 0.9, "zero_constraint": 0.0, "zero_violation_rate": 0.0},
                                   "baseline": {"positive_nll": 0.8, "zero_constraint": 0.0, "zero_violation_rate": 0.0}}}

    tables, summary = evaluate_development(predictions, [record], baseline, observations)

    assert len(tables["single_draw_metrics"]) == 4
    assert len(tables["pixel_mean_metrics"]) == 1
    assert len(tables["baseline_metrics"]) == 1
    assert tables["single_draw_metrics"][0]["sample_id"] == "sample-a"
    assert "single_draw_group_balanced" in summary and "pixel_mean_group_balanced" in summary
    assert summary["paired_bootstrap"]["pixel_mean"]["psnr_db"]["group_count"] == 1
