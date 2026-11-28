from __future__ import annotations

import numpy as np
import pytest
import subprocess
import sys
import torch
from torch import nn
from pathlib import Path

from experiments.phase_e_taskb_likelihood_guided_eval import (
    _predeclared_gate,
    gaussian_gamma_moment_nll,
    likelihood_guided_ddim_sample,
)


def test_gamma_moment_nll_matches_fixed_mean_and_variance_formula():
    image = torch.full((1, 1, 8, 8), 0.5, dtype=torch.float64)
    observation = torch.full((1, 1, 4, 4), 0.5, dtype=torch.float64)

    loss = gaussian_gamma_moment_nll(image, observation)

    variance = (0.5**2) / 4.0
    expected = 0.5 * np.log(variance + 1e-6) * 16
    assert loss.shape == (1,)
    assert float(loss[0]) == pytest.approx(expected, rel=1e-10)


def test_gamma_moment_nll_has_finite_nonzero_input_gradient():
    image = torch.full((1, 1, 8, 8), 0.5, requires_grad=True)
    observation = torch.full((1, 1, 4, 4), 0.8)

    loss = gaussian_gamma_moment_nll(image, observation).sum()
    gradient, = torch.autograd.grad(loss, image)

    assert torch.isfinite(loss)
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0


def test_gamma_moment_nll_rejects_spatial_contract_mismatch():
    image = torch.full((1, 1, 8, 8), 0.5)
    observation = torch.full((1, 1, 5, 4), 0.5)

    with pytest.raises(ValueError, match="2 倍"):
        gaussian_gamma_moment_nll(image, observation)


class _ZeroNoise(nn.Module):
    def forward(self, residual, mean_estimate, low_resolution, timestep):
        return torch.zeros_like(residual)


def test_likelihood_guided_ddim_is_seeded_finite_and_shape_preserving():
    mean_estimate = torch.full((1, 1, 16, 16), 0.5)
    low_resolution = torch.full((1, 1, 8, 8), 0.45)
    betas = torch.linspace(1e-4, 0.02, 1000)
    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)

    first = likelihood_guided_ddim_sample(
        _ZeroNoise(), mean_estimate, low_resolution, alphas_cumprod, seed=123
    )
    repeated = likelihood_guided_ddim_sample(
        _ZeroNoise(), mean_estimate, low_resolution, alphas_cumprod, seed=123
    )

    assert first.shape == mean_estimate.shape
    assert torch.isfinite(first).all()
    assert float(first.min()) >= 0.0 and float(first.max()) <= 1.0
    torch.testing.assert_close(first, repeated, rtol=0, atol=0)


def test_predeclared_gate_rejects_missing_epi_even_when_psnr_passes():
    candidate = {
        "sample_count": 70,
        "epi_valid_sample_count": 69,
        "psnr_db": 26.0,
        "ssim": 0.9,
        "epi_corr": 0.9,
        "bright_mean_ratio": 1.0,
        "background_mean_ratio": 1.0,
        "reprojection_mae": 0.1,
    }
    baseline = {
        "sample_count": 70,
        "epi_valid_sample_count": 70,
        "psnr_db": 25.0,
        "ssim": 0.85,
        "epi_corr": 0.8,
        "bright_mean_ratio": 1.1,
        "background_mean_ratio": 1.1,
        "reprojection_mae": 0.2,
    }
    psnr_bootstrap = {"bootstrap_95ci_low": 0.5}

    meets, epi_complete = _predeclared_gate(candidate, baseline, psnr_bootstrap)

    assert epi_complete is False
    assert meets is False


def test_likelihood_guided_eval_cli_runs_from_repository_path():
    project_root = Path(__file__).resolve().parents[1]
    script = project_root / "experiments" / "phase_e_taskb_likelihood_guided_eval.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "表示域观测似然引导" in result.stdout
