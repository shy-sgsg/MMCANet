from __future__ import annotations

import numpy as np
import pytest

from experiments.phase_e_taskb_metrics import compute_taskb_metrics


def test_taskb_metrics_known_identity_case():
    target = np.linspace(0.05, 0.95, 64 * 64, dtype=np.float32).reshape(64, 64)
    target += (np.indices(target.shape).sum(axis=0) % 2).astype(np.float32) * 0.02
    target = np.clip(target, 0.01, 0.99)
    lr = target.reshape(32, 2, 32, 2).mean(axis=(1, 3))

    metrics = compute_taskb_metrics(target, target, lr)

    assert np.isinf(metrics["psnr_db"])
    assert metrics["ssim"] == pytest.approx(1.0, abs=1e-8)
    assert metrics["epi_corr"] == pytest.approx(1.0, abs=1e-8)
    assert metrics["edge_strength_ratio"] == pytest.approx(1.0, abs=1e-8)
    assert metrics["bright_mean_ratio"] == pytest.approx(1.0, abs=1e-8)
    assert metrics["bright_p95_ratio"] == pytest.approx(1.0, abs=1e-8)
    assert metrics["background_mean_ratio"] == pytest.approx(1.0, abs=1e-8)
    assert metrics["reprojection_mae"] == pytest.approx(0.0, abs=2e-8)


def test_taskb_metrics_rejects_shape_and_range_errors():
    target = np.full((32, 32), 0.5, dtype=np.float32)
    lr = np.full((16, 16), 0.5, dtype=np.float32)

    with pytest.raises(ValueError):
        compute_taskb_metrics(target[:, :-1], target, lr)
    with pytest.raises(ValueError):
        compute_taskb_metrics(np.full_like(target, 1.1), target, lr)


def test_taskb_metrics_can_report_undefined_epi_without_fabricating_a_value():
    target = np.linspace(0.05, 0.95, 64 * 64, dtype=np.float32).reshape(64, 64)
    target += (np.indices(target.shape).sum(axis=0) % 2).astype(np.float32) * 0.02
    target = np.clip(target, 0.01, 0.99)
    lr = target.reshape(32, 2, 32, 2).mean(axis=(1, 3))
    flat_prediction = np.full_like(target, 0.5)

    with pytest.raises(ValueError, match="EPI"):
        compute_taskb_metrics(flat_prediction, target, lr)

    partial = compute_taskb_metrics(
        flat_prediction, target, lr, allow_undefined_epi=True
    )

    assert partial["epi_corr"] is None
    assert partial["edge_strength_ratio"] == pytest.approx(0.0)
    assert np.isfinite(partial["psnr_db"])
    identity = compute_taskb_metrics(target, target, lr, allow_undefined_epi=True)
    assert identity["epi_corr"] == pytest.approx(1.0, abs=1e-8)
