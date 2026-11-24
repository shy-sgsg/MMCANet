from __future__ import annotations

import torch

from experiments.published_baselines_sar_only_20261123_v1.train_sar_only_diffusion import (
    METRIC_NAMES,
    ConditionalUNet,
    _average_draws_by_sample,
    make_beta_schedule,
    q_sample,
)


def test_扩散噪声日程单调且端点固定():
    betas = make_beta_schedule(10, 1e-4, 2e-2)

    assert betas.shape == (10,)
    assert torch.isclose(betas[0], torch.tensor(1e-4), atol=1e-8, rtol=0)
    assert torch.isclose(betas[-1], torch.tensor(2e-2), atol=1e-8, rtol=0)
    assert torch.all(betas[1:] > betas[:-1])


def test_前向扩散按alpha_bar组合干净目标与噪声():
    x0 = torch.ones(2, 1, 3, 3)
    noise = torch.full_like(x0, 2.0)
    alpha_bar = torch.tensor([0.81, 0.25])
    timesteps = torch.tensor([0, 1])

    xt = q_sample(x0, timesteps, noise, alpha_bar)

    expected = alpha_bar[timesteps].sqrt()[:, None, None, None] * x0
    expected += (1 - alpha_bar[timesteps]).sqrt()[:, None, None, None] * noise
    assert torch.allclose(xt, expected)


def test_条件去噪网络输出与噪声目标同形():
    model = ConditionalUNet(base_channels=8, time_dim=32)
    noisy = torch.randn(2, 1, 32, 32)
    condition = torch.randn(2, 1, 32, 32)
    timestep = torch.tensor([3, 7])

    prediction = model(noisy, condition, timestep)

    assert prediction.shape == noisy.shape
    assert torch.isfinite(prediction).all()


def test_单次生成指标先按图像平均后用于配对统计():
    rows = []
    for sample_id, group_id, values in (("a", "g1", (10.0, 14.0)), ("b", "g2", (20.0, 22.0))):
        for value in values:
            row = {"method": "conditional", "sample_id": sample_id, "source_group_id": group_id}
            row.update({metric: value for metric in METRIC_NAMES})
            rows.append(row)

    averaged = _average_draws_by_sample(rows)

    assert [(row["sample_id"], row["source_group_id"], row["psnr_db"]) for row in averaged] == [
        ("a", "g1", 12.0),
        ("b", "g2", 21.0),
    ]
