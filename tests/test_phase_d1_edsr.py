import torch
import pytest

from experiments.phase_d1_edsr_20261124_v1.edsr_sar import (
    build_edsr_sar,
    load_published_edsr_class,
)


def test_edsr_sar_is_single_channel_x2_bounded_and_differentiable():
    edsr_class = load_published_edsr_class()
    model = build_edsr_sar(edsr_class)
    value = torch.rand(2, 1, 32, 40) * 1.7

    prediction = model(value)

    assert prediction.shape == (2, 1, 64, 80)
    assert torch.isfinite(prediction).all()
    assert prediction.min().item() >= 0.0
    assert prediction.max().item() <= 1.0
    prediction.mean().backward()
    gradients = [p.grad for p in model.parameters() if p.requires_grad]
    assert gradients and all(g is not None and torch.isfinite(g).all() for g in gradients)


def test_edsr_sar_rejects_wrong_channel_contract():
    model = build_edsr_sar(load_published_edsr_class())
    with pytest.raises(ValueError, match="单通道"):
        model(torch.rand(1, 3, 32, 32))


def test_edsr_sar_exposes_published_core_and_task_adaptation_identity():
    edsr_class = load_published_edsr_class()
    model = build_edsr_sar(edsr_class)

    assert isinstance(model.core, edsr_class)
    assert len(model.core.body) == 16
    assert model.core.conv_first.in_channels == 1
    assert model.core.conv_last.out_channels == 1
    assert model.experiment_config["身份"] == "TASK_ADAPTED_PUBLISHED"
    assert model.experiment_config["上游源码"] == "BasicSR/basicsr/archs/edsr_arch.py"
