import numpy as np
import torch
from torch import nn

from experiments.phase_d2_observation_aware_20261124_v1.train_prior import (
    epsilon_training_loss,
    prepare_training_batch,
)


class FixedRng:
    def __init__(self, values):
        self.values = iter(values)

    def integers(self, low, high=None):
        value = next(self.values)
        if high is None:
            assert 0 <= value < low
        else:
            assert low <= value < high
        return value


def test_prepare_training_batch_keeps_lr_hr_crop_aligned_and_null_condition_zero():
    lr = np.arange(12, dtype=np.float32).reshape(1, 3, 4)
    target = np.repeat(np.repeat(lr, 2, axis=1), 2, axis=2) / 11.0
    records = [{"lr": lr, "target": target}]

    hr_batch, condition = prepare_training_batch(
        records, np.array([0]), FixedRng([1, 1, 0, 0, 0]), lr_patch=2
    )

    expected = target[:, 2:6, 2:6] * 2.0 - 1.0
    assert hr_batch.shape == (1, 1, 4, 4)
    assert torch.equal(hr_batch[0], torch.from_numpy(expected))
    assert torch.count_nonzero(condition) == 0
    assert condition.shape == hr_batch.shape


def test_epsilon_training_loss_is_finite_and_backpropagates():
    class TinyDenoiser(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(2, 1, kernel_size=1)

        def forward(self, noisy, condition, timesteps):
            return self.conv(torch.cat((noisy, condition), dim=1))

    model = TinyDenoiser()
    target = torch.rand(2, 1, 16, 16) * 2.0 - 1.0
    timesteps = torch.tensor([2, 7])
    noise = torch.randn_like(target)
    alpha_bar = torch.linspace(0.99, 0.10, 10)

    loss = epsilon_training_loss(model, target, timesteps, noise, alpha_bar)

    assert torch.isfinite(loss)
    loss.backward()
    assert model.conv.weight.grad is not None
    assert torch.isfinite(model.conv.weight.grad).all()
