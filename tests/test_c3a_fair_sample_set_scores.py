"""有限经验分布与 IID 底层分布评分的边界回归。"""
from __future__ import annotations

import numpy as np
import pytest

from MMCANetV2.phase_c3a_sample_set import fair_energy_score_iid, fair_variogram_score_iid


def test_fair_energy_score_uses_distinct_pairs() -> None:
    draws = np.array([[[[0.0]]], [[[2.0]]]])
    target = np.array([[[0.0]]])
    assert fair_energy_score_iid(draws, target) == pytest.approx(0.0)


def test_fair_energy_score_matches_plugin_for_identical_draws() -> None:
    draws = np.array([[[[1.0, 0.0]]], [[[1.0, 0.0]]], [[[1.0, 0.0]]]])
    target = np.zeros((1, 1, 2))
    assert fair_energy_score_iid(draws, target) == pytest.approx(1.0 / np.sqrt(2.0))


def test_fair_variogram_score_is_unbiased_u_statistic_form() -> None:
    draws = np.array([[[[0.0, 1.0]]], [[[0.0, 3.0]]]])
    target = np.array([[[0.0, 2.0]]])
    score = fair_variogram_score_iid(draws, target, offsets_yx=((0, 1),), power=1.0)
    # 逐成员平方误差均值为 1；distinct-pair 校正为 2；公平估计可为负。
    assert score == pytest.approx(-1.0)


def test_fair_variogram_matches_plugin_for_identical_draws() -> None:
    draws = np.array([[[[0.0, 2.0]]], [[[0.0, 2.0]]], [[[0.0, 2.0]]]])
    target = np.array([[[0.0, 1.0]]])
    score = fair_variogram_score_iid(draws, target, offsets_yx=((0, 1),), power=1.0)
    assert score == pytest.approx(1.0)


@pytest.mark.parametrize("function", [fair_energy_score_iid, fair_variogram_score_iid])
def test_fair_scores_require_at_least_two_draws(function) -> None:
    draw = np.zeros((1, 1, 1, 2))
    target = np.zeros((1, 1, 2))
    with pytest.raises(ValueError):
        if function is fair_variogram_score_iid:
            function(draw, target, offsets_yx=((0, 1),))
        else:
            function(draw, target)


def test_fair_scores_reject_non_finite_draws() -> None:
    draws = np.zeros((2, 1, 1, 2))
    draws[1, 0, 0, 0] = np.nan
    target = np.zeros((1, 1, 2))
    with pytest.raises(ValueError):
        fair_energy_score_iid(draws, target)
    with pytest.raises(ValueError):
        fair_variogram_score_iid(draws, target, offsets_yx=((0, 1),))
