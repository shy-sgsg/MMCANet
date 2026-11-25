import numpy as np
import pytest

from experiments.diagnostics.analyze_phase_d2_zero_support import summarize_array


def test_zero_support_counts_exact_zeros_and_allows_values_above_one():
    summary = summarize_array(np.array([[[0.0, 0.25], [1.0, 2.0]]], dtype=np.float32))

    assert summary["zero_pixel_count"] == 1
    assert summary["pixel_count"] == 4
    assert summary["zero_fraction"] == pytest.approx(0.25)
    assert summary["positive_pixel_count"] == 3
    assert summary["minimum_positive_value"] == pytest.approx(0.25)


def test_zero_support_rejects_negative_or_nonfinite_lr_values():
    with pytest.raises(ValueError, match="非负"):
        summarize_array(np.array([[[0.0, -0.01]]], dtype=np.float32))
    with pytest.raises(ValueError, match="有限"):
        summarize_array(np.array([[[0.0, np.nan]]], dtype=np.float32))
