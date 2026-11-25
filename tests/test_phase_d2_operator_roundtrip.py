import numpy as np
import pytest

from experiments.diagnostics.verify_phase_d2_operator_roundtrip import error_summary


def test_exact_operator_roundtrip_has_zero_error():
    expected = np.array([[0.0, 0.5], [1.0, 2.0]], dtype=np.float32)

    metrics = error_summary(expected, expected.copy())

    assert metrics["bitwise_equal"] is True
    assert metrics["max_abs_error"] == pytest.approx(0.0)
    assert metrics["mae"] == pytest.approx(0.0)
    assert metrics["rmse"] == pytest.approx(0.0)


def test_operator_roundtrip_reports_known_numeric_error():
    expected = np.array([[0.0, 1.0]], dtype=np.float32)
    regenerated = np.array([[0.0, 0.75]], dtype=np.float32)

    metrics = error_summary(expected, regenerated)

    assert metrics["bitwise_equal"] is False
    assert metrics["max_abs_error"] == pytest.approx(0.25)
    assert metrics["mae"] == pytest.approx(0.125)
    assert metrics["rmse"] == pytest.approx(np.sqrt(0.03125))


def test_operator_roundtrip_rejects_shape_or_nonfinite_input():
    with pytest.raises(ValueError, match="shape"):
        error_summary(np.zeros((2, 2)), np.zeros((2, 3)))
    with pytest.raises(ValueError, match="有限"):
        error_summary(np.array([np.nan]), np.array([0.0]))
