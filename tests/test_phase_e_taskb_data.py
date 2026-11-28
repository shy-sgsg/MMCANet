from __future__ import annotations

import numpy as np
import pytest

from experiments.phase_e_taskb_data import (
    build_fixed_taskb_pair,
    center_crop_reference,
    read_reference_tiff,
)


def test_fixed_taskb_pair_is_deterministic_and_preserves_x2_contract():
    reference = np.full((32, 40), 128, dtype=np.uint8)

    first_lr, first_target = build_fixed_taskb_pair(reference, seed=20261126)
    second_lr, second_target = build_fixed_taskb_pair(reference, seed=20261126)

    assert first_lr.shape == (1, 16, 20)
    assert first_target.shape == (1, 32, 40)
    assert first_lr.dtype == np.float32
    assert first_target.dtype == np.float32
    np.testing.assert_array_equal(first_lr, second_lr)
    np.testing.assert_array_equal(first_target, second_target)
    np.testing.assert_array_equal(
        first_target, np.full_like(first_target, np.float32(128 / 255))
    )
    assert np.isfinite(first_lr).all()


def test_fixed_taskb_pair_rejects_invalid_reference_contracts():
    for reference in (
        np.zeros((7, 8), dtype=np.uint8),
        np.zeros((8, 8, 3), dtype=np.uint8),
        np.full((8, 8), 256, dtype=np.uint16),
        np.full((8, 8), np.nan, dtype=np.float32),
    ):
        with pytest.raises(ValueError):
            build_fixed_taskb_pair(reference, seed=1)


def test_reference_tiff_reader_requires_one_grayscale_band(tmp_path):
    import cv2

    path = tmp_path / "样例.tif"
    expected = np.arange(64, dtype=np.uint8).reshape(8, 8)
    assert cv2.imwrite(str(path), expected)
    actual = read_reference_tiff(path)
    np.testing.assert_array_equal(actual, expected)


def test_reference_center_crop_is_deterministic_and_records_window():
    reference = np.arange(519 * 515, dtype=np.uint32).reshape(519, 515)
    cropped, window = center_crop_reference(reference, size=512)
    assert cropped.shape == (512, 512)
    assert window == {"row_start": 3, "row_stop": 515, "col_start": 1, "col_stop": 513}
    np.testing.assert_array_equal(cropped, reference[3:515, 1:513])


def test_reference_center_crop_rejects_small_or_multichannel_arrays():
    with pytest.raises(ValueError):
        center_crop_reference(np.zeros((511, 512), dtype=np.uint8), size=512)
    with pytest.raises(ValueError):
        center_crop_reference(np.zeros((520, 520, 3), dtype=np.uint8), size=512)
