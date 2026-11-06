from pathlib import Path

import pytest

from scripts.wait_for_c3a_gpu import wait_output_path


def test_wait_output_path_uses_a_new_diagnostic_directory_per_run_id() -> None:
    first = wait_output_path("c3a_gpu_wait_20261106_v7")
    second = wait_output_path("c3a_gpu_wait_20261106_v8")

    assert first != second
    assert first.parts[-2:] == ("diagnostics", "c3a_gpu_wait_20261106_v7")
    assert second.parts[-2:] == ("diagnostics", "c3a_gpu_wait_20261106_v8")


@pytest.mark.parametrize(
    "run_id",
    ["../outside", "c3a_gpu_wait_", "other_20261106_v8", "c3a_gpu_wait_v8/extra"],
)
def test_wait_output_path_rejects_invalid_run_id(run_id: str) -> None:
    with pytest.raises(ValueError):
        wait_output_path(run_id)
