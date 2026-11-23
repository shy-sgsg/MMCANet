import pytest
from types import SimpleNamespace

from scripts import run_phase_c3a_sar_flow as runner


def test_default_preflight_counts_every_cuda_compute_process(monkeypatch):
    assert runner._active_cuda_processes_from_records(
        [
            {"pid": 44415, "process_name": "nautilus"},
            {"pid": 61699, "process_name": "resources"},
            {"pid": 589855, "process_name": "python3.13"},
            {"pid": 700001, "process_name": "python3.13"},
        ]
    ) == [44415, 61699, 589855, 700001]


def test_user_authorized_override_ignores_active_processes_and_low_memory():
    assert runner._resource_preflight_block_reason(
        active_processes=[589855, 700001],
        free_bytes=2 * 1024**3,
        minimum_free_bytes=3 * 1024**3,
        user_authorized_override=True,
    ) is None


def test_default_resource_preflight_preserves_frozen_gates():
    assert runner._resource_preflight_block_reason(
        active_processes=[700001],
        free_bytes=4 * 1024**3,
        minimum_free_bytes=3 * 1024**3,
        user_authorized_override=False,
    ) == "检测到其他 CUDA 计算进程"
    assert runner._resource_preflight_block_reason(
        active_processes=[],
        free_bytes=2 * 1024**3,
        minimum_free_bytes=3 * 1024**3,
        user_authorized_override=False,
    ) == "空闲显存低于冻结协议门槛"


def test_resource_override_requires_an_explicit_chinese_cli_flag():
    assert runner._parse_args([]).user_authorized_resource_override is False
    assert runner._parse_args(["--user-authorized-resource-override"]).user_authorized_resource_override is True


def test_runner_can_resume_a_failed_attempt_into_a_new_experiment_directory():
    destination = runner.REPOSITORY_ROOT / "experiments/diagnostics/c3a_retry_20261123"
    assert runner._resolve_output_path("experiments/diagnostics/c3a_retry_20261123") == destination.resolve()


def test_runner_rejects_output_directories_outside_experiments():
    with pytest.raises(ValueError, match="experiments"):
        runner._resolve_output_path("/tmp/c3a_output")
