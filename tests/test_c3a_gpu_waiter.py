"""C3-A GPU 等待器资源判断的合成测试。"""
from __future__ import annotations

import pytest

from scripts.wait_for_c3a_gpu import (
    EVALUATION_OUTPUT,
    EVALUATION_PROTOCOL,
    MINIMUM_FREE_DISK_BYTES,
    WAIT_OUTPUT,
    gpu_is_ready,
    launch_resources_are_ready,
    parse_gpu_snapshot,
)


def test_gpu_snapshot_parser_reads_gpu_zero_and_deduplicates_other_processes():
    snapshot = parse_gpu_snapshot(
        gpu_csv="NVIDIA GeForce RTX 3050 Laptop GPU, 3200, 4096, 12\n",
        process_csv="123\n123\n456\n",
        current_pid=456,
    )

    assert snapshot == {
        "device_name": "NVIDIA GeForce RTX 3050 Laptop GPU",
        "free_memory_mib": 3200,
        "total_memory_mib": 4096,
        "utilization_percent": 12,
        "other_process_ids": [123],
    }


def test_gpu_launch_requires_no_other_process_and_three_gib_free():
    enough_memory = {
        "free_memory_mib": 3072,
        "other_process_ids": [],
    }
    occupied = {"free_memory_mib": 4096, "other_process_ids": [123]}
    insufficient = {"free_memory_mib": 3071, "other_process_ids": []}

    assert gpu_is_ready(enough_memory, minimum_free_memory_mib=3072)
    assert not gpu_is_ready(occupied, minimum_free_memory_mib=3072)
    assert not gpu_is_ready(insufficient, minimum_free_memory_mib=3072)


def test_launch_uses_two_gib_disk_reserve():
    snapshot = {"free_memory_mib": 4096, "other_process_ids": []}
    reserve = MINIMUM_FREE_DISK_BYTES

    assert reserve == 2 * 1024**3

    assert launch_resources_are_ready(
        snapshot,
        free_disk_bytes=reserve,
        minimum_free_memory_mib=3072,
        minimum_free_disk_bytes=reserve,
    )
    assert not launch_resources_are_ready(
        snapshot,
        free_disk_bytes=reserve - 1,
        minimum_free_memory_mib=3072,
        minimum_free_disk_bytes=reserve,
    )


def test_gpu_snapshot_rejects_unreadable_compute_process_ids():
    with pytest.raises(ValueError, match="进程编号"):
        parse_gpu_snapshot(
            gpu_csv="GPU 0, 4096, 4096, 0\n",
            process_csv="N/A\n",
            current_pid=456,
        )


def test_waiter_targets_unstarted_structural_protocol_and_fresh_record_directory():
    assert EVALUATION_PROTOCOL.name == "multi_sample_stability_c1_v2_20261105_v2.json"
    assert EVALUATION_OUTPUT.name == "c3a_multi_sample_eval_c1_v2_20261105_v2"
    assert WAIT_OUTPUT.name == "c3a_gpu_wait_20261106_v7"
