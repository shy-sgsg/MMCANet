#!/usr/bin/env python3
"""等待 GPU 达到冻结门槛，再串行执行 C3-A 训练与多次采样诊断。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "scripts/run_phase_c3a_sar_flow.py"
EVALUATION_SCRIPT = ROOT / "scripts/evaluate_c3a_multisample.py"
TRAIN_PROTOCOL = ROOT / "protocols/mmcanet_phase_c3a/sar_only_flow_c1_v2_20261105_v1.json"
EVALUATION_PROTOCOL = ROOT / "protocols/mmcanet_phase_c3a/multi_sample_stability_c1_v2_20261105_v2.json"
TRAIN_OUTPUT = ROOT / "experiments/phase_c3a_sar_only_flow_c1_v2_20261105_v1"
EVALUATION_OUTPUT = ROOT / "experiments/diagnostics/c3a_multi_sample_eval_c1_v2_20261105_v2"
WAIT_OUTPUT = ROOT / "experiments/diagnostics/c3a_gpu_wait_20261106_v7"
WAIT_MANIFEST = WAIT_OUTPUT / "manifest.json"
WAIT_COMMAND = WAIT_OUTPUT / "command.txt"
POLL_SECONDS = 60
MINIMUM_FREE_DISK_BYTES = 2 * 1024**3


def wait_output_path(run_id: str) -> Path:
    prefix = "c3a_gpu_wait_"
    suffix = run_id[len(prefix) :] if run_id.startswith(prefix) else ""
    if not suffix or not all(character.isalnum() or character == "_" for character in run_id):
        raise ValueError("等待运行编号须采用 c3a_gpu_wait_<唯一后缀> 格式")
    return ROOT / "experiments/diagnostics" / run_id


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_gpu_snapshot(gpu_csv: str, process_csv: str, *, current_pid: int) -> dict[str, Any]:
    rows = list(csv.reader(gpu_csv.splitlines(), skipinitialspace=True))
    if len(rows) != 1 or len(rows[0]) != 4:
        raise ValueError("无法解析 GPU 0 的显存和利用率信息")
    name, free_memory, total_memory, utilization = (value.strip() for value in rows[0])
    process_values = [value.strip() for value in process_csv.splitlines() if value.strip()]
    if any(not value.isdigit() for value in process_values):
        raise ValueError("CUDA 计算进程编号无法解析；按安全规则不启动训练")
    process_ids = sorted({int(value) for value in process_values if int(value) != current_pid})
    return {
        "device_name": name,
        "free_memory_mib": int(float(free_memory)),
        "total_memory_mib": int(float(total_memory)),
        "utilization_percent": int(float(utilization)),
        "other_process_ids": process_ids,
    }


def gpu_is_ready(snapshot: dict[str, Any], *, minimum_free_memory_mib: int) -> bool:
    return (
        not snapshot["other_process_ids"]
        and snapshot["free_memory_mib"] >= minimum_free_memory_mib
    )


def launch_resources_are_ready(
    snapshot: dict[str, Any],
    *,
    free_disk_bytes: int,
    minimum_free_memory_mib: int,
    minimum_free_disk_bytes: int,
) -> bool:
    return gpu_is_ready(snapshot, minimum_free_memory_mib=minimum_free_memory_mib) and (
        free_disk_bytes >= minimum_free_disk_bytes
    )


def _query_gpu() -> dict[str, Any]:
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--id=0",
            "--query-gpu=name,memory.free,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    processes = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return parse_gpu_snapshot(gpu.stdout, processes.stdout, current_pid=os.getpid())


def _read_run_status(output_dir: Path) -> str | None:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        return str(json.loads(manifest_path.read_text(encoding="utf-8")).get("status"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"无法读取已有运行记录：{manifest_path}（{error}）") from error


def _save_manifest(manifest: dict[str, Any]) -> None:
    WAIT_MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _wait_for_gpu(
    manifest: dict[str, Any],
    *,
    stage: str,
    minimum_free_memory_mib: int,
    minimum_free_disk_bytes: int,
) -> None:
    while True:
        try:
            free_disk_bytes = shutil.disk_usage(ROOT).free
            manifest["最近磁盘查询"] = {
                "时间 UTC": _utc_now(),
                "可用字节": free_disk_bytes,
                "可用 GiB": round(free_disk_bytes / 1024**3, 3),
                "要求可用 GiB": round(minimum_free_disk_bytes / 1024**3, 3),
            }
            snapshot = _query_gpu()
            manifest["最近 GPU 查询"] = {"时间 UTC": _utc_now(), **snapshot}
            manifest["最近查询错误"] = None
            if launch_resources_are_ready(
                snapshot,
                free_disk_bytes=free_disk_bytes,
                minimum_free_memory_mib=minimum_free_memory_mib,
                minimum_free_disk_bytes=minimum_free_disk_bytes,
            ):
                manifest["状态"] = f"GPU 与磁盘预检通过，准备{stage}"
                _save_manifest(manifest)
                print(
                    f"GPU 条件满足：{snapshot['device_name']} 空闲显存 "
                    f"{snapshot['free_memory_mib']}/{snapshot['total_memory_mib']} MiB；"
                    f"磁盘可用 {free_disk_bytes / 1024**3:.2f} GiB；开始{stage}。",
                    flush=True,
                )
                return
            disk_ready = free_disk_bytes >= minimum_free_disk_bytes
            if not disk_ready:
                manifest["状态"] = f"等待磁盘空间后{stage}"
                wait_message = (
                    f"等待磁盘空间：可用 {free_disk_bytes / 1024**3:.2f} GiB，"
                    f"要求至少 {minimum_free_disk_bytes / 1024**3:.2f} GiB；"
                    "等待器不会删除数据或历史结果。"
                )
            else:
                manifest["状态"] = f"等待 GPU 资源后{stage}"
                wait_message = (
                    f"等待 GPU：空闲显存 {snapshot['free_memory_mib']}/{snapshot['total_memory_mib']} MiB，"
                    f"其他 CUDA 计算进程 {len(snapshot['other_process_ids'])} 个。"
                )
            _save_manifest(manifest)
            print(wait_message, flush=True)
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            manifest["状态"] = f"等待资源查询恢复后{stage}"
            manifest["最近查询错误"] = f"{type(error).__name__}：{error}"
            _save_manifest(manifest)
            print(f"GPU 或磁盘查询暂不可用：{error}；{POLL_SECONDS} 秒后重试。", flush=True)
        time.sleep(POLL_SECONDS)


def _write_initial_records() -> dict[str, Any]:
    WAIT_OUTPUT.mkdir(parents=True, exist_ok=False)
    train_protocol = json.loads(TRAIN_PROTOCOL.read_text(encoding="utf-8"))
    evaluation_protocol = json.loads(EVALUATION_PROTOCOL.read_text(encoding="utf-8"))
    train_hash = _sha256(TRAIN_PROTOCOL)
    evaluation_hash = _sha256(EVALUATION_PROTOCOL)
    manifest: dict[str, Any] = {
        "记录格式": "C3-A 训练与多次采样诊断 GPU/磁盘等待记录 v1",
        "运行编号": WAIT_OUTPUT.name,
        "状态": "正在检查 GPU 资源",
        "启动时间 UTC": _utc_now(),
        "轮询间隔秒": POLL_SECONDS,
        "训练协议路径": str(TRAIN_PROTOCOL.relative_to(ROOT)),
        "训练协议 SHA-256": train_hash,
        "多次采样诊断协议路径": str(EVALUATION_PROTOCOL.relative_to(ROOT)),
        "多次采样诊断协议 SHA-256": evaluation_hash,
        "资源门槛 MiB": int(float(train_protocol["resources"]["minimum_free_memory_gib"]) * 1024),
        "磁盘最小可用空间 GiB": MINIMUM_FREE_DISK_BYTES / 1024**3,
        "CUDA 并发进程要求": 0,
        "训练运行目录": str(TRAIN_OUTPUT.relative_to(ROOT)),
        "多次采样输出目录": str(EVALUATION_OUTPUT.relative_to(ROOT)),
        "确认集载荷访问": False,
        "测试集访问": False,
        "测试集使用": False,
        "光学载荷读取": False,
        "等待期间读取影像": False,
        "方法选择或 gate 更新": False,
        "执行命令": [
            [sys.executable, str(TRAIN_SCRIPT)],
            [sys.executable, str(EVALUATION_SCRIPT), "--training-run-dir", str(TRAIN_OUTPUT)],
        ],
        "代码 SHA-256": {
            "scripts/wait_for_c3a_gpu.py": _sha256(Path(__file__).resolve()),
            "scripts/run_phase_c3a_sar_flow.py": _sha256(TRAIN_SCRIPT),
            "scripts/evaluate_c3a_multisample.py": _sha256(EVALUATION_SCRIPT),
        },
    }
    _save_manifest(manifest)
    WAIT_COMMAND.write_text(
        "等待器命令：\n"
        f"{sys.executable} scripts/wait_for_c3a_gpu.py --run-id {WAIT_OUTPUT.name}\n"
        "轮询 GPU 0 资源与项目分区剩余空间；等待期间不读取影像，也不删除文件。\n"
        "达到零个其他 CUDA 计算进程、空闲显存不少于 3,072 MiB 且磁盘至少剩余 2 GiB 后，依次运行：\n"
        f"{sys.executable} scripts/run_phase_c3a_sar_flow.py\n"
        f"{sys.executable} scripts/evaluate_c3a_multisample.py --training-run-dir {TRAIN_OUTPUT}\n",
        encoding="utf-8",
    )
    return manifest


def _run_training(
    manifest: dict[str, Any], minimum_free_memory_mib: int, minimum_free_disk_bytes: int
) -> None:
    while True:
        _wait_for_gpu(
            manifest,
            stage="C3-A 训练",
            minimum_free_memory_mib=minimum_free_memory_mib,
            minimum_free_disk_bytes=minimum_free_disk_bytes,
        )
        manifest["状态"] = "C3-A 训练正在运行"
        manifest["C3-A 训练启动时间 UTC"] = _utc_now()
        _save_manifest(manifest)
        return_code = subprocess.run([sys.executable, str(TRAIN_SCRIPT)], cwd=ROOT, check=False).returncode
        status = _read_run_status(TRAIN_OUTPUT)
        if return_code == 0 and status == "complete":
            manifest["状态"] = "C3-A 训练完成，准备多次采样诊断"
            manifest["C3-A 训练结束时间 UTC"] = _utc_now()
            _save_manifest(manifest)
            print("C3-A 训练完成；后续只进行冻结 development 多次采样诊断。", flush=True)
            return
        if return_code == 2 and status is None:
            manifest["资源预检未通过次数"] = int(manifest.get("资源预检未通过次数", 0)) + 1
            manifest["状态"] = "训练器二次预检未通过，继续等待 GPU"
            _save_manifest(manifest)
            print("训练器二次 GPU 预检未通过；未创建 run 目录或读取数据，稍后重试。", flush=True)
            continue
        raise RuntimeError(f"C3-A 训练未正常完成：退出码 {return_code}，运行状态 {status}")


def _run_evaluation(
    manifest: dict[str, Any], minimum_free_memory_mib: int, minimum_free_disk_bytes: int
) -> None:
    if _read_run_status(EVALUATION_OUTPUT) == "complete":
        manifest["状态"] = "多次采样诊断已完成"
        manifest["结束时间 UTC"] = _utc_now()
        _save_manifest(manifest)
        return
    if EVALUATION_OUTPUT.exists():
        raise RuntimeError("多次采样输出目录已存在且未完成；保留现场并停止，不覆盖或重跑")

    while True:
        _wait_for_gpu(
            manifest,
            stage="development 多次采样诊断",
            minimum_free_memory_mib=minimum_free_memory_mib,
            minimum_free_disk_bytes=minimum_free_disk_bytes,
        )
        manifest["状态"] = "development 多次采样诊断正在运行"
        manifest["多次采样诊断启动时间 UTC"] = _utc_now()
        _save_manifest(manifest)
        process = subprocess.Popen(
            [sys.executable, str(EVALUATION_SCRIPT), "--training-run-dir", str(TRAIN_OUTPUT)],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        output_lines: list[str] = []
        assert process.stdout is not None
        for line in process.stdout:
            output_lines.append(line)
            print(line, end="", flush=True)
        return_code = process.wait()
        status = _read_run_status(EVALUATION_OUTPUT)
        if return_code == 0 and status == "complete":
            manifest["状态"] = "训练与多次采样诊断全部完成"
            manifest["多次采样诊断结束时间 UTC"] = _utc_now()
            manifest["结束时间 UTC"] = _utc_now()
            _save_manifest(manifest)
            print("C3-A 训练与预登记的 development 多次采样诊断均已完成。", flush=True)
            return
        output = "".join(output_lines)
        resource_failure = any(
            marker in output for marker in ("GPU 仍有", "GPU 空闲显存", "无法读取 CUDA 计算进程清单")
        )
        if return_code != 0 and status is None and resource_failure:
            manifest["评价资源预检未通过次数"] = int(manifest.get("评价资源预检未通过次数", 0)) + 1
            manifest["状态"] = "评价器二次 GPU 预检未通过，继续等待 GPU"
            _save_manifest(manifest)
            print("评价器二次 GPU 预检未通过，尚未创建结果目录或读取数据；稍后重试。", flush=True)
            continue
        raise RuntimeError(f"多次采样诊断未正常完成：退出码 {return_code}，运行状态 {status}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="等待 CUDA 资源满足冻结门槛后运行 C3-A 训练与诊断。")
    parser.add_argument("--run-id", required=True, help="本次等待的唯一编号，例如 c3a_gpu_wait_20261106_v8")
    args = parser.parse_args(argv)
    global WAIT_OUTPUT, WAIT_MANIFEST, WAIT_COMMAND
    WAIT_OUTPUT = wait_output_path(args.run_id)
    WAIT_MANIFEST = WAIT_OUTPUT / "manifest.json"
    WAIT_COMMAND = WAIT_OUTPUT / "command.txt"
    if WAIT_OUTPUT.exists():
        raise FileExistsError(f"等待记录目录已存在，拒绝覆盖：{WAIT_OUTPUT}")
    manifest = _write_initial_records()
    train_protocol = json.loads(TRAIN_PROTOCOL.read_text(encoding="utf-8"))
    evaluation_protocol = json.loads(EVALUATION_PROTOCOL.read_text(encoding="utf-8"))
    minimum_free_memory_mib = max(
        int(float(train_protocol["resources"]["minimum_free_memory_gib"]) * 1024),
        int(float(evaluation_protocol["resources"]["minimum_free_memory_gib"]) * 1024),
    )
    minimum_free_disk_bytes = MINIMUM_FREE_DISK_BYTES
    try:
        train_status = _read_run_status(TRAIN_OUTPUT)
        if train_status == "complete":
            manifest["状态"] = "发现 C3-A 训练已完成，准备多次采样诊断"
            _save_manifest(manifest)
        elif TRAIN_OUTPUT.exists():
            raise RuntimeError("C3-A 训练输出目录已存在且未完成；保留现场并停止，不覆盖或重跑")
        else:
            _run_training(manifest, minimum_free_memory_mib, minimum_free_disk_bytes)
        _run_evaluation(manifest, minimum_free_memory_mib, minimum_free_disk_bytes)
        return 0
    except KeyboardInterrupt:
        manifest["状态"] = "等待器被手动中断；未清理或覆盖任何运行结果"
        manifest["中断时间 UTC"] = _utc_now()
        _save_manifest(manifest)
        print("等待器收到中断；所有训练和评价产物均保留。", flush=True)
        return 130
    except Exception as error:
        manifest["状态"] = "执行失败；保留现场"
        manifest["错误类型"] = type(error).__name__
        manifest["错误信息"] = str(error)
        manifest["结束时间 UTC"] = _utc_now()
        _save_manifest(manifest)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
