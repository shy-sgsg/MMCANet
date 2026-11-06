"""Strict epoch-boundary resume helpers for fixed-protocol training runs."""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Optional

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
MMCANET_SRC = ROOT / "MMCANet" / "src"
if str(MMCANET_SRC) not in sys.path:
    sys.path.insert(0, str(MMCANET_SRC))

from experiment_utils import canonical_json_hash, restore_rng_state  # noqa: E402


def build_resume_signature(arguments: Mapping[str, Any], *, resolved_device: str) -> str:
    """Hash all CLI training choices except the run location and resume path."""

    payload = {
        key: value
        for key, value in arguments.items()
        if key not in {"run_dir", "resume"}
    }
    payload["resolved_device"] = str(resolved_device)
    return canonical_json_hash(payload)


def save_checkpoint_atomic(payload: Mapping[str, Any], path: Path) -> None:
    """Replace a checkpoint only after its complete serialized bytes are ready."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, destination)


def _validate_epoch_rows(rows: Any, checkpoint_epoch: int, label: str) -> list[dict[str, Any]]:
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Resume checkpoint lacks {label}")
    normalized: list[dict[str, Any]] = []
    try:
        epochs = [int(row["epoch"]) for row in rows]
        normalized = [dict(row) for row in rows]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Resume checkpoint {label} is invalid") from exc
    if epochs != list(range(1, checkpoint_epoch + 1)):
        raise ValueError(f"Resume checkpoint {label} does not match epoch {checkpoint_epoch}")
    return normalized


def _write_rows_atomic(path: Path, rows: list[Mapping[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def restore_epoch_outputs(
    run_dir: Path,
    epoch_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    *,
    checkpoint_epoch: int,
    best_epoch: int,
    log_names: tuple[str, ...],
) -> None:
    """Rebuild summary files from checkpoint-bound records after interruption."""

    checked_rows = _validate_epoch_rows(epoch_rows, checkpoint_epoch, "epoch log rows")
    if not validation_rows or not all(isinstance(row, Mapping) for row in validation_rows):
        raise ValueError("Resume checkpoint validation rows are invalid")
    try:
        if any(int(row["epoch"]) != checkpoint_epoch for row in validation_rows):
            raise ValueError("Resume checkpoint validation rows do not match its epoch")
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Resume checkpoint validation rows do not match its epoch") from exc
    root = Path(run_dir)
    best_validation = root / "validation_per_image" / f"epoch_{best_epoch:04d}.csv"
    if best_epoch != checkpoint_epoch and not best_validation.is_file():
        raise ValueError("Resume run lacks the selected best-epoch validation table")
    for name in log_names:
        _write_rows_atomic(root / name, checked_rows)
    _write_rows_atomic(
        root / "validation_per_image" / f"epoch_{checkpoint_epoch:04d}.csv",
        [dict(row) for row in validation_rows],
    )


def ensure_best_checkpoint(run_dir: Path, *, checkpoint_epoch: int, best_epoch: int) -> None:
    """Ensure the selected checkpoint agrees with last.pth's recorded best epoch."""

    root = Path(run_dir)
    last_path = root / "checkpoints" / "last.pth"
    best_path = root / "checkpoints" / "best.pth"
    last = torch.load(last_path, map_location="cpu", weights_only=False)
    if not isinstance(last, Mapping) or int(last.get("epoch", -1)) != checkpoint_epoch:
        raise ValueError("Resume last checkpoint epoch is inconsistent")
    last_metadata = last.get("metadata")
    if not isinstance(last_metadata, Mapping) or int(last_metadata.get("best_epoch", -1)) != best_epoch:
        raise ValueError("Resume last checkpoint best epoch is inconsistent")
    selected_matches = False
    if best_path.is_file():
        selected = torch.load(best_path, map_location="cpu", weights_only=False)
        if isinstance(selected, Mapping):
            selected_metadata = selected.get("metadata")
            selected_matches = (
                int(selected.get("epoch", -1)) == best_epoch
                and isinstance(selected_metadata, Mapping)
                and int(selected_metadata.get("best_epoch", -1)) == best_epoch
                and selected_metadata.get("config_hash") == last_metadata.get("config_hash")
            )
    if selected_matches:
        return
    if best_epoch != checkpoint_epoch:
        raise ValueError("Resume selected checkpoint does not match the recorded best epoch")
    temporary = best_path.with_name(f".{best_path.name}.tmp")
    shutil.copyfile(last_path, temporary)
    os.replace(temporary, best_path)


def validate_resume_run(
    run_dir: Path,
    resume_path: Path,
    *,
    expected_config: Mapping[str, Any],
    method: str,
    seed: int,
    config_hash: str,
    resume_signature: str,
    split_hashes: Mapping[str, str],
    fixed_input_manifest_sha256: Mapping[str, str],
) -> dict[str, Any]:
    """Validate that a resume continues one incomplete run in its own folder."""

    run_root = Path(run_dir).resolve()
    checkpoint_path = Path(resume_path).resolve()
    required_checkpoint = (run_root / "checkpoints" / "last.pth").resolve()
    if checkpoint_path != required_checkpoint:
        raise ValueError("Resume must use the same run's checkpoints/last.pth")
    if not run_root.is_dir() or not checkpoint_path.is_file():
        raise FileNotFoundError("Resume requires an existing run directory and last.pth checkpoint")

    config_path = run_root / "config.json"
    manifest_path = run_root / "manifest.json"
    if not config_path.is_file() or not manifest_path.is_file():
        raise ValueError("Resume run is missing config.json or manifest.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if config != dict(expected_config):
        raise ValueError("Resume config.json does not match the requested training configuration")
    if manifest.get("config") != dict(expected_config):
        raise ValueError("Resume manifest config does not match config.json")
    if manifest.get("status") not in {"running", "failed"}:
        raise ValueError(f"Cannot resume run with status {manifest.get('status')!r}")

    expected_fields = {
        "method": method,
        "seed": int(seed),
        "config_hash": config_hash,
        "resume_signature": resume_signature,
        "split_hashes": dict(split_hashes),
        "fixed_input_manifest_sha256": dict(fixed_input_manifest_sha256),
    }
    for field, expected in expected_fields.items():
        if manifest.get(field) != expected:
            raise ValueError(f"Resume manifest {field} mismatch")
    if manifest.get("test_set_used") is not False or manifest.get("test_set_accessed") is not False:
        raise ValueError("Resume manifest must explicitly record false test-access flags")
    return manifest


def load_resume_checkpoint(
    checkpoint_path: Path,
    *,
    map_location: str | torch.device,
    method: str,
    seed: int,
    config_hash: str,
    resume_signature: str,
    split_hashes: Mapping[str, str],
    fixed_input_manifest_sha256: Mapping[str, str],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    scaler: Any,
    train_generator: torch.Generator,
    validation_generator: Optional[torch.Generator],
    ema: Optional[Any] = None,
) -> dict[str, Any]:
    """Strictly restore model, optimizer, schedulers, scaler, RNG, and loader RNG."""

    checkpoint = torch.load(Path(checkpoint_path), map_location=map_location, weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Resume checkpoint must be a mapping")
    metadata = checkpoint.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("Resume checkpoint lacks metadata")

    expected_fields = {
        "method": method,
        "seed": int(seed),
        "config_hash": config_hash,
        "resume_signature": resume_signature,
        "split_hashes": dict(split_hashes),
        "fixed_input_manifest_sha256": dict(fixed_input_manifest_sha256),
    }
    for field, expected in expected_fields.items():
        if metadata.get(field) != expected:
            raise ValueError(f"Resume checkpoint {field} mismatch")
    if metadata.get("test_set_used") is not False or metadata.get("test_set_accessed") is not False:
        raise ValueError("Resume checkpoint must explicitly record false test-access flags")
    if metadata.get("candidate_test_enumerated", False) is not False:
        raise ValueError("Resume checkpoint records candidate test enumeration")

    model_state = checkpoint.get("model")
    optimizer_state = checkpoint.get("optimizer")
    scaler_state = checkpoint.get("scaler")
    rng = checkpoint.get("rng_state")
    generator_states = checkpoint.get("dataloader_generator_states")
    if not isinstance(model_state, Mapping) or not isinstance(optimizer_state, Mapping):
        raise ValueError("Resume checkpoint lacks model or optimizer state")
    if not isinstance(scaler_state, Mapping) or not isinstance(rng, Mapping):
        raise ValueError("Resume checkpoint lacks scaler or RNG state")
    if not isinstance(generator_states, Mapping) or not isinstance(generator_states.get("train"), torch.Tensor):
        raise ValueError("Resume checkpoint lacks the train DataLoader generator state")
    validation_generator_state = generator_states.get("validation")
    if validation_generator is not None and not isinstance(validation_generator_state, torch.Tensor):
        raise ValueError("Resume checkpoint lacks the validation DataLoader generator state")
    if validation_generator is None and validation_generator_state is not None:
        raise ValueError("Resume checkpoint validation DataLoader configuration mismatch")

    scheduler_state = checkpoint.get("scheduler")
    if (scheduler is None) != (scheduler_state is None):
        raise ValueError("Resume checkpoint scheduler configuration mismatch")
    ema_state = checkpoint.get("ema")
    if (ema is None) != (ema_state is None):
        raise ValueError("Resume checkpoint EMA configuration mismatch")
    if ema is not None and not isinstance(ema_state, Mapping):
        raise ValueError("Resume checkpoint EMA state is invalid")

    epoch = checkpoint.get("epoch")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch <= 0:
        raise ValueError("Resume checkpoint epoch must be a positive integer")
    best_epoch = metadata.get("best_epoch")
    best_score = metadata.get("best_score")
    if isinstance(best_epoch, bool) or not isinstance(best_epoch, int) or not 1 <= best_epoch <= epoch:
        raise ValueError("Resume checkpoint best_epoch is invalid")
    if isinstance(best_score, bool) or not isinstance(best_score, (int, float)) or not math.isfinite(float(best_score)):
        raise ValueError("Resume checkpoint best_score is invalid")
    epoch_rows = _validate_epoch_rows(checkpoint.get("epoch_log_rows"), epoch, "epoch log rows")
    validation_rows = checkpoint.get("validation_rows")
    if not isinstance(validation_rows, list) or not validation_rows or not all(
        isinstance(row, Mapping) for row in validation_rows
    ):
        raise ValueError("Resume checkpoint validation rows are invalid")

    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)
    if scheduler is not None:
        scheduler.load_state_dict(scheduler_state)
    scaler.load_state_dict(scaler_state)
    if ema is not None:
        ema.load_state_dict(ema_state)
    train_generator.set_state(torch.as_tensor(generator_states["train"], dtype=torch.uint8, device="cpu").contiguous())
    if validation_generator is not None:
        validation_generator.set_state(
            torch.as_tensor(validation_generator_state, dtype=torch.uint8, device="cpu").contiguous()
        )
    restore_rng_state(rng)
    return {
        "start_epoch": epoch + 1,
        "best_score": float(best_score),
        "best_epoch": best_epoch,
        "epoch_rows": epoch_rows,
        "validation_rows": [dict(row) for row in validation_rows],
    }


def mark_resume_started(
    manifest: dict[str, Any],
    *,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    next_epoch: int,
    command: str,
    resumed_at_unix: Optional[float] = None,
) -> None:
    """Preserve the prior attempt details and mark the continued run active."""

    history = manifest.setdefault("resume_history", [])
    if not isinstance(history, list):
        raise ValueError("Resume manifest history must be a list")
    history.append(
        {
            "resumed_at_unix": float(time.time() if resumed_at_unix is None else resumed_at_unix),
            "from_epoch": int(next_epoch) - 1,
            "checkpoint": str(Path(checkpoint_path).resolve()),
            "checkpoint_sha256": str(checkpoint_sha256),
            "previous_status": manifest.get("status"),
            "previous_end_time_unix": manifest.get("end_time_unix"),
            "previous_error_type": manifest.get("error_type"),
            "previous_error": manifest.get("error"),
            "command": str(command),
        }
    )
    manifest["status"] = "running"
    manifest.pop("end_time_unix", None)
    manifest.pop("error_type", None)
    manifest.pop("error", None)


__all__ = [
    "build_resume_signature",
    "load_resume_checkpoint",
    "mark_resume_started",
    "ensure_best_checkpoint",
    "restore_epoch_outputs",
    "save_checkpoint_atomic",
    "validate_resume_run",
]
