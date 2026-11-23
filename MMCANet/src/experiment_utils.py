"""Reproducibility, split, hashing, and checkpoint helpers for MMCANet."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def canonical_json_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def resolve_manifest_artifact(
    manifest_dir: Path,
    path_text: str,
    *,
    fallback_relative: Optional[Path] = None,
) -> Path:
    """Resolve portable v2 manifest paths and historical cwd-relative paths."""

    root = manifest_dir.resolve()
    candidates: List[Path] = []
    if path_text:
        recorded = Path(path_text)
        if recorded.is_absolute():
            candidates.append(recorded)
        else:
            candidates.extend((root / recorded, recorded))
    if fallback_relative is not None:
        candidates.append(root / fallback_relative)
    seen: set[str] = set()
    attempted: List[str] = []
    for candidate in candidates:
        normalized = str(candidate.resolve(strict=False))
        if normalized in seen:
            continue
        seen.add(normalized)
        attempted.append(normalized)
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Manifest artifact not found (recorded={path_text!r}); attempted={attempted}"
    )


def hash_name_list(names: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for name in names:
        digest.update(name.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def read_split_file(path: str) -> List[str]:
    split_path = Path(path)
    if not split_path.is_file():
        raise FileNotFoundError(f"Split file not found: {path}")
    names: List[str] = []
    seen = set()
    with split_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            token = line.strip()
            if not token or token.startswith("#"):
                continue
            name = Path(token).name
            if name in seen:
                raise ValueError(f"Duplicate split entry {name!r} at {path}:{line_number}")
            seen.add(name)
            names.append(name)
    if not names:
        raise ValueError(f"Split file contains no image names: {path}")
    return names


def paired_image_names(sar_dir: str, opt_dir: str) -> Tuple[set[str], set[str], set[str]]:
    sar_root = Path(sar_dir)
    opt_root = Path(opt_dir)
    if not sar_root.is_dir():
        raise NotADirectoryError(f"SAR directory not found: {sar_dir}")
    if not opt_root.is_dir():
        raise NotADirectoryError(f"Optical directory not found: {opt_dir}")
    sar_names = {
        path.name
        for path in sar_root.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTS
    }
    opt_names = {
        path.name
        for path in opt_root.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTS
    }
    return sar_names, opt_names, sar_names & opt_names


def resolve_split_names(
    sar_dir: str,
    opt_dir: str,
    split_file: str = "",
    *,
    allow_all: bool = False,
) -> List[str]:
    sar_names, opt_names, paired = paired_image_names(sar_dir, opt_dir)
    if split_file:
        requested = read_split_file(split_file)
        missing_sar = [name for name in requested if name not in sar_names]
        missing_opt = [name for name in requested if name not in opt_names]
        if missing_sar or missing_opt:
            raise FileNotFoundError(
                "Split is not fully covered by paired data: "
                f"missing_sar={len(missing_sar)} {missing_sar[:5]}, "
                f"missing_optical={len(missing_opt)} {missing_opt[:5]}"
            )
        return requested
    if not allow_all:
        raise ValueError("A split file is required; use allow_all=True only for explicit smoke/debug runs")
    if not paired:
        raise RuntimeError("No matched SAR/optical image pairs found")
    return sorted(paired)


def assert_disjoint_splits(named_splits: Mapping[str, Sequence[str]]) -> None:
    labels = list(named_splits)
    for index, left_label in enumerate(labels):
        left = set(named_splits[left_label])
        if len(left) != len(named_splits[left_label]):
            raise ValueError(f"Split {left_label} contains duplicate names")
        for right_label in labels[index + 1 :]:
            overlap = sorted(left & set(named_splits[right_label]))
            if overlap:
                raise ValueError(
                    f"Splits {left_label} and {right_label} overlap by {len(overlap)} files: {overlap[:5]}"
                )


def seed_for_name(base_seed: int, name: str) -> int:
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return (int(base_seed) + int.from_bytes(digest[:8], "big")) % (2**63 - 1)


def seed_everything(seed: int, deterministic: bool = True) -> None:
    import torch

    if deterministic:
        # Must be set before the first CUDA/cuBLAS operation.  PyTorch's
        # deterministic-algorithm switch alone otherwise only emits a warning
        # for the batched matrix multiplications used by cross attention.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    import torch

    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def git_state(repo_root: Path) -> Dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    try:
        commit = run("rev-parse", "HEAD")
        status = run("status", "--short", "--untracked-files=all")
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"commit": "unknown", "dirty": None, "error": str(exc)}
    # The raw status can contain every untracked experiment artifact and make
    # each manifest enormous.  Commit and dirty state are sufficient for the
    # reproducibility record; callers that need the full status can run git
    # directly at the run boundary.
    return {"commit": commit, "dirty": bool(status)}


def environment_info() -> Dict[str, Any]:
    import torch

    cuda_devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            cuda_devices.append(
                {
                    "index": index,
                    "name": props.name,
                    "total_memory": props.total_memory,
                    "capability": [props.major, props.minor],
                }
            )
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_devices": cuda_devices,
        "reproducibility_environment": {
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
        },
    }


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False, default=str)
        handle.write("\n")


def checkpoint_model_state(checkpoint: Any) -> Tuple[Mapping[str, torch.Tensor], Optional[Mapping[str, Any]]]:
    """Return state dict and metadata from formal or legacy checkpoints."""

    if not isinstance(checkpoint, Mapping):
        raise ValueError("Unsupported checkpoint: expected a mapping")
    if "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
        if not isinstance(state, Mapping):
            raise ValueError("checkpoint model_state_dict is not a mapping")
        return state, checkpoint
    if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], Mapping):
        return checkpoint["state_dict"], checkpoint
    if checkpoint and all(isinstance(key, str) for key in checkpoint):
        return checkpoint, None
    raise ValueError("Unsupported checkpoint mapping")


def assert_checkpoint_model_config(
    metadata: Optional[Mapping[str, Any]],
    expected_model_config: Mapping[str, Any],
    *,
    allow_legacy: bool = False,
) -> None:
    if metadata is None:
        if allow_legacy:
            return
        raise ValueError("Legacy checkpoint has no model/config metadata; pass explicit legacy override only for audit")
    actual = metadata.get("model_config")
    if not isinstance(actual, Mapping):
        raise ValueError("Checkpoint is missing model_config metadata")
    mismatches = {
        key: {"expected": value, "actual": actual.get(key)}
        for key, value in expected_model_config.items()
        if actual.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Checkpoint/model configuration mismatch: {mismatches}")


def rng_state() -> Dict[str, Any]:
    import torch

    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # ``torch.load(..., map_location="cuda")`` also maps the serialized CPU
    # RNG tensor to CUDA.  PyTorch's RNG setters intentionally accept CPU
    # uint8 tensors only, so normalize both native tensors and legacy lists
    # before restoring.  This keeps formal checkpoint resume device-agnostic.
    torch_state = torch.as_tensor(state["torch"], dtype=torch.uint8, device="cpu").contiguous()
    torch.set_rng_state(torch_state)
    if torch.cuda.is_available() and "cuda" in state:
        cuda_states = [
            torch.as_tensor(item, dtype=torch.uint8, device="cpu").contiguous()
            for item in state["cuda"]
        ]
        torch.cuda.set_rng_state_all(cuda_states)


__all__ = [
    "SUPPORTED_IMAGE_EXTS",
    "assert_checkpoint_model_config",
    "assert_disjoint_splits",
    "canonical_json_hash",
    "checkpoint_model_state",
    "environment_info",
    "git_state",
    "hash_name_list",
    "paired_image_names",
    "read_split_file",
    "resolve_split_names",
    "resolve_manifest_artifact",
    "restore_rng_state",
    "rng_state",
    "seed_everything",
    "seed_for_name",
    "seed_worker",
    "sha256_file",
    "write_json",
]
