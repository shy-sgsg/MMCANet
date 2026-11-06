from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

from scripts import run_phase_c3a_sar_flow as runner
from scripts.training_resume import save_checkpoint_atomic


def _require_helper(name: str):
    helper = getattr(runner, name, None)
    assert callable(helper), f"C3-A runner 缺少续训合同函数 {name}"
    return helper


def _binding() -> dict[str, Any]:
    return {
        "source_manifest_sha256": "source-hash",
        "fixed_input_records": {
            role: {
                "manifest_sha256": f"{role}-manifest",
                "files_csv_sha256": f"{role}-files",
                "sample_count": 1,
                "source_group_count": 1,
                "payload_arrays_read": ["lr", "target"],
            }
            for role in ("train", "development")
        },
        "code_sha256": {"training_entry": "code-a", "Flow": "code-b"},
        "environment_signature": {"Python": "3.x", "PyTorch": "2.x", "device": "cpu"},
        "training_options": {"seed": 17, "epochs": 2, "batch_size": 2},
    }


def _input_manifest(binding: dict[str, Any]) -> dict[str, Any]:
    return {
        "protocol_sha256": "protocol-hash",
        "source_manifest_path": "protocols/mmcanet_phase_c1/v2/source_group_split_20261028_v1/sample_manifest.csv",
        "source_manifest_sha256": binding["source_manifest_sha256"],
        "resume_binding": binding,
        "roles": binding["fixed_input_records"],
        "payload_access": {"train": True, "development": True},
        "confirmation_payload_accessed": False,
        "test_set_accessed": False,
        "test_set_used": False,
        "optical_payload_read": False,
    }


def _train_one_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loader_generator: torch.Generator,
    flow_generator: torch.Generator,
) -> None:
    sample_index = torch.randperm(9, generator=loader_generator)[0].float()
    time_value = torch.rand((), generator=flow_generator)
    flow_noise = torch.randn(2, generator=flow_generator)
    stochastic_offset = random.random() + float(np.random.random()) + float(torch.rand(()))
    inputs = torch.stack((sample_index, time_value, flow_noise[0]))
    prediction = model(inputs)
    target = flow_noise[1] + stochastic_offset
    loss = (prediction.reshape(()) - target).square()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


def _make_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loader_generator: torch.Generator,
    flow_generator: torch.Generator,
    *,
    epoch: int = 1,
    best_epoch: int = 1,
    best_score: float = 1.0,
    protocol_sha256: str = "protocol-hash",
    binding: dict[str, Any] | None = None,
) -> dict[str, Any]:
    helper = _require_helper("_make_checkpoint")
    return helper(
        model,
        optimizer,
        epoch=epoch,
        score=best_score,
        protocol_sha256=protocol_sha256,
        loader_generator=loader_generator,
        flow_generator=flow_generator,
        best_score=best_score,
        best_epoch=best_epoch,
        train_log=[{"轮次": index, "训练速度场 MSE": 1.0 / index} for index in range(1, epoch + 1)],
        development_log=[{"轮次": epoch, "样本 ID": "合成样本"}],
        resume_binding=binding or _binding(),
    )


def _restore(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loader_generator: torch.Generator,
    flow_generator: torch.Generator,
    *,
    protocol_sha256: str = "protocol-hash",
    binding: dict[str, Any] | None = None,
    total_epochs: int = 2,
) -> dict[str, Any]:
    helper = _require_helper("_restore_checkpoint")
    return helper(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        loader_generator=loader_generator,
        flow_generator=flow_generator,
        protocol_sha256=protocol_sha256,
        resume_binding=binding or _binding(),
        total_epochs=total_epochs,
    )


def test_resume_matches_uninterrupted_cpu_training(tmp_path: Path) -> None:
    random.seed(41)
    np.random.seed(41)
    torch.manual_seed(41)
    source_model = nn.Linear(3, 1)
    source_optimizer = torch.optim.Adam(source_model.parameters(), lr=0.01)
    source_loader_generator = torch.Generator().manual_seed(101)
    source_flow_generator = torch.Generator().manual_seed(202)

    _train_one_step(source_model, source_optimizer, source_loader_generator, source_flow_generator)
    checkpoint = _make_checkpoint(
        source_model,
        source_optimizer,
        source_loader_generator,
        source_flow_generator,
    )
    checkpoint_path = tmp_path / "last.pt"
    save_checkpoint_atomic(checkpoint, checkpoint_path)
    _train_one_step(source_model, source_optimizer, source_loader_generator, source_flow_generator)
    expected_model = copy.deepcopy(source_model.state_dict())
    expected_optimizer = copy.deepcopy(source_optimizer.state_dict())
    expected_loader_draw = torch.rand(4, generator=source_loader_generator)
    expected_flow_draw = torch.rand(4, generator=source_flow_generator)

    torch.manual_seed(999)
    resumed_model = nn.Linear(3, 1)
    resumed_optimizer = torch.optim.Adam(resumed_model.parameters(), lr=0.01)
    resumed_loader_generator = torch.Generator().manual_seed(9991)
    resumed_flow_generator = torch.Generator().manual_seed(9992)
    resume_state = _restore(
        checkpoint_path,
        resumed_model,
        resumed_optimizer,
        resumed_loader_generator,
        resumed_flow_generator,
    )
    assert resume_state["start_epoch"] == 2
    assert resume_state["best_epoch"] == 1
    assert resume_state["train_log"] == [{"轮次": 1, "训练速度场 MSE": 1.0}]
    assert resume_state["development_log"] == [{"轮次": 1, "样本 ID": "合成样本"}]

    _train_one_step(resumed_model, resumed_optimizer, resumed_loader_generator, resumed_flow_generator)
    assert all(torch.equal(value, expected_model[name]) for name, value in resumed_model.state_dict().items())
    assert resumed_optimizer.state_dict()["param_groups"] == expected_optimizer["param_groups"]
    for parameter_id, state in resumed_optimizer.state_dict()["state"].items():
        expected_state = expected_optimizer["state"][parameter_id]
        assert all(
            torch.equal(value, expected_state[key]) if isinstance(value, torch.Tensor) else value == expected_state[key]
            for key, value in state.items()
        )
    assert torch.equal(torch.rand(4, generator=resumed_loader_generator), expected_loader_draw)
    assert torch.equal(torch.rand(4, generator=resumed_flow_generator), expected_flow_draw)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize(
    "field", ["fixed_input_records", "code_sha256", "environment_signature", "training_options"]
)
def test_resume_rejects_changed_binding_before_state_restore(tmp_path: Path, field: str) -> None:
    torch.manual_seed(7)
    model = nn.Linear(3, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loader_generator = torch.Generator().manual_seed(1)
    flow_generator = torch.Generator().manual_seed(2)
    checkpoint = _make_checkpoint(model, optimizer, loader_generator, flow_generator)
    path = tmp_path / "last.pt"
    save_checkpoint_atomic(checkpoint, path)

    target_model = nn.Linear(3, 1)
    before = copy.deepcopy(target_model.state_dict())
    target_optimizer = torch.optim.Adam(target_model.parameters(), lr=0.01)
    wrong_binding = _binding()
    wrong_binding[field][next(iter(wrong_binding[field]))] = "changed"
    with pytest.raises(ValueError, match="绑定"):
        _restore(
            path,
            target_model,
            target_optimizer,
            torch.Generator().manual_seed(3),
            torch.Generator().manual_seed(4),
            binding=wrong_binding,
        )
    assert all(torch.equal(value, before[name]) for name, value in target_model.state_dict().items())


def test_resume_rejects_wrong_protocol_hash(tmp_path: Path) -> None:
    model = nn.Linear(3, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loader_generator = torch.Generator().manual_seed(1)
    flow_generator = torch.Generator().manual_seed(2)
    path = tmp_path / "last.pt"
    save_checkpoint_atomic(
        _make_checkpoint(model, optimizer, loader_generator, flow_generator), path
    )

    target_model = nn.Linear(3, 1)
    with pytest.raises(ValueError, match="协议"):
        _restore(
            path,
            target_model,
            torch.optim.Adam(target_model.parameters(), lr=0.01),
            torch.Generator().manual_seed(3),
            torch.Generator().manual_seed(4),
            protocol_sha256="other-protocol",
        )


def test_resume_rejects_missing_generator_state(tmp_path: Path) -> None:
    model = nn.Linear(3, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loader_generator = torch.Generator().manual_seed(1)
    flow_generator = torch.Generator().manual_seed(2)
    checkpoint = _make_checkpoint(model, optimizer, loader_generator, flow_generator)
    checkpoint.pop("flow_generator_state", None)
    path = tmp_path / "last.pt"
    save_checkpoint_atomic(checkpoint, path)

    with pytest.raises(ValueError, match="随机生成器"):
        _restore(
            path,
            nn.Linear(3, 1),
            torch.optim.Adam(nn.Linear(3, 1).parameters(), lr=0.01),
            torch.Generator().manual_seed(3),
            torch.Generator().manual_seed(4),
        )


def test_resume_rejects_epoch_beyond_protocol(tmp_path: Path) -> None:
    model = nn.Linear(3, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    path = tmp_path / "last.pt"
    checkpoint = _make_checkpoint(
        model,
        optimizer,
        torch.Generator().manual_seed(1),
        torch.Generator().manual_seed(2),
        epoch=3,
        best_epoch=2,
    )
    save_checkpoint_atomic(checkpoint, path)

    with pytest.raises(ValueError, match="轮次"):
        _restore(
            path,
            nn.Linear(3, 1),
            torch.optim.Adam(nn.Linear(3, 1).parameters(), lr=0.01),
            torch.Generator().manual_seed(3),
            torch.Generator().manual_seed(4),
            total_epochs=2,
        )


@pytest.mark.parametrize("status", ["complete", "not_started"])
def test_resume_manifest_requires_incomplete_run(status: str) -> None:
    validate = _require_helper("_validate_resume_manifest")
    manifest = {
        "format": "mmcanet_c3a_sar_only_flow_run_v2",
        "status": status,
        "protocol_sha256": "protocol-hash",
        "resume_binding": _binding(),
        "fixed_input_records": _binding()["fixed_input_records"],
        "c1_source_manifest_sha256": _binding()["source_manifest_sha256"],
        "payload_access": {"train": True, "development": True},
        "confirmation_payload_accessed": False,
        "test_set_accessed": False,
        "test_set_used": False,
        "optical_payload_read": False,
    }
    input_manifest = _input_manifest(_binding())
    with pytest.raises(ValueError, match="状态"):
        validate(manifest, input_manifest, protocol_sha256="protocol-hash", resume_binding=_binding())


def test_resume_manifest_rejects_protected_payload_access() -> None:
    validate = _require_helper("_validate_resume_manifest")
    manifest = {
        "format": "mmcanet_c3a_sar_only_flow_run_v2",
        "status": "failed",
        "protocol_sha256": "protocol-hash",
        "resume_binding": _binding(),
        "fixed_input_records": _binding()["fixed_input_records"],
        "c1_source_manifest_sha256": _binding()["source_manifest_sha256"],
        "confirmation_payload_accessed": False,
        "test_set_accessed": False,
        "test_set_used": False,
        "optical_payload_read": False,
        "payload_access": {"train": True, "development": True},
    }
    input_manifest = _input_manifest(_binding())
    input_manifest["optical_payload_read"] = True
    with pytest.raises(ValueError, match="光学"):
        validate(manifest, input_manifest, protocol_sha256="protocol-hash", resume_binding=_binding())


def test_resume_cli_is_explicit_and_default_remains_fresh_run() -> None:
    parse = _require_helper("_parse_args")
    assert parse([]).resume is False
    assert parse(["--resume"]).resume is True


def test_existing_output_requires_explicit_resume(tmp_path: Path) -> None:
    validate = _require_helper("_validate_output_mode")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with pytest.raises(FileExistsError):
        validate(run_dir, resume=False)
    validate(run_dir, resume=True)
    with pytest.raises(FileNotFoundError):
        validate(tmp_path / "missing", resume=True)


def test_run_manifest_json_is_replaced_atomically(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    replacements: list[tuple[Path, Path]] = []
    original_replace = runner.os.replace

    def record_replace(source: str | Path, destination: str | Path) -> None:
        replacements.append((Path(source), Path(destination)))
        original_replace(source, destination)

    monkeypatch.setattr(runner.os, "replace", record_replace)
    destination = tmp_path / "manifest.json"
    runner._json_write(destination, {"状态": "运行中"})

    assert json.loads(destination.read_text(encoding="utf-8")) == {"状态": "运行中"}
    assert replacements == [(tmp_path / ".manifest.json.tmp", destination)]
    assert not (tmp_path / ".manifest.json.tmp").exists()


def test_resume_selects_newest_valid_best_or_last_checkpoint(tmp_path: Path) -> None:
    select = _require_helper("_select_resume_checkpoint")
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    model = nn.Linear(3, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loader_generator = torch.Generator().manual_seed(1)
    flow_generator = torch.Generator().manual_seed(2)
    save_checkpoint_atomic(
        _make_checkpoint(
            model,
            optimizer,
            loader_generator,
            flow_generator,
            epoch=1,
            best_epoch=1,
            best_score=0.5,
        ),
        checkpoint_dir / "last.pt",
    )
    save_checkpoint_atomic(
        _make_checkpoint(
            model,
            optimizer,
            loader_generator,
            flow_generator,
            epoch=2,
            best_epoch=2,
            best_score=0.6,
        ),
        checkpoint_dir / "best.pt",
    )

    selected_path, selected = select(
        checkpoint_dir,
        protocol_sha256="protocol-hash",
        resume_binding=_binding(),
        total_epochs=2,
    )

    assert selected_path == checkpoint_dir / "last.pt"
    assert selected["epoch"] == 2
    restored_last = torch.load(checkpoint_dir / "last.pt", map_location="cpu", weights_only=True)
    assert restored_last["epoch"] == 2
