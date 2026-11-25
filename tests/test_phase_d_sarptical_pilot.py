"""SARptical 有限空间 pilot 的隔离、数据和评价契约。"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest
from scipy.io import savemat

from experiments.phase_d_sarptical_spatial_pilot_20261125 import pilot_data


GROUP_COUNTS = [
    4, 31, 7, 2, 2, 126, 6, 155, 3, 15, 10, 4, 30, 34,
    98, 67, 4, 12, 89, 134, 78, 385, 66, 19, 84, 29, 15, 19,
]


def make_archive(path: Path) -> tuple[Path, dict[str, list[str]]]:
    """建立只含坐标元数据和空 SAR 成员的轻量测试归档。"""
    rows = []
    members = {}
    sample_id = 1000
    for index, count in enumerate(GROUP_COUNTS):
        grid_x, grid_y = index, 0
        group_ids = []
        for _ in range(count):
            rows.append([sample_id, grid_x * 250 + 100, grid_y * 250 + 100, 0])
            group_ids.append(f"point_{sample_id}")
            sample_id += 1
        members[f"{grid_x}_{grid_y}"] = group_ids
    with ZipFile(path, "w") as archive:
        archive.writestr("selectedPntXYZ.mat", savemat_bytes(np.asarray(rows, dtype=np.float64)))
        for group_ids in members.values():
            for name in group_ids:
                archive.writestr(
                    f"patch_SAR_OPT_SQUARE/{name}_ampPatch.mat", b""
                )
    return path, members


def savemat_bytes(values: np.ndarray) -> bytes:
    from io import BytesIO

    buffer = BytesIO()
    savemat(buffer, {"selectedPntXYZ": values})
    return buffer.getvalue()


def test_split_is_reproducible_disjoint_and_has_frozen_counts(tmp_path: Path) -> None:
    archive_path, _ = make_archive(tmp_path / "metadata_only.zip")

    first = pilot_data.build_spatial_split(archive_path, seed=187762)
    second = pilot_data.build_spatial_split(archive_path, seed=187762)

    assert first == second
    assert first["role_group_counts"] == {
        "train": 12,
        "development": 8,
        "confirmation": 8,
    }
    assert first["role_sample_counts"] == {
        "train": 916,
        "development": 306,
        "confirmation": 306,
    }
    group_roles = [row["role"] for row in first["groups"]]
    assert Counter(group_roles) == Counter({"train": 12, "development": 8, "confirmation": 8})
    sample_roles = [row["role"] for row in first["samples"]]
    assert Counter(sample_roles) == Counter({"train": 916, "development": 306, "confirmation": 306})
    assert len({row["sample_id"] for row in first["samples"]}) == 1528
    assert first["confirmation_payload_accessed"] is False
    assert first["test_set_accessed"] is False
    assert first["optical_payload_loaded"] is False


@pytest.mark.parametrize("protected_role", ["confirmation", "test", "optical"])
def test_protected_roles_are_rejected_before_archive_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, protected_role: str
) -> None:
    archive_path, _ = make_archive(tmp_path / "metadata_only.zip")
    split = pilot_data.build_spatial_split(archive_path, seed=187762)

    def forbidden_open(*args, **kwargs):
        raise AssertionError("受保护角色在拒绝前打开了 ZIP")

    monkeypatch.setattr(pilot_data.zipfile, "ZipFile", forbidden_open)
    with pytest.raises(ValueError, match="禁止读取"):
        pilot_data.load_sar_matrix(archive_path, "point_1000", protected_role, split)


def test_confirmation_sample_cannot_be_read_under_another_role(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive_path, _ = make_archive(tmp_path / "metadata_only.zip")
    split = pilot_data.build_spatial_split(archive_path, seed=187762)
    confirmation_id = next(row["sample_id"] for row in split["samples"] if row["role"] == "confirmation")

    def forbidden_open(*args, **kwargs):
        raise AssertionError("样本角色检查前打开了 ZIP")

    monkeypatch.setattr(pilot_data.zipfile, "ZipFile", forbidden_open)
    with pytest.raises(ValueError, match="角色不匹配"):
        pilot_data.load_sar_matrix(archive_path, confirmation_id, "train", split)


def test_mat_parser_accepts_only_one_finite_112_square_matrix() -> None:
    from io import BytesIO

    single = BytesIO()
    savemat(single, {"ampCrop": np.ones((112, 112), dtype=np.float32)})
    parsed = pilot_data.parse_sar_mat_payload(single.getvalue(), "point_1")
    assert parsed.shape == (112, 112)
    assert parsed.dtype == np.float32
    assert np.isfinite(parsed).all()

    wrong_shape = BytesIO()
    savemat(wrong_shape, {"ampCrop": np.ones((111, 112), dtype=np.float32)})
    with pytest.raises(ValueError, match="112×112"):
        pilot_data.parse_sar_mat_payload(wrong_shape.getvalue(), "point_2")

    ambiguous = BytesIO()
    savemat(
        ambiguous,
        {
            "ampCrop": np.ones((112, 112), dtype=np.float32),
            "secondImage": np.zeros((112, 112), dtype=np.float32),
        },
    )
    with pytest.raises(ValueError, match="歧义"):
        pilot_data.parse_sar_mat_payload(ambiguous.getvalue(), "point_3")

    nonfinite = np.ones((112, 112), dtype=np.float32)
    nonfinite[0, 0] = np.nan
    invalid = BytesIO()
    savemat(invalid, {"ampCrop": nonfinite})
    with pytest.raises(ValueError, match="非有限"):
        pilot_data.parse_sar_mat_payload(invalid.getvalue(), "point_4")


def test_pair_preparation_uses_shared_linear_statistics_and_fixed_x2_shape() -> None:
    target = np.linspace(-5.0, 7.0, 112 * 112, dtype=np.float32).reshape(112, 112)
    lr, normalized_target, anchor = pilot_data.prepare_pair_arrays(target, mean=1.25, std=2.5)

    assert lr.shape == (1, 56, 56)
    assert normalized_target.shape == (1, 112, 112)
    assert anchor.shape == (1, 112, 112)
    assert all(array.dtype == np.float32 for array in (lr, normalized_target, anchor))
    assert all(np.isfinite(array).all() for array in (lr, normalized_target, anchor))
    assert np.allclose(normalized_target[0], (target - 1.25) / 2.5, atol=1e-6)


def test_flow_bridge_pair_uses_frozen_linear_path() -> None:
    import torch

    from experiments.phase_d_sarptical_spatial_pilot_20261125.run_pilot import flow_bridge_pair

    target = torch.full((2, 1, 16, 16), 0.25)
    anchor = torch.full_like(target, -0.1)
    noise = torch.arange(target.numel(), dtype=torch.float32).reshape_as(target) / 1000
    time = torch.tensor([0.2, 0.7], dtype=torch.float32)
    state, velocity = flow_bridge_pair(target, anchor, time, noise, sigma=0.5)
    source = anchor + 0.5 * noise
    fraction = time[:, None, None, None]
    assert torch.allclose(state, (1 - fraction) * source + fraction * target)
    assert torch.allclose(velocity, target - source)


@pytest.mark.parametrize("method", ["edsr", "swinir", "flow_bridge"])
def test_models_obey_single_channel_linear_x2_contract(method: str) -> None:
    import torch

    from experiments.phase_d_sarptical_spatial_pilot_20261125.run_pilot import build_model

    model = build_model(method)
    model.eval()
    lr = torch.randn(1, 1, 16, 16)
    with torch.inference_mode():
        if method == "flow_bridge":
            state = torch.randn(1, 1, 32, 32)
            output = model(state, torch.randn_like(state), torch.tensor([0.5]))
            assert tuple(output.shape) == tuple(state.shape)
        else:
            output = model(lr)
            assert tuple(output.shape) == (1, 1, 32, 32)
            assert model.output.__class__.__name__ == "Identity"
    assert torch.isfinite(output).all()


def test_training_batch_updates_parameters_and_schedule_repeats() -> None:
    import torch
    from torch import nn

    from experiments.phase_d_sarptical_spatial_pilot_20261125.run_pilot import (
        make_epoch_schedule,
        train_batch,
    )

    class TinyX2(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(0.0))

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.interpolate(value, scale_factor=2, mode="nearest") + self.scale

    model = TinyX2()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    lr = torch.ones(2, 1, 8, 8)
    target = torch.full((2, 1, 16, 16), 2.0)
    anchor = torch.nn.functional.interpolate(lr, scale_factor=2, mode="nearest")
    before = model.scale.detach().clone()
    result = train_batch("edsr", model, optimizer, lr, target, anchor)
    assert result["loss"] > 0
    assert torch.isfinite(torch.tensor(result["gradient_norm"]))
    assert not torch.equal(before, model.scale.detach())

    records = [
        {"sample_id": f"point_{index}", "lr": np.zeros((1, 56, 56), np.float32)}
        for index in range(8)
    ]
    schedule_a = make_epoch_schedule(records, seed=20261125, patch_lr=48)
    schedule_b = make_epoch_schedule(records, seed=20261125, patch_lr=48)
    assert schedule_a == schedule_b


def test_script_entrypoint_runs_after_training_helpers_are_defined() -> None:
    import ast

    source_path = Path(
        "experiments/phase_d_sarptical_spatial_pilot_20261125/run_pilot.py"
    )
    module = ast.parse(source_path.read_text(encoding="utf-8"))
    train_method = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "train_method"
    )
    entrypoint = next(
        node
        for node in module.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
        and any(
            isinstance(comparator, ast.Constant) and comparator.value == "__main__"
            for comparator in node.test.comparators
        )
    )

    assert train_method.lineno < entrypoint.lineno


def _evaluation_fixture(tmp_path: Path):
    from experiments.phase_d_sarptical_spatial_pilot_20261125.run_pilot import write_prediction_manifest

    yy, xx = np.mgrid[0:112, 0:112]
    target = (xx + yy).astype(np.float32) / 112.0
    record = {
        "sample_id": "point_1",
        "source_group_id": "group_a",
        "lr": np.zeros((1, 56, 56), dtype=np.float32),
        "target": target[None],
        "anchor": target[None],
    }
    prediction_dir = tmp_path / "predictions"
    prediction_dir.mkdir()
    np.save(prediction_dir / "point_1.npy", target + 0.01, allow_pickle=False)
    manifest = write_prediction_manifest(prediction_dir, ["point_1"])
    config = {
        "data_range": 4.0,
        "high_return_threshold": 0.7,
        "background_threshold": 0.3,
        "edge_threshold": 0.1,
    }
    return [record], prediction_dir, manifest, config


def test_evaluator_reports_known_metrics_and_reprojection(tmp_path: Path) -> None:
    from experiments.phase_d_sarptical_spatial_pilot_20261125.run_pilot import evaluate_saved_predictions

    records, prediction_dir, manifest, config = _evaluation_fixture(tmp_path)
    rows = evaluate_saved_predictions(records, prediction_dir, manifest, config, ["point_1"])
    assert len(rows) == 1
    row = rows[0]
    assert row["psnr_db"] > 40
    assert 0.99 <= row["ssim"] <= 1.0
    assert row["rmse"] > 0
    assert row["epi_corr"] > 0.95
    assert row["lr_reprojection_rmse"] > 0


@pytest.mark.parametrize("failure", ["missing", "extra", "wrong_shape", "nan", "order"])
def test_evaluator_fails_closed_on_prediction_contract(
    tmp_path: Path, failure: str
) -> None:
    from experiments.phase_d_sarptical_spatial_pilot_20261125.run_pilot import evaluate_saved_predictions

    records, prediction_dir, manifest, config = _evaluation_fixture(tmp_path)
    if failure == "missing":
        (prediction_dir / "point_1.npy").unlink()
    elif failure == "extra":
        np.save(prediction_dir / "point_2.npy", np.zeros((112, 112), np.float32))
    elif failure == "wrong_shape":
        np.save(prediction_dir / "point_1.npy", np.zeros((111, 112), np.float32))
    elif failure == "nan":
        invalid = np.zeros((112, 112), dtype=np.float32)
        invalid[0, 0] = np.nan
        np.save(prediction_dir / "point_1.npy", invalid)
    elif failure == "order":
        with manifest.open("w", encoding="utf-8") as stream:
            stream.write('{"sample_ids": ["point_2", "point_1"]}')

    with pytest.raises(ValueError):
        evaluate_saved_predictions(records, prediction_dir, manifest, config, ["point_1"])


def test_group_balanced_mean_and_bootstrap_are_deterministic() -> None:
    from experiments.phase_d_sarptical_spatial_pilot_20261125.run_pilot import (
        group_balanced_mean,
        paired_group_bootstrap,
    )

    reference = [
        {"sample_id": "a", "source_group_id": "g1", "psnr_db": 10.0},
        {"sample_id": "b", "source_group_id": "g1", "psnr_db": 12.0},
        {"sample_id": "c", "source_group_id": "g2", "psnr_db": 20.0},
        {"sample_id": "d", "source_group_id": "g2", "psnr_db": 22.0},
    ]
    candidate = [{**row, "psnr_db": row["psnr_db"] + 1.0} for row in reference]
    assert group_balanced_mean(reference, "psnr_db") == pytest.approx(16.0)
    first = paired_group_bootstrap(candidate, reference, "psnr_db", seed=7, draws=1000)
    second = paired_group_bootstrap(candidate, reference, "psnr_db", seed=7, draws=1000)
    assert first == second
    assert first["group_balanced_delta"] == pytest.approx(1.0)
    assert first["ci95_lower"] == pytest.approx(1.0)
    assert first["ci95_upper"] == pytest.approx(1.0)


def test_missing_guardrail_roi_is_reported_and_fails_closed(tmp_path: Path) -> None:
    from experiments.phase_d_sarptical_spatial_pilot_20261125.run_pilot import (
        _research_decision,
        summarize_comparisons,
        write_csv,
        write_decision_review,
    )

    reference_rows = []
    flow_rows = []
    for index in range(8):
        common = {"sample_id": f"sample_{index}", "source_group_id": f"group_{index}"}
        reference_rows.append(
            {
                **common,
                "psnr_db": 20.0,
                "ssim": 0.5,
                "epi_corr": 0.4,
                "rmse": 0.5,
                "bright_mean_abs_error": "NA",
                "bright_abs_error_p95": "NA",
                "edge_strength_abs_error": 0.2,
                "background_abs_bias": 0.2,
            }
        )
        flow_rows.append(
            {
                **common,
                "psnr_db": 20.2,
                "ssim": 0.51,
                "epi_corr": 0.41,
                "rmse": 0.4,
                "bright_mean_abs_error": "NA",
                "bright_abs_error_p95": "NA",
                "edge_strength_abs_error": 0.1,
                "background_abs_bias": 0.1,
            }
        )

    decision = _research_decision(
        {"swinir": reference_rows, "flow_bridge_single_draw": flow_rows},
        {
            "swinir": {"inference_seconds_total": 1.0},
            "flow_bridge_draw_00": {"inference_seconds_total": 10.0},
            **{
                f"flow_bridge_draw_{index:02d}": {"inference_seconds_total": 10.0}
                for index in range(1, 4)
            },
        },
        "swinir",
    )

    bright_guard = decision["guardrails"]["bright_mean_abs_error"]
    assert bright_guard["group_count"] == 0
    assert bright_guard["group_balanced_delta"] is None
    assert bright_guard["ci95_lower"] is None
    assert bright_guard["passed"] is False
    assert decision["conditions"]["brightness_structure_background_guardrails"] is False
    assert decision["decision"] == "STOP_CURRENT_FLOW_BRIDGE_CONFIGURATION"

    comparisons = summarize_comparisons(
        {"swinir": reference_rows, "flow_bridge_single_draw": flow_rows},
        reference_method="swinir",
    )
    bright_comparison = next(
        row for row in comparisons if row["metric"] == "bright_mean_abs_error"
    )
    assert bright_comparison["group_count"] == 0
    assert bright_comparison["group_balanced_delta"] is None
    comparison_path = tmp_path / "comparisons.csv"
    write_csv(comparison_path, comparisons)
    assert "unavailable_reason" in comparison_path.read_text(encoding="utf-8-sig")

    report_rows = {
        "bicubic": reference_rows,
        "edsr": reference_rows,
        "swinir": reference_rows,
        "flow_bridge_single_draw": flow_rows,
        "flow_bridge_pixel_mean": flow_rows,
    }
    report_path = tmp_path / "review.md"
    write_decision_review(
        report_path,
        decision,
        {"run_directory": str(tmp_path)},
        report_rows,
    )
    report = report_path.read_text(encoding="utf-8")
    assert "有效来源组 0/8" in report
    assert "因此该护栏未通过" in report


def test_finalization_cli_can_start_from_repository_root() -> None:
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "experiments/phase_d_sarptical_spatial_pilot_20261125/finalize_saved_run.py",
            "--help",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "恢复 SARptical pilot" in result.stdout
