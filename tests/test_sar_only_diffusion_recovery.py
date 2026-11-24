from __future__ import annotations

import pytest

from experiments.published_baselines_sar_only_20261123_v1 import continue_diffusion_run as recovery
from experiments.published_baselines_sar_only_20261123_v1.continue_diffusion_run import (
    _parse_training_history,
)
from experiments.published_baselines_sar_only_20261123_v1.train_sar_only_diffusion import (
    METRIC_NAMES,
    _source_group_statistics,
)


def test_从控制台日志重建连续训练轮次(tmp_path):
    log_path = tmp_path / "训练进度.log"
    log_path.write_text(
        "conditional 第 01/30 轮：训练 epsilon MSE=0.500000，开发组均衡 MSE=0.450000，峰值显存=520.0 MiB，耗时=101.0 秒\n"
        "residual 第 01/30 轮：训练 epsilon MSE=0.300000，开发组均衡 MSE=0.280000，峰值显存=530.0 MiB，耗时=102.0 秒\n"
        "residual 第 02/30 轮：训练 epsilon MSE=0.250000，开发组均衡 MSE=0.240000，峰值显存=531.0 MiB，耗时=103.0 秒\n",
        encoding="utf-8",
    )

    rows = _parse_training_history(log_path, "residual", expected_epoch=2, total_epochs=30)

    assert [int(row["epoch"]) for row in rows] == [1, 2]
    assert rows[1]["train_epsilon_mse"] == pytest.approx(0.25)
    assert rows[1]["development_group_balanced_epsilon_mse"] == pytest.approx(0.24)
    assert rows[1]["peak_allocated_mib"] == pytest.approx(531.0)
    assert rows[1]["peak_reserved_mib"] == ""
    assert rows[1]["epoch_seconds"] == pytest.approx(103.0)


def test_历史训练轮次缺失时拒绝静默补齐(tmp_path):
    log_path = tmp_path / "训练进度.log"
    log_path.write_text(
        "residual 第 01/30 轮：训练 epsilon MSE=0.300000，开发组均衡 MSE=0.280000，峰值显存=530.0 MiB，耗时=102.0 秒\n"
        "residual 第 03/30 轮：训练 epsilon MSE=0.250000，开发组均衡 MSE=0.240000，峰值显存=531.0 MiB，耗时=103.0 秒\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="轮次不连续"):
        _parse_training_history(log_path, "residual", expected_epoch=3, total_epochs=30)


def test_来源组汇总接受从CSV读取的数值文本():
    metric_values = {metric: "0.5" for metric in METRIC_NAMES}
    mean_rows = [
        {
            "method": "conditional",
            "source_group_id": "来源组甲",
            **metric_values,
            "mean_pixelwise_sample_std": "0.1",
            "draw_pair_mean_absolute_difference": "0.2",
            "output_clipped_fraction": "0.01",
        }
    ]
    draw_rows = [
        {
            "method": "conditional",
            "source_group_id": "来源组甲",
            **metric_values,
            "output_clipped_fraction": "0.01",
        }
    ]
    distribution_rows = [
        {
            "method": "conditional",
            "source_group_id": "来源组甲",
            "pixelwise_crps": "0.3",
            "normalized_image_energy_score": "0.4",
            "adjacent_variogram_score": "0.5",
        }
    ]

    result = _source_group_statistics(draw_rows, mean_rows, distribution_rows)

    assert result[0]["mean_pixelwise_sample_std"] == pytest.approx(0.1)
    assert result[0]["draw_pair_mean_absolute_difference"] == pytest.approx(0.2)
    assert result[0]["output_clipped_fraction_mean"] == pytest.approx(0.01)


def test_finalizer从冻结配置读取SwinIR结果目录(tmp_path):
    run_dir = tmp_path / "扩散运行"
    reference_dir = tmp_path / "SwinIR运行"
    run_dir.mkdir()
    (run_dir / "config.json").write_text(
        '{"swinir_run": "' + str(reference_dir) + '"}', encoding="utf-8"
    )

    resolve = getattr(recovery, "_swinir_reference_run", lambda _: None)

    assert resolve(run_dir) == reference_dir
