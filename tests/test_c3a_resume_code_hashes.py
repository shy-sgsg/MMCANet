from pathlib import Path

from scripts.run_phase_c3a_sar_flow import _resume_code_hashes, _resume_environment_signature


def test_resume_code_hashes_cover_metric_and_baseline_model_dependencies():
    hashes = _resume_code_hashes()
    expected = {
        "MMCANetV2/phase_c3a_metrics.py",
        "MMCANetV2/phase_c2_runner.py",
        "MMCANetV2/phase_c2_matched.py",
        "MMCANet/src/experiment_utils.py",
        "JointJDSR/model.py",
        "scripts/training_resume.py",
    }

    assert expected <= set(hashes)
    assert all(Path(path).is_file() and len(digest) == 64 for path, digest in hashes.items())


def test_resume_binding_captures_cublas_determinism_configuration():
    signature = _resume_environment_signature({"cublas_workspace_config": ":4096:8"})

    assert signature["cublas_workspace_config"] == ":4096:8"
