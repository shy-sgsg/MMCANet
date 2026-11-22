from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments/data_audits/c1_v2_train_development_source_crosswalk_20261107_v1/audit_source_crosswalk.py"
)
SPEC = importlib.util.spec_from_file_location("audit_source_crosswalk", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def test_existing_results_abort_before_reading_audit_inputs(tmp_path, monkeypatch):
    old_result = '{"历史结果": true}\n'
    (tmp_path / "results.json").write_text(old_result, encoding="utf-8")
    monkeypatch.setattr(AUDIT, "OUT", tmp_path)

    def fail_if_input_is_read(_path):
        raise AssertionError("发现已有结果后不应再读取审计输入")

    monkeypatch.setattr(AUDIT, "load_json", fail_if_input_is_read)

    with pytest.raises(FileExistsError, match="拒绝覆盖"):
        AUDIT.main()

    assert (tmp_path / "results.json").read_text(encoding="utf-8") == old_result
