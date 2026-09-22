"""Prospective correction binding uses public synthetic checkouts only."""

import importlib
import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pytest

from automated_phishing_detection import development_execution, execution_preflight

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def api():
    path = ROOT / "src/automated_phishing_detection/development_correction.py"
    assert path.exists(), "missing prospective correction binding"
    return importlib.import_module(
        "automated_phishing_detection.development_correction"
    )


def test_correction_policy_preserves_history_and_limits_new_fits(api):
    profile = json.loads((ROOT / api.PROFILE_PATH).read_bytes())
    api.validate_profile(profile)
    assert profile["new_fits"] == ["random_forest"]
    assert profile["maximum_new_fits"] == 1
    assert profile["stages"] == ["retained_audit", "random_forest"]
    assert profile["protected_evaluation_ready"] is False
    assert profile["original_aggregate_accepted"] is False
    assert (
        profile["rf_probability"]
        == "ordered_float64_sum_of_stored_leaf_probabilities_divided_by_100"
    )
    assert profile["exact_parity"] is True
    assert profile["retry_or_resume"] is False
    assert (
        sha256((ROOT / api.ACCOUNTING_PATH).read_bytes()).hexdigest()
        == api.ACCOUNTING_SHA256
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("protected_evaluation_ready", True),
        ("maximum_new_fits", 2),
        ("maximum_new_fits", True),
        ("exact_parity", False),
        ("retry_or_resume", True),
        ("original_aggregate_accepted", True),
        ("additional_transformer_fits", 1),
        ("stages", ["random_forest", "retained_audit"]),
    ],
)
def test_policy_changes_rejected(api, field, value):
    profile = json.loads((ROOT / api.PROFILE_PATH).read_bytes())
    profile[field] = value
    with pytest.raises(api.CorrectionError, match="profile_policy"):
        api.validate_profile(profile)


def test_binding_checks_supplementary_committed_bytes_and_rechecks_base(
    api, tmp_path, monkeypatch
):
    root = tmp_path / "repo"
    for name in (api.PROFILE_PATH, api.ACCOUNTING_PATH):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / name).read_bytes())
    base = execution_preflight.ExecutionBinding(root, "a" * 40, "b" * 64, (), "{}")
    legacy = development_execution.DevelopmentExecutionBinding(
        base, api.BASE_PROFILE_SHA256, development_execution.METHODS_SHA256, None, b"{}"
    )
    calls = []
    monkeypatch.setattr(api, "bind_development_execution", lambda *a, **kw: legacy)
    monkeypatch.setattr(
        api, "recheck_development_binding", lambda b: calls.append("recheck")
    )
    monkeypatch.setattr(
        execution_preflight, "_committed_files", lambda *a: calls.append(a[2])
    )
    digest = sha256((root / api.PROFILE_PATH).read_bytes()).hexdigest()
    bound = api.bind_correction(
        root, expected_revision=base.revision, expected_profile_sha256=digest
    )
    assert bound.development == legacy
    assert bound.profile_sha256 == digest
    assert sha256(bound.accounting_bytes).hexdigest() == api.ACCOUNTING_SHA256
    assert calls == [
        {api.PROFILE_PATH: digest, api.ACCOUNTING_PATH: api.ACCOUNTING_SHA256},
        "recheck",
    ]
    api.recheck_correction(bound)
    with pytest.raises(api.CorrectionError):
        api.recheck_correction(replace(bound, accounting_bytes=b"{}"))


def test_binding_rejects_wrong_caller_hash(api, tmp_path, monkeypatch):
    root = tmp_path / "repo"
    path = root / api.PROFILE_PATH
    path.parent.mkdir(parents=True)
    path.write_bytes((ROOT / api.PROFILE_PATH).read_bytes())
    base = execution_preflight.ExecutionBinding(root, "a" * 40, "b" * 64, (), "{}")
    legacy = development_execution.DevelopmentExecutionBinding(
        base, api.BASE_PROFILE_SHA256, development_execution.METHODS_SHA256, None, b"{}"
    )
    monkeypatch.setattr(api, "bind_development_execution", lambda *a, **kw: legacy)
    with pytest.raises(api.CorrectionError, match="profile_hash"):
        api.bind_correction(
            root, expected_revision=base.revision, expected_profile_sha256="0" * 64
        )
