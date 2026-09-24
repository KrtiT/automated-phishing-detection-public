"""Prospective zero-fit probe correction binding uses public metadata only."""

import importlib
import inspect
import json
from dataclasses import FrozenInstanceError, replace
from hashlib import sha256
from pathlib import Path

import pytest

from automated_phishing_detection import execution_preflight, seed_probe_execution

ROOT = Path(__file__).resolve().parents[1]
PROFILE = "data/seed-probe-correction-contract-v1.json"
PROFILE_HASH = "46a659ddc809f998a12abaae0f5c6e353c2f965844487faf87c0637b9d7697a4"
BASE_PROFILE_HASH = "4da034b1a46baa599ae04226ee2f4d9a26c2b2d639cac73fa576ff9cb7aa8839"
ACCOUNTING = "reports/secondary-seed-probe-v2-attempt-1.json"
ACCOUNTING_HASH = "cf1fc0e6e41839464def2955b4475492b4839e637d4e563d4c94324057e74cb4"
METHODS_HASH = "eb279404728e498999fc7fd0c7578291373bb80b9816f88b5d7202dfdf637380"
EXPECTED_HISTORY_PINS = {
    "data/seed-probe-execution-contract-v2.json": BASE_PROFILE_HASH,
    "data/seed-probe-execution-contract-v1.json": (
        "cf18fa8c35039c63f896cc62c7aaac8b0847a1abf55676ba67ee65b42340381d"
    ),
    "reports/secondary-seed-probe-v1-attempt-1.json": (
        "3be65bd38c32b8bf8aafa06eede3577a0d1acc212f052d2d9f60768183f535c6"
    ),
    "data/secondary-seed-probe-contract-v1.json": METHODS_HASH,
    "data/secondary-development-contract-v1.json": (
        "f592352593ae64b178a468d5800e780267275a62046a05b31b952f69e424f44f"
    ),
    "reports/secondary-development-correction-v2-summary.json": (
        "663f1117cd33f949b70c35c56810764193f69cae3a37505004d0db27641d829d"
    ),
}


@pytest.fixture
def api():
    profile = ROOT / PROFILE
    module = ROOT / "src/automated_phishing_detection/seed_probe_correction.py"
    assert profile.is_file(), "missing immutable probe-correction profile"
    assert module.is_file(), "missing probe-correction binding"
    return importlib.import_module("automated_phishing_detection.seed_probe_correction")


# SP-CORR-05: metadata pins the exact zero-fit correction and failed-v2 history.
def test_policy_is_zero_fit_single_probe_and_preserves_failed_v2(api):
    assert api.PROFILE_PATH == PROFILE
    assert api.PROFILE_SHA256 == PROFILE_HASH
    assert api.BASE_PROFILE_SHA256 == BASE_PROFILE_HASH
    assert api.ACCOUNTING_PATH == ACCOUNTING
    assert api.ACCOUNTING_SHA256 == ACCOUNTING_HASH
    assert api.METHODS_SHA256 == METHODS_HASH
    assert api.HISTORY_PINS == EXPECTED_HISTORY_PINS
    content = (ROOT / PROFILE).read_bytes()
    assert sha256(content).hexdigest() == PROFILE_HASH
    profile = json.loads(content)
    api.validate_profile(profile)
    assert profile["stages"] == ["retained_seed_audit", "probes"]
    assert profile["retained_seed_stages"] == [
        "seed_42_calibration",
        "seed_43",
        "seed_44",
        "seed_45",
        "seed_46",
    ]
    assert profile["historical_new_fits"] == 4
    assert type(profile["historical_new_fits"]) is int
    assert profile["maximum_new_fits"] == 0
    assert type(profile["maximum_new_fits"]) is int
    assert profile["seed_stage_executions"] == 0
    assert type(profile["seed_stage_executions"]) is int
    assert profile["probe_executions"] == 1
    assert type(profile["probe_executions"]) is int
    assert profile["retry_or_resume"] is False
    assert profile["protected_evaluation_ready"] is False
    assert profile["original_v2_aggregate_accepted"] is False
    assert profile["base_seed_probe_profile_sha256"] == BASE_PROFILE_HASH
    assert profile["stopped_v2_accounting_sha256"] == ACCOUNTING_HASH
    assert profile["methods_sha256"] == METHODS_HASH
    assert profile["development_execution_ready"] is True
    assert profile["scientific_method_changes"] is False
    assert profile["primary_changes"] is False
    assert profile["failure_policy"] == "fail_stop_no_retry_no_resume"
    assert profile["preservation"] == (
        "The original v2 root remains failed, unaccepted, exhausted and unchanged; "
        "v2 itself does not promote its completed seed summaries, while this "
        "correction may accept all five only together after a separate all-or-none "
        "saved-evidence audit as descriptive seed/runtime evidence."
    )
    for field in (
        "parser_scope",
        "audit_scope",
        "execution_policy",
        "acceptance",
        "preservation",
        "research_limitations",
    ):
        assert type(profile[field]) is str
        assert profile[field].strip()


def test_pinned_accounting_remains_failed_unaccepted_and_exhausted(api):
    content = (ROOT / api.ACCOUNTING_PATH).read_bytes()
    assert sha256(content).hexdigest() == api.ACCOUNTING_SHA256
    accounting = json.loads(content)
    assert accounting["status"] == "failed_partial_evidence_retained"
    assert accounting["aggregate_accepted"] is False
    assert accounting["execution"]["profile_sha256"] == api.BASE_PROFILE_SHA256
    assert accounting["execution"]["methods_sha256"] == api.METHODS_SHA256
    assert accounting["execution_observation"]["new_fits"] == 4
    assert accounting["execution_observation"]["retries"] == 0
    assert accounting["execution_observation"]["resumes"] == 0
    assert accounting["authorization"] == {
        "profile_status": "exhausted",
        "retry_authorized": False,
        "resume_authorized": False,
        "refit_authorized": False,
        "required_before_any_new_probe_execution": [
            "separate_prospective_probe_correction_authority_frozen",
            "reviewed",
            "published",
            "ci_passed",
        ],
    }
    assert accounting["access"]["group_test_accessed"] is False
    assert accounting["access"]["phishvn_accessed"] is False
    assert accounting["access"]["external_source_accessed"] is False
    assert accounting["access"]["protected_evaluation_accessed"] is False


def test_every_policy_field_is_immutable(api):
    profile = json.loads((ROOT / api.PROFILE_PATH).read_bytes())
    for field, original in profile.items():
        changed = dict(profile)
        changed[field] = None if original is not None else False
        with pytest.raises(api.SeedProbeCorrectionError, match="profile_policy"):
            api.validate_profile(changed)
    profile["unexpected"] = False
    with pytest.raises(api.SeedProbeCorrectionError, match="profile_policy"):
        api.validate_profile(profile)


def _binding_fixture(api, tmp_path, monkeypatch):
    root = tmp_path / "repo"
    expected = {
        api.PROFILE_PATH: api.PROFILE_SHA256,
        api.ACCOUNTING_PATH: api.ACCOUNTING_SHA256,
        **api.HISTORY_PINS,
    }
    for name in expected:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / name).read_bytes())
    base = execution_preflight.ExecutionBinding(
        root, "a" * 40, "b" * 64, (), '{"fixture":true}'
    )
    seed_probe = seed_probe_execution.SeedProbeExecutionBinding(
        base,
        api.BASE_PROFILE_SHA256,
        seed_probe_execution.BASE_EXECUTION_PROFILE_SHA256,
        seed_probe_execution.STOPPED_ATTEMPT_SHA256,
        seed_probe_execution.METHODS_SHA256,
        seed_probe_execution.ACCEPTED_SHA256,
        None,
        b"{}",
        b"{}",
        (),
        "{}",
        "{}",
        "c" * 64,
        "d" * 64,
    )
    calls = []

    def bind_seed_probe(supplied_root, **kwargs):
        calls.append(("bind", supplied_root, kwargs))
        return seed_probe

    monkeypatch.setattr(api, "bind_seed_probe_execution", bind_seed_probe)
    monkeypatch.setattr(
        api,
        "recheck_seed_probe_binding",
        lambda value: calls.append(("recheck", value)),
    )
    monkeypatch.setattr(
        execution_preflight,
        "_committed_files",
        lambda supplied_root, revision, pins: calls.append(
            ("committed", supplied_root, revision, pins)
        ),
    )
    return root, base, seed_probe, expected, calls


def test_binding_wraps_v2_identity_and_rechecks_all_public_pins(
    api, tmp_path, monkeypatch
):
    root, base, seed_probe, expected, calls = _binding_fixture(
        api, tmp_path, monkeypatch
    )
    bound = api.bind_seed_probe_correction(
        root,
        expected_revision=base.revision,
        expected_profile_sha256=api.PROFILE_SHA256,
    )
    assert bound.seed_probe is seed_probe
    assert bound.base is base
    assert bound.profile_sha256 == api.PROFILE_SHA256
    assert sha256(bound.accounting_bytes).hexdigest() == api.ACCOUNTING_SHA256
    assert bound.protected_evaluation_ready is False
    assert calls == [
        (
            "bind",
            root,
            {
                "expected_revision": base.revision,
                "expected_profile_sha256": api.BASE_PROFILE_SHA256,
            },
        ),
        ("committed", root, base.revision, expected),
        ("recheck", seed_probe),
    ]
    with pytest.raises(FrozenInstanceError):
        bound.profile_sha256 = "0" * 64


def test_binding_accepts_no_research_paths(api):
    assert list(inspect.signature(api.bind_seed_probe_correction).parameters) == [
        "root",
        "expected_revision",
        "expected_profile_sha256",
    ]
    with pytest.raises(TypeError):
        api.bind_seed_probe_correction(
            ROOT,
            expected_revision="a" * 40,
            expected_profile_sha256=api.PROFILE_SHA256,
            training_path=ROOT / "private.jsonl",
        )


@pytest.mark.parametrize("digest", [BASE_PROFILE_HASH, "0" * 64, None, True])
def test_only_new_profile_is_execution_authority(api, monkeypatch, digest):
    monkeypatch.setattr(
        api,
        "bind_seed_probe_execution",
        lambda *a, **k: pytest.fail("v2 binding accessed for rejected authority"),
    )
    with pytest.raises(api.SeedProbeCorrectionError, match="profile_hash_mismatch"):
        api.bind_seed_probe_correction(
            ROOT,
            expected_revision="a" * 40,
            expected_profile_sha256=digest,
        )


@pytest.mark.parametrize("relative", [PROFILE, ACCOUNTING, *EXPECTED_HISTORY_PINS])
def test_each_correction_and_history_pin_is_reread(
    api, tmp_path, monkeypatch, relative
):
    root, base, _, _, _ = _binding_fixture(api, tmp_path, monkeypatch)
    (root / relative).write_bytes(b"{}\n")
    with pytest.raises(api.SeedProbeCorrectionError):
        api.bind_seed_probe_correction(
            root,
            expected_revision=base.revision,
            expected_profile_sha256=api.PROFILE_SHA256,
        )


def test_recheck_rebinds_and_detects_changed_correction(api, tmp_path, monkeypatch):
    root, base, _, _, calls = _binding_fixture(api, tmp_path, monkeypatch)
    bound = api.bind_seed_probe_correction(
        root,
        expected_revision=base.revision,
        expected_profile_sha256=api.PROFILE_SHA256,
    )
    calls.clear()
    api.recheck_seed_probe_correction(bound)
    assert [call[0] for call in calls] == ["bind", "committed", "recheck"]
    with pytest.raises(api.SeedProbeCorrectionError, match="binding_changed"):
        api.recheck_seed_probe_correction(replace(bound, accounting_bytes=b"{}"))


def test_invalid_recheck_type_stops_before_public_access(api, monkeypatch):
    monkeypatch.setattr(
        api,
        "bind_seed_probe_correction",
        lambda *a, **k: pytest.fail("public bytes accessed"),
    )
    with pytest.raises(
        api.SeedProbeCorrectionError, match="invalid_seed_probe_correction_binding"
    ):
        api.recheck_seed_probe_correction(None)
