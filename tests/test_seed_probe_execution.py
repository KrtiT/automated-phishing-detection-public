"""Seed/probe binding checks use public records and invented Git checkouts."""

import copy
import importlib
import json
import subprocess
from dataclasses import FrozenInstanceError, asdict, replace
from hashlib import sha256
from pathlib import Path

import pytest

from automated_phishing_detection import execution_preflight

ROOT = Path(__file__).resolve().parents[1]
PROFILE = "data/seed-probe-execution-contract-v2.json"
V1_PROFILE = "data/seed-probe-execution-contract-v1.json"
STOPPED_ATTEMPT = "reports/secondary-seed-probe-v1-attempt-1.json"
METHODS = "data/secondary-seed-probe-contract-v1.json"
DEVELOPMENT_METHODS = "data/secondary-development-contract-v1.json"
ACCEPTED = "reports/secondary-development-correction-v2-summary.json"
BASE = "data/execution-binding-contract-v2.json"
BASE_HASH = "887f771381927dfe1b9268a45f4e605baf3e9a7caee2b7005cdfe68b1be516e1"
V1_PROFILE_HASH = "cf18fa8c35039c63f896cc62c7aaac8b0847a1abf55676ba67ee65b42340381d"
STOPPED_ATTEMPT_HASH = (
    "3be65bd38c32b8bf8aafa06eede3577a0d1acc212f052d2d9f60768183f535c6"
)
METHODS_HASH = "eb279404728e498999fc7fd0c7578291373bb80b9816f88b5d7202dfdf637380"
ACCEPTED_HASH = "663f1117cd33f949b70c35c56810764193f69cae3a37505004d0db27641d829d"

POLICY_MUTATIONS = [
    ("schema_version", True),
    ("contract_id", "seed-probe-execution-v3"),
    ("date", "2026-09-24"),
    ("base_execution_profile_sha256", "0" * 64),
    ("stopped_attempt_accounting_sha256", "0" * 64),
    ("methods_sha256", "0" * 64),
    ("development_execution_ready", False),
    ("protected_evaluation_ready", True),
    ("scientific_method_changes", True),
    ("primary_changes", True),
    ("stages", ["probes"]),
    ("new_fit_seeds", [42, 43, 44, 45]),
    ("primary_seed_42_fits", True),
    ("prior_attempt_new_fits", True),
    ("maximum_new_fits", True),
    ("authorized_fresh_sequences", True),
    ("retry_or_resume", True),
    ("failure_policy", "retry"),
    ("correction_scope", "changed"),
    ("execution_policy", "changed"),
    ("inheritance", "changed"),
    ("saved_evidence_canonicalization", "changed"),
    ("one_use_enforcement", "changed"),
]


@pytest.fixture
def api():
    path = ROOT / "src/automated_phishing_detection/seed_probe_execution.py"
    assert path.is_file(), "missing separate seed/probe execution binding"
    return importlib.import_module("automated_phishing_detection.seed_probe_execution")


def git(root, *args):
    return (
        subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)
        .stdout.decode()
        .strip()
    )


@pytest.fixture
def checkout(tmp_path, api, monkeypatch):
    root = tmp_path / "checkout"
    root.mkdir()
    public = json.loads((ROOT / BASE).read_bytes())["public_file_sha256"]
    selected = (
        *public,
        PROFILE,
        V1_PROFILE,
        STOPPED_ATTEMPT,
        METHODS,
        DEVELOPMENT_METHODS,
        ACCEPTED,
        BASE,
        "pyproject.toml",
        "uv.lock",
    )
    for name in selected:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / name).read_bytes())
    package = root / "src/automated_phishing_detection"
    package.mkdir(parents=True)
    for name in ("__init__.py", "execution_preflight.py"):
        (package / name).write_text("# Invented source identity fixture.\n")
    git(root, "init", "-q")
    git(root, "add", ".")
    git(
        root,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "-qm",
        "Public binding fixture",
    )
    revision = git(root, "rev-parse", "HEAD")
    base = execution_preflight.ExecutionBinding(
        root, revision, BASE_HASH, tuple(sorted(public.items())), '{"fixture":true}'
    )
    calls = []

    def bind_base(supplied_root, *, expected_revision, expected_contract_sha256):
        assert supplied_root == root
        assert expected_revision == revision
        assert expected_contract_sha256 == BASE_HASH
        execution_preflight._check_checkout(root, revision)
        execution_preflight._committed_files(root, revision, public)
        calls.append("base")
        return base

    monkeypatch.setattr(api, "bind_execution", bind_base)
    monkeypatch.setattr(
        api,
        "recheck_binding",
        lambda bound: bind_base(
            bound.root,
            expected_revision=bound.revision,
            expected_contract_sha256=bound.contract_sha256,
        ),
    )
    return root, revision, sha256((root / PROFILE).read_bytes()).hexdigest(), calls


def bind(api, checkout):
    root, revision, digest, _ = checkout
    return api.bind_seed_probe_execution(
        root, expected_revision=revision, expected_profile_sha256=digest
    )


def test_binding_is_development_only_and_immutable(api, checkout):
    bound = bind(api, checkout)
    assert bound.base.revision == checkout[1]
    assert bound.profile_sha256 == checkout[2] == api.PROFILE_SHA256
    assert bound.base_profile_sha256 == V1_PROFILE_HASH
    assert bound.stopped_attempt_sha256 == STOPPED_ATTEMPT_HASH
    assert bound.methods_sha256 == METHODS_HASH
    assert bound.accepted_development_sha256 == ACCEPTED_HASH
    assert bound.protected_evaluation_ready is False
    assert bound.base.protected_evaluation_ready is False
    assert (
        sha256(bound.preparation_bytes).hexdigest()
        == bound.pins.preparation_summary_sha256
    )
    assert checkout[3] == ["base", "base"]
    with pytest.raises(FrozenInstanceError):
        bound.profile_sha256 = "0" * 64


def test_policy_fixes_sequential_stages_and_no_retry(api):
    value = json.loads((ROOT / PROFILE).read_bytes())
    api.validate_profile(value)
    assert value["base_execution_profile_sha256"] == V1_PROFILE_HASH
    assert value["stopped_attempt_accounting_sha256"] == STOPPED_ATTEMPT_HASH
    assert value["methods_sha256"] == METHODS_HASH
    assert value["development_execution_ready"] is True
    assert value["scientific_method_changes"] is False
    assert value["prior_attempt_new_fits"] == 0
    assert type(value["prior_attempt_new_fits"]) is int
    assert value["authorized_fresh_sequences"] == 1
    assert type(value["authorized_fresh_sequences"]) is int
    assert value["retry_or_resume"] is False
    assert (
        value["stages"]
        == list(api.STAGES)
        == ["seed_42_calibration", "seed_43", "seed_44", "seed_45", "seed_46", "probes"]
    )
    assert value["new_fit_seeds"] == [43, 44, 45, 46]
    assert value["primary_seed_42_fits"] == 0
    assert value["maximum_new_fits"] == 4
    assert value["protected_evaluation_ready"] is False
    assert value["primary_changes"] is False
    assert value["failure_policy"] == "fail_stop_no_retry_no_resume"
    v1 = json.loads((ROOT / V1_PROFILE).read_bytes())
    assert sha256((ROOT / V1_PROFILE).read_bytes()).hexdigest() == V1_PROFILE_HASH
    assert v1["input_roles"] == ["train", "validation"]
    assert v1["partition_reads"]["seed_42_calibration"] == ["validation"]
    assert v1["partition_reads"]["probes"] == ["validation"]
    assert v1["stages"] == value["stages"]
    assert v1["new_fit_seeds"] == value["new_fit_seeds"]
    assert v1["primary_seed_42_fits"] == value["primary_seed_42_fits"]
    assert v1["maximum_new_fits"] == value["maximum_new_fits"]
    assert v1["failure_policy"] == value["failure_policy"]
    for seed in range(43, 47):
        assert v1["partition_reads"][f"seed_{seed}"] == ["train", "validation"]
    assert (
        sha256((ROOT / STOPPED_ATTEMPT).read_bytes()).hexdigest()
        == STOPPED_ATTEMPT_HASH
    )
    assert sha256((ROOT / METHODS).read_bytes()).hexdigest() == METHODS_HASH
    assert sha256((ROOT / ACCEPTED).read_bytes()).hexdigest() == ACCEPTED_HASH


@pytest.mark.parametrize("field,value", POLICY_MUTATIONS)
def test_policy_cannot_be_broadened(api, field, value):
    policy = json.loads((ROOT / PROFILE).read_bytes())
    assert {name for name, _ in POLICY_MUTATIONS} == set(policy)
    policy[field] = value
    with pytest.raises(api.SeedProbeExecutionError, match="profile_policy"):
        api.validate_profile(policy)


def test_unexpected_policy_field_is_rejected(api):
    policy = json.loads((ROOT / PROFILE).read_bytes())
    policy["unexpected"] = False
    with pytest.raises(api.SeedProbeExecutionError, match="profile_policy"):
        api.validate_profile(policy)


def test_old_v1_profile_is_rejected_before_base_binding(api, monkeypatch):
    monkeypatch.setattr(
        api, "bind_execution", lambda *a, **k: pytest.fail("base accessed")
    )
    with pytest.raises(api.SeedProbeExecutionError, match="profile_hash_mismatch"):
        api.bind_seed_probe_execution(
            ROOT,
            expected_revision="a" * 40,
            expected_profile_sha256=V1_PROFILE_HASH,
        )


def test_artifact_hashes_come_from_accepted_public_summaries(api, checkout):
    bound = bind(api, checkout)
    artifacts = dict(bound.primary_artifact_hashes)
    baseline = json.loads((ROOT / "reports/rq1-baseline-v2-summary.json").read_bytes())
    transformer = json.loads(
        (ROOT / "reports/rq1-transformer-cascade-v2-summary.json").read_bytes()
    )
    gmm = json.loads(
        (ROOT / "reports/rq2-gmm-development-v1-summary.json").read_bytes()
    )
    assert artifacts == {
        "length-only.json": baseline["models"]["length-only"]["artifact_sha256"],
        "logistic-l1.json": baseline["models"]["Logistic-L1"]["artifact_sha256"],
        "gmm.json": gmm["artifact_hashes"]["gmm.json"],
        **transformer["artifact_hashes"],
    }
    # The public transformer summary deliberately omits its cutoff and band.
    assert json.loads(bound.public_operating_points_json) == {
        "length_threshold": baseline["models"]["length-only"]["validation_threshold"][
            "threshold"
        ],
        "stage1_threshold": baseline["models"]["Logistic-L1"]["validation_threshold"][
            "threshold"
        ],
        "monitor_boundary": gmm["threshold"],
    }


def test_retained_drift_hashes_come_from_the_accepted_member(api, checkout):
    bound = bind(api, checkout)
    report = json.loads((ROOT / ACCEPTED).read_bytes())
    member = report["completion"]["retained_audit"]["result"]["members"][0]
    summary = member["summary"]["result"]
    assert json.loads(bound.retained_drift_summary_json) == summary
    assert (
        bound.training_reference_sha256
        == summary["private_sha256"]["training-reference.json"]
    )
    assert (
        bound.validation_audit_sha256
        == summary["private_sha256"]["validation-audit.json"]
    )
    assert summary["input_hashes"] == asdict(bound.pins)


def test_transformer_summary_bytes_are_available_without_worker_reread(api, checkout):
    bound = bind(api, checkout)
    relative = "reports/rq1-transformer-cascade-v2-summary.json"
    assert bound.transformer_summary_bytes == (ROOT / relative).read_bytes()
    assert (
        sha256(bound.transformer_summary_bytes).hexdigest()
        == dict(bound.base.source_hashes)[relative]
    )


def test_binding_reads_only_named_public_files(api, checkout, monkeypatch):
    allowed = set(json.loads((ROOT / BASE).read_bytes())["public_file_sha256"])
    allowed.update(
        (
            PROFILE,
            V1_PROFILE,
            STOPPED_ATTEMPT,
            METHODS,
            DEVELOPMENT_METHODS,
            ACCEPTED,
            BASE,
            "pyproject.toml",
            "uv.lock",
            "src/automated_phishing_detection/__init__.py",
            "src/automated_phishing_detection/execution_preflight.py",
        )
    )
    original = execution_preflight._read_regular
    reads = []

    def read_public(root, relative):
        assert relative in allowed, "nonpublic path inspected"
        reads.append(relative)
        return original(root, relative)

    monkeypatch.setattr(execution_preflight, "_read_regular", read_public)
    bind(api, checkout)
    assert ACCEPTED in reads
    assert not any("processed" in name for name in reads)


@pytest.mark.parametrize(
    "pin", [None, True, "A" * 64, "0" * 63, "../profile", "0" * 64]
)
def test_invalid_profile_pin_stops_before_base_access(api, monkeypatch, pin):
    monkeypatch.setattr(
        api, "bind_execution", lambda *a, **k: pytest.fail("base accessed")
    )
    with pytest.raises(
        (api.SeedProbeExecutionError, execution_preflight.ExecutionPreflightError)
    ):
        api.bind_seed_probe_execution(
            ROOT, expected_revision="a" * 40, expected_profile_sha256=pin
        )


def test_dirty_checkout_stops_before_supplement_access(api, checkout, monkeypatch):
    (checkout[0] / "unexpected.txt").write_text("fixture")
    monkeypatch.setattr(
        api, "_read_supplement", lambda *a: pytest.fail("supplement read")
    )
    with pytest.raises(execution_preflight.ExecutionPreflightError, match="clean"):
        bind(api, checkout)


def test_recheck_detects_changed_binding(api, checkout):
    bound = bind(api, checkout)
    api.recheck_seed_probe_binding(bound)
    for field, value in (
        ("methods_sha256", "0" * 64),
        ("validation_audit_sha256", "0" * 64),
        ("public_operating_points_json", "{}"),
    ):
        with pytest.raises(api.SeedProbeExecutionError, match="binding_changed"):
            api.recheck_seed_probe_binding(replace(bound, **{field: value}))


def test_duplicate_profile_keys_rejected(api, checkout):
    content = (
        (checkout[0] / PROFILE)
        .read_bytes()
        .replace(b'"schema_version": 1,', b'"schema_version": 1, "schema_version": 1,')
    )
    with pytest.raises(ValueError):
        api._json(content)


@pytest.mark.parametrize(
    "relative",
    [PROFILE, V1_PROFILE, STOPPED_ATTEMPT, METHODS, DEVELOPMENT_METHODS, ACCEPTED],
)
def test_supplement_must_match_committed_regular_bytes(api, checkout, relative):
    base = bind(api, checkout).base
    (checkout[0] / relative).write_bytes(b"{}\n")
    with pytest.raises(
        (api.SeedProbeExecutionError, execution_preflight.ExecutionPreflightError)
    ):
        api._read_supplement(base, checkout[2])


@pytest.mark.parametrize("relative", [V1_PROFILE, STOPPED_ATTEMPT])
def test_history_supplement_symlink_is_not_followed(api, checkout, relative):
    path = checkout[0] / relative
    target = checkout[0].parent / "outside.json"
    target.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(target)
    with pytest.raises(execution_preflight.ExecutionPreflightError):
        bind(api, checkout)


@pytest.mark.parametrize(
    "change",
    [
        "acceptance",
        "parent_exit",
        "worker_exit",
        "scope",
        "pins",
        "member_order",
        "member_digest",
        "reference_digest",
        "audit_digest",
        "drift_pins",
    ],
)
def test_accepted_drift_chain_rejects_inconsistent_public_records(
    api, checkout, change
):
    bound = bind(api, checkout)
    report = json.loads((ROOT / ACCEPTED).read_bytes())
    completion = report["completion"]
    member = completion["retained_audit"]["result"]["members"][0]
    if change == "acceptance":
        report["status"] = "stopped"
    elif change == "parent_exit":
        report["execution_observation"]["parent_exit_code"] = False
    elif change == "worker_exit":
        completion["worker_exit_codes"]["retained_audit"] = -9
    elif change == "scope":
        completion["protected_evaluation_authorized"] = True
    elif change == "pins":
        completion["execution"]["pins"]["train_sha256"] = "0" * 64
    elif change == "member_order":
        completion["retained_audit"]["result"]["members"].reverse()
    elif change == "member_digest":
        member["public_summary_sha256"] = "0" * 64
    elif change == "reference_digest":
        member["summary"]["private_sha256"]["training-reference.json"] = "0" * 64
    elif change == "audit_digest":
        member["summary"]["result"]["private_sha256"]["validation-audit.json"] = (
            "0" * 64
        )
    else:
        member["summary"]["result"]["input_hashes"]["validation_sha256"] = "0" * 64
    if change != "member_digest":
        member["public_summary_sha256"] = api._digest(member["summary"])
    report["completion_summary_sha256"] = api._digest(completion)
    with pytest.raises(api.SeedProbeExecutionError):
        api._retained_drift(report, bound.pins)


def test_unknown_methods_public_reference_is_rejected(api, checkout):
    bound = bind(api, checkout)
    methods = copy.deepcopy(json.loads((ROOT / METHODS).read_bytes()))
    methods["public_file_sha256"]["reports/unaccepted.json"] = "0" * 64
    with pytest.raises(api.SeedProbeExecutionError, match="methods_public_chain"):
        api._validate_methods_chain(bound.base, methods)


@pytest.mark.parametrize("field", ["primary_weight_sha256", "vocabulary_sha256"])
def test_seed_methods_cannot_substitute_a_primary_artifact(api, checkout, field):
    bound = bind(api, checkout)
    methods = json.loads((ROOT / METHODS).read_bytes())
    methods["seeds"][field] = "0" * 64
    with pytest.raises(api.SeedProbeExecutionError, match="primary_seed_method_chain"):
        api._primary_inputs(bound.base, methods)


def test_binding_does_not_reuse_the_correction_authorization(
    api, checkout, monkeypatch
):
    from automated_phishing_detection import (
        development_correction,
        development_execution,
    )

    def forbidden(*args, **kwargs):
        pytest.fail("an earlier execution profile was reused")

    monkeypatch.setattr(development_correction, "bind_correction", forbidden)
    monkeypatch.setattr(development_execution, "bind_development_execution", forbidden)
    bind(api, checkout)


def test_invalid_recheck_type_is_rejected_before_access(api, monkeypatch):
    monkeypatch.setattr(
        api, "bind_seed_probe_execution", lambda *a, **k: pytest.fail("access")
    )
    with pytest.raises(api.SeedProbeExecutionError, match="invalid_seed_probe_binding"):
        api.recheck_seed_probe_binding(None)


def test_runtime_and_import_identity_are_rechecked(api, checkout, monkeypatch):
    monkeypatch.setattr(
        api,
        "recheck_binding",
        lambda b: (_ for _ in ()).throw(
            execution_preflight.ExecutionPreflightError("runtime changed")
        ),
    )
    with pytest.raises(
        execution_preflight.ExecutionPreflightError, match="runtime changed"
    ):
        bind(api, checkout)


def test_hidden_public_mutation_is_detected_at_final_recheck(
    api, checkout, monkeypatch
):
    original = api._retained_drift

    def changed(*args):
        result = original(*args)
        git(checkout[0], "update-index", "--assume-unchanged", ACCEPTED)
        (checkout[0] / ACCEPTED).write_bytes(b"{}\n")
        assert git(checkout[0], "status", "--porcelain") == ""
        return result

    monkeypatch.setattr(api, "_retained_drift", changed)
    with pytest.raises(api.SeedProbeExecutionError, match="binding_changed"):
        bind(api, checkout)
