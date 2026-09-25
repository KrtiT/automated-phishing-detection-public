"""Development authorization uses public bytes and synthetic checkouts only."""

import copy
import importlib
import json
import subprocess
from dataclasses import FrozenInstanceError, replace
from hashlib import sha256
from pathlib import Path

import pytest

from automated_phishing_detection import execution_preflight

ROOT = Path(__file__).resolve().parents[1]
PROFILE = "data/development-execution-contract-v1.json"
METHODS = "data/secondary-development-contract-v1.json"
BASE = "data/execution-binding-contract-v2.json"
BASE_HASH = "887f771381927dfe1b9268a45f4e605baf3e9a7caee2b7005cdfe68b1be516e1"
METHODS_HASH = "f592352593ae64b178a468d5800e780267275a62046a05b31b952f69e424f44f"


@pytest.fixture
def api():
    assert (
        ROOT / "src/automated_phishing_detection/development_execution.py"
    ).is_file(), "missing separate development execution binding"
    return importlib.import_module("automated_phishing_detection.development_execution")


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
    for name in (*public, PROFILE, METHODS, BASE, "pyproject.toml", "uv.lock"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / name).read_bytes())
    package = root / "src/automated_phishing_detection"
    package.mkdir(parents=True)
    for name in ("__init__.py", "execution_preflight.py"):
        (package / name).write_text("# Synthetic source identity fixture.\n")
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
        "Synthetic public checkout",
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
        execution_preflight._historical_v2_committed_files(root, revision, public)
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
        raising=False,
    )
    return root, revision, sha256((root / PROFILE).read_bytes()).hexdigest(), calls


def bind(api, checkout):
    root, revision, digest, _ = checkout
    return api.bind_development_execution(
        root, expected_revision=revision, expected_profile_sha256=digest
    )


def test_separate_binding_authenticates_development_only(api, checkout):
    bound = bind(api, checkout)
    assert bound.base.revision == checkout[1]
    assert bound.profile_sha256 == checkout[2]
    assert bound.methods_sha256 == METHODS_HASH
    assert bound.base.protected_evaluation_ready is False
    assert bound.protected_evaluation_ready is False
    assert bound.pins.train_sha256 != bound.pins.validation_sha256
    assert (
        sha256(bound.preparation_bytes).hexdigest()
        == bound.pins.preparation_summary_sha256
    )
    with pytest.raises(FrozenInstanceError):
        bound.profile_sha256 = "0" * 64


def test_profile_is_prospective_without_rewriting_existing_contracts(api):
    assert sha256((ROOT / PROFILE).read_bytes()).hexdigest() == (
        "67146228d636c16f02484998741c7a1545da68b209e693620efab22b2676cd43"
    )
    value = json.loads((ROOT / PROFILE).read_bytes())
    assert value["development_execution_ready"] is True
    assert value["protected_evaluation_ready"] is False
    assert value["input_roles"] == ["train", "validation"]
    assert value["steps"] == list(api.STEPS)
    assert value["failure_policy"] == "fail_stop_no_retry_no_resume"
    assert sha256((ROOT / BASE).read_bytes()).hexdigest() == BASE_HASH
    assert sha256((ROOT / METHODS).read_bytes()).hexdigest() == METHODS_HASH


def test_wrong_caller_pin_is_rejected(api, checkout):
    root, revision, _, _ = checkout
    with pytest.raises(api.DevelopmentExecutionError, match="profile_hash"):
        api.bind_development_execution(
            root, expected_revision=revision, expected_profile_sha256="0" * 64
        )


def test_dirty_checkout_rejected_before_supplement_access(api, checkout, monkeypatch):
    (checkout[0] / "unexpected.txt").write_text("fixture")
    monkeypatch.setattr(api, "_read_profile", lambda *a: pytest.fail("supplement read"))
    with pytest.raises(execution_preflight.ExecutionPreflightError, match="clean"):
        bind(api, checkout)


def test_recheck_rejects_changed_binding(api, checkout):
    bound = bind(api, checkout)
    api.recheck_development_binding(bound)
    with pytest.raises(api.DevelopmentExecutionError):
        api.recheck_development_binding(replace(bound, methods_sha256="0" * 64))


@pytest.mark.parametrize(
    "changes",
    [
        {"protected_evaluation_ready": True},
        {"development_execution_ready": 1},
        {"input_roles": ["train", "validation", "group_test"]},
        {"steps": ["formatting"]},
        {"failure_policy": "retry"},
        {"unexpected": True},
    ],
)
def test_profile_cannot_broaden_scope_even_with_new_caller_hash(api, changes):
    value = json.loads((ROOT / PROFILE).read_bytes())
    value.update(changes)
    with pytest.raises(api.DevelopmentExecutionError, match="profile_policy"):
        api._validate_profile(value)


def test_bound_pins_match_all_accepted_public_chains(api, checkout):
    bound = bind(api, checkout)
    baseline = json.loads((ROOT / "reports/rq1-baseline-v2-summary.json").read_bytes())
    gmm = json.loads(
        (ROOT / "reports/rq2-gmm-development-v1-summary.json").read_bytes()
    )
    assert (
        bound.pins.logistic_l1_artifact_sha256
        == baseline["models"]["Logistic-L1"]["artifact_sha256"]
    )
    assert bound.pins.gmm_artifact_sha256 == gmm["artifact_hashes"]["gmm.json"]
    assert bound.pins.train_sha256 == baseline["input_hashes"]["train"]
    assert bound.pins.validation_sha256 == gmm["input_hashes"]["validation"]


def test_committed_supplement_must_be_regular(api, checkout):
    root = checkout[0]
    content = (root / PROFILE).read_bytes()
    target = root.parent / "outside-profile.json"
    target.write_bytes(content)
    (root / PROFILE).unlink()
    (root / PROFILE).symlink_to(target)
    with pytest.raises(execution_preflight.ExecutionPreflightError):
        bind(api, checkout)


@pytest.mark.parametrize("value", [None, True, "A" * 64, "0" * 63, "../profile"])
def test_invalid_pin_is_rejected_before_base_binding(api, monkeypatch, value):
    monkeypatch.setattr(
        api, "bind_execution", lambda *a, **k: pytest.fail("base accessed")
    )
    with pytest.raises(execution_preflight.ExecutionPreflightError):
        api.bind_development_execution(
            ROOT, expected_revision="a" * 40, expected_profile_sha256=value
        )


def test_modified_methods_are_rejected_against_committed_bytes(api, checkout):
    bound = bind(api, checkout)
    (checkout[0] / METHODS).write_bytes(b"{}\n")
    with pytest.raises(
        execution_preflight.ExecutionPreflightError, match="committed blob"
    ):
        api._read_profile(bound.base, bound.profile_sha256)


def test_duplicate_profile_keys_are_rejected(api, checkout):
    bound = bind(api, checkout)
    content = (
        (checkout[0] / PROFILE)
        .read_bytes()
        .replace(b'"schema_version": 1,', b'"schema_version": 1, "schema_version": 1,')
    )
    (checkout[0] / PROFILE).write_bytes(content)
    with pytest.raises(api.DevelopmentExecutionError, match="profile_policy"):
        api._read_profile(bound.base, sha256(content).hexdigest())


@pytest.mark.parametrize("field", ["train", "validation", "logistic_l1", "gmm"])
def test_source_chain_rejects_wrong_method_input_identity(api, checkout, field):
    bound = bind(api, checkout)
    methods = json.loads((ROOT / METHODS).read_bytes())
    target = (
        "partitions" if field in ("train", "validation") else "accepted_model_bytes"
    )
    methods["inputs"][target][field] = "f" * 64
    with pytest.raises(api.DevelopmentExecutionError, match="model_chain"):
        api._development_inputs(bound.base, methods)


@pytest.mark.parametrize("split", ["train", "validation"])
def test_source_chain_rejects_repaired_hashes_with_wrong_counts(api, checkout, split):
    bound = bind(api, checkout)
    methods = json.loads((ROOT / METHODS).read_bytes())
    relative = "reports/rq1-baseline-v2-summary.json"
    baseline = json.loads((checkout[0] / relative).read_bytes())
    baseline["input_counts"][split]["rows"] += 1
    content = json.dumps(baseline).encode()
    (checkout[0] / relative).write_bytes(content)
    hashes = dict(bound.base.source_hashes)
    hashes[relative] = methods["public_file_sha256"][relative] = sha256(
        content
    ).hexdigest()
    changed = replace(bound.base, source_hashes=tuple(sorted(hashes.items())))
    with pytest.raises(api.DevelopmentExecutionError, match="partition_chain"):
        api._development_inputs(changed, methods)


def test_source_chain_rejects_unbound_public_reference(api, checkout):
    bound = bind(api, checkout)
    methods = copy.deepcopy(json.loads((ROOT / METHODS).read_bytes()))
    methods["public_file_sha256"]["reports/not-accepted.json"] = "f" * 64
    with pytest.raises(api.DevelopmentExecutionError, match="public_chain"):
        api._development_inputs(bound.base, methods)


def test_public_bytes_are_rechecked_after_base_authentication(api, checkout):
    bound = bind(api, checkout)
    relative = "reports/phiusiil-preparation-summary.json"
    (checkout[0] / relative).write_bytes(b"{}")
    with pytest.raises(api.DevelopmentExecutionError, match="public_input_hash"):
        api._development_inputs(bound.base, json.loads((ROOT / METHODS).read_bytes()))


def test_source_drift_hidden_from_git_is_rechecked_at_end(api, checkout, monkeypatch):
    root = checkout[0]
    relative = "src/automated_phishing_detection/__init__.py"
    original = api._development_inputs

    def drift(*args):
        result = original(*args)
        git(root, "update-index", "--assume-unchanged", relative)
        (root / relative).write_text("# Changed after initial authentication.\n")
        assert git(root, "status", "--porcelain") == ""
        return result

    monkeypatch.setattr(api, "_development_inputs", drift)
    with pytest.raises(
        execution_preflight.ExecutionPreflightError, match="committed blob"
    ):
        bind(api, checkout)


def test_runtime_and_import_binding_is_rechecked_at_end(api, checkout, monkeypatch):
    def reject(_):
        raise execution_preflight.ExecutionPreflightError("runtime changed")

    monkeypatch.setattr(api, "recheck_binding", reject)
    with pytest.raises(
        execution_preflight.ExecutionPreflightError, match="runtime changed"
    ):
        bind(api, checkout)
