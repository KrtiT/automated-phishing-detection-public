"""Explicit amendment binding and saved checks remain metadata-only and closed."""

import json
from dataclasses import replace
from pathlib import Path

import adopted_study_profile_fixtures as success
import pytest
import study_execution_fixtures as original
from operational_input_fixtures import candidates as candidates
from operational_input_fixtures import manifests as manifests
from study_execution_fixtures import execution_case as execution_case
from study_run_record_fixtures import prepared as prepared
from study_urlnorm_execution_fixtures import (
    POLICY_PATH,
    bound,
    install_policy,
    successful_case,
)
from study_urlnorm_fixtures import digest, policy_bytes, profile, seal
from test_study_operator_saved import hold_snapshot, verify

from automated_phishing_detection import _study_execution_policy as legacy
from automated_phishing_detection._adopted_study_verification import validate_child_root
from automated_phishing_detection._checkpoint_codec import canonical_bytes


def test_amendment_binding_is_metadata_only_with_absent_original_inputs(
    execution_case, monkeypatch
):
    value = seal(profile(execution_case))
    assert not (execution_case.root.parent / "inputs").exists()
    install_policy(execution_case, monkeypatch)
    pin = original.write_envelope(execution_case, value)

    def forbidden(*args, **kwargs):
        pytest.fail("amendment authorization inspected a private path")

    with monkeypatch.context() as scoped:
        scoped.setattr(Path, "stat", forbidden)
        scoped.setattr(Path, "resolve", forbidden)
        result = original.api().bind_study_execution(
            execution_case.root,
            expected_revision=execution_case.base.revision,
            envelope_path=execution_case.envelope_path,
            expected_envelope_sha256=pin,
        )
        original.api().recheck_study_execution(result)
    assert result.policy_bytes == policy_bytes()
    assert result.deadlines == legacy.DEADLINES
    assert not result.base.protected_evaluation_ready
    assert not result.external.protected_evaluation_ready
    assert not result.operational.protected_evaluation_ready
    assert set(execution_case.events) <= {
        ("public", "data/sources.json"),
        ("public", POLICY_PATH),
    }


@pytest.mark.parametrize("change", ["missing", "changed", "symlink", "wrong_name"])
def test_amendment_policy_requires_exact_committed_regular_file(
    execution_case, monkeypatch, change
):
    value = seal(profile(execution_case))
    install_policy(execution_case, monkeypatch)
    previous = original.preflight._git

    def altered(root, *arguments):
        result = previous(root, *arguments)
        if arguments[0] == "cat-file":
            return result + b" " if change == "changed" else result
        if change == "missing":
            return b""
        if change == "symlink":
            return result.replace(b"100644", b"120000")
        return (
            result.replace(b"policy-v3", b"policy-v2")
            if change == "wrong_name"
            else result
        )

    monkeypatch.setattr(original.preflight, "_git", altered)
    with pytest.raises(ValueError):
        original.bind(execution_case, value)


@pytest.mark.parametrize("slot", ["method", "profile", "access"])
def test_incomplete_scientific_authority_stops_before_public_binding(
    execution_case, monkeypatch, slot
):
    value = seal(profile(execution_case))
    del value["decisions"][slot]
    install_policy(execution_case, monkeypatch)

    def forbidden(*args, **kwargs):
        pytest.fail("incomplete amendment authority reached public binding")

    monkeypatch.setattr(original.preflight, "bind_execution", forbidden)
    with pytest.raises(ValueError):
        original.bind(execution_case, value)
    assert not execution_case.events


@pytest.mark.parametrize("field", ["policy_bytes", "profile_bytes", "envelope_bytes"])
def test_recheck_rejects_forged_retained_binding(execution_case, monkeypatch, field):
    result = bound(execution_case, monkeypatch)
    with pytest.raises(ValueError):
        original.api().recheck_study_execution(replace(result, **{field: b"changed"}))


def test_recheck_rejects_envelope_mutation(execution_case, monkeypatch):
    result = bound(execution_case, monkeypatch)
    execution_case.envelope_path.write_bytes(canonical_bytes({"revoked": True}))
    with pytest.raises(ValueError):
        original.api().recheck_study_execution(result)


def test_recheck_rejects_changed_committed_policy(execution_case, monkeypatch):
    result = bound(execution_case, monkeypatch)
    execution_case.policy += b" "
    with pytest.raises(ValueError):
        original.api().recheck_study_execution(result)


@pytest.mark.parametrize("member", ["policy", "profile", "envelope"])
def test_saved_authorization_rejects_rehashed_noncanonical_member(
    execution_case, monkeypatch, member
):
    result = bound(execution_case, monkeypatch)
    payloads = {
        name: getattr(result, f"{name}_bytes")
        for name in ("policy", "profile", "envelope")
    }
    payloads[member] += b" "
    with pytest.raises(ValueError):
        original.api().validate_study_execution_bytes(
            payloads["policy"],
            payloads["profile"],
            payloads["envelope"],
            expected_profile_sha256=digest(payloads["profile"]),
            expected_envelope_sha256=digest(payloads["envelope"]),
        )


def test_saved_hold_retains_amendment_authority_and_original_ten_member_inventory(
    execution_case, prepared, monkeypatch
):
    authorization = bound(execution_case, monkeypatch)
    snapshot = hold_snapshot(authorization, prepared)
    assert len(snapshot.payloads) == 10
    execution = verify(authorization, snapshot)
    assert execution["study_profile_sha256"] == authorization.profile_sha256
    assert (
        json.loads(authorization.profile_bytes)["profile_id"]
        == "study-urlnorm-profile-v1"
    )


@pytest.mark.parametrize("forged_external", [False, True])
def test_saved_success_and_live_child_prefixes_bind_amendment_profile(
    prepared, manifests, monkeypatch, forged_external
):
    case = successful_case(prepared, manifests, monkeypatch)
    authorization, snapshot = success.saved(
        case, external_pin="0" * 64 if forged_external else None
    )
    assert len(snapshot.payloads) == 14
    if forged_external:
        with pytest.raises(ValueError):
            verify(authorization, snapshot)
    else:
        verify(authorization, snapshot)
    for role in ("service", "client"):
        frame, payloads = success.child_context(snapshot, role)
        if forged_external:
            with pytest.raises(ValueError):
                validate_child_root(authorization, frame, payloads)
        else:
            validate_child_root(authorization, frame, payloads)
