"""Study-only adoption authenticates exact invented scope while v3 stays closed."""

import json
from dataclasses import FrozenInstanceError, replace
from hashlib import sha256

import pytest
import study_execution_fixtures as fixtures

from automated_phishing_detection._checkpoint_codec import canonical_bytes

execution_case = fixtures.execution_case


def test_study_execution_entry_exists():
    assert callable(fixtures.api().bind_study_execution)


def test_complete_adoption_retains_original_types_and_immutable_bytes(execution_case):
    case = execution_case
    result = fixtures.bind(case)
    assert result.base is case.base
    assert result.external == case.external and result.operational == case.operational
    assert not result.base.protected_evaluation_ready
    assert not result.external.protected_evaluation_ready
    assert not result.operational.protected_evaluation_ready
    assert result.paths.preparation.source_csv == case.root.parent / "inputs/source-csv"
    assert result.policy_bytes == case.policy
    assert result.profile_sha256 == sha256(result.profile_bytes).hexdigest()
    assert result.envelope_sha256 == sha256(result.envelope_bytes).hexdigest()
    assert result.policy_sha256 == sha256(case.policy).hexdigest()
    assert result.deadlines == dict(startup=300, shutdown=180, terminate=10, kill=10)
    result.deadlines["startup"] = 1
    assert result.deadlines["startup"] == 300
    with pytest.raises(FrozenInstanceError):
        result.policy_bytes = b"changed"


@pytest.mark.parametrize("slot", ["method", "advisor", "profile", "access"])
@pytest.mark.parametrize("status", ["pending", "revoked", "missing", "wrong_scope"])
def test_missing_or_partial_decisions_reject_before_public_binding(
    execution_case, monkeypatch, slot, status
):
    case = execution_case
    envelope = fixtures.seal(case)
    if status == "missing":
        del envelope["decisions"][slot]
    elif status == "wrong_scope":
        envelope["decisions"][slot]["subject_sha256"] = "0" * 64
    else:
        envelope["decisions"][slot]["status"] = status

    def forbidden(*args, **kwargs):
        pytest.fail("incomplete decision reached public binding")

    monkeypatch.setattr(fixtures.preflight, "bind_execution", forbidden)
    with pytest.raises(fixtures.api().StudyExecutionError):
        fixtures.bind(case, envelope)
    assert case.events == []


def test_envelope_is_caller_pinned_before_json_parse(execution_case):
    case = execution_case
    case.envelope_path.write_bytes(b"not-json")
    with pytest.raises(fixtures.api().StudyExecutionError):
        fixtures.api().bind_study_execution(
            case.root,
            expected_revision=case.base.revision,
            envelope_path=case.envelope_path,
            expected_envelope_sha256="0" * 64,
        )
    assert case.events == []


def test_retained_authorization_verifier_joins_all_three_bytes(execution_case):
    result = fixtures.bind(execution_case)
    parsed = fixtures.api().validate_study_execution_bytes(
        result.policy_bytes,
        result.profile_bytes,
        result.envelope_bytes,
        expected_profile_sha256=result.profile_sha256,
        expected_envelope_sha256=result.envelope_sha256,
    )
    assert parsed == json.loads(result.profile_bytes)
    with pytest.raises(fixtures.api().StudyExecutionError):
        fixtures.api().validate_study_execution_bytes(
            result.policy_bytes,
            result.profile_bytes + b" ",
            result.envelope_bytes,
            expected_profile_sha256=result.profile_sha256,
            expected_envelope_sha256=result.envelope_sha256,
        )


def test_recheck_rejects_changed_envelope_and_forged_retained_binding(execution_case):
    result = fixtures.bind(execution_case)
    fixtures.api().recheck_study_execution(result)
    with pytest.raises(fixtures.api().StudyExecutionError):
        fixtures.api().recheck_study_execution(replace(result, policy_bytes=b"changed"))
    execution_case.envelope_path.write_bytes(canonical_bytes({"revoked": True}))
    with pytest.raises(fixtures.api().StudyExecutionError):
        fixtures.api().recheck_study_execution(result)
