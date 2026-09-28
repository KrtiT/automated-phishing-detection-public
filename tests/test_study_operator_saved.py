"""Pure saved checks preserve both governance versions and original inventories."""

import json

import adopted_study_profile_fixtures as success
import pytest
import study_execution_fixtures as original
from operational_input_fixtures import candidates as candidates
from operational_input_fixtures import manifests as manifests
from study_execution_fixtures import execution_case as execution_case
from study_operator_fixtures import (
    effective_profile,
    install_policy,
    policy_bytes,
    seal,
)
from study_run_record_fixtures import prepared as prepared

from automated_phishing_detection import _adopted_study_records as records
from automated_phishing_detection import _study_run_body as body
from automated_phishing_detection._adopted_study_state import AdoptedStudyProgress
from automated_phishing_detection._adopted_study_verification import (
    validate_child_root,
    verify_saved_adopted_authorization,
)
from automated_phishing_detection._checkpoint_codec import canonical_bytes
from automated_phishing_detection.study_root_retention import hold_study_root


def verify(authorization, snapshot):
    return verify_saved_adopted_authorization(
        snapshot,
        expected_profile_sha256=authorization.profile_sha256,
        expected_envelope_sha256=authorization.envelope_sha256,
    )


def hold_snapshot(authorization, prepared):
    state = AdoptedStudyProgress(authorization)
    authorization.paths.attempt.parent.mkdir(mode=0o700)
    identity = records.study_identity(authorization)
    state.reserve(identity)
    barrier, held = body.records.prediction_barrier(
        prepared.preparation, execution=prepared.execution
    )
    assert held
    contents = (
        ("study-intent.json", state.intent),
        (
            "prediction-barrier.json",
            canonical_bytes(json.loads(barrier) | {"execution": state.execution}),
        ),
        ("study-accounting.json", records.study_accounting(state, "whole_study_hold")),
    )
    public = records.public_summary(state.adopted_execution, contents)
    with hold_study_root(
        state.attempt, authorization.paths.public_summary, expected_identity=identity
    ) as writer:
        for name, content in contents:
            writer.append(name, content)
        return writer.complete(extra_outputs={}, public_summary=public)


@pytest.mark.parametrize("version", [1, 2])
def test_saved_hold_keeps_both_versions_and_ten_members(
    execution_case, prepared, monkeypatch, version
):
    if version == 2:
        install_policy(execution_case, monkeypatch)
        authorization = original.bind(
            execution_case, seal(effective_profile(execution_case))
        )
    else:
        authorization = original.bind(execution_case)
    snapshot = hold_snapshot(authorization, prepared)
    assert len(snapshot.payloads) == 10
    assert (
        verify(authorization, snapshot)["study_profile_sha256"]
        == authorization.profile_sha256
    )


def successful_case(prepared, manifests, monkeypatch, version):
    case = success.make_case(prepared, manifests)
    if version == 2:
        case.policy = policy_bytes()
        case.profile = effective_profile(case)
        monkeypatch.setattr(success, "seal", lambda unused, current: seal(current))
    return case


@pytest.mark.parametrize("version", [1, 2])
def test_saved_success_and_operational_child_prefixes_keep_both_versions(
    prepared, manifests, monkeypatch, version
):
    case = successful_case(prepared, manifests, monkeypatch, version)
    authorization, snapshot = success.saved(case)
    assert len(snapshot.payloads) == 14
    verify(authorization, snapshot)
    for role in ("service", "client"):
        frame, payloads = success.child_context(snapshot, role)
        validate_child_root(authorization, frame, payloads)


def test_operator_saved_success_still_rejects_rehashed_external_profile_substitution(
    prepared, manifests, monkeypatch
):
    case = successful_case(prepared, manifests, monkeypatch, 2)
    authorization, snapshot = success.saved(case, external_pin="0" * 64)
    with pytest.raises(ValueError):
        verify(authorization, snapshot)
    for role in ("service", "client"):
        frame, payloads = success.child_context(snapshot, role)
        with pytest.raises(ValueError):
            validate_child_root(authorization, frame, payloads)
