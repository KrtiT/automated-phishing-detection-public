"""Rehashed fixture evidence must still match independently approved scope."""

import json

import pytest
from adopted_study_profile_fixtures import child_context, make_case, saved
from operational_input_fixtures import candidates, manifests
from study_run_record_fixtures import prepared

from automated_phishing_detection import _adopted_study_records as records
from automated_phishing_detection._adopted_study_intent import authenticate_intent
from automated_phishing_detection._adopted_study_verification import (
    validate_child_root,
    verify_saved_adopted_authorization,
)
from automated_phishing_detection._checkpoint_codec import canonical_bytes

__all__ = ["candidates", "manifests", "prepared"]


def verify(authorization, snapshot):
    return verify_saved_adopted_authorization(
        snapshot,
        expected_profile_sha256=authorization.profile_sha256,
        expected_envelope_sha256=authorization.envelope_sha256,
    )


def test_saved_success_and_child_contexts_accept_matching_scope(prepared, manifests):
    authorization, snapshot = saved(make_case(prepared, manifests))
    assert len(snapshot.payloads) == 14
    verify(authorization, snapshot)
    for role in ("service", "client"):
        frame, payloads = child_context(snapshot, role)
        validate_child_root(authorization, frame, payloads)


def test_rehashed_external_profile_substitution_fails_saved_verification(
    prepared, manifests
):
    case = make_case(prepared, manifests)
    authorization, matching = saved(case)
    verify(authorization, matching)
    changed_authorization, changed = saved(case, external_pin="0" * 64)
    assert authorization.profile_bytes == changed_authorization.profile_bytes
    assert authorization.envelope_bytes == changed_authorization.envelope_bytes
    with pytest.raises(ValueError):
        verify(authorization, changed)


@pytest.mark.parametrize("role", ("service", "client"))
def test_rehashed_external_profile_substitution_fails_child_context(
    prepared, manifests, role
):
    authorization, snapshot = saved(
        make_case(prepared, manifests), external_pin="0" * 64
    )
    frame, payloads = child_context(snapshot, role)
    with pytest.raises(ValueError):
        validate_child_root(authorization, frame, payloads)


def test_approved_source_scope_must_match_retained_component(prepared, manifests):
    authorization, snapshot = saved(make_case(prepared, manifests, changed_scope=True))
    with pytest.raises(ValueError):
        verify(authorization, snapshot)


@pytest.mark.parametrize("field", ("runtime_sha256", "source_spec_sha256"))
def test_rehashed_execution_must_match_pinned_operational_component(
    prepared, manifests, field
):
    authorization, snapshot = saved(make_case(prepared, manifests))
    value = json.loads(snapshot.payload("attempt/study-intent.json"))
    scientific = json.loads(records.decoded(value["scientific_intent_bytes"]))
    execution = value["execution"] | {field: "0" * 64}
    value["execution"] = execution
    scientific["execution"] = records.scientific_execution(execution)
    value["scientific_intent_bytes"] = records.encoded(canonical_bytes(scientific))
    with pytest.raises(ValueError):
        authenticate_intent(
            canonical_bytes(value),
            execution,
            authorization.profile_sha256,
            authorization.envelope_sha256,
        )
