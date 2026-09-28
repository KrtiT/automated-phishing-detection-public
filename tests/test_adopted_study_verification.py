"""Saved authorization checking joins retained bytes and independent expectations."""

import json
from dataclasses import replace

import adopted_study_fixtures as fixtures
import pytest
import study_execution_fixtures as execution
from study_execution_fixtures import execution_case
from study_run_record_fixtures import prepared

from automated_phishing_detection import _study_run_body as original
from automated_phishing_detection._checkpoint_codec import canonical_bytes

__all__ = ["execution_case", "prepared"]


def snapshot(execution_case, prepared):
    auth = execution.bind(execution_case)
    records = fixtures.api("_adopted_study_records")
    state = fixtures.api("_adopted_study_state").AdoptedStudyProgress(auth)
    auth.paths.attempt.parent.mkdir(mode=0o700)
    state.reserve(records.study_identity(auth))
    barrier, held = original.records.prediction_barrier(
        prepared.preparation, execution=prepared.execution
    )
    assert held
    barrier = canonical_bytes(json.loads(barrier) | {"execution": state.execution})
    accounting = records.study_accounting(state, "whole_study_hold")
    contents = (
        ("study-intent.json", state.intent),
        ("prediction-barrier.json", barrier),
        ("study-accounting.json", accounting),
    )
    public = records.public_summary(state.adopted_execution, contents)
    with fixtures.api("study_root_retention").hold_study_root(
        state.attempt,
        auth.paths.public_summary,
        expected_identity=records.study_identity(auth),
    ) as writer:
        for name, content in contents:
            writer.append(name, content)
        retained = writer.complete(extra_outputs={}, public_summary=public)
    return auth, retained


def verify(auth, retained, **changes):
    return fixtures.api(
        "_adopted_study_verification"
    ).verify_saved_adopted_authorization(
        retained,
        **(
            {
                "expected_profile_sha256": auth.profile_sha256,
                "expected_envelope_sha256": auth.envelope_sha256,
            }
            | changes
        ),
    )


def test_pure_saved_verification_accepts_exact_independent_expectations(
    execution_case, prepared
):
    auth, retained = snapshot(execution_case, prepared)
    result = verify(auth, retained)
    assert result["study_profile_sha256"] == auth.profile_sha256


@pytest.mark.parametrize(
    "name", ["expected_profile_sha256", "expected_envelope_sha256"]
)
def test_saved_context_cannot_supply_its_own_approved_expectations(
    execution_case, prepared, name
):
    auth, retained = snapshot(execution_case, prepared)
    with pytest.raises(ValueError):
        verify(auth, retained, **{name: "0" * 64})


@pytest.mark.parametrize(
    "name", ["attempt/study-intent.json", "attempt/evidence/study-intent.json"]
)
def test_mixed_candidate_adopted_intent_is_rejected(execution_case, prepared, name):
    auth, retained = snapshot(execution_case, prepared)
    records = fixtures.api("_adopted_study_records")
    intent = json.loads(retained.payload(name))
    legacy = records.decoded(intent["scientific_intent_bytes"])
    mixed = replace(
        retained,
        payloads=tuple(
            (key, legacy if key == name else content)
            for key, content in retained.payloads
        ),
    )
    with pytest.raises(ValueError):
        verify(auth, mixed)


def test_changed_policy_bytes_with_rehashed_intent_still_reject(
    execution_case, prepared
):
    auth, retained = snapshot(execution_case, prepared)
    records = fixtures.api("_adopted_study_records")
    intent = json.loads(retained.payload("attempt/study-intent.json"))
    intent["policy_bytes"] = records.encoded(b"{}\n")
    changed = canonical_bytes(intent)
    replaced = replace(
        retained,
        payloads=tuple(
            (name, changed if name.endswith("study-intent.json") else content)
            for name, content in retained.payloads
        ),
    )
    with pytest.raises(ValueError):
        verify(auth, replaced)
