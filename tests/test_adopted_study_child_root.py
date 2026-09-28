"""A child must join the exact successful retained barrier before private access."""

import json
import os
from dataclasses import replace
from hashlib import sha256

import adopted_study_fixtures as fixtures
import pytest
import study_execution_fixtures as execution
from study_execution_fixtures import execution_case
from study_run_record_fixtures import capacity, prepared

from automated_phishing_detection._checkpoint_codec import canonical_bytes
from automated_phishing_detection._study_admission import AdmissionFrame

__all__ = ["execution_case", "prepared"]


def child_case(execution_case, prepared):
    auth = execution.bind(execution_case)
    records = fixtures.api("_adopted_study_records")
    state = fixtures.api("_adopted_study_state").AdoptedStudyProgress(auth)
    auth.paths.attempt.parent.mkdir(mode=0o700)
    state.reserve(records.study_identity(auth))
    original = fixtures.api("_study_run_records")
    barrier, held = original.prediction_barrier(
        capacity(prepared), execution=prepared.execution
    )
    assert not held
    barrier = canonical_bytes(json.loads(barrier) | {"execution": state.execution})
    return (
        auth,
        child_frame(auth, state, barrier),
        {
            "reservation.json": (auth.paths.attempt / "reservation.json").read_bytes(),
            "study-intent.json": state.intent,
            "prediction-barrier.json": barrier,
        },
    )


def child_frame(auth, state, barrier):
    metadata = json.loads(barrier)
    return AdmissionFrame(
        "internal",
        auth.profile_sha256,
        auth.envelope_sha256,
        os.getpid(),
        "c" * 64,
        state.attempt.reservation_sha256,
        sha256(state.intent).hexdigest(),
        sha256(barrier).hexdigest(),
        metadata["study_preparation_reservation_sha256"],
        metadata["study_preparation_complete_sha256"],
        None,
        None,
        None,
    )


def test_exact_retained_child_context_accepts(execution_case, prepared):
    auth, frame, payloads = child_case(execution_case, prepared)
    fixtures.api("_adopted_study_verification").validate_child_root(
        auth, frame, payloads
    )


@pytest.mark.parametrize(
    "field",
    [
        "profile_sha256",
        "envelope_sha256",
        "root_reservation_sha256",
        "intent_sha256",
        "barrier_sha256",
        "preparation_reservation_sha256",
        "preparation_completion_sha256",
    ],
)
def test_child_context_rejects_each_substituted_join(execution_case, prepared, field):
    auth, frame, payloads = child_case(execution_case, prepared)
    with pytest.raises(ValueError):
        fixtures.api("_adopted_study_verification").validate_child_root(
            auth, replace(frame, **{field: "0" * 64}), payloads
        )


def test_rehashed_hold_barrier_does_not_authorize_child(execution_case, prepared):
    auth, frame, payloads = child_case(execution_case, prepared)
    value = json.loads(payloads["prediction-barrier.json"])
    value["status"] = "whole_study_hold"
    payloads["prediction-barrier.json"] = canonical_bytes(value)
    frame = replace(
        frame, barrier_sha256=sha256(payloads["prediction-barrier.json"]).hexdigest()
    )
    with pytest.raises(ValueError):
        fixtures.api("_adopted_study_verification").validate_child_root(
            auth, frame, payloads
        )


def test_rehashed_intent_cannot_override_adopted_deadlines(execution_case, prepared):
    auth, frame, payloads = child_case(execution_case, prepared)
    records = fixtures.api("_adopted_study_records")
    value = json.loads(payloads["study-intent.json"])
    scientific = json.loads(records.decoded(value["scientific_intent_bytes"]))
    scientific["protective_deadlines_seconds"]["startup"] = 301
    value["scientific_intent_bytes"] = records.encoded(canonical_bytes(scientific))
    payloads["study-intent.json"] = canonical_bytes(value)
    frame = replace(
        frame, intent_sha256=sha256(payloads["study-intent.json"]).hexdigest()
    )
    with pytest.raises(ValueError):
        fixtures.api("_adopted_study_verification").validate_child_root(
            auth, frame, payloads
        )
