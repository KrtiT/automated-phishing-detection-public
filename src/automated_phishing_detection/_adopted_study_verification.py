"""Pure retained-byte authorization joins never grant replay or scientific success."""

import json
from hashlib import sha256
from pathlib import Path

from . import _adopted_study_records as records
from . import _study_root_records as retention
from . import _study_run_public as original
from . import _study_run_schema as schema
from . import execution_receipt as receipt
from ._adopted_study_intent import authenticate_intent as _intent
from ._adopted_study_intent import validate_source_profile


def _reservation(content, expected_digest):
    schema.require(
        type(content) is bytes and sha256(content).hexdigest() == expected_digest
    )
    value = json.loads(content)
    schema.keys(value, {"schema_version", "status", "directory", "identity"})
    schema.require(
        type(value["schema_version"]) is int and value["schema_version"] == 1
    )
    schema.require(value["status"] == "reserved")
    schema.require(receipt._json_bytes(value, "study_reservation") == content)
    execution = value["identity"] | {"reservation_sha256": expected_digest}
    records.scientific_execution(execution)
    return value, execution


def _snapshot(snapshot):
    schema.require(type(snapshot) is retention.StudyRootSnapshot)
    schema.require(type(snapshot.payloads) is tuple)
    for member in snapshot.payloads:
        schema.require(type(member) is tuple and len(member) == 2)
        schema.require(type(member[0]) is str and type(member[1]) is bytes)
    payloads = dict(snapshot.payloads)
    schema.require(len(payloads) == len(snapshot.payloads))
    return payloads


def _publication(payloads, reservation, execution):
    public = json.loads(payloads["public-summary.json"])
    success = public.get("status") == "study_evidence_published"
    order = retention.ORDER if success else retention.HOLD_ORDER
    contents = {name: payloads[f"attempt/{name}"] for name in order}
    outputs = contents | {
        name: payloads[f"attempt/evidence/{name}"]
        for name in (retention.EXTRA_NAMES if success else ())
    }
    expected_names = {
        "attempt/reservation.json",
        "public-summary.json",
        "attempt/finalize.claim",
        "attempt/outcome.json",
    }
    expected_names |= {f"attempt/{name}" for name in order}
    expected_names |= {f"attempt/evidence/{name}" for name in outputs}
    schema.require(set(payloads) == expected_names)
    for name, content in contents.items():
        schema.require(content == payloads[f"attempt/evidence/{name}"])
    attempt = receipt.Attempt(
        Path(reservation["directory"]), execution["reservation_sha256"]
    )
    content = records.public_bytes(
        attempt, reservation["identity"], contents, outputs, success, public
    )
    schema.require(content == payloads["public-summary.json"])
    for name, content in retention.publication(attempt, outputs, content).items():
        schema.require(payloads[f"attempt/{name}"] == content)
    return contents, success


def verify_saved_adopted_authorization(
    snapshot, *, expected_profile_sha256, expected_envelope_sha256
):
    """Check immutable authorization consistency using independently supplied pins."""
    from ._adopted_study_ledger_validation import validate_ledger

    try:
        payloads = _snapshot(snapshot)
        reservation, execution = _reservation(
            payloads["attempt/reservation.json"], snapshot.reservation_sha256
        )
        profile = _intent(
            payloads["attempt/study-intent.json"],
            execution,
            expected_profile_sha256,
            expected_envelope_sha256,
        )
        schema.require(reservation["directory"] == profile["paths"]["attempt"])
        contents, success = _publication(payloads, reservation, execution)
        if success:
            validate_source_profile(contents["source-results.json"], profile)
        accounting = schema.load(contents[retention.ORDER[3]])
        validate_ledger(
            accounting["authorization_ledger"], execution, contents, success
        )
        return execution
    except Exception:
        raise schema.StudyRunRecordError(
            "invalid_adopted_study_authorization"
        ) from None


def validate_child_root(authorization, frame, payloads):
    """Join a live frame to exact retained root checkpoints without private inputs."""
    expected = {"reservation.json", "study-intent.json", "prediction-barrier.json"}
    if frame.role in ("service", "client"):
        expected.add("source-results.json")
    schema.keys(payloads, expected)
    reservation, execution = _reservation(
        payloads["reservation.json"], frame.root_reservation_sha256
    )
    schema.same(reservation["identity"], records.study_identity(authorization))
    schema.require(reservation["directory"] == str(authorization.paths.attempt))
    schema.require(frame.profile_sha256 == authorization.profile_sha256)
    schema.require(frame.envelope_sha256 == authorization.envelope_sha256)
    schema.require(
        sha256(payloads["study-intent.json"]).hexdigest() == frame.intent_sha256
    )
    schema.require(
        sha256(payloads["prediction-barrier.json"]).hexdigest() == frame.barrier_sha256
    )
    profile = _intent(
        payloads["study-intent.json"],
        execution,
        frame.profile_sha256,
        frame.envelope_sha256,
    )
    _child_barrier(frame, payloads, execution, profile)


def _child_barrier(frame, payloads, execution, profile):
    barrier = original._barrier(
        payloads["prediction-barrier.json"],
        records.scientific_execution(execution),
        True,
    )
    schema.require(
        barrier["study_preparation_reservation_sha256"]
        == frame.preparation_reservation_sha256
    )
    schema.require(
        barrier["study_preparation_complete_sha256"]
        == frame.preparation_completion_sha256
    )
    if "source-results.json" in payloads:
        source = payloads["source-results.json"]
        original._sources(source, records.scientific_execution(execution), barrier)
        validate_source_profile(source, profile)
        schema.require(sha256(source).hexdigest() == frame.predecessor_sha256)
        schema.require(
            schema.load(source)["accepted_inputs_sha256"]
            == frame.accepted_inputs_sha256
        )
