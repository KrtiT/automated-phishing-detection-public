"""Join existing cell observations to admissions without retaining source rows."""

from hashlib import sha256

from . import _adopted_study_records as records
from . import _operational_cell_records as cells
from . import _study_run_schema as schema
from ._operational_cell_process_records import _observation
from ._operational_process_records import ProcessObservation
from ._study_admission import decode_admission_frame
from ._study_execution_policy import DEADLINES


def evidence(retained, pair_intent_bytes):
    return {
        "cell_ordinal": retained.cell.ordinal,
        "descriptor_bytes": records.encoded(retained.descriptor_bytes),
        "binding_bytes": records.encoded(retained.binding_bytes),
        "observation_bytes": records.encoded(retained.observation.record),
        "pair_intent_bytes": records.encoded(pair_intent_bytes),
    }


def _role(entry, observation, intent, binding_digest):
    role = entry["role"]
    schema.require(role in ("service", "client"))
    observed = observation[role]
    schema.require(observed["pid"] == entry["launched_pid"])
    schema.require(observed["exit_observed"] is True and observed["exit_code"] == 0)
    schema.require(entry["exit_observed"] is True and entry["exit_code"] == 0)
    schema.require(intent[f"{role}_command_sha256"] == entry["command_sha256"])
    frame = decode_admission_frame(records.decoded(entry["frame_bytes"]))
    schema.require(frame.cell_binding_sha256 == binding_digest)


def _binding(content, descriptor, execution, accepted_digest, projection):
    bound = schema.load(content)
    cells.binding(bound)
    schema.require(bound["descriptor_sha256"] == sha256(descriptor).hexdigest())
    described = schema.load(descriptor)
    cells.descriptor(described)
    schema.same(described["cell"], projection["cell"])
    schema.require(
        described["root_reservation_sha256"] == execution["reservation_sha256"]
    )
    schema.require(described["accepted_inputs_sha256"] == accepted_digest)
    schema.require(bound["cell_reservation_sha256"] == projection["reservation_sha256"])
    return bound["cell_reservation_sha256"]


def _payloads(value, projection):
    schema.keys(
        value,
        {
            "cell_ordinal",
            "descriptor_bytes",
            "binding_bytes",
            "observation_bytes",
            "pair_intent_bytes",
        },
    )
    schema.require(type(value["cell_ordinal"]) is int)
    schema.require(value["cell_ordinal"] == projection["cell"]["ordinal"])
    return {
        name: records.decoded(value[name])
        for name in (
            "descriptor_bytes",
            "binding_bytes",
            "observation_bytes",
            "pair_intent_bytes",
        )
    }


def _observations(payloads, reservation):
    observation, intent = (
        schema.load(payloads[name])
        for name in ("observation_bytes", "pair_intent_bytes")
    )
    schema.require(
        observation["reservation_sha256"] == intent["reservation_sha256"] == reservation
    )
    _observation(
        payloads["observation_bytes"],
        ProcessObservation(payloads["observation_bytes"]),
        reservation,
    )
    schema.keys(
        intent,
        {
            "schema_version",
            "reservation_sha256",
            "service_command_sha256",
            "client_command_sha256",
            "deadlines",
        },
    )
    schema.deadlines(intent["deadlines"])
    schema.same(intent["deadlines"], dict(DEADLINES))
    schema.require(
        type(intent["schema_version"]) is int and intent["schema_version"] == 1
    )
    return observation, intent


def validate(value, entries, execution, accepted_digest, projection):
    payloads = _payloads(value, projection)
    reservation = _binding(
        payloads["binding_bytes"],
        payloads["descriptor_bytes"],
        execution,
        accepted_digest,
        projection,
    )
    observation, intent = _observations(payloads, reservation)
    _hashes(payloads, projection)
    for entry in entries:
        _role(entry, observation, intent, sha256(payloads["binding_bytes"]).hexdigest())


def _hashes(payloads, projection):
    schema.require(
        sha256(payloads["observation_bytes"]).hexdigest()
        == projection["observation_sha256"]
    )
    hashes = projection["snapshot_sha256"]
    for name, key in (
        ("process-pair.json", "observation_bytes"),
        ("process-pair-intent.json", "pair_intent_bytes"),
    ):
        for prefix in ("attempt", "attempt/evidence"):
            schema.require(
                sha256(payloads[key]).hexdigest() == hashes[f"{prefix}/{name}"]
            )
