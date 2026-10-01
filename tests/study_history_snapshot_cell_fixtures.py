"""Invented operational receipts with opaque request and run payloads."""

import json

from adopted_study_profile_evidence_fixtures import _frames
from study_history_snapshot_source_fixtures import (
    digest,
    finish,
    hashes,
    reserve,
)

from automated_phishing_detection import _adopted_study_records as records
from automated_phishing_detection._checkpoint_codec import canonical_bytes
from automated_phishing_detection._operational_cell_protocol import PRIVATE_NAMES


def _lifecycle(observation, descriptor):
    common = dict(
        schema_version=1,
        pid=observation["service"]["pid"],
        workload=descriptor["cell"]["workload"],
    )
    return {
        "service-ready.json": canonical_bytes(
            common | dict(status="ready", host="127.0.0.1", port=12345)
        ),
        "service-cleanup.json": canonical_bytes(common | dict(status="clean")),
        "service-stop.json": canonical_bytes(dict(status="requested")),
    }


def _process(value, reservation, descriptor):
    observation = json.loads(records.decoded(value["observation_bytes"]))
    observation["reservation_sha256"] = reservation
    payloads = _lifecycle(observation, descriptor)
    observation.update(
        readiness_sha256=digest(payloads["service-ready.json"]),
        cleanup_sha256=digest(payloads["service-cleanup.json"]),
    )
    intent = json.loads(records.decoded(value["pair_intent_bytes"]))
    intent["reservation_sha256"] = reservation
    payloads.update(
        {
            "process-pair.json": canonical_bytes(observation),
            "process-pair-intent.json": canonical_bytes(intent),
        }
    )
    value.update(
        observation_bytes=records.encoded(payloads["process-pair.json"]),
        pair_intent_bytes=records.encoded(payloads["process-pair-intent.json"]),
    )
    return payloads, observation, intent


def _roles(outputs, observation, intent, descriptor, binding, primary):
    for role in ("service", "client"):
        command = intent[f"{role}_command_sha256"]
        observed = observation[role]
        for name, content in (
            ("intent", {"command_sha256": command}),
            ("started", {"pid": observed["pid"]}),
            ("process", observed),
        ):
            outputs[f"{role}-{name}.json"] = canonical_bytes(content)
        role_record = dict(
            schema_version=1,
            protocol="operational-role-v1",
            role=role,
            binding_sha256=digest(binding),
            pid=observed["pid"],
            command_sha256=command,
            base_url="http://127.0.0.1:12345",
            workload=descriptor["cell"]["workload"],
        )
        outputs[f"{role}-role.json"] = canonical_bytes(
            role_record | (primary if role == "service" else {})
        )


def _identity(case, value):
    return dict(
        kind="operational_cell",
        protocol="operational-cell-v1",
        **case.source["accepted_inputs"]["execution"],
        operational_profile_sha256=case.execution["operational_profile_sha256"],
        root_reservation_sha256=case.execution["reservation_sha256"],
        descriptor_sha256=digest(records.decoded(value["descriptor_bytes"])),
    )


def _binding(value, reservation):
    binding = json.loads(records.decoded(value["binding_bytes"]))
    binding["cell_reservation_sha256"] = reservation
    value["binding_bytes"] = records.encoded(canonical_bytes(binding))
    return canonical_bytes(binding)


def _cell(case, value, projection):
    descriptor = json.loads(records.decoded(value["descriptor_bytes"]))
    identity = _identity(case, value)
    ordinal = descriptor["cell"]["ordinal"]
    values = {}
    reserve(
        values,
        identity,
        case.profile["paths"]["cells-dir"] + f"/cell-{ordinal:03d}-attempt",
    )
    reservation = identity["reservation_sha256"]
    binding = _binding(value, reservation)
    outputs = {name: b"opaque science " + name.encode() for name in PRIVATE_NAMES}
    process, observation, intent = _process(value, reservation, descriptor)
    outputs.update(process)
    _roles(
        outputs,
        observation,
        intent,
        descriptor,
        binding,
        case.source["accepted_inputs"]["primary"],
    )
    _publish_cell(values, outputs, identity, descriptor)
    projection.update(
        reservation_sha256=reservation,
        snapshot_sha256=hashes(values),
        observation_sha256=digest(outputs["process-pair.json"]),
    )
    return ordinal, reservation, tuple(values.items())


def _publish_cell(values, outputs, execution, descriptor):
    for name, content in outputs.items():
        values[f"attempt/{name}"] = content
        values[f"attempt/evidence/{name}"] = content
    public = dict(
        schema_version=1,
        protocol="operational-cell-v1",
        status="operational_evidence_published",
        execution=execution,
        cell=descriptor["cell"],
        summary=None,
        private_sha256=hashes(outputs),
    )
    finish(values, public, outputs)


def cells(case):
    ledger = case.accounting["authorization_ledger"]
    case.cells = tuple(
        _cell(case, value, projection)
        for value, projection in zip(
            ledger["cell_acceptances"], case.scientific["cells"], strict=False
        )
    )
    bindings = {
        value["cell_ordinal"]: digest(records.decoded(value["binding_bytes"]))
        for value in ledger["cell_acceptances"]
    }
    _frames(ledger, case.execution, bindings)
