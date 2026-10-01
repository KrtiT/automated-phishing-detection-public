"""Closed retained cell facts; bytes do not attest custody or holder exit."""

from . import _operational_cell_records as cells
from . import _study_series_prefix_records as prefix
from ._operational_cell_process_records import _OBSERVATION_FIELDS
from ._operational_cell_protocol import SNAPSHOT_NAMES
from ._operational_process_records import _bytes
from ._stopped_study_process import _observation as clean_observation
from ._stopped_study_process import historical_process
from ._study_execution_policy import DEADLINES
from ._study_series_accounting_ledger import process_role
from ._study_series_accounting_records import decoded, digest, schema
from ._study_series_cell_records import identity
from ._study_series_ledger_records import FACTS, STAGES


def _inputs(slot, context):
    described, bound, reservation = (
        slot[name]
        for name in ("descriptor_bytes", "binding_bytes", "reservation_sha256")
    )
    if described is not None:
        content = decoded(described)
        value = schema.loads(content)
        schema.require(cells.descriptor(value).ordinal == slot["ordinal"])
        schema.require(value["accepted_inputs_sha256"] == digest(context["metadata"]))
        schema.require(value["root_reservation_sha256"] == context["pins"][1])
    if reservation is not None:
        schema.digest(reservation)
        directory = (
            context["profile"]["paths"]["cells_dir"]
            + f"/cell-{slot['ordinal']:03d}-attempt"
        )
        metadata = context["metadata"]
        projected = identity(metadata, decoded(described), digest(metadata))
        expected = prefix.reservation(directory, projected)
        schema.require(reservation == digest(expected))
    if bound is not None:
        value = schema.loads(decoded(bound))
        cells.binding(value)
        schema.require(
            reservation is not None and value["cell_reservation_sha256"] == reservation
        )
        schema.require(value["descriptor_sha256"] == digest(decoded(described)))


def _pair(content, slot, context):
    from ._study_series_accounting_ledger import commands

    value = schema.loads(content, canonical=False)
    schema.require(content == _bytes(value))
    expected = dict(
        schema_version=1,
        reservation_sha256=slot["reservation_sha256"],
        deadlines=dict(DEADLINES),
    )
    expected.update(
        {role + "_command_sha256": pin for role, pin in commands(context, slot).items()}
    )
    schema.same(value, expected)
    return value


def _observation_record(content, slot):
    value = schema.loads(content, canonical=False)
    schema.require(content == _bytes(value))
    schema.keys(value, _OBSERVATION_FIELDS)
    schema.require(
        type(value["schema_version"]) is int and value["schema_version"] == 1
    )
    schema.require(
        slot["reservation_sha256"] is not None
        and value["reservation_sha256"] == slot["reservation_sha256"]
    )
    schema.require(
        value["research_accepted"] is False and type(value["stop_sent"]) is bool
    )
    schema.require(value["status"] in ("pending", "observed", "failed"))
    schema.require(value["failure"] is None or type(value["failure"]) is str)
    schema.require(
        type(value["record_failures"]) is list
        and all(type(name) is str for name in value["record_failures"])
    )
    return value


def _observation(content, slot, entries):
    value = _observation_record(content, slot)
    if value["status"] == "observed":
        clean_observation(content, slot["reservation_sha256"])
    parent = (
        None
        if not entries
        else schema.loads(decoded(entries[0]["frame_bytes"]), canonical=False)[
            "parent_pid"
        ]
    )
    for role in ("service", "client"):
        entry = next((entry for entry in entries if entry["role"] == role), None)
        process_role(value[role], entry, parent)
    for name in ("readiness_sha256", "cleanup_sha256"):
        if value[name] is not None:
            schema.digest(value[name])
    schema.require(
        value["service"]["pid"] is None
        or value["service"]["pid"] != value["client"]["pid"]
    )
    return value


def _snapshot(slot, buffers):
    schema.require(slot["binding_bytes"] is not None)
    pins = slot["snapshot_sha256"]
    schema.keys(pins, SNAPSHOT_NAMES)
    for pin in pins.values():
        schema.digest(pin)
    schema.require(pins["attempt/reservation.json"] == slot["reservation_sha256"])
    for name, content in buffers.items():
        if name in ("observation_bytes", "pair_intent_bytes") and content is not None:
            filename = (
                "process-pair.json"
                if name == "observation_bytes"
                else "process-pair-intent.json"
            )
            schema.require(pins["attempt/" + filename] == digest(content))
    for name in SNAPSHOT_NAMES:
        if name.startswith("attempt/evidence/"):
            schema.require(pins[name] == pins[name.replace("/evidence/", "/")])


def _accepted(slot, buffers, entries):
    schema.require(slot["stage"] is slot["progress_bytes"] is None)
    schema.require(slot["publishing"] is slot["holders_closed"] is True)
    schema.require(
        all(
            buffers[name] is not None
            for name in (
                "descriptor_bytes",
                "binding_bytes",
                "pair_intent_bytes",
                "observation_bytes",
            )
        )
    )
    observation, unused = historical_process(buffers, slot["reservation_sha256"])
    schema.require(len(entries) == 2)
    for entry in entries:
        schema.require(observation[entry["role"]]["pid"] == entry["launched_pid"])
    _snapshot(slot, buffers)
    _process_pins(slot["snapshot_sha256"], observation, entries)


def _process_pins(pins, observation, entries):
    for name, field in (
        ("service-ready.json", "readiness_sha256"),
        ("service-cleanup.json", "cleanup_sha256"),
    ):
        schema.require(pins["attempt/" + name] == observation[field])
    for entry in entries:
        role = entry["role"]
        for suffix, value in (
            ("process", observation[role]),
            ("started", {"pid": entry["launched_pid"]}),
            ("intent", {"command_sha256": entry["command_sha256"]}),
        ):
            schema.require(
                pins[f"attempt/{role}-{suffix}.json"] == digest(_bytes(value))
            )
    schema.require(
        pins["attempt/service-stop.json"] == digest(_bytes({"status": "requested"}))
    )


def validate_slot(slot, ordinal, context, entries):
    schema.keys(slot, {"ordinal", "status", *FACTS})
    schema.require(type(slot["ordinal"]) is int and slot["ordinal"] == ordinal)
    schema.require(slot["status"] in ("accepted", "stopped", "unattempted"))
    if slot["status"] == "unattempted":
        schema.require(all(slot[name] is None for name in FACTS))
        return
    _inputs(slot, context)
    buffers = {
        name: None if slot[name] is None else decoded(slot[name])
        for name in FACTS
        if name.endswith("_bytes")
    }
    if buffers["pair_intent_bytes"] is not None:
        _pair(buffers["pair_intent_bytes"], slot, context)
    for name in ("observation_bytes", "progress_bytes"):
        if buffers[name] is not None:
            _observation(buffers[name], slot, entries)
    if slot["status"] == "accepted":
        _accepted(slot, buffers, entries)
    else:
        schema.require(slot["stage"] in STAGES and slot["holders_closed"] is False)
        schema.require(type(slot["publishing"]) is bool)
        if slot["snapshot_sha256"] is not None:
            schema.require(slot["publishing"] is True)
            _snapshot(slot, buffers)
