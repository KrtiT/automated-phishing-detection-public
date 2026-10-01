"""Authenticate all historical prefix admissions without granting live authority."""

from hashlib import sha256

from . import _adopted_study_cell_evidence as evidence
from . import _study_run_schema as schema
from ._adopted_study_ledger_validation import _entry, _frame, _predecessor
from ._stopped_study_process import historical_process

_FIELDS = {
    "schema_version",
    "protocol",
    "intent_sha256",
    "barrier_sha256",
    "handoff_sha256",
    "source_results_sha256",
    "accepted_inputs_sha256",
    "admissions",
    "cell_acceptances",
}


def _header(ledger, contents, count):
    schema.keys(ledger, _FIELDS)
    schema.require(
        type(ledger["schema_version"]) is int and ledger["schema_version"] == 1
    )
    schema.require(ledger["protocol"] == "study-authorization-ledger-v1")
    for key, name in (
        ("intent_sha256", "study-intent.json"),
        ("barrier_sha256", "prediction-barrier.json"),
        ("source_results_sha256", "source-results.json"),
    ):
        schema.require(ledger[key] == sha256(contents[name]).hexdigest())
    source = schema.load(contents["source-results.json"])
    schema.require(ledger["accepted_inputs_sha256"] == source["accepted_inputs_sha256"])
    entries = ledger["admissions"]
    schema.require(
        type(entries) is list and 2 + 2 * count <= len(entries) <= 4 + 2 * count
    )
    schema.require(type(ledger["cell_acceptances"]) is list)
    schema.require(len(ledger["cell_acceptances"]) == count)
    return entries, source["accepted_inputs"]


def _entries(ledger, execution, entries, barrier, count):
    parent_pid = None
    for position, entry in enumerate(entries):
        frame = _entry(entry)
        _frame(frame, execution, ledger, barrier)
        _predecessor(frame, position, entries, ledger)
        schema.require(entry["accepted"] is (position < 2 + 2 * count))
        schema.require(entry["launched_pid"] != frame.parent_pid)
        schema.require(parent_pid is None or parent_pid == frame.parent_pid)
        parent_pid = frame.parent_pid
    return parent_pid


def _sources(ledger, entries, metadata):
    schema.require(
        ledger["handoff_sha256"]
        == metadata["external"]["execution"]["internal_handoff_sha256"]
    )
    for entry, role in zip(entries[:2], ("internal", "external"), strict=True):
        worker = metadata[role]["worker"]
        schema.require(worker["command_sha256"] == entry["command_sha256"])
        schema.require(worker["exit"]["pid"] == entry["launched_pid"])
        schema.require(worker["exit"]["exit_observed"] == entry["exit_observed"])
        schema.require(worker["exit"]["exit_code"] == entry["exit_code"])


def _cell(value, entries, execution, accepted_digest, projection):
    payloads = evidence._payloads(value, projection)
    reservation = evidence._binding(
        payloads["binding_bytes"],
        payloads["descriptor_bytes"],
        execution,
        accepted_digest,
        projection,
    )
    observation, intent = historical_process(payloads, reservation)
    evidence._hashes(payloads, projection)
    for entry in entries:
        evidence._role(
            entry, observation, intent, sha256(payloads["binding_bytes"]).hexdigest()
        )


def validate_ledger(ledger, execution, contents, projections):
    """Return only a historical parent PID after every retained join succeeds."""
    count = len(projections)
    entries, metadata = _header(ledger, contents, count)
    barrier = schema.load(contents["prediction-barrier.json"])
    parent_pid = _entries(ledger, execution, entries, barrier, count)
    _sources(ledger, entries, metadata)
    for position, (value, projection) in enumerate(
        zip(ledger["cell_acceptances"], projections, strict=True)
    ):
        _cell(
            value,
            entries[2 + 2 * position : 4 + 2 * position],
            execution,
            ledger["accepted_inputs_sha256"],
            projection,
        )
    return parent_pid
