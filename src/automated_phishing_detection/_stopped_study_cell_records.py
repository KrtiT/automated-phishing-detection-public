"""Closed supplied inventory and stopped-service root projection joins."""

from hashlib import sha256

from . import _adopted_study_records as records
from . import _study_run_schema as schema
from ._adopted_study_ledger_validation import _entry

ATTEMPT_NAMES = frozenset(
    f"attempt/{name}"
    for name in (
        "reservation.json",
        "process-pair-intent.json",
        "service-intent.json",
        "service-started.json",
        "service-process.json",
        "service-role.json",
        "service-ready.json",
        "service-stop.json",
        "service-cleanup.json",
        "process-pair.json",
        "finalize.claim",
        "outcome.json",
    )
)
INPUT_NAMES = frozenset(("descriptor.json", "binding.json", "manifest"))


def authenticate_inventory(payloads, expected, names):
    schema.require(type(payloads) is tuple)
    schema.keys(expected, names)
    for member in payloads:
        schema.require(type(member) is tuple and len(member) == 2)
        schema.require(type(member[0]) is str and type(member[1]) is bytes)
    values = dict(payloads)
    schema.require(len(values) == len(payloads))
    schema.keys(values, names)
    hashes = {}
    for name, content in values.items():
        schema.operational.digest(expected[name])
        hashes[name] = sha256(content).hexdigest()
        schema.require(hashes[name] == expected[name])
    return values, tuple(sorted(hashes.items()))


def stopped_admission(root):
    accounting = schema.load(root.accounting_bytes)
    entries = accounting["authorization_ledger"]["admissions"]
    schema.require(len(entries) == 2 + 2 * len(root.accepted_ordinals) + 1)
    entry = entries[-1]
    frame = _entry(entry)
    schema.require(entry["role"] == "service" and entry["accepted"] is False)
    schema.require(entry["exit_observed"] is True and entry["exit_code"] == 0)
    schema.require(entry["launched_pid"] != root.parent_pid)
    scientific = schema.load(records.decoded(accounting["scientific_accounting_bytes"]))
    return entry, frame, scientific["cells"][root.stopped_ordinal - 1]


def authenticate_projection(projection, reservation, attempt):
    schema.require(projection["stage"] == "observation")
    schema.require(projection["publishing"] is False)
    schema.require(projection["observation_sha256"] is None)
    schema.require(projection["reservation_sha256"] == reservation)
    schema.require(
        projection["progress_sha256"]
        == sha256(attempt["attempt/process-pair.json"]).hexdigest()
    )
