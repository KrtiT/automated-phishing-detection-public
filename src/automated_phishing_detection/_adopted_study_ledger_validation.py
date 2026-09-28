"""Retained admission consistency distinguishes issuance from real observations."""

from hashlib import sha256

from . import _adopted_study_records as records
from . import _study_run_schema as schema
from ._study_admission import decode_admission_frame


def _entry(value):
    schema.keys(
        value,
        {
            "role",
            "command_sha256",
            "cell_ordinal",
            "frame_bytes",
            "frame_sha256",
            "issued",
            "launched_pid",
            "exit_observed",
            "exit_code",
            "accepted",
        },
    )
    content = records.decoded(value["frame_bytes"])
    schema.require(sha256(content).hexdigest() == value["frame_sha256"])
    frame = decode_admission_frame(content)
    schema.require(value["issued"] is True)
    schema.require(
        value["role"] == frame.role and value["command_sha256"] == frame.command_sha256
    )
    pid, seen, code = value["launched_pid"], value["exit_observed"], value["exit_code"]
    schema.require(pid is None or type(pid) is int and pid > 0)
    schema.require(type(seen) is bool and type(value["accepted"]) is bool)
    schema.require(type(code) is int and pid is not None if seen else code is None)
    schema.require(not value["accepted"] or seen and code == 0)
    return frame


def _frame(frame, execution, ledger, barrier):
    schema.require(frame.profile_sha256 == execution["study_profile_sha256"])
    schema.require(frame.envelope_sha256 == execution["adoption_envelope_sha256"])
    schema.require(frame.root_reservation_sha256 == execution["reservation_sha256"])
    schema.require(frame.intent_sha256 == ledger["intent_sha256"])
    schema.require(frame.barrier_sha256 == ledger["barrier_sha256"])
    schema.require(
        frame.preparation_reservation_sha256
        == barrier["study_preparation_reservation_sha256"]
    )
    schema.require(
        frame.preparation_completion_sha256
        == barrier["study_preparation_complete_sha256"]
    )


def _position(frame, position, entry):
    expected = (
        "internal"
        if position == 0
        else "external"
        if position == 1
        else "service"
        if position % 2 == 0
        else "client"
    )
    schema.require(frame.role == expected)
    ordinal = None if position < 2 else position // 2
    schema.require(
        entry["cell_ordinal"] is None
        if ordinal is None
        else type(entry["cell_ordinal"]) is int
    )
    schema.require(entry["cell_ordinal"] == ordinal)


def _predecessor(frame, position, entries, ledger):
    _position(frame, position, entries[position])
    if position == 1:
        schema.require(entries[0]["accepted"])
        schema.require(frame.predecessor_sha256 == ledger["handoff_sha256"])
    elif position >= 2:
        schema.require(entries[1]["accepted"])
        schema.require(frame.predecessor_sha256 == ledger["source_results_sha256"])
        schema.require(frame.accepted_inputs_sha256 == ledger["accepted_inputs_sha256"])
        if frame.role == "client":
            previous = _entry(entries[position - 1])
            schema.require(entries[position - 1]["launched_pid"] is not None)
            schema.require(frame.cell_binding_sha256 == previous.cell_binding_sha256)
        elif position > 2:
            schema.require(
                entries[position - 1]["accepted"] and entries[position - 2]["accepted"]
            )


def validate_ledger(ledger, execution, contents, success):
    schema.keys(
        ledger,
        {
            "schema_version",
            "protocol",
            "intent_sha256",
            "barrier_sha256",
            "handoff_sha256",
            "source_results_sha256",
            "accepted_inputs_sha256",
            "admissions",
            "cell_acceptances",
        },
    )
    schema.require(
        type(ledger["schema_version"]) is int and ledger["schema_version"] == 1
    )
    schema.require(ledger["protocol"] == "study-authorization-ledger-v1")
    schema.require(
        ledger["intent_sha256"] == sha256(contents["study-intent.json"]).hexdigest()
    )
    entries = ledger["admissions"]
    schema.require(type(entries) is list and len(entries) <= 252)
    schema.require(type(ledger["cell_acceptances"]) is list)
    if not success:
        _hold(ledger, entries)
        return
    _successful(ledger, entries, execution, contents)


def _hold(ledger, entries):
    schema.require(not entries and not ledger["cell_acceptances"])
    schema.require(
        all(
            ledger[name] is None
            for name in (
                "barrier_sha256",
                "handoff_sha256",
                "source_results_sha256",
                "accepted_inputs_sha256",
            )
        )
    )


def _successful(ledger, entries, execution, contents):
    schema.require(len(entries) == 252)
    schema.require(
        ledger["barrier_sha256"]
        == sha256(contents["prediction-barrier.json"]).hexdigest()
    )
    schema.require(
        ledger["source_results_sha256"]
        == sha256(contents["source-results.json"]).hexdigest()
    )
    schema.require(
        ledger["accepted_inputs_sha256"]
        == schema.load(contents["source-results.json"])["accepted_inputs_sha256"]
    )
    barrier = schema.load(contents["prediction-barrier.json"])
    parent_pid = None
    for position, entry in enumerate(entries):
        frame = _entry(entry)
        _frame(frame, execution, ledger, barrier)
        _predecessor(frame, position, entries, ledger)
        schema.require(entry["accepted"])
        schema.require(parent_pid is None or parent_pid == frame.parent_pid)
        parent_pid = frame.parent_pid
    _acceptances(ledger, entries, execution, contents)


def _acceptances(ledger, entries, execution, contents):
    from ._adopted_study_cell_evidence import validate

    metadata = schema.load(contents["source-results.json"])["accepted_inputs"]
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
    accounting = schema.load(contents["study-accounting.json"])
    scientific = schema.load(records.decoded(accounting["scientific_accounting_bytes"]))
    schema.require(len(ledger["cell_acceptances"]) == 125)
    for position, (value, projection) in enumerate(
        zip(ledger["cell_acceptances"], scientific["cells"], strict=True)
    ):
        validate(
            value,
            entries[2 + 2 * position : 4 + 2 * position],
            execution,
            ledger["accepted_inputs_sha256"],
            projection,
        )
