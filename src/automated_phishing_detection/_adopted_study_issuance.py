"""Retain exact admission bytes before any owned launch can begin."""

import os

from . import _adopted_study_records as records
from . import _study_run_schema as schema
from ._process_support import command_hash
from ._study_admission import AdmissionFrame, ParentAdmission


def _frame(ledger, role, command, predecessor, accepted, binding):
    return AdmissionFrame(
        role=role,
        profile_sha256=ledger.authorization.profile_sha256,
        envelope_sha256=ledger.authorization.envelope_sha256,
        parent_pid=os.getpid(),
        command_sha256=command_hash(command),
        root_reservation_sha256=ledger.attempt.reservation_sha256,
        intent_sha256=ledger.intent_sha256,
        barrier_sha256=ledger.barrier_sha256,
        preparation_reservation_sha256=ledger.preparation.reservation_sha256,
        preparation_completion_sha256=ledger.preparation.completion_sha256,
        predecessor_sha256=predecessor,
        accepted_inputs_sha256=accepted,
        cell_binding_sha256=binding,
    )


def _entry(frame, cell):
    return {
        "role": frame.role,
        "command_sha256": frame.command_sha256,
        "cell_ordinal": None if cell is None else cell.ordinal,
        "frame_bytes": records.encoded(frame.canonical_bytes),
        "frame_sha256": frame.sha256,
        "issued": True,
        "launched_pid": None,
        "exit_observed": False,
        "exit_code": None,
        "accepted": False,
    }


def _launched(entry, pid):
    schema.require(entry["launched_pid"] is None and type(pid) is int and pid > 0)
    entry["launched_pid"] = pid


def _observed(entry, seen, code):
    schema.require(entry["launched_pid"] is not None and type(seen) is bool)
    schema.require(type(code) is int if seen else code is None)
    entry["exit_observed"], entry["exit_code"] = seen, code


def issue_admission(ledger, role, command, predecessor, accepted, binding):
    frame = _frame(ledger, role, command, predecessor, accepted, binding)
    entry = _entry(frame, ledger.current_cell)
    ledger.entries.append(entry)
    return ParentAdmission(
        frame,
        on_launched=lambda pid: _launched(entry, pid),
        on_observed=lambda seen, code: _observed(entry, seen, code),
    )
