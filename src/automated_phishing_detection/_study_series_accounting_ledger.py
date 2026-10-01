"""Authenticate saved ordered admissions without constructing live observations."""

from ._checkpoint_codec import canonical_bytes
from ._process_support import command_hash
from ._study_series_accounting_records import decoded, digest, schema
from ._study_series_child_commands import series_child_command

ENTRY_FIELDS = frozenset(
    (
        "role",
        "cell_ordinal",
        "command_sha256",
        "frame_bytes",
        "frame_sha256",
        "issued",
        "launched_pid",
        "observation_recorded",
        "exit_observed",
        "exit_code",
        "accepted",
    )
)


def process_role(value, entry, parent):
    schema.keys(
        value,
        "pid exit_code exit_observed forced signals stdout_sha256 stderr_sha256".split(),
    )
    for name in ("exit_observed", "forced"):
        schema.require(type(value[name]) is bool)
    pid, code = value["pid"], value["exit_code"]
    schema.require(pid is None or (type(pid) is int and pid > 0 and pid != parent))
    schema.require(type(code) is int if value["exit_observed"] else code is None)
    schema.require(
        type(value["signals"]) is list
        and all(
            type(signal) is int and signal in (15, 9) for signal in value["signals"]
        )
    )
    for name in ("stdout_sha256", "stderr_sha256"):
        if value[name] is not None:
            schema.digest(value[name])
    if pid is not None:
        schema.require(entry is not None and pid == entry["launched_pid"])
    if value["exit_observed"]:
        schema.require(
            pid is not None and entry["exit_observed"] and entry["exit_code"] == code
        )


def commands(context, slot):
    binding = digest(decoded(slot["binding_bytes"]))
    return {
        role: command_hash(
            series_child_command(
                context["public"],
                role,
                cell_ordinal=slot["ordinal"],
                cell_binding_sha256=binding,
            )
        )
        for role in ("service", "client")
    }


def _frame(entry, ledger, context, slot):
    content = decoded(entry["frame_bytes"])
    schema.require(digest(content) == entry["frame_sha256"])
    expected = dict(
        schema_version="study-series-admission-v1",
        role=entry["role"],
        profile_sha256=context["public"].profile_sha256,
        envelope_sha256=context["public"].envelope_sha256,
        parent_pid=ledger["parent_pid"],
        command_sha256=entry["command_sha256"],
        series_reservation_sha256=context["pins"][0],
        segment_reservation_sha256=context["pins"][1],
        origin_reservation_sha256=context["profile"]["origin"][
            "root_reservation_sha256"
        ],
        history_index_sha256=context["profile"]["history"]["index_sha256"],
        intent_sha256=digest(context["intent"]),
        predecessor_sha256=digest(context["imported"]),
        accepted_inputs_sha256=digest(context["metadata"]),
        cell_binding_sha256=digest(decoded(slot["binding_bytes"])),
        segment_ordinal=2,
        cell_ordinal=slot["ordinal"],
    )
    schema.require(content == canonical_bytes(expected)[:-1])


def _facts(entry, parent):
    for name in ("observation_recorded", "exit_observed", "accepted"):
        schema.require(type(entry[name]) is bool)
    pid, code = entry["launched_pid"], entry["exit_code"]
    schema.require(pid is None or (type(pid) is int and pid > 0 and pid != parent))
    schema.require(not entry["observation_recorded"] or pid is not None)
    schema.require(not entry["exit_observed"] or entry["observation_recorded"])
    schema.require(type(code) is int if entry["exit_observed"] else code is None)
    if entry["accepted"]:
        schema.require(
            entry["observation_recorded"] and entry["exit_observed"] and code == 0
        )


def entries(values, ledger, context, slot):
    expected = commands(context, slot) if values else {}
    schema.require(len(values) <= 2)
    for offset, entry in enumerate(values):
        schema.keys(entry, ENTRY_FIELDS)
        schema.require(entry["issued"] is True)
        schema.require(entry["role"] == ("service", "client")[offset])
        schema.require(
            type(entry["cell_ordinal"]) is int
            and entry["cell_ordinal"] == slot["ordinal"]
        )
        schema.require(entry["command_sha256"] == expected[entry["role"]])
        schema.require(entry["accepted"] is (slot["status"] == "accepted"))
        _facts(entry, ledger["parent_pid"])
        _frame(entry, ledger, context, slot)
    if len(values) == 2:
        schema.require(values[0]["launched_pid"] is not None)
        schema.require(values[1]["launched_pid"] != values[0]["launched_pid"])
    if slot["status"] == "accepted":
        schema.require(len(values) == 2)


def _header(value, context):
    expected = dict(
        schema_version=1,
        protocol="study-series-ledger-v1",
        series_reservation_sha256=context["pins"][0],
        segment_reservation_sha256=context["pins"][1],
        intent_sha256=digest(context["intent"]),
        history_import_sha256=digest(context["imported"]),
        accepted_inputs_sha256=digest(context["metadata"]),
        segment_ordinal=2,
        start_ordinal=context["profile"]["segment"]["start_ordinal"],
        end_ordinal=125,
    )
    schema.keys(value, {*expected, "parent_pid", "admissions", "cells"})
    schema.same({name: value[name] for name in expected}, expected)
    schema.require(type(value["parent_pid"]) is int and value["parent_pid"] > 0)
    schema.require(type(value["admissions"]) is list and type(value["cells"]) is list)
    schema.require(len(value["cells"]) == 126 - expected["start_ordinal"])


def validate(value, context, *, complete):
    from ._study_series_accounting_cells import validate_slot

    _header(value, context)
    cursor, pending, stopped = 0, False, False
    for ordinal, slot in enumerate(value["cells"], value["start_ordinal"]):
        status = slot["status"]
        schema.require(status != "accepted" or not pending)
        schema.require(status != "stopped" or not pending)
        pending = pending or status != "accepted"
        stopped = stopped or status == "stopped"
        count = (
            2
            if status == "accepted"
            else len(value["admissions"]) - cursor
            if status == "stopped"
            else 0
        )
        selected = value["admissions"][cursor : cursor + count]
        validate_slot(slot, ordinal, context, selected)
        entries(selected, value, context, slot)
        cursor += count
    schema.require(cursor == len(value["admissions"]))
    schema.require(not complete or (not pending and not stopped))
