"""Append exact current admission facts before opening any pipe or child launch."""

from ._process_support import command_hash
from ._study_series_admission import SeriesAdmissionFrame, SeriesParentAdmission
from ._study_series_ledger_context import digest, schema


def _frame(ledger, role, command):
    return SeriesAdmissionFrame(
        role=role,
        profile_sha256=ledger._public.profile_sha256,
        envelope_sha256=ledger._public.envelope_sha256,
        parent_pid=ledger._parent_pid,
        command_sha256=command_hash(command),
        series_reservation_sha256=ledger._series.reservation_sha256,
        segment_reservation_sha256=ledger._segment.reservation_sha256,
        origin_reservation_sha256=ledger._profile["origin"]["root_reservation_sha256"],
        history_index_sha256=ledger._profile["history"]["index_sha256"],
        intent_sha256=digest(ledger._intent),
        predecessor_sha256=digest(ledger._import),
        accepted_inputs_sha256=digest(ledger._metadata),
        cell_binding_sha256=digest(ledger._current["binding_bytes"]),
        segment_ordinal=2,
        cell_ordinal=ledger._current["cell"].ordinal,
    )


def _entry(frame):
    from ._study_series_ledger_records import encoded

    return dict(
        role=frame.role,
        cell_ordinal=frame.cell_ordinal,
        command_sha256=frame.command_sha256,
        frame_bytes=encoded(frame.canonical_bytes),
        frame_sha256=frame.sha256,
        issued=True,
        launched_pid=None,
        observation_recorded=False,
        exit_observed=False,
        exit_code=None,
        accepted=False,
    )


def _callback(ledger, entry, action, *arguments):
    from .study_series_ledger import _transition

    return _transition(ledger, action, entry, *arguments)


def _current_entry(ledger, entry):
    schema.require(ledger._current is not None and not entry["accepted"])
    schema.require(entry["cell_ordinal"] == ledger._current["cell"].ordinal)
    schema.require(any(member is entry for member in ledger._entries[-2:]))


def _launched(ledger, entry, pid):
    _current_entry(ledger, entry)
    schema.require(not ledger._closed and entry["launched_pid"] is None)
    schema.require(type(pid) is int and pid > 0 and pid != ledger._parent_pid)
    others = ledger._entries[2 * len(ledger._accepted) :]
    schema.require(all(member["launched_pid"] != pid for member in others))
    entry["launched_pid"] = pid


def _observed(ledger, entry, seen, code):
    _current_entry(ledger, entry)
    schema.require(
        entry["launched_pid"] is not None and not entry["observation_recorded"]
    )
    schema.require(type(seen) is bool and (type(code) is int if seen else code is None))
    entry.update(observation_recorded=True, exit_observed=seen, exit_code=code)
    if not seen or code != 0:
        ledger._closed = True


def _command(ledger, role, command):
    from ._study_series_child_commands import series_child_command

    schema.require(not ledger._closed and ledger._current is not None)
    schema.require(type(role) is str and role in ("service", "client"))
    expected = series_child_command(
        ledger._public,
        role,
        cell_ordinal=ledger._current["cell"].ordinal,
        cell_binding_sha256=digest(ledger._current["binding_bytes"]),
    )
    schema.require(type(command) is tuple and command == expected)
    schema.require(
        len(ledger._entries) == 2 * len(ledger._accepted) + (role == "client")
    )
    if role == "client":
        schema.require(ledger._entries[-1]["launched_pid"] is not None)


def issue(ledger, role, command):
    _command(ledger, role, command)
    frame = _frame(ledger, role, command)
    entry = _entry(frame)
    ledger._entries.append(entry)
    ledger._commands.append(command)
    return SeriesParentAdmission(
        frame,
        on_launched=lambda pid: _callback(ledger, entry, _launched, pid),
        on_observed=lambda seen, code: _callback(ledger, entry, _observed, seen, code),
    )
