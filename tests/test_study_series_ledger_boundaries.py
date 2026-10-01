"""Context, first-cell and exact-command boundaries reject before admission."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
from study_series_ledger_fixtures import (
    api,
    candidates,
    commands,
    ledger,
    ledger_cell,
    manifests,
    prefix_case,
    series_case,
    snapshot,
    start,
)
from test_study_series_ledger_ordering import rejection

from automated_phishing_detection._checkpoint_codec import canonical_bytes
from automated_phishing_detection.operational_schedule import cell_for_ordinal

__all__ = ["candidates", "manifests", "series_case", "prefix_case", "ledger_cell"]


@pytest.mark.parametrize(
    "field", ["public_binding", "series_attempt", "segment_attempt"]
)
@pytest.mark.parametrize("value", [None, {}, object()])
def test_exact_initial_binding_and_attempt_types(prefix_case, field, value):
    with rejection():
        ledger(prefix_case, **{field: value})


@pytest.mark.parametrize("field", ["intent_bytes", "import_bytes", "metadata_bytes"])
@pytest.mark.parametrize(
    "value", [None, b"{}\n", bytearray(b"{}\n"), b"PRIVATE_NOT_JSON"]
)
def test_initial_buffers_are_exact_immutable_canonical_context(
    prefix_case, field, value
):
    with rejection():
        ledger(prefix_case, **{field: value})


@pytest.mark.parametrize("field", ["series_attempt", "segment_attempt"])
@pytest.mark.parametrize(
    "part,value",
    [("directory", Path("/wrong/attempt")), ("reservation_sha256", "f" * 64)],
)
def test_actual_attempt_identity_and_exact_directory(prefix_case, field, part, value):
    original = prefix_case.series if field == "series_attempt" else prefix_case.segment
    with rejection():
        ledger(prefix_case, **{field: replace(original, **{part: value})})


@pytest.mark.parametrize("field", ["intent_bytes", "import_bytes", "metadata_bytes"])
@pytest.mark.parametrize("mutation", ["extra", "wrong_series", "wrong_segment"])
def test_reencoded_prefix_cannot_replace_closed_original_joins(
    prefix_case, field, mutation
):
    content = {
        "intent_bytes": prefix_case.intent,
        "import_bytes": prefix_case.imported,
        "metadata_bytes": prefix_case.metadata,
    }[field]
    value = json.loads(content)
    key = {
        "extra": "unreviewed",
        "wrong_series": "series_reservation_sha256",
        "wrong_segment": "segment_reservation_sha256",
    }[mutation]
    value[key] = "f" * 64
    with rejection():
        ledger(prefix_case, **{field: canonical_bytes(value)})


@pytest.mark.parametrize("field", ["policy_bytes", "profile_bytes", "envelope_bytes"])
def test_complete_public_header_is_revalidated(prefix_case, field):
    with rejection():
        ledger(
            prefix_case, public_binding=replace(prefix_case.binding, **{field: b"{}\n"})
        )


@pytest.mark.parametrize(
    "change",
    [
        "earlier",
        "later",
        "attempt",
        "directory",
        "reservation",
        "descriptor",
        "binding",
        "mutable_descriptor",
        "mutable_binding",
    ],
)
def test_start_requires_exact_next_frozen_cell_and_acyclic_actual_binding(
    ledger_cell, change
):
    selected = ledger_cell
    arguments = [
        selected.cell,
        selected.attempt,
        selected.selected.descriptor_bytes,
        selected.binding,
    ]
    replacements = {
        "earlier": (0, cell_for_ordinal(selected.cell.ordinal - 1)),
        "later": (0, cell_for_ordinal(selected.cell.ordinal + 1)),
        "attempt": (1, object()),
        "directory": (1, replace(selected.attempt, directory=Path("/wrong/attempt"))),
        "reservation": (1, replace(selected.attempt, reservation_sha256="e" * 64)),
        "descriptor": (2, b"{}\n"),
        "binding": (3, b"{}\n"),
        "mutable_descriptor": (2, bytearray(arguments[2])),
        "mutable_binding": (3, bytearray(arguments[3])),
    }
    index, value = replacements[change]
    arguments[index] = value
    current = ledger(selected.prefix)
    with rejection():
        current.start_cell(*arguments)
    with rejection():
        start(current, selected)


@pytest.mark.parametrize(
    "kind",
    [
        "role",
        "wrong_role_command",
        "extra",
        "list",
        "interpreter",
        "binding",
        "ordinal",
    ],
)
def test_exact_fixed_command_is_checked_before_appending_entry(ledger_cell, kind):
    current = ledger(ledger_cell.prefix)
    start(current, ledger_cell)
    role, command = "service", commands(ledger_cell)[0]
    if kind == "role":
        role = "internal"
    elif kind == "wrong_role_command":
        command = commands(ledger_cell)[1]
    elif kind == "extra":
        command = (*command, "--override")
    elif kind == "list":
        command = list(command)
    else:
        position = {"interpreter": 0, "binding": -1, "ordinal": -3}[kind]
        command = tuple(
            "changed" if index == position % len(command) else value
            for index, value in enumerate(command)
        )
    with rejection():
        current.issue(role, command)
    current.stop_cell(api().SeriesCellStop("observation"))
    assert snapshot(current)["admissions"] == []
