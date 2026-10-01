"""Supplied inventories and every independently pinned byte remain exact."""

from dataclasses import replace

import pytest
from operational_input_fixtures import candidates, manifests
from stopped_study_cell_fixtures import (
    cell_history,
    change_record,
    invented_cell_history,
    refresh_cell,
    verify_cell,
)
from study_run_record_fixtures import prepared

__all__ = [
    "candidates",
    "manifests",
    "prepared",
    "cell_history",
    "invented_cell_history",
]


@pytest.mark.parametrize(
    "member",
    [
        "client-intent.json",
        "client-started.json",
        "client-process.json",
        "client-role.json",
        "warmup.json",
        "measured.json",
        "run.json",
        "evidence/process-pair.json",
        "public-summary.json",
    ],
)
def test_extra_attempt_records_reject_service_only_history(cell_history, member):
    cell_history.values[member] = b"{}\n"
    refresh_cell(cell_history)
    with pytest.raises(ValueError, match="^invalid_stopped_study_cell_history$"):
        verify_cell(cell_history)


@pytest.mark.parametrize("collection", ["attempt", "input"])
@pytest.mark.parametrize(
    "change", ["missing", "duplicate", "list", "member_list", "nonbytes", "extra"]
)
def test_supplied_payloads_have_exact_closed_shape(cell_history, collection, change):
    attribute = f"{collection}_payloads"
    payloads = getattr(cell_history, attribute)
    if change == "missing":
        payloads = payloads[:-1]
    elif change == "duplicate":
        payloads += (payloads[0],)
    elif change == "list":
        payloads = list(payloads)
    elif change == "member_list":
        payloads = (list(payloads[0]),) + payloads[1:]
    elif change == "nonbytes":
        payloads = ((payloads[0][0], bytearray(payloads[0][1])),) + payloads[1:]
    else:
        payloads += (("unrecognized", b"{}"),)
    setattr(cell_history, attribute, payloads)
    with pytest.raises(ValueError):
        verify_cell(cell_history)


@pytest.mark.parametrize("collection", ["attempt", "input"])
@pytest.mark.parametrize("change", ["missing", "extra", "wrong", "malformed", "notmap"])
def test_independent_inventories_are_exact(cell_history, collection, change):
    pins = getattr(cell_history, f"{collection}_pins")
    first = next(iter(pins))
    if change == "missing":
        pins.pop(first)
    elif change == "extra":
        pins["unexpected"] = "0" * 64
    elif change == "wrong":
        pins[first] = "0" * 64
    elif change == "malformed":
        pins[first] = b"0" * 64
    else:
        setattr(cell_history, f"{collection}_pins", tuple(pins.items()))
    with pytest.raises(ValueError):
        verify_cell(cell_history)


@pytest.mark.parametrize(
    "pin", ["profile", "envelope", "root_snapshot", "root_reservation"]
)
def test_public_entry_rechecks_original_authorization(cell_history, pin):
    overrides = {}
    if pin == "root_reservation":
        cell_history.root.snapshot = replace(
            cell_history.root.snapshot, reservation_sha256="0" * 64
        )
    elif pin == "root_snapshot":
        cell_history.root.pins["attempt/study-accounting.json"] = "0" * 64
    else:
        overrides[f"expected_{pin}_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        verify_cell(cell_history, **overrides)


def test_rehashed_nested_records_cannot_replace_independent_pins(cell_history):
    pins = cell_history.attempt_pins.copy()
    change_record(cell_history, "service-ready.json", port=12345)
    with pytest.raises(ValueError):
        verify_cell(cell_history, expected_attempt_snapshot_sha256=pins)


@pytest.mark.parametrize("name", ["reservation.json", "finalize.claim", "outcome.json"])
def test_receipts_reject_alternate_codec(cell_history, name):
    cell_history.values[name] += b"\n"
    refresh_cell(cell_history)
    with pytest.raises(ValueError):
        verify_cell(cell_history)
