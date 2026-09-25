"""Immutable accounting is administrative state, never constructed exit proof."""

import importlib
import importlib.util
from dataclasses import FrozenInstanceError, replace

import pytest
from study_operational_fixtures import compact

from automated_phishing_detection.operational_schedule import planned_cells


def api():
    name = "automated_phishing_detection.study_operational_records"
    assert importlib.util.find_spec(name), "missing study operational records"
    return importlib.import_module(name)


def test_empty_accounting_contains_every_original_cell_without_invented_attempts():
    slots = api().freeze_cell_accounting(())
    assert type(slots) is tuple and len(slots) == 125
    assert tuple(slot.cell for slot in slots) == planned_cells()
    assert all(slot.status == "unattempted" for slot in slots)
    rows = api().cell_accounting_projection(slots)
    assert len(rows) == 125
    for row in rows:
        assert set(row) == {
            "cell",
            "status",
            "retention",
            "snapshot_sha256",
            "observation_sha256",
            "stage",
            "reservation_sha256",
            "progress_sha256",
            "publishing",
        }
        assert all(
            value is None
            for name, value in row.items()
            if name not in ("cell", "status")
        )


@pytest.mark.parametrize("count", [1, 20, 120, 125])
def test_compact_prefix_retains_actual_references_and_null_suffix(count):
    records = tuple(compact(api(), ordinal) for ordinal in range(1, count + 1))
    slots = api().freeze_cell_accounting(records)
    assert all(slot.status == "accepted" for slot in slots[:count])
    assert all(slot.status == "unattempted" for slot in slots[count:])
    assert all(slot.accepted is record for slot, record in zip(slots, records))
    rows = api().cell_accounting_projection(slots)
    assert all(
        row["retention"] == "compact" and len(row["snapshot_sha256"]) == 36
        for row in rows[:count]
    )
    with pytest.raises(FrozenInstanceError):
        slots[0].accepted = None


def test_stopped_selected_cell_is_not_an_invented_attempt():
    original = b"exact original partial bytes"
    stop = api().StoppedOperationalCell(
        planned_cells()[1], "selection", progress=original
    )
    slots = api().freeze_cell_accounting((compact(api(), 1),), stopped=stop)
    assert slots[1].stopped is stop and slots[1].status == "stopped"
    row = api().cell_accounting_projection(slots)[1]
    assert row["stage"] == "selection" and row["progress_sha256"] is not None
    assert (
        row["observation_sha256"]
        is row["reservation_sha256"]
        is row["publishing"]
        is None
    )


@pytest.mark.parametrize(
    "mutation", ["list", "skip", "duplicate", "wrong_stop", "too_many"]
)
def test_no_sorting_replacement_or_retries(mutation):
    first, second = compact(api(), 1), compact(api(), 2)
    choices = {
        "list": [first],
        "skip": (second,),
        "duplicate": (first, first),
        "wrong_stop": (first,),
        "too_many": (first,) * 126,
    }
    stopped = (
        api().StoppedOperationalCell(planned_cells()[2], "completion")
        if mutation == "wrong_stop"
        else None
    )
    with pytest.raises(api().StudyOperationalRecordError):
        api().freeze_cell_accounting(choices[mutation], stopped=stopped)


@pytest.mark.parametrize(
    "field,value",
    [
        ("run_bytes", bytearray(b"{}\n")),
        ("snapshot_sha256", ()),
        ("descriptor_bytes", b"{}\n"),
        ("observation", object()),
    ],
)
def test_invalid_compact_record_is_integrity_failure(field, value):
    with pytest.raises(api().StudyOperationalRecordError):
        api().freeze_cell_accounting((replace(compact(api(), 1), **{field: value}),))


def test_projection_revalidates_fixed_slots():
    slots = api().freeze_cell_accounting(())
    for invalid in (list(slots), slots[:-1], (slots[1], slots[0], *slots[2:])):
        with pytest.raises(api().StudyOperationalRecordError):
            api().cell_accounting_projection(invalid)
