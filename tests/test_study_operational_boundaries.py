"""Closed administrative joins reject substitutions without inventing evidence."""

import json
from dataclasses import replace

import pytest
from study_operational_fixtures import compact, digest
from test_study_operational_records import api

from automated_phishing_detection import execution_receipt
from automated_phishing_detection.operational_cell_runner import OperationalCellFailure
from automated_phishing_detection.operational_schedule import planned_cells


def changed_public(record, change):
    public = json.loads(record.public_bytes)
    change(public)
    content = execution_receipt._json_bytes(public, "fixture")
    hashes = dict(record.snapshot_sha256) | {"public-summary.json": digest(content)}
    return replace(
        record, public_bytes=content, snapshot_sha256=tuple(sorted(hashes.items()))
    )


@pytest.mark.parametrize("value", [True, 1.0, "1", None])
def test_cell_field_types_are_exact_before_equality(value):
    record = compact(api(), 1)
    record = replace(record, cell=replace(record.cell, ordinal=value))
    with pytest.raises(api().StudyOperationalRecordError):
        api().freeze_cell_accounting((record,))


@pytest.mark.parametrize(
    "field", ["revision", "runtime_sha256", "operational_profile_sha256"]
)
def test_rehashed_second_public_context_cannot_change_study(field):
    first, second = compact(api(), 1), compact(api(), 2)

    def change(public):
        public["execution"][field] = "b" * (40 if field == "revision" else 64)

    second = changed_public(second, change)
    with pytest.raises(api().StudyOperationalRecordError):
        api().freeze_cell_accounting((first, second))


@pytest.mark.parametrize(
    "field,value", [("schema_version", True), ("status", "accepted"), ("extra", 1)]
)
def test_public_closed_schema_is_checked_even_with_fresh_digest(field, value):
    record = changed_public(
        compact(api(), 1), lambda public: public.update({field: value})
    )
    with pytest.raises(api().StudyOperationalRecordError):
        api().freeze_cell_accounting((record,))


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "bad_digest"])
def test_snapshot_hash_inventory_stays_exact_and_immutable(mutation):
    record = compact(api(), 1)
    choices = {
        "duplicate": (*record.snapshot_sha256, record.snapshot_sha256[0]),
        "missing": record.snapshot_sha256[:-1],
        "bad_digest": (
            (record.snapshot_sha256[0][0], "A" * 64),
            *record.snapshot_sha256[1:],
        ),
    }
    with pytest.raises(api().StudyOperationalRecordError):
        api().freeze_cell_accounting(
            (replace(record, snapshot_sha256=choices[mutation]),)
        )


@pytest.mark.parametrize(
    "stage,progress", [("bad stage", None), ("", None), ("selection", bytearray(b"x"))]
)
def test_stopped_metadata_never_normalizes_invalid_values(stage, progress):
    stopped = api().StoppedOperationalCell(planned_cells()[0], stage, progress=progress)
    with pytest.raises(api().StudyOperationalRecordError):
        api().freeze_cell_accounting((), stopped=stopped)


@pytest.mark.parametrize(
    "changes",
    [{"cell": planned_cells()[1]}, {"stage": "completion"}, {"publishing": 1}],
)
def test_actual_failure_must_match_selected_cell_and_stage(changes):
    failure = OperationalCellFailure(
        planned_cells()[0], None, None, None, None, "selection", False
    )
    stopped = api().StoppedOperationalCell(
        planned_cells()[0], "selection", replace(failure, **changes)
    )
    with pytest.raises(api().StudyOperationalRecordError):
        api().freeze_cell_accounting((), stopped=stopped)


def test_projection_views_cannot_mutate_retained_record():
    record = compact(api(), 1)
    slots = api().freeze_cell_accounting((record,))
    projection = api().cell_accounting_projection(slots)
    projection[0]["cell"]["ordinal"] = 9
    projection[0]["snapshot_sha256"].clear()
    again = api().cell_accounting_projection(slots)
    assert again[0]["cell"]["ordinal"] == 1
    assert len(again[0]["snapshot_sha256"]) == 36
