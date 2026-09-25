"""Administrative accounting cannot relabel partial work as a published study."""

import json

import pytest
from study_run_record_fixtures import api, prepared

from automated_phishing_detection import study_operational_records as cells
from automated_phishing_detection._checkpoint_codec import canonical_bytes

__all__ = ["prepared"]


def accounting(prepared, **changes):
    options = dict(
        execution=prepared.execution,
        stage="prediction_barrier",
        status="whole_study_hold",
        internal_status="unattempted",
        external_status="unattempted",
    )
    return api().study_accounting(
        cells.freeze_cell_accounting(()), **(options | changes)
    )


def test_accounting_preserves_exact_fixed_unattempted_projection(prepared):
    content = accounting(prepared)
    value = json.loads(content)
    assert value == {
        "schema_version": 1,
        "protocol": "study-root-v1",
        "status": "whole_study_hold",
        "stage": "prediction_barrier",
        "execution": prepared.execution,
        "internal_status": "unattempted",
        "external_status": "unattempted",
        "cells": cells.cell_accounting_projection(cells.freeze_cell_accounting(())),
    }
    assert len(value["cells"]) == 125
    assert content == canonical_bytes(value)


@pytest.mark.parametrize(
    "member,bad",
    [
        ("status", "study_evidence_published"),
        ("status", "unknown"),
        ("stage", "secret path"),
        ("internal_status", "complete"),
        ("external_status", True),
        ("status", False),
    ],
)
def test_accounting_closed_symbols(prepared, member, bad):
    with pytest.raises(ValueError):
        accounting(prepared, **{member: bad})


@pytest.mark.parametrize(
    "change",
    [
        {"internal_status": "accepted"},
        {"external_status": "stopped"},
        {
            "status": "matrix_accepted",
            "internal_status": "accepted",
            "external_status": "accepted",
        },
        {
            "status": "failed",
            "internal_status": "unattempted",
            "external_status": "accepted",
        },
    ],
)
def test_accounting_cannot_invent_attempted_or_completed_work(prepared, change):
    with pytest.raises(ValueError):
        accounting(prepared, **change)


def test_failed_source_pair_keeps_actual_internal_accepted_status(prepared):
    content = accounting(
        prepared,
        status="failed",
        stage="sources",
        internal_status="accepted",
        external_status="stopped",
    )
    value = json.loads(content)
    assert value["internal_status"] == "accepted"
    assert value["external_status"] == "stopped"
    assert all(slot["status"] == "unattempted" for slot in value["cells"])


def test_accounting_calls_existing_cell_projection_once(prepared, monkeypatch):
    original, calls = cells.cell_accounting_projection, []

    def observe(slots):
        calls.append(slots)
        return original(slots)

    monkeypatch.setattr(cells, "cell_accounting_projection", observe)
    accounting(prepared)
    assert len(calls) == 1
