"""Independent invented-byte checks of fresh series scientific restoration."""

import json
from dataclasses import FrozenInstanceError

import pytest
from study_history_cell_fixtures import republish
from study_series_cell_fixtures import (
    candidates,
    fresh_cell,
    manifests,
    series_case,
    verify,
)
from test_study_series_cell import api

from automated_phishing_detection import _study_history_cell_records as records
from automated_phishing_detection import study_series_inputs
from automated_phishing_detection._checkpoint_codec import canonical_bytes
from automated_phishing_detection._shift_run_codec import dump

__all__ = ["candidates", "fresh_cell", "manifests", "series_case"]


def forbid(*args, **kwargs):
    pytest.fail("unauthenticated content reached a later restoration stage")


@pytest.mark.parametrize(
    "field",
    (
        "expected_descriptor_sha256",
        "expected_binding_sha256",
        "expected_cell_reservation_sha256",
        "expected_snapshot_sha256",
    ),
)
def test_authentication_precedes_input_restoration(fresh_cell, monkeypatch, field):
    monkeypatch.setattr(study_series_inputs, "restore_series_cell_inputs", forbid)
    replacement = "0" * 64
    if field == "expected_snapshot_sha256":
        replacement = dict(fresh_cell.arguments[field])
        replacement["attempt/run.json"] = "0" * 64
    with pytest.raises(api().SeriesCellScienceError):
        verify(fresh_cell, **{field: replacement})


@pytest.mark.parametrize("field", ("metadata_bytes", "profile_bytes", "manifest_bytes"))
def test_context_authentication_precedes_publication(fresh_cell, monkeypatch, field):
    monkeypatch.setattr(records, "publication", forbid)
    with pytest.raises(
        api().SeriesCellScienceError, match="^invalid_series_cell_science$"
    ):
        verify(fresh_cell, **{field: b"invented private malformed context"})


def test_rehashed_run_counter_cannot_replace_full_scientific_validation(fresh_cell):
    document = json.loads(fresh_cell.values["attempt/run.json"])
    run = document.get("run", document)
    run["after_measured"]["admitted_requests"] -= 1
    encoder = (
        dump
        if fresh_cell.inputs.computational.cell.workload == "shift_period"
        else canonical_bytes
    )
    payloads, pins = republish(fresh_cell, {"run.json": encoder(document)})
    with pytest.raises(api().SeriesCellScienceError):
        verify(fresh_cell, payloads=payloads, expected_snapshot_sha256=pins)


def test_rehashed_checkpoint_must_match_complete_run(fresh_cell):
    checkpoint = json.loads(fresh_cell.values["attempt/measured.json"])
    checkpoint["after_warmup"]["admitted_requests"] -= 1
    payloads, pins = republish(fresh_cell, {"measured.json": dump(checkpoint)})
    with pytest.raises(api().SeriesCellScienceError):
        verify(fresh_cell, payloads=payloads, expected_snapshot_sha256=pins)


def test_verified_result_returns_fresh_non_authorizing_views(fresh_cell):
    result = verify(fresh_cell)
    result.run.after_measured.admitted_requests = 0
    result.summary.clear()
    assert result.run == fresh_cell.run
    assert result.summary == fresh_cell.summary
    assert result.authorizes_execution is False
    assert result.inputs.authorizes_execution is False
    with pytest.raises(FrozenInstanceError):
        result.authorizes_execution = True
    with pytest.raises(FrozenInstanceError):
        result.inputs.origin_metadata_bytes = b"{}"


def test_rehashed_shift_trace_remains_bound_to_original_source(fresh_cell):
    if fresh_cell.inputs.computational.cell.workload != "shift_period":
        pytest.skip("offline trace exists only in the shift publication")
    document = json.loads(fresh_cell.values["attempt/run.json"])
    document["trace"]["rows"][0]["stage1_probability"] = 0.21
    payloads, pins = republish(fresh_cell, {"run.json": dump(document)})
    with pytest.raises(api().SeriesCellScienceError):
        verify(fresh_cell, payloads=payloads, expected_snapshot_sha256=pins)
