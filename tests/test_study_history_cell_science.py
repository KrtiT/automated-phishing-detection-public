"""Rehashed scientific lies still fail unchanged full-run and trace kernels."""

import json
from dataclasses import replace

import pytest
from shift_run_codec_fixtures import _complete_case, outcome
from study_history_cell_fixtures import (
    api,
    candidates,
    case,
    digest,
    hashes,
    history,
    manifests,
    republish,
    restore,
)

from automated_phishing_detection import execution_receipt as receipt
from automated_phishing_detection._checkpoint_codec import canonical_bytes
from automated_phishing_detection._shift_run_codec import dump
from automated_phishing_detection.http_replay import ERRORS
from automated_phishing_detection.shift_replay import summarize_shift_run
from automated_phishing_detection.shift_run_codec import encode_shift_run

__all__ = ["candidates", "case", "history", "manifests"]


def _run_wire(history, value):
    return (
        dump(value)
        if history.working.inputs.cell.workload == "shift_period"
        else canonical_bytes(value)
    )


@pytest.mark.parametrize("name", ("warmup.json", "measured.json"))
@pytest.mark.parametrize("change", ("run_index", "rows", "later_counts"))
def test_rehashed_original_checkpoints_must_match(history, name, change):
    value = json.loads(history.values[f"attempt/{name}"])
    if change == "run_index":
        value["run_index"] += 1
    elif change == "rows":
        value["warmup"].pop()
    else:
        value["after_measured"] = {"admitted_requests": 11000}
    payloads, pins = republish(history, {name: dump(value)})
    with pytest.raises(api().HistoricalCellScienceError):
        restore(history, payloads=payloads, expected_snapshot_sha256=pins)


@pytest.mark.parametrize(
    "change", ("missing", "order", "identity", "counter", "duration", "extra")
)
def test_rehashed_run_cannot_omit_or_reclassify_evidence(history, change):
    value = json.loads(history.values["attempt/run.json"])
    run = value.get("run", value)
    if change == "missing":
        run["measured"].pop()
    elif change == "order":
        run["measured"].reverse()
    elif change == "identity":
        run["measured"][0]["record_id"] = "other"
    elif change == "counter":
        run["after_measured"]["admitted_requests"] -= 1
    elif change == "duration":
        run["measured_elapsed_ms"] = -1
    else:
        run["ignored"] = True
    payloads, pins = republish(history, {"run.json": _run_wire(history, value)})
    with pytest.raises(api().HistoricalCellScienceError):
        restore(history, payloads=payloads, expected_snapshot_sha256=pins)


def _replace_summary(payloads, summary):
    values = dict(payloads)
    public = json.loads(values["public-summary.json"])
    public["summary"] = summary
    values["public-summary.json"] = receipt._json_bytes(public, "fixture")
    completion = json.loads(values["attempt/outcome.json"])
    completion["public_summary_sha256"] = digest(values["public-summary.json"])
    values["attempt/outcome.json"] = receipt._json_bytes(completion, "fixture")
    return tuple(values.items()), hashes(values)


@pytest.mark.parametrize(
    "field", ("p99_ms", "request_errors", "request_count", "extra")
)
def test_full_published_summary_not_a_selected_projection_is_verified(history, field):
    summary = dict(history.working.summary)
    summary[field] = summary.get(field, 0) + 1
    payloads, pins = _replace_summary(tuple(history.values.items()), summary)
    with pytest.raises(api().HistoricalCellScienceError):
        restore(history, payloads=payloads, expected_snapshot_sha256=pins)


@pytest.mark.parametrize(
    "field,value",
    (
        ("stage1_probability", 0.21),
        ("negative_log_likelihood", -99.0),
    ),
)
def test_shift_uses_exact_original_offline_primary_stream(history, field, value):
    if history.working.inputs.cell.workload != "shift_period":
        return
    snapshot = history.arguments["external_snapshot"]
    row = replace(
        snapshot.rows[0], primary=replace(snapshot.rows[0].primary, **{field: value})
    )
    changed = replace(snapshot, rows=(row, *snapshot.rows[1:]))
    with pytest.raises(api().HistoricalCellScienceError):
        restore(history, external_snapshot=changed)


@pytest.mark.parametrize(
    "field,value", (("monitor_nll", -99.0), ("stage1_probability", 0.21))
)
def test_rehashed_shift_trace_must_match_original_probabilities(history, field, value):
    if history.working.inputs.cell.workload != "shift_period":
        return
    document = json.loads(history.values["attempt/run.json"])
    document["trace"]["rows"][0][field] = value
    payloads, pins = republish(history, {"run.json": dump(document)})
    with pytest.raises(api().HistoricalCellScienceError):
        restore(history, payloads=payloads, expected_snapshot_sha256=pins)


def test_shift_preserves_every_terminal_error_and_occurrence_drain(history):
    if history.working.inputs.cell.workload != "shift_period":
        return
    original = history.working.run
    errors = sorted(ERRORS)
    phases = {}
    for phase in ("warmup", "measured"):
        rows = list(getattr(original, phase))
        rows[: len(errors)] = [
            outcome(original.plan, phase, position, error)
            for position, error in enumerate(errors)
        ]
        phases[phase] = tuple(rows)
    run, warmup, measured = _complete_case(original.plan, **phases)
    run = replace(run, trace=original.trace)
    payloads, _ = republish(
        history,
        {
            "run.json": encode_shift_run(run),
            "warmup.json": warmup,
            "measured.json": measured,
        },
    )
    payloads, pins = _replace_summary(payloads, summarize_shift_run(run))
    result = restore(history, payloads=payloads, expected_snapshot_sha256=pins)
    assert result.summary["request_errors"] == len(errors)
    assert result.summary["measured_timeout_drain_ms"] == 0.25 * len(errors)
    assert result.run == run
