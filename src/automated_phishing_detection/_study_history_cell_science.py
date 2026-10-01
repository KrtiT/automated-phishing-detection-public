"""Unchanged full run, checkpoint, offline trace and summary computations."""

from ._operational_cell_run_records import decode_run
from .http_replay import summarize_run
from .http_run_checkpoints import verify_http_checkpoints
from .policy_replay import MonitorScore, PairedProbabilities
from .shift_replay import summarize_shift_run, verify_offline_trace
from .shift_run_checkpoints import verify_shift_checkpoints


def _offline(run, inputs, snapshot):
    probabilities = tuple(
        PairedProbabilities(
            row.record.record_id,
            row.primary.stage1_probability,
            row.primary.transformer_probability,
        )
        for row in snapshot.rows
    )
    scores = tuple(
        MonitorScore(row.record.record_id, row.primary.negative_log_likelihood)
        for row in snapshot.rows
    )
    thresholds = inputs.primary["thresholds"]
    verify_offline_trace(
        run,
        probabilities,
        scores,
        stage1_threshold=thresholds["logistic_l1"],
        transformer_threshold=thresholds["transformer"],
        half_width=thresholds["half_width"],
        monitor_boundary=thresholds["monitor_boundary"],
    )


def summary(payloads, inputs, external):
    run = decode_run(payloads["run.json"], inputs)
    if inputs.cell.workload == "shift_period":
        verify_shift_checkpoints(
            payloads["warmup.json"], payloads["measured.json"], run=run
        )
        _offline(run, inputs, external)
        return summarize_shift_run(run)
    verify_http_checkpoints(payloads["warmup.json"], payloads["measured.json"], run=run)
    return summarize_run(run)
