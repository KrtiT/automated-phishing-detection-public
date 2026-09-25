"""Existing complete-run kernels, with no new scientific or measurement policy."""

from .http_replay import summarize_run
from .http_run_checkpoints import verify_http_checkpoints
from .http_run_codec import decode_http_run
from .policy_replay import MonitorScore, PairedProbabilities
from .shift_replay import summarize_shift_run, verify_offline_trace
from .shift_run_checkpoints import verify_shift_checkpoints
from .shift_run_codec import decode_shift_run
from .shift_schema import ShiftPlan


def decode_run(content, inputs):
    cell = inputs.cell
    if cell.workload == "shift_period":
        plan = ShiftPlan(inputs.manifest_sha256, cell.run_index, inputs.requests)
        return decode_shift_run(content, expected_plan=plan)
    return decode_http_run(
        content,
        expected_manifest_sha256=inputs.manifest_sha256,
        expected_requests=inputs.requests,
        expected_prevalence_basis_points=cell.prevalence_basis_points,
        expected_concurrency=cell.concurrency,
        expected_run_index=cell.run_index,
        expected_workload=cell.workload,
    )


def _offline(run, inputs, accepted):
    rows = accepted.external.snapshot.rows
    probabilities = tuple(
        PairedProbabilities(
            row.record.record_id,
            row.primary.stage1_probability,
            row.primary.transformer_probability,
        )
        for row in rows
    )
    scores = tuple(
        MonitorScore(row.record.record_id, row.primary.negative_log_likelihood)
        for row in rows
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


def verify_run(payloads, inputs, accepted):
    run = decode_run(payloads["run.json"], inputs)
    if inputs.cell.workload == "shift_period":
        verify_shift_checkpoints(
            payloads["warmup.json"], payloads["measured.json"], run=run
        )
        _offline(run, inputs, accepted)
        return summarize_shift_run(run)
    verify_http_checkpoints(payloads["warmup.json"], payloads["measured.json"], run=run)
    return summarize_run(run)
