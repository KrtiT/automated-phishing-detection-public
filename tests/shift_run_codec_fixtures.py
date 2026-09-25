"""Complete invented shift runs and their original checkpoint-stage projections."""

from automated_phishing_detection.http_replay import HttpOutcome, ReplayRequest
from automated_phishing_detection.http_schema import DrainResponse, ScanResponse
from automated_phishing_detection.shift_replay import ShiftRun, _ShiftProgress
from automated_phishing_detection.shift_schema import (
    ShiftPlan,
    ShiftStateResponse,
    ShiftTraceRow,
    ShiftWindow,
)


def counts(total):
    return DrainResponse(
        admitted_requests=total,
        completed_requests=total,
        failed_requests=0,
        transformer_forward_attempts=0,
        successful_transformer_scores=0,
    )


def state(plan, phase, total, *, rows=(), complete=False):
    return ShiftStateResponse(
        manifest_sha256=plan.manifest_sha256,
        run_index=plan.run_index,
        measured_count=len(plan.requests),
        warmup_count=plan.warmup_count,
        phase=phase,
        broken=False,
        complete=complete,
        counts=counts(total),
        rows=list(rows),
    )


def outcome(plan, phase, position, error=None):
    identity = plan.request_id(phase, position)
    offset = 0 if phase == "warmup" else plan.warmup_count
    response = ScanResponse(
        request_id=identity,
        admission_sequence=offset + position + 1,
        action="allow",
        probability=0.2,
        stage2_invoked=False,
    )
    return HttpOutcome(
        plan.requests[position].record_id,
        identity,
        2000.0 if error else 1.0,
        None if error else 200,
        error,
        None if error else response,
    )


def trace_row(plan, position):
    ordinal = position + 1
    window = None
    if ordinal >= 256 and (ordinal - 256) % 64 == 0:
        window = ShiftWindow(
            start_position=ordinal - 255, end_position=ordinal, score=1.0, alert=False
        )
    return ShiftTraceRow(
        request_id=plan.request_id("measured", position),
        admission_sequence=plan.warmup_count + ordinal,
        position=ordinal,
        stage1_probability=0.2,
        transformer_probability=None,
        fixed_decision=0,
        decision=0,
        band_selected=False,
        drift_override=False,
        stage2_invoked=False,
        monitor_nll=1.0,
        window=window,
    )


def drains(plan, phase, outcomes):
    offset = 0 if phase == "warmup" else plan.warmup_count
    return [
        {
            "phase": phase,
            "position": position,
            "counts": counts(offset + position + 1),
            "elapsed_ms": 0.25,
        }
        for position, row in enumerate(outcomes)
        if row.error is not None
    ]


def shift_case(*, error=None, measured_count=1000, run_index=1):
    plan = ShiftPlan(
        "a" * 64,
        run_index,
        tuple(
            ReplayRequest(f"invented-{position}", f"https://invented.test/{position}")
            for position in range(measured_count)
        ),
    )
    warmup = tuple(
        outcome(plan, "warmup", position, error if position == 1 else None)
        for position in range(1000)
    )
    measured = tuple(
        outcome(plan, "measured", position, error if position in (0, 999) else None)
        for position in range(measured_count)
    )
    return _complete_case(plan, warmup, measured)


def _complete_case(plan, warmup, measured):
    measured_count = len(measured)
    inline = sum(member["elapsed_ms"] for member in drains(plan, "measured", measured))
    run = ShiftRun(
        plan,
        warmup,
        measured,
        counts(0),
        counts(1000),
        counts(1000 + measured_count),
        state(
            plan,
            "measured",
            1000 + measured_count,
            rows=(trace_row(plan, position) for position in range(measured_count)),
            complete=True,
        ),
        sum(row.elapsed_ms for row in measured) + inline + 1.0,
        0.5,
        inline,
    )
    return run, *checkpoints(run)


def checkpoints(run):
    plan = run.plan
    progress = _ShiftProgress(
        plan.manifest_sha256,
        plan.run_index,
        plan.warmup_count,
        len(plan.requests),
        [True] * plan.warmup_count,
        [False] * len(plan.requests),
        warmup=list(run.warmup),
        phase="warmup",
        stage="warmup_checkpoint",
        initial_state=state(plan, "warmup", 0),
        initial=run.initial,
        occurrence_drains=drains(plan, "warmup", run.warmup),
    )
    warmup = progress.snapshot()
    progress.phase, progress.stage = "measured", "measured_checkpoint"
    progress.measured_started = [True] * len(plan.requests)
    progress.measured = list(run.measured)
    progress.after_warmup = run.after_warmup
    progress.reset_state = state(plan, "measured", plan.warmup_count)
    progress.measured_elapsed_ms = run.measured_elapsed_ms
    progress.measured_timeout_drain_ms = run.measured_timeout_drain_ms
    progress.occurrence_drains.extend(drains(plan, "measured", run.measured))
    return warmup, progress.snapshot()
