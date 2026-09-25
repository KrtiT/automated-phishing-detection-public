"""Serialized live-monitor HTTP characterization, separate from primary H3."""

import asyncio
import json
import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

import httpx
import numpy as np

from . import gmm_monitor, policy_replay
from . import http_replay as http
from .http_schema import DrainResponse, ScanResponse
from .shift_schema import ShiftPlan, ShiftStateResponse


class ShiftReplayError(http.ReplayError):
    def __init__(
        self,
        message,
        *,
        phase="preflight",
        outcomes=(),
        warmup_outcomes=(),
        progress=None,
    ):
        super().__init__(message, progress=progress)
        self.phase = phase
        self.outcomes = tuple(outcomes)
        self.warmup_outcomes = tuple(warmup_outcomes)


@dataclass(frozen=True)
class ShiftRun:
    plan: ShiftPlan
    warmup: tuple[http.HttpOutcome, ...]
    measured: tuple[http.HttpOutcome, ...]
    initial: DrainResponse
    after_warmup: DrainResponse
    after_measured: DrainResponse
    trace: ShiftStateResponse
    measured_elapsed_ms: float
    measured_drain_ms: float
    measured_timeout_drain_ms: float

    @property
    def manifest_sha256(self):
        return self.plan.manifest_sha256

    @property
    def run_index(self):
        return self.plan.run_index

    @property
    def concurrency(self):
        return 1

    @property
    def workload(self):
        return "shift_period"


@dataclass
class _ShiftProgress:
    manifest_sha256: str
    run_index: int
    warmup_count: int
    measured_count: int
    warmup_started: list[bool]
    measured_started: list[bool]
    warmup: list[http.HttpOutcome] = field(default_factory=list)
    measured: list[http.HttpOutcome] = field(default_factory=list)
    phase: str = "preflight"
    stage: str = "client_start"
    initial_state: ShiftStateResponse | None = None
    reset_state: ShiftStateResponse | None = None
    trace: ShiftStateResponse | None = None
    initial: DrainResponse | None = None
    after_warmup: DrainResponse | None = None
    after_measured: DrainResponse | None = None
    occurrence_drains: list[dict] = field(default_factory=list)
    measured_elapsed_ms: float | None = None
    measured_drain_ms: float | None = None
    measured_timeout_drain_ms: float | None = None

    def snapshot(self) -> bytes:
        def wire_model(value):
            if type(value) in (ScanResponse, DrainResponse, ShiftStateResponse):
                return value.model_dump()
            raise TypeError("invalid shift progress value")

        payload = {
            "schema_version": 1,
            "workload": "shift_period",
            "concurrency": 1,
            **asdict(self),
        }
        for phase in ("warmup", "measured"):
            payload[phase].extend(
                [None] * (getattr(self, f"{phase}_count") - len(payload[phase]))
            )
        return (
            json.dumps(
                payload,
                allow_nan=False,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
                default=wire_model,
            )
            + "\n"
        ).encode("ascii")

    def checkpoint(self, phase, retain):
        self.stage = f"{phase}_checkpoint"
        if retain is not None:
            retain(f"{phase}.json", self.snapshot())


async def _control(client, path, payload=None):
    async def exchange():
        response = await client.post(path, json=payload)
        if response.status_code != 200:
            raise ShiftReplayError("shift control rejected; run is incomplete")
        return ShiftStateResponse.model_validate(http._json(response.content))

    try:
        return await asyncio.wait_for(exchange(), timeout=http.DRAIN_DEADLINE_SECONDS)
    except (
        ValueError,
        UnicodeError,
        RecursionError,
        httpx.RequestError,
        asyncio.TimeoutError,
    ) as exc:
        raise ShiftReplayError("shift control failed; run is incomplete") from exc


def _identity(state, plan):
    if (
        state.manifest_sha256 != plan.manifest_sha256
        or state.run_index != plan.run_index
        or state.warmup_count != plan.warmup_count
        or state.measured_count != len(plan.requests)
    ):
        raise ShiftReplayError("service/manifest identity differs")


def _complete_counts(counts, expected):
    return (
        counts.admitted_requests == expected
        and counts.completed_requests == expected
        and counts.failed_requests == 0
        and 0
        <= counts.successful_transformer_scores
        <= counts.transformer_forward_attempts
        <= expected
    )


async def _phase(client, plan, phase, outcomes, progress):
    rows = plan.requests[: plan.warmup_count] if phase == "warmup" else plan.requests
    offset = 0 if phase == "warmup" else plan.warmup_count
    drain_ms = 0.0
    for position, row in enumerate(rows):
        identity = plan.request_id(phase, position)
        progress.stage = phase
        getattr(progress, f"{phase}_started")[position] = True
        outcome = await http._scan(client, row, identity)
        outcomes.append(outcome)
        expected_sequence = offset + position + 1
        if outcome.error is not None:
            progress.stage = f"{phase}_occurrence_drain"
            started = time.perf_counter_ns()
            try:
                drained = await http._drain(client, (identity,))
            except http.ReplayError as exc:
                raise ShiftReplayError(
                    "occurrence drain failed", phase=phase, outcomes=outcomes
                ) from exc
            elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
            drain_ms += elapsed_ms
            progress.occurrence_drains.append(
                {
                    "phase": phase,
                    "position": position,
                    "counts": drained,
                    "elapsed_ms": elapsed_ms,
                }
            )
            if not _complete_counts(drained, expected_sequence):
                raise ShiftReplayError(
                    "missing or failed row; stream is incomplete",
                    phase=phase,
                    outcomes=outcomes,
                )
        elif outcome.response.admission_sequence != expected_sequence:
            raise ShiftReplayError(
                "admission order differs from manifest", phase=phase, outcomes=outcomes
            )
    return tuple(outcomes), drain_ms


async def _replay_phases(client, plan, progress, retain):
    progress.stage = "initial_control"
    progress.initial_state = await _control(client, "/v1/shift/state")
    state = progress.initial_state
    progress.initial = state.counts
    progress.stage = "initial_validation"
    _identity(state, plan)
    if (
        state.phase != "warmup"
        or state.broken
        or state.complete
        or state.rows
        or any(state.counts.model_dump().values())
    ):
        raise ShiftReplayError("shift run requires a fresh service")
    progress.phase = "warmup"
    await _phase(client, plan, "warmup", progress.warmup, progress)
    progress.checkpoint("warmup", retain)
    progress.stage = "warmup_drain"
    progress.after_warmup = await http._drain(
        client, (row.request_id for row in progress.warmup)
    )
    if not _complete_counts(progress.after_warmup, plan.warmup_count):
        raise ShiftReplayError("warmup stream is incomplete")
    progress.stage = "warmup_reset"
    progress.reset_state = await _control(
        client,
        "/v1/shift/reset",
        {"request_ids": [row.request_id for row in progress.warmup]},
    )
    reset = progress.reset_state
    progress.stage = "warmup_reset_validation"
    _identity(reset, plan)
    if (
        reset.phase != "measured"
        or reset.broken
        or reset.complete
        or reset.rows
        or reset.counts != progress.after_warmup
    ):
        raise ShiftReplayError("warmup reset changed counters or retained state")
    progress.phase = "measured"
    started = time.perf_counter_ns()
    _, progress.measured_timeout_drain_ms = await _phase(
        client, plan, "measured", progress.measured, progress
    )
    progress.measured_elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
    progress.checkpoint("measured", retain)
    progress.stage = "measured_drain"
    started = time.perf_counter_ns()
    progress.after_measured = await http._drain(
        client, (row.request_id for row in progress.measured)
    )
    progress.measured_drain_ms = (time.perf_counter_ns() - started) / 1_000_000
    progress.stage = "trace_control"
    progress.trace = await _control(client, "/v1/shift/state")


async def replay_shift_run(
    base_url: str,
    plan: ShiftPlan,
    *,
    retain: Callable[[str, bytes], None] | None = None,
) -> ShiftRun:
    """Retain terminal errors; acknowledge their completion before the next row.

    Plan identity is a claim until the producer binds its private manifest. A
    complete run still needs an offline-trace cross-check and authenticated
    process shutdown before it can be accepted as research evidence.
    """
    if type(plan) is not ShiftPlan:
        raise ShiftReplayError("use a typed shift plan")
    if retain is not None and not callable(retain):
        raise ShiftReplayError("retention callback must be callable")
    base_url = http._loopback_url(base_url)
    transport = httpx.AsyncHTTPTransport(
        retries=0,
        limits=httpx.Limits(
            max_connections=1, max_keepalive_connections=1, keepalive_expiry=5.0
        ),
        http1=True,
        http2=False,
    )
    progress = _ShiftProgress(
        manifest_sha256=plan.manifest_sha256,
        run_index=plan.run_index,
        warmup_count=plan.warmup_count,
        measured_count=len(plan.requests),
        warmup_started=[False] * plan.warmup_count,
        measured_started=[False] * len(plan.requests),
    )
    cancelled = False
    try:
        async with httpx.AsyncClient(
            base_url=base_url,
            transport=transport,
            timeout=http.DRAIN_DEADLINE_SECONDS,
            follow_redirects=False,
            trust_env=False,
        ) as client:
            try:
                await _replay_phases(client, plan, progress, retain)
            except asyncio.CancelledError:
                cancelled = True
                raise
            progress.stage = "client_cleanup"
        result = ShiftRun(
            plan,
            tuple(progress.warmup),
            tuple(progress.measured),
            progress.initial,
            progress.after_warmup,
            progress.after_measured,
            progress.trace,
            progress.measured_elapsed_ms,
            progress.measured_drain_ms,
            progress.measured_timeout_drain_ms,
        )
        progress.stage = "validation"
        validate_shift_run(result)
        return result
    except (Exception, asyncio.CancelledError) as exc:
        if cancelled or http._external_cancellation(exc):
            raise http.ReplayCancelledError(
                "shift replay cancelled; run is incomplete",
                progress=progress.snapshot(),
            ) from exc
        message = (
            str(exc) if isinstance(exc, http.ReplayError) else "shift execution failed"
        )
        raise ShiftReplayError(
            message,
            phase=progress.phase,
            outcomes=(
                progress.measured if progress.phase == "measured" else progress.warmup
            ),
            warmup_outcomes=progress.warmup,
            progress=progress.snapshot(),
        ) from exc


def validate_shift_run(run):
    if type(run) is not ShiftRun or type(run.plan) is not ShiftPlan:
        raise ShiftReplayError("use a typed shift run")
    for snapshot in (run.initial, run.after_warmup, run.after_measured):
        if type(snapshot) is not DrainResponse:
            raise ShiftReplayError("missing drained counters")
        DrainResponse.model_validate(snapshot.model_dump())
    if type(run.trace) is not ShiftStateResponse:
        raise ShiftReplayError("missing typed trace")
    trace = ShiftStateResponse.model_validate(run.trace.model_dump())
    _identity(trace, run.plan)
    if (
        trace.broken
        or not trace.complete
        or trace.phase != "measured"
        or trace.counts != run.after_measured
        or any(run.initial.model_dump().values())
    ):
        raise ShiftReplayError("trace is not a completed fresh stream")
    if (
        len(run.warmup) != run.plan.warmup_count
        or len(run.measured) != len(run.plan.requests)
        or len(trace.rows) != len(run.measured)
    ):
        raise ShiftReplayError("trace or phase length mismatch")
    for phase, before, after in (
        ("warmup", run.initial, run.after_warmup),
        ("measured", run.after_warmup, run.after_measured),
    ):
        counts = http._check_phase(run, phase, before, after)
        if (
            counts["completed_requests"] != len(getattr(run, phase))
            or counts["failed_requests"]
        ):
            raise ShiftReplayError("stream contains incomplete rows")
        for i, outcome in enumerate(getattr(run, phase)):
            if outcome.record_id != run.plan.requests[i].record_id:
                raise ShiftReplayError("source identity/order mismatch")
            if (
                outcome.response is not None
                and outcome.response.admission_sequence
                != before.admitted_requests + i + 1
            ):
                raise ShiftReplayError("serialized admission order mismatch")
    invoked = 0
    nll = []
    override_through = 0
    for i, (row, outcome) in enumerate(zip(trace.rows, run.measured, strict=True)):
        selected = row.band_selected or row.drift_override
        if (
            row.position != i + 1
            or row.request_id != outcome.request_id
            or row.admission_sequence != run.plan.warmup_count + i + 1
            or row.stage2_invoked != selected
            or selected != (row.transformer_probability is not None)
        ):
            raise ShiftReplayError("trace identity or routing inconsistency")
        if row.drift_override != (row.position <= override_through) or (
            (not row.drift_override or row.band_selected)
            and row.fixed_decision != row.decision
        ):
            raise ShiftReplayError("trace violates future-only routing")
        nll.append(row.monitor_nll)
        window_due = row.position >= 256 and (row.position - 256) % 64 == 0
        if window_due != (row.window is not None):
            raise ShiftReplayError("trace complete-window placement differs")
        if row.window is not None:
            _, means = gmm_monitor.window_scores(nll[-256:])
            if (
                row.window.start_position != row.position - 255
                or row.window.end_position != row.position
                or row.window.score != means[0]
            ):
                raise ShiftReplayError("trace complete-window score differs")
            if row.window.alert:
                override_through = row.position + 256
        if outcome.response is not None:
            response = outcome.response
            probability = (
                row.transformer_probability if selected else row.stage1_probability
            )
            if (
                response.admission_sequence != row.admission_sequence
                or response.stage2_invoked != selected
                or response.probability != probability
                or response.action != ("alert" if row.decision else "allow")
            ):
                raise ShiftReplayError("response differs from saved live trace")
        invoked += selected
    for key in ("transformer_forward_attempts", "successful_transformer_scores"):
        if getattr(run.after_measured, key) - getattr(run.after_warmup, key) != invoked:
            raise ShiftReplayError("trace differs from physical forward counts")
    if (
        isinstance(run.measured_timeout_drain_ms, bool)
        or not isinstance(run.measured_timeout_drain_ms, (int, float))
        or not math.isfinite(run.measured_timeout_drain_ms)
        or run.measured_timeout_drain_ms < 0
    ):
        raise ShiftReplayError("invalid timeout drain interval")
    if run.measured_elapsed_ms is None or run.measured_drain_ms is None:
        raise ShiftReplayError("missing measured intervals")
    http._validate_timings(run)
    minimum = (
        math.fsum(row.elapsed_ms for row in run.measured)
        + run.measured_timeout_drain_ms
    )
    if run.measured_elapsed_ms < minimum and not math.isclose(
        run.measured_elapsed_ms, minimum, rel_tol=1e-12, abs_tol=1e-6
    ):
        raise ShiftReplayError("serialized interval omits request or inline drain time")


def verify_offline_trace(
    run,
    probabilities,
    monitor_scores,
    *,
    stage1_threshold,
    transformer_threshold,
    half_width,
    monitor_boundary,
):
    """Check exact saved/live agreement; callers authenticate both input streams."""
    validate_shift_run(run)
    expected = policy_replay.replay_policy(
        probabilities,
        monitor_scores,
        stage1_threshold=stage1_threshold,
        transformer_threshold=transformer_threshold,
        half_width=half_width,
        monitor_boundary=monitor_boundary,
    )
    if len(expected.rows) != len(run.measured):
        raise ShiftReplayError("offline/live lengths differ")
    windows = {window.end_position: window for window in expected.windows}
    for i, (live, reference, pair, monitor) in enumerate(
        zip(run.trace.rows, expected.rows, probabilities, monitor_scores, strict=True)
    ):
        window = windows.get(i + 1)
        if (
            reference.record_id != run.measured[i].record_id
            or live.stage1_probability != pair.stage1_probability
            or (
                live.transformer_probability is not None
                and live.transformer_probability != pair.transformer_probability
            )
            or live.monitor_nll != monitor.negative_log_likelihood
            or live.fixed_decision != reference.fixed_decision
            or live.decision != reference.policy_decision
            or live.band_selected != reference.logical_band
            or live.drift_override != reference.drift_override
            or live.stage2_invoked != reference.logical_stage2_mask
            or (live.window is None) != (window is None)
        ):
            raise ShiftReplayError("offline/live scoring or routing differs")
        if window is not None and (
            live.window.start_position != window.start_position
            or live.window.end_position != window.end_position
            or live.window.score != window.score
            or live.window.alert != window.alert
        ):
            raise ShiftReplayError("offline/live windows differ")


def summarize_shift_run(run):
    validate_shift_run(run)
    counts = http._check_phase(run, "measured", run.after_warmup, run.after_measured)
    total = len(run.measured)
    errors = sum(row.error is not None for row in run.measured)
    quantiles = np.quantile(
        np.asarray([row.elapsed_ms for row in run.measured], dtype=np.float64),
        [0.5, 0.95, 0.99],
        method="linear",
    )
    return {
        "workload": "shift_period",
        "primary_evidence": False,
        "manifest_sha256": run.manifest_sha256,
        "run_index": run.run_index,
        "concurrency": 1,
        "request_count": total,
        "request_errors": errors,
        "request_error_rate": errors / total,
        "p50_ms": float(quantiles[0]),
        "p95_ms": float(quantiles[1]),
        "p99_ms": float(quantiles[2]),
        "measured_elapsed_ms": run.measured_elapsed_ms,
        "measured_drain_ms": run.measured_drain_ms,
        "measured_timeout_drain_ms": run.measured_timeout_drain_ms,
        "client_attempts_per_second": total * 1000 / run.measured_elapsed_ms,
        "successful_responses_per_second": (total - errors)
        * 1000
        / run.measured_elapsed_ms,
        **counts,
    }
