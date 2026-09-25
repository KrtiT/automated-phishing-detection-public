"""Closed-loop HTTP measurements; no dataset reader or official execution runner.

Small fixture runs exercise the same client. Only complete, matching five-run
evidence can produce the primary operational summary. Source provenance and
hardware/artifact binding remain the execution runner's responsibility.
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import math
import re
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from urllib.parse import urlsplit

import httpx
import numpy as np
from pydantic import ValidationError

from .http_schema import HTTP_WORKLOADS, DrainResponse, ScanRequest, ScanResponse
from .hypothesis_evaluation import PrimaryHttpSummary, ReferenceInvocations

DEADLINE_SECONDS = 2.0
DRAIN_DEADLINE_SECONDS = 60.0
CONCURRENCIES = (1, 8, 16, 32, 64, 128)
ERRORS = frozenset(
    {
        "timeout",
        "transport",
        "http_status",
        "invalid_json",
        "invalid_schema",
        "correlation",
    }
)


class ReplayError(ValueError):
    """A run is incomplete or cannot support the declared measurement."""

    def __init__(self, *args, progress: bytes | None = None):
        super().__init__(*args)
        self.progress = progress


class ReplayCancelledError(asyncio.CancelledError):
    """Cancellation with an immutable snapshot of completed HTTP evidence."""

    def __init__(self, *args, progress: bytes):
        super().__init__(*args)
        self.progress = progress


def _exception_chain(error):
    seen = set()
    while isinstance(error, BaseException) and id(error) not in seen:
        seen.add(id(error))
        yield error
        error = error.__cause__ if error.__cause__ is not None else error.__context__


def _external_cancellation(error):
    for current in _exception_chain(error):
        if isinstance(current, asyncio.CancelledError):
            return True
        if isinstance(current, (asyncio.TimeoutError, httpx.TimeoutException)):
            break
    return False


def replay_progress(error: BaseException) -> bytes | None:
    """Recover retained bytes, including Python 3.10 task-cancellation context."""
    for current in _exception_chain(error):
        if (
            isinstance(current, (ReplayError, ReplayCancelledError))
            and type(current.progress) is bytes
        ):
            return current.progress
    return None


@dataclass(frozen=True)
class ReplayRequest:
    record_id: str
    raw_url: str


@dataclass(frozen=True)
class HttpOutcome:
    record_id: str
    request_id: str
    elapsed_ms: float
    status_code: int | None
    error: str | None
    response: ScanResponse | None


@dataclass(frozen=True)
class HttpRun:
    manifest_sha256: str
    prevalence_basis_points: int
    concurrency: int
    run_index: int
    warmup: tuple[HttpOutcome, ...]
    measured: tuple[HttpOutcome, ...]
    initial: DrainResponse
    after_warmup: DrainResponse
    after_measured: DrainResponse
    workload: str = "fixed_cascade"
    measured_elapsed_ms: float | None = None
    measured_drain_ms: float | None = None


@dataclass
class _ReplayProgress:
    manifest_sha256: str
    prevalence_basis_points: int
    concurrency: int
    run_index: int
    workload: str
    warmup: list[HttpOutcome | None]
    measured: list[HttpOutcome | None]
    warmup_started: list[bool]
    measured_started: list[bool]
    stage: str = "client_start"
    initial: DrainResponse | None = None
    after_warmup: DrainResponse | None = None
    after_measured: DrainResponse | None = None
    measured_elapsed_ms: float | None = None
    measured_drain_ms: float | None = None

    def snapshot(self) -> bytes:
        def wire_model(value):
            if type(value) in (ScanResponse, DrainResponse):
                return value.model_dump()
            raise TypeError("invalid HTTP progress value")

        return (
            json.dumps(
                {"schema_version": 1, **asdict(self)},
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


def _metadata(sha, prevalence, concurrency, run_index):
    if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha):
        raise ReplayError("manifest SHA-256 must be lowercase hexadecimal")
    if type(prevalence) is not int or prevalence not in (10, 100, 500):
        raise ReplayError("prevalence must be 10, 100, or 500 basis points")
    if type(concurrency) is not int or concurrency not in CONCURRENCIES:
        raise ReplayError("concurrency is outside the fixed sweep")
    if type(run_index) is not int or not 1 <= run_index <= 5:
        raise ReplayError("run_index must be an integer from 1 through 5")


def _loopback_url(value):
    try:
        parsed = urlsplit(value)
        valid = (
            parsed.scheme == "http"
            and ipaddress.ip_address(parsed.hostname).is_loopback
            and parsed.username is None
            and parsed.password is None
            and parsed.path in ("", "/")
            and not parsed.query
            and not parsed.fragment
            and parsed.port is not None
        )
    except (ValueError, TypeError):
        valid = False
    if not valid:
        raise ReplayError("use an HTTP loopback IP and explicit port, without a path")
    return value.rstrip("/")


def _request_id(sha, concurrency, run_index, phase, position):
    return f"{sha}.{concurrency}.{run_index}.{phase}.{position}"


def _json(body):
    def reject_constant(value):
        raise ValueError("nonfinite JSON constant")

    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    return json.loads(
        body, parse_constant=reject_constant, object_pairs_hook=unique_pairs
    )


async def _scan(client, row, request_id):
    status = None
    error = None
    response = None

    async def exchange():
        nonlocal status, error, response
        received = await client.post(
            "/v1/scan", json={"request_id": request_id, "url": row.raw_url}
        )
        status = received.status_code
        if status != 200:
            error = "http_status"
            return
        try:
            payload = _json(received.content)
        except (ValueError, UnicodeError, RecursionError):
            error = "invalid_json"
            return
        try:
            response = ScanResponse.model_validate(payload)
        except ValidationError:
            error = "invalid_schema"
            return
        if response.request_id != request_id:
            error = "correlation"

    # A worker has already acquired its concurrency slot; local backlog is excluded.
    started = time.perf_counter_ns()
    try:
        await asyncio.wait_for(exchange(), timeout=DEADLINE_SECONDS)
    except (asyncio.TimeoutError, httpx.TimeoutException):
        error = "timeout"
    except httpx.RequestError:
        error = "transport"
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
    # Synchronous JSON/schema validation cannot be interrupted by wait_for.
    if elapsed_ms > DEADLINE_SECONDS * 1000:
        error = "timeout"
    return HttpOutcome(
        row.record_id,
        request_id,
        elapsed_ms,
        status,
        error,
        None if error else response,
    )


async def _phase(client, rows, sha, concurrency, run_index, phase, outcomes, started):
    remaining = iter(enumerate(rows))

    async def worker():
        for position, row in remaining:
            request_id = _request_id(sha, concurrency, run_index, phase, position)
            started[position] = True
            outcomes[position] = await _scan(client, row, request_id)

    tasks = [asyncio.create_task(worker()) for _ in range(min(concurrency, len(rows)))]
    try:
        await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    return tuple(outcomes)


async def _drain(client, request_ids):
    async def exchange():
        response = await client.post(
            "/v1/drain",
            json={"request_ids": list(request_ids)},
            timeout=DRAIN_DEADLINE_SECONDS,
        )
        if response.status_code != 200:
            raise ReplayError("drain did not return HTTP 200")
        return DrainResponse.model_validate(_json(response.content))

    try:
        return await asyncio.wait_for(exchange(), timeout=DRAIN_DEADLINE_SECONDS)
    except (
        ValueError,
        UnicodeError,
        RecursionError,
        httpx.RequestError,
        asyncio.TimeoutError,
    ) as exc:
        raise ReplayError("drain failed; run is incomplete") from exc


async def _replay_phases(client, rows, warmup_count, progress, retain):
    progress.stage = "initial_drain"
    progress.initial = await _drain(client, ())
    if any(progress.initial.model_dump().values()):
        raise ReplayError("each run requires a fresh service and scorer session")
    progress.stage = "warmup"
    warmup = await _phase(
        client,
        rows[:warmup_count],
        progress.manifest_sha256,
        progress.concurrency,
        progress.run_index,
        "warmup",
        progress.warmup,
        progress.warmup_started,
    )
    progress.checkpoint("warmup", retain)
    progress.stage = "warmup_drain"
    progress.after_warmup = await _drain(client, (row.request_id for row in warmup))
    progress.stage = "measured"
    measured_started = time.perf_counter_ns()
    measured = await _phase(
        client,
        rows,
        progress.manifest_sha256,
        progress.concurrency,
        progress.run_index,
        "measured",
        progress.measured,
        progress.measured_started,
    )
    progress.measured_elapsed_ms = (
        time.perf_counter_ns() - measured_started
    ) / 1_000_000
    progress.checkpoint("measured", retain)
    progress.stage = "measured_drain"
    drain_started = time.perf_counter_ns()
    progress.after_measured = await _drain(client, (row.request_id for row in measured))
    progress.measured_drain_ms = (time.perf_counter_ns() - drain_started) / 1_000_000
    return warmup, measured


def _retain_direct_interruption(error, progress):
    if not isinstance(error, (Exception, asyncio.CancelledError)):
        try:
            error.progress = progress.snapshot()
        except BaseException:
            pass


async def replay_run(
    base_url: str,
    requests: tuple[ReplayRequest, ...] | list[ReplayRequest],
    *,
    manifest_sha256: str,
    prevalence_basis_points: int,
    concurrency: int,
    run_index: int,
    warmup_count: int = 1000,
    workload: str = "fixed_cascade",
    retain: Callable[[str, bytes], None] | None = None,
) -> HttpRun:
    """Measure one fresh service session. Cancelled/incomplete runs raise.

    requests must be the manifest order, with original raw URLs. The supplied
    manifest hash is a provenance claim authenticated by the future runner, not
    by this label-free client. Fixture-sized inputs cannot form primary evidence.
    An optional synchronous retain callback receives each completed phase once,
    outside request timing. Failures carry canonical private JSON in progress.
    """
    _metadata(manifest_sha256, prevalence_basis_points, concurrency, run_index)
    if type(workload) is not str or workload not in HTTP_WORKLOADS:
        raise ReplayError("unsupported HTTP workload")
    if retain is not None and not callable(retain):
        raise ReplayError("retain must be a callable or None")
    base_url = _loopback_url(base_url)
    if type(requests) not in (tuple, list) or not requests:
        raise ReplayError("use a nonempty materialized request sequence")
    rows = tuple(requests)
    seen = set()
    for row in rows:
        if type(row) is not ReplayRequest:
            raise ReplayError("use typed replay requests")
        try:
            ScanRequest(request_id=row.record_id, url=row.raw_url)
        except ValidationError as exc:
            raise ReplayError("invalid replay request") from exc
        if row.record_id in seen:
            raise ReplayError("duplicate source record ID")
        seen.add(row.record_id)
    if type(warmup_count) is not int or not 1 <= warmup_count <= len(rows):
        raise ReplayError(
            "warmup_count must be positive and no larger than the manifest"
        )
    limits = httpx.Limits(
        max_connections=concurrency,
        max_keepalive_connections=concurrency,
        keepalive_expiry=5.0,
    )
    transport = httpx.AsyncHTTPTransport(
        retries=0, limits=limits, http1=True, http2=False
    )
    progress = _ReplayProgress(
        manifest_sha256=manifest_sha256,
        prevalence_basis_points=prevalence_basis_points,
        concurrency=concurrency,
        run_index=run_index,
        workload=workload,
        warmup=[None] * warmup_count,
        measured=[None] * len(rows),
        warmup_started=[False] * warmup_count,
        measured_started=[False] * len(rows),
    )
    cancelled = False
    original = None
    try:
        async with httpx.AsyncClient(
            base_url=base_url,
            transport=transport,
            timeout=DEADLINE_SECONDS,
            follow_redirects=False,
            trust_env=False,
        ) as client:
            try:
                warmup, measured = await _replay_phases(
                    client, rows, warmup_count, progress, retain
                )
            except BaseException as error:
                original = error
                cancelled = isinstance(error, asyncio.CancelledError)
                _retain_direct_interruption(error, progress)
                raise
            progress.stage = "client_cleanup"
        result = HttpRun(
            manifest_sha256,
            prevalence_basis_points,
            concurrency,
            run_index,
            warmup,
            measured,
            progress.initial,
            progress.after_warmup,
            progress.after_measured,
            workload,
            progress.measured_elapsed_ms,
            progress.measured_drain_ms,
        )
        progress.stage = "validation"
        _validate_run(result)
        return result
    except BaseException as exc:
        selected = (
            original
            if original is not None and not isinstance(original, Exception)
            else exc
        )
        if not isinstance(selected, (Exception, asyncio.CancelledError)):
            _retain_direct_interruption(selected, progress)
            raise selected from None
        if cancelled or _external_cancellation(exc):
            raise ReplayCancelledError(
                "HTTP replay cancelled; run is incomplete", progress=progress.snapshot()
            ) from exc
        message = str(exc) if isinstance(exc, ReplayError) else "HTTP replay failed"
        raise ReplayError(message, progress=progress.snapshot()) from exc


def _check_phase(run, phase, before, after):
    rows = getattr(run, phase)
    if type(rows) is not tuple or not rows:
        raise ReplayError("phase outcomes must be a nonempty tuple")
    delta = {
        key: getattr(after, key) - value for key, value in before.model_dump().items()
    }
    if any(value < 0 for value in delta.values()):
        raise ReplayError("server counters regressed")
    admitted = delta["admitted_requests"]
    if (
        admitted > len(rows)
        or delta["completed_requests"] + delta["failed_requests"] != admitted
        or not 0
        <= delta["successful_transformer_scores"]
        <= delta["transformer_forward_attempts"]
        <= admitted
        or delta["transformer_forward_attempts"]
        - delta["successful_transformer_scores"]
        > delta["failed_requests"]
    ):
        raise ReplayError("phase counters are inconsistent or not drained")
    identities, sequences = set(), set()
    successful = invoked = 0
    for position, row in enumerate(rows):
        if type(row) is not HttpOutcome:
            raise ReplayError("use typed HTTP outcomes")
        if (
            not isinstance(row.record_id, str)
            or not row.record_id
            or row.record_id in identities
        ):
            raise ReplayError("invalid or duplicate source record ID")
        identities.add(row.record_id)
        if row.request_id != _request_id(
            run.manifest_sha256, run.concurrency, run.run_index, phase, position
        ):
            raise ReplayError("request occurrence identity/order mismatch")
        if (
            isinstance(row.elapsed_ms, bool)
            or not isinstance(row.elapsed_ms, (float, int))
            or not math.isfinite(row.elapsed_ms)
            or row.elapsed_ms < 0
        ):
            raise ReplayError("invalid latency")
        if row.status_code is not None and (
            type(row.status_code) is not int or not 100 <= row.status_code <= 599
        ):
            raise ReplayError("invalid HTTP status")
        if row.error is not None:
            if row.error not in ERRORS or row.response is not None:
                raise ReplayError("invalid terminal error")
            continue
        if (
            type(row.response) is not ScanResponse
            or row.status_code != 200
            or row.elapsed_ms > DEADLINE_SECONDS * 1000
        ):
            raise ReplayError(
                "success requires a valid response within the total deadline"
            )
        response = ScanResponse.model_validate(row.response.model_dump())
        sequence = response.admission_sequence
        if (
            response.request_id != row.request_id
            or sequence in sequences
            or not before.admitted_requests < sequence <= after.admitted_requests
        ):
            raise ReplayError("response identity/admission sequence mismatch")
        sequences.add(sequence)
        successful += 1
        invoked += response.stage2_invoked
    if (
        successful > delta["completed_requests"]
        or invoked > delta["successful_transformer_scores"]
        or delta["transformer_forward_attempts"] > invoked + admitted - successful
    ):
        raise ReplayError(
            "client responses disagree with the drained physical counters"
        )
    if run.workload == "transformer_only" and (
        invoked != successful
        or delta["completed_requests"] > delta["successful_transformer_scores"]
    ):
        raise ReplayError(
            "transformer-only completions require successful physical forwards"
        )
    return delta


def _validate_timings(run):
    timings = (run.measured_elapsed_ms, run.measured_drain_ms)
    if all(value is None for value in timings):
        return
    try:
        if any(
            isinstance(value, bool)
            or not isinstance(value, (float, int))
            or not math.isfinite(value)
            or value <= 0
            for value in timings
        ):
            raise ReplayError("invalid measured phase/drain interval")
        slots = min(run.concurrency, len(run.measured))
        minimum = max(
            max(row.elapsed_ms for row in run.measured),
            math.fsum(row.elapsed_ms for row in run.measured) / slots,
        )
    except OverflowError as exc:
        raise ReplayError("invalid measured phase/drain interval") from exc
    # Closed-loop slots bound total request occupancy; allow only rounding noise.
    if run.measured_elapsed_ms < minimum and not math.isclose(
        run.measured_elapsed_ms, minimum, rel_tol=1e-12, abs_tol=1e-6
    ):
        raise ReplayError("measured phase interval is shorter than closed-loop work")


def _validate_run(run):
    if type(run) is not HttpRun:
        raise ReplayError("use typed HTTP runs")
    if type(run.workload) is not str or run.workload not in HTTP_WORKLOADS:
        raise ReplayError("unsupported HTTP workload")
    _metadata(
        run.manifest_sha256, run.prevalence_basis_points, run.concurrency, run.run_index
    )
    for snapshot in (run.initial, run.after_warmup, run.after_measured):
        if type(snapshot) is not DrainResponse:
            raise ReplayError("missing drained server counters")
        DrainResponse.model_validate(snapshot.model_dump())
    if any(run.initial.model_dump().values()):
        raise ReplayError("run did not begin with fresh counters")
    _check_phase(run, "warmup", run.initial, run.after_warmup)
    _check_phase(run, "measured", run.after_warmup, run.after_measured)
    if tuple(r.record_id for r in run.warmup) != tuple(
        r.record_id for r in run.measured[: len(run.warmup)]
    ):
        raise ReplayError("warmup must reuse the manifest prefix in the same order")
    _validate_timings(run)


def summarize_run(run: HttpRun) -> dict:
    """Descriptive client rates and physical counts, never a primary gate decision."""
    _validate_run(run)
    if run.measured_elapsed_ms is None:
        raise ReplayError("throughput requires measured wall-clock intervals")
    counts = _check_phase(run, "measured", run.after_warmup, run.after_measured)
    total = len(run.measured)
    errors = sum(row.error is not None for row in run.measured)
    quantiles = np.quantile(
        np.asarray([row.elapsed_ms for row in run.measured], dtype=np.float64),
        [0.5, 0.95, 0.99],
        method="linear",
    )
    return {
        "workload": run.workload,
        "manifest_sha256": run.manifest_sha256,
        "prevalence_basis_points": run.prevalence_basis_points,
        "concurrency": run.concurrency,
        "run_index": run.run_index,
        "request_count": total,
        "request_errors": errors,
        "request_error_rate": errors / total,
        "p50_ms": float(quantiles[0]),
        "p95_ms": float(quantiles[1]),
        "p99_ms": float(quantiles[2]),
        "measured_elapsed_ms": run.measured_elapsed_ms,
        "measured_drain_ms": run.measured_drain_ms,
        "client_attempts_per_second": total * 1000 / run.measured_elapsed_ms,
        "successful_responses_per_second": (total - errors)
        * 1000
        / run.measured_elapsed_ms,
        **counts,
    }


def reference_invocations(run: HttpRun) -> ReferenceInvocations:
    """Read actual measured counters, including forwards after client timeouts."""
    _validate_run(run)
    if run.workload != "fixed_cascade":
        raise ReplayError("primary reference requires fixed_cascade workload")
    if (
        run.prevalence_basis_points != 100
        or run.concurrency != 1
        or run.run_index != 1
        or len(run.measured) != 10000
        or len(run.warmup) != 1000
    ):
        raise ReplayError(
            "reference requires the first concurrency-1 run, 10,000-row 1% manifest and 1,000 warmups"
        )
    counts = _check_phase(run, "measured", run.after_warmup, run.after_measured)
    if counts["admitted_requests"] != len(run.measured):
        raise ReplayError(
            "reference request was not admitted; physical-reference evidence is incomplete"
        )
    return ReferenceInvocations(
        len(run.measured),
        counts["transformer_forward_attempts"],
        counts["successful_transformer_scores"],
        counts["completed_requests"],
        counts["failed_requests"],
    )


def primary_http_summary(
    runs: tuple[HttpRun, ...] | list[HttpRun],
) -> PrimaryHttpSummary:
    """Pool all 50,000 measured latencies, including terminal failures."""
    if type(runs) not in (tuple, list) or len(runs) != 5:
        raise ReplayError("primary summary requires five complete runs")
    for run in runs:
        _validate_run(run)
        if run.workload != "fixed_cascade":
            raise ReplayError("primary summary requires fixed_cascade workload")
        if (
            run.concurrency != 64
            or run.prevalence_basis_points != 100
            or len(run.measured) != 10000
            or len(run.warmup) != 1000
        ):
            raise ReplayError(
                "primary runs require concurrency 64, 1% prevalence, 1,000 warmups and 10,000 measured requests"
            )
    if {r.run_index for r in runs} != set(range(1, 6)) or len(
        {r.manifest_sha256 for r in runs}
    ) != 1:
        raise ReplayError("primary runs must be unique repeats of one manifest")
    identities = tuple(row.record_id for row in runs[0].measured)
    if any(tuple(row.record_id for row in run.measured) != identities for run in runs):
        raise ReplayError("primary repeats must preserve manifest order")
    latencies = np.asarray(
        [row.elapsed_ms for run in runs for row in run.measured], dtype=np.float64
    )
    return PrimaryHttpSummary(
        (10000,) * 5,
        64,
        sum(row.error is not None for run in runs for row in run.measured),
        float(np.quantile(latencies, 0.95, method="linear")),
    )
