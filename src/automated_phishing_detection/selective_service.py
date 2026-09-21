"""Single-owner loopback service; bound_runtime supplies authenticated loading."""

from __future__ import annotations

import asyncio
import math
import threading
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future
from contextlib import AbstractContextManager, asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from fastapi import FastAPI, HTTPException

from .http_schema import (
    HTTP_WORKLOADS,
    DrainRequest,
    DrainResponse,
    ScanRequest,
    ScanResponse,
)

if TYPE_CHECKING:
    from .selective_inference import (
        InferenceCounts,
        RequestScores,
        TransformerOnlyScores,
    )


class Scorer(Protocol):
    @property
    def counts(self) -> InferenceCounts: ...

    def scan(self, raw_url: str, *, drift_override: bool = False) -> RequestScores: ...

    def scan_transformer(self, raw_url: str) -> TransformerOnlyScores: ...


class OwnerUnavailable(RuntimeError):
    """The owner cannot admit or complete work."""


class ScoringFailure(RuntimeError):
    """A single request failed without invalidating the owner context."""


@dataclass(frozen=True)
class _Scan:
    request: ScanRequest
    sequence: int
    future: Future


@dataclass(frozen=True)
class _Barrier:
    admitted: int
    future: Future


_STOP = object()


def _public_response(job: _Scan, scores: RequestScores) -> ScanResponse:
    flags = (
        scores.band_selected,
        scores.drift_override,
        scores.logical_stage2_selected,
        scores.transformer_evaluated,
    )
    if any(type(flag) is not bool for flag in flags):
        raise ValueError("invalid scorer flags")
    selected = scores.logical_stage2_selected
    if (
        scores.drift_override
        or selected != scores.band_selected
        or selected != scores.transformer_evaluated
        or selected != (scores.transformer_probability is not None)
        or type(scores.decision) is not int
        or scores.decision not in (0, 1)
        or type(scores.fixed_decision) is not int
        or scores.fixed_decision != scores.decision
        or scores.stage1_probability is None
    ):
        raise ValueError("inconsistent scorer output")
    for probability in (scores.stage1_probability, scores.transformer_probability):
        if probability is not None and (
            isinstance(probability, bool)
            or not isinstance(probability, (int, float))
            or not math.isfinite(probability)
            or not 0 <= probability <= 1
        ):
            raise ValueError("invalid scorer probability")
    return ScanResponse(
        request_id=job.request.request_id,
        admission_sequence=job.sequence,
        action="alert" if scores.decision else "allow",
        probability=(
            scores.transformer_probability if selected else scores.stage1_probability
        ),
        stage2_invoked=selected,
    )


async def _shielded(future: Future):
    wrapped = asyncio.wrap_future(future)
    # Retrieve late failures even when the requesting ASGI task has disappeared.
    wrapped.add_done_callback(
        lambda done: None if done.cancelled() else done.exception()
    )
    return await asyncio.shield(wrapped)


def _public_transformer_response(job, scores):
    if type(scores.decision) is not int or scores.decision not in (0, 1):
        raise ValueError("invalid transformer-only decision")
    return ScanResponse(
        request_id=job.request.request_id,
        admission_sequence=job.sequence,
        action="alert" if scores.decision else "allow",
        probability=scores.probability,
        stage2_invoked=True,
    )


class ScoringOwner:
    """Own one synchronous scorer and serialize admissions, barriers, and shutdown."""

    def __init__(
        self,
        scorer_factory: Callable[[], AbstractContextManager[Scorer]],
        *,
        queue_capacity: int,
        workload: str = "fixed_cascade",
    ) -> None:
        if type(workload) is not str or workload not in HTTP_WORKLOADS:
            raise ValueError("unsupported HTTP workload")
        if type(queue_capacity) is not int or queue_capacity < 1:
            raise ValueError("queue_capacity must be a positive integer")
        self._factory = scorer_factory
        self._workload = workload
        self._capacity = queue_capacity
        self._condition = threading.Condition()
        self._queue: deque = deque()
        self._waiting = 0
        self._admitted = 0
        self._completed = 0
        self._failed = 0
        self._accepting = False
        self._healthy = False
        self._stopping = False
        self._failure = False
        self._closed_ids: set[str] = set()
        self._active: _Scan | _Barrier | None = None
        self._started: Future = Future()
        self._thread = threading.Thread(
            target=self._run, name="selective-inference-owner", daemon=False
        )

    @property
    def admitted_requests(self) -> int:
        with self._condition:
            return self._admitted

    @property
    def queued_requests(self) -> int:
        with self._condition:
            return self._waiting

    @property
    def accepting(self) -> bool:
        with self._condition:
            return self._accepting

    @property
    def healthy(self) -> bool:
        with self._condition:
            return self._healthy

    @property
    def is_alive(self) -> bool:
        return self._thread.is_alive()

    async def start(self) -> None:
        self._thread.start()
        await _shielded(self._started)

    def admit(self, request: ScanRequest) -> Future:
        with self._condition:
            if (
                not self._accepting
                or self._waiting >= self._capacity
                or request.request_id in self._closed_ids
            ):
                raise OwnerUnavailable("scoring service unavailable")
            self._admitted += 1
            self._closed_ids.add(request.request_id)
            future: Future = Future()
            self._queue.append(_Scan(request, self._admitted, future))
            self._waiting += 1
            self._condition.notify()
            return future

    def drain(self, *, request_ids: tuple[str, ...] = ()) -> Future:
        with self._condition:
            if not self._accepting:
                raise OwnerUnavailable("scoring service unavailable")
            # Fence phase IDs even when their HTTP bodies have not arrived yet.
            self._closed_ids.update(request_ids)
            future: Future = Future()
            self._queue.append(_Barrier(self._admitted, future))
            self._condition.notify()
            return future

    async def shutdown(self, *, timeout: float = 60.0) -> None:
        with self._condition:
            self._accepting = False
            if not self._stopping:
                self._stopping = True
                self._queue.append(_STOP)
                self._condition.notify()
        cancelled = False
        if self._thread.ident is not None:
            join = asyncio.create_task(asyncio.to_thread(self._thread.join, timeout))
            while not join.done():
                try:
                    await asyncio.shield(join)
                except asyncio.CancelledError:
                    cancelled = True
            join.result()
        if self._thread.is_alive():
            self._fatal()
            raise RuntimeError("scoring owner shutdown timed out; owner is still alive")
        if self._failure:
            raise RuntimeError("scoring owner failed; cleanup is not confirmed")
        if cancelled:
            raise asyncio.CancelledError

    def _fatal(self) -> None:
        with self._condition:
            self._failure = True
            self._healthy = self._accepting = False
            jobs = list(self._queue)
            self._queue.clear()
            self._waiting = 0
            self._stopping = True
            self._queue.append(_STOP)
            self._condition.notify()
            if self._active is not None:
                jobs.insert(0, self._active)
                self._active = None
            for job in jobs:
                if isinstance(job, (_Scan, _Barrier)) and not job.future.done():
                    if isinstance(job, _Scan):
                        self._failed += 1
                    job.future.set_exception(
                        OwnerUnavailable("scoring service unavailable")
                    )
            if not self._started.done():
                self._started.set_exception(
                    OwnerUnavailable("scoring owner startup failed")
                )

    def _serve(self, scorer: Scorer) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(lambda: bool(self._queue))
                job = self._queue.popleft()
                self._active = job if isinstance(job, (_Scan, _Barrier)) else None
                if isinstance(job, _Scan):
                    self._waiting -= 1
            if job is _STOP:
                return
            try:
                if isinstance(job, _Barrier):
                    counts = scorer.counts
                    result = DrainResponse(
                        admitted_requests=job.admitted,
                        completed_requests=self._completed,
                        failed_requests=self._failed,
                        transformer_forward_attempts=counts.transformer_forward_attempts,
                        successful_transformer_scores=counts.successful_transformer_scores,
                    )
                else:
                    try:
                        if self._workload == "transformer_only":
                            result = _public_transformer_response(
                                job, scorer.scan_transformer(job.request.url)
                            )
                        else:
                            scores = scorer.scan(job.request.url, drift_override=False)
                            result = _public_response(job, scores)
                    except Exception:
                        with self._condition:
                            if not job.future.done():
                                self._failed += 1
                                job.future.set_exception(
                                    ScoringFailure("request scoring failed")
                                )
                            self._active = None
                        continue
                with self._condition:
                    if not job.future.done():
                        if isinstance(job, _Scan):
                            self._completed += 1
                        job.future.set_result(result)
                    self._active = None
            except BaseException:
                self._fatal()
                raise

    def _run(self) -> None:
        try:
            with self._factory() as scorer:
                with self._condition:
                    self._healthy = not self._failure
                    self._accepting = not self._stopping
                    self._started.set_result(None)
                self._serve(scorer)
        except BaseException:
            self._fatal()
        finally:
            with self._condition:
                self._healthy = self._accepting = False


def create_app(
    scorer_factory: Callable[[], AbstractContextManager[Scorer]],
    *,
    queue_capacity: int = 128,
    workload: str = "fixed_cascade",
) -> FastAPI:
    """Build one single-use app; the supplied context is created only on its owner."""
    owner = ScoringOwner(
        scorer_factory, queue_capacity=queue_capacity, workload=workload
    )

    @asynccontextmanager
    async def lifespan(app):
        try:
            await owner.start()
            yield
        finally:
            await owner.shutdown()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.owner = owner
    app.state.workload = workload

    @app.post("/v1/scan", response_model=ScanResponse)
    async def scan(request: ScanRequest) -> ScanResponse:
        try:
            return await _shielded(owner.admit(request))
        except OwnerUnavailable:
            raise HTTPException(503, "scoring service unavailable") from None
        except ScoringFailure:
            raise HTTPException(500, "request scoring failed") from None

    @app.post("/v1/drain", response_model=DrainResponse)
    async def drain(request: DrainRequest | None = None) -> DrainResponse:
        try:
            request_ids = tuple(request.request_ids) if request is not None else ()
            return await _shielded(owner.drain(request_ids=request_ids))
        except OwnerUnavailable:
            raise HTTPException(503, "scoring service unavailable") from None

    return app
