"""One ordered, non-retriable live-monitor workload on the existing owner."""

from concurrent.futures import Future
from dataclasses import asdict, dataclass

from fastapi import HTTPException

from .http_schema import DrainRequest
from .selective_service import (
    OwnerUnavailable,
    ScoringOwner,
    _Barrier,
    _base_app,
    _public_response,
    _shielded,
)
from .shift_schema import ShiftPlan, ShiftStateResponse, ShiftTraceRow


class ShiftControlError(ValueError):
    """A reset would discard failed, missing, or already measured work."""


@dataclass(frozen=True)
class _ShiftControl(_Barrier):
    action: str


class ShiftOwner(ScoringOwner):
    def __init__(self, scorer_factory, plan: ShiftPlan):
        if type(plan) is not ShiftPlan:
            raise ValueError("use a typed shift plan")
        super().__init__(scorer_factory, queue_capacity=1)
        self.plan = plan
        self._phase = "warmup"
        self._broken = False
        self._reset_requested = False
        self._rows = []

    def admit(self, request):
        with self._condition:
            if self._broken:
                raise OwnerUnavailable("shift stream is incomplete")
            index = self._admitted - (
                self.plan.warmup_count if self._phase == "measured" else 0
            )
            limit = (
                self.plan.warmup_count
                if self._phase == "warmup"
                else len(self.plan.requests)
            )
            if (
                self._admitted != self._completed + self._failed
                or not 0 <= index < limit
                or (self._phase == "warmup" and self._reset_requested)
                or request.request_id != self.plan.request_id(self._phase, index)
                or request.url != self.plan.requests[index].raw_url
            ):
                self._broken = True
                raise OwnerUnavailable("shift admission differs from the fixed stream")
            return super().admit(request)

    def control(self, action, request_ids=()):
        with self._condition:
            if not self._accepting:
                raise OwnerUnavailable("scoring service unavailable")
            if action == "reset":
                expected = tuple(
                    self.plan.request_id("warmup", i)
                    for i in range(self.plan.warmup_count)
                )
                if (
                    self._broken
                    or self._reset_requested
                    or self._phase != "warmup"
                    or tuple(request_ids) != expected
                ):
                    raise ShiftControlError("warmup reset is unavailable")
                self._reset_requested = True
                self._closed_ids.update(request_ids)
            elif action != "state":
                raise ShiftControlError("unknown shift control")
            future = Future()
            self._queue.append(_ShiftControl(self._admitted, future, action))
            self._condition.notify()
            return future

    def _score_request(self, scorer, job):
        try:
            live = scorer.scan_shift(job.request.url)
            scores = live.scores
            if type(live.position) is not int or live.position != len(self._rows) + 1:
                raise ValueError("monitor position differs from admitted stream")
            response = _public_response(job, scores, allow_drift=True)
            row = ShiftTraceRow(
                request_id=job.request.request_id,
                admission_sequence=job.sequence,
                position=live.position,
                stage1_probability=scores.stage1_probability,
                transformer_probability=scores.transformer_probability,
                fixed_decision=scores.fixed_decision,
                decision=scores.decision,
                band_selected=scores.band_selected,
                drift_override=scores.drift_override,
                stage2_invoked=scores.transformer_evaluated,
                monitor_nll=live.monitor_nll,
                window=asdict(live.window) if live.window is not None else None,
            )
            self._rows.append(row)
            return response
        except BaseException:
            with self._condition:
                self._broken = True
            raise

    def _barrier_response(self, scorer, job):
        counts = super()._barrier_response(scorer, job)
        if not isinstance(job, _ShiftControl):
            return counts
        if job.action == "reset":
            if (
                self._broken
                or counts.failed_requests
                or counts.completed_requests != self.plan.warmup_count
                or counts.admitted_requests != self.plan.warmup_count
            ):
                self._broken = True
                # A control error is a rejected transition, not owner-thread death.
                job.future.set_exception(
                    ShiftControlError("warmup did not complete in full")
                )
                return None
            scorer.reset_monitor()
            self._rows.clear()
            with self._condition:
                self._phase = "measured"
        return ShiftStateResponse(
            manifest_sha256=self.plan.manifest_sha256,
            run_index=self.plan.run_index,
            measured_count=len(self.plan.requests),
            warmup_count=self.plan.warmup_count,
            phase=self._phase,
            broken=self._broken,
            complete=(
                not self._broken
                and self._phase == "measured"
                and len(self._rows) == len(self.plan.requests)
            ),
            counts=counts,
            rows=list(self._rows),
        )


def create_shift_app(scorer_factory, plan: ShiftPlan):
    owner = ShiftOwner(scorer_factory, plan)
    app = _base_app(owner, "shift_period")

    @app.post("/v1/shift/state", response_model=ShiftStateResponse)
    async def state():
        try:
            return await _shielded(owner.control("state"))
        except OwnerUnavailable:
            raise HTTPException(503, "shift state unavailable") from None

    @app.post("/v1/shift/reset", response_model=ShiftStateResponse)
    async def reset(request: DrainRequest):
        try:
            return await _shielded(owner.control("reset", tuple(request.request_ids)))
        except ShiftControlError:
            raise HTTPException(409, "warmup reset rejected") from None
        except OwnerUnavailable:
            raise HTTPException(503, "shift reset unavailable") from None

    return app
