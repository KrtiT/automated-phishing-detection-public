"""Fixed request plan and private control responses for serialized shift replay."""

import re
from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, ValidationError

from .http_replay import ReplayRequest, _request_id
from .http_schema import DrainResponse, RequestID, ScanRequest

Probability = Annotated[float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)]
FiniteFloat = Annotated[float, Field(strict=True, allow_inf_nan=False)]
BinaryInt = Annotated[int, Field(strict=True, ge=0, le=1)]
PositiveInt = Annotated[int, Field(strict=True, ge=1)]


@dataclass(frozen=True)
class ShiftPlan:
    """A claimed manifest identity; the execution producer authenticates its bytes."""

    manifest_sha256: str
    run_index: int
    requests: tuple[ReplayRequest, ...]
    warmup_count: int = 1000

    def __post_init__(self):
        if type(self.manifest_sha256) is not str or not re.fullmatch(
            r"[0-9a-f]{64}", self.manifest_sha256
        ):
            raise ValueError("invalid manifest SHA-256")
        if type(self.run_index) is not int or not 1 <= self.run_index <= 5:
            raise ValueError("run_index must be 1 through 5")
        if type(self.requests) is not tuple or not self.requests:
            raise ValueError("requests must be a nonempty tuple")
        if type(self.warmup_count) is not int or not 1 <= self.warmup_count <= len(
            self.requests
        ):
            raise ValueError("insufficient or invalid warmup capacity")
        seen = set()
        for row in self.requests:
            if type(row) is not ReplayRequest:
                raise ValueError("use typed replay requests")
            try:
                ScanRequest(request_id=row.record_id, url=row.raw_url)
            except ValidationError as exc:
                raise ValueError("invalid replay request") from exc
            if row.record_id in seen:
                raise ValueError("duplicate source record ID")
            seen.add(row.record_id)

    def request_id(self, phase: str, position: int) -> str:
        if (
            phase not in ("warmup", "measured")
            or type(position) is not int
            or position < 0
        ):
            raise ValueError("invalid phase or position")
        return _request_id(self.manifest_sha256, 1, self.run_index, phase, position)


class ShiftWindow(BaseModel):
    model_config = ConfigDict(extra="forbid")
    start_position: PositiveInt
    end_position: PositiveInt
    score: FiniteFloat
    alert: StrictBool


class ShiftTraceRow(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request_id: RequestID
    admission_sequence: PositiveInt
    position: PositiveInt
    stage1_probability: Probability
    transformer_probability: Probability | None
    fixed_decision: BinaryInt
    decision: BinaryInt
    band_selected: StrictBool
    drift_override: StrictBool
    stage2_invoked: StrictBool
    monitor_nll: FiniteFloat
    window: ShiftWindow | None


class ShiftStateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    workload: Literal["shift_period"] = "shift_period"
    manifest_sha256: Annotated[str, Field(strict=True, pattern=r"^[0-9a-f]{64}$")]
    run_index: Annotated[int, Field(strict=True, ge=1, le=5)]
    measured_count: PositiveInt
    warmup_count: PositiveInt
    phase: Literal["warmup", "measured"]
    broken: StrictBool
    complete: StrictBool
    counts: DrainResponse
    rows: list[ShiftTraceRow]
