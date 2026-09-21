"""Strict wire models for the loopback-only selective inference experiment."""

from typing import Annotated, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictStr,
    field_validator,
)

HTTP_WORKLOADS = ("fixed_cascade", "transformer_only")


def _request_id(value: str) -> str:
    if not value or not value.isprintable() or any(char.isspace() for char in value):
        raise ValueError(
            "request_id must be nonempty printable text without whitespace"
        )
    return value


RequestID = Annotated[StrictStr, AfterValidator(_request_id)]
NonnegativeInt = Annotated[int, Field(strict=True, ge=0)]


class ScanRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: RequestID
    url: Annotated[StrictStr, Field(min_length=1)]


class ScanResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: RequestID
    admission_sequence: Annotated[int, Field(strict=True, ge=1)]
    action: Literal["allow", "alert"]
    probability: Annotated[float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)]
    stage2_invoked: StrictBool


class DrainResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    admitted_requests: NonnegativeInt
    completed_requests: NonnegativeInt
    failed_requests: NonnegativeInt
    transformer_forward_attempts: NonnegativeInt
    successful_transformer_scores: NonnegativeInt


class DrainRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_ids: Annotated[list[RequestID], Field(strict=True)] = Field(
        default_factory=list
    )

    @field_validator("request_ids")
    @classmethod
    def unique_ids(cls, values: list[str]) -> list[str]:
        if len(set(values)) != len(values):
            raise ValueError("drain request_ids must be unique")
        return values
