"""Closed scientific JSON shapes shared by shift run and checkpoint records."""

import json
import math
from dataclasses import asdict, fields

from . import http_replay
from .http_schema import DrainResponse, ScanResponse
from .shift_replay import ShiftRun
from .shift_schema import ShiftPlan, ShiftStateResponse, ShiftTraceRow, ShiftWindow

RUN_FIELDS = {member.name for member in fields(ShiftRun)} | {
    "schema_version",
    "workload",
    "concurrency",
}
OUTCOME_FIELDS = {member.name for member in fields(http_replay.HttpOutcome)}
MODELS = (DrainResponse, ScanResponse, ShiftStateResponse, ShiftTraceRow, ShiftWindow)
_FLOAT_FIELDS = {
    ScanResponse: ("probability",),
    ShiftTraceRow: ("stage1_probability", "transformer_probability", "monitor_nll"),
    ShiftWindow: ("score",),
}


class ShiftRunCodecError(ValueError):
    """A symbolic complete-run or checkpoint consistency rejection."""


def require(condition):
    if not condition:
        raise ShiftRunCodecError("invalid_shift_run_record")


def keys(value, names):
    require(type(value) is dict and set(value) == set(names))


def number(value):
    require(type(value) in (int, float) and math.isfinite(value))
    require(value >= 0)


def _typed_model(value, kind):
    require(type(value) is kind)
    if kind is ShiftStateResponse:
        _typed_model(value.counts, DrainResponse)
        require(type(value.rows) is list)
        for row in value.rows:
            _typed_model(row, ShiftTraceRow)
    if kind is ShiftTraceRow and value.window is not None:
        _typed_model(value.window, ShiftWindow)


def _model_wire(value):
    require(type(value) in MODELS)
    _typed_model(value, type(value))
    return value.model_dump(warnings=False)


def dump(value):
    return (
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            default=_model_wire,
        )
        + "\n"
    ).encode("ascii")


def load(content):
    require(type(content) is bytes)
    value = http_replay._json(content)
    require(content == dump(value))
    return value


def model_shape(value, kind):
    keys(value, kind.model_fields)
    for name in _FLOAT_FIELDS.get(kind, ()):
        member = value[name]
        require(
            (name == "transformer_probability" and member is None)
            or (type(member) is float and math.isfinite(member))
        )
    if kind is ShiftStateResponse:
        model_shape(value["counts"], DrainResponse)
        require(type(value["rows"]) is list)
        for row in value["rows"]:
            model_shape(row, ShiftTraceRow)
    if kind is ShiftTraceRow and value["window"] is not None:
        model_shape(value["window"], ShiftWindow)


def outcomes_shape(values):
    require(type(values) is list)
    for value in values:
        keys(value, OUTCOME_FIELDS)
        for name in ("record_id", "request_id"):
            require(type(value[name]) is str and bool(value[name]))
        number(value["elapsed_ms"])
        status = value["status_code"]
        require(status is None or type(status) is int)
        error = value["error"]
        require(error is None or type(error) is str)
        if value["response"] is not None:
            model_shape(value["response"], ScanResponse)


def outcomes(values):
    return tuple(
        http_replay.HttpOutcome(
            **(
                value
                | {
                    "response": ScanResponse.model_validate(value["response"])
                    if value["response"] is not None
                    else None
                }
            )
        )
        for value in values
    )


def validate_plan(plan):
    require(type(plan) is ShiftPlan)
    ShiftPlan(plan.manifest_sha256, plan.run_index, plan.requests, plan.warmup_count)
    require(type(plan.warmup_count) is int and plan.warmup_count == 1000)
    require(len(plan.requests) >= 1000)


def run_shape(value, plan):
    keys(value, RUN_FIELDS)
    require(type(value["schema_version"]) is int and value["schema_version"] == 1)
    require(type(value["concurrency"]) is int and value["concurrency"] == 1)
    require(value["workload"] == "shift_period")
    require(dump(value["plan"]) == dump(asdict(plan)))
    for name in ("warmup", "measured"):
        outcomes_shape(value[name])
    for name in ("initial", "after_warmup", "after_measured"):
        model_shape(value[name], DrainResponse)
    model_shape(value["trace"], ShiftStateResponse)
    for name in (
        "measured_elapsed_ms",
        "measured_drain_ms",
        "measured_timeout_drain_ms",
    ):
        number(value[name])


def run_wire(run):
    require(type(run) is ShiftRun)
    validate_plan(run.plan)
    return asdict(run) | {
        "schema_version": 1,
        "workload": "shift_period",
        "concurrency": 1,
    }
