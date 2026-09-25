"""Closed HTTP run shapes; no source, process, or filesystem authority."""

import math

from pydantic import TypeAdapter

from .http_replay import HttpOutcome, HttpRun, _metadata
from .http_schema import HTTP_WORKLOADS, DrainResponse, RequestID, ScanResponse

_RECORD_ID = TypeAdapter(RequestID)

RUN_FIELDS = {
    "manifest_sha256",
    "prevalence_basis_points",
    "concurrency",
    "run_index",
    "warmup",
    "measured",
    "initial",
    "after_warmup",
    "after_measured",
    "workload",
    "measured_elapsed_ms",
    "measured_drain_ms",
}
OUTCOME_FIELDS = {
    "record_id",
    "request_id",
    "elapsed_ms",
    "status_code",
    "error",
    "response",
}
RESPONSE_FIELDS = {
    "request_id",
    "admission_sequence",
    "action",
    "probability",
    "stage2_invoked",
}
DRAIN_FIELDS = {
    "admitted_requests",
    "completed_requests",
    "failed_requests",
    "transformer_forward_attempts",
    "successful_transformer_scores",
}


def require(condition):
    if not condition:
        raise ValueError("invalid_http_run_record")


def shape(value, fields):
    require(type(value) is dict and set(value) == fields)


def number(value, *, positive=False):
    require(type(value) in (int, float) and math.isfinite(value))
    require(value > 0 if positive else value >= 0)


def metadata(
    manifest_sha256, prevalence_basis_points, concurrency, run_index, workload
):
    require(type(manifest_sha256) is str)
    require(type(workload) is str and workload in HTTP_WORKLOADS)
    _metadata(manifest_sha256, prevalence_basis_points, concurrency, run_index)
    require(workload != "transformer_only" or prevalence_basis_points == 100)


def response(value):
    if value is None:
        return None
    shape(value, RESPONSE_FIELDS)
    require(type(value["request_id"]) is str and type(value["action"]) is str)
    require(type(value["admission_sequence"]) is int)
    require(type(value["stage2_invoked"]) is bool)
    require(type(value["probability"]) is float)
    number(value["probability"])
    return ScanResponse.model_validate(value)


def drain(value):
    shape(value, DRAIN_FIELDS)
    require(all(type(count) is int and count >= 0 for count in value.values()))
    return DrainResponse.model_validate(value)


def outcome(value):
    shape(value, OUTCOME_FIELDS)
    require(type(value["record_id"]) is str and type(value["request_id"]) is str)
    _RECORD_ID.validate_python(value["record_id"])
    require(value["status_code"] is None or type(value["status_code"]) is int)
    require(value["error"] is None or type(value["error"]) is str)
    number(value["elapsed_ms"])
    return HttpOutcome(**(value | {"response": response(value["response"])}))


def restore(value):
    shape(value, RUN_FIELDS)
    metadata(
        *(
            value[name]
            for name in (
                "manifest_sha256",
                "prevalence_basis_points",
                "concurrency",
                "run_index",
                "workload",
            )
        )
    )
    phases = {}
    for phase, size in (("warmup", 1000), ("measured", 10000)):
        require(type(value[phase]) is list and len(value[phase]) == size)
        phases[phase] = tuple(outcome(row) for row in value[phase])
    number(value["measured_elapsed_ms"], positive=True)
    number(value["measured_drain_ms"], positive=True)
    counters = {
        name: drain(value[name])
        for name in ("initial", "after_warmup", "after_measured")
    }
    return HttpRun(**(value | phases | counters))


def model_payload(value, model):
    require(type(value) is model)
    return value.model_dump(warnings=False)


def outcome_payload(value):
    require(type(value) is HttpOutcome)
    result = {name: getattr(value, name) for name in OUTCOME_FIELDS}
    if value.response is not None:
        result["response"] = model_payload(value.response, ScanResponse)
    return result


def payload(run):
    require(type(run) is HttpRun)
    result = {name: getattr(run, name) for name in RUN_FIELDS}
    for phase, size in (("warmup", 1000), ("measured", 10000)):
        rows = getattr(run, phase)
        require(type(rows) is tuple and len(rows) == size)
        result[phase] = [outcome_payload(row) for row in rows]
    for name in ("initial", "after_warmup", "after_measured"):
        result[name] = model_payload(getattr(run, name), DrainResponse)
    return result
