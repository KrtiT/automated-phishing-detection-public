"""Complete invented runs; no sockets, data readers, or numerical owners."""

import importlib
import importlib.util
import json
from dataclasses import asdict
from functools import lru_cache

from automated_phishing_detection._checkpoint_codec import canonical_bytes
from automated_phishing_detection.http_replay import HttpOutcome, HttpRun, ReplayRequest
from automated_phishing_detection.http_schema import DrainResponse, ScanResponse

MANIFEST = "a" * 64
ERRORS = (
    "timeout",
    "transport",
    "http_status",
    "invalid_json",
    "invalid_schema",
    "correlation",
)


def api():
    name = "automated_phishing_detection.http_run_codec"
    assert importlib.util.find_spec(name), "missing complete HTTP run codec"
    return importlib.import_module(name)


@lru_cache
def requests():
    return tuple(
        ReplayRequest(f"row-{position}", f"HTTPS://例え.test/{position}?a=%2F")
        for position in range(10000)
    )


def outcome(position, phase, workload):
    request_id = f"{MANIFEST}.64.1.{phase}.{position}"
    error = ERRORS[position] if phase == "measured" and position < len(ERRORS) else None
    response = (
        None
        if error
        else ScanResponse(
            request_id=request_id,
            admission_sequence=position + 1 + (1000 if phase == "measured" else 0),
            action="allow",
            probability=0.125,
            stage2_invoked=workload == "transformer_only" or position % 2 == 0,
        )
    )
    return HttpOutcome(
        f"row-{position}",
        request_id,
        2100.0 if error == "timeout" else float(position % 100 + 1),
        None if error == "transport" else 500 if error == "http_status" else 200,
        error,
        response,
    )


def drain(admitted, forwards):
    return DrainResponse(
        admitted_requests=admitted,
        completed_requests=admitted,
        failed_requests=0,
        transformer_forward_attempts=forwards,
        successful_transformer_scores=forwards,
    )


@lru_cache
def complete_run(workload="fixed_cascade"):
    warmup = tuple(outcome(position, "warmup", workload) for position in range(1000))
    measured = tuple(
        outcome(position, "measured", workload) for position in range(10000)
    )
    warmup_forwards = 1000 if workload == "transformer_only" else 500
    measured_forwards = 10000 if workload == "transformer_only" else 4997
    return HttpRun(
        MANIFEST,
        100,
        64,
        1,
        warmup,
        measured,
        drain(0, 0),
        drain(1000, warmup_forwards),
        drain(11000, warmup_forwards + measured_forwards),
        workload,
        10000.0,
        10001.0,
    )


@lru_cache
def wire_bytes(workload="fixed_cascade"):
    run = asdict(complete_run(workload))
    content = json.dumps(run, default=lambda value: value.model_dump())
    return canonical_bytes(
        {"schema_version": 1, "protocol": "http-run-v1", "run": json.loads(content)}
    )


def document():
    return json.loads(wire_bytes())


def arguments(**changes):
    return {
        "expected_manifest_sha256": MANIFEST,
        "expected_requests": requests(),
        "expected_prevalence_basis_points": 100,
        "expected_concurrency": 64,
        "expected_run_index": 1,
        "expected_workload": "fixed_cascade",
    } | changes


def decode(content=None, **changes):
    return api().decode_http_run(
        wire_bytes() if content is None else content, **arguments(**changes)
    )
