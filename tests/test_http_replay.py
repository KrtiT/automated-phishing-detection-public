"""Exercise the client across real loopback sockets, not an ASGI transport."""

import asyncio
import socket
import threading
import time
from contextlib import contextmanager
from dataclasses import replace

import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse

from automated_phishing_detection import http_replay
from automated_phishing_detection.http_replay import (
    HttpOutcome,
    HttpRun,
    ReplayError,
    ReplayRequest,
    primary_http_summary,
    reference_invocations,
    replay_run,
)
from automated_phishing_detection.http_schema import DrainResponse, ScanResponse

SHA = "a" * 64


@contextmanager
def loopback_server(app):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    config = uvicorn.Config(
        app,
        loop="asyncio",
        http="h11",
        ws="none",
        lifespan="on",
        access_log=False,
        log_level="critical",
        proxy_headers=False,
        timeout_graceful_shutdown=5,
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]})
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not server.started:
            if not thread.is_alive() or time.monotonic() >= deadline:
                raise RuntimeError("loopback test server did not start")
            time.sleep(0.005)
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(10)
        sock.close()
        assert not thread.is_alive(), "loopback test server did not stop"


def fixture_app(mode="valid"):
    app = FastAPI()
    state = {"admitted": 0, "completed": 0, "sealed": set(), "calls": []}

    @app.post("/v1/drain")
    async def drain(request: Request):
        payload = await request.json()
        state["sealed"].update(payload["request_ids"])
        return DrainResponse(
            admitted_requests=state["admitted"],
            completed_requests=state["completed"],
            failed_requests=0,
            transformer_forward_attempts=0,
            successful_transformer_scores=0,
        )

    @app.post("/v1/scan")
    async def scan(request: Request):
        row = await request.json()
        state["calls"].append(row)
        if row["request_id"] in state["sealed"]:
            return Response(status_code=503)
        state["admitted"] += 1
        sequence = state["admitted"]
        state["completed"] += 1
        body = {
            "request_id": row["request_id"],
            "admission_sequence": sequence,
            "action": "allow",
            "probability": 0.1,
            "stage2_invoked": False,
        }
        # Keep warm-up valid; exercise the measured request's terminal outcome.
        if ".measured." not in row["request_id"]:
            return body
        if mode == "status":
            return Response(status_code=201)
        if mode == "json":
            return Response("{", media_type="application/json")
        if mode == "duplicate_key":
            return Response('{"action":"allow","action":"alert"}')
        if mode == "nested_json":
            return Response("[" * 1100 + "0" + "]" * 1100)
        if mode == "reset":
            return Response("", headers={"Content-Length": "100"})
        if mode == "schema":
            body["stage2_invoked"] = "false"
        if mode == "correlation":
            body["request_id"] = "some-other-request"
        if mode == "timeout":
            await asyncio.sleep(2.1)
        if mode == "trickle":

            async def chunks():
                for _ in range(5):
                    yield b" "
                    await asyncio.sleep(0.55)
                yield b"{}"

            return StreamingResponse(chunks(), media_type="application/json")
        return body

    return app, state


def run_small(url, records=None, **kwargs):
    return asyncio.run(
        replay_run(
            url,
            records or (ReplayRequest("row-1", "https://example.test/a"),),
            manifest_sha256=SHA,
            prevalence_basis_points=100,
            concurrency=kwargs.pop("concurrency", 1),
            run_index=1,
            warmup_count=1,
            **kwargs,
        )
    )


def test_real_socket_run_separates_warmup_and_records_original_order():
    app, state = fixture_app()
    records = tuple(
        ReplayRequest(f"row-{i}", f"https://example.test/{i}") for i in range(9)
    )
    with loopback_server(app) as url:
        result = run_small(url, records, concurrency=8)
    assert [row.record_id for row in result.measured] == [r.record_id for r in records]
    assert len(result.warmup) == 1
    assert result.initial.admitted_requests == 0
    assert result.after_warmup.completed_requests == 1
    assert result.after_measured.completed_requests == 10
    assert all(row.error is None and row.elapsed_ms > 0 for row in result.measured)
    assert len(state["calls"]) == 10
    assert len(state["sealed"]) == 10
    assert len({row["request_id"] for row in state["calls"]}) == 10
    assert "url" not in result.measured[0].__dataclass_fields__
    with pytest.raises(ReplayError, match="10,000"):
        reference_invocations(result)


@pytest.mark.parametrize(
    ("mode", "error"),
    [
        ("status", "http_status"),
        ("json", "invalid_json"),
        ("duplicate_key", "invalid_json"),
        ("nested_json", "invalid_json"),
        ("reset", "transport"),
        ("schema", "invalid_schema"),
        ("correlation", "correlation"),
        ("timeout", "timeout"),
        ("trickle", "timeout"),
    ],
)
def test_terminal_errors_remain_in_measured_denominator(mode, error):
    app, state = fixture_app(mode)
    with loopback_server(app) as url:
        result = run_small(url)
    assert len(result.measured) == 1
    assert result.measured[0].error == error
    assert result.measured[0].response is None
    assert len(state["calls"]) == 2  # No retries, including after timeout.
    if error == "timeout":
        assert result.measured[0].elapsed_ms >= 2000
    assert result.warmup[0].error is None


def test_connection_refusal_is_incomplete_run_not_zero_error_evidence():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    with pytest.raises(ReplayError, match="drain"):
        run_small(f"http://127.0.0.1:{port}")


def test_synchronous_validation_time_is_inside_total_deadline(monkeypatch):
    original = http_replay._json

    def slow_validation(body):
        value = original(body)
        if ".measured." in value.get("request_id", ""):
            time.sleep(2.01)
        return value

    monkeypatch.setattr(http_replay, "_json", slow_validation)
    app, _ = fixture_app()
    with loopback_server(app) as url:
        result = run_small(url)
    assert result.measured[0].error == "timeout"
    assert result.measured[0].elapsed_ms > 2000
    assert result.measured[0].response is None


def test_cancellation_does_not_return_partial_run():
    app, state = fixture_app("timeout")

    async def cancel(url):
        task = asyncio.create_task(
            replay_run(
                url,
                (ReplayRequest("row-1", "https://example.test/a"),),
                manifest_sha256=SHA,
                prevalence_basis_points=100,
                concurrency=1,
                run_index=1,
                warmup_count=1,
            )
        )
        while len(state["calls"]) < 2:
            await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    with loopback_server(app) as url:
        asyncio.run(asyncio.wait_for(cancel(url), timeout=5))


def test_warmup_failures_do_not_cause_retry_or_skip_measurement():
    app, state = fixture_app()

    @app.middleware("http")
    async def failed_warmup(request, call_next):
        if request.url.path == "/v1/scan" and len(state["calls"]) == 0:
            row = await request.json()
            state["calls"].append(row)
            return Response(status_code=503)
        return await call_next(request)

    with loopback_server(app) as url:
        result = run_small(url)
    assert result.warmup[0].error == "http_status"
    assert result.after_warmup.admitted_requests == 0
    assert result.measured[0].error is None
    assert len(state["calls"]) == 2


@pytest.mark.parametrize(
    "url",
    [
        "https://127.0.0.1:80",
        "http://example.com",
        "http://u:p@127.0.0.1",
        "http://127.0.0.1/x",
        "http://127.0.0.1/?x=1",
    ],
)
def test_replay_is_loopback_only(url):
    with pytest.raises(ReplayError, match="loopback"):
        run_small(url)


def synthetic_run(run_index, *, concurrency=64, slow=0, errors=0):
    def rows(phase, count):
        result = []
        for i in range(count):
            request_id = f"{SHA}.{concurrency}.{run_index}.{phase}.{i}"
            error = "http_status" if phase == "measured" and i < errors else None
            latency = 400.0 if phase == "warmup" or i < slow else 100.0
            result.append(
                HttpOutcome(
                    record_id=f"row-{i}",
                    request_id=request_id,
                    elapsed_ms=latency,
                    status_code=500 if error else 200,
                    error=error,
                    response=None
                    if error
                    else ScanResponse(
                        request_id=request_id,
                        admission_sequence=i + (1 if phase == "warmup" else 1001),
                        action="allow",
                        probability=0.1,
                        stage2_invoked=False,
                    ),
                )
            )
        return tuple(result)

    def counts(n):
        return DrainResponse(
            admitted_requests=n,
            completed_requests=n,
            failed_requests=0,
            transformer_forward_attempts=0,
            successful_transformer_scores=0,
        )

    return HttpRun(
        SHA,
        100,
        concurrency,
        run_index,
        rows("warmup", 1000),
        rows("measured", 10000),
        counts(0),
        counts(1000),
        counts(11000),
    )


def test_primary_summary_pools_individual_measured_latencies_and_errors():
    runs = tuple(
        synthetic_run(i, slow=600 if i == 1 else 0, errors=10) for i in range(1, 6)
    )
    summary = primary_http_summary(runs)
    assert summary.run_request_counts == (10000,) * 5
    assert summary.pooled_p95_ms == 100.0  # Mean of run quantiles would be 160.
    assert summary.request_errors == 50
    counts = reference_invocations(synthetic_run(1, concurrency=1))
    assert counts.request_count == counts.completed_requests == 10000
    assert counts.forward_attempts == 0


@pytest.mark.parametrize(
    "change",
    [
        "missing_run",
        "duplicate_run",
        "wrong_concurrency",
        "wrong_prevalence",
        "wrong_hash",
        "missing_request",
        "duplicate_request",
        "wrong_warmup",
        "undrained",
        "counter_regression",
        "unaccounted_failed_forward",
        "unaccounted_successful_forward",
    ],
)
def test_primary_summary_rejects_incomplete_or_mixed_evidence(change):
    runs = [synthetic_run(i) for i in range(1, 6)]
    if change == "missing_run":
        runs.pop()
    elif change == "duplicate_run":
        runs[4] = runs[0]
    elif change == "wrong_concurrency":
        runs[0] = replace(runs[0], concurrency=8)
    elif change == "wrong_prevalence":
        runs[0] = replace(runs[0], prevalence_basis_points=10)
    elif change == "wrong_hash":
        runs[0] = replace(runs[0], manifest_sha256="b" * 64)
    elif change == "missing_request":
        runs[0] = replace(runs[0], measured=runs[0].measured[:-1])
    elif change == "duplicate_request":
        runs[0] = replace(
            runs[0], measured=(runs[0].measured[0],) + runs[0].measured[:-1]
        )
    elif change == "wrong_warmup":
        runs[0] = replace(runs[0], warmup=runs[0].warmup[:-1])
    elif change == "undrained":
        runs[0] = replace(
            runs[0],
            after_measured=runs[0].after_measured.model_copy(
                update={"admitted_requests": 11001}
            ),
        )
    elif change == "counter_regression":
        runs[0] = replace(
            runs[0],
            after_warmup=runs[0].after_warmup.model_copy(
                update={"transformer_forward_attempts": 1}
            ),
        )
    elif change == "unaccounted_failed_forward":
        runs[0] = replace(
            runs[0],
            after_measured=runs[0].after_measured.model_copy(
                update={"transformer_forward_attempts": 1}
            ),
        )
    elif change == "unaccounted_successful_forward":
        runs[0] = replace(
            runs[0],
            after_measured=runs[0].after_measured.model_copy(
                update={
                    "transformer_forward_attempts": 1,
                    "successful_transformer_scores": 1,
                }
            ),
        )
    with pytest.raises(ReplayError):
        primary_http_summary(tuple(runs))


@pytest.mark.parametrize(("concurrency", "run_index"), [(64, 1), (1, 2)])
def test_reference_invocation_run_cannot_be_chosen_after_results(
    concurrency, run_index
):
    with pytest.raises(ReplayError, match="first concurrency-1"):
        reference_invocations(synthetic_run(run_index, concurrency=concurrency))


def test_unadmitted_request_does_not_become_an_invented_scorer_failure():
    run = synthetic_run(1, concurrency=1)
    last = replace(run.measured[-1], status_code=None, error="transport", response=None)
    run = replace(
        run,
        measured=run.measured[:-1] + (last,),
        after_measured=run.after_measured.model_copy(
            update={"admitted_requests": 10999, "completed_requests": 10999}
        ),
    )
    # It is valid client evidence, but incomplete physical-reference evidence.
    http_replay._validate_run(run)
    with pytest.raises(ReplayError, match="not admitted"):
        reference_invocations(run)
