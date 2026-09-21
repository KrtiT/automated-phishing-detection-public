"""The transformer comparator must not silently execute the cascade's stage one."""

import asyncio
import math
import threading
from dataclasses import replace

import pytest
from test_http_replay import loopback_server, synthetic_run
from test_transformer_inference import _build_fixture, _load_fixture

from automated_phishing_detection import fixed_cascade, http_replay
from automated_phishing_detection.http_replay import (
    ReplayError,
    ReplayRequest,
    primary_http_summary,
    reference_invocations,
    replay_run,
)
from automated_phishing_detection.selective_inference import SelectiveCascade
from automated_phishing_detection.selective_service import create_app


def forbid_stage_one(*args, **kwargs):
    pytest.fail("transformer-only workload executed stage one")


def comparator_run(run_index=1, *, concurrency=64):
    run = synthetic_run(run_index, concurrency=concurrency)

    def invoked(rows):
        return tuple(
            replace(
                row, response=row.response.model_copy(update={"stage2_invoked": True})
            )
            for row in rows
        )

    def counts(snapshot):
        return snapshot.model_copy(
            update={
                "transformer_forward_attempts": snapshot.admitted_requests,
                "successful_transformer_scores": snapshot.admitted_requests,
            }
        )

    return replace(
        run,
        workload="transformer_only",
        warmup=invoked(run.warmup),
        measured=invoked(run.measured),
        after_warmup=counts(run.after_warmup),
        after_measured=counts(run.after_measured),
    )


@pytest.mark.parametrize("threshold", [0.0, 1.0])
def test_transformer_only_uses_real_forward_without_stage_one(
    tmp_path, monkeypatch, threshold
):
    loaded = replace(
        _load_fixture(_build_fixture(tmp_path)), transformer_threshold=threshold
    )
    monkeypatch.setattr(
        fixed_cascade, "score_logistic_l1_authoritative", forbid_stage_one
    )
    with SelectiveCascade(loaded, _fixture_cpu=True) as scorer:
        assert hasattr(scorer, "scan_transformer"), "missing transformer-only scorer"
        result = scorer.scan_transformer("https://example.test/only-transformer")
        assert 0 <= result.probability <= 1
        assert result.decision == int(result.probability >= threshold)
        assert scorer.counts.transformer_forward_attempts == 1
        assert scorer.counts.successful_transformer_scores == 1
        assert scorer.counts.completed_requests == 1
        assert scorer.counts.failed_requests == 0


def test_failed_transformer_forward_remains_a_physical_attempt(tmp_path, monkeypatch):
    loaded = _load_fixture(_build_fixture(tmp_path))
    monkeypatch.setattr(
        fixed_cascade, "score_logistic_l1_authoritative", forbid_stage_one
    )

    def fail(*args):
        raise RuntimeError("synthetic failed forward")

    monkeypatch.setattr(loaded._model, "forward", fail)
    with SelectiveCascade(loaded, _fixture_cpu=True) as scorer:
        assert hasattr(scorer, "scan_transformer"), "missing transformer-only scorer"
        with pytest.raises(ValueError, match="failed forward"):
            scorer.scan_transformer("https://example.test/")
        assert scorer.counts.transformer_forward_attempts == 1
        assert scorer.counts.successful_transformer_scores == 0
        assert scorer.counts.completed_requests == 0
        assert scorer.counts.failed_requests == 1


def test_transformer_only_real_socket_workload_keeps_mode_and_owner(
    tmp_path, monkeypatch
):
    events = []
    monkeypatch.setattr(
        fixed_cascade, "score_logistic_l1_authoritative", forbid_stage_one
    )

    def factory():
        events.append(threading.get_ident())
        return SelectiveCascade(
            _load_fixture(_build_fixture(tmp_path)), _fixture_cpu=True
        )

    app = create_app(factory, workload="transformer_only")
    with loopback_server(app) as url:
        result = asyncio.run(
            replay_run(
                url,
                (
                    ReplayRequest("a", "https://example.test/a"),
                    ReplayRequest("b", "https://example.test/b"),
                ),
                manifest_sha256="a" * 64,
                prevalence_basis_points=100,
                concurrency=1,
                run_index=1,
                warmup_count=1,
                workload="transformer_only",
            )
        )
    assert result.workload == app.state.workload == "transformer_only"
    assert len(events) == 1
    assert events[0] != threading.get_ident()
    assert result.after_warmup.transformer_forward_attempts == 1
    assert result.after_measured.transformer_forward_attempts == 3
    assert all(
        row.error is None and row.response.stage2_invoked for row in result.measured
    )
    assert not app.state.owner.is_alive
    assert result.measured_elapsed_ms >= max(row.elapsed_ms for row in result.measured)
    assert result.measured_drain_ms > 0
    summary = http_replay.summarize_run(result)
    assert summary["request_count"] == 2
    assert summary["request_errors"] == 0
    assert summary["transformer_forward_attempts"] == 2
    assert summary["client_attempts_per_second"] == 2000 / result.measured_elapsed_ms
    assert summary["workload"] == "transformer_only"


def test_comparator_runs_cannot_enter_primary_h3_or_invocation_gate():
    runs = tuple(comparator_run(i) for i in range(1, 6))
    with pytest.raises(ReplayError, match="fixed_cascade"):
        primary_http_summary(runs)
    with pytest.raises(ReplayError, match="fixed_cascade"):
        reference_invocations(comparator_run(concurrency=1))


@pytest.mark.parametrize(
    "workload", [None, True, "policy", "fixed", "transformer-only"]
)
def test_unknown_service_mode_fails_before_factory(workload):
    with pytest.raises(ValueError, match="workload"):
        create_app(lambda: pytest.fail("factory called"), workload=workload)


@pytest.mark.parametrize("phase", ["warmup", "measured"])
def test_comparator_rejects_successful_requests_without_transformer_calls(phase):
    run = comparator_run()
    changed_rows = tuple(
        replace(row, response=row.response.model_copy(update={"stage2_invoked": False}))
        for row in getattr(run, phase)
    )
    warmup_forwards = 0 if phase == "warmup" else len(run.warmup)
    total_forwards = warmup_forwards + (len(run.measured) if phase == "warmup" else 0)
    run = replace(
        run,
        **{
            phase: changed_rows,
            "after_warmup": run.after_warmup.model_copy(
                update={
                    "transformer_forward_attempts": warmup_forwards,
                    "successful_transformer_scores": warmup_forwards,
                }
            ),
            "after_measured": run.after_measured.model_copy(
                update={
                    "transformer_forward_attempts": total_forwards,
                    "successful_transformer_scores": total_forwards,
                }
            ),
        },
    )
    with pytest.raises(ReplayError, match="transformer.only"):
        http_replay._validate_run(run)


def test_comparator_cannot_hide_missing_forward_behind_client_timeout():
    run = comparator_run()
    timed_out = replace(
        run.measured[0],
        response=None,
        status_code=None,
        error="timeout",
        elapsed_ms=2000.0,
    )
    run = replace(
        run,
        measured=(timed_out, *run.measured[1:]),
        after_measured=run.after_measured.model_copy(
            update={
                "transformer_forward_attempts": run.after_measured.transformer_forward_attempts
                - 1,
                "successful_transformer_scores": run.after_measured.successful_transformer_scores
                - 1,
            }
        ),
    )
    with pytest.raises(ReplayError, match="transformer.only"):
        http_replay._validate_run(run)


@pytest.mark.parametrize(
    "failure",
    ["before_forward", "in_forward", "after_score", "client_timeout", "unadmitted"],
)
def test_comparator_keeps_legitimate_errors_and_late_completion(failure):
    run = comparator_run()
    completed = run.after_measured.completed_requests - int(failure != "client_timeout")
    failed = int(failure not in ("client_timeout", "unadmitted"))
    attempts = run.after_measured.transformer_forward_attempts - int(
        failure in ("before_forward", "unadmitted")
    )
    scores = run.after_measured.successful_transformer_scores - int(
        failure in ("before_forward", "in_forward", "unadmitted")
    )
    error_row = replace(
        run.measured[-1],
        response=None,
        status_code=None if failure == "client_timeout" else 500,
        error="timeout" if failure == "client_timeout" else "http_status",
        elapsed_ms=2000.0,
    )
    run = replace(
        run,
        measured=(*run.measured[:-1], error_row),
        after_measured=run.after_measured.model_copy(
            update={
                "admitted_requests": completed + failed,
                "completed_requests": completed,
                "failed_requests": failed,
                "transformer_forward_attempts": attempts,
                "successful_transformer_scores": scores,
            }
        ),
    )
    http_replay._validate_run(run)


@pytest.mark.parametrize("concurrency", [1, 8, 64, 128])
def test_phase_interval_cannot_be_shorter_than_closed_loop_work(concurrency):
    run = synthetic_run(1, concurrency=concurrency)
    run = replace(
        run,
        measured_elapsed_ms=max(row.elapsed_ms for row in run.measured),
        measured_drain_ms=1.0,
    )
    with pytest.raises(ReplayError, match="interval"):
        http_replay.summarize_run(run)


@pytest.mark.parametrize("concurrency", [1, 8, 64, 128])
@pytest.mark.parametrize("round_down", [False, True])
def test_phase_interval_accepts_concurrent_overlap_and_rounding_tolerance(
    concurrency, round_down
):
    run = synthetic_run(1, concurrency=concurrency)
    bound = math.fsum(row.elapsed_ms for row in run.measured) / concurrency
    interval = math.nextafter(bound, 0.0) if round_down else bound
    run = replace(run, measured_elapsed_ms=interval, measured_drain_ms=1.0)
    summary = http_replay.summarize_run(run)
    assert summary["client_attempts_per_second"] == len(run.measured) * 1000 / interval


@pytest.mark.parametrize("field", ["measured_elapsed_ms", "measured_drain_ms"])
@pytest.mark.parametrize(
    "value",
    [
        None,
        True,
        0,
        -1,
        "1",
        float("nan"),
        float("inf"),
        pytest.param(10**1000, id="oversized_integer"),
    ],
)
def test_invalid_timing_metadata_is_rejected(field, value):
    run = replace(synthetic_run(1), measured_elapsed_ms=20000.0, measured_drain_ms=1.0)
    with pytest.raises(ReplayError, match="interval"):
        http_replay.summarize_run(replace(run, **{field: value}))
