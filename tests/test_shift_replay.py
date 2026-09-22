"""Real-socket tests use invented URLs and an instrumented synthetic monitor."""

import asyncio
import threading
from dataclasses import replace
from pathlib import Path

import pytest
from test_http_replay import loopback_server
from test_shift_service import SyntheticMonitor, make_plan

from automated_phishing_detection import http_replay, shift_schema, shift_service


@pytest.fixture
def replay():
    assert (
        Path(__file__).resolve().parents[1]
        / "src/automated_phishing_detection/shift_replay.py"
    ).is_file(), "missing serialized shift client"
    from automated_phishing_detection import shift_replay

    return shift_replay


def test_real_socket_run_retains_order_and_excludes_warmup(replay):
    plan = make_plan(shift_schema)
    app = shift_service.create_shift_app(SyntheticMonitor, plan)
    with loopback_server(app) as url:
        run = asyncio.run(replay.replay_shift_run(url, plan))
    assert len(run.measured) == 3 and len(run.warmup) == 1
    assert [r.response.admission_sequence for r in run.measured] == [2, 3, 4]
    summary = replay.summarize_shift_run(run)
    assert summary["workload"] == "shift_period"
    assert summary["request_count"] == 3
    assert summary["completed_requests"] == 3
    assert summary["primary_evidence"] is False
    with pytest.raises(http_replay.ReplayError):
        http_replay.primary_http_summary([run] * 5)


def test_timeout_is_drained_before_next_row_without_retry(replay, monkeypatch):
    blocked, release = threading.Event(), threading.Event()

    class SlowMonitor(SyntheticMonitor):
        def scan_shift(self, raw_url):
            if self.position == 0:
                blocked.set()
                assert release.wait(10), "client did not drain the timed-out row"
                release.clear()
                blocked.clear()
            return super().scan_shift(raw_url)

    plan = make_plan(shift_schema)
    holder, released_ids = [], []
    original_drain = http_replay._drain

    async def release_on_drain(client, request_ids):
        request_ids = tuple(request_ids)
        if blocked.is_set():
            # The real request deadline must expire before scoring can finish.
            assert request_ids == (
                plan.request_id("warmup" if not released_ids else "measured", 0),
            )
            released_ids.extend(request_ids)
            release.set()
        return await original_drain(client, request_ids)

    monkeypatch.setattr(http_replay, "_drain", release_on_drain)

    def factory():
        holder.append(SlowMonitor())
        return holder[-1]

    app = shift_service.create_shift_app(factory, plan)
    with loopback_server(app) as url:
        try:
            run = asyncio.run(replay.replay_shift_run(url, plan))
        finally:
            release.set()
    assert run.warmup[0].error == "timeout"
    assert run.measured[0].error == "timeout"
    assert sum(row.error is not None for row in run.measured) == 1
    assert len(run.trace.rows) == 3
    assert run.after_measured.admitted_requests == 4
    assert len(released_ids) == 2
    assert run.measured_timeout_drain_ms > 0
    assert len([event for event, _ in holder[0].events if event == "scan"]) == 4


def test_failed_row_stops_stream_and_retains_partial_outcomes(replay):
    plan = make_plan(shift_schema)

    class FailingMeasured(SyntheticMonitor):
        def reset_monitor(self):
            super().reset_monitor()
            self.fail_at = 2

    app = shift_service.create_shift_app(FailingMeasured, plan)
    with loopback_server(app) as url:
        with pytest.raises(replay.ShiftReplayError) as caught:
            asyncio.run(replay.replay_shift_run(url, plan))
    assert caught.value.phase == "measured"
    assert len(caught.value.outcomes) == 2
    assert caught.value.outcomes[-1].error == "http_status"
    assert app.state.owner.admitted_requests == 3


def test_wrong_manifest_handshake_fails_before_scan(replay):
    plan = make_plan(shift_schema)
    app = shift_service.create_shift_app(SyntheticMonitor, plan)
    with loopback_server(app) as url:
        with pytest.raises(replay.ShiftReplayError, match="identity"):
            asyncio.run(
                replay.replay_shift_run(url, replace(plan, manifest_sha256="b" * 64))
            )
    assert app.state.owner.admitted_requests == 0


def test_tampered_completed_trace_cannot_be_summarized(replay):
    plan = make_plan(shift_schema)
    app = shift_service.create_shift_app(SyntheticMonitor, plan)
    with loopback_server(app) as url:
        run = asyncio.run(replay.replay_shift_run(url, plan))
    row = run.trace.rows[0].model_copy(update={"position": 2})
    changed = run.trace.model_copy(update={"rows": [row, *run.trace.rows[1:]]})
    with pytest.raises(http_replay.ReplayError, match="trace"):
        replay.summarize_shift_run(replace(run, trace=changed))


@pytest.mark.parametrize(
    "change",
    [
        "early_window",
        "future_drift",
        "fixed_decision",
        "snapshot_bool",
        "drain_bool",
        "drain_occupancy",
    ],
)
def test_inconsistent_evidence_is_rejected(replay, change):
    plan = make_plan(shift_schema)
    app = shift_service.create_shift_app(SyntheticMonitor, plan)
    with loopback_server(app) as url:
        run = asyncio.run(replay.replay_shift_run(url, plan))
    row = run.trace.rows[0]
    if change == "early_window":
        row = row.model_copy(
            update={
                "window": shift_schema.ShiftWindow(
                    start_position=1, end_position=1, score=1.0, alert=True
                )
            }
        )
    elif change == "future_drift":
        row = row.model_copy(
            update={
                "drift_override": True,
                "stage2_invoked": True,
                "transformer_probability": 0.2,
            }
        )
    elif change == "fixed_decision":
        row = row.model_copy(update={"fixed_decision": 1})
    elif change == "snapshot_bool":
        run = replace(
            run, initial=run.initial.model_copy(update={"admitted_requests": False})
        )
    elif change == "drain_bool":
        run = replace(run, measured_timeout_drain_ms=False)
    else:
        run = replace(run, measured_timeout_drain_ms=run.measured_elapsed_ms)
    run = replace(
        run, trace=run.trace.model_copy(update={"rows": [row, *run.trace.rows[1:]]})
    )
    with pytest.raises((http_replay.ReplayError, ValueError)):
        replay.summarize_shift_run(run)


@pytest.mark.parametrize(
    "failed_control,phase,count", [(2, "warmup", 1), (3, "measured", 3)]
)
def test_control_failures_preserve_collected_outcomes(
    replay, monkeypatch, failed_control, phase, count
):
    plan = make_plan(shift_schema)
    original = replay._control
    calls = 0

    async def fail(client, path, payload=None):
        nonlocal calls
        calls += 1
        if calls == failed_control:
            raise replay.ShiftReplayError("synthetic control failure")
        return await original(client, path, payload)

    monkeypatch.setattr(replay, "_control", fail)
    app = shift_service.create_shift_app(SyntheticMonitor, plan)
    with loopback_server(app) as url:
        with pytest.raises(replay.ShiftReplayError) as caught:
            asyncio.run(replay.replay_shift_run(url, plan))
    assert caught.value.phase == phase
    assert len(caught.value.outcomes) == count


def test_offline_trace_requires_exact_saved_scores(replay):
    from automated_phishing_detection.policy_replay import (
        MonitorScore,
        PairedProbabilities,
    )

    plan = make_plan(shift_schema)
    app = shift_service.create_shift_app(SyntheticMonitor, plan)
    with loopback_server(app) as url:
        run = asyncio.run(replay.replay_shift_run(url, plan))
    pairs = tuple(PairedProbabilities(r.record_id, 0.2, 0.7) for r in plan.requests)
    nll = tuple(MonitorScore(r.record_id, 1.0) for r in plan.requests)
    kwargs = dict(
        stage1_threshold=0.5,
        transformer_threshold=0.5,
        half_width=0.1,
        monitor_boundary=1.0,
    )
    replay.verify_offline_trace(run, pairs, nll, **kwargs)
    with pytest.raises(replay.ShiftReplayError, match="differs"):
        replay.verify_offline_trace(
            run,
            pairs,
            (replace(nll[0], negative_log_likelihood=2.0), *nll[1:]),
            **kwargs,
        )


def test_real_core_over_tcp_resets_warmup_alert_and_matches_offline(replay, tmp_path):
    from contextlib import contextmanager

    from test_live_monitor import gmm as gmm_fixture
    from test_transformer_inference import _build_fixture, _load_fixture

    from automated_phishing_detection import gmm_monitor
    from automated_phishing_detection.live_monitor import LiveMonitor
    from automated_phishing_detection.policy_replay import (
        MonitorScore,
        PairedProbabilities,
    )
    from automated_phishing_detection.selective_inference import SelectiveCascade
    from automated_phishing_detection.url_features import extract_url_features

    model = replace(
        _load_fixture(_build_fixture(tmp_path)), stage1_threshold=1.0, half_width=0.0
    )
    gmm = gmm_fixture.__wrapped__()
    raw_url = "https://safe.example/a"
    with SelectiveCascade(model, _fixture_cpu=True) as scorer:
        saved = scorer.score_all(raw_url)
        portable = model.stage1_model.score_urls((raw_url,))[0]
        nll = float(
            gmm_monitor.score_feature_matrix(
                [(*extract_url_features(raw_url), portable)], gmm
            )[0]
        )
    assert saved.band_selected is False and nll > 0
    plan = shift_schema.ShiftPlan(
        "a" * 64,
        1,
        tuple(http_replay.ReplayRequest(f"row-{i}", raw_url) for i in range(257)),
        warmup_count=256,
    )

    @contextmanager
    def factory():
        with SelectiveCascade(model, _fixture_cpu=True) as scorer:
            yield LiveMonitor(
                scorer, stage1_model=model.stage1_model, gmm=gmm, boundary=0.0
            )

    app = shift_service.create_shift_app(factory, plan)
    with loopback_server(app) as url:
        run = asyncio.run(replay.replay_shift_run(url, plan))
    assert not any(row.drift_override for row in run.trace.rows[:256])
    assert run.trace.rows[255].window.alert is True
    assert run.trace.rows[256].drift_override is True
    assert run.after_warmup.transformer_forward_attempts == 0
    assert run.after_measured.transformer_forward_attempts == 1
    replay.verify_offline_trace(
        run,
        tuple(
            PairedProbabilities(
                row.record_id, saved.stage1_probability, saved.transformer_probability
            )
            for row in plan.requests
        ),
        tuple(MonitorScore(row.record_id, nll) for row in plan.requests),
        stage1_threshold=model.stage1_threshold,
        transformer_threshold=model.transformer_threshold,
        half_width=model.half_width,
        monitor_boundary=0.0,
    )
