"""Real-socket tests use invented URLs and an instrumented synthetic monitor."""

import asyncio
import json
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
    progress = json.loads(caught.value.progress)
    assert progress["stage"] == "measured_occurrence_drain"
    assert progress["measured_started"] == [True, True, False]
    assert progress["measured"][1]["error"] == "http_status"
    assert progress["measured"][2] is None
    assert progress["occurrence_drains"][0]["phase"] == "measured"
    assert progress["occurrence_drains"][0]["position"] == 1
    assert progress["occurrence_drains"][0]["counts"]["failed_requests"] == 1
    assert progress["occurrence_drains"][0]["elapsed_ms"] > 0


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
    "failed_control,phase,count",
    [(1, "preflight", 0), (2, "warmup", 1), (3, "measured", 3)],
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
    assert type(caught.value.progress) is bytes
    progress = json.loads(caught.value.progress)
    assert progress["manifest_sha256"] == plan.manifest_sha256
    assert progress["run_index"] == plan.run_index
    assert progress["workload"] == "shift_period"
    assert progress["concurrency"] == 1
    assert progress["warmup_count"] == 1
    assert progress["measured_count"] == 3
    assert "prevalence_basis_points" not in progress
    assert progress["phase"] == phase
    assert (
        progress["stage"]
        == {
            1: "initial_control",
            2: "warmup_reset",
            3: "trace_control",
        }[failed_control]
    )
    assert progress["measured_started"] == [failed_control == 3] * 3
    assert progress["trace"] is None


@pytest.mark.parametrize("failed_drain", [1, 2])
@pytest.mark.parametrize("timeout", [False, True])
def test_phase_drain_failures_retain_outcomes_and_completed_times(
    replay, monkeypatch, failed_drain, timeout
):
    original = http_replay._drain
    calls = 0

    async def fail(client, request_ids):
        nonlocal calls
        calls += 1
        if calls == failed_drain:
            if timeout:
                try:
                    await asyncio.wait_for(asyncio.Event().wait(), timeout=0.001)
                except asyncio.TimeoutError as exc:
                    raise http_replay.ReplayError("invented drain timeout") from exc
            raise http_replay.ReplayError("invented drain failure")
        return await original(client, request_ids)

    monkeypatch.setattr(http_replay, "_drain", fail)
    plan = make_plan(shift_schema)
    app = shift_service.create_shift_app(SyntheticMonitor, plan)
    with loopback_server(app) as url:
        with pytest.raises(replay.ShiftReplayError) as caught:
            asyncio.run(replay.replay_shift_run(url, plan))
    progress = json.loads(caught.value.progress)
    assert progress["stage"] == (
        "warmup_drain" if failed_drain == 1 else "measured_drain"
    )
    assert progress["warmup_started"] == [True]
    assert progress["warmup"][0]["error"] is None
    assert progress["measured_started"] == [failed_drain == 2] * 3
    assert (progress["measured_elapsed_ms"] is None) is (failed_drain == 1)
    assert progress["measured_drain_ms"] is None
    assert progress["after_measured"] is None


@pytest.mark.parametrize("failure", ["validation", "client_cleanup"])
def test_final_failure_retains_live_trace_and_both_phases(replay, monkeypatch, failure):
    if failure == "validation":

        def fail(run):
            raise replay.ShiftReplayError("invented final validation failure")

        monkeypatch.setattr(replay, "validate_shift_run", fail)
    else:
        original = replay.httpx.AsyncClient.__aexit__

        async def fail(client, *args):
            await original(client, *args)
            raise RuntimeError("invented client cleanup failure")

        monkeypatch.setattr(replay.httpx.AsyncClient, "__aexit__", fail)
    plan = make_plan(shift_schema)
    app = shift_service.create_shift_app(SyntheticMonitor, plan)
    with loopback_server(app) as url:
        with pytest.raises(replay.ShiftReplayError) as caught:
            asyncio.run(replay.replay_shift_run(url, plan))
    assert caught.value.phase == "measured"
    assert len(caught.value.outcomes) == 3
    assert len(caught.value.warmup_outcomes) == 1
    progress = json.loads(caught.value.progress)
    assert progress["stage"] == failure
    assert progress["trace"]["complete"] is True
    assert len(progress["trace"]["rows"]) == 3
    assert progress["reset_state"]["phase"] == "measured"
    assert progress["initial_state"]["phase"] == "warmup"
    assert progress["after_measured"]["completed_requests"] == 4
    assert progress["measured_elapsed_ms"] > 0
    assert progress["measured_drain_ms"] > 0
    assert progress["measured_timeout_drain_ms"] == 0


@pytest.mark.parametrize("cleanup_failure", [False, True])
def test_cancellation_retains_completed_prefix_and_stops_serial_scan(
    replay, monkeypatch, cleanup_failure
):
    original = http_replay._scan
    entered, stopped = None, None

    async def scan(client, row, request_id):
        if request_id.endswith(".measured.1"):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()
        return await original(client, row, request_id)

    monkeypatch.setattr(http_replay, "_scan", scan)
    if cleanup_failure:
        cleanup = replay.httpx.AsyncClient.__aexit__

        async def fail(client, *args):
            await cleanup(client, *args)
            raise RuntimeError("invented cleanup failure during cancellation")

        monkeypatch.setattr(replay.httpx.AsyncClient, "__aexit__", fail)
    plan = make_plan(shift_schema)
    app = shift_service.create_shift_app(SyntheticMonitor, plan)

    async def cancel(url):
        nonlocal entered, stopped
        entered, stopped = asyncio.Event(), asyncio.Event()
        task = asyncio.create_task(replay.replay_shift_run(url, plan))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError) as caught:
            await task
        assert task.cancelled() and stopped.is_set()
        progress = json.loads(http_replay.replay_progress(caught.value))
        assert progress["stage"] == progress["phase"] == "measured"
        assert progress["warmup_started"] == [True]
        assert progress["measured_started"] == [True, True, False]
        assert progress["measured"][0]["error"] is None
        assert progress["measured"][1:] == [None, None]
        assert progress["measured_elapsed_ms"] is None

    with loopback_server(app) as url:
        asyncio.run(asyncio.wait_for(cancel(url), timeout=5))
    assert app.state.owner.admitted_requests == 2


@pytest.mark.parametrize("failed_checkpoint", ["warmup.json", "measured.json"])
def test_failed_checkpoint_does_not_retry_or_submit_more_rows(
    replay, failed_checkpoint
):
    checkpoints = []

    def retain(name, content):
        assert type(content) is bytes
        checkpoints.append((name, content))
        if name == failed_checkpoint:
            raise OSError("invented checkpoint failure")

    plan = make_plan(shift_schema)
    app = shift_service.create_shift_app(SyntheticMonitor, plan)
    with loopback_server(app) as url:
        with pytest.raises(replay.ShiftReplayError) as caught:
            asyncio.run(replay.replay_shift_run(url, plan, retain=retain))
    names = ["warmup.json"]
    if failed_checkpoint == "measured.json":
        names.append("measured.json")
    assert [name for name, _ in checkpoints] == names
    assert app.state.owner.admitted_requests == (1 if len(names) == 1 else 4)
    progress = json.loads(caught.value.progress)
    assert progress["stage"] == failed_checkpoint.replace(".json", "_checkpoint")
    assert progress["measured_started"] == [len(names) == 2] * 3


def test_checkpoints_do_not_include_callback_time_or_mutable_response_aliases(
    replay, monkeypatch
):
    ticks, checkpoints = 0, []

    def clock():
        nonlocal ticks
        ticks += 1_000_000
        return ticks

    def retain(name, content):
        nonlocal ticks
        checkpoints.append((name, content))
        ticks += 100_000_000_000

    monkeypatch.setattr(replay.time, "perf_counter_ns", clock)
    plan = make_plan(shift_schema)
    app = shift_service.create_shift_app(SyntheticMonitor, plan)
    with loopback_server(app) as url:
        run = asyncio.run(replay.replay_shift_run(url, plan, retain=retain))
    assert [name for name, _ in checkpoints] == ["warmup.json", "measured.json"]
    assert run.measured_elapsed_ms < 100
    assert run.measured_drain_ms < 100
    assert all(row.elapsed_ms < 100 for row in run.measured)
    run.measured[0].response.probability = 0.99
    run.after_warmup.completed_requests = 999
    snapshot = json.loads(checkpoints[1][1])
    assert snapshot["measured"][0]["response"]["probability"] != 0.99
    assert snapshot["after_warmup"]["completed_requests"] == 1
    assert snapshot["after_measured"] is None
    assert json.loads(checkpoints[0][1])["measured"] == [None] * 3


def test_cancellation_first_received_during_cleanup_remains_cancelled(
    replay, monkeypatch
):
    original = replay.httpx.AsyncClient.__aexit__
    entered = None

    async def cleanup(client, *args):
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError as exc:
            await original(client, *args)
            raise RuntimeError("invented cleanup failure after cancellation") from exc

    monkeypatch.setattr(replay.httpx.AsyncClient, "__aexit__", cleanup)
    plan = make_plan(shift_schema)
    app = shift_service.create_shift_app(SyntheticMonitor, plan)

    async def cancel(url):
        nonlocal entered
        entered = asyncio.Event()
        task = asyncio.create_task(replay.replay_shift_run(url, plan))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError) as caught:
            await task
        assert task.cancelled()
        progress = json.loads(http_replay.replay_progress(caught.value))
        assert progress["stage"] == "client_cleanup"
        assert progress["trace"]["complete"] is True
        assert all(row is not None for row in progress["measured"])

    with loopback_server(app) as url:
        asyncio.run(asyncio.wait_for(cancel(url), timeout=5))


@pytest.mark.parametrize(
    "control_number,stage,state_field,completed",
    [
        (1, "initial_validation", "initial_state", 0),
        (2, "warmup_reset_validation", "reset_state", 1),
        (3, "validation", "trace", 4),
    ],
)
def test_rejected_control_state_is_retained_before_validation(
    replay, monkeypatch, control_number, stage, state_field, completed
):
    original = replay._control
    calls = 0

    async def invalid(client, path, payload=None):
        nonlocal calls
        calls += 1
        state = await original(client, path, payload)
        if calls == control_number:
            return state.model_copy(update={"broken": True})
        return state

    monkeypatch.setattr(replay, "_control", invalid)
    plan = make_plan(shift_schema)
    app = shift_service.create_shift_app(SyntheticMonitor, plan)
    with loopback_server(app) as url:
        with pytest.raises(replay.ShiftReplayError) as caught:
            asyncio.run(replay.replay_shift_run(url, plan))
    progress = json.loads(caught.value.progress)
    assert progress["stage"] == stage
    assert progress[state_field]["broken"] is True
    assert progress[state_field]["counts"]["completed_requests"] == completed
    assert app.state.owner.admitted_requests == completed
    assert b"https://" not in caught.value.progress


@pytest.mark.parametrize(
    "control_number,stage,completed",
    [(1, "initial_control", 0), (2, "warmup_reset", 1), (3, "trace_control", 4)],
)
def test_cancelled_control_joins_before_retaining_progress(
    replay, monkeypatch, control_number, stage, completed
):
    original = replay._control
    entered, stopped = None, None
    calls = 0

    async def block(client, path, payload=None):
        nonlocal calls
        calls += 1
        if calls == control_number:
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()
        return await original(client, path, payload)

    monkeypatch.setattr(replay, "_control", block)
    plan = make_plan(shift_schema)
    app = shift_service.create_shift_app(SyntheticMonitor, plan)

    async def cancel(url):
        nonlocal entered, stopped
        entered, stopped = asyncio.Event(), asyncio.Event()
        task = asyncio.create_task(replay.replay_shift_run(url, plan))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError) as caught:
            await task
        assert task.cancelled() and stopped.is_set()
        progress = json.loads(http_replay.replay_progress(caught.value))
        assert progress["stage"] == stage
        assert progress["warmup_started"] == [completed > 0]
        assert progress["measured_started"] == [completed == 4] * 3

    with loopback_server(app) as url:
        asyncio.run(asyncio.wait_for(cancel(url), timeout=5))
    assert app.state.owner.admitted_requests == completed


def test_invalid_retention_callback_fails_before_starting_client(replay, monkeypatch):
    def forbid_client(*args, **kwargs):
        raise AssertionError("client must not start")

    monkeypatch.setattr(replay.httpx, "AsyncClient", forbid_client)
    with pytest.raises(replay.ShiftReplayError, match="callback must be callable"):
        asyncio.run(
            replay.replay_shift_run(
                "http://127.0.0.1:12345", make_plan(shift_schema), retain=False
            )
        )


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
