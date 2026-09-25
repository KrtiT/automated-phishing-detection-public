"""Capture real phase-stage callbacks with invented full operational outcomes."""

import asyncio
import importlib
import importlib.util

from http_run_codec_fixtures import complete_run, requests

from automated_phishing_detection import http_replay


def api():
    name = "automated_phishing_detection.http_run_checkpoints"
    assert importlib.util.find_spec(name), "missing HTTP checkpoint linkage"
    return importlib.import_module(name)


def _phase_source(run):
    async def phase(client, rows, digest, concurrency, index, name, outcomes, started):
        expected = getattr(run, name)
        assert len(rows) == len(expected)
        assert tuple(row.record_id for row in rows) == tuple(
            row.record_id for row in expected
        )
        outcomes[:] = expected
        started[:] = [True] * len(expected)
        return list(expected)

    return phase


def _drain_source(run):
    values = iter((run.initial, run.after_warmup, run.after_measured))

    async def drain(client, request_ids):
        tuple(request_ids)
        return next(values)

    return drain


def checkpoint_case(monkeypatch, workload="fixed_cascade"):
    run = complete_run(workload)
    progress = http_replay._ReplayProgress(
        run.manifest_sha256,
        run.prevalence_basis_points,
        run.concurrency,
        run.run_index,
        run.workload,
        [None] * 1000,
        [None] * 10000,
        [False] * 1000,
        [False] * 10000,
    )
    retained = {}
    ticks = iter((0, 10_000_000_000, 10_000_000_000, 20_001_000_000))
    with monkeypatch.context() as patch:
        patch.setattr(http_replay, "_phase", _phase_source(run))
        patch.setattr(http_replay, "_drain", _drain_source(run))
        patch.setattr(http_replay.time, "perf_counter_ns", lambda: next(ticks))
        asyncio.run(
            http_replay._replay_phases(
                None, requests(), 1000, progress, retained.__setitem__
            )
        )
    return run, retained["warmup.json"], retained["measured.json"]
