"""Full invented schedules with shared read-only outcomes and real validators."""

import importlib
import importlib.util
from dataclasses import replace
from functools import cache

import pytest
from shift_run_codec_fixtures import shift_case

from automated_phishing_detection.http_replay import HttpOutcome, HttpRun
from automated_phishing_detection.http_schema import DrainResponse, ScanResponse
from automated_phishing_detection.operational_schedule import planned_cells


def api():
    name = "automated_phishing_detection.operational_summary"
    assert importlib.util.find_spec(name), "missing complete operational summary"
    return importlib.import_module(name)


def counters(admitted, forwards):
    return DrainResponse(
        admitted_requests=admitted,
        completed_requests=admitted,
        failed_requests=0,
        transformer_forward_attempts=forwards,
        successful_transformer_scores=forwards,
    )


def _outcome(manifest, concurrency, repeat, phase, position):
    identity = f"{manifest}.{concurrency}.{repeat}.{phase}.{position}"
    response = None
    if position == 0 and (phase == "warmup" or repeat != 5):
        response = ScanResponse(
            request_id=identity,
            admission_sequence=1 if phase == "warmup" else 801,
            action="allow",
            probability=0.1,
            stage2_invoked=True,
        )
    latency = (
        10000.0 if phase == "measured" and repeat == 5 and position < 600 else 2100.0
    )
    if response is not None:
        latency = 1.0
    return HttpOutcome(
        f"row-{position}",
        identity,
        latency,
        200 if response else None,
        None if response else "timeout",
        response,
    )


@cache
def _phase(manifest, concurrency, repeat, phase):
    return tuple(
        _outcome(manifest, concurrency, repeat, phase, position)
        for position in range(1000 if phase == "warmup" else 10000)
    )


def http_run(cell):
    manifest = {100: "a", 10: "b", 500: "c"}[cell.prevalence_basis_points] * 64
    phases = tuple(
        _phase(manifest, cell.concurrency, cell.run_index, name)
        for name in ("warmup", "measured")
    )
    forwards = 800 if cell.workload == "transformer_only" else 400
    return HttpRun(
        manifest,
        cell.prevalence_basis_points,
        cell.concurrency,
        cell.run_index,
        *phases,
        counters(0, 0),
        counters(800, forwards),
        counters(8800, 11 * forwards),
        cell.workload,
        100000000.0,
        101000000.0,
    )


@pytest.fixture(scope="module")
def matrix():
    runs = tuple(
        http_run(cell)
        if cell.workload != "shift_period"
        else shift_case(error="timeout", measured_count=1001, run_index=cell.run_index)[
            0
        ]
        for cell in planned_cells()
    )
    yield runs
    _phase.cache_clear()


@pytest.fixture(scope="module")
def report(matrix):
    return api().summarize_operational_runs(matrix)


def rebind_outcome(row, old, new):
    identity = row.request_id.replace(old, new)
    response = row.response
    if response is not None:
        response = response.model_copy(update={"request_id": identity})
    return replace(row, request_id=identity, response=response)


def substitute_http(run, change):
    if change == "hash":
        phases = {
            phase: tuple(
                rebind_outcome(row, run.manifest_sha256, "d" * 64)
                for row in getattr(run, phase)
            )
            for phase in ("warmup", "measured")
        }
        return replace(run, **phases, manifest_sha256="d" * 64)
    phases = {}
    for phase in ("warmup", "measured"):
        rows = getattr(run, phase)
        phases[phase] = (
            replace(rows[0], record_id=rows[1].record_id),
            replace(rows[1], record_id=rows[0].record_id),
            *rows[2:],
        )
    return replace(run, **phases)


def substitute_shift_hash(run):
    old, new = run.manifest_sha256, "d" * 64
    phases = {
        phase: tuple(rebind_outcome(row, old, new) for row in getattr(run, phase))
        for phase in ("warmup", "measured")
    }
    trace = run.trace.model_copy(
        update={
            "manifest_sha256": new,
            "rows": [
                row.model_copy(update={"request_id": row.request_id.replace(old, new)})
                for row in run.trace.rows
            ],
        }
    )
    return replace(
        run, **phases, plan=replace(run.plan, manifest_sha256=new), trace=trace
    )
