"""Pure complete-matrix descriptions, not accepted-cell or hypothesis evidence.

Inputs establish caller consistency only. Source, raw HTTP URL dispatch,
process ownership, shift checkpoints, and acceptance remain separate bindings.
"""

import numpy as np

from ._checkpoint_codec import canonical_bytes
from .http_replay import HttpRun, summarize_run
from .http_run_codec import encode_http_run
from .operational_schedule import OperationalCell, planned_cells, validate_cell
from .shift_replay import ShiftRun, summarize_shift_run
from .shift_run_codec import encode_shift_run

_COUNTERS = (
    "admitted_requests",
    "completed_requests",
    "failed_requests",
    "transformer_forward_attempts",
    "successful_transformer_scores",
)


class OperationalSummaryError(ValueError):
    """The inputs cannot form a complete finite operational description."""


def _require(condition):
    if not condition:
        raise OperationalSummaryError("invalid_operational_summary")


def _scheduled(runs):
    _require(type(runs) is tuple and len(runs) == 125)
    for run, cell in zip(runs, planned_cells(), strict=True):
        expected_type = ShiftRun if cell.workload == "shift_period" else HttpRun
        _require(type(run) is expected_type)
        prevalence = run.prevalence_basis_points if expected_type is HttpRun else None
        validate_cell(
            OperationalCell(
                cell.ordinal, run.workload, prevalence, run.concurrency, run.run_index
            )
        )


def _identities(runs):
    manifests = {}
    for run in runs[:120]:
        identity = (run.manifest_sha256, tuple(row.record_id for row in run.measured))
        prevalence = run.prevalence_basis_points
        _require(identity == manifests.setdefault(prevalence, identity))
    expected = (runs[120].manifest_sha256, runs[120].plan.requests)
    for run in runs[121:]:
        _require((run.manifest_sha256, run.plan.requests) == expected)


def _run_summary(run, ordinal):
    if type(run) is HttpRun:
        encode_http_run(run)
        summary = summarize_run(run)
    else:
        encode_shift_run(run)
        summary = summarize_shift_run(run)
    return summary | {"cell_ordinal": ordinal}


def _physical_counts(summaries, total):
    counts = {name: sum(run[name] for run in summaries) for name in _COUNTERS}
    return counts | {
        "physical_invocation_fraction": counts["transformer_forward_attempts"] / total
    }


def _pooled(runs, summaries):
    total = sum(run["request_count"] for run in summaries)
    errors = sum(run["request_errors"] for run in summaries)
    latencies = np.fromiter(
        (row.elapsed_ms for run in runs for row in run.measured),
        dtype=np.float64,
        count=total,
    )
    quantiles = np.quantile(latencies, [0.5, 0.95, 0.99], method="linear")
    return {
        "request_count": total,
        "request_errors": errors,
        "request_error_rate": errors / total,
        "p50_ms": float(quantiles[0]),
        "p95_ms": float(quantiles[1]),
        "p99_ms": float(quantiles[2]),
        **_physical_counts(summaries, total),
    }


def _group(runs, first_ordinal):
    first = runs[0]
    summaries = [
        _run_summary(run, first_ordinal + offset) for offset, run in enumerate(runs)
    ]
    return {
        "workload": first.workload,
        "prevalence_basis_points": first.prevalence_basis_points
        if type(first) is HttpRun
        else None,
        "concurrency": first.concurrency,
        "manifest_sha256": first.manifest_sha256,
        "run_indices": [run.run_index for run in runs],
        "run_request_counts": [run["request_count"] for run in summaries],
        **_pooled(runs, summaries),
        "run_summaries": summaries,
    }


def summarize_operational_runs(runs: tuple[HttpRun | ShiftRun, ...]) -> dict:
    """Pool every scheduled repeat; physical fractions use all client attempts."""
    try:
        _scheduled(runs)
        _identities(runs)
        result = {
            "schema_version": 1,
            "protocol": "operational-descriptive-summary-v1",
            "groups": [
                _group(runs[offset : offset + 5], offset + 1)
                for offset in range(0, 125, 5)
            ],
        }
        canonical_bytes(result)
        return result
    except Exception:
        raise OperationalSummaryError("invalid_operational_summary") from None
