"""Historical cell replay retains exact original contexts and every error."""

from dataclasses import FrozenInstanceError

import pytest
from study_history_cell_fixtures import (
    candidates,
    case,
    history,
    manifests,
    restore,
)

__all__ = ["candidates", "case", "history", "manifests"]


def test_full_sized_original_publication_reconstructs(history):
    result = restore(history)
    assert result.payloads == tuple(history.values.items())
    assert result.inputs == history.working.inputs
    assert result.summary_bytes == history.working.summary_bytes
    assert result.summary == history.working.summary
    assert result.reservation_sha256 == history.working.reservation_sha256
    assert len(result.run.warmup) == 1000
    assert len(result.run.measured) == len(history.working.inputs.requests)
    assert result.run == history.working.run


def test_result_is_immutable_and_run_summary_are_fresh(history):
    result = restore(history)
    with pytest.raises(FrozenInstanceError):
        result.summary_bytes = b"{}"
    with pytest.raises(FrozenInstanceError):
        result.inputs.manifest_bytes = b""
    result.summary.clear()
    result.inputs.primary["thresholds"].clear()
    result.inputs.execution.clear()
    result.run.after_measured.admitted_requests = 0
    assert result.summary == history.working.summary
    assert result.inputs.primary == history.working.inputs.primary
    assert result.inputs.execution == history.working.inputs.execution
    assert result.run == history.working.run
    assert not hasattr(result, "accepted")


def test_terminal_errors_are_preserved_without_selection(history):
    result = restore(history)
    original = history.working.run
    assert tuple(row.error for row in result.run.measured) == tuple(
        row.error for row in original.measured
    )
    if result.inputs.cell.workload != "shift_period":
        assert result.summary["request_errors"] == 6
        assert {row.error for row in result.run.measured if row.error} == {
            "timeout",
            "transport",
            "http_status",
            "invalid_json",
            "invalid_schema",
            "correlation",
        }
