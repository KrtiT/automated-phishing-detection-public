"""Phase bytes describe their historical stage, not later final drain evidence."""

import json

import pytest
from http_run_checkpoints_fixtures import api, checkpoint_case


@pytest.mark.parametrize("workload", ["fixed_cascade", "transformer_only"])
def test_actual_phase_callbacks_join_complete_run_without_later_state(
    monkeypatch, workload
):
    run, warmup, measured = checkpoint_case(monkeypatch, workload)
    assert api().verify_http_checkpoints(warmup, measured, run=run) is None
    first, second = json.loads(warmup), json.loads(measured)
    assert first["after_warmup"] is None
    assert not any(first["measured_started"])
    assert all(value is None for value in first["measured"])
    assert second["after_warmup"] == run.after_warmup.model_dump()
    assert second["measured_elapsed_ms"] == run.measured_elapsed_ms
    assert second["after_measured"] is second["measured_drain_ms"] is None
    assert sum(row["error"] is not None for row in second["measured"]) == 6
