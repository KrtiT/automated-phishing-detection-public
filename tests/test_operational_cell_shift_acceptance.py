"""Shift acceptance reuses exact retained source scores, never numerical owners."""

from dataclasses import replace

import pytest
from operational_cell_acceptance_fixtures import changed_payload, verify
from operational_cell_shift_fixtures import shift_case
from operational_input_fixtures import candidates, case, manifests
from test_operational_cell_identity import api

__all__ = ["candidates", "case", "manifests"]


@pytest.fixture(scope="module")
def working_case(case):
    return shift_case(api(), case)


def test_complete_shift_matches_existing_offline_trace(working_case):
    result = verify(api(), working_case)
    assert result.summary["workload"] == "shift_period"
    assert result.summary["primary_evidence"] is False
    assert result.summary["request_count"] == 1001
    assert len(result.run.trace.rows) == 1001
    result.run.trace.rows[0].monitor_nll = 999
    assert result.run.trace.rows[0].monitor_nll == -100


@pytest.mark.parametrize(
    "field,value", [("negative_log_likelihood", -99.0), ("stage1_probability", 0.21)]
)
def test_offline_live_mismatch_is_not_replaced_by_rescoring(working_case, field, value):
    accepted = working_case.arguments["accepted"]
    original = accepted.external.snapshot
    row = replace(
        original.rows[0], primary=replace(original.rows[0].primary, **{field: value})
    )
    external = replace(
        accepted.external, snapshot=replace(original, rows=(row, *original.rows[1:]))
    )
    with pytest.raises(api().OperationalCellAcceptanceError):
        verify(api(), working_case, accepted=replace(accepted, external=external))


@pytest.mark.parametrize("name", ["warmup.json", "measured.json"])
def test_original_shift_phase_bytes_must_match(working_case, name):
    changed = changed_payload(
        working_case, name, lambda value: value.update({"run_index": 2})
    )
    with pytest.raises(api().OperationalCellAcceptanceError):
        verify(api(), changed)
