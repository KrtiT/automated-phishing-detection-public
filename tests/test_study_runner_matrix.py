"""Fixed125 orchestration with explicitly mocked science and cell boundaries."""

import json

import pytest
import study_runner_fixtures as fixtures
from study_run_record_fixtures import capacity, prepared
from study_runner_matrix_fixtures import install

__all__ = ["prepared"]


def test_all125_cells_run_once_in_order_before_original_reduction(
    tmp_path, prepared, monkeypatch
):
    case = fixtures.setup(tmp_path, prepared, monkeypatch, capacity(prepared))
    install(case, monkeypatch)
    result = fixtures.execute(case)
    assert [event for event in case.events if type(event) is int] == list(range(1, 126))
    assert case.events.count("sources") == case.events.count("reduce") == 1
    assert case.events.index("sources") < case.events.index(1)
    assert (
        case.events.index(125)
        < case.events.index("reduce")
        < case.events.index("release_preparation")
    )
    assert result.sources is case.sources and result.accepted is case.accepted
    assert all(slot.status == "accepted" for slot in result.cells)
    assert len(result.snapshot.payloads) == 14
    assert (
        json.loads(result.snapshot.payload("public-summary.json"))["status"]
        == "study_evidence_published"
    )


@pytest.mark.parametrize("ordinal", [1, 21, 125])
def test_failed_cell_stops_without_retry_or_omitted_suffix(
    tmp_path, prepared, monkeypatch, ordinal
):
    case = fixtures.setup(tmp_path, prepared, monkeypatch, capacity(prepared))
    install(case, monkeypatch, fail_ordinal=ordinal)
    with pytest.raises(case.module.StudyRunError) as caught:
        fixtures.execute(case)
    assert [event for event in case.events if type(event) is int] == list(
        range(1, ordinal + 1)
    )
    assert "reduce" not in case.events
    slots = caught.value.study_failure.cells
    assert all(slot.status == "accepted" for slot in slots[: ordinal - 1])
    assert slots[ordinal - 1].status == "stopped"
    assert all(slot.status == "unattempted" for slot in slots[ordinal:])
    assert not case.paths.public_summary.exists()
    accounting = json.loads((case.paths.attempt / "study-accounting.json").read_bytes())
    assert accounting["status"] == "failed"
    assert len(accounting["cells"]) == 125


def test_reduction_failure_keeps_all125_accepted_cells(tmp_path, prepared, monkeypatch):
    case = fixtures.setup(tmp_path, prepared, monkeypatch, capacity(prepared))
    install(case, monkeypatch, reduction_error=ValueError("original kernel rejected"))
    with pytest.raises(case.module.StudyRunError) as caught:
        fixtures.execute(case)
    assert all(slot.status == "accepted" for slot in caught.value.study_failure.cells)
    assert len(case.returns) == 125 and case.events.count("reduce") == 1
    assert not case.paths.public_summary.exists()
