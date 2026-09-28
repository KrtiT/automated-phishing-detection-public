"""Full adopted coordination with explicitly constructed scientific observations."""

import json

import adopted_study_fixtures as fixtures
import pytest
from adopted_study_matrix_fixtures import install
from study_run_record_fixtures import capacity, prepared

__all__ = ["prepared"]


def test_full_adopted_root_joins_252_admissions_and_125_cells(
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
    assert all(slot.status == "accepted" for slot in result.cells)
    assert len(result.snapshot.payloads) == 14
    public = json.loads(result.snapshot.payload("public-summary.json"))
    assert public["protocol"] == "adopted-study-root-v1"
    assert public["status"] == "study_evidence_published"
    accounting = json.loads(result.snapshot.payload("attempt/study-accounting.json"))
    ledger = accounting["authorization_ledger"]
    assert len(ledger["admissions"]) == 252 and len(ledger["cell_acceptances"]) == 125
    assert all(entry["accepted"] for entry in ledger["admissions"])


@pytest.mark.parametrize("ordinal", [1, 21, 125])
def test_failed_adopted_cell_preserves_prefix_without_retry(
    tmp_path, prepared, monkeypatch, ordinal
):
    case = fixtures.setup(tmp_path, prepared, monkeypatch, capacity(prepared))
    install(case, monkeypatch, fail_ordinal=ordinal)
    with pytest.raises(case.module.StudyRunError) as caught:
        fixtures.execute(case)
    assert [event for event in case.events if type(event) is int] == list(
        range(1, ordinal + 1)
    )
    assert "reduce" not in case.events and not case.paths.public_summary.exists()
    failure = case.module.adopted_study_failure(caught.value)
    slots = failure.scientific.cells
    assert all(slot.status == "accepted" for slot in slots[: ordinal - 1])
    assert slots[ordinal - 1].status == "stopped"
    assert all(slot.status == "unattempted" for slot in slots[ordinal:])
    ledger = json.loads(failure.authorization_ledger)
    assert len(ledger["cell_acceptances"]) == ordinal - 1
    assert len(ledger["admissions"]) == 2 + 2 * ordinal
    assert all(entry["accepted"] for entry in ledger["admissions"][:-2])
    assert all(not entry["accepted"] for entry in ledger["admissions"][-2:])


def test_adopted_reduction_failure_preserves_every_accepted_cell(
    tmp_path, prepared, monkeypatch
):
    case = fixtures.setup(tmp_path, prepared, monkeypatch, capacity(prepared))
    install(case, monkeypatch, reduction_error=ValueError("invented reduction failure"))
    with pytest.raises(case.module.StudyRunError) as caught:
        fixtures.execute(case)
    assert all(slot.status == "accepted" for slot in caught.value.study_failure.cells)
    assert len(case.returns) == 125 and case.events.count("reduce") == 1
    ledger = json.loads(
        case.module.adopted_study_failure(caught.value).authorization_ledger
    )
    assert len(ledger["admissions"]) == 252 and len(ledger["cell_acceptances"]) == 125
    assert not case.paths.public_summary.exists()
