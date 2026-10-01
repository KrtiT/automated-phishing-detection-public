"""Full synthetic suffix execution composes unchanged science and held publication."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from study_series_reduction_fixtures import matrix, reduce
from study_series_runner_fixtures import candidates, manifests, run, series_case, setup
from study_series_runner_process_fixtures import install
from test_study_series_runner import api

__all__ = ["candidates", "manifests", "series_case", "matrix"]


def complete_case(tmp_path, series_case, matrix, monkeypatch):
    source = SimpleNamespace(**vars(series_case))
    source.origin_bytes = matrix.arguments["selected_metadata_bytes"]
    source.profile = json.loads(matrix.arguments["profile_bytes"])
    source.internal = matrix.arguments["internal_snapshot"]
    source.external = matrix.arguments["external_snapshot"]
    case = setup(tmp_path, source, monkeypatch)
    case.history.historical_prefix = matrix.historical
    install(case, monkeypatch)
    return case


def test_all53_real_cells_accounting_and_original125_reducer_publish_once(
    tmp_path, series_case, matrix, monkeypatch
):
    case = complete_case(tmp_path, series_case, matrix, monkeypatch)
    result = run(case)
    assert [entry for entry in case.events if isinstance(entry, tuple)] == [
        ("cell", ordinal) for ordinal in range(73, 126)
    ]
    assert len(result.completed) == 53
    assert result.reduced == reduce(matrix)
    accounting = json.loads(result.segment_accounting_bytes)
    assert accounting["status"] == "complete"
    assert all(cell["status"] == "accepted" for cell in accounting["ledger"]["cells"])
    assert len(accounting["ledger"]["admissions"]) == 106
    for role in ("series", "segment"):
        path = Path(case.source.profile["paths"][f"{role}_public_summary"])
        public = json.loads(path.read_bytes())
        assert public["status"] == f"{role}_evidence_published"
        assert public["sessions"]["single_session"] is False
    assert case.events[-3:] == ["history_closed", "public_check", "supervisor_check"]


def test_real_stopped_second_cell_preserves_first_acceptance_and_unattempted_suffix(
    tmp_path, series_case, monkeypatch
):
    case = setup(tmp_path, series_case, monkeypatch)
    install(case, monkeypatch, stop_ordinal=74)
    with pytest.raises(ValueError) as caught:
        run(case)
    failure = api().series_run_failure(caught.value)
    assert len(failure.completed) == 1
    accounting = json.loads(failure.segment_accounting_bytes)
    cells = accounting["ledger"]["cells"]
    assert [cell["status"] for cell in cells[:2]] == ["accepted", "stopped"]
    assert all(cell["status"] == "unattempted" for cell in cells[2:])
    assert len(accounting["ledger"]["admissions"]) == 2
    assert not Path(case.source.profile["paths"]["series_public_summary"]).exists()
