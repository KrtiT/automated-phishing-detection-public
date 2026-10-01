import json

import pytest
from study_series_cell_fixtures import (
    candidates,
    fresh_cell,
    manifests,
    series_case,
    verify,
)

from automated_phishing_detection import _study_history_cell_science as science

__all__ = ["candidates", "fresh_cell", "manifests", "series_case"]


def test_full_new_context_publication_preserves_original_mathematics(fresh_cell):
    result = verify(fresh_cell)
    assert result.run == fresh_cell.run
    assert result.summary == fresh_cell.summary
    assert result.inputs == fresh_cell.inputs
    assert result.authorizes_execution is False
    current = result.inputs.computational
    assert (
        json.loads(current.accepted_bytes)["kind"]
        == "study-series-operational-inputs-v1"
    )
    origin = json.loads(result.inputs.origin_metadata_bytes)
    assert origin["execution"]["revision"] != current.execution["revision"]
    assert origin["primary"] == current.primary
    assert result.summary["request_errors"] == (2 if current.cell.ordinal == 121 else 6)


def test_full_run_and_checkpoints_replayed_once_with_selected_original_source(
    fresh_cell, monkeypatch
):
    calls = []
    original = science.summary

    def tracked(payloads, inputs, external):
        calls.append((inputs, external))
        return original(payloads, inputs, external)

    monkeypatch.setattr(science, "summary", tracked)
    result = verify(fresh_cell)
    assert calls == [
        (result.inputs.computational, fresh_cell.arguments["external_snapshot"])
    ]


@pytest.mark.parametrize("name", ("attempt/run.json", "attempt/evidence/run.json"))
def test_rehashed_one_sided_payload_substitution_rejects(fresh_cell, name):
    from study_series_input_fixtures import digest
    from test_study_series_cell import api

    values = fresh_cell.values | {name: fresh_cell.values[name] + b" "}
    pins = {member: digest(content) for member, content in values.items()}
    with pytest.raises(api().SeriesCellScienceError):
        verify(
            fresh_cell, payloads=tuple(values.items()), expected_snapshot_sha256=pins
        )
