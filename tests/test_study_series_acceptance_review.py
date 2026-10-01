"""Independent invented-observation checks of current-series cell acceptance."""

import json
from hashlib import sha256

import pytest
from study_series_acceptance_fixtures import api, arguments, forbid, payloads, working
from study_series_cell_fixtures import candidates, fresh_cell, manifests, series_case

from automated_phishing_detection import _study_history_cell_science as science
from automated_phishing_detection._checkpoint_codec import canonical_bytes
from automated_phishing_detection._operational_cell_process_records import (
    verify_process_records,
)
from automated_phishing_detection._operational_process_records import (
    ProcessObservation,
    _bytes,
)

__all__ = ["candidates", "fresh_cell", "manifests", "series_case"]


def test_actual_observation_is_forwarded_before_scientific_reconstruction(
    fresh_cell, monkeypatch
):
    module, options, calls = api(), arguments(fresh_cell), []
    original_science = science.summary

    def observed(values, **keywords):
        assert keywords["observation"] is options["observation"]
        assert values["process-pair.json"] == options["observation"].record
        calls.append("observation")
        return verify_process_records(values, **keywords)

    def reconstructed(*values):
        assert calls == ["observation"]
        calls.append("science")
        return original_science(*values)

    monkeypatch.setattr(module, "verify_process_records", observed)
    monkeypatch.setattr(science, "summary", reconstructed)
    result = module.verify_series_working_cell(payloads(fresh_cell), **options)
    assert calls == ["observation", "science"]
    assert result.summary == fresh_cell.summary


@pytest.mark.parametrize(
    "field", ("expected_metadata_sha256", "expected_profile_sha256")
)
def test_independent_pin_failure_precedes_process_and_science(
    fresh_cell, monkeypatch, field
):
    monkeypatch.setattr(api(), "verify_process_records", forbid)
    monkeypatch.setattr(science, "summary", forbid)
    with pytest.raises(api().SeriesCellAcceptanceError, match="^invalid_series_cell$"):
        working(fresh_cell, **{field: "0" * 64})


def alternate_service(values):
    changed = dict(values)
    pair = json.loads(changed["process-pair.json"])
    pair["service"]["pid"] = 987
    changed["service-process.json"] = _bytes(pair["service"])
    changed["service-started.json"] = _bytes({"pid": 987})
    for name, pin in (
        ("service-ready.json", "readiness_sha256"),
        ("service-cleanup.json", "cleanup_sha256"),
    ):
        changed[name] = _bytes(json.loads(changed[name]) | {"pid": 987})
        pair[pin] = sha256(changed[name]).hexdigest()
    changed["service-role.json"] = canonical_bytes(
        json.loads(changed["service-role.json"]) | {"pid": 987}
    )
    changed["process-pair.json"] = _bytes(pair)
    return changed


def test_fully_rejoined_alternate_child_cannot_replace_actual_observation(
    fresh_cell, monkeypatch
):
    changed = alternate_service(payloads(fresh_cell))
    options = arguments(fresh_cell)
    verify_process_records(
        changed,
        inputs=fresh_cell.inputs.computational,
        observation=ProcessObservation(changed["process-pair.json"]),
        reservation=options["attempt"].reservation_sha256,
        service_command=options["service_command"],
        client_command=options["client_command"],
        expected_deadlines=options["expected_deadlines"],
    )
    monkeypatch.setattr(science, "summary", forbid)
    with pytest.raises(api().SeriesCellAcceptanceError, match="^invalid_series_cell$"):
        working(fresh_cell, values=tuple(changed.items()))
