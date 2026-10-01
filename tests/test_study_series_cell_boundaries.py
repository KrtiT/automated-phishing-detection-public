import json

import pytest
from study_series_cell_fixtures import (
    candidates,
    fresh_cell,
    manifests,
    series_case,
    verify,
)
from study_series_input_fixtures import digest
from test_study_history_external_boundaries import forbid_live
from test_study_series_cell import api

from automated_phishing_detection import execution_receipt as receipt
from automated_phishing_detection._operational_cell_results import (
    VerifiedOperationalCell,
    VerifiedOperationalWorking,
)
from automated_phishing_detection.operational_inputs import AcceptedOperationalInputs

__all__ = ["candidates", "fresh_cell", "manifests", "series_case"]


def forbid(*args, **kwargs):
    pytest.fail("series science fabricated original acceptance")


def test_no_io_new_forwards_or_fabricated_legacy_acceptance(fresh_cell, monkeypatch):
    forbid_live(monkeypatch)
    for record in (
        AcceptedOperationalInputs,
        VerifiedOperationalWorking,
        VerifiedOperationalCell,
    ):
        monkeypatch.setattr(record, "__init__", forbid)
    result = verify(fresh_cell)
    assert result.summary == fresh_cell.summary


@pytest.mark.parametrize(
    "field",
    (
        "expected_metadata_sha256",
        "expected_profile_sha256",
        "expected_descriptor_sha256",
        "expected_binding_sha256",
        "expected_cell_reservation_sha256",
    ),
)
def test_each_independent_pin_is_required(fresh_cell, field):
    with pytest.raises(api().SeriesCellScienceError):
        verify(fresh_cell, **{field: "0" * 64})


@pytest.mark.parametrize("kind", ("missing", "extra", "duplicate", "list"))
def test_inventory_shape_remains_closed(fresh_cell, kind):
    values = tuple(fresh_cell.values.items())
    variants = dict(
        missing=values[:-1],
        extra=values + (("extra", b"x"),),
        duplicate=values + (values[0],),
        list=list(values),
    )
    with pytest.raises(api().SeriesCellScienceError):
        verify(fresh_cell, payloads=variants[kind])


def public_replacement(case, public):
    values = dict(case.values)
    content = receipt._json_bytes(public, "invented")
    values["public-summary.json"] = content
    outcome = json.loads(values["attempt/outcome.json"])
    outcome["public_summary_sha256"] = digest(content)
    values["attempt/outcome.json"] = receipt._json_bytes(outcome, "invented")
    return tuple(values.items()), {name: digest(body) for name, body in values.items()}


@pytest.mark.parametrize(
    "field",
    (
        "revision",
        "runtime_sha256",
        "root_reservation_sha256",
        "operational_profile_sha256",
        "descriptor_sha256",
    ),
)
def test_rehashed_public_old_or_wrong_execution_cannot_replace_new_context(
    fresh_cell, field
):
    public = json.loads(fresh_cell.values["public-summary.json"])
    public["execution"][field] = "0" * (40 if field == "revision" else 64)
    payloads, pins = public_replacement(fresh_cell, public)
    with pytest.raises(api().SeriesCellScienceError):
        verify(fresh_cell, payloads=payloads, expected_snapshot_sha256=pins)


def test_rehashed_summary_cannot_erase_terminal_errors(fresh_cell):
    public = json.loads(fresh_cell.values["public-summary.json"])
    public["summary"]["request_errors"] = 0
    payloads, pins = public_replacement(fresh_cell, public)
    with pytest.raises(api().SeriesCellScienceError):
        verify(fresh_cell, payloads=payloads, expected_snapshot_sha256=pins)
