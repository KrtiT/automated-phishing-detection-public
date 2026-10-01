import json

import pytest
from study_series_acceptance_fixtures import api, working
from study_series_cell_fixtures import (
    candidates,
    fresh_cell,
    manifests,
    series_case,
    verify,
)

from automated_phishing_detection import execution_receipt as receipt

__all__ = ["candidates", "fresh_cell", "manifests", "series_case"]


def test_full_working_and_publication_preserve_new_context_and_original_science(
    fresh_cell,
):
    result = working(fresh_cell)
    assert result.summary == fresh_cell.summary
    assert result.inputs == fresh_cell.inputs
    assert result.run == fresh_cell.run
    assert result.authorizes_execution is False
    public = api().build_series_cell_public(
        result, reservation_sha256=result.reservation_sha256
    )
    assert public == json.loads(fresh_cell.values["public-summary.json"])
    published = api().verify_series_published_cell(
        tuple(fresh_cell.values.items()),
        working=result,
        expected_public_bytes=receipt._json_bytes(public, "invented"),
    )
    assert published == verify(fresh_cell)


def test_identity_projection_precedes_reservation_without_same_parent_relabelling(
    fresh_cell,
):
    inputs = fresh_cell.inputs.computational
    identity = api().series_cell_identity(
        inputs.accepted_bytes,
        inputs.descriptor_bytes,
        expected_metadata_sha256=fresh_cell.arguments["expected_metadata_sha256"],
    )
    assert (
        identity
        == json.loads(fresh_cell.values["attempt/reservation.json"])["identity"]
    )
    assert "reservation_sha256" not in identity


@pytest.mark.parametrize(
    "field,value",
    (
        ("service_command", ("wrong",)),
        ("client_command", ("wrong",)),
        ("expected_deadlines", dict(startup=301, shutdown=180, terminate=10, kill=10)),
        ("expected_metadata_sha256", "0" * 64),
        ("expected_profile_sha256", "0" * 64),
        ("observation", None),
    ),
)
def test_working_requires_independent_actual_parent_context(fresh_cell, field, value):
    with pytest.raises(api().SeriesCellAcceptanceError):
        working(fresh_cell, **{field: value})
