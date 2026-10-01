"""Publication must match held working bytes without recomputing science."""

import json
import os
from dataclasses import FrozenInstanceError, replace

import pytest
from study_series_acceptance_fixtures import (
    accepted_working,
    api,
    forbid,
    forbid_constructors,
    publish,
)
from study_series_cell_fixtures import candidates, fresh_cell, manifests, series_case
from test_study_history_external_boundaries import forbid_live

from automated_phishing_detection import _study_history_cell_science as science
from automated_phishing_detection import execution_receipt as receipt
from automated_phishing_detection._checkpoint_codec import canonical_bytes
from automated_phishing_detection._operational_cell_protocol import SNAPSHOT_NAMES

__all__ = ["accepted_working", "candidates", "fresh_cell", "manifests", "series_case"]


@pytest.mark.parametrize("name", SNAPSHOT_NAMES)
def test_every_original_publication_buffer_matches_working(
    fresh_cell, accepted_working, name
):
    values = fresh_cell.values | {name: fresh_cell.values[name] + b"\n"}
    with pytest.raises(api().SeriesCellAcceptanceError):
        publish(fresh_cell, accepted_working, values=tuple(values.items()))


@pytest.mark.parametrize(
    "kind", ("missing", "extra", "duplicate", "list", "member", "mutable")
)
def test_exact36_publication_inventory_is_closed(fresh_cell, accepted_working, kind):
    values = tuple(fresh_cell.values.items())
    variants = {
        "missing": values[:-1],
        "extra": (*values, ("extra", b"x")),
        "duplicate": (*values, values[0]),
        "list": list(values),
        "member": (list(values[0]), *values[1:]),
        "mutable": ((values[0][0], bytearray(values[0][1])), *values[1:]),
    }
    with pytest.raises(api().SeriesCellAcceptanceError):
        publish(fresh_cell, accepted_working, values=variants[kind])


@pytest.mark.parametrize(
    "field", ("summary", "private_sha256", "execution", "cell", "extra")
)
def test_caller_cannot_replace_expected_public_projection(
    fresh_cell, accepted_working, field
):
    value = json.loads(fresh_cell.values["public-summary.json"])
    value[field] = {}
    content = receipt._json_bytes(value, "invented")
    values = fresh_cell.values | {"public-summary.json": content}
    with pytest.raises(api().SeriesCellAcceptanceError):
        publish(
            fresh_cell, accepted_working, values=tuple(values.items()), content=content
        )


@pytest.mark.parametrize(
    "name,field,value",
    (
        ("attempt/finalize.claim", "operation", "failure"),
        ("attempt/finalize.claim", "reservation_sha256", "0" * 64),
        ("attempt/finalize.claim", "schema_version", True),
        ("attempt/outcome.json", "status", "failed"),
        ("attempt/outcome.json", "private_sha256", {}),
        ("attempt/outcome.json", "public_summary_sha256", "0" * 64),
        ("attempt/outcome.json", "extra", True),
    ),
)
def test_full_claim_and_outcome_projection_is_required(
    fresh_cell, accepted_working, name, field, value
):
    changed = json.loads(fresh_cell.values[name]) | {field: value}
    values = fresh_cell.values | {name: receipt._json_bytes(changed, "invented")}
    with pytest.raises(api().SeriesCellAcceptanceError):
        publish(fresh_cell, accepted_working, values=tuple(values.items()))


@pytest.mark.parametrize("field", ("summary_bytes", "payloads", "reservation_sha256"))
def test_modified_working_context_cannot_back_original_publication(
    fresh_cell, accepted_working, field
):
    values = dict(accepted_working.payloads)
    values["run.json"] += b"\n"
    replacements = {
        "summary_bytes": canonical_bytes({}),
        "payloads": tuple(values.items()),
        "reservation_sha256": "0" * 64,
    }
    changed = replace(accepted_working, **{field: replacements[field]})
    with pytest.raises(api().SeriesCellAcceptanceError):
        publish(fresh_cell, changed)


def test_result_views_are_fresh_frozen_and_never_authority(
    fresh_cell, accepted_working
):
    result = publish(fresh_cell, accepted_working)
    for value in (accepted_working, result):
        with pytest.raises(FrozenInstanceError):
            value.summary_bytes = b"{}"
        value.summary.clear()
        value.run.after_measured.admitted_requests = 0
        assert value.summary == fresh_cell.summary
        assert value.run == fresh_cell.run
        assert value.authorizes_execution is False
    private = accepted_working.private_outputs
    private.clear()
    assert (
        accepted_working.private_outputs["run.json"]
        == fresh_cell.values["attempt/run.json"]
    )
    assert result.payloads == tuple(fresh_cell.values.items())


def test_publication_does_not_recompute_math_or_accept_wrong_reservation(
    fresh_cell, accepted_working, monkeypatch
):
    forbid_live(monkeypatch)
    forbid_constructors(monkeypatch)
    monkeypatch.setattr(os, "open", forbid)
    monkeypatch.setattr(science, "summary", forbid)
    assert publish(fresh_cell, accepted_working).summary == fresh_cell.summary
    with pytest.raises(api().SeriesCellAcceptanceError):
        api().build_series_cell_public(accepted_working, reservation_sha256="0" * 64)
