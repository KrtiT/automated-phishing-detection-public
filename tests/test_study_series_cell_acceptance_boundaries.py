"""Closed working inventories and independently retained current-series context."""

import os
from dataclasses import replace

import pytest
from study_series_acceptance_fixtures import (
    api,
    arguments,
    forbid,
    forbid_constructors,
    payloads,
    working,
)
from study_series_cell_fixtures import candidates, fresh_cell, manifests, series_case
from test_study_history_external_boundaries import forbid_live

from automated_phishing_detection import _study_history_cell_science as science
from automated_phishing_detection._operational_cell_protocol import WORKING_NAMES

__all__ = ["candidates", "fresh_cell", "manifests", "series_case"]


@pytest.mark.parametrize("name", WORKING_NAMES)
def test_every_original_working_buffer_is_required(fresh_cell, name):
    values = dict(payloads(fresh_cell))
    values[name] += b"\n"
    with pytest.raises(api().SeriesCellAcceptanceError):
        working(fresh_cell, values=tuple(values.items()))


@pytest.mark.parametrize(
    "kind", ("missing", "extra", "duplicate", "list", "member", "mutable")
)
def test_exact17_inventory_has_only_immutable_unique_pairs(fresh_cell, kind):
    values = payloads(fresh_cell)
    variants = {
        "missing": values[:-1],
        "extra": (*values, ("extra", b"x")),
        "duplicate": (*values, values[0]),
        "list": list(values),
        "member": (list(values[0]), *values[1:]),
        "mutable": ((values[0][0], bytearray(values[0][1])), *values[1:]),
    }
    with pytest.raises(api().SeriesCellAcceptanceError):
        working(fresh_cell, values=variants[kind])


@pytest.mark.parametrize(
    "field",
    (
        "revision",
        "execution_contract_sha256",
        "runtime_sha256",
        "source_spec_sha256",
        "root_reservation_sha256",
        "operational_profile_sha256",
        "descriptor_sha256",
    ),
)
def test_independent_identity_prevents_old_or_other_context(
    fresh_cell, monkeypatch, field
):
    identity = arguments(fresh_cell)["expected_identity"]
    identity[field] = "0" * (40 if field == "revision" else 64)
    monkeypatch.setattr(science, "summary", forbid)
    with pytest.raises(api().SeriesCellAcceptanceError):
        working(fresh_cell, expected_identity=identity)


@pytest.mark.parametrize(
    "field",
    (
        "accepted_bytes",
        "descriptor_bytes",
        "binding_bytes",
        "manifest_bytes",
        "requests",
        "cell",
    ),
)
def test_each_computational_input_is_revalidated_before_math(
    fresh_cell, monkeypatch, field
):
    original = fresh_cell.inputs.computational
    value = getattr(original, field)
    changed = (
        value + b"\n"
        if type(value) is bytes
        else value[:-1]
        if type(value) is tuple
        else replace(value, ordinal=value.ordinal + 1)
    )
    inputs = replace(
        fresh_cell.inputs, computational=replace(original, **{field: changed})
    )
    monkeypatch.setattr(science, "summary", forbid)
    with pytest.raises(api().SeriesCellAcceptanceError):
        working(fresh_cell, inputs=inputs)


@pytest.mark.parametrize("role", ("internal", "external"))
def test_substituted_scientific_snapshot_cannot_reach_math(
    fresh_cell, monkeypatch, role
):
    name = f"{role}_snapshot"
    snapshot = fresh_cell.arguments[name]
    values = snapshot.payloads
    changed = replace(
        snapshot, payloads=((values[0][0], values[0][1] + b"x"), *values[1:])
    )
    monkeypatch.setattr(science, "summary", forbid)
    with pytest.raises(api().SeriesCellAcceptanceError):
        working(fresh_cell, **{name: changed})


@pytest.mark.parametrize(
    "kind", ("legacy_carrier", "origin_bytes", "metadata", "profile", "origin_view")
)
def test_old_metadata_or_bad_current_pins_cannot_reach_math(
    fresh_cell, monkeypatch, kind
):
    carrier = fresh_cell.inputs
    changes = {
        "legacy_carrier": {"inputs": carrier.computational},
        "origin_bytes": {
            "inputs": replace(
                carrier,
                computational=replace(
                    carrier.computational, accepted_bytes=carrier.origin_metadata_bytes
                ),
            )
        },
        "metadata": {"expected_metadata_sha256": "0" * 64},
        "profile": {"expected_profile_sha256": "0" * 64},
        "origin_view": {"inputs": replace(carrier, origin_metadata_bytes=b"{}")},
    }
    monkeypatch.setattr(science, "summary", forbid)
    with pytest.raises(api().SeriesCellAcceptanceError):
        working(fresh_cell, **changes[kind])


def test_verification_never_creates_an_observation_or_reads_models(
    fresh_cell, monkeypatch
):
    options, values, module = arguments(fresh_cell), payloads(fresh_cell), api()
    forbid_live(monkeypatch)
    forbid_constructors(monkeypatch)
    monkeypatch.setattr(os, "open", forbid)
    result = module.verify_series_working_cell(values, **options)
    assert result.summary == fresh_cell.summary
    assert result.authorizes_execution is False
    assert (
        result.inputs.origin_metadata_bytes == fresh_cell.inputs.origin_metadata_bytes
    )
