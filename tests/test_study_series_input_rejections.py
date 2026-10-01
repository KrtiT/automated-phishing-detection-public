"""Rehashed declarations cannot replace original scientific and runtime identity."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
from study_series_adoption_fixtures import digest
from study_series_input_fixtures import (
    api,
    candidates,
    descriptor,
    manifests,
    metadata,
    restore,
    series_case,
)

from automated_phishing_detection._checkpoint_codec import canonical_bytes

__all__ = ["candidates", "manifests", "series_case"]


@pytest.mark.parametrize(
    ("group", "field"),
    (
        ("execution", "runtime_sha256"),
        ("scientific_pins", "source_spec_sha256"),
        ("scientific_pins", "primary_metadata_sha256"),
        ("scientific_pins", "internal_bindings_sha256"),
        ("scientific_pins", "external_bindings_sha256"),
        ("origin", "root_reservation_sha256"),
        ("origin", "preparation_reservation_sha256"),
        ("origin", "preparation_complete_sha256"),
    ),
)
def test_profile_science_runtime_and_ancestry_pins_match_original_sources(
    series_case, group, field
):
    changed = SimpleNamespace(**vars(series_case))
    changed.profile = deepcopy(series_case.profile)
    changed.profile[group][field] = "0" * 64
    with pytest.raises(api().SeriesInputError):
        metadata(changed)


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("expected_origin_sha256", "0" * 64),
        ("expected_profile_sha256", "0" * 64),
        ("series_reservation_sha256", "2" * 64),
        ("segment_reservation_sha256", "1" * 64),
        ("series_reservation_sha256", "f" * 64),
        ("segment_reservation_sha256", "f" * 64),
        ("series_reservation_sha256", True),
        ("segment_reservation_sha256", "A" * 64),
    ),
)
def test_independent_identity_pins_are_required_and_reservations_are_distinct(
    series_case, name, value
):
    with pytest.raises(api().SeriesInputError):
        metadata(series_case, **{name: value})


@pytest.mark.parametrize(
    "field",
    (
        "schema_version",
        "kind",
        "root_reservation_sha256",
        "series_reservation_sha256",
        "profile_sha256",
        "history_index_sha256",
        "execution",
        "operational_profile_sha256",
        "primary",
        "origin",
    ),
)
@pytest.mark.parametrize("operation", ("missing", "changed"))
def test_closed_new_metadata_rejects_rehashed_missing_and_substituted_members(
    series_case, field, operation
):
    value = json.loads(metadata(series_case))
    if operation == "missing":
        value.pop(field)
    else:
        value[field] = False
    with pytest.raises(api().SeriesInputError):
        descriptor(series_case, content=canonical_bytes(value))


def test_new_metadata_never_accepts_legacy_kind_even_after_rehash(series_case):
    value = json.loads(metadata(series_case))
    value["kind"] = "same-parent-operational-inputs-v1"
    with pytest.raises(api().SeriesInputError):
        descriptor(series_case, content=canonical_bytes(value))


@pytest.mark.parametrize(
    "field",
    ("revision", "execution_contract_sha256", "runtime_sha256", "source_spec_sha256"),
)
def test_current_computational_execution_is_exactly_profile_bound(series_case, field):
    value = json.loads(metadata(series_case))
    value["execution"][field] = "0" * (40 if field == "revision" else 64)
    with pytest.raises(api().SeriesInputError):
        descriptor(series_case, content=canonical_bytes(value))


@pytest.mark.parametrize(
    "field",
    (
        "expected_metadata_sha256",
        "expected_profile_sha256",
        "expected_binding_sha256",
        "expected_cell_reservation_sha256",
    ),
)
def test_restoration_requires_each_independent_identity(series_case, field):
    with pytest.raises(api().SeriesInputError):
        restore(series_case, **{field: "0" * 64})


@pytest.mark.parametrize(
    "field",
    (
        "metadata_bytes",
        "profile_bytes",
        "descriptor_bytes",
        "binding_bytes",
        "manifest_bytes",
    ),
)
def test_all_restored_bytes_reject_substitution(series_case, field):
    with pytest.raises(
        api().SeriesInputError, match="^invalid_series_operational_inputs$"
    ):
        restore(series_case, **{field: b"{}\n"})


def test_new_primary_must_remain_exactly_original_primary(series_case):
    value = json.loads(metadata(series_case))
    value["primary"]["artifact_hashes"]["gmm.json"] = "0" * 64
    with pytest.raises(api().SeriesInputError):
        descriptor(series_case, content=canonical_bytes(value))


def test_profile_binding_cannot_be_swapped_with_a_rehashed_other_profile(series_case):
    changed = deepcopy(series_case.profile)
    changed["series_id"] = "invented-other-series"
    assert digest(changed) != digest(series_case.profile)
    with pytest.raises(api().SeriesInputError):
        descriptor(series_case, content=metadata(series_case), profile_value=changed)
