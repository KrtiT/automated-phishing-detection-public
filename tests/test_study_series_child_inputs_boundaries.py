"""Fully repinned child bytes still require every origin/current identity join."""

import json
from dataclasses import asdict, replace

import pytest
from study_series_child_inputs_fixtures import (
    api,
    candidates,
    child_case,
    digest,
    manifests,
    rebound,
    relink_origin,
    restored,
    series_case,
)

from automated_phishing_detection import _operational_input_schema as schema
from automated_phishing_detection._checkpoint_codec import canonical_bytes
from automated_phishing_detection.operational_cell_inputs import bind_cell_descriptor
from automated_phishing_detection.operational_schedule import cell_for_ordinal

__all__ = ["candidates", "child_case", "manifests", "series_case"]


def set_nested(value, path, replacement):
    for name in path[:-1]:
        value = value[name]
    value[path[-1]] = replacement


@pytest.mark.parametrize(
    "path",
    (
        ("root_reservation_sha256",),
        ("operational_profile_sha256",),
        ("execution", "revision"),
        ("execution", "execution_contract_sha256"),
        ("execution", "runtime_sha256"),
        ("execution", "source_spec_sha256"),
        ("internal", "execution", "preparation_summary_sha256"),
        ("internal", "execution", "study_preparation_reservation_sha256"),
        ("internal", "execution", "study_preparation_complete_sha256"),
        ("internal", "snapshot_sha256", "attempt/evidence/bindings.json"),
        ("external", "snapshot_sha256", "attempt/evidence/bindings.json"),
        ("external", "execution", "source_profile_sha256"),
    ),
)
def test_legacy_valid_relinked_origin_must_match_series_profile(child_case, path):
    value = json.loads(child_case.metadata)
    origin = value["origin"]
    set_nested(origin, path, "0" * (40 if path[-1] == "revision" else 64))
    relink_origin(origin)
    schema.validate_metadata(origin)
    selected = rebound(child_case, metadata_value=value)
    assert selected.metadata != child_case.metadata
    with pytest.raises(api().SeriesChildInputError):
        restored(selected)


@pytest.mark.parametrize(
    "field",
    (
        "revision",
        "execution_contract_sha256",
        "runtime_sha256",
        "source_spec_sha256",
    ),
)
def test_rebound_current_execution_cannot_replace_original_contract(child_case, field):
    value = json.loads(child_case.metadata)
    value["execution"][field] = "0" * (40 if field == "revision" else 64)
    with pytest.raises(api().SeriesChildInputError):
        restored(rebound(child_case, metadata_value=value))


@pytest.mark.parametrize("both", (False, True))
def test_rebound_primary_remains_bound_to_profile(child_case, both):
    value = json.loads(child_case.metadata)
    value["primary"]["artifact_hashes"]["gmm.json"] = "0" * 64
    if both:
        value["origin"]["primary"] = value["primary"]
    with pytest.raises(api().SeriesChildInputError):
        restored(rebound(child_case, metadata_value=value))


@pytest.mark.parametrize(
    "field",
    ("kind", "schema_version", "history_index_sha256", "operational_profile_sha256"),
)
def test_rebound_current_metadata_identity_is_closed(child_case, field):
    value = json.loads(child_case.metadata)
    value[field] = False if field == "schema_version" else "0" * 64
    with pytest.raises(api().SeriesChildInputError):
        restored(rebound(child_case, metadata_value=value))


@pytest.mark.parametrize(
    "field", ("series_reservation_sha256", "root_reservation_sha256")
)
def test_rebound_new_reservations_cannot_equal_original(child_case, field):
    value = json.loads(child_case.metadata)
    value[field] = value["origin"]["root_reservation_sha256"]
    described = json.loads(child_case.descriptor)
    described["root_reservation_sha256"] = value["root_reservation_sha256"]
    with pytest.raises(api().SeriesChildInputError):
        restored(rebound(child_case, metadata_value=value, described=described))


def test_fully_rebound_descriptor_cannot_name_accepted_prefix(child_case):
    described = json.loads(child_case.descriptor)
    described["cell"] = asdict(cell_for_ordinal(72))
    with pytest.raises(api().SeriesChildInputError):
        restored(rebound(child_case, described=described))


def test_rehashed_profile_suffix_must_include_admitted_cell(child_case):
    profile = json.loads(child_case.profile)
    profile["segment"]["start_ordinal"] = child_case.frame.cell_ordinal + 1
    with pytest.raises(api().SeriesChildInputError):
        restored(rebound(child_case, profile_value=profile))


@pytest.mark.parametrize("field", ("root_reservation_sha256", "accepted_inputs_sha256"))
def test_repinned_descriptor_still_joins_metadata(child_case, field):
    described = json.loads(child_case.descriptor)
    described[field] = "0" * 64
    content = canonical_bytes(described)
    binding = bind_cell_descriptor(content, cell_reservation_sha256="3" * 64)
    frame = replace(child_case.frame, cell_binding_sha256=digest(binding))
    assert content != child_case.descriptor
    with pytest.raises(api().SeriesChildInputError):
        restored(
            child_case, descriptor_bytes=content, binding_bytes=binding, frame=frame
        )


def mutated_manifest(case):
    if case.frame.cell_ordinal == 121:
        return b"".join(reversed(case.manifest.splitlines(keepends=True)))
    value = json.loads(case.manifest)
    source = value["records"][0]["record_id"].split(":")
    source[1] = "0" * 64
    value["records"][0]["record_id"] = ":".join(source)
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def test_fully_rebound_manifest_requires_original_source_identity(child_case):
    content = mutated_manifest(child_case)
    assert content != child_case.manifest
    with pytest.raises(api().SeriesChildInputError):
        restored(rebound(child_case, body=content))


@pytest.mark.parametrize("field", ("schema_version", "kind", "primary", "origin"))
def test_new_metadata_cannot_omit_required_fields(child_case, field):
    value = json.loads(child_case.metadata)
    value.pop(field)
    if field == "origin":
        content = canonical_bytes(value)
        frame = replace(child_case.frame, accepted_inputs_sha256=digest(content))
        with pytest.raises(api().SeriesChildInputError):
            restored(child_case, metadata_bytes=content, frame=frame)
        return
    with pytest.raises(api().SeriesChildInputError):
        restored(rebound(child_case, metadata_value=value))
