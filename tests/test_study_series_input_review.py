"""Independent exact-source and old/new authority boundary checks."""

import copy
import json

import pytest
from study_series_input_fixtures import (
    candidates,
    manifests,
    metadata,
    restore,
    series_case,
)

from automated_phishing_detection import _operational_cell_records as original
from automated_phishing_detection._checkpoint_codec import canonical_bytes
from automated_phishing_detection.external_replay_codec import (
    decode_external_replay_manifest,
)
from automated_phishing_detection.replay_manifest_codec import decode_replay_manifest

__all__ = ["candidates", "manifests", "series_case"]


@pytest.mark.parametrize("ordinal", (73, 111, 121))
def test_exact_original_request_order_and_raw_values_survive_transition(
    series_case, ordinal
):
    result = restore(series_case, ordinal)
    inputs = result.computational
    options = dict(expected_sha256=inputs.manifest_sha256)
    if ordinal < 121:
        rows = decode_replay_manifest(
            inputs.manifest_bytes,
            expected_prevalence_basis_points=inputs.cell.prevalence_basis_points,
            **options,
        ).records
    else:
        rows = decode_external_replay_manifest(inputs.manifest_bytes, **options)
    assert tuple((row.record_id, row.raw_url) for row in inputs.requests) == tuple(
        (row.record_id, row.raw_url) for row in rows
    )
    assert len(inputs.requests[:1000]) == 1000
    assert result.origin_metadata_bytes == series_case.origin_bytes


def test_new_carrier_cannot_authenticate_as_old_same_parent_metadata(series_case):
    inputs = restore(series_case).computational
    with pytest.raises(ValueError):
        original.authenticate(
            inputs.accepted_bytes,
            inputs.descriptor_bytes,
            inputs.binding_bytes,
            expected_binding_sha256=inputs.binding_sha256,
            expected_cell_reservation_sha256="3" * 64,
        )


def test_source_and_profile_objects_are_unchanged_by_new_context(series_case):
    before = copy.deepcopy(series_case.profile)
    original = series_case.origin_bytes
    contents = metadata(series_case)
    result = restore(series_case)
    assert series_case.profile == before
    assert series_case.origin_bytes == original
    assert canonical_bytes(json.loads(contents)["origin"]) == original
    assert result.origin_metadata_bytes == original
