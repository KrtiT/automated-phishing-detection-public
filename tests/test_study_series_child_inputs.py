from dataclasses import FrozenInstanceError, replace

import pytest
from study_series_child_inputs_fixtures import (
    api,
    candidates,
    child_case,
    manifests,
    restored,
    series_case,
)

__all__ = ["candidates", "manifests", "series_case", "child_case"]


def test_complete_child_carrier_matches_stronger_parent_restoration(child_case):
    result = restored(child_case)
    assert result == child_case.expected
    assert result.authorizes_execution is False
    assert result.computational.accepted_bytes == child_case.metadata
    with pytest.raises(FrozenInstanceError):
        result.authorizes_execution = True


@pytest.mark.parametrize(
    "name",
    (
        "metadata_bytes",
        "profile_bytes",
        "descriptor_bytes",
        "binding_bytes",
        "manifest_bytes",
    ),
)
def test_unpinned_child_bytes_reject(child_case, name):
    with pytest.raises(api().SeriesChildInputError):
        restored(child_case, **{name: b"changed"})


@pytest.mark.parametrize(
    "name",
    (
        "profile_sha256",
        "history_index_sha256",
        "series_reservation_sha256",
        "segment_reservation_sha256",
        "origin_reservation_sha256",
        "accepted_inputs_sha256",
        "cell_binding_sha256",
    ),
)
def test_frame_context_substitution_rejects(child_case, name):
    with pytest.raises(api().SeriesChildInputError):
        restored(child_case, frame=replace(child_case.frame, **{name: "0" * 64}))


def test_different_cell_or_reservation_rejects(child_case):
    with pytest.raises(api().SeriesChildInputError):
        restored(child_case, frame=replace(child_case.frame, cell_ordinal=125))
    with pytest.raises(api().SeriesChildInputError):
        restored(child_case, expected_cell_reservation_sha256="0" * 64)
