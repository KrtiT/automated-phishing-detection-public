"""Series metadata preserves original sources without claiming a new observation."""

import json
from dataclasses import FrozenInstanceError

import pytest
from study_series_adoption_fixtures import digest
from study_series_input_fixtures import (
    candidates,
    descriptor,
    manifests,
    metadata,
    restore,
    series_case,
)

from automated_phishing_detection import _operational_input_schema as legacy
from automated_phishing_detection.operational_cell_inputs import (
    RestoredOperationalCell,
    restore_cell_inputs,
)

__all__ = ["candidates", "manifests", "series_case"]


def test_metadata_uses_current_execution_and_keeps_original_accepted_metadata(
    series_case,
):
    value = json.loads(metadata(series_case))
    origin = json.loads(series_case.origin_bytes)
    assert value["kind"] == "study-series-operational-inputs-v1"
    assert value["origin"] == origin
    assert value["primary"] == origin["primary"]
    assert value["root_reservation_sha256"] == "2" * 64
    assert (
        value["execution"]["revision"] == series_case.profile["execution"]["revision"]
    )
    assert value["profile_sha256"] == digest(series_case.profile)
    assert (
        value["history_index_sha256"] == series_case.profile["history"]["index_sha256"]
    )
    with pytest.raises(legacy.OperationalInputError):
        legacy.validate_metadata(value)


@pytest.mark.parametrize("ordinal", (73, 91, 120, 121, 125))
def test_internal_and_shift_descriptors_restore_truthful_computational_carrier(
    series_case, ordinal
):
    result = restore(series_case, ordinal)
    assert result.origin_metadata_bytes == series_case.origin_bytes
    assert result.authorizes_execution is False
    computational = result.computational
    assert type(computational) is RestoredOperationalCell
    assert (
        json.loads(computational.accepted_bytes)["kind"]
        == "study-series-operational-inputs-v1"
    )
    assert computational.cell.ordinal == ordinal
    assert len(computational.requests) == (1001 if ordinal >= 121 else 10000)
    assert computational.primary == json.loads(series_case.origin_bytes)["primary"]
    assert (
        computational.operational_profile_sha256
        == series_case.profile["components"]["current_operational"]
    )
    assert "origin_metadata_bytes" not in repr(result)
    with pytest.raises(FrozenInstanceError):
        result.computational = None


def test_legacy_restore_never_accepts_the_new_metadata_kind(series_case):
    inputs = restore(series_case).computational
    with pytest.raises(legacy.OperationalInputError):
        restore_cell_inputs(
            inputs.accepted_bytes,
            inputs.descriptor_bytes,
            inputs.binding_bytes,
            inputs.manifest_bytes,
            expected_binding_sha256=inputs.binding_sha256,
            expected_cell_reservation_sha256="3" * 64,
        )


@pytest.mark.parametrize("ordinal", (True, 1, 72, 126, "73"))
def test_only_exact_suffix_ordinals_are_available(series_case, ordinal):
    with pytest.raises(ValueError, match="^invalid_series_operational_inputs$"):
        descriptor(series_case, ordinal)
