"""Invented scientific restoration uses original kernels without live authority."""

from dataclasses import FrozenInstanceError

import pytest
from study_history_internal_fixtures import (
    api,
    historical_internal,
    inputs,
    published,
    runner,
    verify,
)
from study_history_internal_prepared_fixtures import as_prepared_history

from automated_phishing_detection.internal_source_handoff import (
    VerifiedInternalSnapshot,
)

__all__ = ["historical_internal", "inputs", "published", "runner"]


def test_historical_internal_science_api_exists():
    assert callable(api().verify_historical_internal_science)


def test_complete_internal_science_returns_immutable_retained_data(historical_internal):
    result = verify(historical_internal)
    assert type(result) is VerifiedInternalSnapshot
    assert dict(result.payloads) == historical_internal.payloads
    assert len(result.records) == result.public_summary["row_count"]
    assert set(result.manifests) == {10, 100, 500}
    assert result.overlap_domains
    with pytest.raises(FrozenInstanceError):
        result.records = ()


def test_retained_preparation_ancestry_remains_original(historical_internal):
    case = as_prepared_history(historical_internal)
    result = verify(case)
    assert dict(result.payloads) == case.payloads
    assert result.public_summary["execution"] == case.execution


@pytest.mark.parametrize(
    "field",
    ["study_preparation_reservation_sha256", "study_preparation_complete_sha256"],
)
def test_independent_preparation_context_is_required(historical_internal, field):
    case = as_prepared_history(historical_internal)
    with pytest.raises(ValueError):
        verify(case, expected_execution=case.execution | {field: "c" * 64})
