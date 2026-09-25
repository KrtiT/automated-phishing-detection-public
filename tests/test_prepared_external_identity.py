"""Retained external identity projections use authenticated bytes only."""

import pytest
from prepared_external_fixtures import (
    inputs,
    preparation_api,
    preparation_case,
    prepared_case,
    runner,
)

from automated_phishing_detection import source_runner
from automated_phishing_detection._external_source_records import external_identity

__all__ = ["inputs", "preparation_api", "preparation_case", "prepared_case", "runner"]


def test_prepared_identity_does_not_reopen_public_sources(prepared_case, monkeypatch):
    case = prepared_case

    def forbidden(*args, **kwargs):
        pytest.fail("byte-only identity reopened public metadata")

    monkeypatch.setattr(source_runner, "_public_sources", forbidden)
    result = external_identity(
        case.binding, case.profile, case.handoff, preparation=case.preparation
    )
    assert (
        result["study_preparation_complete_sha256"]
        == case.preparation.completion_sha256
    )
