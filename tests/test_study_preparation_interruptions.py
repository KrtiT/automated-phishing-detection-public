"""Failure retention preserves the first interruption and already available bytes."""

from types import SimpleNamespace

import pytest
from study_preparation_runner_fixtures import preparation_api

__all__ = ["preparation_api"]


def test_failure_retention_interruption_preserves_existing_progress(
    preparation_api, monkeypatch
):
    interruption = KeyboardInterrupt()
    state = SimpleNamespace(
        writer=SimpleNamespace(snapshot=lambda: b"private progress"),
        attempt=object(),
        stage="external_source",
    )

    def interrupted(*args, **kwargs):
        raise interruption

    monkeypatch.setattr(preparation_api, "record_failure", interrupted)
    result = preparation_api._failure(state, ValueError("original failure"))
    assert result is interruption
    assert result.preparation_progress == b"private progress"


@pytest.mark.parametrize("original", [ValueError(), KeyboardInterrupt(), SystemExit(0)])
def test_first_interruption_wins_across_progress_and_failure_record(
    preparation_api, monkeypatch, original
):
    first, later = KeyboardInterrupt(), SystemExit(0)

    def snapshot():
        raise first

    def recorded(*args, **kwargs):
        raise later

    state = SimpleNamespace(
        writer=SimpleNamespace(snapshot=snapshot), attempt=object(), stage="source"
    )
    monkeypatch.setattr(preparation_api, "record_failure", recorded)
    selected = preparation_api._failure(state, original)
    assert selected is (first if isinstance(original, Exception) else original)
