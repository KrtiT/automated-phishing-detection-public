"""Historical access stays behind adoption and real public binding checks."""

import json
from dataclasses import replace
from importlib import import_module

import pytest
from study_series_execution_fixtures import bind, public_case

from automated_phishing_detection._checkpoint_codec import canonical_bytes

__all__ = ["public_case"]


def api():
    return import_module("automated_phishing_detection.study_series_history")


def test_candidate_policy_stops_before_any_historical_read(public_case, monkeypatch):
    module = api()
    binding = bind(public_case)
    candidate = json.loads(binding.policy_bytes) | {
        "status": "development_candidate_header_only"
    }
    binding = replace(binding, policy_bytes=canonical_bytes(candidate))
    calls = []
    monkeypatch.setattr(module, "recheck_series_public_execution", lambda unused: None)
    monkeypatch.setattr(module.HistoryFiles, "read", lambda *args: calls.append(args))
    with pytest.raises(module.SeriesHistoryError):
        with module.hold_series_history(binding):
            pytest.fail("candidate execution gate opened")
    assert calls == []


def test_public_recheck_stops_before_historical_read(public_case, monkeypatch):
    module = api()
    binding = bind(public_case)
    calls = []

    def reject(unused):
        raise ValueError("invented public binding failure")

    monkeypatch.setattr(module, "require_final_policy", lambda _: None)
    monkeypatch.setattr(module, "recheck_series_public_execution", reject)
    monkeypatch.setattr(module.HistoryFiles, "read", lambda *args: calls.append(args))
    with pytest.raises(module.SeriesHistoryError):
        with module.hold_series_history(binding):
            pytest.fail("invalid public binding opened history")
    assert calls == []
