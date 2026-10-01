"""The loader owns verification order and preserves interruption precedence."""

from importlib import import_module
from types import SimpleNamespace

import pytest


def setup(monkeypatch):
    module = import_module("automated_phishing_detection.study_series_history")
    events = []
    reader = SimpleNamespace(check=lambda: events.append("files_checked"))
    origin = SimpleNamespace(metadata_bytes=b"original metadata")
    index, document, profile = object(), object(), object()
    monkeypatch.setattr(module, "HistoryFiles", lambda: reader)
    monkeypatch.setattr(
        module, "require_final_policy", lambda _: events.append("policy")
    )
    monkeypatch.setattr(
        module, "recheck_series_public_execution", lambda _: events.append("public")
    )
    monkeypatch.setattr(module, "archives", lambda *args: (index, document, profile))
    monkeypatch.setattr(module, "restore_origin", lambda *args: origin)
    monkeypatch.setattr(
        module, "verify_physical", lambda *args: events.append("physical")
    )
    monkeypatch.setattr(
        module, "restore_sources", lambda *args: ("internal", "external")
    )
    monkeypatch.setattr(module, "restore_cells", lambda *args: ("cell1", "cell2"))
    return module, events, reader, index


def test_yields_verified_data_and_rechecks_public_and_files_on_exit(monkeypatch):
    module, events, reader, index = setup(monkeypatch)
    with module.hold_series_history(object()) as history:
        assert history.index is index
        assert history.origin_metadata_bytes == b"original metadata"
        assert history.historical_prefix == ("cell1", "cell2")
        assert history.internal_snapshot == "internal"
        assert history.external_snapshot == "external"
        assert events[:2] == ["public", "policy"]
        assert "physical" in events
        events.clear()
        history.check()
        assert events == ["public", "policy", "files_checked"]
    assert events[-3:] == ["public", "policy", "files_checked"]


def test_failed_science_never_yields_and_rechecks_retained_files(monkeypatch):
    module, events, reader, index = setup(monkeypatch)

    def reject(*args):
        raise ValueError("invented private detail")

    monkeypatch.setattr(module, "restore_cells", reject)
    with pytest.raises(module.SeriesHistoryError, match="^invalid_series_history$"):
        with module.hold_series_history(object()):
            pytest.fail("invalid history yielded")
    assert "files_checked" in events


def test_caller_failure_is_not_relabeled_as_historical_rejection(monkeypatch):
    module, events, reader, index = setup(monkeypatch)
    original = RuntimeError("caller failed")
    with pytest.raises(RuntimeError) as caught:
        with module.hold_series_history(object()):
            raise original
    assert caught.value is original


@pytest.mark.parametrize("interruption", [KeyboardInterrupt, SystemExit])
def test_interruption_is_not_replaced_by_cleanup_failure(monkeypatch, interruption):
    module, events, reader, index = setup(monkeypatch)
    original = interruption("original stop")

    def changed():
        raise ValueError("changed history")

    with pytest.raises(interruption) as caught:
        with module.hold_series_history(object()):
            reader.check = changed
            raise original
    assert caught.value is original
