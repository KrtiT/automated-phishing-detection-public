"""Closed series-root flags and live supervisor admission before public binding."""

import asyncio
import importlib.util
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest


def api():
    path = Path(__file__).parents[1] / "scripts/run_study_series.py"
    assert path.is_file(), "missing supervised series root CLI"
    spec = importlib.util.spec_from_file_location("series_root_test_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def arguments():
    return [
        "--repo-root",
        "/invented/repository",
        "--expected-revision",
        "a" * 40,
        "--expected-profile-sha256",
        "b" * 64,
        "--envelope",
        "/invented/envelope.json",
        "--expected-envelope-sha256",
        "c" * 64,
    ]


@pytest.mark.parametrize(
    "extra",
    (
        ["--cell", "73"],
        ["--timeout", "999"],
        ["--model", "private"],
        ["--expected-profile-sha", "b" * 64],
    ),
)
def test_root_cli_has_no_workload_path_or_deadline_overrides(extra):
    with pytest.raises(SystemExit):
        api()._parser().parse_args(arguments() + extra)


def test_supervisor_is_consumed_before_any_binder_or_loader(monkeypatch):
    module, events, public = api(), [], object()

    @contextmanager
    def supervised(arguments):
        events.append("supervised")
        yield SimpleNamespace(check=lambda: events.append("live"))
        events.append("supervisor_closed")

    def bind(*args, **kwargs):
        assert events == ["supervised"]
        events.append("bound")
        return public

    async def run(selected, *, hold_history, lifecycle_check):
        assert selected is public and hold_history is module.hold_series_history
        lifecycle_check()
        events.append("ran")

    monkeypatch.setattr(module, "consume_series_supervision", supervised)
    monkeypatch.setattr(module, "bind_series_public_execution", bind)
    monkeypatch.setattr(module, "_run_series_bound", run)
    asyncio.run(module._run(module._parser().parse_args(arguments())))
    assert events == ["supervised", "bound", "live", "ran", "supervisor_closed"]


def test_missing_supervisor_never_opens_public_or_history(monkeypatch):
    module = api()

    def reject(arguments):
        raise ValueError("missing_live_supervisor")

    monkeypatch.setattr(module, "consume_series_supervision", reject)
    monkeypatch.setattr(
        module,
        "bind_series_public_execution",
        lambda *args, **kwargs: pytest.fail("binder reached"),
    )
    with pytest.raises(ValueError, match="missing_live_supervisor"):
        asyncio.run(module._run(module._parser().parse_args(arguments())))
