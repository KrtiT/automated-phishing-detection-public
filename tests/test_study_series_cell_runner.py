"""Private series cell orchestration must not become an independent access route."""

from importlib import import_module
from importlib.util import find_spec

import pytest
from study_series_input_fixtures import candidates, manifests, series_case

__all__ = ["candidates", "manifests", "series_case"]


def api():
    name = "automated_phishing_detection.study_series_cell_runner"
    assert find_spec(name) is not None, "missing bounded series cell runner"
    return import_module(name)


def test_only_private_series_composition_is_exposed():
    import inspect

    module = api()
    assert not hasattr(module, "run_series_cell")
    assert tuple(inspect.signature(module._run_series_cell).parameters) == (
        "public",
        "metadata_bytes",
        "internal_snapshot",
        "external_snapshot",
        "cell",
        "admissions",
    )


def test_real_retention_and_science_accept_only_after_holders_exit(
    tmp_path, series_case, monkeypatch
):
    from study_series_cell_runner_fixtures import (
        execute,
        inputs,
        install_observer,
        setup,
    )
    from study_series_ledger_fixtures import snapshot

    case = setup(tmp_path, series_case, monkeypatch)
    install_observer(case, monkeypatch)
    with inputs(case):
        result = execute(case)
    assert result.inputs.computational == case.selected.inputs.computational
    saved = snapshot(case.ledger)
    assert saved["cells"][0]["status"] == "accepted"
    assert saved["cells"][0]["holders_closed"] is True
    assert all(entry["accepted"] for entry in saved["admissions"])
    assert case.events == ["observe"]


def test_development_policy_rejects_before_creating_cell_outputs(
    tmp_path, series_case, monkeypatch
):
    from study_series_cell_runner_fixtures import execute, setup

    case = setup(tmp_path, series_case, monkeypatch, allow_candidate=False)
    with pytest.raises(ValueError):
        execute(case)
    assert not case.selected.attempt.directory.exists()
