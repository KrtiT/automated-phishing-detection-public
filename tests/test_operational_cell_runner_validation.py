"""Reject unscheduled, aliased or incompletely specified cells before reservation."""

from dataclasses import replace

import pytest
from operational_cell_runner_fixtures import execute, setup
from operational_input_fixtures import candidates, manifests

__all__ = ["candidates", "manifests"]


@pytest.mark.parametrize("change", ["missing", "extra", "list", "overflow", "nan"])
def test_closed_deadline_mapping(tmp_path, manifests, monkeypatch, change):
    case = setup(tmp_path, manifests, monkeypatch)
    deadlines = case.deadlines.copy()
    if change == "missing":
        deadlines.pop("kill")
    elif change == "extra":
        deadlines["retry"] = 1
    elif change == "list":
        deadlines = list(deadlines.items())
    else:
        deadlines["startup"] = 10**1000 if change == "overflow" else float("nan")
    with pytest.raises(case.module.OperationalCellExecutionError):
        execute(case, deadlines=deadlines)
    assert not case.paths.attempt.exists()


@pytest.mark.parametrize("change", ["same", "nested", "checkout", "artifacts"])
def test_paths_must_not_alias_private_roles(tmp_path, manifests, monkeypatch, change):
    case = setup(tmp_path, manifests, monkeypatch)
    bad = {
        "same": case.paths.accepted_inputs_directory,
        "nested": case.paths.accepted_inputs_directory / "child",
        "checkout": case.binding.root / "private",
        "artifacts": case.artifacts.transformer_bundle / "child",
    }[change]
    with pytest.raises(case.module.OperationalCellExecutionError):
        execute(case, paths=replace(case.paths, cell_input_directory=bad))
    assert not case.paths.attempt.exists()


@pytest.mark.parametrize(
    "member", ["cell_input_directory", "attempt", "public_summary"]
)
def test_existing_outputs_are_never_reused(tmp_path, manifests, monkeypatch, member):
    case = setup(tmp_path, manifests, monkeypatch)
    destination = getattr(case.paths, member)
    destination.write_bytes(b"existing")
    with pytest.raises(case.module.OperationalCellExecutionError):
        execute(case)
    assert destination.read_bytes() == b"existing"
    assert "observe" not in case.events


@pytest.mark.parametrize(
    "change",
    ["ordinal", "concurrency", "run_index", "workload", "prevalence_basis_points"],
)
def test_cell_must_be_exact_fixed_schedule(tmp_path, manifests, monkeypatch, change):
    case = setup(tmp_path, manifests, monkeypatch)
    wrong = {
        "ordinal": True,
        "concurrency": 64,
        "run_index": 2,
        "workload": "fixed_cascade",
        "prevalence_basis_points": 100,
    }
    case.cell = replace(case.cell, **{change: wrong[change]})
    with pytest.raises(case.module.OperationalCellExecutionError):
        execute(case)
    assert not case.paths.attempt.exists()


@pytest.mark.parametrize("mutable", [False, True])
def test_profile_is_exact_frozen_projection(tmp_path, manifests, monkeypatch, mutable):
    case = setup(tmp_path, manifests, monkeypatch)
    content = bytearray(case.profile.canonical_bytes) if mutable else b"{}\n"
    case.profile = replace(case.profile, canonical_bytes=content)
    with pytest.raises(case.module.OperationalCellExecutionError):
        execute(case)
    assert not case.paths.attempt.exists()
