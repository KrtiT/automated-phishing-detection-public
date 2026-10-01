"""Independent pre-IO context and exact held-source boundary checks."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from study_series_cell_runner_fixtures import (
    candidates,
    execute,
    inputs,
    manifests,
    series_case,
    setup,
)
from study_series_ledger_fixtures import snapshot
from test_study_series_cell_runner import api

from automated_phishing_detection.operational_schedule import cell_for_ordinal

__all__ = ["candidates", "manifests", "series_case"]


def forbidden(*args, **kwargs):
    pytest.fail("invalid context reached output or retained-input IO")


def changed_context(case, change):
    if change == "ledger_type":
        return {"admissions": SimpleNamespace()}
    if change == "public_object":
        return {"public": replace(case.prefix.binding)}
    if change == "metadata":
        return {"metadata_bytes": case.prefix.metadata + b" "}
    if change == "next_cell":
        return {"cell": cell_for_ordinal(case.selected.cell.ordinal + 1)}
    if change == "parent_pid":
        case.ledger._parent_pid += 1
    elif change == "closed":
        case.ledger._closed = True
    elif change == "active_cell":
        case.ledger.start_cell(
            case.selected.cell,
            case.selected.attempt,
            case.selected.selected.descriptor_bytes,
            case.selected.binding,
        )
    return {}


@pytest.mark.parametrize(
    "change",
    (
        "ledger_type",
        "public_object",
        "metadata",
        "next_cell",
        "parent_pid",
        "closed",
        "active_cell",
    ),
)
def test_invalid_association_rejects_before_any_private_io_or_ledger_stop(
    tmp_path, series_case, monkeypatch, change
):
    case = setup(tmp_path, series_case, monkeypatch)
    changes = changed_context(case, change)
    monkeypatch.setattr(api().context.original, "absent_outputs", forbidden)
    monkeypatch.setattr(api().context.storage, "hold", forbidden)
    monkeypatch.setattr(api().SeriesCellProgress, "reserve", forbidden)
    monkeypatch.setattr(case.ledger, "stop_cell", forbidden)
    with pytest.raises(ValueError):
        execute(case, **changes)
    assert not case.selected.attempt.directory.exists()


def test_initial_public_recheck_rejects_before_output_scan_and_retains_validation_stop(
    tmp_path, series_case, monkeypatch
):
    case = setup(tmp_path, series_case, monkeypatch)

    def changed_public(public):
        raise ValueError("invented changed public binding")

    monkeypatch.setattr(api(), "recheck_series_public_execution", changed_public)
    monkeypatch.setattr(api().context.original, "absent_outputs", forbidden)
    monkeypatch.setattr(api().context.storage, "hold", forbidden)
    with pytest.raises(ValueError):
        execute(case)
    saved = snapshot(case.ledger)
    assert saved["admissions"] == []
    assert saved["cells"][0]["stage"] == "validation"
    assert saved["cells"][0]["reservation_sha256"] is None
    assert all(cell["status"] == "unattempted" for cell in saved["cells"][1:])


def test_existing_attempt_is_preserved_without_selection_or_reservation(
    tmp_path, series_case, monkeypatch
):
    case = setup(tmp_path, series_case, monkeypatch)
    case.selected.attempt.directory.mkdir(mode=0o700)
    marker = case.selected.attempt.directory / "existing"
    marker.write_bytes(b"invented retained previous attempt")
    monkeypatch.setattr(api().context, "select", forbidden)
    monkeypatch.setattr(api().SeriesCellProgress, "reserve", forbidden)
    with pytest.raises(ValueError):
        execute(case)
    assert marker.read_bytes() == b"invented retained previous attempt"
    assert snapshot(case.ledger)["cells"][0]["reservation_sha256"] is None


@pytest.mark.parametrize(
    "name", ("accepted-inputs.json", "descriptor.json", "binding.json", "manifest")
)
def test_every_held_buffer_must_equal_selected_inputs_before_science_and_observation(
    tmp_path, series_case, monkeypatch, name
):
    case = setup(tmp_path, series_case, monkeypatch)
    original = api().context.transport._contents

    def swapped(*args):
        return original(*args) | {name: b"invented substituted bytes"}

    monkeypatch.setattr(api().context.transport, "_contents", swapped)
    monkeypatch.setattr(api().context, "restore_series_cell_inputs", forbidden)
    monkeypatch.setattr(api(), "observe_series_operational_children", forbidden)
    with inputs(case), pytest.raises(ValueError):
        execute(case)
    saved = snapshot(case.ledger)
    assert saved["admissions"] == []
    assert saved["cells"][0]["stage"] == "input_retention"
    assert (
        saved["cells"][0]["reservation_sha256"]
        == case.selected.attempt.reservation_sha256
    )
    directory = Path(case.prefix.source.profile["paths"]["cells_dir"])
    assert sorted(path.name for path in directory.iterdir()) == [
        "cell-073-attempt",
        "cell-073-inputs",
    ]
