"""Independent selected-file lifecycle checks with explicitly opaque IO fixtures."""

from dataclasses import replace
from importlib import import_module
from pathlib import Path

import pytest
from study_series_child_prefix_fixtures import api, hold, setup
from study_series_execution_fixtures import bind, public_case

from automated_phishing_detection import _study_child_root as legacy

__all__ = ["public_case"]


def forbidden(*arguments, **keywords):
    pytest.fail("series prefix enumerated a growing root or invoked legacy semantics")


def test_root_growth_never_enumerates_inventory_or_uses_legacy_validator(
    tmp_path, monkeypatch
):
    case = setup(tmp_path, monkeypatch)
    with monkeypatch.context() as guard:
        guard.setattr(api().os, "listdir", forbidden)
        guard.setattr(api().os, "scandir", forbidden)
        guard.setattr(api().files, "check", forbidden)
        guard.setattr(legacy, "validate_child_root", forbidden)
        with hold(case) as payloads:
            (case.paths["series"] / "unrelated-accounting.json").write_bytes(b"later")
            (case.paths["segment"] / "later-evidence").mkdir(mode=0o700)
            assert dict(payloads) == case.values
    assert len(case.calls) == 1


@pytest.mark.parametrize("kind", ("series", "segment"))
def test_replaced_root_directory_rejects_even_if_held_file_bytes_survive(
    tmp_path, monkeypatch, kind
):
    case = setup(tmp_path, monkeypatch)
    selected = case.paths[kind]
    with pytest.raises(api().SeriesChildPrefixError):
        with hold(case):
            selected.rename(tmp_path / f"retained-{kind}")
            selected.mkdir(mode=0o700)


def test_cleanup_rejection_retains_original_process_failure_context(
    tmp_path, monkeypatch
):
    case = setup(tmp_path, monkeypatch)
    original = RuntimeError("invented private caller failure")
    original.progress = b"retained process progress"
    original.operational_failure = object()
    with pytest.raises(api().SeriesChildPrefixError) as caught:
        with hold(case):
            (case.paths["segment"] / "history-import.json").write_bytes(b"changed")
            raise original
    assert caught.value.progress is original.progress
    assert caught.value.operational_failure is original.operational_failure
    assert str(caught.value) == "invalid_series_child_prefix"
    assert caught.value.__suppress_context__ is True


def test_system_exit_survives_selected_file_cleanup_failure(tmp_path, monkeypatch):
    case = setup(tmp_path, monkeypatch)
    original = SystemExit(19)
    with pytest.raises(SystemExit) as caught:
        with hold(case):
            (case.paths["series"] / "reservation.json").write_bytes(b"changed")
            raise original
    assert caught.value is original


@pytest.mark.parametrize(
    "field,value",
    (
        ("root", Path("/invented/other-root")),
        ("revision", "c" * 40),
        ("contract_sha256", "c" * 64),
    ),
)
def test_child_command_rejects_relabelled_current_base(public_case, field, value):
    commands = import_module(
        "automated_phishing_detection._study_series_child_commands"
    )
    original = bind(public_case)
    changed = replace(original, base=replace(original.base, **{field: value}))
    with pytest.raises(ValueError, match="^invalid_series_child_command$"):
        commands.series_child_command(
            changed, "service", cell_ordinal=73, cell_binding_sha256="3" * 64
        )
