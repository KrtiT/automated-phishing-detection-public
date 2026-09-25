"""Private invented output parents stay bound before and after study work."""

import importlib
import importlib.util
from dataclasses import fields, is_dataclass, replace
from pathlib import Path

import pytest
import study_run_context_fixtures as fixtures


def api():
    name = "automated_phishing_detection._study_run_paths"
    assert importlib.util.find_spec(name), "missing held study output paths"
    return importlib.import_module(name)


def relocate(value, root):
    if isinstance(value, Path):
        return root / value.relative_to("/invented")
    if is_dataclass(value):
        return replace(
            value,
            **{
                member.name: relocate(getattr(value, member.name), root)
                for member in fields(value)
            },
        )
    return value


def setup(tmp_path):
    paths = relocate(fixtures.paths(fixtures.api()), tmp_path)
    paths.cells_directory.mkdir(parents=True, mode=0o700)
    paths.cells_directory.parent.chmod(0o700)
    return paths


def test_output_holder_never_creates_or_reads_source_inputs(tmp_path):
    paths = setup(tmp_path)
    with api().hold_study_paths(paths) as held:
        held.check(())
        assert not paths.preparation.source_csv.exists()
        assert not paths.attempt.exists()
    assert list(paths.cells_directory.iterdir()) == []


@pytest.mark.parametrize(
    "name", ["attempt", "public_summary", "accepted_inputs_directory"]
)
def test_existing_root_outputs_reject_before_study_reservation(tmp_path, name):
    paths = setup(tmp_path)
    getattr(paths, name).write_bytes(b"existing")
    with pytest.raises(ValueError):
        with api().hold_study_paths(paths):
            pytest.fail("existing output admitted")
    assert getattr(paths, name).read_bytes() == b"existing"


@pytest.mark.parametrize("mode", [0o755, 0o777, 0o750])
def test_cell_parent_requires_private_mode(tmp_path, mode):
    paths = setup(tmp_path)
    paths.cells_directory.chmod(mode)
    with pytest.raises(ValueError):
        with api().hold_study_paths(paths):
            pytest.fail("nonprivate parent admitted")


def test_cell_parent_requires_empty_initial_inventory(tmp_path):
    paths = setup(tmp_path)
    (paths.cells_directory / "cell-001-attempt").mkdir(mode=0o700)
    with pytest.raises(ValueError):
        with api().hold_study_paths(paths):
            pytest.fail("historical attempt admitted")


def test_cell_parent_replacement_is_detected_after_body(tmp_path):
    paths = setup(tmp_path)
    with pytest.raises(ValueError):
        with api().hold_study_paths(paths):
            paths.cells_directory.rename(tmp_path / "old-cells")
            paths.cells_directory.mkdir(mode=0o700)


def test_unknown_cell_entry_rejects_without_deleting_anything(tmp_path):
    paths = setup(tmp_path)
    added = paths.cells_directory / "unplanned"
    with pytest.raises(ValueError):
        with api().hold_study_paths(paths):
            added.write_bytes(b"retain")
    assert added.read_bytes() == b"retain"


def test_allowed_partial_cell_inventory_is_not_a_completed_matrix(tmp_path):
    paths = setup(tmp_path)
    with api().hold_study_paths(paths) as held:
        (paths.cells_directory / "cell-001-attempt").mkdir(mode=0o700)
        held.check(("cell-001-attempt",))
        with pytest.raises(ValueError):
            held.check(())


def test_first_interruption_survives_late_parent_replacement(tmp_path):
    paths = setup(tmp_path)
    first = KeyboardInterrupt("first")
    with pytest.raises(KeyboardInterrupt) as caught:
        with api().hold_study_paths(paths):
            paths.cells_directory.rename(tmp_path / "old-cells")
            paths.cells_directory.mkdir(mode=0o700)
            raise first
    assert caught.value is first


def test_cell_paths_are_fixed_and_do_not_accept_arbitrary_ordinals(tmp_path):
    paths = setup(tmp_path)
    module = api()
    for ordinal in range(1, 126):
        cell = module.cell_paths(paths, ordinal)
        assert cell.accepted_inputs_directory == paths.accepted_inputs_directory
        assert cell.attempt == paths.cells_directory / f"cell-{ordinal:03d}-attempt"
        assert (
            cell.cell_input_directory
            == paths.cells_directory / f"cell-{ordinal:03d}-inputs"
        )
        assert (
            cell.public_summary
            == paths.cells_directory / f"cell-{ordinal:03d}-summary.json"
        )
    for invalid in (True, 0, 126, "1"):
        with pytest.raises(ValueError):
            module.cell_paths(paths, invalid)
