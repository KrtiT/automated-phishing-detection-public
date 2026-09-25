"""Private orchestration checks do not establish real process execution."""

import inspect
import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest
from operational_cell_runner_fixtures import api, execute, orchestration, setup
from operational_input_fixtures import candidates, manifests

__all__ = ["candidates", "manifests"]


def test_one_reservation_observation_and_completion_with_original_bytes(
    tmp_path, manifests, monkeypatch
):
    case = setup(tmp_path, manifests, monkeypatch)
    orchestration(case, monkeypatch)
    result = execute(case)
    assert result.observation is case.observation and result.snapshot is case.snapshot
    assert case.events == [
        "recheck",
        "holder_enter",
        "observe",
        "complete",
        "holder_exit",
        "recheck",
    ]
    assert case.inputs.accepted_bytes == case.accepted.metadata_bytes
    reservation = json.loads((case.paths.attempt / "reservation.json").read_bytes())
    assert reservation["identity"] == case.identity
    assert set(item.name for item in case.paths.cell_input_directory.iterdir()) == {
        "descriptor.json",
        "binding.json",
        "manifest",
    }
    with pytest.raises(FrozenInstanceError):
        result.observation = None
    with pytest.raises(Exception):
        execute(case)
    assert case.events.count("observe") == 1


@pytest.mark.parametrize("member", ["startup", "shutdown", "terminate", "kill"])
@pytest.mark.parametrize("bad", [None, False, 0, -1, float("inf"), "1"])
def test_deadlines_reject_before_reservation(
    tmp_path, manifests, monkeypatch, member, bad
):
    case = setup(tmp_path, manifests, monkeypatch)
    with pytest.raises(api().OperationalCellExecutionError):
        execute(case, deadlines=case.deadlines | {member: bad})
    assert (
        not case.paths.attempt.exists() and not case.paths.cell_input_directory.exists()
    )


@pytest.mark.parametrize(
    "member", ["cell_input_directory", "attempt", "public_summary"]
)
@pytest.mark.parametrize("bad", [Path("relative"), Path("/with/../parent"), "/string"])
def test_output_paths_are_strict_before_reservation(
    tmp_path, manifests, monkeypatch, member, bad
):
    case = setup(tmp_path, manifests, monkeypatch)
    with pytest.raises(api().OperationalCellExecutionError):
        execute(case, paths=replace(case.paths, **{member: bad}))
    assert not case.paths.attempt.exists()


@pytest.mark.parametrize("field", ["execution", "operational_profile_sha256"])
def test_accepted_context_mismatch_never_reserves(
    tmp_path, manifests, monkeypatch, field
):
    case = setup(tmp_path, manifests, monkeypatch)
    from automated_phishing_detection._checkpoint_codec import canonical_bytes

    value = json.loads(case.accepted.metadata_bytes)
    value[field] = {} if field == "execution" else "f" * 64
    case.accepted = replace(case.accepted, metadata_bytes=canonical_bytes(value))
    with pytest.raises(api().OperationalCellExecutionError):
        execute(case)
    assert not case.paths.attempt.exists()


def test_no_public_independent_cell_or_command_override():
    module = api()
    assert not hasattr(module, "run_operational_cell")
    assert tuple(inspect.signature(module._run_bound_cell).parameters) == (
        "binding",
        "profile",
        "accepted",
        "cell",
        "paths",
        "artifacts",
        "deadlines",
    )
