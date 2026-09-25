"""Saved acceptance boundary on invented bytes; process lineage is a unit seam."""

import importlib
import importlib.util
from collections import Counter
from dataclasses import asdict, replace

import pytest

from automated_phishing_detection import source_runner
from automated_phishing_detection._owned_process_exit import OwnedProcessExit
from automated_phishing_detection._process_support import command_hash
from automated_phishing_detection.owned_worker import WorkerObservation


def module():
    name = "automated_phishing_detection.external_source_completion"
    assert importlib.util.find_spec(name), "missing observed external completion"
    return importlib.import_module(name)


def test_observed_external_completion_api_exists():
    module()


def verify(api, case, **changes):
    arguments = {
        "expected_handoff": case.handoff,
        "worker": case.worker,
        "command": case.command,
    }
    return api.verify_external_completion_snapshot(
        case.binding, case.paths, **(arguments | changes)
    )


def bind_fixture(api, case, monkeypatch):
    events = []
    monkeypatch.setattr(
        api, "resolve_external_source_profile", lambda unused: case.profile
    )
    monkeypatch.setattr(api, "recheck_binding", lambda unused: events.append("recheck"))
    return events


def _exit_mutations():
    return {
        "unknown": OwnedProcessExit(123, False, None),
        "nonzero": OwnedProcessExit(123, True, 2),
        "signal": OwnedProcessExit(123, True, -9),
        "boolean_zero": OwnedProcessExit(123, True, False),
        "float_zero": OwnedProcessExit(123, True, 0.0),
        "bad_pid": OwnedProcessExit(0, True, 0),
        "boolean_pid": OwnedProcessExit(True, True, 0),
    }


@pytest.mark.parametrize("count", [0, 1, 255, 256, 319, 320])
def test_reconstructs_provenance_science_and_outer_links_once(
    tmp_path, monkeypatch, count
):
    from external_completion_fixtures import external_completion_case

    api = module()
    case = external_completion_case(tmp_path, monkeypatch, count)
    events = bind_fixture(api, case, monkeypatch)
    original = source_runner._read_file_once
    reads = Counter()

    def counted(filename, **options):
        assert filename == case.paths.public_summary or filename.is_relative_to(
            case.paths.attempt
        )
        reads[filename] += 1
        return original(filename, **options)

    monkeypatch.setattr(source_runner, "_read_file_once", counted)
    result = verify(api, case)
    assert asdict(result.replay) == asdict(case.produced.replay)
    assert result.public_summary == case.expectedpublic
    assert result.profile_bytes == case.profile.canonical_bytes
    assert len(result.payloads) == len(reads) == 76
    assert set(reads.values()) == {1}
    assert events == ["recheck"]


@pytest.mark.parametrize(
    "change",
    [
        "unknown",
        "nonzero",
        "signal",
        "boolean_zero",
        "float_zero",
        "wrong_command",
        "bad_pid",
        "boolean_pid",
        "bad_stdout",
        "bad_stderr",
    ],
)
def test_worker_rejection_precedes_binding_and_any_paths(monkeypatch, change):
    api = module()
    command = ("invented", "worker")
    worker = WorkerObservation(
        command_hash(command), OwnedProcessExit(123, True, 0), "a" * 64, "b" * 64
    )
    exits = _exit_mutations()
    if change in exits:
        worker = replace(worker, exit=exits[change])
    else:
        field = {
            "wrong_command": "command_sha256",
            "bad_stdout": "stdout_sha256",
            "bad_stderr": "stderr_sha256",
        }[change]
        worker = replace(worker, **{field: "invalid"})

    def forbidden(*args, **kwargs):
        pytest.fail("invalid worker reached binding or file access")

    monkeypatch.setattr(api, "resolve_external_source_profile", forbidden)
    monkeypatch.setattr(source_runner, "_read_file_once", forbidden)
    with pytest.raises(api.ExternalCompletionVerificationError):
        api.verify_external_completion_snapshot(
            None, None, expected_handoff=None, worker=worker, command=command
        )
