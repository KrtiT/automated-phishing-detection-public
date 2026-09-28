"""Fixed adopted-child argv is metadata, not permission to consume inputs."""

import importlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def api():
    name = "automated_phishing_detection._study_child_commands"
    assert importlib.util.find_spec(name), "missing fixed study child commands"
    return importlib.import_module(name)


def authorization():
    return SimpleNamespace(
        base=SimpleNamespace(root=Path("/invented/repo"), revision="a" * 40),
        envelope_path=Path("/invented/private/approval.json"),
        envelope_sha256="b" * 64,
    )


def test_internal_command_has_only_fixed_review_identity():
    module = api()
    command = module.internal_command(authorization())
    assert command == (
        sys.executable,
        "/invented/repo/scripts/run_study_child.py",
        "--role",
        "internal",
        "--repo-root",
        "/invented/repo",
        "--expected-revision",
        "a" * 40,
        "--envelope",
        "/invented/private/approval.json",
        "--expected-envelope-sha256",
        "b" * 64,
    )
    assert not any("admission" in argument for argument in command)


def test_external_command_adds_only_actual_parent_handoff_transport():
    module = api()
    transport = SimpleNamespace(
        directory=Path("/invented/private/handoff"),
        expected_handoff_sha256="c" * 64,
    )
    command = module.external_command(authorization(), transport)
    assert command[3] == "external"
    assert command[-4:] == (
        "--internal-transport",
        str(transport.directory),
        "--expected-handoff-sha256",
        "c" * 64,
    )


@pytest.mark.parametrize("role", ["service", "client"])
def test_cell_command_cannot_override_models_or_workload(role):
    command = api().cell_command(authorization(), role, 125, "d" * 64)
    assert command[3] == role
    assert command[-4:] == (
        "--cell-ordinal",
        "125",
        "--expected-binding-sha256",
        "d" * 64,
    )
    for option in ("--length-only", "--deadline", "--resume", "--command"):
        assert option not in command


@pytest.mark.parametrize("ordinal", [0, 126, True, 1.0, "1"])
def test_invalid_cell_ordinal_rejects(ordinal):
    with pytest.raises(ValueError):
        api().cell_command(authorization(), "service", ordinal, "d" * 64)


@pytest.mark.parametrize("role", ["internal", "external", "worker", ""])
def test_other_roles_cannot_use_cell_command(role):
    with pytest.raises(ValueError):
        api().cell_command(authorization(), role, 1, "d" * 64)


@pytest.mark.parametrize("digest", [None, "", "D" * 64, "a" * 63])
def test_bad_binding_hash_rejects(digest):
    with pytest.raises(ValueError):
        api().cell_command(authorization(), "client", 1, digest)


@pytest.mark.parametrize("value", [Path("relative"), Path("/invented/../bad")])
def test_ambiguous_envelope_path_rejects(value):
    reviewed = authorization()
    reviewed.envelope_path = value
    with pytest.raises(ValueError):
        api().internal_command(reviewed)
