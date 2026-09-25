"""Fixed argv construction is pure and does not admit the closed candidate."""

import importlib
import importlib.util
import inspect
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from operational_profile_fixtures import COMMON, profile_case, resolve

from automated_phishing_detection._checkpoint_codec import canonical_bytes
from automated_phishing_detection.bound_models import ArtifactPaths

__all__ = ["profile_case"]

ARTIFACT_ARGUMENTS = (
    "--length-only",
    "--logistic-l1",
    "--transformer-bundle",
    "--gmm",
)
ARTIFACT_VALUES = (
    "/invented-private/length",
    "/invented-private/logistic",
    "/invented-private/transformer",
    "/invented-private/gmm",
)


def api():
    name = "automated_phishing_detection._operational_cell_commands"
    assert importlib.util.find_spec(name), "missing fixed operational command builder"
    return importlib.import_module(name)


def options():
    root = Path("/invented-private")
    return {
        "accepted_inputs_directory": root / "accepted-inputs",
        "cell_input_directory": root / "cell-inputs/cell-001",
        "expected_binding_sha256": "d" * 64,
        "artifacts": ArtifactPaths(
            *(root / name for name in ("length", "logistic", "transformer", "gmm"))
        ),
    }


def test_fixed_commands_use_actual_python_and_bound_scripts(profile_case):
    profile, arguments = resolve(profile_case), options()
    service, client = api().build_cell_commands(
        profile_case.binding, profile, **arguments
    )
    assert client[0] == service[0] == sys.executable
    assert client[1] == str(
        profile_case.binding.root / "scripts/run_operational_client.py"
    )
    assert service[1] == str(
        profile_case.binding.root / "scripts/run_operational_service.py"
    )
    assert client[2::2] == tuple(COMMON)
    assert service[2::2] == (*COMMON, *ARTIFACT_ARGUMENTS)
    common = expected_common(profile_case, profile, arguments)
    assert client[3::2] == common
    assert service[3::2] == (*common, *ARTIFACT_VALUES)
    assert profile.protected_evaluation_ready is False


def expected_common(case, profile, arguments):
    return (
        str(case.binding.root),
        "a" * 40,
        "b" * 64,
        profile.profile_sha256,
        str(arguments["accepted_inputs_directory"]),
        str(arguments["cell_input_directory"]),
        "d" * 64,
    )


@pytest.mark.parametrize(
    "member", ["accepted_inputs_directory", "cell_input_directory"]
)
@pytest.mark.parametrize(
    "bad", ["/private", Path("relative"), Path("/with/../parent"), Path("/nul\0")]
)
def test_bad_input_path_rejects_before_io(profile_case, member, bad):
    arguments = options() | {member: bad}
    with pytest.raises(api().OperationalCommandError):
        api().build_cell_commands(
            profile_case.binding, resolve(profile_case), **arguments
        )


@pytest.mark.parametrize(
    "member", ["length_only", "logistic_l1", "transformer_bundle", "gmm"]
)
def test_each_artifact_path_requires_absolute_path(profile_case, member):
    arguments = options()
    arguments["artifacts"] = replace(
        arguments["artifacts"], **{member: Path("relative")}
    )
    with pytest.raises(api().OperationalCommandError):
        api().build_cell_commands(
            profile_case.binding, resolve(profile_case), **arguments
        )


@pytest.mark.parametrize("bad", [True, "F" * 64, "short", None])
def test_binding_pin_is_exact_digest(profile_case, bad):
    arguments = options() | {"expected_binding_sha256": bad}
    with pytest.raises(api().OperationalCommandError):
        api().build_cell_commands(
            profile_case.binding, resolve(profile_case), **arguments
        )


@pytest.mark.parametrize(
    "field", ["execution", "commands", "protective_deadlines_seconds"]
)
def test_caller_rewritten_profile_cannot_change_argv(profile_case, field):
    profile = resolve(profile_case)
    value = profile.projection() | {field: {}}
    altered = replace(profile, canonical_bytes=canonical_bytes(value))
    with pytest.raises(api().OperationalCommandError):
        api().build_cell_commands(profile_case.binding, altered, **options())


def test_command_builder_exposes_no_execution_overrides(profile_case, monkeypatch):
    import subprocess

    profile = resolve(profile_case)

    def forbidden(*arguments, **keywords):
        pytest.fail("pure command builder performed I/O or launched work")

    for owner, name in ((Path, "open"), (Path, "resolve"), (subprocess, "Popen")):
        monkeypatch.setattr(owner, name, forbidden)
    api().build_cell_commands(profile_case.binding, profile, **options())
    assert tuple(inspect.signature(api().build_cell_commands).parameters) == (
        "binding",
        "profile",
        "accepted_inputs_directory",
        "cell_input_directory",
        "expected_binding_sha256",
        "artifacts",
    )


def test_mutable_profile_buffer_is_not_a_fixed_command_identity(profile_case):
    profile = resolve(profile_case)
    altered = replace(profile, canonical_bytes=bytearray(profile.canonical_bytes))
    with pytest.raises(api().OperationalCommandError):
        api().build_cell_commands(profile_case.binding, altered, **options())
