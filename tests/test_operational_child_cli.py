"""Fixed child commands expose no workload, endpoint, readiness or retry bypass."""

import importlib.util
from pathlib import Path

import pytest

from automated_phishing_detection._operational_cell_protocol import (
    COMMON_ARGUMENTS,
    SERVICE_ARGUMENTS,
)

ROOT = Path(__file__).resolve().parents[1]


def script(role):
    path = ROOT / "scripts" / f"run_operational_{role}.py"
    assert path.is_file(), "missing fixed operational child script"
    spec = importlib.util.spec_from_file_location(f"operational_{role}_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def arguments(role):
    names = SERVICE_ARGUMENTS if role == "service" else COMMON_ARGUMENTS
    return [
        value
        for name in names
        for value in (
            name,
            "/invented/value" if name.endswith(("dir", "root")) else "value",
        )
    ]


@pytest.mark.parametrize("role", ["service", "client"])
def test_cli_preserves_exact_fixed_argument_inventory(role):
    api = script(role)
    actual = {
        name for action in api.parser()._actions for name in action.option_strings
    }
    expected = set(SERVICE_ARGUMENTS if role == "service" else COMMON_ARGUMENTS)
    assert actual == expected | {"-h", "--help"}
    assert api.parser().allow_abbrev is False


@pytest.mark.parametrize("role", ["service", "client"])
@pytest.mark.parametrize(
    "extra",
    [
        "--worker",
        "--workload",
        "--base-url",
        "--timeout",
        "--ready",
        "--resume",
        "--expected-rev",
    ],
)
def test_cli_rejects_every_unapproved_override(role, extra):
    with pytest.raises(SystemExit) as caught:
        script(role).parser().parse_args(arguments(role) + [extra, "value"])
    assert caught.value.code == 2


@pytest.mark.parametrize("role", ["service", "client"])
def test_cli_calls_only_fixed_runner_and_redacts_failure(role, monkeypatch, capsys):
    api, calls = script(role), []

    async def failed(root, **kwargs):
        calls.append((root, kwargs))
        raise OSError("/private/secret")

    monkeypatch.setattr(api, f"run_operational_{role}", failed)
    assert api.main(arguments(role)) == 2
    assert len(calls) == 1
    assert "/private" not in capsys.readouterr().err
