"""Explicit prepared-only worker options never impersonate original sources."""

import importlib.util
from pathlib import Path

import pytest
from test_external_source_cli import OPTION_NAMES

NAMES = tuple(
    name for name in OPTION_NAMES if name not in ("archive", "suffix-rules")
) + (
    "preparation",
    "expected-preparation-reservation-sha256",
    "expected-preparation-completion-sha256",
)


@pytest.fixture
def cli():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts/run_prepared_external_evaluation.py"
    )
    assert path.is_file(), "missing prepared-only external CLI"
    spec = importlib.util.spec_from_file_location("prepared_external_cli_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def arguments():
    return [value for name in NAMES for value in (f"--{name}", f"invented/{name}")]


def test_exact_required_options(cli):
    actions = cli.parser()._actions[1:]
    assert {action.option_strings[0] for action in actions} == {
        f"--{name}" for name in NAMES
    }
    assert all(action.required for action in actions)


@pytest.mark.parametrize("name", NAMES)
def test_each_option_is_required(cli, name):
    values = arguments()
    position = values.index(f"--{name}")
    del values[position : position + 2]
    with pytest.raises(SystemExit) as caught:
        cli.parser().parse_args(values)
    assert caught.value.code == 2


@pytest.mark.parametrize(
    "name",
    [
        "archive",
        "suffix-rules",
        "source-csv",
        "resume",
        "override",
        "expected-preparation-completion",
    ],
)
def test_original_override_and_abbreviated_options_are_rejected(cli, name):
    with pytest.raises(SystemExit) as caught:
        cli.parser().parse_args(arguments() + [f"--{name}", "invented"])
    assert caught.value.code == 2


def test_dispatch_is_once_and_preserves_exact_preparation_expectations(
    cli, monkeypatch
):
    calls = []
    monkeypatch.setattr(
        cli,
        "run_prepared_external_evaluation",
        lambda *args, **kwargs: calls.append(kwargs),
    )
    assert cli.main(arguments()) == 0
    assert len(calls) == 1
    assert calls[0]["paths"].preparation == Path("invented/preparation")
    assert (
        calls[0]["expected_preparation_reservation_sha256"]
        == "invented/expected-preparation-reservation-sha256"
    )
    assert (
        calls[0]["expected_preparation_completion_sha256"]
        == "invented/expected-preparation-completion-sha256"
    )
