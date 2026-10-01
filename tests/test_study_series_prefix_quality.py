"""Independent quality probes for immutable declarations and exact fixed argv."""

import builtins
import os
from importlib import import_module
from pathlib import Path

import pytest
from study_series_prefix_fixtures import (
    api,
    candidates,
    imported,
    manifests,
    prefix_case,
    series_case,
)

from automated_phishing_detection import _study_history_cell_inputs as history
from automated_phishing_detection import _study_series_input_context as inputs

__all__ = ["candidates", "manifests", "prefix_case", "series_case"]


def forbidden(*arguments, **keywords):
    pytest.fail("prefix declaration invoked IO or repeated scientific restoration")


def test_all_declaration_builders_and_validator_stay_byte_only(
    prefix_case, monkeypatch
):
    case = prefix_case
    for owner, name in (
        (builtins, "open"),
        (os, "open"),
        (Path, "resolve"),
        (history, "sources"),
        (history, "restore"),
        (inputs, "authenticate"),
        (inputs, "build"),
    ):
        monkeypatch.setattr(owner, name, forbidden)
    assert imported(case) == case.imported
    assert (
        api().segment_intent_bytes(
            case.binding, case.series, case.segment, case.imported, case.metadata
        )
        == case.intent
    )
    assert (
        api().validate_series_child_prefix(case.binding, case.frame, case.payloads)
        is None
    )
    assert case.binding.authorizes_execution is False


@pytest.mark.parametrize(
    "executable",
    [
        "/invented/../venv/python",
        "/invented//venv/python",
        "/invented/./venv/python",
        "//invented/venv/python",
    ],
)
@pytest.mark.parametrize("role", ["service", "client"])
def test_fixed_command_preserves_exact_interpreter_without_resolving(
    prefix_case, monkeypatch, executable, role
):
    commands = import_module(
        "automated_phishing_detection._study_series_child_commands"
    )
    monkeypatch.setattr(commands.sys, "executable", executable)
    monkeypatch.setattr(Path, "resolve", forbidden)
    monkeypatch.setattr(os, "open", forbidden)
    result = commands.series_child_command(
        prefix_case.binding, role, cell_ordinal=125, cell_binding_sha256="3" * 64
    )
    assert type(result) is tuple and len(result) == 18
    assert result[0] == executable
    assert result[2:4] == ("--role", role)
    assert result[-4:] == (
        "--cell-ordinal",
        "125",
        "--expected-binding-sha256",
        "3" * 64,
    )
    assert "--workload" not in result and "--model" not in result
