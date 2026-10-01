from importlib import import_module
from importlib.util import find_spec

import pytest
from study_series_execution_fixtures import bind, public_case

__all__ = ["public_case"]


def api():
    name = "automated_phishing_detection._study_series_child_commands"
    assert find_spec(name), "missing fixed series child commands"
    return import_module(name)


@pytest.mark.parametrize("role", ("service", "client"))
def test_exact_series_command_has_no_path_or_workload_overrides(public_case, role):
    binding = bind(public_case)
    result = api().series_child_command(
        binding, role, cell_ordinal=73, cell_binding_sha256="3" * 64
    )
    assert result == (
        api().sys.executable,
        str(binding.base.root / "scripts/run_study_series_child.py"),
        "--role",
        role,
        "--repo-root",
        str(binding.base.root),
        "--expected-revision",
        binding.base.revision,
        "--expected-profile-sha256",
        binding.profile_sha256,
        "--envelope",
        str(binding.envelope_path),
        "--expected-envelope-sha256",
        binding.envelope_sha256,
        "--cell-ordinal",
        "73",
        "--expected-binding-sha256",
        "3" * 64,
    )
    assert binding.authorizes_execution is False


@pytest.mark.parametrize("ordinal", (True, 1, 72, 126, "73", 73.0, None))
def test_invalid_or_historical_ordinal_rejects(public_case, ordinal):
    with pytest.raises(ValueError):
        api().series_child_command(
            bind(public_case),
            "service",
            cell_ordinal=ordinal,
            cell_binding_sha256="3" * 64,
        )


@pytest.mark.parametrize("role", ("internal", "external", "", None, True))
def test_no_source_or_unsupported_role(public_case, role):
    with pytest.raises(ValueError):
        api().series_child_command(
            bind(public_case), role, cell_ordinal=73, cell_binding_sha256="3" * 64
        )


def test_actual_absolute_interpreter_spelling_is_preserved(public_case, monkeypatch):
    binding = bind(public_case)
    executable = "/invented/../environment/bin/python"
    monkeypatch.setattr(api().sys, "executable", executable)
    command = api().series_child_command(
        binding, "service", cell_ordinal=73, cell_binding_sha256="3" * 64
    )
    assert command[0] == executable


@pytest.mark.parametrize(
    "executable", ("", "relative/python", "/bad\0python", None, True)
)
def test_invalid_interpreter_rejects(public_case, monkeypatch, executable):
    binding = bind(public_case)
    monkeypatch.setattr(api().sys, "executable", executable)
    with pytest.raises(ValueError):
        api().series_child_command(
            binding, "service", cell_ordinal=73, cell_binding_sha256="3" * 64
        )
