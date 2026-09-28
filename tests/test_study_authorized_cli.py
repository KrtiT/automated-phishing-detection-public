"""Help and parsing never authorize a study or a standalone child."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def execute(script, arguments):
    target = ROOT / "scripts" / script
    assert target.is_file(), "missing study-only CLI"
    return subprocess.run(
        [sys.executable, str(target), *arguments],
        env=os.environ | {"PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


@pytest.mark.parametrize("script", ["run_adopted_study.py", "run_study_child.py"])
def test_help_has_no_path_deadline_command_or_resume_override(script):
    result = execute(script, ["--help"])
    assert result.returncode == 0 and result.stderr == ""
    for name in ("--repo-root", "--expected-revision", "--envelope"):
        assert name in result.stdout
    assert "--expected-envelope-sha256" in result.stdout
    for name in ("--source-csv", "--length-only", "--deadline", "--resume", "--worker"):
        assert name not in result.stdout


@pytest.mark.parametrize("script", ["run_adopted_study.py", "run_study_child.py"])
def test_help_does_not_import_model_or_execution_modules(script):
    target = ROOT / "scripts" / script
    assert target.is_file(), "missing study-only CLI"
    program = """
import runpy
import sys
sys.argv = [sys.argv[1], '--help']
try:
    runpy.run_path(sys.argv[0], run_name='__main__')
except SystemExit as error:
    assert error.code == 0
else:
    raise AssertionError('help did not exit')
assert 'torch' not in sys.modules
assert 'automated_phishing_detection.study_execution' not in sys.modules
assert 'automated_phishing_detection.adopted_study_runner' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", program, str(target)],
        env=os.environ | {"PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def identity_arguments():
    return [
        "--repo-root",
        "/invented/repo",
        "--expected-revision",
        "a" * 40,
        "--envelope",
        "/invented/approval.json",
        "--expected-envelope-sha256",
        "b" * 64,
    ]


@pytest.mark.parametrize("role", ["internal", "external", "service", "client"])
def test_roles_cannot_accept_resume_override(role):
    result = execute(
        "run_study_child.py", ["--role", role, *identity_arguments(), "--resume"]
    )
    assert result.returncode == 2
    assert "Study child completed" not in result.stdout


def test_internal_rejects_external_handoff_options():
    result = execute(
        "run_study_child.py",
        ["--role", "internal", *identity_arguments(), "--internal-transport", "/bad"],
    )
    assert result.returncode == 2 and "unrecognized arguments" in result.stderr


def test_external_requires_transport_and_digest():
    result = execute(
        "run_study_child.py", ["--role", "external", *identity_arguments()]
    )
    assert result.returncode == 2 and "required" in result.stderr


@pytest.mark.parametrize("role", ["service", "client"])
def test_operational_role_requires_exact_cell_identity(role):
    result = execute("run_study_child.py", ["--role", role, *identity_arguments()])
    assert result.returncode == 2 and "required" in result.stderr
