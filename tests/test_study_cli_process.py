"""Actual entry processes use invented closed bindings, never protected sources."""

import os
import subprocess
import sys
from pathlib import Path

from study_cli_fixtures import NAMES, SCRIPT, arguments


def environment():
    return os.environ | {"PYTHONPATH": str(SCRIPT.parents[1] / "src")}


def test_actual_script_help_requires_no_runner_or_model_imports():
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        env=environment(),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0 and completed.stderr == ""
    for name in NAMES:
        assert f"--{name}" in completed.stdout
    assert "--deadline" not in completed.stdout and "--resume" not in completed.stdout


def test_help_finishes_before_scientific_dependencies_are_imported():
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
assert 'automated_phishing_detection.study_runner' not in sys.modules
assert 'automated_phishing_detection.bound_models' not in sys.modules
assert 'torch' not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", program, str(SCRIPT)],
        env=environment(),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def closed_program():
    return """
import runpy
import sys
from types import SimpleNamespace
from automated_phishing_detection import study_runner
def closed(*arguments, **keywords):
    return SimpleNamespace(protected_evaluation_ready=False)
def forbidden(*arguments, **keywords):
    raise AssertionError('private execution reached')
study_runner.bind_execution = closed
study_runner._run_bound_study = forbidden
study_runner.resolve_external_source_profile = forbidden
study_runner.resolve_operational_profile = forbidden
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
"""


def test_actual_entry_process_stops_at_invented_false_gate(tmp_path):
    values = arguments()
    for name in ("attempt", "preparation-attempt", "archive", "source-csv"):
        values[values.index(f"--{name}") + 1] = str(tmp_path / name)
    completed = subprocess.run(
        [sys.executable, "-c", closed_program(), str(SCRIPT), *values],
        env=environment(),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 2
    assert completed.stdout == ""
    assert completed.stderr == "Study execution stopped: execution_failed\n"
    assert list(tmp_path.iterdir()) == []


def test_real_public_preflight_rejects_invented_non_repository(tmp_path):
    values = arguments()
    values[values.index("--repo-root") + 1] = str(tmp_path)
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), *values],
        env=environment(),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 2
    assert completed.stdout == ""
    assert completed.stderr == "Study execution stopped: execution_failed\n"
    assert list(Path(tmp_path).iterdir()) == []
