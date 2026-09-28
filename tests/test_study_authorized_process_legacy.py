"""An exact invented study envelope does not open the old actual worker CLI."""

import subprocess
from pathlib import Path

from study_authorized_process_root_fixtures import setup
from test_source_runner import inputs, runner

from automated_phishing_detection._prepared_internal_process import _worker_command

__all__ = ["inputs", "runner"]


def test_old_prepared_cli_remains_closed_with_study_profile_present(
    tmp_path, inputs, monkeypatch
):
    case = setup(inputs, tmp_path, monkeypatch)
    auth = case.authorization
    script = "scripts/run_prepared_internal_evaluation.py"
    (case.root / script).write_bytes((Path(__file__).parents[1] / script).read_bytes())
    command = _worker_command(
        auth.base,
        auth.paths.internal,
        reservation_sha256=case.preparation.reservation_sha256,
        completion_sha256=case.preparation.completion_sha256,
    )
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode != 0
    assert not auth.paths.internal.attempt.exists()
    assert auth.base.protected_evaluation_ready is False
    assert auth.external.protected_evaluation_ready is False
    assert auth.operational.protected_evaluation_ready is False
