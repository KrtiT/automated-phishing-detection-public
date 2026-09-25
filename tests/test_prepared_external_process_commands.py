"""Prepared parent commands bind retained input identity, never raw source paths."""

from types import SimpleNamespace

import pytest
from test_external_source_process_commands import command_case

from automated_phishing_detection import external_source_process as process
from automated_phishing_detection._external_source_records import (
    PreparedExternalRunPaths,
)


def test_prepared_command_has_only_retained_input_flags():
    assert callable(getattr(process, "_prepared_worker_command", None))
    binding, old, transport = command_case()
    paths = PreparedExternalRunPaths(
        old.archive,
        old.artifacts,
        old.secondary_artifacts,
        old.drift_artifacts,
        old.attempt,
        old.public_summary,
    )
    preparation = SimpleNamespace(
        reservation_sha256="e" * 64, completion_sha256="f" * 64
    )
    command = process._prepared_worker_command(binding, paths, transport, preparation)
    options = dict(zip(command[2::2], command[3::2], strict=True))
    assert command[1].endswith("/scripts/run_prepared_external_evaluation.py")
    assert not {"--archive", "--suffix-rules", "--source-csv"} & set(options)
    assert options["--preparation"] == str(paths.preparation)
    assert options["--expected-preparation-reservation-sha256"] == "e" * 64
    assert options["--expected-preparation-completion-sha256"] == "f" * 64
    assert options["--expected-handoff-sha256"] == transport.expected_handoff_sha256


def test_prepared_command_rejects_original_path_type():
    assert callable(getattr(process, "_prepared_worker_command", None))
    binding, paths, transport = command_case()
    with pytest.raises(process.ExternalSourceExecutionError):
        process._prepared_worker_command(binding, paths, transport, object())
