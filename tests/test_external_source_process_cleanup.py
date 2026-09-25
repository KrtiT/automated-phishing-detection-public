"""Later transport failures cannot erase the original observed worker progress."""

import pytest
from test_external_source_process_observation import context

from automated_phishing_detection import _internal_transport_io
from automated_phishing_detection.owned_worker import WorkerExecutionError


@pytest.mark.parametrize("cleanup_type", [RuntimeError, KeyboardInterrupt, SystemExit])
def test_transport_cleanup_preserves_progress_from_replaced_worker_error(
    tmp_path, monkeypatch, cleanup_type
):
    api, case = context(tmp_path, monkeypatch)
    worker_error = WorkerExecutionError(
        "worker_wait_failed", progress=b"known-progress"
    )
    cleanup_error = cleanup_type("private-cleanup-canary")
    original = _internal_transport_io.RetainedFiles.close

    def fail_observation(command):
        raise worker_error

    def fail_cleanup(owner):
        original(owner)
        raise cleanup_error

    monkeypatch.setattr(api, "observe_worker", fail_observation)
    monkeypatch.setattr(_internal_transport_io.RetainedFiles, "close", fail_cleanup)
    with pytest.raises(BaseException) as rejected:
        api._run_observed_external(case.binding, case.paths, case.handoff)
    failure = rejected.value.external_failure
    assert failure.worker_progress == b"known-progress"
    assert failure.worker is None
    if not issubclass(cleanup_type, Exception):
        assert rejected.value is cleanup_error
