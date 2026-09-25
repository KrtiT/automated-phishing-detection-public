"""Public metadata acquisition closes borrowed descriptors before interruption."""

import os
import signal
from types import SimpleNamespace

import pytest
from test_external_source_process_commands import command_case
from test_prepared_external_io import _closed

from automated_phishing_detection import execution_receipt as receipt
from automated_phishing_detection import external_source_runner as worker
from automated_phishing_detection import prepared_external_process as pair
from automated_phishing_detection.execution_preflight import ExecutionBinding


def public_call(module, binding):
    arguments = dict(
        expected_revision=binding.revision,
        expected_contract_sha256=binding.contract_sha256,
        expected_preparation_reservation_sha256=object(),
        expected_preparation_completion_sha256=object(),
    )
    if module is worker:
        return worker.run_prepared_external_evaluation(
            binding.root,
            **arguments,
            paths=object(),
            internal_transport=object(),
            expected_handoff_sha256=object(),
        )
    return pair.run_prepared_internal_external_process(
        binding.root, **arguments, internal_paths=object(), external_paths=object()
    )


def public_setup(module, binding, tmp_path, monkeypatch, descriptors):
    original = receipt._open_directory

    def opened(path):
        descriptor = original(path)
        descriptors.append(descriptor)
        os.kill(os.getpid(), signal.SIGINT)
        return descriptor

    def metadata(*args):
        with receipt._directory(tmp_path):
            return SimpleNamespace(protected_evaluation_ready=False)

    monkeypatch.setattr(receipt, "_open_directory", opened)
    monkeypatch.setattr(module, "bind_execution", lambda *args, **kwargs: binding)
    monkeypatch.setattr(
        ExecutionBinding, "protected_evaluation_ready", property(lambda _: True)
    )
    owner = worker.body if module is worker else pair.process
    monkeypatch.setattr(owner, "resolve_external_source_profile", metadata)


@pytest.mark.parametrize("module", [worker, pair])
def test_public_profile_fd_is_owned_before_signal_delivery(
    module, tmp_path, monkeypatch
):
    value, unused, unused_transport = command_case()
    binding = ExecutionBinding(
        value.root, value.revision, value.contract_sha256, (), "{}"
    )
    descriptors, first = [], KeyboardInterrupt("invented")
    public_setup(module, binding, tmp_path, monkeypatch, descriptors)

    def interrupted(signum, frame):
        raise first

    previous = signal.signal(signal.SIGINT, interrupted)
    try:
        with pytest.raises(KeyboardInterrupt) as caught:
            public_call(module, binding)
        assert caught.value is first
    finally:
        signal.signal(signal.SIGINT, previous)
    assert descriptors and all(_closed(descriptor) for descriptor in descriptors)
