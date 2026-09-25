"""Numerical cleanup priorities use CPU fixtures and never run a forward."""

import asyncio

import pytest
from test_selective_inference import loaded as loaded
from test_selective_inference import module as module
from test_selective_inference import (
    restore_deterministic_mode as restore_deterministic_mode,
)


@pytest.mark.parametrize(
    "kind", (KeyboardInterrupt, asyncio.CancelledError, SystemExit)
)
@pytest.mark.parametrize("phase", ("entry", "body"))
def test_selective_cleanup_keeps_original_and_releases_owner(
    loaded, module, monkeypatch, kind, phase
):
    original, calls = kind("first"), []
    deterministic = module.torch.use_deterministic_algorithms

    def broken(enabled, **kwargs):
        calls.append(enabled)
        deterministic(enabled, **kwargs)
        if len(calls) == 1:
            if phase == "entry":
                raise original
        else:
            raise OSError("restore")

    monkeypatch.setattr(module.torch, "use_deterministic_algorithms", broken)
    with pytest.raises(BaseException) as caught:
        with module.SelectiveCascade(loaded, _fixture_cpu=True):
            raise original
    assert caught.value is original
    assert len(calls) == 2
    assert module._PROCESS_SESSION.acquire(blocking=False)
    module._PROCESS_SESSION.release()
    monkeypatch.setattr(module.torch, "use_deterministic_algorithms", deterministic)


@pytest.mark.parametrize("phase", ("construction", "registration"))
def test_cleanup_setup_failure_does_not_leak_process_lock(
    loaded, module, monkeypatch, phase
):
    original = KeyboardInterrupt("setup")
    stack_type = module.CleanupStack

    class BrokenStack(stack_type):
        def __init__(self):
            if phase == "construction":
                raise original
            super().__init__()

        def callback(self, *args, **kwargs):
            raise original

    monkeypatch.setattr(module, "CleanupStack", BrokenStack)
    with pytest.raises(BaseException) as caught:
        module.SelectiveCascade(loaded, _fixture_cpu=True).__enter__()
    free = module._PROCESS_SESSION.acquire(blocking=False)
    module._PROCESS_SESSION.release()
    assert caught.value is original
    assert free
