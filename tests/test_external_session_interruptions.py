"""External session teardown preserves original interruptions without scoring."""

import asyncio
from contextlib import contextmanager

import pytest
from test_bound_external_runtime import external_fixture


def install_failures(runtime, fixture, monkeypatch, original, phase):
    @contextmanager
    def scorer(unused):
        if phase == "entry":
            raise original
        try:
            yield fixture.scorer
        finally:
            raise OSError("restoration")

    def recheck(binding):
        fixture.events.append("check")
        if fixture.events.count("check") == 3:
            raise KeyboardInterrupt("repeated")

    monkeypatch.setattr(runtime, "SelectiveCascade", scorer)
    monkeypatch.setattr(runtime, "recheck_binding", recheck)


@pytest.mark.parametrize(
    "kind", (KeyboardInterrupt, asyncio.CancelledError, SystemExit)
)
@pytest.mark.parametrize("phase", ("entry", "body"))
def test_external_session_preserves_interruption_through_final_recheck(
    monkeypatch, tmp_path, kind, phase
):
    from automated_phishing_detection import bound_external_runtime as runtime

    fixture = external_fixture(runtime, monkeypatch, tmp_path)
    original = kind("first")
    install_failures(runtime, fixture, monkeypatch, original, phase)
    with pytest.raises(BaseException) as caught:
        with runtime.open_bound_external_session(
            fixture.binding,
            fixture.primary_paths,
            fixture.secondary_paths,
            fixture.drift_paths,
        ):
            raise original
    assert caught.value is original
    assert fixture.events[-1] == "check"
