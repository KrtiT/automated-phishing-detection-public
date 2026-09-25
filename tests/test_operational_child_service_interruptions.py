"""Direct service interruptions may not turn into ordinary cleanup failures."""

import asyncio
import os
import signal
import socket

import pytest
from test_selective_service import factory_for

from automated_phishing_detection import operational_service
from automated_phishing_detection.selective_service import create_app


@pytest.mark.parametrize(
    "cleanup", [None, OSError("cleanup"), KeyboardInterrupt("later")]
)
def test_first_service_sigint_survives_cleanup(monkeypatch, cleanup):
    captured, closed = [], []

    async def serve(self, **kwargs):
        try:
            os.kill(os.getpid(), signal.SIGINT)
        except KeyboardInterrupt as error:
            captured.append(error)
            raise

    async def shutdown(*args):
        closed.append(True)
        if cleanup is not None:
            raise cleanup

    monkeypatch.setattr(operational_service.uvicorn.Server, "serve", serve)
    monkeypatch.setattr(operational_service, "_shutdown", shutdown)
    caught = exercise()
    assert caught is captured[0] and closed == [True]


def exercise():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        stop_read, stop_write = os.pipe()
        try:
            with pytest.raises(KeyboardInterrupt) as caught:
                asyncio.run(
                    operational_service.serve_service(
                        create_app(factory_for([])),
                        listener,
                        stop_read,
                        retain=lambda *args: None,
                    )
                )
        finally:
            os.close(stop_read)
            os.close(stop_write)
    return caught.value


@pytest.mark.parametrize("later", [OSError("remove"), KeyboardInterrupt("remove")])
def test_stop_reader_removal_cannot_skip_cleanup(monkeypatch, later):
    first, closed = KeyboardInterrupt("first"), []

    async def serve(self, **kwargs):
        loop = asyncio.get_running_loop()
        original = loop.remove_reader

        def remove(descriptor):
            loop.remove_reader = original
            original(descriptor)
            raise later

        loop.remove_reader = remove
        raise first

    async def shutdown(*args):
        closed.append(True)

    monkeypatch.setattr(operational_service.uvicorn.Server, "serve", serve)
    monkeypatch.setattr(operational_service, "_shutdown", shutdown)
    assert exercise() is first
    assert closed == [True]
