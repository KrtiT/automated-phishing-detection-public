"""Retain service lifecycle evidence; this alone is not bound run provenance."""

import asyncio
import json
import os
import socket

import uvicorn

from .selective_service import ScoringOwner


class OperationalServiceError(ValueError):
    def __init__(self, check_id):
        self.check_id = check_id
        super().__init__(check_id)


def _config(app):
    return uvicorn.Config(
        app,
        workers=1,
        loop="asyncio",
        http="h11",
        ws="none",
        lifespan="on",
        reload=False,
        access_log=False,
        proxy_headers=False,
        timeout_keep_alive=5,
        backlog=2048,
        timeout_graceful_shutdown=60,
        log_level="critical",
    )


def _record(retain, name, value):
    content = (
        json.dumps(
            {"schema_version": 1, **value},
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()
    try:
        retain(name, content)
    except Exception:
        raise OperationalServiceError("service_checkpoint_failed") from None


async def _shutdown(server, owner, completed):
    failure = None
    try:
        if server.started and not completed:
            await server.shutdown()
        elif not completed and hasattr(server, "lifespan"):
            await asyncio.wait_for(server.lifespan.shutdown(), timeout=60)
    except BaseException as error:
        failure = error
    try:
        await owner.shutdown()
    except BaseException as error:
        if failure is None or isinstance(failure, Exception):
            failure = error
    if failure is not None:
        raise failure from None


async def _cleanup_result(server, owner, completed):
    try:
        await _shutdown(server, owner, completed)
    except BaseException as error:
        return error


async def serve_service(app, listener, stop_fd, *, retain):
    """Serve one constructed owner and certify cleanup, never caller identity.

    The caller owns the socket and control descriptors. The eventual official
    supervisor must bind the app, install records create-only, and independently
    observe both service and client process exits before accepting a run.
    """
    owner = getattr(app.state, "owner", None)
    if not isinstance(owner, ScoringOwner):
        raise OperationalServiceError("invalid_service_owner")
    if (
        not isinstance(listener, socket.socket)
        or listener.family != socket.AF_INET
        or listener.type != socket.SOCK_STREAM
        or listener.getsockname()[0] != "127.0.0.1"
    ):
        raise OperationalServiceError("invalid_service_listener")
    if type(stop_fd) is not int or stop_fd < 0 or not callable(retain):
        raise OperationalServiceError("invalid_service_control")
    identity = {"pid": os.getpid(), "workload": app.state.workload}
    ready = False
    stopped = None

    class RetainedServer(uvicorn.Server):
        async def startup(self, sockets=None):
            nonlocal ready
            await super().startup(sockets=sockets)
            if self.started:
                _record(
                    retain,
                    "service-ready.json",
                    {
                        **identity,
                        "status": "ready",
                        "host": "127.0.0.1",
                        "port": listener.getsockname()[1],
                    },
                )
                ready = True

    server = RetainedServer(_config(app))
    loop = asyncio.get_running_loop()

    def stop():
        nonlocal stopped
        try:
            stopped = os.read(stop_fd, 4096) == b"stop\n" and ready
        except OSError:
            stopped = False
        loop.remove_reader(stop_fd)
        server.should_exit = True

    completed = False
    failure = None
    loop.add_reader(stop_fd, stop)
    try:
        await server.serve(sockets=[listener])
        completed = True
    except BaseException as error:
        failure = error
    finally:
        try:
            loop.remove_reader(stop_fd)
        except BaseException as error:
            if failure is None or isinstance(failure, Exception):
                failure = error
    cleanup = asyncio.create_task(_cleanup_result(server, owner, completed))
    cancelled = isinstance(failure, asyncio.CancelledError)
    while not cleanup.done():
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            cancelled = True
        except BaseException as error:
            if failure is None or isinstance(failure, Exception):
                failure = error
    try:
        cleanup_error = cleanup.result()
        if cleanup_error is not None:
            if failure is None or isinstance(failure, Exception):
                failure = cleanup_error
            cancelled = cancelled or isinstance(cleanup_error, asyncio.CancelledError)
    except asyncio.CancelledError:
        cancelled = True
    except BaseException as error:
        if failure is None or isinstance(failure, Exception):
            failure = error
    if failure is not None and not isinstance(
        failure, (Exception, asyncio.CancelledError)
    ):
        raise failure from None
    if cancelled:
        raise asyncio.CancelledError from None
    if failure is not None:
        raise OperationalServiceError("service_cleanup_failed") from None
    if not ready:
        raise OperationalServiceError("service_not_ready")
    if stopped is not True:
        raise OperationalServiceError("invalid_stop_control")
    _record(retain, "service-cleanup.json", {**identity, "status": "clean"})
