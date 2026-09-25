import asyncio
import importlib
import json
import os
import select
import socket
import subprocess
import sys
import threading
from pathlib import Path

import httpx
import pytest
from test_selective_service import SyntheticScorer, factory_for

from automated_phishing_detection.selective_service import create_app


@pytest.fixture
def operational():
    path = Path(__file__).resolve().parents[1] / (
        "src/automated_phishing_detection/operational_service.py"
    )
    assert path.is_file(), "missing operational service lifecycle"
    return importlib.import_module("automated_phishing_detection.operational_service")


async def exercise_service(operational, *, stop=b"stop\n", **options):
    holder, records = [], []
    app = create_app(factory_for(holder, **options))
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    base_url = f"http://127.0.0.1:{listener.getsockname()[1]}"
    stop_read, stop_write = os.pipe()
    ready = asyncio.Event()

    def retain(name, content):
        value = json.loads(content)
        records.append((name, value))
        if name == "service-ready.json":
            assert app.state.owner.is_alive
            ready.set()
        if name == "service-cleanup.json":
            assert not app.state.owner.is_alive
            assert holder[0].events[-1][0] == "exit"

    task = asyncio.create_task(
        operational.serve_service(app, listener, stop_read, retain=retain)
    )
    try:
        await asyncio.wait_for(ready.wait(), 5)
        async with httpx.AsyncClient(trust_env=False) as client:
            response = await client.post(
                base_url + "/v1/scan",
                json={"request_id": "one", "url": "https://example.test"},
            )
        assert response.status_code == 200
        if stop is not None:
            os.write(stop_write, stop)
        os.close(stop_write)
        stop_write = None
        try:
            await asyncio.wait_for(task, 5)
        except operational.OperationalServiceError as error:
            return holder, records, error
        return holder, records, None
    finally:
        if stop_write is not None:
            os.close(stop_write)
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        os.close(stop_read)
        listener.close()


def test_real_socket_ready_and_clean_receipts_follow_owner_lifecycle(operational):
    holder, records, error = asyncio.run(exercise_service(operational))
    assert error is None
    assert [name for name, _ in records] == [
        "service-ready.json",
        "service-cleanup.json",
    ]
    assert records[0][1]["pid"] == os.getpid()
    assert records[0][1]["workload"] == "fixed_cascade"
    assert records[0][1]["host"] == "127.0.0.1"
    assert records[1][1]["status"] == "clean"
    assert [name for name, _ in holder[0].events].count("exit") == 1


def test_cleanup_failure_cannot_be_hidden_by_uvicorn_return(operational):
    holder, records, error = asyncio.run(
        exercise_service(operational, exit_error=RuntimeError("private detail"))
    )
    assert isinstance(error, operational.OperationalServiceError)
    assert error.check_id == "service_cleanup_failed"
    assert "private" not in str(error)
    assert [name for name, _ in records] == ["service-ready.json"]
    assert holder[0].events[-1][0] == "exit"


@pytest.mark.parametrize("stop", [None, b"wrong\n", b"stop\nstop\n"])
def test_lost_or_invalid_supervisor_does_not_publish_clean_success(operational, stop):
    holder, records, error = asyncio.run(exercise_service(operational, stop=stop))
    assert isinstance(error, operational.OperationalServiceError)
    assert error.check_id == "invalid_stop_control"
    assert [name for name, _ in records] == ["service-ready.json"]
    assert holder[0].events[-1][0] == "exit"


def test_frozen_uvicorn_configuration(operational):
    config = operational._config(create_app(factory_for([])))
    assert config.workers == 1
    assert config.loop == "asyncio"
    assert config.http == "h11"
    assert config.ws == "none"
    assert config.lifespan == "on"
    assert config.reload is False
    assert config.access_log is False
    assert config.proxy_headers is False
    assert config.timeout_keep_alive == 5
    assert config.backlog == 2048
    assert config.timeout_graceful_shutdown == 60


@pytest.mark.parametrize(
    "failed_record", ["service-ready.json", "service-cleanup.json"]
)
def test_checkpoint_failure_is_sanitized_and_owner_is_joined(
    operational, failed_record
):
    async def exercise():
        holder, records = [], []
        app = create_app(factory_for(holder))
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        stop_read, stop_write = os.pipe()

        def retain(name, content):
            records.append(name)
            if name == failed_record:
                raise OSError("private checkpoint path")
            if name == "service-ready.json":
                os.write(stop_write, b"stop\n")

        try:
            with pytest.raises(operational.OperationalServiceError) as caught:
                await asyncio.wait_for(
                    operational.serve_service(app, listener, stop_read, retain=retain),
                    5,
                )
            assert "private" not in str(caught.value)
            assert records.count(failed_record) == 1
            assert not app.state.owner.is_alive
            assert holder[0].events[-1][0] == "exit"
        finally:
            os.close(stop_read)
            os.close(stop_write)
            listener.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("exit_failure", [False, True])
def test_cancellation_joins_owner_without_clean_receipt(operational, exit_failure):
    async def exercise():
        holder, records = [], []
        app = create_app(
            factory_for(
                holder,
                exit_error=RuntimeError("private cleanup") if exit_failure else None,
            )
        )
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        stop_read, stop_write = os.pipe()
        ready = asyncio.Event()

        def retain(name, content):
            records.append(name)
            ready.set()

        task = asyncio.create_task(
            operational.serve_service(app, listener, stop_read, retain=retain)
        )
        try:
            await asyncio.wait_for(ready.wait(), 5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 5)
            assert task.cancelled()
            assert records == ["service-ready.json"]
            assert not app.state.owner.is_alive
        finally:
            os.close(stop_read)
            os.close(stop_write)
            listener.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("exit_failure", [False, True])
def test_actual_service_child_exit_and_retained_cleanup(operational, exit_failure):
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    stop_read, stop_write = os.pipe()
    code = """
import asyncio, json, socket, sys
from automated_phishing_detection.operational_service import (
    OperationalServiceError, serve_service,
)
from automated_phishing_detection.selective_service import create_app
from test_selective_service import factory_for
listener = socket.socket(fileno=int(sys.argv[1]))
error = RuntimeError('private fixture') if sys.argv[3] == 'True' else None
app = create_app(factory_for([], exit_error=error))
def retain(name, content):
    print(json.dumps({'name': name, 'record': json.loads(content)}), flush=True)
try:
    asyncio.run(serve_service(app, listener, int(sys.argv[2]), retain=retain))
except OperationalServiceError:
    sys.exit(2)
"""
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            code,
            str(listener.fileno()),
            str(stop_read),
            str(exit_failure),
        ],
        pass_fds=(listener.fileno(), stop_read),
        env={**os.environ, "PYTHONPATH": os.pathsep.join(("src", "tests"))},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        assert select.select([process.stdout], [], [], 10)[0], "child did not start"
        ready = json.loads(process.stdout.readline())
        assert ready["name"] == "service-ready.json"
        assert ready["record"]["pid"] == process.pid
        os.write(stop_write, b"stop\n")
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == (2 if exit_failure else 0), stderr.decode()
        records = [json.loads(line) for line in stdout.splitlines()]
        assert [record["name"] for record in records] == (
            [] if exit_failure else ["service-cleanup.json"]
        )
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        os.close(stop_read)
        os.close(stop_write)
        listener.close()


def test_stop_before_readiness_closes_listener_without_success(operational):
    async def exercise():
        holder, records = [], []
        app = create_app(factory_for(holder))
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        stop_read, stop_write = os.pipe()
        os.write(stop_write, b"stop\n")
        try:
            with pytest.raises(operational.OperationalServiceError):
                await asyncio.wait_for(
                    operational.serve_service(
                        app,
                        listener,
                        stop_read,
                        retain=lambda name, content: records.append(name),
                    ),
                    5,
                )
            assert "service-cleanup.json" not in records
            assert not app.state.owner.is_alive
            assert listener.fileno() == -1
        finally:
            os.close(stop_read)
            os.close(stop_write)
            listener.close()

    asyncio.run(exercise())


def test_startup_cancellation_finishes_lifespan_task(operational):
    async def exercise():
        entered, release = threading.Event(), threading.Event()
        records = []

        class GatedScorer(SyntheticScorer):
            def __enter__(self):
                entered.set()
                assert release.wait(5)
                return super().__enter__()

        app = create_app(GatedScorer)
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        stop_read, stop_write = os.pipe()
        task = asyncio.create_task(
            operational.serve_service(
                app,
                listener,
                stop_read,
                retain=lambda name, content: records.append(name),
            )
        )
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            task.cancel()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 5)
            assert not app.state.owner.is_alive
            assert not records
            assert not any(
                "LifespanOn.main" in pending.get_coro().__qualname__
                for pending in asyncio.all_tasks()
            )
        finally:
            release.set()
            os.close(stop_read)
            os.close(stop_write)
            listener.close()

    asyncio.run(exercise())
