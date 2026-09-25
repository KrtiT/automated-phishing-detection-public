import asyncio
import importlib
import json
import os
import signal
import subprocess
import sys
import time
from hashlib import sha256
from pathlib import Path

import pytest

from automated_phishing_detection import execution_receipt as receipt


@pytest.fixture
def operational():
    path = Path(__file__).resolve().parents[1] / (
        "src/automated_phishing_detection/operational_process.py"
    )
    assert path.is_file(), "missing parent process observation primitive"
    return importlib.import_module("automated_phishing_detection.operational_process")


COMMON = """
import json, os, signal, socket, sys, time
from pathlib import Path
directory = Path(os.environ['APD_ATTEMPT_DIRECTORY'])
mode = sys.argv[1]
def record(name, value):
    with (directory / name).open('xb') as stream:
        stream.write(json.dumps(value, sort_keys=True).encode() + b'\\n')
        stream.flush()
        os.fsync(stream.fileno())
"""
SERVICE = (
    COMMON
    + """
listener = socket.socket(fileno=int(os.environ['APD_LISTENER_FD']))
listener.listen()
if mode == 'startup_death':
    sys.exit(11)
if mode == 'startup_hang':
    time.sleep(60)
identity = {'schema_version': 1, 'pid': os.getpid(), 'workload': 'fixed_cascade'}
ready = {**identity, 'status': 'ready', 'host': '127.0.0.1',
         'port': listener.getsockname()[1]}
if mode == 'wrong_ready':
    ready['pid'] += 1
if mode != 'missing_ready':
    record('service-ready.json', ready)
os.write(int(os.environ['APD_READY_FD']), b'ready\\n')
os.close(int(os.environ['APD_READY_FD']))
if mode == 'ignore_stop':
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    time.sleep(60)
if mode == 'term_zero':
    signal.signal(signal.SIGTERM, lambda *args: sys.exit(0))
    time.sleep(60)
if mode == 'early_clean':
    while not (directory / 'allow-early-exit.json').exists():
        time.sleep(0.005)
    record('service-cleanup.json', {**identity, 'status': 'clean'})
    sys.exit(0)
if mode == 'service_dies':
    while not (directory / 'client-started.json').exists():
        time.sleep(0.005)
    sys.exit(12)
assert os.read(int(os.environ['APD_STOP_FD']), 4096) == b'stop\\n'
if mode == 'slow_cleanup':
    time.sleep(0.2)
if mode == 'cleanup_nonzero':
    sys.exit(13)
if mode == 'wrong_cleanup':
    identity['pid'] += 1
record('service-cleanup.json', {**identity, 'status': 'clean'})
print('private-service-output')
"""
)
CLIENT = (
    COMMON
    + """
assert 'APD_LISTENER_FD' not in os.environ
assert 'APD_STOP_FD' not in os.environ
assert 'APD_READY_FD' not in os.environ
record('measured.json', {'fixture': True})
print('private-client-output', flush=True)
if mode == 'nonzero':
    sys.exit(17)
if mode == 'signal':
    os.kill(os.getpid(), signal.SIGTERM)
if mode == 'block':
    time.sleep(60)
"""
)


def inputs(tmp_path, service="success", client="success"):
    attempt = receipt.reserve_attempt(tmp_path / "attempt", identity={"fixture": True})
    return attempt, {
        "service_command": (sys.executable, "-c", SERVICE, service),
        "client_command": (sys.executable, "-c", CLIENT, client),
        "startup_timeout_seconds": 5.0,
        "shutdown_timeout_seconds": 1.0,
        "terminate_timeout_seconds": 0.2,
        "kill_timeout_seconds": 1.0,
    }


def assert_reaped(progress):
    for role in ("service", "client"):
        observed = progress[role]
        if observed["pid"] is not None:
            assert type(observed["exit_code"]) is int
            with pytest.raises(ProcessLookupError):
                os.kill(observed["pid"], 0)


def test_real_children_are_observed_and_private_output_is_only_hashed(
    operational, tmp_path
):
    attempt, options = inputs(tmp_path)
    observed = asyncio.run(operational.observe_process_pair(attempt, **options))
    progress = json.loads(observed.record)
    assert progress["status"] == "observed"
    assert progress["research_accepted"] is False
    assert progress["service"]["exit_code"] == 0
    assert progress["client"]["exit_code"] == 0
    assert progress["service"]["exit_observed"] is True
    assert progress["client"]["exit_observed"] is True
    assert progress["service"]["forced"] is False
    assert progress["client"]["forced"] is False
    assert (
        progress["service"]["stdout_sha256"]
        == sha256(b"private-service-output\n").hexdigest()
    )
    assert (
        progress["client"]["stdout_sha256"]
        == sha256(b"private-client-output\n").hexdigest()
    )
    assert (
        progress["readiness_sha256"]
        == sha256((attempt.directory / "service-ready.json").read_bytes()).hexdigest()
    )
    assert (
        progress["cleanup_sha256"]
        == sha256((attempt.directory / "service-cleanup.json").read_bytes()).hexdigest()
    )
    assert (attempt.directory / "process-pair.json").read_bytes() == observed.record
    assert not (attempt.directory / "finalize.claim").exists()
    for path in attempt.directory.iterdir():
        assert b"private-" not in path.read_bytes()
    assert_reaped(progress)


@pytest.mark.parametrize("service", ["startup_death", "wrong_ready", "missing_ready"])
def test_unready_service_cannot_launch_client(operational, tmp_path, service):
    attempt, options = inputs(tmp_path, service=service)
    with pytest.raises(operational.OperationalProcessError) as caught:
        asyncio.run(operational.observe_process_pair(attempt, **options))
    progress = json.loads(caught.value.progress)
    assert progress["status"] == "failed"
    assert progress["client"]["pid"] is None
    assert progress["client"]["exit_code"] is None
    assert not (attempt.directory / "client-intent.json").exists()
    assert_reaped(progress)


@pytest.mark.parametrize("client,code", [("nonzero", 17), ("signal", -signal.SIGTERM)])
def test_failed_client_retains_checkpoint_and_gracefully_stops_service(
    operational, tmp_path, client, code
):
    attempt, options = inputs(tmp_path, client=client)
    with pytest.raises(operational.OperationalProcessError) as caught:
        asyncio.run(operational.observe_process_pair(attempt, **options))
    progress = json.loads(caught.value.progress)
    assert progress["client"]["exit_code"] == code
    assert progress["service"]["exit_code"] == 0
    assert progress["service"]["forced"] is False
    assert progress["cleanup_sha256"] is not None
    assert (attempt.directory / "measured.json").is_file()
    assert_reaped(progress)


@pytest.mark.parametrize("service", ["cleanup_nonzero", "wrong_cleanup"])
def test_cleanup_must_be_observed_and_match_owned_service(
    operational, tmp_path, service
):
    attempt, options = inputs(tmp_path, service=service)
    with pytest.raises(operational.OperationalProcessError) as caught:
        asyncio.run(operational.observe_process_pair(attempt, **options))
    progress = json.loads(caught.value.progress)
    assert progress["client"]["exit_code"] == 0
    assert progress["status"] == "failed"
    assert_reaped(progress)


def test_service_death_stops_owned_client(operational, tmp_path):
    attempt, options = inputs(tmp_path, service="service_dies", client="block")
    with pytest.raises(operational.OperationalProcessError) as caught:
        asyncio.run(operational.observe_process_pair(attempt, **options))
    progress = json.loads(caught.value.progress)
    assert progress["service"]["exit_code"] == 12
    assert progress["client"]["forced"] is True
    assert_reaped(progress)


def test_forced_shutdown_never_appears_clean(operational, tmp_path):
    attempt, options = inputs(tmp_path, service="ignore_stop")
    options["shutdown_timeout_seconds"] = 0.1
    with pytest.raises(operational.OperationalProcessError) as caught:
        asyncio.run(operational.observe_process_pair(attempt, **options))
    progress = json.loads(caught.value.progress)
    assert progress["service"]["forced"] is True
    assert progress["service"]["signals"] == [signal.SIGTERM, signal.SIGKILL]
    assert progress["service"]["exit_code"] == -signal.SIGKILL
    assert progress["cleanup_sha256"] is None
    assert_reaped(progress)


def test_forced_shutdown_cannot_be_success_even_with_zero_exit(operational, tmp_path):
    attempt, options = inputs(tmp_path, service="term_zero")
    options["shutdown_timeout_seconds"] = 0.1
    with pytest.raises(operational.OperationalProcessError) as caught:
        asyncio.run(operational.observe_process_pair(attempt, **options))
    progress = json.loads(caught.value.progress)
    assert progress["service"]["exit_code"] == 0
    assert progress["service"]["forced"] is True
    assert_reaped(progress)


def test_service_zero_exit_before_parent_stop_is_not_clean(
    operational, tmp_path, monkeypatch
):
    attempt, options = inputs(tmp_path, service="early_clean")
    original = operational._run

    async def finish_without_stop(children, observations, service, client):
        await original(children, observations, service, client)
        (attempt.directory / "allow-early-exit.json").write_bytes(b"{}\n")
        await children.wait_exit("service", 5)

    monkeypatch.setattr(operational, "_run", finish_without_stop)
    with pytest.raises(operational.OperationalProcessError) as caught:
        asyncio.run(operational.observe_process_pair(attempt, **options))
    progress = json.loads(caught.value.progress)
    assert progress["service"]["exit_code"] == 0
    assert progress["stop_sent"] is False
    assert_reaped(progress)


def test_startup_timeout_is_not_a_request_deadline(operational, tmp_path):
    attempt, options = inputs(tmp_path, service="startup_hang")
    options["startup_timeout_seconds"] = 0.05
    options["shutdown_timeout_seconds"] = 0.05
    with pytest.raises(operational.OperationalProcessError) as caught:
        asyncio.run(operational.observe_process_pair(attempt, **options))
    progress = json.loads(caught.value.progress)
    assert progress["failure"] == "startup_timeout"
    assert progress["client"]["pid"] is None
    assert_reaped(progress)


def test_repeated_parent_cancellation_joins_children_and_retains_progress(
    operational, tmp_path
):
    attempt, options = inputs(tmp_path, service="slow_cleanup", client="block")

    async def exercise():
        task = asyncio.create_task(operational.observe_process_pair(attempt, **options))
        while not (attempt.directory / "measured.json").exists():
            await asyncio.sleep(0.005)
        task.cancel()
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError) as caught:
            await task
        assert task.cancelled()
        progress = json.loads(operational.process_progress(caught.value))
        assert progress["status"] == "failed"
        assert progress["failure"] == "parent_cancelled"
        assert progress["service"]["exit_code"] == 0
        assert_reaped(progress)

    asyncio.run(asyncio.wait_for(exercise(), timeout=10))


@pytest.mark.parametrize(
    "failed_record",
    [
        "service-started.json",
        "client-started.json",
        "service-stop.json",
        "service-process.json",
        "process-pair.json",
    ],
)
def test_record_failure_never_retries_or_leaks_children(
    operational, tmp_path, monkeypatch, failed_record
):
    attempt, options = inputs(tmp_path)
    original = operational._record
    writes = []

    def fail(attempt, name, content):
        writes.append(name)
        if name == failed_record:
            raise OSError("private failing path")
        return original(attempt, name, content)

    monkeypatch.setattr(operational, "_record", fail)
    with pytest.raises(operational.OperationalProcessError) as caught:
        asyncio.run(operational.observe_process_pair(attempt, **options))
    assert "private" not in str(caught.value)
    assert writes.count(failed_record) == 1
    assert (attempt.directory / "process-pair-intent.json").exists()
    progress = json.loads(caught.value.progress)
    assert failed_record in progress["record_failures"]
    assert_reaped(progress)


def test_installed_intent_consumes_attempt_without_relaunch(
    operational, tmp_path, monkeypatch
):
    attempt, options = inputs(tmp_path)
    asyncio.run(operational.observe_process_pair(attempt, **options))

    def forbid(*args, **kwargs):
        raise AssertionError("attempt cannot launch twice")

    monkeypatch.setattr(subprocess, "Popen", forbid)
    with pytest.raises(operational.OperationalProcessError):
        asyncio.run(operational.observe_process_pair(attempt, **options))


@pytest.mark.parametrize("phase", ["run", "cleanup"])
def test_real_parent_interrupt_retains_observations_and_joins_children(
    operational, tmp_path, phase
):
    attempt, options = inputs(
        tmp_path,
        service="slow_cleanup" if phase == "run" else "ignore_stop",
        client="block" if phase == "run" else "success",
    )
    options["shutdown_timeout_seconds"] = 2.0
    trigger = "measured.json" if phase == "run" else "service-stop.json"
    parent_code = """
import asyncio, json, sys
from pathlib import Path
from automated_phishing_detection.execution_receipt import Attempt
from automated_phishing_detection.operational_process import observe_process_pair
attempt = Attempt(Path(sys.argv[1]), sys.argv[2])
options = json.loads(sys.argv[3])
for key in ('service_command', 'client_command'):
    options[key] = tuple(options[key])
asyncio.run(observe_process_pair(attempt, **options))
"""
    parent = subprocess.Popen(
        [
            sys.executable,
            "-c",
            parent_code,
            str(attempt.directory),
            attempt.reservation_sha256,
            json.dumps(options),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + 10
        while not (attempt.directory / trigger).exists():
            assert parent.poll() is None
            assert time.monotonic() < deadline
            time.sleep(0.01)
        parent.send_signal(signal.SIGINT)
        parent.communicate(timeout=10)
        assert parent.returncode != 0
        progress = json.loads((attempt.directory / "process-pair.json").read_bytes())
        assert progress["failure"] in {"parent_cancelled", "parent_interrupted"}
        assert (attempt.directory / "measured.json").is_file()
        assert_reaped(progress)
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.communicate(timeout=5)
        if not (attempt.directory / "process-pair.json").exists():
            for role in ("service", "client"):
                started = attempt.directory / f"{role}-started.json"
                if started.is_file():
                    try:
                        os.kill(json.loads(started.read_bytes())["pid"], signal.SIGKILL)
                    except ProcessLookupError:
                        pass


def test_interrupt_at_popen_return_still_owns_and_reaps_service(
    operational, tmp_path, monkeypatch
):
    attempt, options = inputs(tmp_path)
    original = subprocess.Popen
    spawned = []

    def interrupt(*args, **kwargs):
        process = original(*args, **kwargs)
        spawned.append(process)
        signal.raise_signal(signal.SIGINT)
        return process

    monkeypatch.setattr(subprocess, "Popen", interrupt)
    try:
        with pytest.raises(KeyboardInterrupt) as caught:
            asyncio.run(operational.observe_process_pair(attempt, **options))
        progress = json.loads(operational.process_progress(caught.value))
        assert len(spawned) == 1
        assert progress["service"]["pid"] == spawned[0].pid
        assert_reaped(progress)
    finally:
        for process in spawned:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)


def test_client_cleanup_error_does_not_skip_service_join(
    operational, tmp_path, monkeypatch
):
    attempt, options = inputs(tmp_path, client="block")
    original = operational.OwnedChildren.force
    children_seen = []

    async def fail(children, role):
        children_seen.append(children)
        await original(children, role)
        if role == "client":
            raise OSError("private client cleanup error")

    monkeypatch.setattr(operational.OwnedChildren, "force", fail)

    async def exercise():
        task = asyncio.create_task(operational.observe_process_pair(attempt, **options))
        while not (attempt.directory / "measured.json").exists():
            await asyncio.sleep(0.005)
        task.cancel()
        with pytest.raises(asyncio.CancelledError) as caught:
            await task
        progress = json.loads(operational.process_progress(caught.value))
        assert progress["service"]["exit_code"] == 0
        assert_reaped(progress)

    try:
        asyncio.run(asyncio.wait_for(exercise(), timeout=10))
    finally:
        for children in children_seen:
            for process in children.processes.values():
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=5)


@pytest.mark.parametrize("value", [True, 0, -1, float("inf"), float("nan"), "5"])
def test_invalid_deadlines_do_not_consume_attempt(operational, tmp_path, value):
    attempt, options = inputs(tmp_path)
    options["shutdown_timeout_seconds"] = value
    with pytest.raises(
        operational.OperationalProcessError, match="protective_deadline"
    ):
        asyncio.run(operational.observe_process_pair(attempt, **options))
    assert not (attempt.directory / "process-pair-intent.json").exists()


def test_resource_setup_failure_preserves_intent_without_private_error_text(
    operational, tmp_path, monkeypatch
):
    from automated_phishing_detection import _operational_process_children as children

    attempt, options = inputs(tmp_path)

    def fail(*args, **kwargs):
        raise OSError("private socket setup detail")

    async def exercise():
        with monkeypatch.context() as context:
            context.setattr(children.socket, "socket", fail)
            with pytest.raises(operational.OperationalProcessError) as caught:
                await operational.observe_process_pair(attempt, **options)
            assert "private" not in str(caught.value)
            progress = json.loads(caught.value.progress)
            assert progress["service"]["pid"] is None

    asyncio.run(exercise())
    assert (attempt.directory / "process-pair-intent.json").is_file()


@pytest.mark.parametrize(
    "client,code", [("success", 0), ("nonzero", 17), ("signal", -15)]
)
def test_externally_reaped_exit_remains_unknown(
    operational, tmp_path, monkeypatch, client, code
):
    attempt, options = inputs(tmp_path, client=client)
    original = subprocess.Popen
    kill = os.kill
    signaled = []

    def observe_signal(pid, number):
        signaled.append(pid)
        return kill(pid, number)

    def externally_reap(command, **kwargs):
        process = original(command, **kwargs)
        if command == options["client_command"]:
            waited_pid, status = os.waitpid(process.pid, 0)
            assert waited_pid == process.pid
            assert os.waitstatus_to_exitcode(status) == code
        return process

    monkeypatch.setattr(subprocess, "Popen", externally_reap)
    monkeypatch.setattr(os, "kill", observe_signal)
    with pytest.raises(operational.OperationalProcessError) as caught:
        asyncio.run(operational.observe_process_pair(attempt, **options))
    progress = json.loads(caught.value.progress)
    assert progress["client"]["exit_code"] is None
    assert progress["client"]["exit_observed"] is False
    assert progress["client"]["stdout_sha256"] is None
    assert progress["service"]["exit_code"] == 0
    assert progress["client"]["pid"] not in signaled
    installed = json.loads((attempt.directory / "client-process.json").read_bytes())
    assert installed["exit_code"] is None


def test_installed_intent_write_failure_cannot_launch_or_resume(
    operational, tmp_path, monkeypatch
):
    attempt, options = inputs(tmp_path)
    original = operational._record

    def fail_after_install(attempt, name, content):
        original(attempt, name, content)
        raise OSError("private sync failure")

    def forbid(*args, **kwargs):
        raise AssertionError("intent write did not succeed")

    monkeypatch.setattr(operational, "_record", fail_after_install)
    monkeypatch.setattr(subprocess, "Popen", forbid)
    for _ in range(2):
        with pytest.raises(operational.OperationalProcessError) as caught:
            asyncio.run(operational.observe_process_pair(attempt, **options))
        progress = json.loads(caught.value.progress)
        assert progress["service"]["pid"] is None
    assert (attempt.directory / "process-pair-intent.json").is_file()


@pytest.mark.parametrize("role", ["service", "client"])
def test_launch_failure_never_invents_an_exit(operational, tmp_path, role):
    attempt, options = inputs(tmp_path)
    options[f"{role}_command"] = (str(tmp_path / "missing-private-executable"),)
    with pytest.raises(operational.OperationalProcessError) as caught:
        asyncio.run(operational.observe_process_pair(attempt, **options))
    progress = json.loads(caught.value.progress)
    assert progress[role]["pid"] is None
    assert progress[role]["exit_code"] is None
    assert "private" not in str(caught.value)
    assert_reaped(progress)


def test_setup_interrupt_closes_already_owned_listener(
    operational, tmp_path, monkeypatch
):
    attempt, options = inputs(tmp_path)
    listeners = []

    def interrupt(children):
        listeners.append(children.listener)
        raise KeyboardInterrupt

    monkeypatch.setattr(operational.OwnedChildren, "_pipe", interrupt)
    try:
        with pytest.raises(KeyboardInterrupt):
            asyncio.run(operational.observe_process_pair(attempt, **options))
        assert listeners[0].fileno() == -1
    finally:
        for listener in listeners:
            listener.close()


@pytest.mark.parametrize("kind", ["socket", "pipe"])
def test_interrupt_between_descriptor_allocation_and_registration_closes_all(
    operational, tmp_path, monkeypatch, kind
):
    from automated_phishing_detection import _operational_process_children as children

    attempt, options = inputs(tmp_path)
    descriptors, sockets = [], []
    original = children.socket.socket if kind == "socket" else os.pipe

    def interrupt(*args, **kwargs):
        allocated = original(*args, **kwargs)
        if kind == "socket":
            sockets.append(allocated)
            descriptors.append(allocated.fileno())
        else:
            descriptors.extend(allocated)
        signal.raise_signal(signal.SIGINT)
        return allocated

    async def exercise():
        with monkeypatch.context() as context:
            context.setattr(
                children.socket if kind == "socket" else os, kind, interrupt
            )
            with pytest.raises(KeyboardInterrupt):
                await operational.observe_process_pair(attempt, **options)
        for descriptor in descriptors:
            with pytest.raises(OSError):
                os.fstat(descriptor)

    try:
        asyncio.run(exercise())
    finally:
        for owned_socket in sockets:
            owned_socket.close()
        for descriptor in descriptors:
            try:
                os.close(descriptor)
            except OSError:
                pass
