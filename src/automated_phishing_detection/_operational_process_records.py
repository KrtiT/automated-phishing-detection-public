"""Create-only observations of owned processes, not scientific authorization."""

import json
from dataclasses import dataclass
from hashlib import sha256

from . import execution_receipt as receipt
from ._owned_process_exit import observe_owned_exit
from .source_runner import _json, _read_file_once


class OperationalProcessError(ValueError):
    def __init__(self, check_id, *, progress=None):
        self.check_id = check_id
        self.progress = progress
        super().__init__(check_id)


@dataclass(frozen=True)
class ProcessObservation:
    record: bytes


def _bytes(value):
    return receipt._json_bytes(value, "process observation") + b"\n"


def _record(attempt, name, content):
    with receipt._attempt_directory(attempt) as directory:
        if any(
            receipt._entry(directory, entry) is not None
            for entry in ("finalize.claim", "outcome.json", "evidence")
        ):
            raise OperationalProcessError("attempt_finalized")
        receipt._install_record(directory, name, content)


def _stream_hash(stream):
    stream.seek(0)
    digest = sha256()
    while chunk := stream.read(1024 * 1024):
        digest.update(chunk)
    return digest.hexdigest()


class Observations:
    def __init__(self, attempt, writer):
        self.attempt, self.writer = attempt, writer
        self.ready = None
        self.exited_roles = set()
        self.value = {
            "schema_version": 1,
            "reservation_sha256": attempt.reservation_sha256,
            "status": "pending",
            "research_accepted": False,
            "failure": None,
            "record_failures": [],
            "readiness_sha256": None,
            "cleanup_sha256": None,
            "stop_sent": False,
        }
        for role in ("service", "client"):
            self.value[role] = {
                "pid": None,
                "exit_code": None,
                "exit_observed": False,
                "forced": False,
                "signals": [],
                "stdout_sha256": None,
                "stderr_sha256": None,
            }

    def snapshot(self):
        return _bytes(self.value)

    def claim(self, service_command, client_command, deadlines):
        self.install(
            "process-pair-intent.json",
            {
                "schema_version": 1,
                "reservation_sha256": self.attempt.reservation_sha256,
                "service_command_sha256": command_hash(service_command),
                "client_command_sha256": command_hash(client_command),
                "deadlines": deadlines,
            },
        )

    def fail(self, check_id):
        if self.value["failure"] is None:
            self.value["failure"] = check_id
        self.value["status"] = "failed"

    def install(self, name, value):
        try:
            self.writer(self.attempt, name, _bytes(value))
        except Exception:
            self.value["record_failures"].append(name)
            self.fail("record_write_failed")
            raise OperationalProcessError(
                "record_write_failed", progress=self.snapshot()
            ) from None

    def try_install(self, name, value):
        try:
            self.install(name, value)
        except OperationalProcessError:
            pass

    def lifecycle(self, name, *, pid, port=None):
        content = _read_file_once(self.attempt.directory / name)
        key = "readiness_sha256" if port is not None else "cleanup_sha256"
        self.value[key] = sha256(content).hexdigest()
        value = _json(content)
        fields = {"schema_version", "pid", "workload", "status"}
        if port is not None:
            fields |= {"host", "port"}
        valid = (
            type(value) is dict
            and set(value) == fields
            and type(value["schema_version"]) is int
            and value["schema_version"] == 1
            and type(value["pid"]) is int
            and value["pid"] == pid
            and type(value["workload"]) is str
            and bool(value["workload"])
        )
        if not valid:
            raise OperationalProcessError("invalid_lifecycle_record")
        self._lifecycle_values(value, port)
        return value

    def _lifecycle_values(self, value, port):
        if port is not None:
            valid = (
                value["status"] == "ready"
                and value["host"] == "127.0.0.1"
                and type(value["port"]) is int
                and value["port"] == port
            )
        else:
            valid = (
                self.ready is not None
                and value["status"] == "clean"
                and value["workload"] == self.ready["workload"]
            )
        if not valid:
            raise OperationalProcessError("invalid_lifecycle_record")

    def exited(self, role, process):
        if role in self.exited_roles:
            return True
        observed = observe_owned_exit(process)
        if observed is None:
            return False
        self.exited_roles.add(role)
        if not observed.exit_observed:
            self.fail(f"{role}_exit_unobserved")
            return True
        self.value[role].update(exit_code=observed.exit_code, exit_observed=True)
        return True

    def exits(self, children):
        for role in children.processes:
            children.exited(role)
            observed = self.value[role]
            code = observed["exit_code"]
            if code is not None:
                output, errors = children.streams[role]
                observed["stdout_sha256"] = _stream_hash(output)
                observed["stderr_sha256"] = _stream_hash(errors)
            if code != 0 or observed["forced"]:
                self.fail(f"{role}_exit_unsuccessful")
            self.try_install(f"{role}-process.json", observed)

    def finish(self, children):
        self.exits(children)
        if not self.value["stop_sent"]:
            self.fail("service_stop_unobserved")
        if "service" in children.processes:
            try:
                self.lifecycle(
                    "service-cleanup.json", pid=children.processes["service"].pid
                )
            except Exception:
                self.fail("service_cleanup_unverified")
        if self.value["failure"] is None:
            self.value["status"] = "observed"
        self.try_install("process-pair.json", self.value)
        return self.snapshot()


def command_hash(command):
    content = json.dumps(list(command), ensure_ascii=True, separators=(",", ":"))
    return sha256(content.encode("ascii")).hexdigest()
