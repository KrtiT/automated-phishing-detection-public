"""Invented process records explicitly confer no owned-exit authority."""

from hashlib import sha256

from automated_phishing_detection._operational_process_records import (
    ProcessObservation,
    _bytes,
)
from automated_phishing_detection._process_support import command_hash
from automated_phishing_detection.operational_role_records import (
    OperationalRoleContext,
    build_client_role,
    build_service_role,
)


def digest(content):
    return sha256(content).hexdigest()


def role_value(pid):
    return {
        "pid": pid,
        "exit_code": 0,
        "exit_observed": True,
        "forced": False,
        "signals": [],
        "stdout_sha256": digest(b""),
        "stderr_sha256": digest(b""),
    }


def _initial(inputs, reservation):
    common = {"schema_version": 1, "pid": 321, "workload": inputs.cell.workload}
    ready = _bytes(common | {"status": "ready", "host": "127.0.0.1", "port": 54321})
    cleanup = _bytes(common | {"status": "clean"})
    observation = {
        "schema_version": 1,
        "reservation_sha256": reservation,
        "status": "observed",
        "research_accepted": False,
        "failure": None,
        "record_failures": [],
        "readiness_sha256": digest(ready),
        "cleanup_sha256": digest(cleanup),
        "stop_sent": True,
        "service": role_value(321),
        "client": role_value(654),
    }
    return {
        "service-ready.json": ready,
        "service-cleanup.json": cleanup,
        "service-stop.json": _bytes({"status": "requested"}),
        "process-pair.json": _bytes(observation),
    }, observation


def process_records(inputs, reservation, commands, deadlines):
    payloads, observation = _initial(inputs, reservation)
    payloads["process-pair-intent.json"] = _bytes(
        {
            "schema_version": 1,
            "reservation_sha256": reservation,
            "service_command_sha256": command_hash(commands[0]),
            "client_command_sha256": command_hash(commands[1]),
            "deadlines": deadlines,
        }
    )
    for role, command in zip(("service", "client"), commands, strict=True):
        value = observation[role]
        context = OperationalRoleContext(
            value["pid"], command, "http://127.0.0.1:54321"
        )
        payloads[f"{role}-intent.json"] = _bytes(
            {"command_sha256": command_hash(command)}
        )
        payloads[f"{role}-started.json"] = _bytes({"pid": value["pid"]})
        payloads[f"{role}-process.json"] = _bytes(value)
        payloads[f"{role}-role.json"] = (
            build_service_role(inputs, context, primary=inputs.primary)
            if role == "service"
            else build_client_role(inputs, context)
        )
    return payloads, ProcessObservation(_bytes(observation))
