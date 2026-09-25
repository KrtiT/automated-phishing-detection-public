"""Join historical process schemas to one parent's actual retained observation."""

import math
from hashlib import sha256

from . import _operational_input_schema as schema
from ._operational_process_records import ProcessObservation, _bytes
from ._process_support import command_hash
from .operational_role_records import OperationalRoleContext, verify_role_record

_OBSERVATION_FIELDS = frozenset(
    (
        "schema_version",
        "reservation_sha256",
        "status",
        "research_accepted",
        "failure",
        "record_failures",
        "readiness_sha256",
        "cleanup_sha256",
        "stop_sent",
        "service",
        "client",
    )
)


def _role(value):
    schema.keys(
        value,
        {
            "pid",
            "exit_code",
            "exit_observed",
            "forced",
            "signals",
            "stdout_sha256",
            "stderr_sha256",
        },
    )
    schema.require(type(value["pid"]) is int and value["pid"] > 0)
    schema.require(type(value["exit_code"]) is int and value["exit_code"] == 0)
    schema.require(value["exit_observed"] is True and value["forced"] is False)
    schema.require(type(value["signals"]) is list and value["signals"] == [])
    schema.digest(value["stdout_sha256"])
    schema.digest(value["stderr_sha256"])


def _observation(content, observation, reservation):
    schema.require(type(observation) is ProcessObservation)
    schema.require(type(observation.record) is bytes and content == observation.record)
    value = schema.loads(content, canonical=False)
    schema.require(_bytes(value) == content)
    schema.keys(value, _OBSERVATION_FIELDS)
    schema.require(
        type(value["schema_version"]) is int and value["schema_version"] == 1
    )
    schema.require(value["reservation_sha256"] == reservation)
    schema.require(
        value["status"] == "observed" and value["research_accepted"] is False
    )
    schema.require(value["failure"] is None and value["stop_sent"] is True)
    schema.require(
        type(value["record_failures"]) is list and value["record_failures"] == []
    )
    for role in ("service", "client"):
        _role(value[role])
    schema.require(value["service"]["pid"] != value["client"]["pid"])
    schema.digest(value["readiness_sha256"])
    schema.digest(value["cleanup_sha256"])
    return value


def _lifecycle(payloads, observation, workload):
    for name, key in (
        ("service-ready.json", "readiness_sha256"),
        ("service-cleanup.json", "cleanup_sha256"),
    ):
        schema.require(sha256(payloads[name]).hexdigest() == observation[key])
    ready = schema.loads(payloads["service-ready.json"], canonical=False)
    schema.keys(ready, {"schema_version", "pid", "workload", "status", "host", "port"})
    schema.require(type(ready["port"]) is int and 1 <= ready["port"] <= 65535)
    common = {
        "schema_version": 1,
        "pid": observation["service"]["pid"],
        "workload": workload,
    }
    expected = common | {"status": "ready", "host": "127.0.0.1", "port": ready["port"]}
    schema.require(payloads["service-ready.json"] == _bytes(expected))
    schema.require(
        payloads["service-cleanup.json"] == _bytes(common | {"status": "clean"})
    )
    schema.require(payloads["service-stop.json"] == _bytes({"status": "requested"}))
    return f"http://127.0.0.1:{ready['port']}"


def _intents(payloads, reservation, commands, deadlines):
    schema.keys(deadlines, {"startup", "shutdown", "terminate", "kill"})
    schema.require(
        all(
            type(value) in (int, float) and math.isfinite(value) and value > 0
            for value in deadlines.values()
        )
    )
    expected = {
        "schema_version": 1,
        "reservation_sha256": reservation,
        "service_command_sha256": command_hash(commands[0]),
        "client_command_sha256": command_hash(commands[1]),
        "deadlines": deadlines,
    }
    schema.require(payloads["process-pair-intent.json"] == _bytes(expected))


def verify_process_records(
    payloads,
    *,
    inputs,
    observation,
    reservation,
    service_command,
    client_command,
    expected_deadlines,
):
    value = _observation(payloads["process-pair.json"], observation, reservation)
    endpoint = _lifecycle(payloads, value, inputs.cell.workload)
    commands = service_command, client_command
    _intents(payloads, reservation, commands, expected_deadlines)
    for role, command in zip(("service", "client"), commands, strict=True):
        record = value[role]
        context = OperationalRoleContext(record["pid"], command, endpoint)
        for name, expected in (
            ("intent", {"command_sha256": command_hash(command)}),
            ("started", {"pid": record["pid"]}),
            ("process", record),
        ):
            schema.require(payloads[f"{role}-{name}.json"] == _bytes(expected))
        verify_role_record(
            payloads[f"{role}-role.json"], inputs=inputs, context=context, role=role
        )
