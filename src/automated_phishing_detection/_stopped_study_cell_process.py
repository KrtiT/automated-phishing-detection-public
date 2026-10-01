"""Authenticate recorded cancellation; neither event timing nor cause is inferred."""

from hashlib import sha256

from . import _operational_cell_process_records as process
from . import _study_run_schema as schema
from ._checkpoint_codec import canonical_bytes
from ._study_execution_policy import DEADLINES


def _pair(content, reservation):
    value = schema.operational.loads(content, canonical=False)
    schema.require(process._bytes(value) == content)
    schema.keys(value, process._OBSERVATION_FIELDS)
    schema.require(
        type(value["schema_version"]) is int and value["schema_version"] == 1
    )
    schema.require(value["reservation_sha256"] == reservation)
    schema.require(
        value["status"] == "failed" and value["failure"] == "parent_cancelled"
    )
    schema.require(value["research_accepted"] is False and value["stop_sent"] is True)
    schema.require(
        type(value["record_failures"]) is list and not value["record_failures"]
    )
    process._role(value["service"])
    schema.same(
        value["client"],
        dict(
            pid=None,
            exit_code=None,
            exit_observed=False,
            forced=False,
            signals=[],
            stdout_sha256=None,
            stderr_sha256=None,
        ),
    )
    return value


def _intent(attempt, reservation, entry):
    content = attempt["attempt/process-pair-intent.json"]
    intent = schema.load(content)
    schema.operational.digest(intent["client_command_sha256"])
    expected = dict(
        schema_version=1,
        reservation_sha256=reservation,
        service_command_sha256=entry["command_sha256"],
        client_command_sha256=intent["client_command_sha256"],
        deadlines=dict(DEADLINES),
    )
    schema.require(content == process._bytes(expected))


def _service(attempt, observation, entry):
    service = observation["service"]
    schema.require(service["pid"] == entry["launched_pid"])
    values = {
        "service-intent.json": dict(command_sha256=entry["command_sha256"]),
        "service-started.json": dict(pid=service["pid"]),
        "service-process.json": service,
    }
    for name, value in values.items():
        schema.require(attempt[f"attempt/{name}"] == process._bytes(value))


def _lifecycle(attempt, observation, cell):
    content = attempt["attempt/service-ready.json"]
    ready = schema.load(content)
    schema.require(type(ready["port"]) is int and 1 <= ready["port"] <= 65535)
    common = dict(
        schema_version=1, pid=observation["service"]["pid"], workload=cell.workload
    )
    schema.require(
        content
        == process._bytes(
            common | dict(status="ready", host="127.0.0.1", port=ready["port"])
        )
    )
    schema.require(
        observation["readiness_sha256"] in (None, sha256(content).hexdigest())
    )
    cleanup = attempt["attempt/service-cleanup.json"]
    schema.require(cleanup == process._bytes(common | dict(status="clean")))
    schema.require(observation["cleanup_sha256"] == sha256(cleanup).hexdigest())
    schema.require(
        attempt["attempt/service-stop.json"] == process._bytes(dict(status="requested"))
    )
    return f"http://127.0.0.1:{ready['port']}"


def _role(attempt, cell, metadata, entry, frame, endpoint):
    expected = (
        dict(
            schema_version=1,
            protocol="operational-role-v1",
            role="service",
            binding_sha256=frame.cell_binding_sha256,
            pid=entry["launched_pid"],
            command_sha256=entry["command_sha256"],
            base_url=endpoint,
            workload=cell.workload,
        )
        | metadata["primary"]
    )
    schema.require(attempt["attempt/service-role.json"] == canonical_bytes(expected))


def authenticate_process(attempt, reservation, cell, metadata, entry, frame):
    observation = _pair(attempt["attempt/process-pair.json"], reservation)
    _intent(attempt, reservation, entry)
    _service(attempt, observation, entry)
    endpoint = _lifecycle(attempt, observation, cell)
    _role(attempt, cell, metadata, entry, frame, endpoint)
