"""Validate historical process bytes without representing a new observation."""

from . import _operational_cell_process_records as process
from . import _study_run_schema as schema
from ._study_execution_policy import DEADLINES


def _observation(content, reservation):
    value = schema.operational.loads(content, canonical=False)
    schema.require(process._bytes(value) == content)
    schema.keys(value, process._OBSERVATION_FIELDS)
    schema.require(
        type(value["schema_version"]) is int and value["schema_version"] == 1
    )
    schema.require(value["reservation_sha256"] == reservation)
    schema.require(
        value["status"] == "observed" and value["research_accepted"] is False
    )
    schema.require(value["failure"] is None and value["stop_sent"] is True)
    schema.require(
        type(value["record_failures"]) is list and not value["record_failures"]
    )
    for role in ("service", "client"):
        process._role(value[role])
    schema.require(value["service"]["pid"] != value["client"]["pid"])
    schema.operational.digest(value["readiness_sha256"])
    schema.operational.digest(value["cleanup_sha256"])
    return value


def historical_process(payloads, reservation):
    observation = _observation(payloads["observation_bytes"], reservation)
    intent = schema.load(payloads["pair_intent_bytes"])
    schema.keys(
        intent,
        {
            "schema_version",
            "reservation_sha256",
            "service_command_sha256",
            "client_command_sha256",
            "deadlines",
        },
    )
    schema.require(
        type(intent["schema_version"]) is int and intent["schema_version"] == 1
    )
    schema.require(intent["reservation_sha256"] == reservation)
    schema.deadlines(intent["deadlines"])
    schema.same(intent["deadlines"], dict(DEADLINES))
    for role in ("service", "client"):
        schema.operational.digest(intent[f"{role}_command_sha256"])
    return observation, intent
