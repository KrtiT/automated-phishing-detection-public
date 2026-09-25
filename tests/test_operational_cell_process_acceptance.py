"""Self-consistent saved failure markers never turn observed failure into success."""

import json
from types import SimpleNamespace

import pytest
from operational_cell_acceptance_fixtures import digest, http_case, verify
from operational_input_fixtures import candidates, case, manifests
from test_operational_cell_identity import api

from automated_phishing_detection._operational_process_records import (
    ProcessObservation,
    _bytes,
)

__all__ = ["candidates", "case", "manifests"]


@pytest.fixture(scope="module")
def working_case(case):
    return http_case(api(), case)


def changed_observation(original, mutate):
    payloads = dict(original.payloads)
    value = json.loads(payloads["process-pair.json"])
    mutate(value, payloads)
    content = _bytes(value)
    payloads["process-pair.json"] = content
    arguments = original.arguments | {"observation": ProcessObservation(content)}
    return SimpleNamespace(
        payloads=payloads, arguments=arguments, inputs=original.inputs
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("status", "failed"),
        ("research_accepted", True),
        ("stop_sent", False),
        ("failure", "failure"),
        ("record_failures", ["run.json"]),
        ("schema_version", True),
        ("extra", 1),
        ("readiness_sha256", "0" * 64),
        ("cleanup_sha256", "0" * 64),
    ],
)
def test_even_supplied_observation_must_be_complete(working_case, key, value):
    changed = changed_observation(
        working_case, lambda record, payloads: record.update({key: value})
    )
    with pytest.raises(api().OperationalCellAcceptanceError):
        verify(api(), changed)


@pytest.mark.parametrize("role", ["service", "client"])
@pytest.mark.parametrize(
    "key,value",
    [
        ("pid", True),
        ("exit_code", False),
        ("exit_code", 17),
        ("exit_code", None),
        ("exit_observed", False),
        ("forced", True),
        ("signals", [15]),
        ("stdout_sha256", "bad"),
        ("stderr_sha256", "A" * 64),
        ("extra", 1),
    ],
)
def test_owned_role_requires_exact_observed_zero_exit(working_case, role, key, value):
    changed = changed_observation(
        working_case, lambda record, payloads: record[role].update({key: value})
    )
    with pytest.raises(api().OperationalCellAcceptanceError):
        verify(api(), changed)


@pytest.mark.parametrize(
    "name,key,value",
    [
        ("service-ready.json", "port", True),
        ("service-ready.json", "port", 0),
        ("service-ready.json", "port", 65536),
        ("service-ready.json", "host", "localhost"),
        ("service-ready.json", "pid", 654),
        ("service-ready.json", "schema_version", True),
        ("service-cleanup.json", "workload", "other"),
        ("service-cleanup.json", "status", "failed"),
        ("service-cleanup.json", "port", 1),
    ],
)
def test_hash_matched_lifecycle_still_requires_owned_scheduled_identity(
    working_case, name, key, value
):
    def mutate(record, payloads):
        lifecycle = json.loads(payloads[name])
        lifecycle[key] = value
        payloads[name] = _bytes(lifecycle)
        digest_key = (
            "readiness_sha256" if name == "service-ready.json" else "cleanup_sha256"
        )
        record[digest_key] = digest(payloads[name])

    with pytest.raises(api().OperationalCellAcceptanceError):
        verify(api(), changed_observation(working_case, mutate))


@pytest.mark.parametrize("value", [True, 0, -1, float("inf"), float("nan")])
def test_no_invalid_or_implicit_deadline(working_case, value):
    deadlines = working_case.arguments["expected_deadlines"] | {"startup": value}
    with pytest.raises(api().OperationalCellAcceptanceError):
        verify(api(), working_case, expected_deadlines=deadlines)


def test_symbolic_errors_do_not_leak_payload_values(working_case):
    changed = changed_observation(
        working_case,
        lambda record, payloads: record.update({"failure": "private-url.example"}),
    )
    with pytest.raises(api().OperationalCellAcceptanceError) as caught:
        verify(api(), changed)
    assert str(caught.value) == "invalid_operational_cell"
