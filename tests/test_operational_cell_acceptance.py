"""Pure joins use complete frozen-size HTTP records and retained parent context."""

from dataclasses import FrozenInstanceError, replace

import pytest
from operational_cell_acceptance_fixtures import changed_payload, http_case, verify
from operational_input_fixtures import candidates, case, manifests
from test_operational_cell_identity import api

from automated_phishing_detection._operational_cell_protocol import PRIVATE_NAMES

__all__ = ["candidates", "case", "manifests"]


@pytest.fixture(scope="module")
def working_case(case):
    return http_case(api(), case)


def test_full_http_working_retains_bytes_and_all_terminal_errors(working_case):
    result = verify(api(), working_case)
    assert dict(result.payloads) == working_case.payloads
    assert set(result.private_outputs) == set(PRIVATE_NAMES)
    assert result.summary["request_count"] == 10000
    assert result.summary["request_errors"] == 6
    assert result.summary["request_error_rate"] == 6 / 10000
    assert result.inputs is working_case.inputs
    assert result.accepted is working_case.arguments["accepted"]
    with pytest.raises(FrozenInstanceError):
        result.payloads = ()
    result.summary["request_count"] = 1
    result.private_outputs.clear()
    result.run.after_measured.admitted_requests = 0
    assert result.summary["request_count"] == 10000
    assert result.run.after_measured.admitted_requests == 11000
    assert len(result.private_outputs) == 16


@pytest.mark.parametrize(
    "name,key,value",
    [
        ("service-role.json", "pid", 999),
        ("client-role.json", "base_url", "http://127.0.0.1:1"),
        ("service-started.json", "pid", True),
        ("service-process.json", "exit_code", 17),
        ("client-intent.json", "command_sha256", "0" * 64),
        ("process-pair-intent.json", "reservation_sha256", "0" * 64),
        ("service-stop.json", "status", "not_requested"),
        ("service-cleanup.json", "workload", "transformer_only"),
        ("warmup.json", "stage", "measured_checkpoint"),
        ("measured.json", "measured_elapsed_ms", 0),
    ],
)
def test_rejects_record_join_mutations(working_case, name, key, value):
    changed = changed_payload(working_case, name, lambda row: row.update({key: value}))
    with pytest.raises(api().OperationalCellAcceptanceError):
        verify(api(), changed)


@pytest.mark.parametrize(
    "key,value",
    [
        ("status", "failed"),
        ("research_accepted", True),
        ("stop_sent", False),
        ("failure", "service_exit_unsuccessful"),
        ("record_failures", ["run.json"]),
        ("schema_version", True),
        ("extra", 1),
    ],
)
def test_saved_observation_cannot_replace_actual_expected(working_case, key, value):
    changed = changed_payload(
        working_case, "process-pair.json", lambda row: row.update({key: value})
    )
    with pytest.raises(api().OperationalCellAcceptanceError):
        verify(api(), changed)


def test_rejects_changed_expected_context_and_typed_request_projection(working_case):
    contexts = [
        {
            "expected_identity": working_case.arguments["expected_identity"]
            | {"extra": 1}
        },
        {"service_command": ("/invented/different",)},
        {
            "expected_deadlines": working_case.arguments["expected_deadlines"]
            | {"startup": 9}
        },
        {
            "inputs": replace(
                working_case.inputs, requests=working_case.inputs.requests[::-1]
            )
        },
    ]
    for changes in contexts:
        with pytest.raises(api().OperationalCellAcceptanceError):
            verify(api(), working_case, **changes)


@pytest.mark.parametrize("kind", ["missing", "extra", "duplicate", "list", "mutable"])
def test_exact_immutable_working_inventory(working_case, kind):
    pairs = tuple(working_case.payloads.items())
    variants = {
        "missing": pairs[:-1],
        "extra": (*pairs, ("extra", b"x")),
        "duplicate": (*pairs, pairs[0]),
        "list": list(pairs),
        "mutable": ((pairs[0][0], bytearray(pairs[0][1])), *pairs[1:]),
    }
    with pytest.raises(api().OperationalCellAcceptanceError):
        api().verify_working_cell(variants[kind], **working_case.arguments)
