"""Rehashed historical cell evidence preserves original schemas and deadlines."""

import json

import pytest
from operational_input_fixtures import candidates, manifests
from stopped_study_authorization_fixtures import (
    make_stopped,
    refresh_accounting,
    repin_evidence,
    verify,
)
from study_run_record_fixtures import prepared

from automated_phishing_detection import _adopted_study_records as records
from automated_phishing_detection._checkpoint_codec import canonical_bytes

__all__ = ["candidates", "manifests", "prepared"]


@pytest.mark.parametrize(
    "target,field,value",
    [
        ("observation", "unknown", True),
        ("observation", "schema_version", True),
        ("observation", "research_accepted", True),
        ("observation", "stop_sent", False),
        ("observation", "record_failures", ["failure"]),
        ("observation", "reservation_sha256", "0" * 64),
        ("observation", "status", "failed"),
        ("observation", "failure", "cancelled"),
        ("service", "forced", True),
        ("service", "signals", [15]),
        ("service", "pid", 999),
        ("client", "pid", 321),
        ("client", "exit_code", 1),
        ("service", "stdout_sha256", "bad"),
        ("intent", "unknown", True),
        ("intent", "schema_version", True),
        ("intent", "service_command_sha256", "0" * 64),
        ("intent", "client_command_sha256", "0" * 64),
        ("intent", "reservation_sha256", "0" * 64),
        ("deadlines", "startup", 301),
        ("deadlines", "shutdown", 181),
        ("deadlines", "terminate", 11),
        ("deadlines", "kill", 11),
    ],
)
def test_rehashed_process_evidence_preserves_contract(
    prepared, manifests, target, field, value
):
    case = make_stopped(prepared, manifests)
    intent = target in ("intent", "deadlines")
    key = "pair_intent_bytes" if intent else "observation_bytes"
    filename = "process-pair-intent.json" if intent else "process-pair.json"
    bundle = case.accounting["authorization_ledger"]["cell_acceptances"][0]
    record = json.loads(records.decoded(bundle[key]))
    destination = (
        record[target] if target in ("service", "client", "deadlines") else record
    )
    destination[field] = value
    repin_evidence(case, key, filename, record)
    with pytest.raises(ValueError):
        verify(case)


@pytest.mark.parametrize(
    "key,field,value",
    [
        ("descriptor_bytes", "root_reservation_sha256", "0" * 64),
        ("descriptor_bytes", "accepted_inputs_sha256", "0" * 64),
        ("descriptor_bytes", "unknown", True),
        ("binding_bytes", "descriptor_sha256", "0" * 64),
        ("binding_bytes", "cell_reservation_sha256", "0" * 64),
        ("binding_bytes", "unknown", True),
    ],
)
def test_binding_and_descriptor_remain_original(prepared, manifests, key, field, value):
    case = make_stopped(prepared, manifests)
    bundle = case.accounting["authorization_ledger"]["cell_acceptances"][0]
    record = json.loads(records.decoded(bundle[key])) | {field: value}
    bundle[key] = records.encoded(canonical_bytes(record))
    refresh_accounting(case)
    with pytest.raises(ValueError):
        verify(case)


@pytest.mark.parametrize(
    "name",
    [
        "attempt/process-pair.json",
        "attempt/evidence/process-pair.json",
        "attempt/process-pair-intent.json",
        "attempt/evidence/process-pair-intent.json",
    ],
)
def test_snapshot_hashes_match_retained_process_bytes(prepared, manifests, name):
    case = make_stopped(prepared, manifests)
    case.scientific["cells"][0]["snapshot_sha256"][name] = "0" * 64
    refresh_accounting(case)
    with pytest.raises(ValueError):
        verify(case)
