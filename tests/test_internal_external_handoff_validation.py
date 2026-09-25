"""Closed handoff schemas and caller-supplied identity checks."""

import json
from dataclasses import replace
from hashlib import sha256

import pytest
import test_internal_external_handoff as fixtures

from automated_phishing_detection._checkpoint_codec import canonical_bytes

completion = fixtures.completion
handoff_api = fixtures.handoff_api
inputs = fixtures.inputs
observed_worker = fixtures.observed_worker
published = fixtures.published
runner = fixtures.runner
verifier = fixtures.verifier


@pytest.fixture
def payloads(handoff_api, completion):
    return handoff_api.build_internal_handoff(completion)


def verify_changed(api, payloads, content):
    return api.verify_internal_handoff(
        content,
        payloads.overlap_bytes,
        expected_handoff_sha256=sha256(content).hexdigest(),
    )


@pytest.mark.parametrize(
    ("location", "field", "value"),
    [
        ("", "schema_version", True),
        ("", "schema_version", 1.0),
        ("", "kind", "standalone-acceptance"),
        ("", "extra", "rejected"),
        ("", "execution", []),
        ("", "worker", []),
        ("", "snapshot_sha256", []),
        ("execution", "kind", "external_evaluation"),
        ("execution", "source_interface", "saved_domain_set"),
        ("execution", "scientific_checkpoint_protocol", "other"),
        ("execution", "revision", "a" * 39),
        ("execution", "execution_contract_sha256", "A" * 64),
        ("execution", "source_spec_sha256", "a" * 64),
        ("execution", "reservation_sha256", "a" * 64),
        ("execution", "partition_sha256", "a" * 64),
        ("execution", "source_csv_sha256", "a" * 64),
        ("execution", "extra", None),
        ("worker", "command_sha256", True),
        ("worker", "stdout_sha256", None),
        ("worker", "stderr_sha256", "a" * 63),
        ("worker", "extra", None),
        ("worker.exit", "pid", True),
        ("worker.exit", "pid", 1.0),
        ("worker.exit", "pid", 0),
        ("worker.exit", "pid", -1),
        ("worker.exit", "exit_observed", 1),
        ("worker.exit", "exit_observed", False),
        ("worker.exit", "exit_code", False),
        ("worker.exit", "exit_code", 0.0),
        ("worker.exit", "exit_code", None),
        ("worker.exit", "exit_code", 17),
        ("worker.exit", "extra", None),
        ("snapshot_sha256", "public-summary.json", "A" * 64),
        ("snapshot_sha256", "extra", "a" * 64),
    ],
)
def test_rehashed_envelope_schema_or_link_mutations_rejected(
    handoff_api, payloads, location, field, value
):
    envelope = json.loads(payloads.handoff_bytes)
    target = envelope
    for component in location.split(".") if location else ():
        target = target[component]
    target[field] = value
    with pytest.raises(ValueError):
        verify_changed(handoff_api, payloads, canonical_bytes(envelope))


@pytest.mark.parametrize(
    "location", ["", "execution", "worker", "worker.exit", "snapshot_sha256"]
)
def test_missing_envelope_fields_rejected(handoff_api, payloads, location):
    envelope = json.loads(payloads.handoff_bytes)
    target = envelope
    for component in location.split(".") if location else ():
        target = target[component]
    target.pop(next(iter(target)))
    with pytest.raises(ValueError):
        verify_changed(handoff_api, payloads, canonical_bytes(envelope))


@pytest.mark.parametrize("mutation", ["whitespace", "duplicate", "nan", "list"])
def test_noncanonical_or_ambiguous_json_rejected(handoff_api, payloads, mutation):
    mutations = {
        "whitespace": payloads.handoff_bytes + b" ",
        "duplicate": b'{"schema_version":1,' + payloads.handoff_bytes[1:],
        "nan": payloads.handoff_bytes.replace(
            b'"schema_version":1', b'"schema_version":NaN'
        ),
        "list": b"[]\n",
    }
    with pytest.raises(ValueError):
        verify_changed(handoff_api, payloads, mutations[mutation])


@pytest.mark.parametrize("location", ["handoff", "overlap", "expected"])
@pytest.mark.parametrize("value", [None, True, "bytes", bytearray(b"bytes")])
def test_validator_rejects_nonexact_input_types(handoff_api, payloads, location, value):
    values = {
        "handoff": payloads.handoff_bytes,
        "overlap": payloads.overlap_bytes,
        "expected": sha256(payloads.handoff_bytes).hexdigest(),
    }
    values[location] = value
    with pytest.raises(ValueError):
        handoff_api.verify_internal_handoff(
            values["handoff"],
            values["overlap"],
            expected_handoff_sha256=values["expected"],
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("pid", True),
        ("pid", 0),
        ("exit_observed", 1),
        ("exit_observed", False),
        ("exit_code", False),
        ("exit_code", 0.0),
        ("exit_code", 17),
    ],
)
def test_builder_rejects_invalid_typed_observation(
    handoff_api, completion, field, value
):
    worker = replace(
        completion.worker, exit=replace(completion.worker.exit, **{field: value})
    )
    with pytest.raises(ValueError):
        handoff_api.build_internal_handoff(replace(completion, worker=worker))


@pytest.mark.parametrize("field", ["command_sha256", "stdout_sha256", "stderr_sha256"])
def test_builder_rejects_invalid_worker_hash(handoff_api, completion, field):
    worker = replace(completion.worker, **{field: "invalid"})
    with pytest.raises(ValueError):
        handoff_api.build_internal_handoff(replace(completion, worker=worker))


def test_builder_rejects_nonexact_domain_strings(handoff_api, completion):
    class InventedDomain(str):
        pass

    snapshot = replace(
        completion.snapshot,
        overlap_domains=frozenset(
            InventedDomain(domain) for domain in completion.snapshot.overlap_domains
        ),
    )
    with pytest.raises(ValueError):
        handoff_api.build_internal_handoff(replace(completion, snapshot=snapshot))
