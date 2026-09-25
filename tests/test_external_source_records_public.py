"""Closed outer payload hashes preserve pure nested composition semantics."""

import builtins
import json
from pathlib import Path

import pytest
import test_external_source_records as fixtures

from automated_phishing_detection import (
    execution_preflight,
    external_producer,
    phishvn,
    saved_external_evidence,
)
from automated_phishing_detection._checkpoint_codec import canonical_bytes

inputs = fixtures.inputs
published = fixtures.published
runner = fixtures.runner
verifier = fixtures.verifier
observed_worker = fixtures.observed_worker
completion = fixtures.completion
handoff_api = fixtures.handoff_api
records_case = fixtures.records_case


@pytest.mark.parametrize("change", ["missing", "extra", "nonbytes", "wrong_type"])
def test_public_builder_requires_exact_thirty_six_byte_payloads(records_case, change):
    outputs = records_case.outputs.copy()
    if change == "missing":
        outputs.pop("all-scores.jsonl")
    elif change == "extra":
        outputs["unknown"] = b"unexpected"
    elif change == "nonbytes":
        outputs["all-scores.jsonl"] = bytearray(b"mutable")
    else:
        outputs = []
    with pytest.raises(fixtures.api().ExternalSourceExecutionError):
        fixtures.build(records_case, private_outputs=outputs)


@pytest.mark.parametrize(
    "field,value",
    [
        ("protected_evaluation_authorized", True),
        ("protected_evaluation_authorized", 0),
        ("source_binding", "authenticated"),
        ("private_sha256", {}),
    ],
)
def test_nested_composition_flags_and_exact_scientific_inventory_are_checked(
    records_case, field, value
):
    composition = records_case.composition | {field: value}
    with pytest.raises(fixtures.api().ExternalSourceExecutionError):
        fixtures.build(records_case, composition=composition)


@pytest.mark.parametrize(
    "change", ["extra", "wrong_hash", "changed_science", "nonfinite"]
)
def test_nested_summary_cannot_conceal_changed_scientific_payloads(
    records_case, change
):
    composition = json.loads(canonical_bytes(records_case.composition))
    outputs = records_case.outputs.copy()
    if change == "extra":
        composition["private_sha256"]["publisher-source.json"] = "a" * 64
    elif change == "wrong_hash":
        composition["private_sha256"]["all-scores.jsonl"] = "a" * 64
    elif change == "changed_science":
        outputs["all-scores.jsonl"] = b"changed"
    else:
        composition["row_count"] = float("nan")
    with pytest.raises(fixtures.api().ExternalSourceExecutionError):
        fixtures.build(records_case, composition=composition, private_outputs=outputs)


@pytest.mark.parametrize(
    "content",
    [
        b"{}",
        b"[]\n",
        b'{"duplicate":1,"duplicate":1}\n',
        b'{"value":NaN}\n',
        b"private-canary",
    ],
)
def test_publisher_summary_requires_canonical_science_dictionary(records_case, content):
    outputs = records_case.outputs | {"publisher-summary.json": content}
    with pytest.raises(fixtures.api().ExternalSourceExecutionError) as rejected:
        fixtures.build(records_case, private_outputs=outputs)
    assert "private-canary" not in str(rejected.value)


@pytest.mark.parametrize("value", [None, True, "invalid", "A" * 64])
def test_outer_reservation_identity_requires_exact_digest(records_case, value):
    with pytest.raises(fixtures.api().ExternalSourceExecutionError):
        fixtures.build(records_case, reservation_sha256=value)


@pytest.mark.parametrize("change", ["missing", "extra", "mismatch"])
def test_supplied_identity_must_match_retained_source_context(records_case, change):
    identity = fixtures.api().external_identity(
        records_case.binding, records_case.profile, records_case.handoff
    )
    if change == "missing":
        identity.pop("internal_overlap_sha256")
    elif change == "extra":
        identity["unexpected"] = True
    else:
        identity["archive_sha256"] = "a" * 64
    with pytest.raises(fixtures.api().ExternalSourceExecutionError):
        fixtures.build(records_case, identity=identity)


def test_scientific_tuple_projection_preserves_exact_canonical_inner_bytes(
    records_case,
):
    composition = records_case.composition | {"invented_gates": ({"decision": None},)}
    public = fixtures.build(records_case, composition=composition)
    assert canonical_bytes(public["composition"]) == canonical_bytes(composition)


def test_outer_builders_do_no_io_runtime_probes_preparation_or_inference(
    records_case, monkeypatch
):
    module = fixtures.api()

    def forbidden(*args, **kwargs):
        pytest.fail("byte-only record construction performed external work")

    for owner, name in (
        (builtins, "open"),
        (Path, "read_bytes"),
        (Path, "stat"),
        (Path, "open"),
        (execution_preflight, "recheck_binding"),
        (execution_preflight, "_read_regular"),
        (phishvn, "prepare_external_rows"),
        (external_producer, "produce_external_evidence"),
        (saved_external_evidence, "reconstruct_external_evidence"),
    ):
        monkeypatch.setattr(owner, name, forbidden)
    identity = module.external_identity(
        records_case.binding, records_case.profile, records_case.handoff
    )
    assert (
        fixtures.build(records_case, identity=identity)[
            "protected_evaluation_authorized"
        ]
        is False
    )
