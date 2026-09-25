"""Closed provenance inventories, independent expectations and mutable-view rejection."""

import json
from dataclasses import replace
from hashlib import sha256

import pytest
import test_external_source_provenance as provenance_fixtures
from test_external_source_provenance import (
    NAMES,
    PREPARED_NAMES,
    build,
    prepared_outputs,
    verify,
)

from automated_phishing_detection import phishvn, protocol_preflight

provenance_api = provenance_fixtures.provenance_api
provenance_case = provenance_fixtures.provenance_case
inputs = provenance_fixtures.inputs
published = provenance_fixtures.published
runner = provenance_fixtures.runner
verifier = provenance_fixtures.verifier
observed_worker = provenance_fixtures.observed_worker
completion = provenance_fixtures.completion
handoff_api = provenance_fixtures.handoff_api


def rejected(api):
    return pytest.raises(
        api.ExternalSourceProvenanceError, match="^invalid_external_source_provenance$"
    )


@pytest.mark.parametrize("name", sorted(NAMES))
def test_missing_provenance_is_rejected(provenance_api, provenance_case, name):
    provenance = build(provenance_api, provenance_case)
    del provenance[name]
    with rejected(provenance_api):
        verify(provenance_api, provenance_case, provenance)


@pytest.mark.parametrize("name", sorted(NAMES))
def test_mutable_payload_is_rejected(provenance_api, provenance_case, name):
    provenance = build(provenance_api, provenance_case)
    provenance[name] = bytearray(provenance[name])
    with rejected(provenance_api):
        verify(provenance_api, provenance_case, provenance)


@pytest.mark.parametrize("target", ["provenance", "prepared"])
def test_additional_payload_is_rejected(provenance_api, provenance_case, target):
    provenance = build(provenance_api, provenance_case)
    outputs = prepared_outputs(provenance_case.prepared)
    (provenance if target == "provenance" else outputs)["extra.json"] = b"{}\n"
    with rejected(provenance_api):
        verify(provenance_api, provenance_case, provenance, outputs)


@pytest.mark.parametrize("name", sorted(PREPARED_NAMES))
@pytest.mark.parametrize("change", ["missing", "changed", "mutable"])
def test_prepared_outputs_are_exact(provenance_api, provenance_case, name, change):
    provenance = build(provenance_api, provenance_case)
    outputs = prepared_outputs(provenance_case.prepared)
    if change == "missing":
        del outputs[name]
    else:
        outputs[name] = (
            outputs[name] + b" " if change == "changed" else bytearray(outputs[name])
        )
    with rejected(provenance_api):
        verify(provenance_api, provenance_case, provenance, outputs)


@pytest.mark.parametrize(
    "field,value",
    [
        ("expected_handoff", b"substituted"),
        ("expected_overlap", b"substituted"),
        ("suffix_rules_sha256", "0" * 64),
        ("reservation_sha256", "0" * 64),
        ("execution", {"kind": "different_parent"}),
    ],
)
def test_parent_expectations_are_not_taken_from_saved_receipts(
    provenance_api, provenance_case, field, value
):
    provenance = build(provenance_api, provenance_case)
    with rejected(provenance_api):
        verify(provenance_api, provenance_case, provenance, **{field: value})


@pytest.mark.parametrize(
    "field,value",
    [
        ("expected_handoff", b"changed"),
        ("expected_overlap", b"changed"),
        ("suffix_rules_sha256", "0" * 64),
    ],
)
def test_expected_inputs_checked_before_parsing(
    provenance_api, provenance_case, monkeypatch, field, value
):
    provenance = build(provenance_api, provenance_case)

    def forbidden(*args, **kwargs):
        pytest.fail("parsed before checking parent expectations")

    monkeypatch.setattr(json, "loads", forbidden)
    with rejected(provenance_api):
        verify(provenance_api, provenance_case, provenance, **{field: value})


@pytest.mark.parametrize("view", ["rows", "split_counts", "summary", "private"])
def test_builder_rejects_mutated_decoder_views(provenance_api, provenance_case, view):
    case = provenance_case
    if view == "rows":
        case.decoded = replace(case.decoded, rows=case.decoded.rows[:-1])
    elif view == "split_counts":
        case.decoded.published_split_counts["test"] += 1
    elif view == "summary":
        case.decoded.public_summary["mapping_counts"]["mapped"] += 1
    else:
        case.decoded.private_outputs["publisher-source.json"] += b" "
    with rejected(provenance_api):
        build(provenance_api, case)


@pytest.mark.parametrize("view", ["rows", "summary", "private", "without_overlap"])
def test_builder_rejects_prepared_view_inconsistency(
    provenance_api, provenance_case, view
):
    case = provenance_case
    if view == "rows":
        case.prepared = replace(case.prepared, retained=case.prepared.retained[:-1])
    elif view == "summary":
        case.prepared.public_summary["retained_test_rows"] += 1
    elif view == "private":
        case.prepared.private_outputs["retained-test.jsonl"] += b" "
    else:
        case.prepared = phishvn.prepare_external_rows(
            case.decoded.rows,
            published_split_counts=case.decoded.published_split_counts,
            test_split="test",
            suffix_rules=protocol_preflight.parse_suffix_rules(case.suffix.decode()),
            phiusiil_domains=frozenset(),
        )
    with rejected(provenance_api):
        build(provenance_api, case)


def test_even_self_consistent_psl_substitution_breaks_internal_chain(
    provenance_api, provenance_case
):
    case = provenance_case
    provenance = build(provenance_api, case)
    changed = case.suffix + b"example\n"
    with rejected(provenance_api):
        build(provenance_api, case, suffix_rules=changed)
    provenance["suffix-rules.dat"] = changed
    with rejected(provenance_api):
        verify(
            provenance_api,
            case,
            provenance,
            suffix_rules_sha256=sha256(changed).hexdigest(),
        )


@pytest.mark.parametrize(
    "field,value", [("execution", []), ("reservation_sha256", True)]
)
def test_invalid_parent_context_is_symbolic(
    provenance_api, provenance_case, field, value
):
    provenance = build(provenance_api, provenance_case)
    with rejected(provenance_api):
        build(provenance_api, provenance_case, **{field: value})
    with rejected(provenance_api):
        verify(provenance_api, provenance_case, provenance, **{field: value})
