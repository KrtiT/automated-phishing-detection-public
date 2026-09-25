"""Receipt binding and rehashed forgery tests over invented provenance only."""

import json
from hashlib import sha256

import pytest
import test_external_source_provenance as fixtures
from test_external_source_provenance import build, prepared_outputs, verify
from test_external_source_provenance_rejection import rejected

from automated_phishing_detection import phishvn
from automated_phishing_detection._checkpoint_codec import canonical_bytes

provenance_api = fixtures.provenance_api
provenance_case = fixtures.provenance_case
inputs = fixtures.inputs
published = fixtures.published
runner = fixtures.runner
verifier = fixtures.verifier
observed_worker = fixtures.observed_worker
completion = fixtures.completion
handoff_api = fixtures.handoff_api


def test_declared_counts_still_require_reconstruction(
    provenance_api, provenance_case, monkeypatch
):
    provenance = build(provenance_api, provenance_case)
    receipt = json.loads(provenance["external-source-reconstruction.json"])
    receipt["counts"]["published_split_counts"]["test"] += 1
    provenance["external-source-reconstruction.json"] = canonical_bytes(receipt)
    original, calls = phishvn.prepare_external_rows, []

    def observed(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(phishvn, "prepare_external_rows", observed)
    with rejected(provenance_api):
        verify(provenance_api, provenance_case, provenance)
    assert calls == [True]


def test_receipt_binds_all_predecessors_and_declared_counts(
    provenance_api, provenance_case
):
    case = provenance_case
    provenance = build(provenance_api, case)
    receipt = json.loads(provenance["external-source-reconstruction.json"])
    assert receipt == {
        "schema_version": 1,
        "execution": case.execution,
        "reservation_sha256": case.reservation,
        "provenance_sha256": {
            name: sha256(content).hexdigest()
            for name, content in provenance.items()
            if name != "external-source-reconstruction.json"
        },
        "prepared_sha256": {
            name: sha256(content).hexdigest()
            for name, content in prepared_outputs(case.prepared).items()
        },
        "counts": {
            "published_split_counts": case.decoded.published_split_counts,
            "mapping_counts": case.decoded.public_summary["mapping_counts"],
            "preparation": case.prepared.public_summary,
        },
    }


@pytest.mark.parametrize(
    "field",
    [
        "execution",
        "reservation_sha256",
        "counts",
        "prepared_sha256",
        "provenance_sha256",
        "schema_version",
        "extra",
    ],
)
def test_reconstruction_receipt_is_closed(provenance_api, provenance_case, field):
    provenance = build(provenance_api, provenance_case)
    receipt = json.loads(provenance["external-source-reconstruction.json"])
    receipt[field] = True
    provenance["external-source-reconstruction.json"] = canonical_bytes(receipt)
    with rejected(provenance_api):
        verify(provenance_api, provenance_case, provenance)


def test_rehashed_derived_forgery_cannot_replace_reconstruction(
    provenance_api, provenance_case
):
    case = provenance_case
    provenance = build(provenance_api, case)
    outputs = prepared_outputs(case.prepared)
    outputs["retained-test.jsonl"] = b""
    receipt = json.loads(provenance["external-source-reconstruction.json"])
    receipt["prepared_sha256"]["retained-test.jsonl"] = sha256(b"").hexdigest()
    provenance["external-source-reconstruction.json"] = canonical_bytes(receipt)
    with rejected(provenance_api):
        verify(provenance_api, case, provenance, outputs)


def test_rehashed_publisher_identity_cannot_replace_parent_pins(
    provenance_api, provenance_case
):
    case = provenance_case
    provenance = build(provenance_api, case)
    source = json.loads(provenance["publisher-source.json"])
    summary = json.loads(provenance["publisher-summary.json"])
    source["input_archive"]["archive_sha256"] = "0" * 64
    summary["input_archive"] = source["input_archive"]
    provenance["publisher-source.json"] = canonical_bytes(source)
    summary["private_sha256"]["publisher-source.json"] = sha256(
        provenance["publisher-source.json"]
    ).hexdigest()
    provenance["publisher-summary.json"] = canonical_bytes(summary)
    receipt = json.loads(provenance["external-source-reconstruction.json"])
    receipt["provenance_sha256"] = {
        name: sha256(content).hexdigest()
        for name, content in provenance.items()
        if name != "external-source-reconstruction.json"
    }
    provenance["external-source-reconstruction.json"] = canonical_bytes(receipt)
    with rejected(provenance_api):
        verify(provenance_api, case, provenance)
