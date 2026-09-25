"""Six-file provenance reconstruction over invented publisher and internal bytes."""

import importlib
import importlib.util
from hashlib import sha256
from types import SimpleNamespace

import pytest
import test_internal_external_handoff as internal_fixtures
from phishvn_source_fixtures import decode, members, record

from automated_phishing_detection import phishvn, protocol_preflight
from automated_phishing_detection._checkpoint_codec import canonical_bytes
from automated_phishing_detection._phishvn_archive import PhishVNSourcePins

inputs = internal_fixtures.inputs
published = internal_fixtures.published
runner = internal_fixtures.runner
verifier = internal_fixtures.verifier
observed_worker = internal_fixtures.observed_worker
completion = internal_fixtures.completion
handoff_api = internal_fixtures.handoff_api

NAMES = frozenset(
    {
        "publisher-source.json",
        "publisher-summary.json",
        "suffix-rules.dat",
        "internal-source-overlap.json",
        "internal-source-handoff.json",
        "external-source-reconstruction.json",
    }
)
PREPARED_NAMES = frozenset(
    {
        "retained-test.jsonl",
        "quarantine.jsonl",
        "inventory.json",
        "preparation-summary.json",
    }
)


@pytest.fixture
def provenance_api():
    name = "automated_phishing_detection.external_source_provenance"
    assert importlib.util.find_spec(name), "missing six-file external provenance"
    return importlib.import_module(name)


def prepared_outputs(prepared):
    return prepared.private_outputs | {
        "preparation-summary.json": canonical_bytes(prepared.public_summary)
    }


def _publisher_rows():
    return [
        record("train", "train", url="https://shared-domain.com/train"),
        record("validation", "val", url="https://validation.com/"),
        record("phishing", url="https://phishing.com/", channel="opaque-canary"),
        record("overlap", url="https://quarantined-label.com/private"),
        record("cross-split", url="https://shared-domain.com/test"),
        record(
            "control",
            url="https://control.com/",
            source="tranco",
            tier="silver",
            label="benign",
        ),
    ]


@pytest.fixture
def provenance_case(completion, handoff_api):
    handoff = handoff_api.build_internal_handoff(completion)
    decoded = decode(members(_publisher_rows()))
    suffix = b"com\nco.uk\n"
    prepared = phishvn.prepare_external_rows(
        decoded.rows,
        published_split_counts=decoded.published_split_counts,
        test_split="test",
        suffix_rules=protocol_preflight.parse_suffix_rules(suffix.decode()),
        phiusiil_domains=completion.snapshot.overlap_domains,
    )
    return SimpleNamespace(
        decoded=decoded,
        prepared=prepared,
        suffix=suffix,
        handoff=handoff,
        execution={"kind": "invented_external", "revision": "c" * 40},
        reservation="d" * 64,
        pins=PhishVNSourcePins(**decoded.public_summary["input_archive"]),
    )


def build(api, case, **changes):
    arguments = (
        dict(
            suffix_rules=case.suffix,
            internal_handoff=case.handoff.handoff_bytes,
            internal_overlap=case.handoff.overlap_bytes,
            execution=case.execution,
            reservation_sha256=case.reservation,
        )
        | changes
    )
    return api.build_external_provenance(case.decoded, case.prepared, **arguments)


def verify(api, case, provenance, outputs=None, **changes):
    arguments = (
        dict(
            pins=case.pins,
            expected_handoff=case.handoff.handoff_bytes,
            expected_overlap=case.handoff.overlap_bytes,
            execution=case.execution,
            reservation_sha256=case.reservation,
            suffix_rules_sha256=sha256(case.suffix).hexdigest(),
        )
        | changes
    )
    return api.verify_external_provenance(
        provenance,
        prepared_outputs(case.prepared) if outputs is None else outputs,
        **arguments,
    )


def test_roundtrip_preserves_exact_six_payloads_and_preparation(
    provenance_api, provenance_case
):
    case = provenance_case
    provenance = build(provenance_api, case)
    assert set(provenance) == NAMES
    assert all(type(content) is bytes for content in provenance.values())
    assert (
        provenance["publisher-source.json"]
        == case.decoded.private_outputs["publisher-source.json"]
    )
    assert provenance["publisher-summary.json"] == canonical_bytes(
        case.decoded.public_summary
    )
    assert provenance["internal-source-overlap.json"] == case.handoff.overlap_bytes
    assert provenance["internal-source-handoff.json"] == case.handoff.handoff_bytes
    assert provenance["suffix-rules.dat"] == case.suffix
    restored = verify(provenance_api, case, provenance)
    assert restored == case.prepared
    assert [row.is_phishing for row in restored.retained] == [1, None]
    assert {row.published_id for row in restored.quarantine} == {
        "train",
        "cross-split",
        "overlap",
    }
    assert b"opaque-canary" in provenance["publisher-source.json"]
    assert restored.public_summary["protected_evaluation_authorized"] is False


def test_no_path_archive_model_or_forward_reads(
    provenance_api, provenance_case, monkeypatch
):
    from pathlib import Path

    from automated_phishing_detection import (
        bound_models,
        external_producer,
        phishvn_source,
    )

    def forbidden(*args, **kwargs):
        pytest.fail("provenance performed a source/model read or forward")

    for name in ("open", "read_bytes", "read_text", "stat"):
        monkeypatch.setattr(Path, name, forbidden)
    monkeypatch.setattr(phishvn_source, "decode_phishvn_archive", forbidden)
    monkeypatch.setattr(bound_models, "load_bound_models", forbidden)
    monkeypatch.setattr(external_producer, "produce_external_evidence", forbidden)
    provenance = build(provenance_api, provenance_case)
    assert (
        verify(provenance_api, provenance_case, provenance) == provenance_case.prepared
    )


def test_returned_preparation_does_not_share_mutable_views(
    provenance_api, provenance_case
):
    case = provenance_case
    provenance = build(provenance_api, case)
    restored = verify(provenance_api, case, provenance)
    restored.private_outputs.clear()
    restored.public_summary["role_counts"].clear()
    assert verify(provenance_api, case, provenance) == case.prepared
