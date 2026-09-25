"""Independent receipt bindings reject before publisher restoration or preparation."""

import json

import pytest
import test_external_source_provenance as fixtures
from test_external_source_provenance import NAMES, PREPARED_NAMES, build, verify

from automated_phishing_detection import phishvn, phishvn_source
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


def _forbid_derivation(api, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("receipt bindings were not checked before derivation")

    monkeypatch.setattr(api, "restore_phishvn_source", forbidden)
    monkeypatch.setattr(phishvn_source, "_decode_rows", forbidden)
    monkeypatch.setattr(phishvn, "prepare_external_rows", forbidden)


@pytest.mark.parametrize(
    "field,value",
    [
        ("execution", {"kind": "other_parent"}),
        ("reservation_sha256", "0" * 64),
    ],
)
@pytest.mark.parametrize("changed_side", ["parent", "receipt"])
def test_context_checked_before_any_derivation(
    provenance_api, provenance_case, monkeypatch, field, value, changed_side
):
    provenance = build(provenance_api, provenance_case)
    arguments = {}
    if changed_side == "parent":
        arguments[field] = value
    else:
        receipt = json.loads(provenance["external-source-reconstruction.json"])
        receipt[field] = value
        provenance["external-source-reconstruction.json"] = canonical_bytes(receipt)
    _forbid_derivation(provenance_api, monkeypatch)
    with pytest.raises(provenance_api.ExternalSourceProvenanceError):
        verify(provenance_api, provenance_case, provenance, **arguments)


@pytest.mark.parametrize(
    "group,name",
    [
        *(
            ("provenance_sha256", name)
            for name in sorted(NAMES - {"external-source-reconstruction.json"})
        ),
        *(("prepared_sha256", name) for name in sorted(PREPARED_NAMES)),
    ],
)
def test_every_known_payload_digest_checked_before_derivation(
    provenance_api, provenance_case, monkeypatch, group, name
):
    provenance = build(provenance_api, provenance_case)
    receipt = json.loads(provenance["external-source-reconstruction.json"])
    receipt[group][name] = "0" * 64
    provenance["external-source-reconstruction.json"] = canonical_bytes(receipt)
    _forbid_derivation(provenance_api, monkeypatch)
    with pytest.raises(provenance_api.ExternalSourceProvenanceError):
        verify(provenance_api, provenance_case, provenance)


@pytest.mark.parametrize(
    "change", ["boolean_version", "extra", "noncanonical", "counts_shape"]
)
def test_receipt_shape_checked_before_derivation(
    provenance_api, provenance_case, monkeypatch, change
):
    provenance = build(provenance_api, provenance_case)
    receipt = json.loads(provenance["external-source-reconstruction.json"])
    if change == "boolean_version":
        receipt["schema_version"] = True
    elif change == "extra":
        receipt["extra"] = None
    elif change == "counts_shape":
        receipt["counts"] = []
    content = canonical_bytes(receipt)
    provenance["external-source-reconstruction.json"] = content + (
        b" " if change == "noncanonical" else b""
    )
    _forbid_derivation(provenance_api, monkeypatch)
    with pytest.raises(provenance_api.ExternalSourceProvenanceError):
        verify(provenance_api, provenance_case, provenance)
