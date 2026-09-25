"""Canonical and rehashed forgeries must not change frozen external evidence."""

import copy
import json
from dataclasses import replace
from hashlib import sha256

import pytest
from saved_external_fixtures import saved_external_bundle
from test_saved_external_evidence import module

from automated_phishing_detection import saved_evidence
from automated_phishing_detection._saved_external_bindings import PRIVATE_OUTPUTS


def rehashed_summary(produced, private):
    public = copy.deepcopy(produced.public_summary)
    public["private_sha256"] = {
        name: sha256(content).hexdigest() for name, content in private.items()
    }
    return saved_evidence._json_bytes(public)


@pytest.mark.parametrize("name", sorted(PRIVATE_OUTPUTS))
@pytest.mark.parametrize("rehash", [False, True])
def test_each_private_output_mutation_rejected(monkeypatch, name, rehash):
    produced = saved_external_bundle(monkeypatch).produced
    private = dict(produced.private_outputs)
    private[name] += b"\n"
    public = (
        rehashed_summary(produced, private)
        if rehash
        else saved_evidence._json_bytes(produced.public_summary)
    )
    reconstructor = module()
    with pytest.raises(reconstructor.SavedExternalEvidenceError):
        reconstructor.reconstruct_external_evidence(private, public)


@pytest.mark.parametrize(
    "mutation", ["authorized", "role_count", "monitor", "hypothesis"]
)
def test_public_aggregate_or_authority_forgery_rejected(monkeypatch, mutation):
    produced = saved_external_bundle(monkeypatch).produced
    public = copy.deepcopy(produced.public_summary)
    if mutation == "authorized":
        public["protected_evaluation_authorized"] = True
    elif mutation == "role_count":
        public["role_counts"]["gold"] += 1
    elif mutation == "monitor":
        public["monitors"]["gmm"]["reason"] = "private-url-leak"
    else:
        public["primary"]["hypotheses"]["H2"]["decision"] = "supported"
    reconstructor = module()
    with pytest.raises(reconstructor.SavedExternalEvidenceError) as caught:
        reconstructor.reconstruct_external_evidence(
            produced.private_outputs, saved_evidence._json_bytes(public)
        )
    assert "private-url-leak" not in str(caught.value)


@pytest.mark.parametrize(
    "field", ["routing.json", "monitors.json", "predictions.jsonl", "secondary.json"]
)
def test_canonical_rehashed_derived_changes_rejected(monkeypatch, field):
    produced = saved_external_bundle(monkeypatch).produced
    private = dict(produced.private_outputs)
    if field == "predictions.jsonl":
        rows = [json.loads(line) for line in private[field].splitlines()]
        rows[0]["standardized_monitor_features"][0] += 0.01
        private[field] = b"".join(saved_evidence._json_bytes(row) for row in rows)
    else:
        value = json.loads(private[field])
        if field == "monitors.json":
            value[0]["reason"] = "private-url-leak"
        else:
            value["invented"] = True
        private[field] = saved_evidence._json_bytes(value)
    reconstructor = module()
    with pytest.raises(reconstructor.SavedExternalEvidenceError):
        reconstructor.reconstruct_external_evidence(
            private, rehashed_summary(produced, private)
        )


@pytest.mark.parametrize(
    "mutation", ["scale", "portable", "logistic", "gmm", "training_count"]
)
def test_arithmetic_crossbinds_full_accepted_reference(monkeypatch, mutation):
    from automated_phishing_detection._saved_external_arithmetic import (
        verify_retained_arithmetic,
    )

    bundle = saved_external_bundle(monkeypatch)
    binding = json.loads(bundle.produced.private_outputs["bindings.json"])
    reference = bundle.session.drift.reference
    if mutation == "scale":
        reference = replace(reference, scaler_scale=(2.0,) * 26)
    elif mutation == "portable":
        reference = replace(reference, portable_state_sha256="f" * 64)
    elif mutation == "training_count":
        reference = replace(
            reference, psi=replace(reference.psi, training_row_count=999)
        )
    else:
        field = (
            "logistic_l1_artifact_sha256"
            if mutation == "logistic"
            else "gmm_artifact_sha256"
        )
        reference = replace(
            reference, pins=replace(reference.pins, **{field: "f" * 64})
        )
    with pytest.raises(ValueError):
        verify_retained_arithmetic((), binding, reference)
