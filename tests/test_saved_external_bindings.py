"""Strict external byte envelopes over invented source and artifact identities."""

import copy
import importlib
import importlib.util
import json
from hashlib import sha256

import pytest
from external_composition_fixtures import composition_inputs, producer_module
from test_saved_evidence import expect_synthetic_binding

from automated_phishing_detection import saved_evidence


def module():
    name = "automated_phishing_detection._saved_external_bindings"
    assert importlib.util.find_spec(name), "saved external bindings missing"
    return importlib.import_module(name)


@pytest.fixture
def bundle(monkeypatch):
    prepared, session, *_ = composition_inputs(monkeypatch)
    result = producer_module().produce_external_evidence(prepared, session)
    expect_synthetic_binding(
        saved_evidence, monkeypatch, result.private_outputs["bindings.json"]
    )
    return result


def test_snapshot_closes_inventory_and_copies_mapping(bundle):
    private = dict(bundle.private_outputs)
    public = saved_evidence._json_bytes(bundle.public_summary)
    snapshot, summary = module().snapshot_outputs(private, public)
    private.clear()
    assert len(snapshot) == 30
    assert saved_evidence._json_bytes(summary) == public
    assert snapshot == bundle.private_outputs


@pytest.mark.parametrize("mutation", ["missing", "extra", "mutable", "subclass"])
def test_snapshot_rejects_nonclosed_or_mutable_inventory(bundle, mutation):
    private = dict(bundle.private_outputs)
    if mutation == "missing":
        private.pop("primary-completion.json")
    elif mutation == "extra":
        private["private-url-leak"] = b""
    elif mutation == "mutable":
        private["bindings.json"] = bytearray(private["bindings.json"])
    else:
        private = type("CustomMapping", (dict,), {})(private)
    with pytest.raises(ValueError):
        module().snapshot_outputs(
            private, saved_evidence._json_bytes(bundle.public_summary)
        )


@pytest.mark.parametrize("content", [b"{}", b"{}\n\n", b'{"a":1,"a":1}\n'])
def test_snapshot_rejects_noncanonical_summary(bundle, content):
    with pytest.raises(ValueError):
        module().snapshot_outputs(bundle.private_outputs, content)


def test_snapshot_rejects_replaced_private_bytes(bundle):
    private = bundle.private_outputs | {"secondary.json": b"{}\n"}
    with pytest.raises(ValueError):
        module().snapshot_outputs(
            private, saved_evidence._json_bytes(bundle.public_summary)
        )


def test_restores_exact_binding_and_accepted_drift_chain(bundle):
    bindings = module().restore_bindings(bundle.private_outputs)
    reference = module().restore_reference(bundle.private_outputs, bindings)
    assert bindings == json.loads(bundle.private_outputs["bindings.json"])
    assert (
        reference.training_reference_sha256
        == sha256(bundle.private_outputs["training-reference.json"]).hexdigest()
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "schema_bool",
        "extra",
        "cutoff",
        "count_bool",
        "seed_float",
        "preparation",
        "source",
    ],
)
def test_rehashed_core_or_envelope_changes_are_rejected(bundle, mutation):
    bindings = json.loads(bundle.private_outputs["bindings.json"])
    if mutation == "schema_bool":
        bindings["schema_version"] = True
    elif mutation == "extra":
        bindings["private-url-leak"] = "private-value"
    elif mutation == "cutoff":
        bindings["thresholds"]["logistic_l1"] += 0.01
    elif mutation == "count_bool":
        bindings["gmm_audit"]["alert_count"] = True
    elif mutation == "seed_float":
        bindings["secondary"]["seeds"][0]["seed"] = 42.0
    elif mutation == "preparation":
        bindings["preparation"]["retained_test_rows"] += 1
    else:
        bindings["source_binding"] = "officially_authenticated"
    private = bundle.private_outputs | {
        "bindings.json": saved_evidence._json_bytes(bindings)
    }
    with pytest.raises(ValueError):
        module().restore_bindings(private)


@pytest.mark.parametrize(
    "mutation", ["pins", "portable", "extra_hash", "hash", "report"]
)
def test_drift_envelope_cannot_replace_accepted_reference(bundle, mutation):
    bindings = copy.deepcopy(module().restore_bindings(bundle.private_outputs))
    private = dict(bundle.private_outputs)
    if mutation == "pins":
        bindings["drift"]["pins"]["train_sha256"] = "f" * 64
    elif mutation == "portable":
        bindings["drift"]["portable_state_sha256"] = "f" * 64
    elif mutation == "extra_hash":
        bindings["drift"]["private_sha256"]["private-value"] = "f" * 64
    elif mutation == "hash":
        bindings["drift"]["private_sha256"]["training-reference.json"] = "f" * 64
    else:
        private["drift-accepted-report.json"] = b"not accepted JSON"
        bindings["drift"]["private_sha256"]["drift-accepted-report.json"] = sha256(
            private["drift-accepted-report.json"]
        ).hexdigest()
    with pytest.raises(ValueError):
        module().restore_reference(private, bindings)
