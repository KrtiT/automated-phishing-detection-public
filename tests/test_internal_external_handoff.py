"""Same-parent byte projection from invented, actually verified completions."""

import importlib.util
import json
import sys
from dataclasses import FrozenInstanceError, asdict, replace
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest
import test_source_completion as completion_fixtures

from automated_phishing_detection._checkpoint_codec import canonical_bytes
from automated_phishing_detection.internal_process_handoff import (
    ObservedInternalCompletion,
)
from automated_phishing_detection.owned_worker import observe_worker

inputs = completion_fixtures.inputs
published = completion_fixtures.published
runner = completion_fixtures.runner
verifier = completion_fixtures.verifier


@pytest.fixture
def handoff_api():
    name = "automated_phishing_detection.internal_external_handoff"
    assert importlib.util.find_spec(name), "missing same-parent byte handoff"
    return importlib.import_module(name)


@pytest.fixture(scope="module")
def observed_worker():
    return observe_worker((sys.executable, "-c", "pass"))


@pytest.fixture
def completion(verifier, published, observed_worker):
    binding, paths, _ = published
    snapshot = verifier.verify_internal_completion_snapshot(
        binding, paths, producer_exit_code=0
    )
    return ObservedInternalCompletion(observed_worker, snapshot)


def verified_domains(api, payloads, *, expected=None):
    return api.verify_internal_handoff(
        payloads.handoff_bytes,
        payloads.overlap_bytes,
        expected_handoff_sha256=expected or sha256(payloads.handoff_bytes).hexdigest(),
    )


def test_projection_binds_all_retained_bytes_and_actual_worker(handoff_api, completion):
    payloads = handoff_api.build_internal_handoff(completion)
    assert type(payloads) is handoff_api.InternalHandoffPayloads
    assert type(payloads.handoff_bytes) is type(payloads.overlap_bytes) is bytes
    expected = {
        "schema_version": 1,
        "kind": "same-parent-internal-handoff-v1",
        "execution": completion.public_summary["execution"],
        "worker": asdict(completion.worker),
        "snapshot_sha256": {
            name: sha256(content).hexdigest()
            for name, content in completion.snapshot.payloads
        },
    }
    assert len(expected["snapshot_sha256"]) == 35
    assert payloads.handoff_bytes == canonical_bytes(expected)
    assert payloads.overlap_bytes == completion.snapshot.payload(
        "attempt/checkpoints/source-overlap.json"
    )
    assert (
        verified_domains(handoff_api, payloads) == completion.snapshot.overlap_domains
    )
    with pytest.raises(FrozenInstanceError):
        payloads.handoff_bytes = b"changed"


def test_projection_keeps_quarantined_only_original_domains(handoff_api, completion):
    payloads = handoff_api.build_internal_handoff(completion)
    domains = verified_domains(handoff_api, payloads)
    assert {"quarantined-label.com", "quarantined-conflict.com"} <= domains
    assert {
        record.registrable_domain for record in completion.snapshot.records
    } < domains


def test_projection_and_validation_never_read_paths_or_population(
    handoff_api, completion, monkeypatch
):
    from automated_phishing_detection import (
        bound_secondary,
        evaluation_producer,
        phiusiil,
    )

    def forbidden(*args, **kwargs):
        pytest.fail("handoff reopened a path or reconstructed original source")

    for method in ("read_bytes", "read_text", "open", "stat"):
        monkeypatch.setattr(Path, method, forbidden)
    monkeypatch.setattr(phiusiil, "_parse_csv_rows", forbidden)
    monkeypatch.setattr(evaluation_producer, "produce_internal_evidence", forbidden)
    monkeypatch.setattr(evaluation_producer, "score_primary_url", forbidden)
    monkeypatch.setattr(bound_secondary, "score_bound_secondary", forbidden)
    monkeypatch.setattr(type(completion.snapshot), "population", property(forbidden))
    payloads = handoff_api.build_internal_handoff(completion)
    assert (
        verified_domains(handoff_api, payloads) == completion.snapshot.overlap_domains
    )


def test_expected_identity_checked_before_any_json_parse(
    handoff_api, completion, monkeypatch
):
    payloads = handoff_api.build_internal_handoff(completion)

    def forbidden(*args, **kwargs):
        pytest.fail("JSON parsed before authentication")

    monkeypatch.setattr(json, "loads", forbidden)
    with pytest.raises(ValueError, match="handoff_identity_mismatch"):
        verified_domains(handoff_api, payloads, expected="0" * 64)
    with pytest.raises(ValueError, match="handoff_identity_mismatch"):
        handoff_api.verify_internal_handoff(
            b"not JSON", b"not overlap", expected_handoff_sha256="0" * 64
        )


@pytest.mark.parametrize(
    "value", [None, {}, SimpleNamespace(worker=None, snapshot=None)]
)
def test_builder_requires_exact_completion_type(handoff_api, value):
    with pytest.raises(ValueError):
        handoff_api.build_internal_handoff(value)


@pytest.mark.parametrize("field", ["worker", "snapshot"])
def test_builder_rejects_structural_impostors(handoff_api, completion, field):
    changed = replace(completion, **{field: SimpleNamespace()})
    with pytest.raises(ValueError):
        handoff_api.build_internal_handoff(changed)


@pytest.mark.parametrize("change", ["missing", "extra", "duplicate", "mutable", "list"])
def test_builder_requires_complete_immutable_snapshot(handoff_api, completion, change):
    payloads = completion.snapshot.payloads
    mutations = {
        "missing": payloads[:-1],
        "extra": (*payloads, ("unexpected", b"content")),
        "duplicate": (*payloads, payloads[0]),
        "mutable": ((payloads[0][0], bytearray(payloads[0][1])), *payloads[1:]),
        "list": list(payloads),
    }
    snapshot = replace(completion.snapshot, payloads=mutations[change])
    with pytest.raises(ValueError):
        handoff_api.build_internal_handoff(replace(completion, snapshot=snapshot))


@pytest.mark.parametrize("domains", [set(), frozenset(), frozenset({True})])
def test_builder_rejects_domains_different_from_retained_rows(
    handoff_api, completion, domains
):
    snapshot = replace(completion.snapshot, overlap_domains=domains)
    with pytest.raises(ValueError):
        handoff_api.build_internal_handoff(replace(completion, snapshot=snapshot))


@pytest.mark.parametrize("field", ["completion", "snapshot", "worker", "exit"])
def test_builder_rejects_subclassed_observation_records(handoff_api, completion, field):
    original = {
        "completion": completion,
        "snapshot": completion.snapshot,
        "worker": completion.worker,
        "exit": completion.worker.exit,
    }[field]
    subclass = type("InventedSubclass", (type(original),), {})
    changed = subclass(**vars(original))
    if field == "exit":
        changed = replace(completion, worker=replace(completion.worker, exit=changed))
    elif field != "completion":
        changed = replace(completion, **{field: changed})
    with pytest.raises(ValueError):
        handoff_api.build_internal_handoff(changed)
