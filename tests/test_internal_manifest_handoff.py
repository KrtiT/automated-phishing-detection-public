"""Retain already reconstructed manifest outcomes without another source pass."""

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import test_evaluation_manifest as manifest_fixtures
import test_internal_source_handoff as handoff_fixtures
import test_source_completion as completion_fixtures

from automated_phishing_detection import evaluation_producer, internal_source_handoff
from automated_phishing_detection.hypothesis_evaluation import SavedPopulation

candidates = manifest_fixtures.candidates
manifests = manifest_fixtures.manifests
inputs = completion_fixtures.inputs
published = completion_fixtures.published
runner = completion_fixtures.runner
verifier = completion_fixtures.verifier


def test_snapshot_exposes_retained_manifest_outcomes():
    assert isinstance(
        getattr(internal_source_handoff.VerifiedInternalSnapshot, "manifests", None),
        property,
    ), "missing immutable manifest handoff"


@pytest.fixture
def outcomes(manifests):
    return evaluation_producer._manifests(manifests[100].records)


def frozen(outcomes):
    attempt, public = Path("/invented/attempt"), Path("/invented/public.json")
    return internal_source_handoff.freeze_internal_snapshot(
        {public: b'{"fixture":true}'},
        {},
        attempt=attempt,
        public_summary=public,
        population=SavedPopulation((), {}),
        overlap_domains=frozenset(),
        manifests=outcomes,
    )


def test_prepared_and_insufficient_outcomes_are_retained_exactly(outcomes):
    snapshot = frozen(outcomes)
    assert snapshot.manifest_outcomes == tuple(sorted(outcomes.items()))
    assert all(snapshot.manifests[key] is value for key, value in outcomes.items())
    prepared = snapshot.manifests[100]
    assert prepared.status == "prepared"
    assert len(prepared.manifest.records) == 10000
    assert len(prepared.manifest.warmup_records) == 1000
    assert snapshot.manifests[10] == evaluation_producer.ManifestOutcome(
        "insufficient_capacity", insufficient_label=0, required=9990, available=9900
    )
    assert snapshot.manifests[500] == evaluation_producer.ManifestOutcome(
        "insufficient_capacity", insufficient_label=1, required=500, available=100
    )


def test_outer_mappings_are_fresh_and_nested_values_are_frozen(outcomes):
    snapshot = frozen(outcomes)
    expected = dict(outcomes)
    outcomes.clear()
    first = snapshot.manifests
    first.clear()
    assert snapshot.manifests == expected
    assert snapshot.manifests is not snapshot.manifests
    assert type(snapshot.manifest_outcomes) is tuple
    prepared = snapshot.manifests[100]
    with pytest.raises(FrozenInstanceError):
        snapshot.manifest_outcomes = ()
    with pytest.raises(FrozenInstanceError):
        prepared.status = "changed"
    with pytest.raises(FrozenInstanceError):
        prepared.manifest.sha256 = "0" * 64
    with pytest.raises(FrozenInstanceError):
        prepared.manifest.records[0].raw_url = "https://changed.example"
    assert type(prepared.manifest.records) is tuple


def test_manifest_field_does_not_expose_private_rows_in_repr(outcomes):
    snapshot = frozen(outcomes)
    assert "manifest_outcomes" not in repr(snapshot)
    assert outcomes[100].manifest.records[0].raw_url not in repr(snapshot)


def test_completion_reuses_its_single_reconstructed_mapping(
    verifier, published, monkeypatch
):
    binding, paths, unused = published
    original = verifier.reconstruct_internal_evidence_and_population
    reconstructed = []

    def observed(*args):
        result = original(*args)
        reconstructed.append(result[0].manifests)
        return result

    monkeypatch.setattr(
        verifier, "reconstruct_internal_evidence_and_population", observed
    )
    snapshot = verifier.verify_internal_completion_snapshot(
        binding, paths, producer_exit_code=0
    )
    assert len(reconstructed) == 1
    assert snapshot.manifests == reconstructed[0]
    assert all(
        snapshot.manifests[key] is value for key, value in reconstructed[0].items()
    )
    assert all(
        value.status == "insufficient_capacity" for value in snapshot.manifests.values()
    )
    expected = snapshot.manifests
    reconstructed[0].clear()
    assert snapshot.manifests == expected


def test_manifest_views_do_not_read_parse_reconstruct_or_sample(outcomes, monkeypatch):
    from automated_phishing_detection import evaluation_manifest, saved_evidence

    snapshot = frozen(outcomes)

    def forbidden(*args, **kwargs):
        pytest.fail("manifest view repeated input processing")

    monkeypatch.setattr("builtins.open", forbidden)
    monkeypatch.setattr(Path, "read_bytes", forbidden)
    monkeypatch.setattr(internal_source_handoff.json, "loads", forbidden)
    monkeypatch.setattr(evaluation_manifest, "build_manifest", forbidden)
    monkeypatch.setattr(evaluation_manifest, "_permutation", forbidden)
    monkeypatch.setattr(
        saved_evidence, "reconstruct_internal_evidence_and_population", forbidden
    )
    assert snapshot.manifests == snapshot.manifests == outcomes


def test_manifest_views_survive_saved_path_replacement(verifier, published):
    binding, paths, unused = published
    snapshot = verifier.verify_internal_completion_snapshot(
        binding, paths, producer_exit_code=0
    )
    expected, public, payloads = (
        snapshot.manifests,
        snapshot.public_summary,
        snapshot.payloads,
    )
    for path in handoff_fixtures._retained_paths(binding, paths).values():
        path.rename(path.with_suffix(path.suffix + ".detached"))
        path.write_bytes(b"unverified replacement")
    assert snapshot.manifests == expected
    assert snapshot.public_summary == public
    assert snapshot.payloads == payloads


def test_legacy_public_result_and_inventory_are_unchanged(verifier, published):
    binding, paths, unused = published
    snapshot = verifier.verify_internal_completion_snapshot(
        binding, paths, producer_exit_code=0
    )
    assert snapshot.public_summary == verifier.verify_internal_completion(
        binding, paths, producer_exit_code=0
    )
    assert len(snapshot.payloads) == 35
    assert set(dict(snapshot.payloads)) == set(
        handoff_fixtures._retained_paths(binding, paths)
    )
