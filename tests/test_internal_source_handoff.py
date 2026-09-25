"""Immutable handoff from verifier reads of invented producer outputs only."""

import json
from collections import Counter
from dataclasses import FrozenInstanceError

import pytest
import test_source_completion as completion_fixtures

from automated_phishing_detection.internal_scientific_checkpoints import (
    SCIENTIFIC_CHECKPOINT_NAMES,
)

inputs = completion_fixtures.inputs
published = completion_fixtures.published
runner = completion_fixtures.runner
verifier = completion_fixtures.verifier


def _retained_paths(binding, paths):
    attempt_names = {
        "reservation.json",
        "finalize.claim",
        "outcome.json",
        *(f"evidence/{name}" for name in completion_fixtures.PRIVATE_NAMES),
        *(f"checkpoints/{name}" for name in completion_fixtures.CHECKPOINT_NAMES),
        *(f"scientific-checkpoints/{name}" for name in SCIENTIFIC_CHECKPOINT_NAMES),
    }
    return {
        **{f"attempt/{name}": paths.attempt / name for name in attempt_names},
        "public-summary.json": paths.public_summary,
        **{f"source/{name}": binding.root / name for name, _ in binding.source_hashes},
    }


def _snapshot(verifier, binding, paths):
    companion = getattr(verifier, "verify_internal_completion_snapshot", None)
    assert callable(companion), "missing immutable completion snapshot companion"
    return companion(binding, paths, producer_exit_code=0)


def test_snapshot_preserves_all_verified_bytes_and_legacy_public_api(
    verifier, published
):
    binding, paths, _ = published
    expected = {
        name: path.read_bytes()
        for name, path in _retained_paths(binding, paths).items()
    }
    legacy = verifier.verify_internal_completion(binding, paths, producer_exit_code=0)
    snapshot = _snapshot(verifier, binding, paths)
    assert len(snapshot.payloads) == len(expected) == 35
    assert dict(snapshot.payloads) == expected
    assert (
        snapshot.public_summary == legacy == json.loads(expected["public-summary.json"])
    )
    for name, content in expected.items():
        assert snapshot.payload(name) == content
    with pytest.raises(KeyError):
        snapshot.payload("attempt/missing.json")


def _observe_reads_and_reconstruction(verifier, runner, monkeypatch, expected_paths):
    original_read = runner._read_file_once
    original_reconstruction = verifier.reconstruct_internal_evidence_and_population
    reads, populations = Counter(), []

    def observed_read(path, **kwargs):
        assert path in expected_paths
        reads[path] += 1
        return original_read(path, **kwargs)

    def observed_reconstruction(*args):
        result = original_reconstruction(*args)
        populations.append(result[1])
        return result

    monkeypatch.setattr(runner, "_read_file_once", observed_read)
    monkeypatch.setattr(
        verifier,
        "reconstruct_internal_evidence_and_population",
        observed_reconstruction,
    )
    return reads, populations


def test_snapshot_reads_once_and_reconstructs_population_once(
    verifier, runner, published, monkeypatch
):
    binding, paths, events = published
    expected_paths = _retained_paths(binding, paths).values()
    reads, populations = _observe_reads_and_reconstruction(
        verifier, runner, monkeypatch, expected_paths
    )
    initial_rechecks = events.count("recheck")
    snapshot = _snapshot(verifier, binding, paths)
    assert len(populations) == 1
    assert snapshot.population == snapshot.population == populations[0]
    assert reads == Counter(expected_paths)
    assert events.count("recheck") == initial_rechecks + 2
    populations[0].predictions.clear()
    assert set(snapshot.population.predictions) == {
        "length_only",
        "logistic_l1",
        "transformer",
        "cascade",
    }


def test_snapshot_survives_replacement_of_every_verified_path(verifier, published):
    binding, paths, _ = published
    snapshot = _snapshot(verifier, binding, paths)
    original_public, original_population = snapshot.public_summary, snapshot.population
    for name, path in _retained_paths(binding, paths).items():
        original = snapshot.payload(name)
        path.rename(path.with_suffix(path.suffix + ".detached"))
        path.write_bytes(b"replacement is not verified evidence")
        assert snapshot.payload(name) == original
    assert snapshot.public_summary == original_public
    assert snapshot.population == original_population


def test_public_and_population_mutation_cannot_change_snapshot(verifier, published):
    binding, paths, _ = published
    snapshot = _snapshot(verifier, binding, paths)
    original_public, original_population = snapshot.public_summary, snapshot.population
    mutable_public, mutable_population = snapshot.public_summary, snapshot.population
    mutable_public["execution"].clear()
    mutable_population.predictions.clear()
    assert snapshot.public_summary == original_public
    assert snapshot.population == original_population
    assert type(snapshot.payloads) is tuple
    assert type(snapshot.records) is tuple
    assert type(snapshot.prediction_columns) is tuple
    assert all(type(column) is tuple for _, column in snapshot.prediction_columns)
    with pytest.raises(FrozenInstanceError):
        snapshot.payloads = ()
    with pytest.raises(FrozenInstanceError):
        snapshot.population.records[0].label = 1
    with pytest.raises(FrozenInstanceError):
        snapshot.population.predictions["cascade"][0].decision = 1


def test_snapshot_includes_original_domains_before_quarantine(verifier, published):
    binding, paths, _ = published
    snapshot = _snapshot(verifier, binding, paths)
    overlap = json.loads(snapshot.payload("attempt/checkpoints/source-overlap.json"))
    assert type(snapshot.overlap_domains) is frozenset
    assert snapshot.overlap_domains == frozenset(overlap["domains"])
    assert {
        "quarantined-label.com",
        "quarantined-conflict.com",
    } <= snapshot.overlap_domains
    assert {
        record.registrable_domain for record in snapshot.records
    } < snapshot.overlap_domains


def test_snapshot_rejects_file_change_during_final_state_check(
    verifier, runner, published, monkeypatch
):
    binding, paths, _ = published
    companion = getattr(verifier, "verify_internal_completion_snapshot", None)
    assert callable(companion), "missing immutable completion snapshot companion"
    original_recheck = runner.recheck_binding
    rechecks = []

    def replace_before_final_check(*args):
        rechecks.append(args)
        if len(rechecks) == 2:
            paths.public_summary.write_bytes(b"replaced after verification")
        return original_recheck(*args)

    monkeypatch.setattr(runner, "recheck_binding", replace_before_final_check)
    with pytest.raises(verifier.CompletionVerificationError, match="output_changed"):
        companion(binding, paths, producer_exit_code=0)
