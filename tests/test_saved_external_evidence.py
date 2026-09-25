"""Saved-only external reconstruction over invented complete evidence streams."""

import importlib
import importlib.util
from dataclasses import asdict
from pathlib import Path

import pytest

from automated_phishing_detection import (
    bound_models,
    bound_secondary,
    character_transformer,
    evaluation_producer,
    external_producer,
    phishvn,
    saved_evidence,
    secondary_transformer,
    source_runner,
    transformer_inference,
)


def module():
    name = "automated_phishing_detection.saved_external_evidence"
    assert importlib.util.find_spec(name), "saved external reconstruction missing"
    return importlib.import_module(name)


@pytest.mark.parametrize("count", [0, 1, 255, 256, 319, 320])
def test_reconstructs_complete_saved_stream(monkeypatch, count):
    reconstructor = module()
    from saved_external_fixtures import saved_external_bundle

    bundle = saved_external_bundle(monkeypatch, count)
    produced = bundle.produced
    replay = reconstructor.reconstruct_external_evidence(
        produced.private_outputs, saved_evidence._json_bytes(produced.public_summary)
    )
    assert asdict(replay) == asdict(produced.replay)
    assert len(replay.rows) == count
    assert all(
        row.record.is_phishing is None
        for row in replay.rows
        if row.record.role == "tranco"
    )
    assert produced.public_summary["protected_evaluation_authorized"] is False


@pytest.mark.parametrize(
    "private", [None, [], (), {"private-url": bytearray(b"secret")}]
)
def test_malformed_input_has_only_symbolic_failure(private):
    reconstructor = module()
    with pytest.raises(reconstructor.SavedExternalEvidenceError) as caught:
        reconstructor.reconstruct_external_evidence(private, b"{}\n")
    assert str(caught.value) == "invalid_saved_external_evidence"
    assert caught.value.__suppress_context__ is True


def test_never_uses_original_mapping_after_snapshot(monkeypatch):
    reconstructor = module()
    from saved_external_fixtures import saved_external_bundle

    produced = saved_external_bundle(monkeypatch).produced
    original = dict(produced.private_outputs)
    restore = reconstructor.restore_bindings

    def clear_original(snapshot):
        original.clear()
        return restore(snapshot)

    monkeypatch.setattr(reconstructor, "restore_bindings", clear_original)
    replay = reconstructor.reconstruct_external_evidence(
        original, saved_evidence._json_bytes(produced.public_summary)
    )
    assert original == {}
    assert replay == produced.replay


def test_full_secondary_control_prefix_routes_before_gold_selection(monkeypatch):
    import saved_external_fixtures
    from external_replay_fixtures import _prepared

    monkeypatch.setattr(saved_external_fixtures, "prepared_external", _prepared)
    produced = saved_external_fixtures.saved_external_bundle(monkeypatch, 320).produced
    replay = module().reconstruct_external_evidence(
        produced.private_outputs, saved_evidence._json_bytes(produced.public_summary)
    )
    assert {row.record.role for row in replay.rows[:256]} == {"secondary", "tranco"}
    assert replay.monitors[0].windows[0].alert is True
    assert all(not row.drift_override for row in replay.rows[:256])
    assert replay.rows[256].record.role == "gold"
    assert replay.rows[256].drift_override is True
    assert replay.rows[256].logical_stage2_mask is True
    assert len(replay.evidence.populations["gold"].records) == 32


def _forbid_source_and_model_access(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("saved verification attempted forbidden I/O or scoring")

    for target, name in (
        (Path, "read_bytes"),
        (Path, "read_text"),
        (bound_models, "load_bound_models"),
        (bound_secondary, "load_bound_secondary"),
        (bound_secondary, "score_bound_secondary"),
        (source_runner, "_read_file_once"),
        (evaluation_producer, "score_primary_url"),
        (external_producer, "produce_external_evidence"),
        (phishvn, "prepare_external_rows"),
        (secondary_transformer, "load_secondary_transformer_bytes"),
        (secondary_transformer, "score_secondary_transformer_urls"),
        (transformer_inference, "_load_transformer_cascade_bytes"),
        (character_transformer.CharacterTransformer, "forward"),
    ):
        monkeypatch.setattr(target, name, forbidden)


def test_byte_reconstruction_forbids_source_model_and_forward_access(monkeypatch):
    reconstructor = module()
    from saved_external_fixtures import saved_external_bundle

    produced = saved_external_bundle(monkeypatch).produced
    _forbid_source_and_model_access(monkeypatch)
    replay = reconstructor.reconstruct_external_evidence(
        produced.private_outputs, saved_evidence._json_bytes(produced.public_summary)
    )
    assert replay == produced.replay
