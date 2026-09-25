"""Immutable same-parent evidence views from invented saved external results."""

import importlib
import importlib.util
import json
from dataclasses import FrozenInstanceError, asdict
from pathlib import Path

import pytest
from saved_external_fixtures import saved_external_bundle

from automated_phishing_detection._checkpoint_codec import canonical_bytes
from automated_phishing_detection._external_completion_files import ExternalFileSnapshot


def module():
    name = "automated_phishing_detection.external_source_handoff"
    assert importlib.util.find_spec(name), "missing immutable external handoff"
    return importlib.import_module(name)


def test_external_handoff_api_exists():
    module()


@pytest.mark.parametrize("count", [0, 1, 320])
def test_freeze_keeps_typed_results_and_exact_retained_bytes(monkeypatch, count):
    api = module()
    produced = saved_external_bundle(monkeypatch, count).produced
    public = canonical_bytes(produced.public_summary)
    files = ExternalFileSnapshot((("public-summary.json", public),))
    profile = canonical_bytes({"fixture": True})
    snapshot = api.freeze_external_snapshot(files, profile, produced.replay)
    assert type(snapshot) is api.VerifiedExternalSnapshot
    assert snapshot.payload("public-summary.json") == public
    assert snapshot.profile_bytes == profile
    assert snapshot.public_summary == json.loads(public)
    assert asdict(snapshot.replay) == asdict(produced.replay)
    with pytest.raises(FrozenInstanceError):
        snapshot.payloads = ()
    assert "host" not in repr(snapshot)
    assert "role_counts" not in repr(snapshot)


def test_fresh_views_and_original_mutations_cannot_change_retained_evidence(
    monkeypatch,
):
    api = module()
    produced = saved_external_bundle(monkeypatch, 5).produced
    expected = asdict(produced.replay)
    public = canonical_bytes(produced.public_summary)
    files = ExternalFileSnapshot((("public-summary.json", public),))
    snapshot = api.freeze_external_snapshot(files, b"{}\n", produced.replay)
    snapshot.public_summary.clear()
    views = (snapshot.replay, produced.replay)
    for replay in views:
        for population in replay.evidence.populations.values():
            population.predictions.clear()
        replay.evidence.populations.clear()
        replay.evidence.controls.predictions.clear()
        replay.evidence.role_counts.clear()
    assert snapshot.public_summary == json.loads(public)
    assert asdict(snapshot.replay) == expected


def test_handoff_views_do_not_read_reconstruct_or_score(monkeypatch):
    api = module()
    produced = saved_external_bundle(monkeypatch).produced
    expected = asdict(produced.replay)
    files = ExternalFileSnapshot((("public-summary.json", b'{"fixture":true}\n'),))

    def forbidden(*args, **kwargs):
        pytest.fail("handoff reopened paths or recomputed evidence")

    from automated_phishing_detection import (
        external_producer,
        external_replay,
        saved_external_evidence,
    )

    for method in ("read_bytes", "read_text", "open", "stat"):
        monkeypatch.setattr(Path, method, forbidden)
    monkeypatch.setattr(external_producer, "produce_external_evidence", forbidden)
    monkeypatch.setattr(external_replay, "replay_external_scores", forbidden)
    monkeypatch.setattr(
        saved_external_evidence, "reconstruct_external_evidence", forbidden
    )
    snapshot = api.freeze_external_snapshot(files, b"{}\n", produced.replay)
    assert asdict(snapshot.replay) == expected
    assert snapshot.public_summary == {"fixture": True}
