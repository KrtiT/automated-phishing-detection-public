"""Repaired checkpoint links cannot replace observations checked by retained math."""

import json
from hashlib import sha256

import pytest
from saved_external_fixtures import saved_external_bundle
from test_saved_external_evidence import module
from test_saved_external_mutations import rehashed_summary

from automated_phishing_detection import external_replay, saved_evidence
from automated_phishing_detection._external_secondary_checkpoints import (
    SECONDARY_CHECKPOINTS,
)


def _jsonl(rows):
    return b"".join(saved_evidence._json_bytes(row) for row in rows)


def _primary_change(private, field):
    rows = [json.loads(line) for line in private["primary-scores.jsonl"].splitlines()]
    original = rows[0]["primary"][field]
    rows[0]["primary"][field] = original + 0.001 if type(original) is float else "{}"
    private["primary-scores.jsonl"] = _jsonl(rows)
    joined = [json.loads(line) for line in private["all-scores.jsonl"].splitlines()]
    joined[0]["primary"] = rows[0]["primary"]
    private["all-scores.jsonl"] = _jsonl(joined)
    return sha256(private["primary-scores.jsonl"]).hexdigest()


def _repair_checkpoint_links(private, primary_hash):
    receipt = json.loads(private["primary-completion.json"])
    receipt["primary_scores_sha256"] = primary_hash
    private["primary-completion.json"] = saved_evidence._json_bytes(receipt)
    for name in SECONDARY_CHECKPOINTS:
        column = json.loads(private[name])
        column["primary_scores_sha256"] = primary_hash
        private[name] = saved_evidence._json_bytes(column)
    completion = json.loads(private["secondary-completion.json"])
    completion["primary_scores_sha256"] = primary_hash
    completion["checkpoint_sha256"] = {
        name: sha256(private[name]).hexdigest() for name in SECONDARY_CHECKPOINTS
    }
    private["secondary-completion.json"] = saved_evidence._json_bytes(completion)


@pytest.mark.parametrize(
    "field",
    [
        "monitor_probability",
        "negative_log_likelihood",
        "length_scoring_audit_json",
        "stage1_scoring_audit_json",
    ],
)
def test_relinked_arithmetic_forgery_rejected_before_derived_replay(monkeypatch, field):
    produced = saved_external_bundle(monkeypatch).produced
    private = dict(produced.private_outputs)
    _repair_checkpoint_links(private, _primary_change(private, field))
    verifier = saved_evidence._verify_loaded_monitor_path
    checked = []

    def observe(*args):
        checked.append(True)
        return verifier(*args)

    def forbidden(*args, **kwargs):
        pytest.fail("unverified retained arithmetic reached derived replay")

    monkeypatch.setattr(saved_evidence, "_verify_loaded_monitor_path", observe)
    monkeypatch.setattr(external_replay, "replay_external_scores", forbidden)
    reconstructor = module()
    with pytest.raises(reconstructor.SavedExternalEvidenceError):
        reconstructor.reconstruct_external_evidence(
            private, rehashed_summary(produced, private)
        )
    assert checked == [True]
