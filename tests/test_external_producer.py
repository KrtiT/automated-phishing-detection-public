import json
from dataclasses import asdict
from hashlib import sha256

import pytest
from external_composition_fixtures import composition_inputs, producer_module

from automated_phishing_detection import evaluation_producer


@pytest.mark.parametrize("count", [0, 5, 320])
def test_external_composition_retains_complete_inventory_and_counts(monkeypatch, count):
    module = producer_module()
    prepared, session, primary_calls, secondary_calls = composition_inputs(
        monkeypatch, count
    )
    writes = []
    result = module.produce_external_evidence(
        prepared, session, retain=lambda *args: writes.append(args)
    )
    assert len(result.replay.rows) == count
    assert dict(writes) == result.private_outputs
    assert len(writes) == len(result.private_outputs) == 30
    assert all(len(calls) == count for calls in primary_calls)
    assert len(secondary_calls) == (11 if count else 0)
    assert (
        result.public_summary["offline_inference_counts"][
            "transformer_forward_attempts"
        ]
        == count
    )
    assert result.public_summary["row_count"] == count
    assert result.public_summary["protected_evaluation_authorized"] is False
    assert result.public_summary["source_binding"] == "caller_supplied_preparation_only"
    assert result.public_summary["private_sha256"] == {
        name: sha256(content).hexdigest()
        for name, content in result.private_outputs.items()
    }
    assert b"host" not in json.dumps(result.public_summary).encode()
    assert b"host" not in repr(result).encode()


def test_preparation_and_bindings_are_retained_before_any_primary_score(monkeypatch):
    module = producer_module()
    prepared, session, _, _ = composition_inputs(monkeypatch)
    names = []
    original = evaluation_producer.score_primary_url

    def score(*args):
        assert names[:4] == [
            "retained-test.jsonl",
            "quarantine.jsonl",
            "inventory.json",
            "preparation-summary.json",
        ]
        assert {
            "bindings.json",
            "training-reference.json",
            "validation-audit.json",
        } <= set(names)
        return original(*args)

    monkeypatch.setattr(evaluation_producer, "score_primary_url", score)
    module.produce_external_evidence(
        prepared, session, retain=lambda name, content: names.append(name)
    )
    assert names.index("primary-completion.json") < names.index(
        "secondary-tabular-formatting.json"
    )
    assert names.index("secondary-completion.json") < names.index("all-scores.jsonl")
    assert names.index("all-scores.jsonl") < names.index("routing.json")
    assert names.index("predictions.jsonl") < names.index("secondary.json")


def test_joined_scores_precede_derived_arithmetic_and_summaries(monkeypatch):
    module = producer_module()
    prepared, session, _, _ = composition_inputs(monkeypatch)
    writes = {}
    original = module.external_replay.replay_external_scores

    def replay(*args):
        assert "all-scores.jsonl" in writes
        return original(*args)

    monkeypatch.setattr(module.external_replay, "replay_external_scores", replay)
    result = module.produce_external_evidence(
        prepared,
        session,
        retain=lambda name, content: writes.__setitem__(name, content),
    )
    assert writes["predictions.jsonl"] == b"".join(
        module.evaluation_producer._json_bytes(asdict(row))
        for row in result.replay.rows
    )
    secondary = json.loads(writes["secondary.json"])
    assert len(secondary["detector_columns"]) == 22
    assert secondary["populations"]["tranco"]["unlabeled_count"] == 1
    assert (
        result.public_summary["primary"]["hypotheses"]["H2"]["decision"]
        == "not_supported"
    )
