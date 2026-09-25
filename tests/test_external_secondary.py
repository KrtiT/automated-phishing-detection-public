"""Completed secondary columns bind identities before the next member starts."""

import json
from dataclasses import asdict
from hashlib import sha256
from types import ModuleType, SimpleNamespace

import pytest
from bound_secondary_column_fixtures import scoring as scoring
from external_secondary_fixtures import CHECKPOINTS, primary_scores
from external_secondary_fixtures import module as module
from external_secondary_fixtures import phase as phase

from automated_phishing_detection import bound_secondary, evaluation_producer


def _assert_binding(
    saved: dict, bound: bound_secondary.BoundSecondary, index: int
) -> None:
    if index < 7:
        assert (
            saved["binding"]["artifact_sha256"] == bound.tabular[index].artifact_sha256
        )
        assert saved["binding"]["threshold"] == bound.tabular[index].threshold
        assert saved["column"]["singleton_calls"] == 2
    else:
        member = bound.seeds[index - 7]
        assert saved["binding"]["weights_sha256"] == member.weights_sha256
        assert saved["binding"]["half_width"] == member.half_width
        assert saved["binding"]["transformer_threshold"] == member.transformer_threshold
        assert saved["binding"]["stage1_threshold"] == bound.stage1_threshold


def test_complete_columns_bind_primary_ids_artifacts_scores_and_counts(
    module: ModuleType,
    phase: SimpleNamespace,
) -> None:
    writes = []
    result = module.score_external_secondary(
        phase.primary,
        phase.bound,
        retain=lambda name, content: writes.append((name, content)),
    )
    assert tuple(result.private_outputs) == (*CHECKPOINTS, "secondary-completion.json")
    assert writes == list(result.private_outputs.items())
    for index, name in enumerate(CHECKPOINTS):
        content = result.private_outputs[name]
        saved = json.loads(content)
        assert content == evaluation_producer._json_bytes(saved)
        assert saved["schema_version"] == 1
        assert (
            saved["primary_scores_sha256"]
            == sha256(phase.primary.checkpoint_bytes).hexdigest()
        )
        assert saved["record_ids"] == [row.record_id for row in phase.primary.records]
        assert len(saved["column"]["scores"]) == 2
        _assert_binding(saved, phase.bound, index)
    assert result.scoring.counts.reused_primary_transformer_scores == 2


def test_final_receipt_binds_all_twelve_columns_and_exact_scoring(
    module: ModuleType,
    phase: SimpleNamespace,
) -> None:
    result = module.score_external_secondary(phase.primary, phase.bound)
    receipt = json.loads(result.private_outputs["secondary-completion.json"])
    assert receipt["phase"] == "external_secondary"
    assert receipt["row_count"] == 2
    assert (
        receipt["primary_scores_sha256"]
        == sha256(phase.primary.checkpoint_bytes).hexdigest()
    )
    assert receipt["checkpoint_sha256"] == {
        name: sha256(result.private_outputs[name]).hexdigest() for name in CHECKPOINTS
    }
    assert receipt["inference_counts"] == json.loads(
        json.dumps(asdict(result.scoring.counts))
    )
    assert (
        receipt["secondary_scores_sha256"]
        == sha256(evaluation_producer._json_bytes(asdict(result.scoring))).hexdigest()
    )


def test_empty_primary_still_completes_all_columns_without_model_calls(
    module: ModuleType,
    scoring: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary = primary_scores(monkeypatch, 0)
    result = module.score_external_secondary(primary, scoring.bound)
    assert tuple(result.private_outputs) == (*CHECKPOINTS, "secondary-completion.json")
    assert result.scoring.rows == ()
    assert scoring.events == scoring.singletons == []
    assert all(
        json.loads(result.private_outputs[name])["column"]["scores"] == []
        for name in CHECKPOINTS
    )
    assert result.scoring.counts.reused_primary_transformer_scores == 0


def test_retention_occurs_before_the_next_members_inference(
    module: ModuleType,
    phase: SimpleNamespace,
) -> None:
    def retain(name: str, content: bytes) -> None:
        phase.events.append(("retained", name))

    module.score_external_secondary(phase.primary, phase.bound, retain=retain)
    for index, member in enumerate(phase.bound.tabular[:-1]):
        retained = phase.events.index(("retained", CHECKPOINTS[index]))
        following = phase.events.index(("tabular", phase.bound.tabular[index + 1].name))
        assert retained < following
    assert phase.events.index(("retained", CHECKPOINTS[7])) < phase.events.index(
        ("load", 43)
    )


def test_completed_secondary_is_exactly_the_existing_bound_kernel(
    module: ModuleType,
    phase: SimpleNamespace,
) -> None:
    expected = bound_secondary.score_bound_secondary(
        phase.bound,
        tuple(row.raw_url for row in phase.primary.records),
        tuple(row.stage1_probability for row in phase.primary.scores),
        tuple(row.transformer_probability for row in phase.primary.scores),
    )
    actual = module.score_external_secondary(phase.primary, phase.bound)
    assert actual.scoring == expected
