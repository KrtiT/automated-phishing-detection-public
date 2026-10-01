"""Independent historical internal byte and non-authority specification checks."""

import json

import pytest
from study_history_internal_fixtures import (
    api,
    historical_internal,
    inputs,
    published,
    republish,
    runner,
    verify,
)

__all__ = ["historical_internal", "inputs", "published", "runner"]


@pytest.mark.parametrize(
    "name",
    [
        "source/data/sources.json",
        "public-summary.json",
        "attempt/reservation.json",
        "attempt/evidence/predictions.jsonl",
        "attempt/checkpoints/source-overlap.json",
        "attempt/scientific-checkpoints/context.json",
    ],
)
def test_all_member_classes_authenticate_before_reconstruction(
    historical_internal, monkeypatch, name
):
    def forbidden(*args, **kwargs):
        pytest.fail("unmatched byte identity reached scientific reconstruction")

    monkeypatch.setattr(api(), "restore", forbidden)
    historical_internal.payloads[name] += b" "
    with pytest.raises(ValueError, match="^invalid_historical_internal_science$"):
        verify(historical_internal)


def test_republished_authority_flag_cannot_confer_permission(historical_internal):
    public = json.loads(historical_internal.payloads["public-summary.json"])
    public["protected_evaluation_authorized"] = True
    republish(historical_internal, public)
    with pytest.raises(ValueError, match="^invalid_historical_internal_science$"):
        verify(historical_internal)


def test_snapshot_views_do_not_mutate_retained_science(historical_internal):
    result = verify(historical_internal)
    expected = result.public_summary
    result.public_summary.clear()
    result.population.predictions.clear()
    result.manifests.clear()
    historical_internal.payloads.clear()
    assert result.public_summary == expected
    assert result.population.predictions
    assert set(result.manifests) == {10, 100, 500}
    assert not hasattr(result, "worker")
    assert not hasattr(result, "protected_evaluation_ready")
