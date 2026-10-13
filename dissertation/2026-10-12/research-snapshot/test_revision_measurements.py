"""Fixture-only measurement tests; no research artifacts or timing conclusions."""

import json

import pytest
import revision_measurements as measurements


class Admission:
    def __init__(self):
        self.calls = 0
        self.revoked = False

    def __call__(self):
        self.calls += 1
        if self.revoked:
            raise ValueError("revoked")


def test_engine_scores_only_selected_expert_and_counts_actual_stages():
    called = []
    engine = measurements.RoutingEngine(
        lambda raw: called.append("structural") or 0.10,
        lambda raw: called.append("transformer") or 0.90,
        transformer_threshold=0.5,
    )
    assert engine.scan("unused", "original")["decision"] == 0
    assert called == ["structural"]
    result = engine.scan("unused", "symmetric")
    assert result["decision"] == 1
    assert called == ["structural", "structural", "transformer"]
    assert engine.counts == {
        "stage1_started": 2,
        "stage1_completed": 2,
        "stage2_started": 1,
        "stage2_completed": 1,
        "requests_completed": 2,
        "requests_failed": 0,
    }
    engine.scan("unused", "transformer")
    assert called[-1] == "transformer" and engine.counts["stage1_started"] == 2


def test_control_pays_for_stage1_but_membership_is_not_score_dependent():
    engine = measurements.RoutingEngine(
        lambda raw: 0.99, lambda raw: 0.01, transformer_threshold=0.5
    )
    assert (
        engine.scan("unused", "symmetric_control", control_selected=True)["decision"]
        == 0
    )
    assert engine.counts["stage1_completed"] == engine.counts["stage2_completed"] == 1


def test_expert_failure_retains_started_and_failed_counts():
    def failing(raw):
        raise ValueError("expert_failure")

    engine = measurements.RoutingEngine(
        lambda raw: 0.1, failing, transformer_threshold=0.5
    )
    with pytest.raises(ValueError):
        engine.scan("unused", "symmetric")
    assert engine.counts["stage2_started"] == 1
    assert engine.counts["stage2_completed"] == 0
    assert engine.counts["requests_failed"] == 1


def test_unknown_path_never_scores():
    def forbidden(raw):
        pytest.fail("unknown path performed computation")

    engine = measurements.RoutingEngine(forbidden, forbidden, transformer_threshold=0.5)
    with pytest.raises(ValueError, match="unknown_scoring_path"):
        engine.scan("unused", "unplanned")


def test_full_fixture_schedule_excludes_warmups_and_keeps_all_arms(tmp_path):
    records = [
        {
            "record_id": "one",
            "raw_url": "one",
            "stage1_probability": 0.1,
            "transformer_probability": 0.9,
        },
        {
            "record_id": "two",
            "raw_url": "two",
            "stage1_probability": 0.9,
            "transformer_probability": 0.01,
        },
    ]
    probabilities = {row["raw_url"]: row for row in records}
    engine = measurements.RoutingEngine(
        lambda raw: probabilities[raw]["stage1_probability"],
        lambda raw: probabilities[raw]["transformer_probability"],
        transformer_threshold=0.5,
    )
    admission = Admission()
    report = measurements.run_cost(records, engine, tmp_path, admission, warmups=3)
    assert len(report["arms"]) == 30
    assert {entry["repetition"] for entry in report["arms"]} == set(range(5))
    assert report["arms"][0]["path"] == "structural"
    assert report["arms"][6]["path"] == "transformer"
    assert all(arm["counts"]["requests_completed"] == 2 for arm in report["arms"])
    symmetric = [arm for arm in report["arms"] if arm["path"] == "symmetric"]
    assert all(arm["counts"]["stage2_completed"] == 1 for arm in symmetric)
    assert len(list(tmp_path.glob("arm-*.jsonl"))) == 30
    assert report["http_service_measurement"] is False
    with pytest.raises(FileExistsError):
        measurements.run_cost(records, engine, tmp_path, admission, warmups=3)


def test_cost_failure_leaves_prefix_not_success_or_retry(tmp_path):
    admission = Admission()

    def revoke(raw):
        admission.revoked = True
        return 0.1

    engine = measurements.RoutingEngine(
        revoke, lambda raw: 0.9, transformer_threshold=0.5
    )
    records = [
        {
            "record_id": "one",
            "raw_url": "one",
            "stage1_probability": 0.1,
            "transformer_probability": 0.9,
        }
    ]
    with pytest.raises(ValueError, match="revoked"):
        measurements.run_cost(records, engine, tmp_path, admission, warmups=0)
    stopped = json.loads((tmp_path / "cost-failure.json").read_bytes())
    assert stopped["completed_arms"] == 0
    assert not (tmp_path / "cost-summary.json").exists()


def test_actual_forward_hook_mismatch_prevents_completed_arm(tmp_path):
    engine = measurements.RoutingEngine(
        lambda raw: 0.1,
        lambda raw: 0.9,
        transformer_threshold=0.5,
        forward_counts=lambda: {"started": 0, "completed": 0},
    )
    records = [
        {
            "record_id": "one",
            "raw_url": "one",
            "stage1_probability": 0.1,
            "transformer_probability": 0.9,
        }
    ]
    with pytest.raises(
        ValueError, match="underlying_transformer_forward_counts_differ"
    ):
        measurements.run_cost(records, engine, tmp_path, Admission(), warmups=0)
    failure = json.loads((tmp_path / "cost-failure.json").read_bytes())
    assert failure["completed_arms"] == 1
    assert failure["active_arm"]["path"] == "transformer"
    assert not (tmp_path / "cost-summary.json").exists()
