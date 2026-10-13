"""Fixture tests of retained-prefix scoring and spelling comparisons."""

import json

import pytest
import revision_tasks as tasks


def test_formatting_checks_original_before_companions_and_preserves_all(tmp_path):
    calls = []

    def score(raw):
        calls.append(raw)
        return 0.8 if raw.startswith("https:") else 0.1

    saved = [
        {
            "record": {
                "record_id": "one",
                "raw_url": "https://safe.test/A?x=Z",
                "role": "gold",
                "is_phishing": 1,
            },
            "primary": {
                "stage1_probability": 0.8,
                "stage1_decision": 1,
                "features": [1.0],
            },
        }
    ]
    report = tasks.run_formatting(
        saved, score, lambda raw: [1.0], ["constant"], tmp_path, lambda: None
    )
    assert calls == [
        "https://safe.test/A?x=Z",
        "http://safe.test/A?x=Z",
        "https://safe.test/A?x=Z",
    ]
    assert report["groups"]["gold"]["http"]["true_positive"] == 0
    assert report["groups"]["gold"]["https"]["true_positive"] == 1
    assert report["original_replay_verified"] is True
    assert (
        len((tmp_path / "formatting-predictions.jsonl").read_text().splitlines()) == 1
    )


def test_original_mismatch_stops_before_companion_scoring(tmp_path):
    calls = []

    def score(raw):
        calls.append(raw)
        return 0.7

    saved = [
        {
            "record": {
                "record_id": "one",
                "raw_url": "https://safe.test",
                "role": "gold",
                "is_phishing": 1,
            },
            "primary": {
                "stage1_probability": 0.8,
                "stage1_decision": 1,
                "features": [1.0],
            },
        }
    ]
    with pytest.raises(ValueError, match="original_replay_mismatch"):
        tasks.run_formatting(
            saved, score, lambda raw: [1.0], ["constant"], tmp_path, lambda: None
        )
    assert calls == ["https://safe.test"]
    assert (tmp_path / "formatting-failure.json").exists()
    assert not (tmp_path / "formatting-summary.json").exists()


def test_new_cohort_outputs_retain_all_completed_predictions_before_error(tmp_path):
    records = [
        {"record_id": str(number), "raw_url": "https://safe.test"}
        for number in range(3)
    ]

    class Scorer:
        def __init__(self):
            self.calls = 0

        def score_all(self, raw):
            self.calls += 1
            if self.calls == 3:
                raise ValueError("model failed")
            return type(
                "Result",
                (),
                {"stage1_probability": 0.8, "transformer_probability": 0.7},
            )()

    with pytest.raises(ValueError):
        tasks.run_cohort(records, Scorer(), 0.5, tmp_path, lambda: None)
    lines = (tmp_path / "routing-predictions.jsonl").read_text().splitlines()
    assert len(lines) == 2
    failure = json.loads((tmp_path / "routing-failure.json").read_bytes())
    assert failure["completed_predictions"] == 2
    assert failure["active_record_id"] == "2"
    assert not (tmp_path / "routing-summary.json").exists()
