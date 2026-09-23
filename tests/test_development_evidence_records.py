"""Public summaries retain the producer's JSON numbers and original byte links."""

import json
from hashlib import sha256
from pathlib import Path

from automated_phishing_detection import execution_receipt


def test_accepted_development_report_preserves_original_summary_hashes():
    path = (
        Path(__file__).resolve().parents[1]
        / "reports/secondary-development-correction-v2-summary.json"
    )
    report = json.loads(path.read_bytes())
    summary = report["completion"]

    def digest(value):
        return sha256(execution_receipt._json_bytes(value, "summary")).hexdigest()

    assert digest(summary) == report["completion_summary_sha256"]
    assert report["status"] == "accepted_development_evidence"
    assert type(report["execution_observation"]["parent_exit_code"]) is int
    assert report["execution_observation"]["parent_exit_code"] == 0
    assert summary["worker_exit_codes"] == {"retained_audit": 0, "random_forest": 0}
    assert all(type(code) is int for code in summary["worker_exit_codes"].values())
    assert type(summary["new_fits"]) is int and summary["new_fits"] == 1
    assert summary["original_aggregate_accepted"] is False
    assert summary["protected_evaluation_authorized"] is False
    for stage, key in (
        ("retained_audit", "retained_audit"),
        ("random_forest", "corrected_random_forest"),
    ):
        assert digest(summary[key]) == report["receipt_sha256"][f"{stage}.json"]
    for member in summary["retained_audit"]["result"]["members"]:
        assert digest(member["summary"]) == member["public_summary_sha256"]
    formatting = summary["retained_audit"]["result"]["members"][1]["summary"]["result"]
    assert (
        type(formatting["scoring_audit"]["max_absolute_decision_difference"]) is float
    )
    assert (
        type(formatting["scoring_audit"]["max_absolute_probability_difference"])
        is float
    )
    assert type(formatting["validation_threshold"]["observed_fpr"]) is float
