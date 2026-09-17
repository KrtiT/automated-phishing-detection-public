import importlib.util
import json
from hashlib import sha256
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/describe_gmm_audit.py"
CONTRACT_ID = "rq2-gmm-development-v1"
CONTRACT_HASH = "22d32088b05e74432704f9671ab76ba28b4f573ead418846b23bc366315cb393"


def test_saved_trace_descriptor_exists():
    assert SCRIPT.is_file(), "the saved-trace descriptor has not been implemented"


@pytest.fixture
def descriptor():
    assert SCRIPT.is_file(), "the saved-trace descriptor has not been implemented"
    spec = importlib.util.spec_from_file_location("describe_gmm_audit", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _encode(value):
    return json.dumps(value, sort_keys=True).encode()


def _fixture(calibration=None, audit=None, threshold=238.45):
    calibration = list(map(float, range(252))) if calibration is None else calibration
    audit = [threshold + 1] * 28 + [0.0] * 224 if audit is None else audit
    traces = {"schema_version": 1, "contract_id": CONTRACT_ID}
    counts = {}
    offset = 0
    for name, scores in (("calibration", calibration), ("audit", audit)):
        rows = 256 + 64 * (len(scores) - 1)
        traces[name] = {
            "window_scores": scores,
            "window_end_positions": list(range(256, rows + 1, 64)),
            "input_row_positions": list(range(offset, offset + rows)),
            "domains": ["PRIVATE_DOMAIN_CANARY"],
            "record_ids": ["PRIVATE_RECORD_CANARY"],
        }
        counts[name] = {"rows": rows, "complete_windows": len(scores)}
        offset += rows
    alerts = sum(score > threshold for score in audit)
    summary = {
        "schema_version": 1,
        "status": "completed_development_validation",
        "analysis_stage": "development_validation_only",
        "contract": {"id": CONTRACT_ID, "sha256": CONTRACT_HASH},
        "threshold": threshold,
        "calibration_window_count": len(calibration),
        "audit_window_count": len(audit),
        "audit_alert_count": alerts,
        "audit_alert_fraction": alerts / len(audit),
        "false_alert_gate_met": 20 * alerts <= len(audit),
        "input_counts": {**counts, "validation": {"rows": offset}},
    }
    return summary, traces


def _describe(descriptor, summary, traces):
    audit_bytes = _encode(traces)
    summary["artifact_hashes"] = {
        "validation-audit.json": sha256(audit_bytes).hexdigest()
    }
    summary_bytes = _encode(summary)
    loaded = descriptor._load_summary(summary_bytes, sha256(summary_bytes).hexdigest())
    return descriptor._describe_audit(loaded, audit_bytes)


def test_fixed_quantiles_original_gate_and_finite_outlier(descriptor):
    summary, traces = _fixture()
    traces["calibration"]["window_scores"][-1] = 482882.36575319275
    result = _describe(descriptor, summary, traces)
    calibration = result["streams"]["calibration"]
    assert result["analysis_stage"] == "post_hoc_descriptive_only"
    assert calibration["score_summary"] == {
        "min": 0.0,
        "median": 125.5,
        "p90": 225.9,
        "p95": 238.45,
        "max": 482882.36575319275,
    }
    assert calibration["above_original_threshold_count"] == 13
    assert result["original_audit"] == {
        "threshold": 238.45,
        "alert_rule": "score > threshold",
        "alert_count": 28,
        "window_count": 252,
        "alert_fraction": 28 / 252,
        "gate": "20 * alert_windows <= complete_windows",
        "maximum_allowed_alerts": 12,
        "false_alert_gate_met": False,
    }
    assert (
        result["source_hashes"]["validation_audit_sha256"]
        == sha256(_encode(traces)).hexdigest()
    )
    assert (
        result["source_hashes"]["summary_sha256"]
        == sha256(_encode(summary)).hexdigest()
    )


def test_overlap_counts_and_consecutive_runs(descriptor):
    summary, traces = _fixture([0.0] * 6, [10, 10, -10, 10, -10, 10], 0.0)
    result = _describe(descriptor, summary, traces)
    audit = result["streams"]["audit"]
    assert audit["consecutive_alert_run_count"] == 3
    assert audit["consecutive_alert_run_length_histogram"] == {"1": 2, "2": 1}
    assert audit["alert_window_pair_counts_by_shared_rows"] == {
        "64": 1,
        "128": 2,
        "192": 1,
    }
    assert audit["unique_rows_covered_by_alert_windows"] == 576
    assert audit["alert_window_row_memberships_counting_overlap"] == 1024
    assert result["cross_stream_membership_overlap_count"] == 0
    assert result["streams"]["calibration"]["above_original_threshold_count"] == 0


def test_output_excludes_private_fields_and_traces(descriptor):
    result = _describe(descriptor, *_fixture())
    output = json.dumps(result)
    for forbidden in (
        "PRIVATE_",
        "domains",
        "record_ids",
        "input_row_positions",
        "window_scores",
        "window_end_positions",
        "paths",
    ):
        assert forbidden not in output


def test_summary_pin_rejects_before_reading_private_file(descriptor, tmp_path):
    summary = tmp_path / "summary.json"
    summary.write_bytes(b"not the official summary")
    with pytest.raises(ValueError, match="summary SHA-256"):
        descriptor.describe_audit(summary, tmp_path / "MUST_NOT_BE_READ.json")
    assert descriptor.EXPECTED_SUMMARY_SHA256 == (
        "6f695138a302e854e1e5af590152e289486affe8ccdf75510ca9a5dcaad3b523"
    )


def test_trace_hash_rejects_before_json_parsing(descriptor):
    summary, traces = _fixture()
    _describe(descriptor, summary, traces)
    with pytest.raises(ValueError, match="audit SHA-256"):
        descriptor._describe_audit(summary, b"PRIVATE_CORRUPTED_JSON")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "1"])
def test_rejects_nonfinite_or_nonnumeric_scores(descriptor, value):
    summary, traces = _fixture()
    traces["audit"]["window_scores"][0] = value
    with pytest.raises(ValueError, match="finite numeric"):
        _describe(descriptor, summary, traces)


@pytest.mark.parametrize("mutation", ["end", "length", "duplicate", "overlap", "range"])
def test_rejects_inconsistent_windows_and_membership(descriptor, mutation):
    summary, traces = _fixture()
    audit = traces["audit"]
    if mutation == "end":
        audit["window_end_positions"][0] = 255
    elif mutation == "length":
        audit["window_scores"].pop()
    elif mutation == "duplicate":
        audit["input_row_positions"][1] = audit["input_row_positions"][0]
    elif mutation == "overlap":
        audit["input_row_positions"][0] = traces["calibration"]["input_row_positions"][
            0
        ]
    else:
        audit["input_row_positions"][-1] = summary["input_counts"]["validation"]["rows"]
    with pytest.raises(ValueError):
        _describe(descriptor, summary, traces)


@pytest.mark.parametrize(
    "field", ["audit_alert_count", "audit_window_count", "threshold"]
)
def test_rejects_invalid_summary_counts_or_threshold(descriptor, field):
    summary, traces = _fixture()
    summary[field] = True
    with pytest.raises(ValueError):
        _describe(descriptor, summary, traces)


def test_rejects_disagreement_with_original_result(descriptor):
    summary, traces = _fixture()
    summary["audit_alert_count"] = 27
    with pytest.raises(ValueError, match="original audit"):
        _describe(descriptor, summary, traces)


@pytest.mark.parametrize("target", ["summary", "trace"])
def test_rejects_wrong_contract(descriptor, target):
    summary, traces = _fixture()
    if target == "summary":
        summary["contract"]["sha256"] = "0" * 64
    else:
        traces["contract_id"] = "different-contract"
    with pytest.raises(ValueError, match="contract"):
        _describe(descriptor, summary, traces)


def test_cli_prints_only_json_and_does_not_expose_paths_on_failure(
    descriptor, tmp_path, capsys
):
    result = descriptor.main(
        [
            "--summary",
            str(tmp_path / "PRIVATE_MISSING_SUMMARY"),
            "--audit",
            str(tmp_path / "PRIVATE_MISSING_AUDIT"),
        ]
    )
    captured = capsys.readouterr()
    assert result == 2
    assert captured.err == ""
    assert json.loads(captured.out)["status"] == "failed"
    assert "PRIVATE_" not in captured.out


def test_cli_rejects_threshold_override(descriptor, capsys):
    assert descriptor.main(["--threshold", "0"]) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "failed"
