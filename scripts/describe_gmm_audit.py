"""Describe authenticated saved GMM windows without rescoring or retuning."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from hashlib import sha256
from pathlib import Path

import numpy as np

EXPECTED_SUMMARY_SHA256 = (
    "6f695138a302e854e1e5af590152e289486affe8ccdf75510ca9a5dcaad3b523"
)
CONTRACT_ID = "rq2-gmm-development-v1"
CONTRACT_SHA256 = "22d32088b05e74432704f9671ab76ba28b4f573ead418846b23bc366315cb393"
WINDOW_LENGTH = 256
WINDOW_STRIDE = 64


def _finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _count(value):
    if type(value) is not int or value < 0:
        raise ValueError("counts must be nonnegative integers")
    return value


def _json_object(content):
    try:
        value = json.loads(content)
    except (ValueError, UnicodeError) as error:
        raise ValueError("input is not valid JSON") from error
    if type(value) is not dict:
        raise ValueError("input must be a JSON object")
    return value


def _load_summary(content, expected_sha256):
    """The hash parameter is a private synthetic-fixture seam, not a CLI option."""
    digest = sha256(content).hexdigest()
    if digest != expected_sha256:
        raise ValueError("summary SHA-256 does not match the pinned record")
    summary = _json_object(content)
    try:
        if (
            type(summary["schema_version"]) is not int
            or summary["schema_version"] != 1
            or summary["contract"]["id"] != CONTRACT_ID
            or summary["contract"]["sha256"] != CONTRACT_SHA256
        ):
            raise ValueError("summary contract identity is invalid")
        if (
            summary["status"] != "completed_development_validation"
            or summary["analysis_stage"] != "development_validation_only"
        ):
            raise ValueError("summary is not a completed development result")
        if not _finite_number(summary["threshold"]):
            raise ValueError("original threshold must be finite numeric")
        for name in (
            "calibration_window_count",
            "audit_window_count",
            "audit_alert_count",
        ):
            _count(summary[name])
        if (
            not _finite_number(summary["audit_alert_fraction"])
            or type(summary["false_alert_gate_met"]) is not bool
        ):
            raise ValueError("original audit result is invalid")
        trace_hash = summary["artifact_hashes"]["validation-audit.json"]
        if (
            type(trace_hash) is not str
            or len(trace_hash) != 64
            or any(character not in "0123456789abcdef" for character in trace_hash)
        ):
            raise ValueError("audit SHA-256 is invalid")
    except (KeyError, TypeError) as error:
        raise ValueError("summary structure is invalid") from error
    return {**summary, "_summary_sha256": digest}


def _stream_description(trace, counts, total_rows, threshold):
    scores = trace["window_scores"]
    ends = trace["window_end_positions"]
    positions = trace["input_row_positions"]
    rows = _count(counts["rows"])
    windows = _count(counts["complete_windows"])
    if (
        type(scores) is not list
        or not scores
        or not all(_finite_number(score) for score in scores)
    ):
        raise ValueError("window scores must be a nonempty finite numeric list")
    if (
        type(positions) is not list
        or len(positions) != rows
        or any(
            type(position) is not int or not 0 <= position < total_rows
            for position in positions
        )
        or positions != sorted(set(positions))
    ):
        raise ValueError("stream membership positions are invalid")
    if (
        type(ends) is not list
        or any(type(end) is not int for end in ends)
        or ends != list(range(WINDOW_LENGTH, rows + 1, WINDOW_STRIDE))
        or len(scores) != len(ends)
        or len(scores) != windows
    ):
        raise ValueError("saved window alignment or counts are invalid")

    quantiles = np.quantile(scores, [0, 0.5, 0.9, 0.95, 1], method="linear")
    if not np.all(np.isfinite(quantiles)):
        raise ValueError("score summaries must be finite numeric")
    alerted = [index for index, score in enumerate(scores) if score > threshold]
    runs = []
    previous = None
    covered = set()
    pairs = Counter({64: 0, 128: 0, 192: 0})
    for offset, index in enumerate(alerted):
        if previous is None or index != previous + 1:
            runs.append(1)
        else:
            runs[-1] += 1
        previous = index
        covered.update(positions[ends[index] - WINDOW_LENGTH : ends[index]])
        for other in alerted[offset + 1 :]:
            shared = WINDOW_LENGTH - (ends[other] - ends[index])
            if shared <= 0:
                break
            pairs[shared] += 1
    description = {
        "window_count": windows,
        "row_count": rows,
        "score_summary": dict(
            zip(
                ("min", "median", "p90", "p95", "max"),
                map(float, quantiles),
                strict=True,
            )
        ),
        "above_original_threshold_count": len(alerted),
        "above_original_threshold_fraction": len(alerted) / windows,
        "equal_original_threshold_count": sum(score == threshold for score in scores),
        "consecutive_alert_run_count": len(runs),
        "consecutive_alert_run_length_histogram": {
            str(length): count for length, count in sorted(Counter(runs).items())
        },
        "alert_window_pair_counts_by_shared_rows": {
            str(shared): count for shared, count in sorted(pairs.items())
        },
        "unique_rows_covered_by_alert_windows": len(covered),
        "alert_window_row_memberships_counting_overlap": len(alerted) * WINDOW_LENGTH,
        "rows_outside_complete_windows": rows - ends[-1],
    }
    return description, set(positions)


def _describe_audit(summary, content):
    """Authenticate bytes before parsing; only aggregate saved score/position fields."""
    digest = sha256(content).hexdigest()
    if digest != summary["artifact_hashes"]["validation-audit.json"]:
        raise ValueError("audit SHA-256 does not match the pinned summary")
    traces = _json_object(content)
    try:
        if (
            type(traces["schema_version"]) is not int
            or traces["schema_version"] != 1
            or traces["contract_id"] != CONTRACT_ID
        ):
            raise ValueError("audit contract identity is invalid")
        total_rows = _count(summary["input_counts"]["validation"]["rows"])
        streams, membership = {}, {}
        for name in ("calibration", "audit"):
            streams[name], membership[name] = _stream_description(
                traces[name],
                summary["input_counts"][name],
                total_rows,
                summary["threshold"],
            )
            if streams[name]["window_count"] != summary[f"{name}_window_count"]:
                raise ValueError("summary window counts disagree with saved windows")
        overlap = len(membership["calibration"] & membership["audit"])
        if overlap or sum(map(len, membership.values())) != total_rows:
            raise ValueError(
                "calibration and audit membership must partition validation"
            )
        if streams["calibration"]["score_summary"]["p95"] != summary["threshold"]:
            raise ValueError(
                "saved calibration quantile disagrees with original threshold"
            )
        audit = streams["audit"]
        alerts, windows = audit["above_original_threshold_count"], audit["window_count"]
        gate_met = 20 * alerts <= windows
        if (
            alerts != summary["audit_alert_count"]
            or alerts / windows != summary["audit_alert_fraction"]
            or gate_met != summary["false_alert_gate_met"]
        ):
            raise ValueError("saved windows disagree with the original audit result")
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError("audit or summary structure is invalid") from error
    return {
        "schema_version": 1,
        "analysis_stage": "post_hoc_descriptive_only",
        "contract_id": CONTRACT_ID,
        "source_hashes": {
            "summary_sha256": summary["_summary_sha256"],
            "validation_audit_sha256": digest,
        },
        "quantile_method": "linear",
        "original_audit": {
            "threshold": summary["threshold"],
            "alert_rule": "score > threshold",
            "alert_count": alerts,
            "window_count": windows,
            "alert_fraction": alerts / windows,
            "gate": "20 * alert_windows <= complete_windows",
            "maximum_allowed_alerts": windows // 20,
            "false_alert_gate_met": gate_met,
        },
        "cross_stream_membership_overlap_count": overlap,
        "streams": streams,
    }


def describe_audit(summary_path: Path, audit_path: Path) -> dict:
    """Describe this recorded run only; reject an unpinned summary before audit access."""
    summary = _load_summary(summary_path.read_bytes(), EXPECTED_SUMMARY_SHA256)
    return _describe_audit(summary, audit_path.read_bytes())


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError("arguments must be --summary PATH --audit PATH")


def main(argv=None) -> int:
    parser = _Parser(add_help=False, allow_abbrev=False)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    try:
        arguments = parser.parse_args(argv)
        result = describe_audit(arguments.summary, arguments.audit)
    except OSError:
        result = {"status": "failed", "error": "could not read a required input"}
    except ValueError as error:
        result = {"status": "failed", "error": str(error)}
    print(json.dumps(result, allow_nan=False, sort_keys=True, separators=(",", ":")))
    return 2 if result.get("status") == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
