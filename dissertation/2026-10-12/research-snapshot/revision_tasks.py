"""The F and R tasks receive live parent admission, not standalone authority."""

import math
import os
from collections import Counter

import numpy as np
from revision_analysis import analyze_routing, force_scheme
from revision_population import _bytes, _write
from routing_replay import THRESHOLD


def _stream(path):
    return os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb")


def _append(stream, record):
    stream.write(_bytes(record))
    stream.flush()


def _formatting_summary(results, feature_names):
    groups = {}
    for role in sorted({row["role"] for row in results}):
        group = [row for row in results if row["role"] == role]
        variants = {}
        for variant in ("original", "http", "https"):
            counts = Counter(
                (row["is_phishing"], row["variants"][variant]["decision"])
                for row in group
                if row["is_phishing"] is not None
            )
            positive = sum(counts[(1, decision)] for decision in (0, 1))
            negative = sum(counts[(0, decision)] for decision in (0, 1))
            shifts = [
                row["variants"][variant]["probability"]
                - row["variants"]["original"]["probability"]
                for row in group
            ]
            deltas = np.asarray(
                [row["variants"][variant]["features"] for row in group]
            ) - np.asarray([row["variants"]["original"]["features"] for row in group])
            variants[variant] = {
                "rows": len(group),
                "true_positive": counts[(1, 1)],
                "false_negative": counts[(1, 0)],
                "false_positive": counts[(0, 1)],
                "true_negative": counts[(0, 0)],
                "recall": counts[(1, 1)] / positive if positive else None,
                "fpr": counts[(0, 1)] / negative if negative else None,
                "alerts": sum(row["variants"][variant]["decision"] for row in group),
                "decision_flips": sum(
                    row["variants"][variant]["decision"]
                    != row["variants"]["original"]["decision"]
                    for row in group
                ),
                "probability_shift": {
                    "mean": float(np.mean(shifts)),
                    "min": min(shifts),
                    "max": max(shifts),
                    "median": float(np.median(shifts)),
                },
                "features": {
                    name: {
                        "changed_rows": int(np.count_nonzero(deltas[:, index])),
                        "mean_delta": float(deltas[:, index].mean()),
                        "min_delta": float(deltas[:, index].min()),
                        "max_delta": float(deltas[:, index].max()),
                    }
                    for index, name in enumerate(feature_names)
                },
            }
        groups[role] = variants
    return {
        "groups": groups,
        "rows": len(results),
        "original_replay_verified": True,
        "intervention": "prefix_spelling_only_frozen_model_and_threshold",
        "threshold": THRESHOLD,
        "original_hypotheses_reclassified": False,
    }


def run_formatting(saved_rows, score, features, feature_names, root, require_admission):
    require_admission()
    _write(
        root / "formatting-started.json",
        _bytes({"rows": len(saved_rows), "variants": ["original", "http", "https"]}),
    )
    results = []
    active = None
    try:
        with (
            _stream(root / "formatting-events.jsonl") as events,
            _stream(root / "formatting-predictions.jsonl") as output,
        ):
            for saved in saved_rows:
                require_admission()
                record = saved["record"]
                active = record["record_id"]
                raw = record["raw_url"]
                result = {
                    "record_id": active,
                    "role": record["role"],
                    "is_phishing": None
                    if record["role"] == "tranco"
                    else record["is_phishing"],
                    "variants": {},
                }
                for variant in ("original", "http", "https"):
                    require_admission()
                    text = raw if variant == "original" else force_scheme(raw, variant)
                    values = list(features(text))
                    probability = score(text)
                    if (
                        len(values) != len(feature_names)
                        or any(not math.isfinite(value) for value in values)
                        or not 0 <= probability <= 1
                    ):
                        raise ValueError("invalid_formatting_output")
                    current = {
                        "features": values,
                        "probability": probability,
                        "decision": int(probability >= THRESHOLD),
                    }
                    _append(
                        events, {"record_id": active, "variant": variant, **current}
                    )
                    if variant == "original" and (
                        probability != saved["primary"]["stage1_probability"]
                        or current["decision"] != saved["primary"]["stage1_decision"]
                        or values != saved["primary"]["features"]
                    ):
                        raise ValueError("original_replay_mismatch")
                    result["variants"][variant] = current
                _append(output, result)
                results.append(result)
            os.fsync(events.fileno())
            os.fsync(output.fileno())
        report = _formatting_summary(results, feature_names)
        require_admission()
        _write(root / "formatting-summary.json", _bytes(report))
        return report
    except BaseException as error:
        _write(
            root / "formatting-failure.json",
            _bytes(
                {
                    "error_type": type(error).__name__,
                    "completed_records": len(results),
                    "active_record_id": active,
                }
            ),
        )
        raise


def run_cohort(rows, scorer, transformer_threshold, root, require_admission):
    require_admission()
    _write(
        root / "routing-started.json",
        _bytes({"rows": len(rows), "fit": False, "threshold_selection": False}),
    )
    predictions = []
    active = None
    try:
        with _stream(root / "routing-predictions.jsonl") as output:
            for row in rows:
                require_admission()
                active = row["record_id"]
                scores = scorer.score_all(row["raw_url"])
                result = {
                    **row,
                    "stage1_probability": scores.stage1_probability,
                    "transformer_probability": scores.transformer_probability,
                }
                _append(output, result)
                predictions.append(result)
            os.fsync(output.fileno())
        summary = analyze_routing(
            predictions, transformer_threshold=transformer_threshold
        )
        require_admission()
        _write(root / "routing-summary.json", _bytes(summary))
        return predictions, summary
    except BaseException as error:
        _write(
            root / "routing-failure.json",
            _bytes(
                {
                    "error_type": type(error).__name__,
                    "completed_predictions": len(predictions),
                    "active_record_id": active,
                }
            ),
        )
        raise
