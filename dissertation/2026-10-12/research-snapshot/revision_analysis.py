"""Fixed post-review analysis, without model loading, fitting or data access."""

import math
from collections import Counter, defaultdict
from dataclasses import asdict
from urllib.parse import urlsplit

import numpy as np
from routing_replay import THRESHOLD, gate_mask, random_mask
from scipy.stats import beta
from sklearn.metrics import average_precision_score, roc_auc_score

from automated_phishing_detection.paired_inference import (
    PairedTable,
    paired_exact_envelope,
    paired_score_interval,
)


def force_scheme(raw, scheme):
    parsed = urlsplit(raw)
    if (
        scheme not in {"http", "https"}
        or parsed.scheme.lower() not in {"http", "https"}
        or not parsed.hostname
    ):
        raise ValueError("absolute_http_input_required")
    return scheme + ":" + raw.split(":", 1)[1]


def routing_decisions(rows, transformer_threshold):
    if (
        not rows
        or not math.isfinite(transformer_threshold)
        or not 0 <= transformer_threshold <= 1
    ):
        raise ValueError("invalid_routing_inputs")
    identifiers = [row["record_id"] for row in rows]
    if len(set(identifiers)) != len(rows):
        raise ValueError("duplicate_identity")
    for row in rows:
        if any(
            type(row[name]) not in (int, float) or not 0 <= row[name] <= 1
            for name in ("stage1_probability", "transformer_probability")
        ):
            raise ValueError("invalid_model_probability")
    scores = [row["stage1_probability"] for row in rows]
    structural = [int(score >= THRESHOLD) for score in scores]
    transformer = [
        int(row["transformer_probability"] >= transformer_threshold) for row in rows
    ]
    masks = {
        "structural": [False] * len(rows),
        "transformer": [True] * len(rows),
        "original": gate_mask(scores, "symmetric", 0),
        "symmetric": gate_mask(scores, "symmetric", 0.20),
        "directional": gate_mask(scores, "below", 0.20),
    }
    masks["symmetric_control"] = random_mask(identifiers, sum(masks["symmetric"]))
    masks["directional_control"] = random_mask(identifiers, sum(masks["directional"]))
    decisions = {
        name: [
            expert if selected else baseline
            for expert, baseline, selected in zip(
                transformer, structural, mask, strict=True
            )
        ]
        for name, mask in masks.items()
    }
    return decisions, masks


def date_sensitivity(rows, candidate, reference):
    if not rows or len(rows) != len(candidate) or len(rows) != len(reference):
        raise ValueError("invalid_date_sensitivity_shape")
    groups = defaultdict(list)
    for row, left, right in zip(rows, candidate, reference, strict=True):
        if not row["collected_at"].strip() or any(
            type(value) is not int or value not in (0, 1) for value in (left, right)
        ):
            raise ValueError("invalid_dated_pair")
        groups[row["collected_at"]].append(left - right)
    ordered = [groups[name] for name in sorted(groups)]
    sums = np.asarray([sum(values) for values in ordered], dtype=np.int64)
    counts = np.asarray([len(values) for values in ordered], dtype=np.int64)
    generator = np.random.Generator(np.random.PCG64(20261012))
    indices = generator.integers(len(ordered), size=(2000, len(ordered)))
    effects = sums[indices].sum(axis=1) / counts[indices].sum(axis=1)
    lower, upper = np.quantile(effects, [0.025, 0.975], method="linear")
    return {
        "method": "source_event_date_cluster_percentile_sensitivity",
        "groups": len(ordered),
        "largest_group": int(counts.max()),
        "replicates": 2000,
        "generator": "PCG64",
        "seed": 20261012,
        "lower": float(lower),
        "upper": float(upper),
        "inferential_decision": False,
        "campaign_independence_established": False,
    }


def _metrics(labels, decisions, baseline, mask):
    counts = Counter(zip(labels, decisions, strict=True))
    positives, negatives = sum(labels), len(labels) - sum(labels)
    true_positive, false_positive = counts[(1, 1)], counts[(0, 1)]
    result = {
        "rows": len(labels),
        "positive": positives,
        "negative": negatives,
        "true_positive": true_positive,
        "false_negative": counts[(1, 0)],
        "false_positive": false_positive,
        "true_negative": counts[(0, 0)],
        "recall": true_positive / positives,
        "fpr": false_positive / negatives,
        "fpr_upper_95_iid_conditional": float(
            beta.ppf(0.95, false_positive + 1, negatives - false_positive)
        )
        if false_positive < negatives
        else 1.0,
        "logical_calls": sum(mask),
        "call_fraction": sum(mask) / len(mask),
    }
    for label, population in ((1, "positive"), (0, "negative")):
        selected = [
            (decision, original, call)
            for truth, decision, original, call in zip(
                labels, decisions, baseline, mask, strict=True
            )
            if truth == label
        ]
        result[population + "_calls"] = sum(
            call for decision, original, call in selected
        )
        result[population + "_repairs"] = sum(
            original != label and decision == label
            for decision, original, call in selected
        )
        result[population + "_harms"] = sum(
            original == label and decision != label
            for decision, original, call in selected
        )
    return result


def analyze_routing(rows, *, transformer_threshold):
    if len({row["registrable_domain"] for row in rows}) != len(rows):
        raise ValueError("one_row_per_domain_required")
    labels = [row["is_phishing"] for row in rows]
    if any(type(label) is not int or label not in (0, 1) for label in labels) or set(
        labels
    ) != {0, 1}:
        raise ValueError("both_binary_label_populations_required")
    decisions, masks = routing_decisions(rows, transformer_threshold)
    systems = {
        name: _metrics(labels, values, decisions["structural"], masks[name])
        for name, values in decisions.items()
    }
    positive_indices = [index for index, label in enumerate(labels) if label]
    positive_rows = [rows[index] for index in positive_indices]
    contrasts = {}
    for candidate, reference in (
        ("original", "transformer"),
        ("symmetric", "transformer"),
        ("symmetric", "structural"),
        ("directional", "transformer"),
    ):
        left = [decisions[candidate][index] for index in positive_indices]
        right = [decisions[reference][index] for index in positive_indices]
        counts = Counter(zip(left, right, strict=True))
        table = PairedTable(*(counts[key] for key in ((1, 1), (1, 0), (0, 1), (0, 0))))
        contrasts[candidate + "_minus_" + reference] = {
            "cells": asdict(table),
            "exact": asdict(paired_exact_envelope(table)),
            "newcombe10_sensitivity": asdict(paired_score_interval(table)),
            "date_cluster_sensitivity": date_sensitivity(positive_rows, left, right),
            "coverage": "individual_contrast_conditional_on_iid_labeled_pairs",
        }
    statements = {}
    for name in ("original", "symmetric"):
        lower = contrasts[name + "_minus_transformer"]["exact"]["lower"]
        statements[name + "_noninferiority"] = {
            "margin": -0.02,
            "lower": lower,
            "supported": lower >= -0.02,
            "interpretation": "conditional_recall_only_not_joint_deployment_adequacy",
        }
    requirements = {
        "positive_recall_difference_lower_bound": contrasts[
            "symmetric_minus_structural"
        ]["exact"]["lower"]
        > 0,
        "observed_fpr_at_most_one_percent": systems["symmetric"]["fpr"] <= 0.01,
        "calls_at_most_thirty_percent": systems["symmetric"]["call_fraction"] <= 0.30,
    }
    statements["symmetric_benefit_and_budget"] = {
        "conditions": requirements,
        "supported": all(requirements.values()),
    }
    return {
        "systems": systems,
        "contrasts": contrasts,
        "primary_statements": statements,
        "ranking": {
            name: {
                "roc_auc": float(roc_auc_score(labels, [row[column] for row in rows])),
                "average_precision": float(
                    average_precision_score(labels, [row[column] for row in rows])
                ),
            }
            for name, column in (
                ("structural", "stage1_probability"),
                ("transformer", "transformer_probability"),
            )
        },
        "original_hypotheses_reclassified": False,
        "estimand": "domain_balanced_retained_source_frame",
    }
