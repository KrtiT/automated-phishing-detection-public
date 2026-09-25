"""Invented full-stream inputs for external replay composition."""

import importlib
import importlib.util
import json
from dataclasses import replace

from test_external_primary import external_session
from test_retained_drift import _snapshots

from automated_phishing_detection import (
    evaluation_producer,
    phishvn,
    protocol_preflight,
)
from automated_phishing_detection._external_primary_progress import primary_rows_bytes
from automated_phishing_detection.external_primary import (
    _receipt,
    score_external_primary,
)
from automated_phishing_detection.retained_drift import load_retained_drift_reference


def replay_module():
    name = "automated_phishing_detection.external_replay"
    assert importlib.util.find_spec(name) is not None, "external replay missing"
    return importlib.import_module(name)


def _prepared(count):
    mappings = (
        ("ncsc", "phishing", "silver"),
        ("tranco", "reference_negative", "control"),
        ("ncsc", "phishing", "gold"),
        ("trusted_registry", "legitimate", "certified"),
    )
    rows = tuple(
        phishvn.NormalizedExternalRow(
            f"row-{index}",
            "test",
            index + 1,
            f"https://host{index}.example{index}.com/path",
            *mappings[index % 2 if index < 256 else 2 + index % 2],
            "test",
        )
        for index in range(count)
    )
    return phishvn.prepare_external_rows(
        rows,
        published_split_counts={"train": 0, "val": 0, "test": count},
        test_split="test",
        suffix_rules=protocol_preflight.parse_suffix_rules("com\n"),
        phiusiil_domains=frozenset(),
    )


def inputs(monkeypatch, count=320):
    prepared = _prepared(count)
    session, _ = external_session(monkeypatch)
    session.evaluation.primary.models.monitor_boundary = 1.0
    primary = score_external_primary(prepared, session)
    secondary = evaluation_producer.score_bound_secondary(
        session.evaluation.secondary,
        tuple(row.raw_url for row in primary.records),
        tuple(row.stage1_probability for row in primary.scores),
        tuple(row.transformer_probability for row in primary.scores),
    )
    reference = load_retained_drift_reference(**_snapshots())
    return primary, secondary, reference


def relink_primary(primary, *, scores=None, thresholds=None):
    scores = primary.scores if scores is None else scores
    thresholds = dict(primary.thresholds) if thresholds is None else thresholds
    checkpoint = primary_rows_bytes(primary.records, scores)
    source_hash = json.loads(primary.receipt_bytes)["retained_test_sha256"]
    return replace(
        primary,
        scores=scores,
        thresholds=tuple(thresholds.items()),
        checkpoint_bytes=checkpoint,
        receipt_bytes=_receipt(
            source_hash, checkpoint, thresholds, primary.inference_counts
        ),
    )
