"""Invented prepared rows only, with no source files or model execution."""

import importlib
import importlib.util
import json
from functools import cache
from hashlib import sha256

from automated_phishing_detection.evaluation_manifest import ManifestRecord
from automated_phishing_detection.phishvn import (
    NormalizedExternalRow,
    prepare_external_rows,
)
from automated_phishing_detection.protocol_preflight import parse_suffix_rules

SOURCE = "a" * 64
ROLE_FIELDS = {
    "gold": ("ncsc", "phishing", "gold"),
    "certified": ("trusted_registry", "legitimate", "certified"),
    "silver": ("ncsc", "phishing", "silver"),
    "bronze": ("chongluadao_openphish", "phishing", "bronze"),
    "tranco": ("tranco", "reference_negative", "control"),
}


def api():
    name = "automated_phishing_detection.study_feasibility"
    assert importlib.util.find_spec(name), "missing preparation-only feasibility"
    return importlib.import_module(name)


@cache
def internal_rows(negatives=1, positives=2, positive_domains=2):
    records = []
    for position in range(negatives + positives):
        positive = position >= negatives
        domain = (
            f"positive-{(position - negatives) % positive_domains}.test"
            if positive
            else "negative.test"
        )
        raw_url = f"https://{domain}/record-{position}"
        records.append(
            ManifestRecord(
                f"phiusiil-row-v1:{SOURCE}:{position + 1:016x}",
                raw_url,
                sha256(raw_url.encode("utf-8")).hexdigest(),
                domain,
                int(positive),
                "group_test",
            )
        )
    return tuple(records)


def external_row(position, role, gold_domains):
    domain = (
        f"gold-{position % gold_domains}.test"
        if role == "gold" and gold_domains
        else f"external-{position}.test"
    )
    return NormalizedExternalRow(
        f"external-{position}",
        "test",
        position,
        f"https://{domain}/record-{position}",
        *ROLE_FIELDS[role],
        "test",
    )


def _quarantined(start, count):
    return tuple(
        NormalizedExternalRow(
            f"quarantine-{position}",
            "test",
            start + position,
            "invalid URL",
            "ncsc",
            "phishing",
            "gold",
            "test",
        )
        for position in range(1, count + 1)
    )


@cache
def external_rows(
    roles=("gold", "gold", "certified", "silver", "bronze", "tranco"),
    *,
    gold_domains=None,
    quarantine_count=0,
):
    rows = tuple(
        external_row(position, role, gold_domains)
        for position, role in enumerate(roles, 1)
    )
    quarantined = _quarantined(len(rows), quarantine_count)
    return prepare_external_rows(
        (*rows, *quarantined),
        published_split_counts={
            "train": 0,
            "val": 0,
            "test": len(rows) + len(quarantined),
        },
        test_split="test",
        suffix_rules=parse_suffix_rules("test\n"),
        phiusiil_domains=frozenset(),
    )


def assess(internal=None, external=None):
    return json.loads(
        api().assess_preparation_feasibility(
            internal_rows() if internal is None else internal,
            external_rows() if external is None else external,
        )
    )


def shortages(result):
    return {item["requirement"]: item for item in result["shortages"]}
