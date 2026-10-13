"""Independent retained-frame reconstruction, without fitting or predictions."""

from collections import defaultdict
from hashlib import sha256

from automated_phishing_detection.phiusiil import PreparationError, canonicalize_url
from automated_phishing_detection.protocol_preflight import (
    PreflightError,
    normalize_hostname,
    registrable_domain,
)


def _hash(prefix, value):
    return sha256((prefix + value).encode()).hexdigest()


def verify_selection(document, rules, exposed, observed):
    records = []
    labels = defaultdict(set)
    splits = defaultdict(set)
    for ordinal, original in enumerate(document["rows"]):
        row = dict(
            zip(
                document["headers"][original["source_member"]],
                original["cells"],
                strict=True,
            )
        )
        try:
            canonical = canonicalize_url(row["url_norm"])
            host = normalize_hostname(canonical)
            domain = registrable_domain(host, rules)
        except (PreparationError, PreflightError):
            continue
        splits[domain].add(row["split"])
        labels[canonical].add(row["label"])
        records.append((row, original, ordinal, canonical, host, domain))
    candidates = defaultdict(list)
    for row, original, ordinal, canonical, host, domain in records:
        population = None
        if row["source"] == "tinnhiemmang" and row["label"] == "phishing":
            population = "gold_positive"
        if (
            row["source"] in {"tinnhiem_org", "tinnhiem_web"}
            and row["label"] == "benign"
        ):
            population = "certified_negative"
        if (
            population is None
            or row["tier"] != "gold"
            or not row["label_source"].strip()
            or row["split"] not in {"train", "val"}
            or domain in exposed
            or "test" in splits[domain]
            or {"train", "val"} <= splits[domain]
            or len(labels[canonical]) != 1
            or host != row["domain"].lower().rstrip(".")
        ):
            continue
        selected = {
            "record_id": _hash("advisor-review-20261012:id:", canonical),
            "raw_url": row["url_norm"],
            "canonical_url": canonical,
            "registrable_domain": domain,
            "is_phishing": int(row["label"] == "phishing"),
            "population": population,
            "source": row["source"],
            "tier": row["tier"],
            "publisher_split": row["split"],
            "collected_at": row["collected_at"],
            "label_source": row["label_source"],
            "source_locator": {
                "publisher_ordinal": ordinal,
                "source_member": original["source_member"],
                "file_position": original["file_position"],
            },
        }
        candidates[domain].append(selected)
    expected = []
    for domain, rows in candidates.items():
        if len({row["is_phishing"] for row in rows}) != 1:
            raise ValueError("selection_cross_label_domain")
        expected.append(
            sorted(
                rows,
                key=lambda row: (
                    _hash("advisor-review-20261012:row:", row["canonical_url"]),
                    row["canonical_url"],
                    row["source_locator"]["publisher_ordinal"],
                ),
            )[0]
        )
    expected.sort(
        key=lambda row: (
            _hash("advisor-review-20261012:order:", row["registrable_domain"]),
            row["registrable_domain"],
        )
    )
    if expected != observed:
        raise ValueError("independent_selection_disagrees")
