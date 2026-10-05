"""Deterministic admission of the single specified follow-up benchmark."""

import csv
import io
from collections import Counter, defaultdict
from hashlib import sha256

from .phiusiil import PreparationError, canonicalize_url
from .protocol_preflight import PreflightError, normalize_hostname, registrable_domain

LABELS = {"legitimate": 0, "phishing": 1}


def prepare_population(content, suffix_rules, *, blocked_domains):
    """Retain all eligible URL/label pairs without fitting or scoring anything."""
    reader = csv.DictReader(io.StringIO(content.decode("utf-8"), newline=""))
    fields = reader.fieldnames
    if (
        fields is None
        or len(fields) != len(set(fields))
        or not {"url", "status"}.issubset(fields)
    ):
        raise ValueError("unrecognized or duplicate publisher header")
    rows = []
    canonical_labels = defaultdict(set)
    source_hash = sha256(content).hexdigest()
    overlap_counts = {name: 0 for name in blocked_domains}
    for ordinal, source in enumerate(reader, start=1):
        if None in source or any(value is None for value in source.values()):
            raise ValueError("publisher row width differs from header")
        if source["status"] not in LABELS:
            raise ValueError("unrecognized publisher label")
        row = {
            "record_id": f"{source_hash}:{ordinal:08d}",
            "source_ordinal": ordinal,
            "raw_url": source["url"],
            "publisher_label": source["status"],
            "is_phishing": LABELS[source["status"]],
            "canonical_url_sha256": None,
            "registrable_domain": None,
            "reasons": [],
        }
        try:
            canonical = canonicalize_url(row["raw_url"])
            domain = registrable_domain(normalize_hostname(canonical), suffix_rules)
        except (PreparationError, PreflightError):
            row["reasons"].append("invalid_url")
        else:
            canonical_hash = sha256(canonical.encode("utf-8")).hexdigest()
            row["canonical_url_sha256"] = canonical_hash
            row["registrable_domain"] = domain
            canonical_labels[canonical_hash].add(row["is_phishing"])
            for name, domains in blocked_domains.items():
                if domain in domains:
                    row["reasons"].append(f"{name}_domain_overlap")
                    overlap_counts[name] += 1
        rows.append(row)
    seen = set()
    retained, quarantine = [], []
    for row in rows:
        canonical_hash = row["canonical_url_sha256"]
        if canonical_hash is not None:
            if len(canonical_labels[canonical_hash]) > 1:
                row["reasons"].append("conflicting_canonical_labels")
            elif canonical_hash in seen:
                row["reasons"].append("same_label_canonical_duplicate")
            seen.add(canonical_hash)
        if row["reasons"]:
            quarantine.append(row)
        else:
            retained.append(
                {key: value for key, value in row.items() if key != "reasons"}
            )
    counts = {
        str(label): sum(row["is_phishing"] == label for row in retained)
        for label in (0, 1)
    }
    domain_counts = {
        str(label): len(
            {
                row["registrable_domain"]
                for row in retained
                if row["is_phishing"] == label
            }
        )
        for label in (0, 1)
    }
    return {
        "retained": retained,
        "quarantine": quarantine,
        "summary": {
            "protocol": "bounded-followup-population-v1",
            "input_sha256": source_hash,
            "input_rows": len(rows),
            "input_class_counts": dict(
                Counter(str(row["is_phishing"]) for row in rows)
            ),
            "retained_rows": len(retained),
            "quarantined_rows": len(quarantine),
            "quarantine_reason_incidences": dict(
                Counter(reason for row in quarantine for reason in row["reasons"])
            ),
            "overlap_before_filtering": overlap_counts,
            "retained_class_counts": counts,
            "retained_class_domain_counts": domain_counts,
            "admitted": all(
                counts[key] >= 1000 and domain_counts[key] >= 250 for key in counts
            ),
            "model_predictions_performed": False,
        },
    }
