"""Prediction-blind necessary population capacity, not scientific estimability.

Caller consistency does not authenticate source completeness or authorize access.
Shortage categories describe task dependencies, not inferential population roles.
The coordinator separately retains preparation identities and administrative holds.
"""

from collections import Counter

from ._checkpoint_codec import canonical_bytes
from ._external_inputs import validate_prepared_external
from .evaluation_manifest import ManifestRecord, _validated_candidates
from .phishvn import PreparedExternal


class StudyFeasibilityError(ValueError):
    """Preparation is invalid; no population shortage can be inferred."""


def _internal_counts(records):
    if type(records) is not tuple:
        raise StudyFeasibilityError("invalid_preparation_feasibility")
    rows = _validated_candidates(records)
    positive = tuple(row for row in rows if row.is_phishing == 1)
    return {
        "retained_rows": len(rows),
        "positive_rows": len(positive),
        "negative_rows": len(rows) - len(positive),
        "positive_domains": len({row.registrable_domain for row in positive}),
    }


def _secondary_support(rows, source, tier):
    selected = tuple(
        row
        for row in rows
        if (row.source_group, row.confidence_tier, row.source_class, row.role)
        == (source, tier, "phishing", "secondary")
    )
    return {
        "rows": len(selected),
        "domains": len({row.registrable_domain for row in selected}),
    }


def _external_counts(prepared):
    rows = validate_prepared_external(prepared)
    roles = Counter(row.role for row in rows)
    return {
        "input_test_rows": prepared.public_summary["input_test_rows"],
        "quarantined_test_rows": prepared.public_summary["quarantined_test_rows"],
        "retained_test_rows": len(rows),
        "gold_rows": roles["gold"],
        "gold_positive_domains": len(
            {row.registrable_domain for row in rows if row.role == "gold"}
        ),
        "certified_rows": roles["certified"],
        "tranco_rows": roles["tranco"],
        "secondary_rows": roles["secondary"],
        "complete_windows": max(0, 1 + (len(rows) - 256) // 64),
        "secondary_positive_strata": {
            "ncsc_silver": _secondary_support(rows, "ncsc", "silver"),
            "chongluadao_openphish_bronze": _secondary_support(
                rows, "chongluadao_openphish", "bronze"
            ),
        },
    }


def _requirements(internal, external):
    requirements = [
        ("internal_negative_rows", "primary", 1, internal["negative_rows"]),
        ("internal_positive_domains", "primary", 2, internal["positive_domains"]),
        ("certified_rows", "primary", 1, external["certified_rows"]),
        ("gold_positive_domains", "primary", 2, external["gold_positive_domains"]),
        ("tranco_rows", "primary", 1, external["tranco_rows"]),
        ("external_complete_windows", "primary", 1, external["complete_windows"]),
    ]
    for prevalence in (100, 10, 500):
        category = "primary" if prevalence == 100 else "sensitivity"
        for label, required in (
            ("negative", 10000 - prevalence),
            ("positive", prevalence),
        ):
            requirements.append(
                (
                    f"http_{prevalence}_{label}_rows",
                    category,
                    required,
                    internal[f"{label}_rows"],
                )
            )
    requirements.append(
        ("shift_warmup_rows", "descriptive", 1000, external["retained_test_rows"])
    )
    return requirements


def _shortages(internal, external):
    return [
        {
            "requirement": name,
            "category": category,
            "required": required,
            "available": available,
        }
        for name, category, required, available in _requirements(internal, external)
        if available < required
    ]


def assess_preparation_feasibility(
    internal: tuple[ManifestRecord, ...], external: PreparedExternal
) -> bytes:
    """Project existing prerequisites without sampling, scoring, or gate decisions."""
    try:
        internal_counts = _internal_counts(internal)
        external_counts = _external_counts(external)
        return canonical_bytes(
            {
                "schema_version": 1,
                "protocol": "study-preparation-feasibility-v1",
                "scope": "necessary_population_capacity_only",
                "counts": {"internal": internal_counts, "external": external_counts},
                "shortages": _shortages(internal_counts, external_counts),
            }
        )
    except Exception:
        raise StudyFeasibilityError("invalid_preparation_feasibility") from None
