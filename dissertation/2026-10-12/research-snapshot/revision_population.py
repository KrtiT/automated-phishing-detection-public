"""Prospective identity-only selection for the bounded post-review cohort."""

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import exposure_audit
from retained_domain_preflight import PINS, _records

from automated_phishing_detection.protocol_preflight import parse_suffix_rules

EXPECTED_COUNTS = {"gold_positive": 1392, "certified_negative": 1262}


def _digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _bytes(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()


def _population(row):
    if row["tier"] != "gold" or not row["label_source"].strip():
        return None
    if row["source"] == "tinnhiemmang" and row["label"] == "phishing":
        return "gold_positive"
    if row["source"] in {"tinnhiem_org", "tinnhiem_web"} and row["label"] == "benign":
        return "certified_negative"
    return None


def _summary(rows):
    strata = {}
    for population in EXPECTED_COUNTS:
        selected = [row for row in rows if row["population"] == population]
        dates = Counter(
            row["collected_at"] for row in selected if row["collected_at"].strip()
        )
        strata[population] = {
            "rows": len(selected),
            "domains": len(selected),
            "missing_date_rows": sum(
                not row["collected_at"].strip() for row in selected
            ),
            "distinct_dates": len(dates),
            "largest_date_group": max(dates.values(), default=0),
            "source_counts": dict(
                sorted(Counter(row["source"] for row in selected).items())
            ),
        }
    return {
        "status": "identity_verified_not_execution_admission",
        "counts": {name: values["rows"] for name, values in strata.items()},
        "strata": strata,
        "model_predictions": 0,
        "execution_authorized": False,
        "estimand": "domain_balanced_retained_source_frame",
        "independence": "distinct_domains_not_verified_independent_campaigns",
    }


def select_cohort(
    document, rules, excluded_domains, *, expected_counts=EXPECTED_COUNTS
):
    records = _records(document, rules)
    split_domains, canonical_labels = defaultdict(set), defaultdict(set)
    for row in records:
        if row["registered"]:
            split_domains[row["split"]].add(row["registered"])
            canonical_labels[row["canonical"]].add(row["label"])
    excluded = (
        excluded_domains
        | split_domains["test"]
        | (split_domains["train"] & split_domains["val"])
    )
    eligible = defaultdict(list)
    for ordinal, row in enumerate(records):
        population = _population(row)
        if (
            population is None
            or not row["registered"]
            or not row["hostname_matches"]
            or row["split"] not in {"train", "val"}
            or row["registered"] in excluded
            or len(canonical_labels[row["canonical"]]) != 1
        ):
            continue
        eligible[row["registered"]].append((row, ordinal, population))
    result = []
    for domain, candidates in eligible.items():
        if len({row["label"] for row, unused, population in candidates}) != 1:
            raise ValueError("cross_label_domain")
        row, ordinal, population = min(
            candidates,
            key=lambda entry: (
                _digest("advisor-review-20261012:row:" + entry[0]["canonical"]),
                entry[0]["canonical"],
                entry[1],
            ),
        )
        original = document["rows"][ordinal]
        if type(original["file_position"]) is not int or original["file_position"] < 0:
            raise ValueError("invalid_publisher_locator")
        result.append(
            {
                "record_id": _digest("advisor-review-20261012:id:" + row["canonical"]),
                "raw_url": row["url_norm"],
                "canonical_url": row["canonical"],
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
        )
    result.sort(
        key=lambda row: (
            _digest("advisor-review-20261012:order:" + row["registrable_domain"]),
            row["registrable_domain"],
        )
    )
    summary = _summary(result)
    if _bytes(summary["counts"]) != _bytes(expected_counts):
        raise ValueError("cohort_count_mismatch")
    if len({row["record_id"] for row in result}) != len(result):
        raise ValueError("duplicate_cohort_identity")
    return result, summary


def select_workload(rows, *, positives=10, negatives=990):
    remaining = {1: positives, 0: negatives}
    selected = []
    for row in rows:
        label = row["is_phishing"]
        if type(label) is not int or label not in (0, 1):
            raise ValueError("invalid_workload_label")
        if remaining[label] > 0:
            selected.append(row)
            remaining[label] -= 1
    if any(remaining.values()):
        raise ValueError("workload_shortage")
    return selected


def verify_roster(baseline, current):
    fields = ("inventory", "input_sha256", "unresolved_identity_rows")
    if (
        type(current["unresolved_identity_rows"]) is not int
        or current["unresolved_identity_rows"] != 0
        or _bytes({name: baseline[name] for name in fields})
        != _bytes({name: current[name] for name in fields})
    ):
        raise ValueError("exposure_roster_changed_or_unresolved")


def checked_history(context, baseline, rules):
    paths = exposure_audit.discover(context)
    if [str(path.relative_to(context)) for path in paths] != [
        entry["path"] for entry in baseline["inventory"]
    ]:
        raise ValueError("exposure_roster_members_changed")
    known = {}
    for path in paths:
        if path.name.endswith("predictions.jsonl"):
            continue
        with path.open() as stream:
            for line in stream:
                identifier, domain = exposure_audit.identity(json.loads(line), rules)
                if domain is None or (
                    identifier in known and known[identifier] != domain
                ):
                    raise ValueError("exposure_roster_identity_mismatch")
                known[identifier] = domain
    inventory, exposed = [], set()
    for path in paths:
        entry, domains = exposure_audit.scan(path, context, rules, known)
        inventory.append(entry)
        exposed.update(domains)
    verify_roster(
        baseline,
        {
            "inventory": inventory,
            "input_sha256": PINS,
            "unresolved_identity_rows": sum(
                entry["unresolved_rows"] for entry in inventory
            ),
        },
    )
    return exposed


def _write(path, content):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def prepare(context, output):
    revision = Path(__file__).resolve().parent
    preparation = (
        context / "gwu_working/study-urlnorm-v1-2026-09-30-attempt-4/preparation"
    )
    sources = {
        "source": preparation / "publisher-source.json",
        "suffix": preparation / "suffix-rules.dat",
        "overlap": preparation / "source-overlap.json",
        "followup": context
        / "dissertation/followup-20261001/population-admission-v1-attempt-2/retained.jsonl",
    }
    content = {name: path.read_bytes() for name, path in sources.items()}
    if any(
        hashlib.sha256(value).hexdigest() != PINS[name]
        for name, value in content.items()
    ):
        raise ValueError("metadata_input_hash_mismatch")
    roster_bytes = (revision / "retained-exposure-audit-v2.json").read_bytes()
    roster = json.loads(roster_bytes)
    rules = parse_suffix_rules(content["suffix"].decode())
    excluded = checked_history(context, roster, rules)
    excluded.update(json.loads(content["overlap"])["domains"])
    excluded.update(
        json.loads(line)["registrable_domain"]
        for line in content["followup"].splitlines()
    )
    rows, summary = select_cohort(json.loads(content["source"]), rules, excluded)
    workload = select_workload(rows)
    products = {
        "cohort.jsonl": b"".join(_bytes(row) for row in rows),
        "workload.jsonl": b"".join(_bytes(row) for row in workload),
    }
    summary.update(
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_sha256": PINS,
            "exposure_roster_sha256": hashlib.sha256(roster_bytes).hexdigest(),
            "specification_sha256": hashlib.sha256(
                (revision / "EXPERIMENT_SPEC.txt").read_bytes()
            ).hexdigest(),
            "selector_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "output_sha256": {
                name: hashlib.sha256(value).hexdigest()
                for name, value in products.items()
            },
            "workload_counts": {"positive": 10, "negative": 990},
            "history_files": len(roster["inventory"]),
        }
    )
    output.mkdir(mode=0o700)
    try:
        for name, value in products.items():
            _write(output / name, value)
        _write(output / "summary.json", _bytes(summary))
    except BaseException as error:
        _write(
            output / "preparation-failure.json",
            _bytes({"error_type": type(error).__name__}),
        )
        raise
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(args.context.resolve(), args.output.resolve()),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
