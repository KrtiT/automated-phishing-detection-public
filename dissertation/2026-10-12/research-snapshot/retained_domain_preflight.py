"""Aggregate retained metadata domain feasibility; never fit, score or admit."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from retained_metadata_preflight import FIELDS, SOURCE_SHA256

from automated_phishing_detection.phiusiil import PreparationError, canonicalize_url
from automated_phishing_detection.protocol_preflight import (
    PreflightError,
    normalize_hostname,
    parse_suffix_rules,
    registrable_domain,
)

PINS = {
    "source": SOURCE_SHA256,
    "suffix": "65365c4c9a4a6f746d53aadc758ab6b08aa10bb1379fea8ac353e381bca4b62e",
    "overlap": "10f824f87e4b451d62d77350872c94dff48b909dce9526c77703fd04149164d4",
    "followup": "731c231a1c39147583c6472d2b7242d8962eabde6ec1b05a5c5c9574ef17f96c",
}


def _records(document, rules):
    records = []
    required = (*FIELDS, "domain", "url_norm", "collected_at", "label_source")
    for row in document["rows"]:
        headers = document["headers"][row["source_member"]]
        if len(headers) != len(row["cells"]) or len(set(headers)) != len(headers):
            raise ValueError("publisher header and cell alignment is invalid")
        selected = {name: row["cells"][headers.index(name)] for name in required}
        if selected["split"] not in {"train", "val", "test"} or selected[
            "label"
        ] not in {"phishing", "benign"}:
            raise ValueError("unrecognized split or label")
        selected["canonical"] = selected["registered"] = None
        selected["hostname_matches"] = False
        try:
            canonical = canonicalize_url(selected["url_norm"])
            hostname = normalize_hostname(canonical)
            registered = registrable_domain(hostname, rules)
            selected.update(
                canonical=canonical,
                registered=registered,
                hostname_matches=hostname == selected["domain"].lower().rstrip("."),
            )
        except (PreparationError, PreflightError):
            pass
        records.append(selected)
    return records


def domain_summary(
    document, rules, original_domains, followup_domains, *, historical_domains=None
):
    records = _records(document, rules)
    split_domains = defaultdict(set)
    canonical_labels = defaultdict(set)
    groups = defaultdict(list)
    for row in records:
        groups[tuple(row[name] for name in FIELDS)].append(row)
        if row["registered"]:
            split_domains[row["split"]].add(row["registered"])
            canonical_labels[row["canonical"]].add(row["label"])
    exclusions = {
        "original_domain_overlap": original_domains,
        "followup_domain_overlap": followup_domains,
        "publisher_test_domain": split_domains["test"],
        "train_validation_domain_overlap": split_domains["train"]
        & split_domains["val"],
    }
    if historical_domains is not None:
        exclusions["historical_domain_overlap"] = historical_domains
    result, primary = [], defaultdict(list)
    for key, rows in sorted(groups.items()):
        valid = [row for row in rows if row["registered"]]
        counts = Counter(row["registered"] for row in valid)
        conflicts = [
            row for row in valid if len(canonical_labels[row["canonical"]]) > 1
        ]
        candidates = [
            row
            for row in valid
            if (
                row["split"] in {"train", "val"}
                and row["hostname_matches"]
                and len(canonical_labels[row["canonical"]]) == 1
                and not any(
                    row["registered"] in domains for domains in exclusions.values()
                )
            )
        ]
        for row in candidates:
            if (
                row["tier"] == "gold"
                and row["source"] == "tinnhiemmang"
                and row["label"] == "phishing"
            ):
                primary["gold_positive"].append(row)
            elif (
                row["tier"] == "gold"
                and row["source"] in {"tinnhiem_org", "tinnhiem_web"}
                and row["label"] == "benign"
            ):
                primary["certified_negative"].append(row)
        result.append(
            dict(zip(FIELDS, key, strict=True))
            | {
                "rows": len(rows),
                "valid_rows": len(valid),
                "invalid_url_rows": len(rows) - len(valid),
                "registered_domains": len(counts),
                "largest_domain_rows": max(counts.values(), default=0),
                "largest_ten_domain_sizes": sorted(counts.values(), reverse=True)[:10],
                "canonical_duplicate_excess": len(valid)
                - len({row["canonical"] for row in valid}),
                "conflicting_canonical_label_rows": len(conflicts),
                "publisher_hostname_mismatch_rows": sum(
                    not row["hostname_matches"] for row in valid
                ),
                "missing_collection_time_rows": sum(
                    not row["collected_at"].strip() for row in rows
                ),
                "missing_label_source_rows": sum(
                    not row["label_source"].strip() for row in rows
                ),
                "exclusion_incidences": {
                    name: sum(row["registered"] in domains for row in valid)
                    for name, domains in exclusions.items()
                },
                "candidate_upper_bound_rows": len(candidates),
                "candidate_upper_bound_unique_urls": len(
                    {row["canonical"] for row in candidates}
                ),
                "candidate_upper_bound_domains": len(
                    {row["registered"] for row in candidates}
                ),
            }
        )
    return {
        "groups": result,
        "primary_candidate_upper_bounds": {
            population: {
                "rows": len(rows),
                "unique_urls": len({row["canonical"] for row in rows}),
                "registered_domains": len({row["registered"] for row in rows}),
                "missing_collection_time_rows": sum(
                    not row["collected_at"].strip() for row in rows
                ),
                "distinct_collection_dates": len(
                    {row["collected_at"] for row in rows if row["collected_at"].strip()}
                ),
            }
            for population, rows in sorted(primary.items())
        },
        "primary_candidate_cross_label_domains": len(
            {row["registered"] for row in primary["gold_positive"]}
            & {row["registered"] for row in primary["certified_negative"]}
        ),
        "publisher_train_validation_domain_overlap": len(
            exclusions["train_validation_domain_overlap"]
        ),
        "status": "aggregate_upper_bound_only_not_independent_population_admission",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (*PINS, "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    contents = {}
    for name, expected in PINS.items():
        content = getattr(args, name).read_bytes()
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError(f"{name} input hash mismatch")
        contents[name] = content
    rules = parse_suffix_rules(contents["suffix"].decode())
    original = set(json.loads(contents["overlap"])["domains"])
    followup = {
        json.loads(line)["registrable_domain"]
        for line in contents["followup"].splitlines()
    }
    if any(
        registrable_domain(domain, rules) != domain for domain in original | followup
    ):
        raise ValueError("exposure inventory contains non-registered domains")
    report = domain_summary(json.loads(contents["source"]), rules, original, followup)
    report.update(
        input_sha256=PINS,
        original_blocked_domains=len(original),
        followup_blocked_domains=len(followup),
        privacy="Aggregate counts only; no URLs, domain strings or record identifiers exported.",
        limitations="Distinct registered domains do not establish independent campaigns, label accuracy or absence of other historical exposure.",
    )
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "groups": len(report["groups"])}))


if __name__ == "__main__":
    main()
