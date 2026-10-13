"""Local-history identity audit, without model outputs or execution authority."""

import argparse
import hashlib
import json
from pathlib import Path

from retained_domain_preflight import PINS, domain_summary

from automated_phishing_detection.phiusiil import canonicalize_url
from automated_phishing_detection.protocol_preflight import (
    normalize_hostname,
    parse_suffix_rules,
    registrable_domain,
)


def identity(row, rules):
    record = row.get("record", row)
    identifier = record.get("record_id")
    if type(identifier) is not str or not identifier.strip():
        raise ValueError("missing historical record identity")
    domain = record.get("registrable_domain")
    if domain is not None:
        if type(domain) is not str or registrable_domain(domain, rules) != domain:
            raise ValueError("invalid historical registered domain")
    if "raw_url" in record:
        derived = registrable_domain(
            normalize_hostname(canonicalize_url(record["raw_url"])), rules
        )
        if domain is not None and derived != domain:
            raise ValueError("historical URL and domain disagree")
        domain = derived
    return identifier, domain


def scan(path, context, rules, known):
    domains = set()
    digest = hashlib.sha256()
    rows = joined = unresolved = 0
    with path.open("rb") as stream:
        for line in stream:
            digest.update(line)
            rows += 1
            identifier, domain = identity(json.loads(line), rules)
            reference = known.get(identifier)
            if domain is not None and reference is not None and domain != reference:
                raise ValueError("historical identity domain disagreement")
            if domain is None:
                domain = reference
                joined += domain is not None
            if domain is None:
                unresolved += 1
            else:
                domains.add(domain)
    return {
        "path": str(path.relative_to(context)),
        "sha256": digest.hexdigest(),
        "rows": rows,
        "registered_domains": len(domains),
        "identity_joined_rows": joined,
        "unresolved_rows": unresolved,
    }, domains


def discover(context):
    patterns = (
        "gwu_working/study-*-attempt-*/**/retained*.jsonl",
        "gwu_working/study-*-attempt-*/**/predictions.jsonl",
        "gwu_working/secondary-*-attempt-*/**/*predictions.jsonl",
        "gwu_working/automated-phishing-detection-public/data/processed/phiusiil*/train.jsonl",
        "gwu_working/automated-phishing-detection-public/data/processed/phiusiil*/validation.jsonl",
        "gwu_working/automated-phishing-detection-public/data/processed/phiusiil*/test.jsonl",
        "gwu_working/automated-phishing-detection-public/data/processed/phiusiil*/group_test.jsonl",
        "dissertation/followup-20261001/population-admission-*/retained.jsonl",
        "dissertation/followup-20261001/detection-evaluate-*/predictions.jsonl",
    )
    paths = sorted({path for pattern in patterns for path in context.glob(pattern)})
    if not paths:
        raise ValueError("historical exposure inventory is empty")
    return paths


def build_report(context):
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
    contents = {name: path.read_bytes() for name, path in sources.items()}
    if any(
        hashlib.sha256(content).hexdigest() != PINS[name]
        for name, content in contents.items()
    ):
        raise ValueError("pinned metadata source differs")
    rules = parse_suffix_rules(contents["suffix"].decode())
    original = set(json.loads(contents["overlap"])["domains"])
    followup = {
        json.loads(line)["registrable_domain"]
        for line in contents["followup"].splitlines()
    }
    paths = discover(context)
    known = {}
    for path in paths:
        if path.name.endswith("predictions.jsonl"):
            continue
        with path.open() as stream:
            for line in stream:
                identifier, domain = identity(json.loads(line), rules)
                if domain is None:
                    raise ValueError("identity reference lacks registered domain")
                if identifier in known and known[identifier] != domain:
                    raise ValueError("identity reference domain disagreement")
                known[identifier] = domain
    inventory, exposed = [], set()
    for path in paths:
        entry, domains = scan(path, context, rules, known)
        inventory.append(entry)
        exposed.update(domains)
    report = domain_summary(
        json.loads(contents["source"]),
        rules,
        original,
        followup,
        historical_domains=exposed,
    )
    unresolved = sum(entry["unresolved_rows"] for entry in inventory)
    report.update(
        input_sha256=PINS,
        inventory=inventory,
        history_files=len(inventory),
        distinct_history_contents=len({entry["sha256"] for entry in inventory}),
        historical_registered_domains=len(exposed),
        historical_domains_outside_prior_block=len(exposed - original - followup),
        unresolved_identity_rows=unresolved,
        local_inventory_identity_coverage="complete_for_discovered_roster"
        if unresolved == 0
        else "incomplete",
        privacy="Aggregates and file identities only; no URLs, domains or record IDs exported.",
        status="metadata_upper_bound_not_admission",
        limitations="Discovered local history only; not proof of no use elsewhere. Distinct domains do not certify campaign independence or labels. No model outcomes read or new predictions generated.",
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.context)
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                name: report[name]
                for name in (
                    "history_files",
                    "distinct_history_contents",
                    "historical_registered_domains",
                    "unresolved_identity_rows",
                    "primary_candidate_upper_bounds",
                    "status",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
