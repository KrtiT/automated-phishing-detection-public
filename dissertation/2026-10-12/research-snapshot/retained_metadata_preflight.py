"""Aggregate retained publisher metadata only; no URLs, scores or model execution."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

SOURCE_SHA256 = "97b3e71f6c3569ba9d98150efae0a986bcd19788d7f380db38a4b2a948589292"
FIELDS = ("split", "source", "tier", "label")


def summarize(document):
    groups = defaultdict(lambda: {"rows": 0, "domains": set(), "missing_domain": 0})
    for row in document["rows"]:
        headers = document["headers"][row["source_member"]]
        if len(headers) != len(row["cells"]):
            raise ValueError("publisher header and cell counts differ")
        selected = {
            name: row["cells"][headers.index(name)] for name in (*FIELDS, "domain")
        }
        group = groups[tuple(selected[name] for name in FIELDS)]
        group["rows"] += 1
        if selected["domain"]:
            group["domains"].add(selected["domain"])
        else:
            group["missing_domain"] += 1
    return [
        dict(zip(FIELDS, key, strict=True))
        | {
            "rows": value["rows"],
            "distinct_publisher_domain_strings": len(value["domains"]),
            "empty_publisher_domain_rows": value["missing_domain"],
        }
        for key, value in sorted(groups.items())
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    content = args.source.read_bytes()
    if hashlib.sha256(content).hexdigest() != SOURCE_SHA256:
        raise ValueError("retained source differs from publisher-summary pin")
    document = json.loads(content)
    if len(document["rows"]) != 53116:
        raise ValueError("retained publisher row count differs")
    report = {
        "source_sha256": SOURCE_SHA256,
        "status": "metadata_feasibility_only_not_population_admission",
        "groups": summarize(document),
        "privacy": "Aggregate counts only; no URLs, domains, identifiers or row-level labels exported.",
        "independence": "Publisher domain counts are not validated independent sampling units.",
    }
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
