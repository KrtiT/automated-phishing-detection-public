"""Admit the named follow-up benchmark once, without models or predictions."""

import argparse
import hashlib
import json
import os
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from automated_phishing_detection.followup_population import prepare_population
from automated_phishing_detection.phiusiil import PreparationError, canonicalize_url
from automated_phishing_detection.protocol_preflight import (
    PreflightError,
    normalize_hostname,
    parse_suffix_rules,
    registrable_domain,
)


def digest(content):
    return hashlib.sha256(content).hexdigest()


def write(path, content):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as output:
        output.write(content)


def json_bytes(value):
    return (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context-root", required=True, type=Path)
    parser.add_argument("--attempt-number", type=int, default=1)
    arguments = parser.parse_args()
    if arguments.attempt_number < 1:
        parser.error("attempt number must be positive")
    context = arguments.context_root.resolve(strict=True)
    followup = context / "dissertation/followup-20261001"
    suffix = (
        "" if arguments.attempt_number == 1 else f"-attempt-{arguments.attempt_number}"
    )
    attempt = followup / ("population-admission-v1" + suffix)
    attempt.mkdir(mode=0o700)
    specification = followup / "comparison-specification-v1.md"
    source_files = followup / "resources/hannousse-files.json"
    source = json.loads(source_files.read_bytes())
    if len(source) != 1 or source[0]["filename"] != "dataset_B_05_2020.csv":
        raise ValueError("publisher file inventory changed")
    expected = "21093e2902e5441c86a6daf95e86e7c332046e477fdf109a579d7bd81e586d6c"
    details = source[0]["content_details"]
    if details["sha256_hash"] != expected or details["size"] != 3661166:
        raise ValueError("publisher hash or size changed")
    write(
        attempt / "intent.json",
        json_bytes(
            {
                "started_at": datetime.now(timezone.utc).isoformat(),
                "specification_sha256": digest(specification.read_bytes()),
                "script_sha256": digest(Path(__file__).read_bytes()),
                "publisher_inventory_sha256": digest(source_files.read_bytes()),
                "expected_csv_sha256": expected,
                "scope": "data eligibility only; no model access, fitting or predictions",
            }
        ),
    )
    try:
        request = urllib.request.Request(
            details["download_url"],
            headers={"User-Agent": "Dissertation-method-review/1.0"},
        )
        with urllib.request.urlopen(request, timeout=90) as response:
            content = response.read(details["size"] + 1)
            retrieval = {
                "url": details["download_url"],
                "final_url": response.url,
                "status": response.status,
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
            }
        write(attempt / source[0]["filename"], content)
        if digest(content) != expected or len(content) != details["size"]:
            raise ValueError("download does not match pinned publisher bytes")
        preparation = (
            context / "gwu_working/study-urlnorm-v1-2026-09-30-attempt-4/preparation"
        )
        suffix_bytes = (preparation / "suffix-rules.dat").read_bytes()
        overlap_bytes = (preparation / "source-overlap.json").read_bytes()
        if (
            digest(suffix_bytes)
            != "65365c4c9a4a6f746d53aadc758ab6b08aa10bb1379fea8ac353e381bca4b62e"
        ):
            raise ValueError("suffix rules changed")
        if (
            digest(overlap_bytes)
            != "10f824f87e4b451d62d77350872c94dff48b909dce9526c77703fd04149164d4"
        ):
            raise ValueError("PhiUSIIL source-overlap manifest changed")
        suffix_rules = parse_suffix_rules(suffix_bytes.decode("utf-8"))
        internal = json.loads(overlap_bytes)
        publisher_bytes = (preparation / "publisher-source.json").read_bytes()
        if (
            digest(publisher_bytes)
            != "97b3e71f6c3569ba9d98150efae0a986bcd19788d7f380db38a4b2a948589292"
        ):
            raise ValueError("PhishVN publisher-source manifest changed")
        publisher = json.loads(publisher_bytes)
        if len(internal["rows"]) != 235795 or len(publisher["rows"]) != 53116:
            raise ValueError("incomplete original corpus manifest")
        external_domains = set()
        invalid_fields = 0
        for row in publisher["rows"]:
            fields = publisher["headers"][row["source_member"]]
            values = dict(zip(fields, row["cells"], strict=True))
            for field in ("url", "url_norm"):
                try:
                    canonical = canonicalize_url(values[field])
                    domain = registrable_domain(
                        normalize_hostname(canonical), suffix_rules
                    )
                except (PreparationError, PreflightError):
                    invalid_fields += 1
                else:
                    external_domains.add(domain)
        result = prepare_population(
            content,
            suffix_rules,
            blocked_domains={
                "phiusiil": set(internal["domains"]),
                "phishvn": external_domains,
            },
        )
        summary = result["summary"]
        if summary["input_rows"] != 11430 or summary["input_class_counts"] != {
            "0": 5715,
            "1": 5715,
        }:
            raise ValueError("publisher population does not match declared counts")
        for name in ("retained", "quarantine"):
            payload = b"".join(
                json_bytes(row).replace(b"\n", b"") + b"\n" for row in result[name]
            )
            write(attempt / f"{name}.jsonl", payload)
        summary.update(
            {
                "retrieval": retrieval,
                "specification_sha256": digest(specification.read_bytes()),
                "source_overlap_sha256": digest(overlap_bytes),
                "publisher_source_sha256": digest(publisher_bytes),
                "suffix_rules_sha256": digest(suffix_bytes),
                "phiusiil_blocked_domains": len(internal["domains"]),
                "phishvn_blocked_domains": len(external_domains),
                "phishvn_unparseable_url_fields": invalid_fields,
                "output_hashes": {
                    name: digest((attempt / name).read_bytes())
                    for name in ("retained.jsonl", "quarantine.jsonl")
                },
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        write(attempt / "summary.json", json_bytes(summary))
        print(json.dumps(summary, indent=2))
    except BaseException as error:
        write(
            attempt / "failure.json",
            json_bytes(
                {
                    "type": type(error).__name__,
                    "message": str(error),
                    "at": datetime.now(timezone.utc).isoformat(),
                }
            ),
        )
        raise


if __name__ == "__main__":
    main()
