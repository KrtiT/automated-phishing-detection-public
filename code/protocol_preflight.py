import argparse
from dataclasses import dataclass
import hashlib
import ipaddress
import json
from pathlib import Path
import sys
from urllib.parse import urlsplit


class PreflightError(ValueError):
    pass


REQUIRED_RECORD_FIELDS = frozenset({"record_id", "url", "label", "split"})
VALID_SPLITS = ("train", "validation", "test")
ASCII_LABEL_CHARACTERS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789-")


@dataclass(frozen=True)
class SuffixRules:
    exact: frozenset[str]
    wildcard: frozenset[str]
    exception: frozenset[str]


def _validate_ascii_label(label: str) -> None:
    if not 1 <= len(label) <= 63:
        raise PreflightError("hostname label length is invalid")
    if any(character not in ASCII_LABEL_CHARACTERS for character in label):
        raise PreflightError("hostname label contains an invalid character")
    if label.startswith("-") or label.endswith("-"):
        raise PreflightError("hostname label starts or ends with a hyphen")
    if label.startswith("xn--"):
        try:
            decoded = label.encode("ascii").decode("idna")
            round_trip = decoded.encode("idna").decode("ascii").lower()
        except UnicodeError as exc:
            raise PreflightError("hostname A-label is not valid IDNA") from exc
        if round_trip != label:
            raise PreflightError("hostname A-label does not round-trip")


def _ascii_domain(hostname: str) -> str:
    if not isinstance(hostname, str) or not hostname:
        raise PreflightError("hostname is missing")
    if any(character.isspace() for character in hostname):
        raise PreflightError("hostname contains whitespace")
    if "%" in hostname:
        raise PreflightError("hostname contains percent encoding")

    try:
        domain = hostname.lower().encode("idna").decode("ascii").lower()
    except UnicodeError as exc:
        raise PreflightError("hostname is not valid IDNA") from exc
    if domain.endswith("."):
        domain = domain[:-1]
    if len(domain) > 253:
        raise PreflightError("hostname exceeds 253 characters")
    labels = domain.split(".")
    if not domain or any(not label for label in labels):
        raise PreflightError("hostname contains an empty label")
    for label in labels:
        _validate_ascii_label(label)
    return domain


def _reject_ip_literal(domain: str) -> None:
    try:
        ipaddress.ip_address(domain)
    except ValueError:
        return
    raise PreflightError("IP-literal hostnames are not supported")


def normalize_hostname(url: str) -> str:
    if not isinstance(url, str) or not url:
        raise PreflightError("URL must be a nonempty string")
    if any(character in "\t\r\n" for character in url):
        raise PreflightError("URL contains a prohibited control character")
    try:
        parsed = urlsplit(url)
        hostname = parsed.hostname
        _ = parsed.port
    except (TypeError, ValueError) as exc:
        raise PreflightError("URL is malformed") from exc
    if not parsed.scheme or not parsed.netloc:
        raise PreflightError("URL must be absolute")
    authority = parsed.netloc.rsplit("@", 1)[-1]
    if authority.startswith("["):
        raise PreflightError("bracketed hostnames are not supported")
    if hostname is None:
        raise PreflightError("URL hostname is missing")

    domain = _ascii_domain(hostname)
    _reject_ip_literal(domain)
    return domain


def parse_suffix_rules(text: str) -> SuffixRules:
    if not isinstance(text, str):
        raise PreflightError("suffix rules must be text")
    exact: set[str] = set()
    wildcard: set[str] = set()
    exception: set[str] = set()

    for raw_line in text.splitlines():
        rule = raw_line.split("//", 1)[0].strip()
        if not rule:
            continue
        target = exact
        if rule.startswith("!"):
            target, rule = exception, rule[1:]
        elif rule.startswith("*."):
            target, rule = wildcard, rule[2:]
        if not rule or "*" in rule or "!" in rule:
            raise PreflightError("suffix rule is malformed")
        target.add(_ascii_domain(rule))

    if not (exact or wildcard or exception):
        raise PreflightError("suffix rule file contains no rules")
    return SuffixRules(frozenset(exact), frozenset(wildcard), frozenset(exception))


def _rule_matches(domain: str, rule: str) -> bool:
    return domain == rule or domain.endswith(f".{rule}")


def _longest_rule_length(domain: str, rules: frozenset[str]) -> int:
    lengths = (
        rule.count(".") + 1 for rule in rules if _rule_matches(domain, rule)
    )
    return max(lengths, default=0)


def _public_suffix_length(labels: list[str], rules: SuffixRules) -> int:
    domain = ".".join(labels)
    exception_length = _longest_rule_length(domain, rules.exception)
    if exception_length:
        return exception_length - 1

    exact_length = _longest_rule_length(domain, rules.exact)
    wildcard_length = 0
    for rule in rules.wildcard:
        base_length = rule.count(".") + 1
        if len(labels) > base_length and _rule_matches(domain, rule):
            wildcard_length = max(wildcard_length, base_length + 1)
    return max(1, exact_length, wildcard_length)


def registrable_domain(hostname: str, rules: SuffixRules) -> str:
    domain = _ascii_domain(hostname)
    _reject_ip_literal(domain)
    labels = domain.split(".")
    suffix_length = _public_suffix_length(labels, rules)
    if len(labels) <= suffix_length:
        raise PreflightError("hostname has no registrable domain")
    return ".".join(labels[-(suffix_length + 1) :])


def registrable_domain_for_url(url: str, rules: SuffixRules) -> str:
    return registrable_domain(normalize_hostname(url), rules)


def _validated_record(record: object) -> tuple[str, str, int, str]:
    if not isinstance(record, dict) or set(record) != REQUIRED_RECORD_FIELDS:
        raise PreflightError("record fields must be exactly record_id, url, label, split")
    record_id, url = record["record_id"], record["url"]
    label, split = record["label"], record["split"]
    if not isinstance(record_id, str) or not record_id:
        raise PreflightError("record_id must be a nonempty string")
    if not isinstance(url, str) or not url:
        raise PreflightError("url must be a nonempty string")
    if type(label) is not int or label not in (0, 1):
        raise PreflightError("label must be integer 0 or 1")
    if split not in VALID_SPLITS:
        raise PreflightError("split must be train, validation, or test")
    return record_id, url, label, split


def _check_record_history(record_id, url, label, record_ids, url_labels):
    if record_id in record_ids:
        raise PreflightError("duplicate record_id")
    record_ids.add(record_id)
    if url in url_labels:
        if url_labels[url] != label:
            raise PreflightError("label conflict for url")
        raise PreflightError("duplicate url")
    url_labels[url] = label


def _summary(record_count, domain_splits, split_records, split_domains):
    return {
        "status": "valid",
        "record_count": record_count,
        "registrable_domain_count": len(domain_splits),
        "splits": {
            split: {
                "record_count": split_records[split],
                "registrable_domain_count": len(split_domains[split]),
            }
            for split in VALID_SPLITS
        },
    }


def validate_manifest(manifest: object, rules: SuffixRules) -> dict:
    if not isinstance(manifest, dict):
        raise PreflightError("manifest must be an object")
    if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        raise PreflightError("schema_version must be 1")
    records = manifest.get("records")
    if not isinstance(records, list):
        raise PreflightError("records must be a list")

    record_ids, url_labels, domain_splits = set(), {}, {}
    split_records = {split: 0 for split in VALID_SPLITS}
    split_domains = {split: set() for split in VALID_SPLITS}
    for record in records:
        record_id, url, label, split = _validated_record(record)
        _check_record_history(record_id, url, label, record_ids, url_labels)
        domain = registrable_domain_for_url(url, rules)
        if domain in domain_splits and domain_splits[domain] != split:
            raise PreflightError(f"domain {domain} appears in multiple splits")
        domain_splits[domain] = split
        split_records[split] += 1
        split_domains[split].add(domain)
    return _summary(len(records), domain_splits, split_records, split_domains)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Validate a synthetic split manifest.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--suffix-rules", required=True, type=Path)
    args = parser.parse_args(argv)

    try:
        manifest_bytes = args.manifest.read_bytes()
        suffix_rules_bytes = args.suffix_rules.read_bytes()
        manifest = json.loads(manifest_bytes.decode("utf-8"))
        rules = parse_suffix_rules(suffix_rules_bytes.decode("utf-8"))
        summary = validate_manifest(manifest, rules)
    except (OSError, UnicodeError, json.JSONDecodeError, PreflightError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    summary["manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
    summary["suffix_rules_sha256"] = hashlib.sha256(suffix_rules_bytes).hexdigest()
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
