from dataclasses import dataclass
import ipaddress
from urllib.parse import urlsplit


class PreflightError(ValueError):
    pass


@dataclass(frozen=True)
class SuffixRules:
    exact: frozenset[str]
    wildcard: frozenset[str]
    exception: frozenset[str]


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
    if not domain or any(not label for label in domain.split(".")):
        raise PreflightError("hostname contains an empty label")
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
