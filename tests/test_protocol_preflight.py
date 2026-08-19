from pathlib import Path
import sys

import pytest


BASE_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = BASE_DIR / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.append(str(CODE_DIR))

from protocol_preflight import (  # noqa: E402
    PreflightError,
    normalize_hostname,
    parse_suffix_rules,
    registrable_domain_for_url,
)


SUFFIX_RULES = parse_suffix_rules(
    """
    // Synthetic rules only.
    com
    co.uk
    example
    *.ck
    !www.ck
    """
)


def test_normalizes_case_terminal_dot_and_port():
    url = "https://Login.Shop.Example.COM.:8443/path"

    assert registrable_domain_for_url(url, SUFFIX_RULES) == "example.com"


def test_uses_longest_multi_label_suffix():
    url = "https://a.department.example.co.uk"

    assert registrable_domain_for_url(url, SUFFIX_RULES) == "example.co.uk"


def test_converts_unicode_hostname_to_idna_ascii():
    url = "https://BÜCHER.example"

    assert registrable_domain_for_url(url, SUFFIX_RULES) == "xn--bcher-kva.example"


def test_normalizes_idna_dot_separator():
    assert normalize_hostname("https://shop.example.com\u3002/") == "shop.example.com"


def test_applies_wildcard_suffix_rule():
    url = "https://shop.foo.ck"

    assert registrable_domain_for_url(url, SUFFIX_RULES) == "shop.foo.ck"


def test_applies_exception_suffix_rule():
    url = "https://a.www.ck"

    assert registrable_domain_for_url(url, SUFFIX_RULES) == "www.ck"


def test_rejects_relative_url():
    with pytest.raises(PreflightError):
        registrable_domain_for_url("example.com/path", SUFFIX_RULES)


def test_rejects_ip_literal():
    with pytest.raises(PreflightError):
        registrable_domain_for_url("https://127.0.0.1/path", SUFFIX_RULES)


def test_rejects_percent_encoded_hostname():
    with pytest.raises(PreflightError):
        registrable_domain_for_url("https://exa%6dple.com/", SUFFIX_RULES)


def test_rejects_bracketed_ipvfuture_hostname():
    with pytest.raises(PreflightError):
        registrable_domain_for_url("https://[v1.foo]/", SUFFIX_RULES)


def test_rejects_bad_port():
    with pytest.raises(PreflightError):
        registrable_domain_for_url("https://example.com:not-a-port", SUFFIX_RULES)


def test_rejects_whitespace_in_hostname():
    with pytest.raises(PreflightError):
        registrable_domain_for_url("https://bad host.com/path", SUFFIX_RULES)


def test_rejects_missing_host():
    with pytest.raises(PreflightError):
        registrable_domain_for_url("https:///path", SUFFIX_RULES)


def test_rejects_comment_only_suffix_rules():
    with pytest.raises(PreflightError, match="suffix rule"):
        parse_suffix_rules("\n // first comment\n\n// second comment\n")
