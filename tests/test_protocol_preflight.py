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
    validate_manifest,
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


@pytest.mark.parametrize(
    "url",
    [
        pytest.param("https://bad_host.example/", id="underscore"),
        pytest.param("https://-bad.example/", id="leading-hyphen"),
        pytest.param("https://bad-.example/", id="trailing-hyphen"),
        pytest.param("https://xn--abc.example/", id="malformed-a-label"),
    ],
)
def test_rejects_invalid_ascii_hostname_labels(url):
    with pytest.raises(PreflightError):
        normalize_hostname(url)


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


def _record(record_id, url, label=0, split="train"):
    return dict(record_id=record_id, url=url, label=label, split=split)


def _manifest(*records):
    return {"schema_version": 1, "records": list(records)}


def _assert_rejected(manifest, message):
    with pytest.raises(PreflightError, match=message):
        validate_manifest(manifest, SUFFIX_RULES)


def test_validates_manifest_and_summarizes_every_split():
    manifest = _manifest(
        _record("one", "https://login.example.com", split="train"),
        _record("two", "https://www.example.com", 1, "train"),
        _record("three", "https://sample.co.uk", split="validation"),
        _record("four", "https://shop.foo.ck", 1, "test"),
    )

    assert validate_manifest(manifest, SUFFIX_RULES) == {
        "status": "valid",
        "record_count": 4,
        "registrable_domain_count": 3,
        "splits": {
            "train": {"record_count": 2, "registrable_domain_count": 1},
            "validation": {"record_count": 1, "registrable_domain_count": 1},
            "test": {"record_count": 1, "registrable_domain_count": 1},
        },
    }


def test_rejects_duplicate_record_id():
    manifest = _manifest(
        _record("same", "https://one.example.com"),
        _record("same", "https://two.example.com"),
    )
    _assert_rejected(manifest, "duplicate record_id")


def test_rejects_exact_duplicate_url_with_same_label():
    url = "https://login.example.com/path"
    manifest = _manifest(_record("one", url), _record("two", url))
    _assert_rejected(manifest, "duplicate url")


def test_reports_label_conflict_before_duplicate_url():
    url = "https://login.example.com/path"
    manifest = _manifest(_record("one", url), _record("two", url, 1))
    _assert_rejected(manifest, "label conflict")


def test_rejects_registrable_domain_in_multiple_splits():
    manifest = _manifest(
        _record("one", "https://login.example.com", split="train"),
        _record("two", "https://shop.example.com", split="test"),
    )

    _assert_rejected(manifest, "appears in multiple splits")


@pytest.mark.parametrize(
    ("field", "value"),
    [("record_id", ""), ("url", ""), ("label", True), ("label", 2), ("split", "dev")],
)
def test_rejects_invalid_record_values(field, value):
    record = _record("one", "https://example.com")
    record[field] = value
    _assert_rejected(_manifest(record), field)


@pytest.mark.parametrize(
    "record",
    [
        {"record_id": "one", "url": "https://example.com", "label": 0},
        {**_record("one", "https://example.com"), "domain": "example.com"},
    ],
)
def test_rejects_inexact_record_fields(record):
    _assert_rejected(_manifest(record), "record fields")


@pytest.mark.parametrize(
    ("manifest", "message"),
    [
        ([], "manifest"),
        ({"schema_version": 2, "records": []}, "schema_version"),
        ({"schema_version": 1, "records": {}}, "records"),
    ],
)
def test_rejects_invalid_manifest_shape(manifest, message):
    _assert_rejected(manifest, message)
