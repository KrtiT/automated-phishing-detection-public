"""Synthetic-only known answers for the prospective URL probe primitives."""

import importlib
import inspect
from dataclasses import FrozenInstanceError

import pytest

OPERATORS = (
    "ascii_scheme_host_uppercase",
    "percent_escape_hex_uppercase",
    "first_literal_path_alphanumeric_percent_encode",
)
FEATURE_NAMES = (
    "has_uppercase_ascii_scheme",
    "has_uppercase_ascii_host",
    "has_lowercase_hex_percent_escape",
    "has_explicit_scheme_default_port",
    "has_empty_path",
)


def _module():
    return importlib.import_module("automated_phishing_detection.secondary_probes")


def _results(raw_url):
    return {result.operator: result for result in _module().perturb_url(raw_url)}


def test_operators_are_noncomposed_and_preserve_exact_original_and_output():
    raw_url = "hTtPs://User:%aa@ExAmple.COM.:00443/a%2fb?x=%af#%b0"
    results = _module().perturb_url(raw_url)

    assert tuple(result.operator for result in results) == OPERATORS
    assert tuple(result.original_url for result in results) == (raw_url,) * 3
    assert tuple(result.output_url for result in results) == (
        "HTTPS://User:%aa@EXAMPLE.COM.:00443/a%2fb?x=%af#%b0",
        "hTtPs://User:%AA@ExAmple.COM.:00443/a%2Fb?x=%AF#%B0",
        "hTtPs://User:%aa@ExAmple.COM.:00443/%61%2fb?x=%af#%b0",
    )
    assert all(result.eligible and result.changed for result in results)


def test_api_contains_only_raw_url_input_and_fixed_feature_order():
    probes = _module()
    assert probes.PERTURBATION_OPERATORS == OPERATORS
    assert probes.FORMATTING_FEATURE_NAMES == FEATURE_NAMES
    for function in (probes.perturb_url, probes.extract_formatting_features):
        assert tuple(inspect.signature(function).parameters) == ("raw_url",)


def test_results_are_immutable():
    result = _module().perturb_url("https://example.com/a")[0]
    with pytest.raises(FrozenInstanceError):
        result.output_url = "https://different.example/a"


@pytest.mark.parametrize(
    ("raw_url", "expected"),
    (
        (
            "https://User:PASS@example.com/a?Q=ab#Frag",
            "HTTPS://User:PASS@EXAMPLE.COM/a?Q=ab#Frag",
        ),
        (
            "http://first@last:pw@example.com:81/?#",
            "HTTP://first@last:pw@EXAMPLE.COM:81/?#",
        ),
        ("HtTpS://xn--bcher-kva.example./%aa", "HTTPS://XN--BCHER-KVA.EXAMPLE./%aa"),
        (
            "https://b\u00fccher.example/\u00e9?x=\u00df#\u03b2",
            "HTTPS://B\u00fcCHER.EXAMPLE/\u00e9?x=\u00df#\u03b2",
        ),
        ("HTTPS://\u00dc.example/", "HTTPS://\u00dc.EXAMPLE/"),
        ("http://example.com:00080?#", "HTTP://EXAMPLE.COM:00080?#"),
    ),
)
def test_ascii_case_operator_preserves_every_other_character(raw_url, expected):
    result = _results(raw_url)[OPERATORS[0]]
    assert result.output_url == expected
    assert result.eligible is True
    assert result.changed is True


@pytest.mark.parametrize(
    ("raw_url", "eligible", "changed", "expected"),
    (
        ("https://example.com/a", False, False, "https://example.com/a"),
        ("https://example.com/%20%2F%AB", True, False, "https://example.com/%20%2F%AB"),
        (
            "https://u%af:p%bC@example.com/%aa?q=%b0#%cD",
            True,
            True,
            "https://u%AF:p%BC@example.com/%AA?q=%B0#%CD",
        ),
        (
            "https://example.com/%252f?x=Ab#%41a",
            True,
            False,
            "https://example.com/%252f?x=Ab#%41a",
        ),
    ),
)
def test_percent_operator_only_uppercases_valid_triplets(
    raw_url, eligible, changed, expected
):
    result = _results(raw_url)[OPERATORS[1]]
    assert result.output_url == expected
    assert result.eligible is eligible
    assert result.changed is changed


@pytest.mark.parametrize(
    ("raw_url", "expected", "eligible"),
    (
        ("https://example.com/Ab9?x=1#F", "https://example.com/%41b9?x=1#F", True),
        ("https://example.com/0a", "https://example.com/%30a", True),
        (
            "https://example.com/%41%2f_B?q=A#b",
            "https://example.com/%41%2f_%42?q=A#b",
            True,
        ),
        ("https://example.com/%2541a", "https://example.com/%25%341a", True),
        (
            "https://example.com/\u00e9\u0661-A",
            "https://example.com/\u00e9\u0661-%41",
            True,
        ),
        (
            "https://User:pw@example.com?x=A#b",
            "https://User:pw@example.com?x=A#b",
            False,
        ),
        ("https://example.com/%41%2f?x=A#b", "https://example.com/%41%2f?x=A#b", False),
        (
            "https://example.com/\u00e9\u0661-._~!/?#",
            "https://example.com/\u00e9\u0661-._~!/?#",
            False,
        ),
    ),
)
def test_path_operator_skips_escapes_unicode_and_nonpath_components(
    raw_url, expected, eligible
):
    result = _results(raw_url)[OPERATORS[2]]
    assert result.output_url == expected
    assert result.eligible is eligible
    assert result.changed is eligible


def test_eligibility_distinguishes_eligible_noop_from_ineligible_unchanged():
    results = _module().perturb_url("HTTPS://EXAMPLE.COM/%20?#")
    assert tuple(result.eligible for result in results) == (True, True, False)
    assert tuple(result.changed for result in results) == (False, False, False)
    assert all(result.output_url == result.original_url for result in results)


@pytest.mark.parametrize(
    ("raw_url", "expected"),
    (
        ("https://example.com/a", (0, 0, 0, 0, 0)),
        ("hTtPs://ExAmple.COM:00443?x=%af#", (1, 1, 1, 1, 1)),
        ("http://example.com:00080/", (0, 0, 0, 1, 0)),
        ("https://example.com:80/", (0, 0, 0, 0, 0)),
        ("http://example.com:443/", (0, 0, 0, 0, 0)),
        ("https://USER:PASS@example.com?x=A#B", (0, 0, 0, 0, 1)),
        ("https://u%aa:p%BB@example.com/", (0, 0, 1, 0, 0)),
        ("https://example.com/%2f", (0, 0, 1, 0, 0)),
        ("https://example.com/?x=%aF", (0, 0, 1, 0, 0)),
        ("https://example.com/#%Ab", (0, 0, 1, 0, 0)),
        ("https://example.com/%25af?q=lowercase#%41a", (0, 0, 0, 0, 0)),
        ("https://\u00dc.example", (0, 0, 0, 0, 1)),
        ("https://xn--bcher-kva.example/", (0, 0, 0, 0, 0)),
        ("https://XN--BCHER-KVA.example/", (0, 1, 0, 0, 0)),
        ("https://example.com?#", (0, 0, 0, 0, 1)),
        ("https://example.com/?#", (0, 0, 0, 0, 0)),
    ),
)
def test_formatting_features_are_exact_raw_serialization_indicators(raw_url, expected):
    actual = _module().extract_formatting_features(raw_url)
    assert actual == tuple(float(value) for value in expected)
    assert all(type(value) is float for value in actual)


class _StringSubclass(str):
    pass


@pytest.mark.parametrize(
    "function_name", ("perturb_url", "extract_formatting_features")
)
@pytest.mark.parametrize(
    "raw_url",
    (
        None,
        "",
        17,
        b"https://example.com/a",
        _StringSubclass("https://example.com/a"),
        "example.com/a",
        "//example.com/a",
        "ftp://example.com/a",
        "https:///a",
        "https://example.com/a%ZZ",
        "https://example.com/a%a",
        "https://example.com/a%",
        "https://u%xx@example.com/a",
        "https://example.com/a?x=%xy",
        "https://example.com/a#%xy",
        "https://ex%61mple.com/a",
        "https://127.0.0.1/a",
        "https://[2001:db8::a]/a",
        "https://[not-ipv6]/a",
        "https://example.com:/a",
        "https://example.com:65536/a",
        "https://example.com:abc/a",
        "https://example.com:\u0668\u0660/a",
        "https://xn--invalid-/a",
        "https://example.com/white space",
        " https://example.com/a",
        "https://example.com/a\n",
        "https://example.com/\x00",
        "https://example.com/\\a",
        "https://example.com/\ud800",
    ),
)
def test_invalid_urls_fail_closed_under_existing_preparation_rules(
    function_name, raw_url
):
    probes = _module()
    with pytest.raises(
        probes.SecondaryProbeError,
        match="^raw_url is missing or invalid under canonical-url-v1$",
    ):
        getattr(probes, function_name)(raw_url)
