"""Pure raw-URL feature extraction for the frozen RQ1 baselines."""

import math
import re
from collections import Counter
from urllib.parse import urlsplit

from .phiusiil import PreparationError, canonicalize_url
from .protocol_preflight import PreflightError, normalize_hostname

FEATURE_NAMES = (
    "raw_url_codepoint_length",
    "raw_url_utf8_byte_length",
    "hostname_ascii_length",
    "path_codepoint_length",
    "query_codepoint_length",
    "fragment_codepoint_length",
    "hostname_label_count",
    "hostname_ascii_digit_count",
    "hostname_hyphen_count",
    "hostname_punycode_label_count",
    "path_segment_count",
    "query_parameter_count",
    "raw_url_ascii_letter_count",
    "raw_url_ascii_digit_count",
    "raw_url_other_codepoint_count",
    "raw_url_ascii_digit_ratio",
    "raw_url_other_codepoint_ratio",
    "raw_url_unique_codepoint_count",
    "raw_url_utf8_byte_entropy_bits",
    "percent_escape_count",
    "is_https",
    "has_userinfo",
    "has_explicit_port",
    "has_query_delimiter",
    "has_fragment_delimiter",
)

_INVALID_URL_MESSAGE = "raw_url is missing or invalid under canonical-url-v1"
_PERCENT_ESCAPE = re.compile(r"%[0-9A-Fa-f]{2}")


class FeatureExtractionError(ValueError):
    """Raised when a URL does not satisfy the frozen preparation rules."""


def _utf8_byte_entropy(raw_bytes: bytes) -> float:
    byte_count = len(raw_bytes)
    return -math.fsum(
        (frequency / byte_count) * math.log2(frequency / byte_count)
        for frequency in Counter(raw_bytes).values()
    )


def extract_url_features(raw_url: object) -> tuple[float, ...]:
    """Return the 25 frozen RQ1 features in ``FEATURE_NAMES`` order."""
    try:
        canonicalize_url(raw_url)
        hostname = normalize_hostname(raw_url)
        parsed = urlsplit(raw_url)
        raw_bytes = raw_url.encode("utf-8")
    except (PreparationError, PreflightError, TypeError, ValueError):
        raise FeatureExtractionError(_INVALID_URL_MESSAGE) from None

    codepoint_length = len(raw_url)
    ascii_letter_count = sum(
        character.isascii() and character.isalpha() for character in raw_url
    )
    ascii_digit_count = sum(
        character.isascii() and character.isdigit() for character in raw_url
    )
    other_codepoint_count = codepoint_length - (ascii_letter_count + ascii_digit_count)
    hostname_labels = hostname.split(".")
    authority = parsed.netloc.rsplit("@", 1)[-1]
    before_fragment = raw_url.split("#", 1)[0]

    values = (
        codepoint_length,
        len(raw_bytes),
        len(hostname),
        len(parsed.path),
        len(parsed.query),
        len(parsed.fragment),
        len(hostname_labels),
        sum(character.isdigit() for character in hostname),
        hostname.count("-"),
        sum(label.startswith("xn--") for label in hostname_labels),
        sum(bool(segment) for segment in parsed.path.split("/")),
        0 if not parsed.query else len(parsed.query.split("&")),
        ascii_letter_count,
        ascii_digit_count,
        other_codepoint_count,
        ascii_digit_count / codepoint_length,
        other_codepoint_count / codepoint_length,
        len(set(raw_url)),
        _utf8_byte_entropy(raw_bytes),
        len(_PERCENT_ESCAPE.findall(raw_url)),
        parsed.scheme.lower() == "https",
        "@" in parsed.netloc,
        ":" in authority,
        "?" in before_fragment,
        "#" in raw_url,
    )
    return tuple(float(value) for value in values)
