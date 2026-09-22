"""Pure prospective development-only URL perturbations and formatting features.

These definitions implement the fixed five indicators and three operators in
secondary-analysis-v1. Inputs use existing canonical-url-v1 acceptance, including
its exclusion of malformed escapes and IP literals, plus UTF-8 encodability as in
the primary URL extractor. Canonicalized spellings are never used as output.

Uppercase indicators mean presence of at least one raw ASCII uppercase letter;
host excludes userinfo and port. Default-port equality is numeric (80 for HTTP,
443 for HTTPS), including leading-zero spellings. Percent indicators/operators
inspect valid escapes in every raw component. Path encoding skips whole existing
escapes and encodes the first literal ASCII alphanumeric with uppercase hex.

Each operator starts from the original, never another operator's output. Case
perturbation is eligible for every accepted URL; percent-case perturbation needs
an escape; path encoding needs a literal ASCII alphanumeric in the path. Eligible
no-ops have eligible=True and changed=False; ineligible results are unchanged.
Raw originals and exact outputs remain available in immutable result mappings.

No URL is fetched, no data is read or fitted, and no outcome label is inherited.
These primitives do not establish adversarial success or primary H2 evidence.
"""

import re
from dataclasses import dataclass
from urllib.parse import urlsplit

from .phiusiil import PreparationError, canonicalize_url

PERTURBATION_OPERATORS = (
    "ascii_scheme_host_uppercase",
    "percent_escape_hex_uppercase",
    "first_literal_path_alphanumeric_percent_encode",
)
FORMATTING_FEATURE_NAMES = (
    "has_uppercase_ascii_scheme",
    "has_uppercase_ascii_host",
    "has_lowercase_hex_percent_escape",
    "has_explicit_scheme_default_port",
    "has_empty_path",
)

_INVALID_URL_MESSAGE = "raw_url is missing or invalid under canonical-url-v1"
_PERCENT_ESCAPE = re.compile(r"%[0-9A-Fa-f]{2}")
_ASCII_UPPERCASE = re.compile(r"[A-Z]")
_PATH_TOKEN = re.compile(r"%[0-9A-Fa-f]{2}|([A-Za-z0-9])")
_UPPERCASE_TRANSLATION = str.maketrans(
    "abcdefghijklmnopqrstuvwxyz", "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
)


class SecondaryProbeError(ValueError):
    """The URL does not satisfy the existing frozen preparation policy."""


@dataclass(frozen=True)
class PerturbationResult:
    """One independent operator's eligibility and exact original/output pair."""

    operator: str
    original_url: str
    output_url: str
    eligible: bool
    changed: bool


@dataclass(frozen=True)
class _RawParts:
    scheme_end: int
    host_start: int
    host_end: int
    path_start: int
    path_end: int
    port: int | None


def _raw_parts(raw_url: object) -> _RawParts:
    try:
        canonicalize_url(raw_url)
        raw_url.encode("utf-8")
        parsed = urlsplit(raw_url)
    except (PreparationError, TypeError, ValueError):
        raise SecondaryProbeError(_INVALID_URL_MESSAGE) from None

    scheme_end = raw_url.index(":")
    authority_start = scheme_end + 3
    authority_end = authority_start + len(parsed.netloc)
    host_start = authority_start + parsed.netloc.rfind("@") + 1
    host_port = raw_url[host_start:authority_end]
    host_end = host_start + host_port.rfind(":") if ":" in host_port else authority_end
    return _RawParts(
        scheme_end=scheme_end,
        host_start=host_start,
        host_end=host_end,
        path_start=authority_end,
        path_end=authority_end + len(parsed.path),
        port=parsed.port,
    )


def perturb_url(raw_url: object) -> tuple[PerturbationResult, ...]:
    """Return the three noncomposed probes in ``PERTURBATION_OPERATORS`` order."""
    parts = _raw_parts(raw_url)
    case_output = (
        raw_url[: parts.scheme_end].translate(_UPPERCASE_TRANSLATION)
        + raw_url[parts.scheme_end : parts.host_start]
        + raw_url[parts.host_start : parts.host_end].translate(_UPPERCASE_TRANSLATION)
        + raw_url[parts.host_end :]
    )
    percent_output = _PERCENT_ESCAPE.sub(lambda match: match[0].upper(), raw_url)
    path_output = raw_url
    path_eligible = False
    path = raw_url[parts.path_start : parts.path_end]
    for match in _PATH_TOKEN.finditer(path):
        if match[1] is not None:
            index = parts.path_start + match.start()
            path_output = (
                raw_url[:index] + f"%{ord(match[1]):02X}" + raw_url[index + 1 :]
            )
            path_eligible = True
            break

    outputs = (case_output, percent_output, path_output)
    eligibility = (True, _PERCENT_ESCAPE.search(raw_url) is not None, path_eligible)
    return tuple(
        PerturbationResult(
            operator=operator,
            original_url=raw_url,
            output_url=output,
            eligible=eligible,
            changed=output != raw_url,
        )
        for operator, output, eligible in zip(
            PERTURBATION_OPERATORS, outputs, eligibility
        )
    )


def extract_formatting_features(raw_url: object) -> tuple[float, ...]:
    """Return only the five fixed raw-serialization indicators, without fitting."""
    parts = _raw_parts(raw_url)
    scheme = raw_url[: parts.scheme_end]
    host = raw_url[parts.host_start : parts.host_end]
    values = (
        _ASCII_UPPERCASE.search(scheme) is not None,
        _ASCII_UPPERCASE.search(host) is not None,
        any(
            any(character in "abcdef" for character in match[0][1:])
            for match in _PERCENT_ESCAPE.finditer(raw_url)
        ),
        parts.port == {"http": 80, "https": 443}[scheme.lower()],
        parts.path_start == parts.path_end,
    )
    return tuple(float(value) for value in values)
