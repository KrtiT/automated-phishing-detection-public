"""Deterministic in-memory replay selection from prepared group-test records.

The caller supplies a materialized list or tuple whose preparation and pinned-PSL
provenance have already been verified. This module checks domain syntax and its
relationship to the URL hostname, but does not perform public-suffix extraction.
It never reads source files, fetches URLs, or writes manifests.
"""

import json
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256

import numpy as np

from .phiusiil import PreparationError, canonicalize_url
from .protocol_preflight import PreflightError, normalize_hostname

_PREVALENCES = (10, 100, 500)
_MEASURED_COUNT = 10000
_WARMUP_COUNT = 1000
_RECORD_ID = re.compile(r"phiusiil-row-v1:([0-9a-f]{64}):([0-9a-f]{16})")
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class ManifestRecord:
    """Prepared private record; raw URL spelling is preserved during replay."""

    record_id: str
    raw_url: str
    canonical_url_sha256: str
    registrable_domain: str
    is_phishing: int
    split: str


@dataclass(frozen=True)
class ReplayManifest:
    """Ordered measured rows and a digest binding their complete private payload."""

    prevalence_basis_points: int
    records: tuple[ManifestRecord, ...]
    sha256: str

    @property
    def warmup_records(self) -> tuple[ManifestRecord, ...]:
        """Replay this prefix separately; do not count warmup in measurements."""
        return self.records[:_WARMUP_COUNT]


class InsufficientStratum(ValueError):
    """A requested class lacks enough candidates; no substitute is selected."""

    def __init__(
        self, prevalence_basis_points: int, label: int, required: int, available: int
    ) -> None:
        self.prevalence_basis_points = prevalence_basis_points
        self.label = label
        self.required = required
        self.available = available
        super().__init__(
            f"insufficient stratum for prevalence_basis_points={prevalence_basis_points}: "
            f"label={label}, required={required}, available={available}"
        )


def _validate_record(record: ManifestRecord) -> str:
    if type(record) is not ManifestRecord:
        raise TypeError("every candidate must be a ManifestRecord")
    if type(record.is_phishing) is not int or record.is_phishing not in (0, 1):
        raise ValueError("is_phishing must be exact integer 0 or 1")
    if type(record.split) is not str or record.split != "group_test":
        raise ValueError("split must be group_test")
    identifier = (
        _RECORD_ID.fullmatch(record.record_id)
        if type(record.record_id) is str
        else None
    )
    if identifier is None or int(identifier[2], 16) == 0:
        raise ValueError("record_id must be a one-based phiusiil-row-v1 identifier")
    if (
        type(record.canonical_url_sha256) is not str
        or _SHA256.fullmatch(record.canonical_url_sha256) is None
    ):
        raise ValueError("canonical_url_sha256 must be a lowercase SHA-256")

    try:
        canonical = canonicalize_url(record.raw_url)
        canonical_hash = sha256(canonical.encode("utf-8")).hexdigest()
    except (PreparationError, UnicodeError) as exc:
        raise ValueError(
            "raw_url must satisfy canonical-url-v1 and encode as UTF-8"
        ) from exc
    if canonical_hash != record.canonical_url_sha256:
        raise ValueError("canonical_url_sha256 does not match raw_url")

    domain = record.registrable_domain
    if type(domain) is not str or not domain.isascii() or "." not in domain:
        raise ValueError("registrable_domain must be a canonical ASCII domain")
    try:
        normalized_domain = normalize_hostname(f"https://{domain}/")
    except PreflightError as exc:
        raise ValueError("registrable_domain must be a canonical ASCII domain") from exc
    hostname = normalize_hostname(canonical)
    if normalized_domain != domain or not (
        hostname == domain or hostname.endswith(f".{domain}")
    ):
        raise ValueError(
            "registrable_domain must be canonical and match the URL hostname"
        )
    return identifier[1]


def _validated_records(
    records: Sequence[ManifestRecord],
) -> tuple[ManifestRecord, ...]:
    if type(records) not in (list, tuple):
        raise TypeError("records must be a materialized list or tuple")
    snapshot = tuple(records)
    record_ids: set[str] = set()
    canonical_hashes: set[str] = set()
    source_hash = None
    for record in snapshot:
        record_source_hash = _validate_record(record)
        if source_hash is not None and source_hash != record_source_hash:
            raise ValueError("all record_ids must identify the same source CSV hash")
        source_hash = record_source_hash
        if record.record_id in record_ids:
            raise ValueError("duplicate record_id")
        if record.canonical_url_sha256 in canonical_hashes:
            raise ValueError("duplicate canonical_url_sha256")
        record_ids.add(record.record_id)
        canonical_hashes.add(record.canonical_url_sha256)
    return snapshot


def _validated_candidates(
    records: Sequence[ManifestRecord],
) -> tuple[ManifestRecord, ...]:
    return tuple(
        sorted(_validated_records(records), key=lambda record: record.record_id)
    )


def _permutation(size: int, prevalence_basis_points: int, purpose: int) -> np.ndarray:
    generator = np.random.Generator(
        np.random.PCG64(
            np.random.SeedSequence([20260816, 1, prevalence_basis_points, purpose])
        )
    )
    return generator.permutation(np.arange(size, dtype=np.int64))


def _manifest_bytes(
    prevalence_basis_points: int, records: tuple[ManifestRecord, ...]
) -> bytes:
    payload = {
        "schema_version": 1,
        "algorithm_id": "replay-manifest-v1",
        "prevalence_basis_points": prevalence_basis_points,
        "records": [asdict(record) for record in records],
    }
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _manifest_hash(
    prevalence_basis_points: int, records: tuple[ManifestRecord, ...]
) -> str:
    return sha256(_manifest_bytes(prevalence_basis_points, records)).hexdigest()


def build_manifest(
    records: Sequence[ManifestRecord], *, prevalence_basis_points: int
) -> ReplayManifest:
    """Select exactly 10,000 rows using replay-manifest-v1 and NumPy 2.2.6.

    Only exact integer prevalences 10, 100, and 500 basis points are supported.
    Validate every candidate before capacity checks or selection. Each class is
    sampled without replacement, with independent purpose-specific RNG streams.
    Different prevalence manifests may overlap and are not independent samples.
    """
    if (
        type(prevalence_basis_points) is not int
        or prevalence_basis_points not in _PREVALENCES
    ):
        raise ValueError(
            "prevalence_basis_points must be exact integer 10, 100, or 500"
        )
    candidates = _validated_candidates(records)
    pools = tuple(
        tuple(record for record in candidates if record.is_phishing == label)
        for label in (0, 1)
    )
    required = (_MEASURED_COUNT - prevalence_basis_points, prevalence_basis_points)
    for label in (0, 1):
        if len(pools[label]) < required[label]:
            raise InsufficientStratum(
                prevalence_basis_points, label, required[label], len(pools[label])
            )
    selected = tuple(
        pools[label][index]
        for label in (0, 1)
        for index in _permutation(len(pools[label]), prevalence_basis_points, label)[
            : required[label]
        ]
    )
    ordered = tuple(
        selected[index]
        for index in _permutation(_MEASURED_COUNT, prevalence_basis_points, 2)
    )
    return ReplayManifest(
        prevalence_basis_points=prevalence_basis_points,
        records=ordered,
        sha256=_manifest_hash(prevalence_basis_points, ordered),
    )
