"""Deterministic preparation of the PhiUSIIL development dataset."""

import csv
import ctypes
import errno
import io
import json
import os
import shutil
import sys
import tempfile
import unicodedata
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
from urllib.parse import urlsplit

from .proposed_label_contract import map_phiusiil_label
from .protocol_preflight import (
    PreflightError,
    SuffixRules,
    normalize_hostname,
    parse_suffix_rules,
    registrable_domain,
)

CANONICAL_URL_VERSION = "canonical-url-v1"
DOMAIN_SPLIT_VERSION = "phiusiil-domain-split-v1"
SPLIT_SEED = "20260816"
SOURCE_CONTRACT_ID = "phiusiil-development-v1"
SPLITS = ("train", "validation", "group_test")
SPLIT_WEIGHTS = (70, 15, 15)
_LOWERCASE_HEX = frozenset("0123456789abcdef")


class PreparationError(ValueError):
    """Raised when deterministic preparation cannot satisfy its contract."""


@dataclass(frozen=True)
class PhiUSIILRow:
    ordinal: int
    raw_url: object
    native_label_cell: object


@dataclass(frozen=True)
class ResolvedRecord:
    record_id: str
    raw_url: str
    canonical_url: str
    canonical_url_sha256: str
    registrable_domain: str
    native_label: int
    is_phishing: int
    split: str | None = None


@dataclass(frozen=True)
class QuarantineRecord:
    record_id: str
    canonical_url_sha256: str | None
    reason_code: str


@dataclass(frozen=True)
class Resolution:
    retained: tuple[ResolvedRecord, ...]
    quarantine: tuple[QuarantineRecord, ...]
    input_row_count: int
    native_label_counts: dict[str, int]
    canonicalized_url_groups: int


@dataclass(frozen=True)
class _Candidate:
    record_id: str
    raw_url: str
    canonical_url: str
    canonical_url_sha256: str
    registrable_domain: str
    native_label: int | None
    is_phishing: int | None


def _uppercase_percent_escapes(value: str) -> str:
    result: list[str] = []
    index = 0
    while index < len(value):
        character = value[index]
        if character != "%":
            result.append(character)
            index += 1
            continue
        if index + 2 >= len(value):
            raise PreparationError("URL contains a malformed percent escape")
        digits = value[index + 1 : index + 3]
        if any(digit.lower() not in _LOWERCASE_HEX for digit in digits):
            raise PreparationError("URL contains a malformed percent escape")
        result.append(f"%{digits.upper()}")
        index += 3
    return "".join(result)


def canonicalize_url(raw_url: object) -> str:
    if type(raw_url) is not str or not raw_url:
        raise PreparationError("URL must be an exact nonempty string")
    if "\\" in raw_url:
        raise PreparationError("URL contains a prohibited backslash")
    if any(
        character.isspace() or unicodedata.category(character) == "Cc"
        for character in raw_url
    ):
        raise PreparationError("URL contains raw whitespace or a control character")

    try:
        parsed = urlsplit(raw_url)
        port = parsed.port
    except (TypeError, ValueError) as exc:
        raise PreparationError("URL is malformed") from exc

    scheme = parsed.scheme.lower()
    if scheme not in ("http", "https") or not parsed.netloc:
        raise PreparationError("URL must be absolute HTTP or HTTPS")

    userinfo, separator, host_port = parsed.netloc.rpartition("@")
    if not separator:
        userinfo, host_port = "", parsed.netloc
    if host_port.startswith("["):
        raise PreparationError("bracketed hostnames are not supported")

    explicit_port = ":" in host_port
    if explicit_port:
        raw_host, _, raw_port = host_port.rpartition(":")
        if (
            not raw_host
            or not raw_port
            or any(character not in "0123456789" for character in raw_port)
        ):
            raise PreparationError("URL port must be a nonempty decimal integer")

    try:
        hostname = normalize_hostname(raw_url)
    except PreflightError as exc:
        raise PreparationError("URL hostname is invalid or unsupported") from exc

    canonical_authority = ""
    if separator:
        canonical_authority = f"{_uppercase_percent_escapes(userinfo)}@"
    canonical_authority += hostname
    if explicit_port and not (
        (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
    ):
        canonical_authority += f":{port}"

    path = _uppercase_percent_escapes(parsed.path) if parsed.path else "/"
    query = _uppercase_percent_escapes(parsed.query)
    fragment = _uppercase_percent_escapes(parsed.fragment)
    before_fragment = raw_url.split("#", 1)[0]
    has_query_delimiter = "?" in before_fragment
    has_fragment_delimiter = "#" in raw_url

    canonical = f"{scheme}://{canonical_authority}{path}"
    if has_query_delimiter:
        canonical += f"?{query}"
    if has_fragment_delimiter:
        canonical += f"#{fragment}"
    return canonical


def _validate_sha256(value: object, field_name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _LOWERCASE_HEX for character in value)
    ):
        raise PreparationError(f"{field_name} must be a lowercase SHA-256")
    return value


def record_id_for_row(csv_sha256: str, ordinal: int) -> str:
    source_hash = _validate_sha256(csv_sha256, "csv_sha256")
    if type(ordinal) is not int or not 1 <= ordinal <= 0xFFFFFFFFFFFFFFFF:
        raise PreparationError("data-row ordinal must be an unsigned 64-bit integer")
    return f"phiusiil-row-v1:{source_hash}:{ordinal:016x}"


def resolve_rows(
    rows: Iterable[PhiUSIILRow], *, csv_sha256: str, suffix_rules: SuffixRules
) -> Resolution:
    source_hash = _validate_sha256(csv_sha256, "csv_sha256")
    groups: dict[str, list[_Candidate]] = {}
    quarantine: list[QuarantineRecord] = []
    native_counts = {"0": 0, "1": 0, "invalid": 0}
    seen_ordinals: set[int] = set()
    input_row_count = 0

    for row in rows:
        input_row_count += 1
        if not isinstance(row, PhiUSIILRow):
            raise PreparationError("each input must be a PhiUSIILRow")
        if row.ordinal in seen_ordinals:
            raise PreparationError("data-row ordinals must be unique")
        seen_ordinals.add(row.ordinal)
        record_id = record_id_for_row(source_hash, row.ordinal)

        native_label = None
        if type(row.native_label_cell) is str and row.native_label_cell in ("0", "1"):
            native_label = int(row.native_label_cell)
            native_counts[row.native_label_cell] += 1
        else:
            native_counts["invalid"] += 1
        decision = map_phiusiil_label(native_label)

        try:
            canonical_url = canonicalize_url(row.raw_url)
            domain = registrable_domain(normalize_hostname(canonical_url), suffix_rules)
        except (PreparationError, PreflightError):
            quarantine.append(
                QuarantineRecord(record_id, None, "invalid_or_missing_url")
            )
            continue

        canonical_hash = sha256(canonical_url.encode("utf-8")).hexdigest()
        groups.setdefault(canonical_url, []).append(
            _Candidate(
                record_id=record_id,
                raw_url=row.raw_url,
                canonical_url=canonical_url,
                canonical_url_sha256=canonical_hash,
                registrable_domain=domain,
                native_label=native_label,
                is_phishing=decision.is_phishing,
            )
        )

    retained: list[ResolvedRecord] = []
    for canonical_url in sorted(groups):
        candidates = sorted(groups[canonical_url], key=lambda item: item.record_id)
        if any(candidate.is_phishing is None for candidate in candidates):
            quarantine.extend(
                QuarantineRecord(
                    candidate.record_id,
                    candidate.canonical_url_sha256,
                    "invalid_phiusiil_native_label",
                )
                for candidate in candidates
            )
            continue

        mapped_labels = {candidate.is_phishing for candidate in candidates}
        if len(mapped_labels) != 1:
            quarantine.extend(
                QuarantineRecord(
                    candidate.record_id,
                    candidate.canonical_url_sha256,
                    "canonical_url_conflicting_mapping",
                )
                for candidate in candidates
            )
            continue

        selected = candidates[0]
        if selected.native_label is None or selected.is_phishing is None:
            raise AssertionError("validated mapping is unexpectedly missing")
        retained.append(
            ResolvedRecord(
                record_id=selected.record_id,
                raw_url=selected.raw_url,
                canonical_url=selected.canonical_url,
                canonical_url_sha256=selected.canonical_url_sha256,
                registrable_domain=selected.registrable_domain,
                native_label=selected.native_label,
                is_phishing=selected.is_phishing,
            )
        )
        quarantine.extend(
            QuarantineRecord(
                duplicate.record_id,
                duplicate.canonical_url_sha256,
                "canonical_url_duplicate_same_mapping",
            )
            for duplicate in candidates[1:]
        )

    return Resolution(
        retained=tuple(sorted(retained, key=lambda item: item.record_id)),
        quarantine=tuple(sorted(quarantine, key=lambda item: item.record_id)),
        input_row_count=input_row_count,
        native_label_counts=native_counts,
        canonicalized_url_groups=len(groups),
    )


def allocate_domain_counts(domain_count: int) -> dict[str, int]:
    if type(domain_count) is not int or domain_count < 0:
        raise PreparationError("domain_count must be a nonnegative exact integer")
    counts = [domain_count * weight // 100 for weight in SPLIT_WEIGHTS]
    remainders = [domain_count * weight % 100 for weight in SPLIT_WEIGHTS]
    unallocated = domain_count - sum(counts)
    remainder_order = sorted(
        range(len(SPLITS)), key=lambda index: (-remainders[index], index)
    )
    for index in remainder_order[:unallocated]:
        counts[index] += 1
    return dict(zip(SPLITS, counts))


def assign_splits(records: Iterable[ResolvedRecord]) -> tuple[ResolvedRecord, ...]:
    materialized = tuple(records)
    if any(not isinstance(record, ResolvedRecord) for record in materialized):
        raise PreparationError("split inputs must be resolved PhiUSIIL records")

    domains = {record.registrable_domain for record in materialized}
    try:
        ranked_domains = sorted(
            domains,
            key=lambda domain: (
                sha256(
                    DOMAIN_SPLIT_VERSION.encode("ascii")
                    + b"\0"
                    + SPLIT_SEED.encode("ascii")
                    + b"\0"
                    + domain.encode("ascii")
                ).digest(),
                domain,
            ),
        )
    except UnicodeEncodeError as exc:
        raise PreparationError("registrable domains must be ASCII") from exc

    allocation = allocate_domain_counts(len(ranked_domains))
    domain_splits: dict[str, str] = {}
    cursor = 0
    for split in SPLITS:
        next_cursor = cursor + allocation[split]
        domain_splits.update(
            (domain, split) for domain in ranked_domains[cursor:next_cursor]
        )
        cursor = next_cursor

    assigned = tuple(
        sorted(
            (
                replace(record, split=domain_splits[record.registrable_domain])
                for record in materialized
            ),
            key=lambda record: record.record_id,
        )
    )

    for split in SPLITS:
        split_records = [record for record in assigned if record.split == split]
        if not split_records:
            raise PreparationError(f"{split} split is empty")
        if {record.is_phishing for record in split_records} != {0, 1}:
            raise PreparationError(f"{split} split must contain both classes")

    canonical_urls = {record.canonical_url for record in assigned}
    if len(canonical_urls) != len(assigned):
        raise PreparationError("canonical URL duplicate survived resolution")
    observed_domain_splits: dict[str, str] = {}
    for record in assigned:
        previous = observed_domain_splits.setdefault(
            record.registrable_domain, record.split or ""
        )
        if previous != record.split:
            raise PreparationError("registrable domain crosses splits")
    return assigned


_SOURCE_SPEC_FIELDS = frozenset(
    {"contract_id", "schema_version", "phiusiil", "public_suffix_list"}
)
_PHIUSIIL_SOURCE_FIELDS = frozenset(
    {
        "uci_dataset_id",
        "archive_url",
        "archive_sha256",
        "csv_filename",
        "csv_sha256",
        "license",
        "page_url",
    }
)
_PSL_SOURCE_FIELDS = frozenset(
    {"url", "upstream_url", "version", "commit", "sha256", "license"}
)
_QUARANTINE_REASONS = (
    "invalid_or_missing_url",
    "invalid_phiusiil_native_label",
    "canonical_url_conflicting_mapping",
    "canonical_url_duplicate_same_mapping",
)
_OFFICIAL_PHIUSIIL_METADATA = {
    "uci_dataset_id": 967,
    "archive_url": "https://archive.ics.uci.edu/static/public/967/phiusiil%2Bphishing%2Burl%2Bdataset.zip",
    "archive_sha256": "0a639fd03aea6308c5b1c10c92aa23c2ce1505447a9137271865cd0badc9a59a",
    "csv_filename": "PhiUSIIL_Phishing_URL_Dataset.csv",
    "license": "CC BY 4.0",
    "page_url": "https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset",
}
_OFFICIAL_PSL_METADATA = {
    "url": "https://raw.githubusercontent.com/publicsuffix/list/0f1fa47ec45056a19c2fdcd32a08442de9715d12/public_suffix_list.dat",
    "upstream_url": "https://publicsuffix.org/list/public_suffix_list.dat",
    "version": "commit-pinned snapshot",
    "commit": "0f1fa47ec45056a19c2fdcd32a08442de9715d12",
    "license": "MPL-2.0",
}


def _object_without_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise PreparationError("source spec contains a duplicate JSON key")
        result[key] = value
    return result


def _require_nonempty_string(mapping: dict, key: str, section: str) -> str:
    value = mapping.get(key)
    if type(value) is not str or not value:
        raise PreparationError(f"{section}.{key} must be a nonempty string")
    return value


def _load_source_spec(source_spec_bytes: bytes) -> dict:
    source_spec = json.loads(
        source_spec_bytes.decode("utf-8"),
        object_pairs_hook=_object_without_duplicate_keys,
    )
    if not isinstance(source_spec, dict) or set(source_spec) != _SOURCE_SPEC_FIELDS:
        raise PreparationError(
            "source spec fields must be exactly contract_id, schema_version, phiusiil, and public_suffix_list"
        )
    if source_spec["contract_id"] != SOURCE_CONTRACT_ID:
        raise PreparationError(f"source spec contract_id must be {SOURCE_CONTRACT_ID}")
    if (
        type(source_spec["schema_version"]) is not int
        or source_spec["schema_version"] != 1
    ):
        raise PreparationError("source spec schema_version must be 1")

    phiusiil_source = source_spec["phiusiil"]
    if (
        not isinstance(phiusiil_source, dict)
        or set(phiusiil_source) != _PHIUSIIL_SOURCE_FIELDS
    ):
        raise PreparationError("source spec PhiUSIIL fields are invalid")
    if (
        type(phiusiil_source["uci_dataset_id"]) is not int
        or phiusiil_source["uci_dataset_id"] != 967
    ):
        raise PreparationError("source spec PhiUSIIL UCI dataset ID must be 967")
    for key in ("archive_url", "csv_filename", "license", "page_url"):
        _require_nonempty_string(phiusiil_source, key, "phiusiil")
    _validate_sha256(phiusiil_source.get("archive_sha256"), "archive_sha256")
    _validate_sha256(phiusiil_source.get("csv_sha256"), "csv_sha256")
    if phiusiil_source["license"] != "CC BY 4.0":
        raise PreparationError("source spec PhiUSIIL license must be CC BY 4.0")
    csv_filename = phiusiil_source["csv_filename"]
    if Path(csv_filename).name != csv_filename:
        raise PreparationError("source spec CSV filename must be a basename")
    if any(
        phiusiil_source[key] != expected
        for key, expected in _OFFICIAL_PHIUSIIL_METADATA.items()
    ):
        raise PreparationError("source spec PhiUSIIL metadata is not recognized")

    psl_source = source_spec["public_suffix_list"]
    if not isinstance(psl_source, dict) or set(psl_source) != _PSL_SOURCE_FIELDS:
        raise PreparationError("source spec public suffix list fields are invalid")
    for key in ("url", "upstream_url", "version", "commit", "license"):
        _require_nonempty_string(psl_source, key, "public_suffix_list")
    _validate_sha256(psl_source.get("sha256"), "public_suffix_list.sha256")
    commit = psl_source["commit"]
    if len(commit) != 40 or any(
        character not in _LOWERCASE_HEX for character in commit
    ):
        raise PreparationError("source spec PSL commit must be lowercase hexadecimal")
    if psl_source["license"] != "MPL-2.0":
        raise PreparationError("source spec PSL license must be MPL-2.0")
    if any(
        psl_source[key] != expected for key, expected in _OFFICIAL_PSL_METADATA.items()
    ):
        raise PreparationError("source spec PSL metadata is not recognized")
    return source_spec


def _parse_csv_rows(csv_bytes: bytes) -> tuple[PhiUSIILRow, ...]:
    text = csv_bytes.decode("utf-8-sig")
    reader = csv.reader(io.StringIO(text, newline=""), strict=True)
    try:
        header = next(reader)
    except StopIteration as exc:
        raise PreparationError("CSV header is missing") from exc
    if len(set(header)) != len(header):
        raise PreparationError("CSV header names must be unique")
    if "URL" not in header or "label" not in header:
        raise PreparationError("CSV header must include URL and label")
    url_index = header.index("URL")
    label_index = header.index("label")

    rows = []
    for ordinal, values in enumerate(reader, start=1):
        raw_url = values[url_index] if url_index < len(values) else None
        native_label = values[label_index] if label_index < len(values) else None
        rows.append(PhiUSIILRow(ordinal, raw_url, native_label))
    return tuple(rows)


def _jsonl_bytes(records: Iterable[dict]) -> bytes:
    return b"".join(
        (
            json.dumps(record, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
            + "\n"
        ).encode("ascii")
        for record in records
    )


def _private_output_contents(
    assigned: tuple[ResolvedRecord, ...], resolution: Resolution
) -> dict[str, bytes]:
    contents = {}
    for split in SPLITS:
        contents[f"{split}.jsonl"] = _jsonl_bytes(
            {
                "record_id": record.record_id,
                "raw_url": record.raw_url,
                "canonical_url_sha256": record.canonical_url_sha256,
                "registrable_domain": record.registrable_domain,
                "is_phishing": record.is_phishing,
                "split": record.split,
            }
            for record in assigned
            if record.split == split
        )
    contents["quarantine.jsonl"] = _jsonl_bytes(
        {
            "record_id": record.record_id,
            "canonical_url_sha256": record.canonical_url_sha256,
            "reason_code": record.reason_code,
        }
        for record in resolution.quarantine
    )
    return contents


def _build_summary(
    assigned: tuple[ResolvedRecord, ...],
    resolution: Resolution,
    source_spec: dict,
    source_spec_sha256: str,
    output_hashes: dict[str, str],
) -> dict:
    local_counts = Counter(record.is_phishing for record in assigned)
    quarantine_counts = Counter(record.reason_code for record in resolution.quarantine)
    split_counts = {}
    for split in SPLITS:
        records = [record for record in assigned if record.split == split]
        classes = Counter(record.is_phishing for record in records)
        split_counts[split] = {
            "row_count": len(records),
            "domain_count": len({record.registrable_domain for record in records}),
            "class_counts": {"0": classes[0], "1": classes[1]},
        }

    return {
        "schema_version": 1,
        "source_spec_sha256": source_spec_sha256,
        "declared_sources": source_spec,
        "algorithms": {
            "record_identifier_version": "phiusiil-row-v1",
            "canonicalization_version": CANONICAL_URL_VERSION,
            "domain_split_version": DOMAIN_SPLIT_VERSION,
            "seed": SPLIT_SEED,
            "allocation_version": "hamilton-largest-remainder-v1",
            "allocation_basis": "unique_ascii_domain_groups",
            "split_percentages": dict(zip(SPLITS, SPLIT_WEIGHTS)),
        },
        "overall_counts": {
            "input_rows": resolution.input_row_count,
            "canonicalized_url_groups": resolution.canonicalized_url_groups,
            "retained_rows": len(assigned),
            "retained_domains": len({record.registrable_domain for record in assigned}),
            "quarantined_rows": len(resolution.quarantine),
        },
        "native_label_counts": resolution.native_label_counts,
        "local_label_counts": {"0": local_counts[0], "1": local_counts[1]},
        "splits": split_counts,
        "quarantine_reason_counts": {
            reason: quarantine_counts[reason] for reason in _QUARANTINE_REASONS
        },
        "output_hashes": output_hashes,
    }


def _write_private_file(path: Path, content: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


def _write_summary_temp(summary_path: Path, content: bytes) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{summary_path.name}.tmp-", dir=summary_path.parent
    )
    try:
        os.fchmod(descriptor, 0o644)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return Path(temporary_name)


def _path_lexists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def _raise_publication_error(destination: Path, error_number: int) -> None:
    if error_number in (errno.EEXIST, errno.ENOTEMPTY):
        raise PreparationError("publication destination already exists")
    raise OSError(error_number, os.strerror(error_number), os.fspath(destination))


def _publish_path_without_replace(source: Path, destination: Path) -> None:
    """Atomically publish a file or directory without replacing a destination."""
    if sys.platform == "darwin":
        function = ctypes.CDLL(None, use_errno=True).renameatx_np
        function.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        function.restype = ctypes.c_int
        result = function(
            -2,
            os.fsencode(source),
            -2,
            os.fsencode(destination),
            0x00000004,
        )
    elif sys.platform.startswith("linux"):
        library = ctypes.CDLL(None, use_errno=True)
        try:
            function = library.renameat2
        except AttributeError as exc:
            raise PreparationError(
                "atomic no-replace publication is unavailable"
            ) from exc
        function.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        function.restype = ctypes.c_int
        result = function(
            -100,
            os.fsencode(source),
            -100,
            os.fsencode(destination),
            0x00000001,
        )
    elif os.name == "nt":
        try:
            os.rename(source, destination)
        except FileExistsError as exc:
            raise PreparationError("publication destination already exists") from exc
        return
    else:
        raise PreparationError("atomic no-replace publication is unavailable")

    if result != 0:
        _raise_publication_error(destination, ctypes.get_errno())


def _publish_staged_artifacts(
    *,
    output_dir: Path,
    summary_path: Path,
    assigned: tuple[ResolvedRecord, ...],
    resolution: Resolution,
    source_spec: dict,
    source_spec_sha256: str,
) -> dict:
    temporary_output = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    temporary_summary: Path | None = None
    try:
        os.chmod(temporary_output, 0o700)
        contents = _private_output_contents(assigned, resolution)
        output_hashes = {}
        for filename, content in contents.items():
            _write_private_file(temporary_output / filename, content)
            output_hashes[filename] = sha256(content).hexdigest()

        checksum_content = "".join(
            f"{output_hashes[filename]}  {filename}\n"
            for filename in sorted(output_hashes)
        ).encode("ascii")
        _write_private_file(temporary_output / "SHA256SUMS", checksum_content)
        output_hashes["SHA256SUMS"] = sha256(checksum_content).hexdigest()

        summary = _build_summary(
            assigned,
            resolution,
            source_spec,
            source_spec_sha256,
            output_hashes,
        )
        summary_bytes = (
            json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
        ).encode("ascii")
        temporary_summary = _write_summary_temp(summary_path, summary_bytes)

        _publish_path_without_replace(temporary_output, output_dir)
        temporary_output = None
        _publish_path_without_replace(temporary_summary, summary_path)
        temporary_summary = None
        return summary
    finally:
        if temporary_output is not None:
            shutil.rmtree(temporary_output, ignore_errors=True)
        if temporary_summary is not None:
            temporary_summary.unlink(missing_ok=True)


def prepare_phiusiil(
    *,
    csv_path: Path,
    suffix_rules_path: Path,
    source_spec_path: Path,
    output_dir: Path,
    summary_path: Path,
) -> dict:
    csv_path = Path(csv_path)
    suffix_rules_path = Path(suffix_rules_path)
    source_spec_path = Path(source_spec_path)
    output_dir = Path(output_dir)
    summary_path = Path(summary_path)

    if _path_lexists(output_dir):
        raise PreparationError("output directory already exists")
    if _path_lexists(summary_path):
        raise PreparationError("summary path already exists")
    if not output_dir.parent.is_dir():
        raise PreparationError("output directory parent must already exist")
    if not summary_path.parent.is_dir():
        raise PreparationError("summary parent must already exist")
    try:
        summary_path.resolve(strict=False).relative_to(output_dir.resolve(strict=False))
    except ValueError:
        pass
    else:
        raise PreparationError("summary must be outside the private output directory")

    source_spec_bytes = source_spec_path.read_bytes()
    source_spec = _load_source_spec(source_spec_bytes)
    if csv_path.name != source_spec["phiusiil"]["csv_filename"]:
        raise PreparationError("CSV filename does not match the source spec")

    csv_bytes = csv_path.read_bytes()
    suffix_rules_bytes = suffix_rules_path.read_bytes()
    csv_sha256 = sha256(csv_bytes).hexdigest()
    suffix_rules_sha256 = sha256(suffix_rules_bytes).hexdigest()
    if csv_sha256 != source_spec["phiusiil"]["csv_sha256"]:
        raise PreparationError("CSV SHA-256 mismatch")
    if suffix_rules_sha256 != source_spec["public_suffix_list"]["sha256"]:
        raise PreparationError("PSL SHA-256 mismatch")

    suffix_rules = parse_suffix_rules(suffix_rules_bytes.decode("utf-8"))
    rows = _parse_csv_rows(csv_bytes)
    resolution = resolve_rows(rows, csv_sha256=csv_sha256, suffix_rules=suffix_rules)
    assigned = assign_splits(resolution.retained)
    return _publish_staged_artifacts(
        output_dir=output_dir,
        summary_path=summary_path,
        assigned=assigned,
        resolution=resolution,
        source_spec=source_spec,
        source_spec_sha256=sha256(source_spec_bytes).hexdigest(),
    )
