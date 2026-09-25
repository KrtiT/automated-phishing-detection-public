"""Complete operational HTTP records, with caller-consistency checks only.

Expected hashes refer to the upstream full manifest, not a request projection.
Outcomes do not retain URLs and cannot prove which input bytes were dispatched.
Source acceptance and observing-parent process/cell authority remain separate.
"""

from typing import Literal

from . import _http_run_codec as schema
from ._checkpoint_codec import canonical_bytes
from .http_replay import HttpRun, ReplayRequest, _json, summarize_run
from .http_schema import ScanRequest


class HttpRunCodecError(ValueError):
    """A saved record is not a complete caller-consistent operational HTTP run."""


def _expected_requests(requests):
    schema.require(type(requests) is tuple and len(requests) == 10000)
    record_ids = []
    for request in requests:
        schema.require(type(request) is ReplayRequest)
        schema.require(type(request.record_id) is str and type(request.raw_url) is str)
        ScanRequest(request_id=request.record_id, url=request.raw_url)
        record_ids.append(request.record_id)
    schema.require(len(set(record_ids)) == len(record_ids))
    return tuple(record_ids)


def _validated_run(value):
    run = schema.restore(value)
    canonical_bytes(summarize_run(run))
    return run


def encode_http_run(run: HttpRun) -> bytes:
    """Encode a complete 1,000-warmup/10,000-measured run without I/O."""
    try:
        value = schema.payload(run)
        _validated_run(value)
        return canonical_bytes(
            {"schema_version": 1, "protocol": "http-run-v1", "run": value}
        )
    except Exception:
        raise HttpRunCodecError("invalid_http_run_record") from None


def _decode(content, expected, record_ids):
    schema.require(type(content) is bytes)
    value = _json(content)
    schema.shape(value, {"schema_version", "protocol", "run"})
    schema.require(
        type(value["schema_version"]) is int and value["schema_version"] == 1
    )
    schema.require(value["protocol"] == "http-run-v1")
    schema.require(canonical_bytes(value) == content)
    schema.shape(value["run"], schema.RUN_FIELDS)
    schema.require(
        canonical_bytes({name: value["run"][name] for name in expected})
        == canonical_bytes(expected)
    )
    run = _validated_run(value["run"])
    schema.require(tuple(row.record_id for row in run.measured) == record_ids)
    schema.require(tuple(row.record_id for row in run.warmup) == record_ids[:1000])
    return run


def decode_http_run(
    content: bytes,
    *,
    expected_manifest_sha256: str,
    expected_requests: tuple[ReplayRequest, ...],
    expected_prevalence_basis_points: int,
    expected_concurrency: int,
    expected_run_index: int,
    expected_workload: Literal["fixed_cascade", "transformer_only"],
) -> HttpRun:
    """Restore a complete run; independent caller inputs confer no authority."""
    try:
        schema.metadata(
            expected_manifest_sha256,
            expected_prevalence_basis_points,
            expected_concurrency,
            expected_run_index,
            expected_workload,
        )
        record_ids = _expected_requests(expected_requests)
        expected = {
            "manifest_sha256": expected_manifest_sha256,
            "prevalence_basis_points": expected_prevalence_basis_points,
            "concurrency": expected_concurrency,
            "run_index": expected_run_index,
            "workload": expected_workload,
        }
        return _decode(content, expected, record_ids)
    except Exception:
        raise HttpRunCodecError("invalid_http_run_record") from None
