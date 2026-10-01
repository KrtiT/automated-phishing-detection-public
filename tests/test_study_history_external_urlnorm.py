"""The original amended representation survives historical scientific replay."""

from dataclasses import asdict
from types import SimpleNamespace

from study_history_external_fixtures import arguments, digest, verify
from study_urlnorm_scientific_fixtures import (
    inputs,
    preparation_api,
    preparation_case,
    runner,
    scientific_case,
    score_and_verify,
)
from test_study_history_external_boundaries import forbid_live

from automated_phishing_detection.study_history_internal import (
    verify_historical_internal_science,
)

__all__ = ["inputs", "preparation_api", "preparation_case", "runner", "scientific_case"]


def restore_internal(case, original):
    payloads = dict(original.payloads)
    return verify_historical_internal_science(
        payloads,
        expected_snapshot_sha256={
            name: digest(content) for name, content in payloads.items()
        },
        expected_execution=original.public_summary["execution"],
        expected_source_sha256={
            name: dict(case.binding.source_hashes)[name]
            for name in (
                "data/sources.json",
                "reports/phiusiil-preparation-summary.json",
            )
        },
        expected_attempt_directory=str(case.internal_paths.attempt),
    )


def test_retained_urlnorm_science_reconstructs_without_new_forwards(
    scientific_case, monkeypatch
):
    case = scientific_case
    internal, original, supplied = score_and_verify(case, monkeypatch)
    payloads = dict(original.payloads)
    history = SimpleNamespace(payloads=payloads, arguments=arguments(case, payloads))
    forbid_live(monkeypatch)
    restored_internal = restore_internal(case, internal)
    result = verify(history)
    assert restored_internal == internal
    assert asdict(result.replay) == asdict(original.replay)
    assert result.public_summary == original.public_summary
    assert tuple(row.record.raw_url for row in result.rows) == case.expected_urls
    assert supplied and all(urls == case.expected_urls for urls in supplied)
    assert result.payload("attempt/evidence/publisher-source.json") == (
        case.preparation.payload("publisher-source.json")
    )
