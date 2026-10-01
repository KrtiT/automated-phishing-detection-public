"""Pure original source, preparation, execution and receipt authentication."""

from hashlib import sha256
from pathlib import PurePosixPath

from . import _internal_handoff_validation as internal
from . import _study_run_schema as schema
from . import baselines, phiusiil, source_runner
from ._study_history_snapshot_sources import internal_snapshot

SOURCES = {
    "data/sources.json": "source_spec_sha256",
    "reports/phiusiil-preparation-summary.json": "preparation_summary_sha256",
}


def _source_pins(values, execution, expected):
    schema.keys(expected, SOURCES)
    for name, field in SOURCES.items():
        schema.operational.digest(expected[name])
        schema.require(sha256(values[f"source/{name}"]).hexdigest() == expected[name])
        schema.require(execution[field] == expected[name])


def _source(values, execution, pins):
    source = phiusiil._load_source_spec(values["source/data/sources.json"])
    report = source_runner._json(
        values["source/reports/phiusiil-preparation-summary.json"]
    )
    preparation = baselines._validate_preparation_summary(report)
    schema.require(report["source_spec_sha256"] == pins["data/sources.json"])
    schema.require(phiusiil._matches_exactly(report["declared_sources"], source))
    expected = _population(source, preparation)
    for field, name in (
        ("partition_sha256", "expected_sha256"),
        ("source_csv_sha256", "source_csv_sha256"),
        ("suffix_rules_sha256", "suffix_rules_sha256"),
    ):
        schema.require(execution[field] == expected[name])
    return expected, report


def _population(source, preparation):
    split = preparation["splits"]["group_test"]
    schema.require(split["domain_count"] <= split["row_count"])
    return {
        "expected_sha256": preparation["output_hashes"]["group_test.jsonl"],
        "source_csv_sha256": preparation["source_csv_sha256"],
        "suffix_rules_sha256": source["public_suffix_list"]["sha256"],
        "expected_row_count": split["row_count"],
        "expected_domain_count": split["domain_count"],
        "expected_class_counts": dict(split["class_counts"]),
    }


def authenticate(payloads, expected, execution, sources, directory):
    schema.keys(payloads, internal.SNAPSHOT_NAMES)
    schema.require(type(directory) is str and "\0" not in directory)
    path = PurePosixPath(directory)
    schema.require(path.is_absolute() and ".." not in path.parts)
    schema.require(str(path) == directory)
    internal.execution(execution, expected)
    values = internal_snapshot(
        tuple(payloads.items()),
        expected,
        {"execution": execution, "snapshot_sha256": expected},
        {"paths": {"internal-attempt": directory}},
    )
    _source_pins(values, execution, sources)
    source, report = _source(values, execution, sources)
    return values, source, report
