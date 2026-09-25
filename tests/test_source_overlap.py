"""Complete source-domain reconstruction over invented CSV buffers only."""

import builtins
import csv
import io
import json
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from hashlib import sha256
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import pytest

from automated_phishing_detection import phiusiil, protocol_preflight


def encode(value):
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode("ascii")


@pytest.fixture
def source():
    rows = [
        (f"https://{prefix}.domain-{index:02d}.example/", label)
        for index in range(20)
        for prefix, label in (("left", "0"), ("right", "1"))
    ]
    rows.extend(
        [
            ("https://invalid-label.example/path", "invalid"),
            ("https://conflict.example/path", "0"),
            ("https://CONFLICT.example:443/path", "1"),
            ("https://duplicate.example/path", "0"),
            ("https://DUPLICATE.example:443/path", "0"),
            ("not-a-url-secret-canary", "0"),
            ("https://127.0.0.1/", "1"),
            ("https://com/", "0"),
            ("https://invalid-escape.example/%GG", "0"),
            ("", "1"),
            ("https://missing-label.example/", ""),
            ("https://extra.domain-00.example/later", "0"),
        ]
    )
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(("URL", "label"))
    writer.writerows(rows)
    csv_bytes = stream.getvalue().encode("utf-8")
    suffix_bytes = b"com\nexample\n"
    source_spec = {
        "contract_id": phiusiil.SOURCE_CONTRACT_ID,
        "schema_version": 2,
        "phiusiil": {
            **deepcopy(phiusiil._OFFICIAL_PHIUSIIL_METADATA),
            "csv_sha256": sha256(csv_bytes).hexdigest(),
        },
        "public_suffix_list": {
            **phiusiil._OFFICIAL_PSL_METADATA,
            "sha256": sha256(suffix_bytes).hexdigest(),
        },
    }
    source_bytes = encode(source_spec)
    rules = protocol_preflight.parse_suffix_rules(suffix_bytes.decode())
    parsed = phiusiil._parse_csv_rows(csv_bytes)
    resolution = phiusiil.resolve_rows(
        parsed, csv_sha256=sha256(csv_bytes).hexdigest(), suffix_rules=rules
    )
    assigned = phiusiil.assign_splits(resolution.retained)
    outputs = phiusiil._private_output_contents(assigned, resolution)
    hashes = {name: sha256(content).hexdigest() for name, content in outputs.items()}
    sums = "".join(f"{hashes[name]}  {name}\n" for name in sorted(hashes)).encode(
        "ascii"
    )
    hashes["SHA256SUMS"] = sha256(sums).hexdigest()
    report = phiusiil._build_summary(
        assigned, resolution, source_spec, sha256(source_bytes).hexdigest(), hashes
    )
    report_bytes = encode(report)
    return SimpleNamespace(
        buffers={
            "csv_bytes": csv_bytes,
            "suffix_rules_bytes": suffix_bytes,
            "source_spec_bytes": source_bytes,
            "preparation_summary_bytes": report_bytes,
        },
        pins={
            "source_csv_sha256": sha256(csv_bytes).hexdigest(),
            "suffix_rules_sha256": sha256(suffix_bytes).hexdigest(),
            "source_spec_sha256": sha256(source_bytes).hexdigest(),
            "preparation_summary_sha256": sha256(report_bytes).hexdigest(),
        },
        raw_rows=rows,
        parsed=parsed,
        resolution=resolution,
        assigned=assigned,
        outputs=outputs,
        report=report,
    )


def module():
    return import_module("automated_phishing_detection.source_overlap")


def reconstruct(source, **buffers):
    api = module()
    return api.reconstruct_source_overlap(
        **{**source.buffers, **buffers}, pins=api.SourceOverlapPins(**source.pins)
    )


def test_complete_universe_includes_valid_domains_before_all_exclusions(source):
    result = reconstruct(source)

    assert result.overlap_domains == frozenset(
        {f"domain-{index:02d}.example" for index in range(20)}
        | {
            "invalid-label.example",
            "conflict.example",
            "duplicate.example",
            "missing-label.example",
        }
    )
    retained_domains = {row.registrable_domain for row in source.assigned}
    assert result.overlap_domains - retained_domains == {
        "invalid-label.example",
        "conflict.example",
        "missing-label.example",
    }
    assert result.group_test_bytes == source.outputs["group_test.jsonl"]
    assert (
        result.public_summary["reconstructed_output_sha256"]
        == source.report["output_hashes"]
    )


def test_manifest_covers_every_original_row_once_with_explicit_invalid_accounting(
    source,
):
    result = reconstruct(source)
    assert set(result.private_outputs) == {"source-overlap.json"}
    content = result.private_outputs["source-overlap.json"]
    manifest = json.loads(content)
    rows = manifest["rows"]

    assert [row["source_ordinal"] for row in rows] == list(
        range(1, len(source.raw_rows) + 1)
    )
    assert [row["record_id"] for row in rows] == [
        phiusiil.record_id_for_row(source.pins["source_csv_sha256"], ordinal)
        for ordinal in range(1, len(source.raw_rows) + 1)
    ]
    assert manifest["domains"] == sorted(result.overlap_domains)
    assert manifest["input_hashes"] == source.pins
    assert manifest["reconstructed_output_sha256"] == source.report["output_hashes"]
    assert rows[40]["status"] == "valid_url_and_domain"
    assert rows[40]["registrable_domain"] == "invalid-label.example"
    assert rows[41]["canonical_url_sha256"] == rows[42]["canonical_url_sha256"]
    assert rows[43]["canonical_url_sha256"] == rows[44]["canonical_url_sha256"]
    assert rows[43]["registrable_domain"] == rows[44]["registrable_domain"]
    invalid = [row for row in rows if row["status"] == "invalid_url_or_domain"]
    assert [row["source_ordinal"] for row in invalid] == [46, 47, 48, 49, 50]
    assert all(row["registrable_domain"] is None for row in invalid)
    assert all(row["canonical_url_sha256"] is None for row in invalid)
    assert result.public_summary["counts"] == {
        "input_rows": 52,
        "valid_url_domain_rows": 47,
        "invalid_url_domain_rows": 5,
        "original_valid_domains": 24,
        "retained_domains": 21,
        "valid_quarantined_rows": 5,
        "quarantined_only_domains": 3,
    }
    assert content == (
        json.dumps(
            manifest,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("ascii")


def test_result_is_immutable_and_public_summary_contains_no_row_values(source):
    result = reconstruct(source)
    with pytest.raises(FrozenInstanceError):
        result.group_test_bytes = b"changed"
    summary = result.public_summary
    assert summary["source_binding"] == "caller_supplied_pins_only"
    assert summary["protected_evaluation_authorized"] is False
    assert summary["private_sha256"] == {
        name: sha256(content).hexdigest()
        for name, content in result.private_outputs.items()
    }
    public = json.dumps(summary)
    assert all(raw_url not in public for raw_url, _ in source.raw_rows if raw_url)
    assert all(domain not in public for domain in result.overlap_domains)
    assert "rows" not in summary
    assert "private_outputs=" not in repr(result)
    assert "group_test_bytes=" not in repr(result)
    assert "overlap_domains=" not in repr(result)


def test_pure_reconstruction_reads_no_paths_and_parses_csv_once(source, monkeypatch):
    from sklearn.linear_model import LogisticRegression
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler

    api = module()
    calls = []
    original = phiusiil._parse_csv_rows

    def parse(content):
        assert content is source.buffers["csv_bytes"]
        calls.append("csv")
        return original(content)

    def forbidden(*args, **kwargs):
        pytest.fail("source reconstruction attempted file I/O or old publication")

    monkeypatch.setattr(phiusiil, "_parse_csv_rows", parse)
    for owner, name in (
        (builtins, "open"),
        (Path, "open"),
        (Path, "read_bytes"),
        (Path, "read_text"),
        (phiusiil, "_prepare_phiusiil"),
        (phiusiil, "_publish_staged_artifacts"),
        (LogisticRegression, "fit"),
        (LogisticRegression, "predict_proba"),
        (GaussianMixture, "fit"),
        (GaussianMixture, "score_samples"),
        (StandardScaler, "fit"),
    ):
        monkeypatch.setattr(owner, name, forbidden)
    result = reconstruct(source)
    assert type(result) is api.ReconstructedSource
    assert calls == ["csv"]
    assert result.group_test_bytes == source.outputs["group_test.jsonl"]


@pytest.mark.parametrize("removed_ordinal", [1, 41, 42, 45, 46])
def test_missing_source_row_cannot_survive_preparation_reconstruction(
    source, monkeypatch, removed_ordinal
):
    api = module()
    original = phiusiil._parse_csv_rows

    def incomplete(content):
        return tuple(row for row in original(content) if row.ordinal != removed_ordinal)

    monkeypatch.setattr(phiusiil, "_parse_csv_rows", incomplete)
    with pytest.raises(api.SourceOverlapError, match="preparation_mismatch"):
        reconstruct(source)


@pytest.mark.parametrize(
    "name",
    [
        "csv_bytes",
        "suffix_rules_bytes",
        "source_spec_bytes",
        "preparation_summary_bytes",
    ],
)
def test_any_buffer_digest_failure_precedes_all_parsing(source, monkeypatch, name):
    api = module()

    def forbidden(*args, **kwargs):
        pytest.fail("parsing preceded complete input authentication")

    monkeypatch.setattr(phiusiil, "_parse_csv_rows", forbidden)
    monkeypatch.setattr(phiusiil, "_load_source_spec", forbidden)
    monkeypatch.setattr(protocol_preflight, "parse_suffix_rules", forbidden)
    with pytest.raises(api.SourceOverlapError, match="hash_mismatch") as rejected:
        reconstruct(source, **{name: source.buffers[name] + b"secret-canary"})
    assert "secret-canary" not in str(rejected.value)


@pytest.mark.parametrize(
    "change", ["source_csv", "source_psl", "prep_source", "prep_pin"]
)
def test_relinked_cross_binding_failure_precedes_row_parsing(
    source, monkeypatch, change
):
    api = module()
    spec = json.loads(source.buffers["source_spec_bytes"])
    report = deepcopy(source.report)
    if change == "source_csv":
        spec["phiusiil"]["csv_sha256"] = "f" * 64
    elif change == "source_psl":
        spec["public_suffix_list"]["sha256"] = "f" * 64
    elif change == "prep_source":
        report["declared_sources"]["phiusiil"]["csv_sha256"] = "f" * 64
    else:
        report["source_spec_sha256"] = "f" * 64
    source.buffers["source_spec_bytes"] = encode(spec)
    source.pins["source_spec_sha256"] = sha256(
        source.buffers["source_spec_bytes"]
    ).hexdigest()
    source.buffers["preparation_summary_bytes"] = encode(report)
    source.pins["preparation_summary_sha256"] = sha256(
        source.buffers["preparation_summary_bytes"]
    ).hexdigest()
    monkeypatch.setattr(
        phiusiil,
        "_parse_csv_rows",
        lambda *args: pytest.fail("row parsing preceded binding"),
    )
    with pytest.raises(api.SourceOverlapError, match="binding_mismatch"):
        reconstruct(source)


@pytest.mark.parametrize(
    "output",
    [
        "train.jsonl",
        "validation.jsonl",
        "group_test.jsonl",
        "quarantine.jsonl",
        "SHA256SUMS",
    ],
)
def test_every_existing_output_hash_is_checked(source, output):
    api = module()
    report = deepcopy(source.report)
    report["output_hashes"][output] = "f" * 64
    source.buffers["preparation_summary_bytes"] = encode(report)
    source.pins["preparation_summary_sha256"] = sha256(
        source.buffers["preparation_summary_bytes"]
    ).hexdigest()
    with pytest.raises(api.SourceOverlapError, match="preparation_mismatch"):
        reconstruct(source)


@pytest.mark.parametrize(
    "field",
    [
        "overall_counts",
        "native_label_counts",
        "local_label_counts",
        "splits",
        "quarantine_reason_counts",
        "algorithms",
        "label_mapping",
        "extra",
    ],
)
def test_relinked_preparation_count_or_method_change_is_rejected(source, field):
    api = module()
    report = deepcopy(source.report)
    report[field] = {"changed": 1}
    source.buffers["preparation_summary_bytes"] = encode(report)
    source.pins["preparation_summary_sha256"] = sha256(
        source.buffers["preparation_summary_bytes"]
    ).hexdigest()
    with pytest.raises(api.SourceOverlapError, match="preparation_mismatch"):
        reconstruct(source)


def test_generated_cross_split_domain_change_cannot_replace_accepted_bytes(
    source, monkeypatch
):
    api = module()
    original = phiusiil.assign_splits

    def change(records):
        assigned = list(original(records))
        first = assigned[0]
        replacement = "validation" if first.split != "validation" else "train"
        assigned[0] = replace(first, split=replacement)
        return tuple(assigned)

    monkeypatch.setattr(phiusiil, "assign_splits", change)
    with pytest.raises(api.SourceOverlapError, match="preparation_mismatch"):
        reconstruct(source)


@pytest.mark.parametrize(
    "name",
    [
        "csv_bytes",
        "suffix_rules_bytes",
        "source_spec_bytes",
        "preparation_summary_bytes",
    ],
)
def test_mutable_or_nonbyte_inputs_are_rejected(source, name):
    api = module()
    with pytest.raises(api.SourceOverlapError, match="invalid_.*_bytes"):
        reconstruct(source, **{name: bytearray(source.buffers[name])})


def test_malformed_source_error_is_symbolic_without_source_values(source):
    api = module()
    content = b'URL,label\n"unterminated-secret-canary,0\n'
    source.buffers["csv_bytes"] = content
    source.pins["source_csv_sha256"] = sha256(content).hexdigest()
    spec = json.loads(source.buffers["source_spec_bytes"])
    spec["phiusiil"]["csv_sha256"] = source.pins["source_csv_sha256"]
    source.buffers["source_spec_bytes"] = encode(spec)
    source.pins["source_spec_sha256"] = sha256(
        source.buffers["source_spec_bytes"]
    ).hexdigest()
    report = deepcopy(source.report)
    report["declared_sources"] = spec
    report["source_spec_sha256"] = source.pins["source_spec_sha256"]
    source.buffers["preparation_summary_bytes"] = encode(report)
    source.pins["preparation_summary_sha256"] = sha256(
        source.buffers["preparation_summary_bytes"]
    ).hexdigest()

    with pytest.raises(api.SourceOverlapError) as rejected:
        reconstruct(source)
    assert "secret-canary" not in str(rejected.value)


@pytest.mark.parametrize(
    "buffer_name,pin_name",
    [
        ("source_spec_bytes", "source_spec_sha256"),
        ("preparation_summary_bytes", "preparation_summary_sha256"),
    ],
)
def test_deeply_nested_public_json_has_symbolic_error(
    source, monkeypatch, buffer_name, pin_name
):
    api = module()
    content = b"[" * 2000 + b"0" + b"]" * 2000
    source.buffers[buffer_name] = content
    source.pins[pin_name] = sha256(content).hexdigest()
    monkeypatch.setattr(
        phiusiil,
        "_parse_csv_rows",
        lambda *args: pytest.fail("row parsing preceded public binding"),
    )

    with pytest.raises(api.SourceOverlapError) as rejected:
        reconstruct(source)
    assert str(rejected.value) == "source_reconstruction_failed"
    assert rejected.value.__suppress_context__
