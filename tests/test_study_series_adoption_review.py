"""Independent header checks use invented declarations and no referenced files."""

import builtins
import os
import subprocess

import pytest
from study_series_adoption_fixtures import api, digest, make_case, refresh, validate

from automated_phishing_detection._checkpoint_codec import canonical_bytes

OUTPUTS = (
    "series_attempt",
    "series_public_summary",
    "segment_attempt",
    "segment_public_summary",
    "historical_inputs_dir",
    "cells_dir",
    "physical_records_dir",
)


@pytest.mark.parametrize("output", OUTPUTS)
@pytest.mark.parametrize("source", ("origin", "ancestor", "history"))
def test_every_output_is_disjoint_from_declared_original_inputs(output, source):
    case = make_case()
    original = case.profile["origin"]["profile"]
    protected = {
        "origin": original["paths"]["accepted-inputs-dir"],
        "ancestor": original["continuation"]["prior_profile"]["paths"]["cells-dir"],
        "history": case.profile["history"]["exposure_record_path"],
    }
    case.profile["paths"][output] = protected[source] + "/nested"
    refresh(case)
    with pytest.raises(api().SeriesAdoptionError):
        validate(case)


@pytest.mark.parametrize("output", OUTPUTS)
def test_output_cannot_contain_a_declared_history_file(output):
    case = make_case()
    case.profile["history"]["eligible_prefix_review_path"] = (
        case.profile["paths"][output] + "/review.json"
    )
    refresh(case)
    with pytest.raises(api().SeriesAdoptionError):
        validate(case)


@pytest.mark.parametrize("output", OUTPUTS[1:])
@pytest.mark.parametrize("suffix", ("", "/nested"))
def test_output_pairs_cannot_alias_or_nest(output, suffix):
    case = make_case()
    case.profile["paths"][output] = case.profile["paths"]["series_attempt"] + suffix
    refresh(case)
    with pytest.raises(api().SeriesAdoptionError):
        validate(case)


@pytest.mark.parametrize("member", ("policy", "profile", "envelope"))
@pytest.mark.parametrize("change", ("duplicate", "newline", "utf16"))
def test_each_input_requires_canonical_bytes_after_repin(member, change):
    case = make_case()
    content = canonical_bytes(getattr(case, member))
    changed = {
        "duplicate": b'{"schema_version":1,' + content[1:],
        "newline": content + b"\n",
        "utf16": content.decode().encode("utf-16"),
    }[change]
    overrides = {f"{member}_bytes": changed}
    if member == "policy":
        case.profile["policy_sha256"] = digest(changed)
        refresh(case)
    else:
        overrides[f"expected_{member}_sha256"] = digest(changed)
    with pytest.raises(api().SeriesAdoptionError):
        validate(case, **overrides)


@pytest.mark.parametrize("record", ("operator_directive", "technical_rebind"))
@pytest.mark.parametrize(
    "timestamp",
    ("2026-09-30T01:00:00", "2026-09-30T01:00:00+00:00", "2026-09-30T01:00:00.1Z"),
)
def test_recorded_times_use_original_exact_utc_syntax(record, timestamp):
    case = make_case()
    case.envelope[record]["recorded_at"] = timestamp
    refresh(case)
    with pytest.raises(api().SeriesAdoptionError):
        validate(case)


def test_header_never_opens_files_or_starts_processes(monkeypatch):
    case = make_case()

    def forbidden(*args, **kwargs):
        pytest.fail("pre-access metadata verification attempted IO")

    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(os, "open", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    result = validate(case)
    assert result.authorizes_execution is False
    assert result.scope == "closed_series_adoption_metadata_only"
