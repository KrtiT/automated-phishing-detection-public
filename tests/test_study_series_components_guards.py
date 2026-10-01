"""No IO, retrospective binding construction, or recheck grants authority."""

import builtins
import json
import os
import subprocess
from pathlib import Path

import pytest
from study_series_components_fixtures import api, arguments, case, repin_audit
from study_urlnorm_fixtures import digest
from test_study_series_components_boundaries import reject

from automated_phishing_detection import _external_source_profile as external
from automated_phishing_detection import _operational_profile as operational
from automated_phishing_detection import _study_series_components_science as science
from automated_phishing_detection import bound_models, execution_preflight
from automated_phishing_detection._checkpoint_codec import canonical_bytes


def forbidden(*args, **kwargs):
    raise AssertionError("pure transition attempted IO or live binding construction")


def test_verifier_performs_no_io_recheck_or_old_binding_construction(monkeypatch):
    values = arguments(case())
    module = api()
    for owner, names in (
        (builtins, ("open",)),
        (os, ("open", "listdir", "scandir")),
        (Path, ("open", "read_bytes", "read_text", "resolve", "stat", "lstat")),
        (subprocess, ("Popen", "run")),
        (
            execution_preflight,
            ("recheck_binding", "preflight_execution", "_read_regular"),
        ),
        (external, ("resolve_external_source_profile", "_suffix")),
        (operational, ("resolve_operational_profile",)),
        (bound_models, ("bind_models", "_read_public")),
    ):
        for name in names:
            if hasattr(owner, name):
                monkeypatch.setattr(owner, name, forbidden)
    for target in (
        execution_preflight.ExecutionBinding,
        external.CandidateExternalProfile,
        operational.CandidateOperationalProfile,
    ):
        monkeypatch.setattr(target, "__init__", forbidden)
    assert (
        module.verify_series_component_transition(**values).authorizes_execution
        is False
    )


def test_public_report_is_pinned_before_existing_json_parser(monkeypatch):
    values = arguments(case())
    values["audit_summary_bytes"] = b"PRIVATE_CORRUPTED_REPORT"
    monkeypatch.setattr(science.inputs, "loads", forbidden)
    reject(values)


def test_original_public_report_format_need_not_match_checkpoint_codec():
    selected = case()
    report = json.dumps(json.loads(selected.audit), indent=2).encode()
    repin_audit(selected, report)
    assert api().verify_series_component_transition(**arguments(selected))


@pytest.mark.parametrize(
    "kind", ["duplicate", "nonfinite", "array", "malformed", "private"]
)
def test_authenticated_public_report_still_uses_existing_closed_parser(kind):
    selected = case()
    report = json.loads(selected.audit)
    contents = {
        "duplicate": selected.audit.replace(
            b'"schema_version":1', b'"schema_version":1,"schema_version":1'
        ),
        "nonfinite": selected.audit.replace(b'"threshold":2.0', b'"threshold":NaN'),
        "array": b"[]",
        "malformed": b"PRIVATE_MALFORMED",
        "private": canonical_bytes(report | {"warnings": [{"raw_url": "PRIVATE"}]}),
    }
    repin_audit(selected, contents[kind])
    reject(arguments(selected))


@pytest.mark.parametrize("kind", ["profile", "external", "operational"])
@pytest.mark.parametrize("style", ["pretty", "duplicate", "missing_newline"])
def test_repinned_candidate_and_profile_json_must_be_canonical(kind, style):
    selected = case()
    values = arguments(selected)
    content = (
        values["profile_bytes"] if kind == "profile" else values[kind].canonical_bytes
    )
    changed = {
        "pretty": json.dumps(json.loads(content), indent=2).encode(),
        "duplicate": content.replace(
            b'"schema_version":1', b'"schema_version":1,"schema_version":1', 1
        ),
        "missing_newline": content.rstrip(b"\n"),
    }[style]
    if kind == "profile":
        values.update(profile_bytes=changed, expected_profile_sha256=digest(changed))
    else:
        selected.after = tuple(
            changed if index == (kind == "operational") else member
            for index, member in enumerate(selected.after)
        )
        selected.profile["components"][f"current_{kind}"] = digest(changed)
        values = arguments(selected)
    reject(values)


@pytest.mark.parametrize("kind", ["external", "operational"])
def test_original_component_pin_is_independent_of_current_candidate(kind):
    selected = case()
    profile = selected.profile
    false_original = digest(b"independent-other-original")
    profile["components"][f"original_{kind}"] = false_original
    profile["origin"]["profile"]["components"][kind] = false_original
    profile["origin"]["profile_sha256"] = digest(profile["origin"]["profile"])
    reject(arguments(selected))


@pytest.mark.parametrize("field", ["url", "upstream_url", "sha256"])
def test_external_suffix_is_preserved_by_full_original_hash_not_selected_whitelist(
    field,
):
    from study_series_components_fixtures import rewrite_component

    selected = case()
    value = json.loads(selected.after[0])
    value["public_suffix_list"][field] = digest(b"changed suffix")
    rewrite_component(selected, "external", value)
    reject(arguments(selected))
