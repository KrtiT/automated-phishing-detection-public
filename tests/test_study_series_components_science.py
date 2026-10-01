"""Full independently original components preserve every scientific field."""

import json

import pytest
from study_series_components_fixtures import (
    arguments,
    case,
    repin_audit,
    rewrite_component,
)
from study_urlnorm_fixtures import digest
from test_study_series_components_boundaries import reject

from automated_phishing_detection._checkpoint_codec import canonical_bytes


@pytest.mark.parametrize("kind", ["external", "operational"])
@pytest.mark.parametrize(
    "field",
    [
        "revision",
        "execution_contract_sha256",
        "runtime_sha256",
        "source_spec_sha256",
        "extra",
    ],
)
def test_full_component_execution_dictionary_is_joined(kind, field):
    selected = case()
    value = json.loads(selected.after[("external", "operational").index(kind)])
    value["execution"][field] = "c" * (40 if field == "revision" else 64)
    rewrite_component(selected, kind, value)
    reject(arguments(selected))


@pytest.mark.parametrize(
    "kind,field",
    [
        ("external", "publisher"),
        ("external", "public_suffix_list"),
        ("external", "retention"),
        ("external", "implementation_sha256"),
        ("operational", "bound_file_sha256"),
        ("operational", "commands"),
        ("operational", "retention"),
        ("operational", "whole_study"),
        ("operational", "manifest_sha256"),
        ("operational", "protective_deadlines_seconds"),
        ("operational", "pre_prediction_policy"),
        ("operational", "session_exclusivity"),
    ],
)
def test_any_scientific_projection_field_must_match_original_digest(kind, field):
    selected = case()
    value = json.loads(selected.after[("external", "operational").index(kind)])
    value[field] = {"changed": "scientific canary"}
    rewrite_component(selected, kind, value)
    reject(arguments(selected))


@pytest.mark.parametrize("kind", ["external", "operational"])
@pytest.mark.parametrize(
    "field",
    [
        "schema_version",
        "profile_id",
        "status",
        "protected_evaluation_ready",
        "protected_evaluation_authorized",
        "extra",
    ],
)
def test_rehashed_current_candidate_stays_closed_and_unready(kind, field):
    selected = case()
    value = json.loads(selected.after[("external", "operational").index(kind)])
    value[field] = True
    rewrite_component(selected, kind, value)
    reject(arguments(selected))


@pytest.mark.parametrize(
    "mutation",
    ["missing", "duplicate", "reverse", "reference", "primary", "cell", "float"],
)
def test_all_125_cells_and_reference_primary_ordinals_are_frozen(mutation):
    selected = case()
    value = json.loads(selected.after[1])
    schedule = value["schedule"]
    if mutation == "missing":
        schedule["cells"].pop()
    elif mutation == "duplicate":
        schedule["cells"][1] = schedule["cells"][0]
    elif mutation == "reverse":
        schedule["cells"].reverse()
    elif mutation in ("reference", "float"):
        schedule["reference_ordinal"] = 2 if mutation == "reference" else 1.0
    elif mutation == "primary":
        schedule["primary_http_ordinals"] = [22, 23, 24, 25, 26]
    else:
        schedule["cells"][100]["extra"] = "changed"
    rewrite_component(selected, "operational", value)
    selected.profile["scientific_pins"]["schedule_sha256"] = digest(schedule)
    reject(arguments(selected))


@pytest.mark.parametrize("name", ["schedule_sha256", "original_h2_audit_sha256"])
def test_independent_scientific_pins_not_report_or_subobject_digests(name):
    selected = case()
    selected.profile["scientific_pins"][name] = digest(selected.audit)
    reject(arguments(selected))


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "completed"),
        ("audit_alert_count", 27),
        ("audit_alert_count", 28.0),
        ("audit_alert_count", True),
        ("audit_window_count", 253),
        ("audit_window_count", 252.0),
        ("audit_alert_fraction", 0),
        ("false_alert_gate_met", True),
        ("input_counts", {"audit": {"complete_windows": 251}}),
        ("artifact_hashes", {"validation-audit.json": digest(b"other-audit")}),
        ("extra", "PRIVATE_CANARY"),
        ("candidates", [{"private": True}]),
    ],
)
def test_independently_repinned_public_audit_retains_fixed_failure(field, value):
    selected = case()
    report = json.loads(selected.audit)
    report[field] = value
    repin_audit(selected, canonical_bytes(report))
    reject(arguments(selected))
