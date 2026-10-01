"""Fully rehashed synthetic mutations cannot widen the closed series header."""

import pytest
from study_series_adoption_fixtures import (
    api,
    digest,
    make_case,
    refresh,
    selected,
    validate,
)

MUTATIONS = (
    ("profile.schema_version", True),
    ("profile.profile_id", "study-execution-profile-v1"),
    ("profile.series_id", ""),
    ("profile.operator", "Different operator"),
    ("profile.execution.revision", "a" * 40),
    ("profile.execution.contract_sha256", "0" * 64),
    ("profile.origin.attempt_ordinal", True),
    ("profile.origin.attempt_ordinal", 3),
    ("profile.origin.revision", "c" * 40),
    ("profile.origin.profile_sha256", "0" * 64),
    ("profile.origin.policy_sha256", "0" * 64),
    ("profile.origin.method_sha256", "0" * 64),
    ("profile.components.original_external", "0" * 64),
    ("profile.components.original_operational", "0" * 64),
    ("profile.segment.ordinal", True),
    ("profile.segment.ordinal", 3),
    ("profile.segment.start_ordinal", True),
    ("profile.segment.start_ordinal", 1),
    ("profile.segment.start_ordinal", 126),
    ("profile.segment.start_ordinal", 73.0),
    ("profile.segment.end_ordinal", 124),
    ("profile.segment.session_id", "fixture-1-urlnorm"),
    ("profile.segment.session_requirements.power", "battery"),
    ("profile.segment.operator_commitment", "best_effort"),
    ("profile.invocation.script", "scripts/run_adopted_study.py"),
    ("profile.invocation.arguments.expected-revision", "c" * 40),
    ("profile.paths.series_attempt", "relative"),
    ("profile.paths.series_attempt", "/invented/repository/nested"),
    ("profile.paths.series_attempt", "/invented/fresh/cells_dir/nested"),
    ("profile.paths.series_attempt", "/invented/history/index.json"),
    ("profile.paths.series_attempt", "/invented/outputs/attempt-urlnorm/nested"),
    ("profile.paths.series_attempt", "/invented/outputs/attempt/nested"),
    ("profile.paths.series_attempt", "/invented//alias"),
    ("profile.paths.series_attempt", "/invented/../escape"),
    ("operator_directive.schema_version", True),
    ("operator_directive.decision", "pending"),
    ("operator_directive.scope", "development_only"),
    ("operator_directive.operator", "Different operator"),
    ("operator_directive.decision_text", ""),
    ("operator_directive.reviewed_proposal_sha256", "0" * 64),
    ("operator_directive.advisor_decision_obtained", True),
    ("operator_directive.advisor_decision_obtained", 0),
    (
        "operator_directive.personally_reviewed_future_code_or_profile_hash_claimed",
        True,
    ),
    ("operator_directive.recorded_at", "2026-02-30T00:00:00Z"),
    ("technical_rebind.recorded_at", "2026-09-29T00:00:00Z"),
    ("technical_rebind.advisor_decision_obtained", True),
    ("technical_rebind.scope", "governance_only"),
    ("decisions.segment_execution.status", "approved"),
    ("decisions.historical_access.operator", "Different operator"),
    ("decisions.amendment.subject_kind", "method"),
    ("revoked", True),
)


@pytest.mark.parametrize(("path", "value"), MUTATIONS)
def test_rehashed_semantic_mutation_rejects(path, value):
    case = make_case()
    parent, _, name = path.rpartition(".")
    target = selected(case.envelope, parent) if parent else case.envelope
    target[name] = value
    refresh(case)
    with pytest.raises(api().SeriesAdoptionError):
        validate(case)


@pytest.mark.parametrize(
    "path",
    (
        "technical_rebind.directive_sha256",
        "technical_rebind.profile_sha256",
        "technical_rebind.history_index_sha256",
        "decisions.amendment.subject_sha256",
        "decisions.historical_access.directive_sha256",
        "decisions.segment_execution.subject_sha256",
    ),
)
def test_join_mismatch_rejects_without_repairing_join(path):
    case = make_case()
    parent, _, name = path.rpartition(".")
    selected(case.envelope, parent)[name] = "0" * 64
    with pytest.raises(api().SeriesAdoptionError):
        validate(case)


@pytest.mark.parametrize(
    "mutation", ("change", "delete", "add", "alias", "overlap", "missing_addition")
)
def test_complete_scope_cannot_omit_change_or_widen_original_bytes(mutation):
    case = make_case()
    transition = case.profile["transition"]
    unchanged, added = transition["unchanged_sha256"], transition["added_sha256"]
    original_name = next(iter(unchanged))
    if mutation == "change":
        unchanged[original_name] = "0" * 64
    elif mutation == "delete":
        unchanged.pop(original_name)
    elif mutation in ("add", "alias"):
        added["src/unreviewed.py" if mutation == "add" else "src/../alias.py"] = (
            "0" * 64
        )
    elif mutation == "overlap":
        added[original_name] = unchanged[original_name]
    else:
        added.pop(next(iter(added)))
    case.profile["source_artifact_scope"] = unchanged | added
    refresh(case)
    with pytest.raises(api().SeriesAdoptionError):
        validate(case)


@pytest.mark.parametrize(
    ("path", "value"),
    (
        ("status", "development_candidate_header_only"),
        ("legacy_readiness", True),
        ("protective_deadlines_seconds.startup", 301),
        ("session_requirements.power", "battery"),
        ("amendment.representation", "raw_url"),
        ("admission_roles", ["internal", "external"]),
        ("scripts.child", "scripts/run_study_child.py"),
        ("origin_policy_sha256", "0" * 64),
        ("permitted_added_paths", ["src/**/*.py"]),
    ),
)
def test_fully_rehashed_unreviewed_policy_rejects(path, value):
    case = make_case()
    parent, _, name = path.rpartition(".")
    target = selected(case.policy, parent) if parent else case.policy
    target[name] = value
    case.profile["policy_sha256"] = digest(case.policy)
    case.profile["amendment_sha256"] = digest(case.policy["amendment"])
    refresh(case)
    with pytest.raises(api().SeriesAdoptionError):
        validate(case)
