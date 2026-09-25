"""One strict score checker for internal records and nullable external records."""

from copy import deepcopy

import pytest
from saved_score_fixtures import saved_scores as saved_scores
from saved_score_fixtures import validate

from automated_phishing_detection import saved_evidence


@pytest.mark.parametrize("external", [False, True])
def test_shared_checker_requires_only_raw_url_record_identity(saved_scores, external):
    rows, bindings = saved_scores
    row = deepcopy(rows[0])
    row["record"] = {"raw_url": row["record"]["raw_url"]}
    if external:
        row["record"].update({"is_phishing": None, "role": "tranco_control"})
    original = deepcopy(row)
    assert validate(row, bindings) is None
    assert row == original


@pytest.mark.parametrize("fault", ["boolean_count", "float_count", "float_seed"])
def test_internal_parser_rejects_equal_but_wrong_typed_observations(
    saved_scores, fault
):
    rows, bindings = saved_scores
    if fault == "float_seed":
        rows[0]["secondary_seeds"][0]["seed"] = 42.0
    else:
        rows[0]["inference_counts"]["completed_requests"] = (
            True if fault == "boolean_count" else 1.0
        )
    content = b"".join(saved_evidence._json_bytes(row) for row in rows)
    with pytest.raises(saved_evidence.SavedEvidenceError):
        saved_evidence._parse_rows(content, bindings)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("stage1_probability", True),
        ("transformer_probability", float("nan")),
        ("length_probability", 10**400),
        ("cascade_probability", -1.0),
        ("length_decision", True),
        ("stage1_decision", 0.0),
        ("transformer_decision", 2),
        ("cascade_decision", False),
        ("band_selected", 0),
        ("negative_log_likelihood", float("inf")),
        ("monitor_probability", -0.1),
        ("features", [True] * 25),
        ("features", [0.0] * 24),
        ("length_scoring_audit_json", '{"private": "reason"}'),
        ("stage1_scoring_audit_json", '{"duplicate":1,"duplicate":1}'),
    ],
)
def test_shared_checker_rejects_malformed_primary_fields(saved_scores, field, value):
    rows, bindings = saved_scores
    rows[0][field] = value
    with pytest.raises(saved_evidence.SavedEvidenceError) as caught:
        validate(rows[0], bindings)
    assert "private" not in str(caught.value)
    assert rows[0]["record"]["raw_url"] not in str(caught.value)


@pytest.mark.parametrize("member", range(7))
def test_every_saved_tabular_decision_uses_its_own_threshold(saved_scores, member):
    rows, bindings = saved_scores
    score = rows[0]["secondary_tabular"][member]
    score["decision"] = 1 - score["decision"]
    with pytest.raises(saved_evidence.SavedEvidenceError):
        validate(rows[0], bindings)


@pytest.mark.parametrize("member", range(5))
@pytest.mark.parametrize(
    "field", ["transformer_decision", "cascade_decision", "band_selected"]
)
def test_every_seed_uses_its_frozen_threshold_and_band(saved_scores, member, field):
    rows, bindings = saved_scores
    score = rows[0]["secondary_seeds"][member]
    score[field] = not score[field] if field == "band_selected" else 1 - score[field]
    with pytest.raises(saved_evidence.SavedEvidenceError):
        validate(rows[0], bindings)


@pytest.mark.parametrize("member", range(5))
def test_seed_cascade_probability_matches_selected_component(saved_scores, member):
    rows, bindings = saved_scores
    score = rows[0]["secondary_seeds"][member]
    score["cascade_probability"] = 0.123456789
    with pytest.raises(saved_evidence.SavedEvidenceError):
        validate(rows[0], bindings)


def test_seed_42_reuses_exact_primary_probability(saved_scores):
    rows, bindings = saved_scores
    rows[0]["secondary_seeds"][0]["transformer_probability"] = 0.123456789
    with pytest.raises(saved_evidence.SavedEvidenceError):
        validate(rows[0], bindings)


def test_internal_parser_calls_shared_checker_for_every_row(saved_scores, monkeypatch):
    rows, bindings = saved_scores
    checker = getattr(saved_evidence, "_validate_score_row", None)
    assert callable(checker), "missing shared saved-score validator"
    checked = []

    def observe(row, bound):
        checked.append(row)
        return checker(row, bound)

    monkeypatch.setattr(saved_evidence, "_validate_score_row", observe)
    content = b"".join(saved_evidence._json_bytes(row) for row in rows)
    assert saved_evidence._parse_rows(content, bindings) == tuple(checked)
    assert len(checked) == len(rows)


@pytest.mark.parametrize("fault", ["empty", "order", "partition", "nullable_label"])
def test_internal_record_partition_contract_remains_unchanged(saved_scores, fault):
    rows, bindings = saved_scores
    if fault == "empty":
        rows.clear()
    elif fault == "order":
        rows.reverse()
    elif fault == "partition":
        bindings["partition_sha256"] = "f" * 64
    else:
        rows[0]["record"]["is_phishing"] = None
    content = b"".join(saved_evidence._json_bytes(row) for row in rows)
    with pytest.raises(saved_evidence.SavedEvidenceError):
        saved_evidence._parse_rows(content, bindings)


@pytest.mark.parametrize("family", ["secondary_tabular", "secondary_seeds"])
@pytest.mark.parametrize("fault", ["missing", "tuple", "order", "extra_field"])
def test_shared_checker_preserves_exact_score_family_shapes(
    saved_scores, family, fault
):
    rows, bindings = saved_scores
    if fault == "missing":
        rows[0][family].pop()
    elif fault == "tuple":
        rows[0][family] = tuple(rows[0][family])
    elif fault == "order":
        rows[0][family].reverse()
    else:
        rows[0][family][0]["private"] = "private-untrusted-value"
    with pytest.raises(saved_evidence.SavedEvidenceError) as caught:
        validate(rows[0], bindings)
    assert "private" not in str(caught.value)
