"""Independent selection audit and bounded runner sequencing fixtures."""

import copy

import pytest
import revision_preflight as preflight
from revision_population import select_cohort
from test_revision_population import benign, document

from automated_phishing_detection.protocol_preflight import parse_suffix_rules


def fixture():
    source = document([{}, {"url_norm": "https://positive.test/longer"}, benign()])
    rules = parse_suffix_rules("test")
    rows, unused = select_cohort(
        source,
        rules,
        set(),
        expected_counts={"gold_positive": 1, "certified_negative": 1},
    )
    return source, rules, rows


def test_independent_selection_matches_and_detects_omission():
    source, rules, rows = fixture()
    preflight.verify_selection(source, rules, set(), rows)
    with pytest.raises(ValueError, match="selection"):
        preflight.verify_selection(source, rules, set(), rows[:-1])


@pytest.mark.parametrize(
    "field,value",
    [
        ("raw_url", "https://wrong.test/"),
        ("is_phishing", 0),
        ("collected_at", "changed"),
        ("record_id", "wrong"),
    ],
)
def test_independent_selection_checks_preserved_fields(field, value):
    source, rules, rows = fixture()
    altered = copy.deepcopy(rows)
    positive = next(row for row in altered if row["is_phishing"] == 1)
    positive[field] = value
    with pytest.raises(ValueError, match="selection"):
        preflight.verify_selection(source, rules, set(), altered)


def test_independent_selection_checks_exposure_and_order():
    source, rules, rows = fixture()
    with pytest.raises(ValueError, match="selection"):
        preflight.verify_selection(source, rules, {"positive.test"}, rows)
    with pytest.raises(ValueError, match="selection"):
        preflight.verify_selection(source, rules, set(), rows[::-1])
