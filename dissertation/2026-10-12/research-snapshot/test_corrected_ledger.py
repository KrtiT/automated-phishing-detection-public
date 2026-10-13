"""Decision reconciliation from aggregates, without research execution."""

import copy
import csv
import importlib
import importlib.util
import json
from pathlib import Path

import pytest

REVISION = Path(__file__).resolve().parent
AGGREGATES = (
    REVISION.parents[1]
    / "gwu_working/dissertation-publication-20261004/dissertation/2026-10-04/aggregate-data"
)


@pytest.fixture
def ledger():
    assert importlib.util.find_spec("corrected_ledger"), "correction ledger missing"
    return importlib.import_module("corrected_ledger")


@pytest.fixture
def inputs():
    with (AGGREGATES / "primary-gates.csv").open() as stream:
        gates = list(csv.DictReader(stream))
    audit = json.loads((REVISION / "paired-and-dependence-audit.json").read_text())
    return gates, audit


def test_all_original_checks_remain_and_only_noninferiority_changes(ledger, inputs):
    gates, audit = inputs
    before = copy.deepcopy(gates)
    rows = ledger.corrected_rows(gates, audit)
    assert gates == before
    assert len(rows) == 22
    assert [row["name"] for row in rows] == [row["name"] for row in gates]
    assert sum(row["original_status"] == "pass" for row in rows) == 9
    assert sum(row["corrected_status"] == "pass" for row in rows) == 8
    changed = [
        row["name"] for row in rows if row["original_status"] != row["corrected_status"]
    ]
    assert changed == ["gold.cascade_minus_transformer"]
    for row in rows:
        assert row["original_threshold"] == row["corrected_threshold"]


def test_corrected_gold_recall_has_nonzero_width(ledger, inputs):
    rows = {row["name"]: row for row in ledger.corrected_rows(*inputs)}
    recall = rows["gold.cascade_minus_transformer"]
    assert recall["corrected_lower"] == pytest.approx(-0.061533043469273366)
    assert recall["corrected_upper"] == pytest.approx(0.061533043469273366)
    assert recall["method"] == "bonferroni_clopper_pearson_discordance"
    assert rows["gold.logistic_l1_minus_length_only"]["corrected_lower"] > 0


def test_fpr_gates_do_not_change_to_confidence_bound_rules(ledger, inputs):
    rows = ledger.corrected_rows(*inputs)
    for row in rows:
        if row["name"].endswith((".fpr", ".alert_rate")):
            assert row["corrected_estimate"] == float(row["original_estimate"])
            assert row["method"] == "unchanged_observed_rate_gate"
            assert row["corrected_lower"] is None
            assert row["corrected_upper"] is None


def test_degenerate_clustered_and_stateful_intervals_are_not_certified(ledger, inputs):
    rows = {row["name"]: row for row in ledger.corrected_rows(*inputs)}
    for name in ("internal.cascade_minus_logistic_l1", "gold.policy_minus_cascade"):
        assert rows[name]["corrected_lower"] is None
        assert rows[name]["corrected_upper"] is None
        assert rows[name]["corrected_status"] == "fail"
        assert "not population certainty" in rows[name]["interpretation"]


@pytest.mark.parametrize("fault", ["missing", "duplicate", "threshold", "count"])
def test_unexpected_input_changes_are_rejected(ledger, inputs, fault):
    gates, audit = copy.deepcopy(inputs)
    if fault == "missing":
        gates.pop()
    elif fault == "duplicate":
        gates[-1] = gates[0]
    elif fault == "threshold":
        next(row for row in gates if row["name"] == "gold.cascade_minus_transformer")[
            "threshold"
        ] = "-0.05"
    else:
        audit["paired_contrasts"]["gold.cascade_minus_transformer"]["cells"][
            "candidate_only"
        ] = 1
    with pytest.raises(ValueError):
        ledger.corrected_rows(gates, audit)


def test_all_joint_hypotheses_remain_unsupported(ledger, inputs):
    rows = ledger.corrected_rows(*inputs)
    for hypothesis in ("H1", "H2", "H3"):
        selected = [row for row in rows if row["hypothesis"] == hypothesis]
        assert selected
        assert not all(row["corrected_status"] == "pass" for row in selected)
