"""Invented paired decisions exercise the bounded score-replay specification."""

import importlib

import pytest


def module():
    return importlib.import_module("routing_replay")


def row(identifier, score, label, structural, transformer, role="internal"):
    return {
        "record": {
            "record_id": identifier,
            "is_phishing": label,
            "registrable_domain": identifier + ".test",
            "role": role,
        },
        "stage1_probability": score,
        "stage1_decision": structural,
        "transformer_decision": transformer,
        "band_selected": score == 0.5,
        "cascade_decision": transformer if score == 0.5 else structural,
    }


def test_zero_band_routes_exact_ties_only():
    replay = module()
    assert replay.gate_mask([0.49, 0.5, 0.51], "symmetric", 0, threshold=0.5) == [
        False,
        True,
        False,
    ]


def test_directional_band_boundaries_are_explicit():
    replay = module()
    scores = [0.25, 0.5, 0.75]
    assert replay.gate_mask(scores, "below", 0.25, threshold=0.5) == [
        True,
        False,
        False,
    ]
    assert replay.gate_mask(scores, "above", 0.25, threshold=0.5) == [False, True, True]


@pytest.mark.parametrize(
    "scores,family,width",
    [
        ([float("nan")], "above", 0.1),
        ([1.1], "above", 0.1),
        ([0.5], "unknown", 0.1),
        ([0.5], "above", -0.1),
    ],
)
def test_invalid_gates_fail(scores, family, width):
    with pytest.raises(ValueError):
        module().gate_mask(scores, family, width)


def test_benefit_and_harm_are_reported_separately():
    rows = [
        row("a", 0.4, 1, 0, 1),
        row("b", 0.6, 0, 0, 1),
        row("c", 0.7, 1, 1, 0),
        row("d", 0.8, 0, 1, 0),
    ]
    result = module().summarize(rows, [True] * 4)
    assert result["true_positive"] == result["false_positive"] == 1
    assert result["positive_repairs"] == result["positive_harms"] == 1
    assert result["negative_repairs"] == result["negative_harms"] == 1
    assert result["logical_calls"] == 4


def test_tranco_is_not_counted_as_known_negative():
    result = module().summarize([row("a", 0.5, 0, 1, 1, "tranco")], [True])
    assert result["alerts"] == 1
    assert result["false_positive"] == 0
    assert result["fpr"] is None


def test_random_control_matches_budget_without_order_or_label_dependence():
    replay = module()
    identifiers = ["a", "b", "c", "d"]
    chosen = {
        identifier
        for identifier, routed in zip(identifiers, replay.random_mask(identifiers, 2))
        if routed
    }
    reversed_ids = list(reversed(identifiers))
    assert chosen == {
        identifier
        for identifier, routed in zip(reversed_ids, replay.random_mask(reversed_ids, 2))
        if routed
    }
    assert len(chosen) == 2
    with pytest.raises(ValueError):
        replay.random_mask(["a", "a"], 1)


def test_saved_original_decision_and_mask_must_replay():
    rows = [row("a", 0.5, 1, 0, 1), row("b", 0.6, 0, 0, 1)]
    assert module().verify_original(rows, threshold=0.5)["exact_threshold_ties"] == 1
    rows[0]["band_selected"] = False
    with pytest.raises(ValueError):
        module().verify_original(rows, threshold=0.5)


def test_selection_uses_all_constraints_and_never_relaxes_them():
    candidates = [
        {
            "name": "high_fpr",
            "width": 0.1,
            "metrics": {
                "fpr": 0.02,
                "recall": 0.99,
                "logical_call_fraction": 0.1,
                "true_positive": 99,
                "false_positive": 2,
                "logical_calls": 10,
            },
        },
        {
            "name": "eligible",
            "width": 0.1,
            "metrics": {
                "fpr": 0.01,
                "recall": 0.98,
                "logical_call_fraction": 0.2,
                "true_positive": 98,
                "false_positive": 1,
                "logical_calls": 20,
            },
        },
        {
            "name": "expensive",
            "width": 0.2,
            "metrics": {
                "fpr": 0.00,
                "recall": 0.99,
                "logical_call_fraction": 0.31,
                "true_positive": 99,
                "false_positive": 0,
                "logical_calls": 31,
            },
        },
    ]
    assert module().select(candidates, reference_recall=1.0) == "eligible"
    assert module().select([candidates[0], candidates[2]], reference_recall=1.0) is None
