"""Invented summary inputs for the joint reducer, without inference or reads."""

import importlib
import importlib.util
from dataclasses import replace

import pytest
from test_hypothesis_evaluation import passing_evidence, population

from automated_phishing_detection import hypothesis_evaluation as evaluator
from automated_phishing_detection.evaluation_stream import ExternalEvidence
from automated_phishing_detection.policy_replay import PolicyReplay


@pytest.fixture
def study():
    name = "automated_phishing_detection.study_evidence"
    assert importlib.util.find_spec(name), "missing joint study evidence reducer"
    return importlib.import_module(name)


def evidence():
    values = passing_evidence(evaluator)
    external = ExternalEvidence(
        {role: values["populations"][role] for role in ("gold", "certified")},
        values["controls"],
        values["external_windows"],
        PolicyReplay((), (), None),
        {"gold": 8, "certified": 100, "tranco": 100, "secondary": 0},
    )
    return {
        "internal": values["populations"]["internal"],
        "external": external,
        "reference": values["reference"],
        "http": values["http"],
    }


def internal_population(labels):
    return population(
        evaluator,
        labels,
        {
            "length_only": [0] * len(labels),
            "logistic_l1": [1] * len(labels),
            "cascade": [1] * len(labels),
            "transformer": [1] * len(labels),
        },
        "internal",
    )


def empty_external():
    original = evidence()["external"]
    return replace(
        original,
        populations={
            role: population(
                evaluator, (), {name: () for name in saved.predictions}, role
            )
            for role, saved in original.populations.items()
        },
        controls=evaluator.SavedControls((), {"cascade": (), "transformer": ()}),
        external_windows=evaluator.WindowCounts(0, 0),
        role_counts={"gold": 0, "certified": 0, "tranco": 0, "secondary": 0},
    )


def cells(result):
    return {cell.test_id: cell for cell in result.ablation_family.cells}


def without_model(saved, model):
    return replace(
        saved,
        predictions={
            name: values for name, values in saved.predictions.items() if name != model
        },
    )
