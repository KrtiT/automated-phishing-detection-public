"""Exact finite-sample power checks use hypothetical probabilities only."""

import importlib

import pytest
from scipy.stats import multinomial

from automated_phishing_detection.paired_inference import (
    PairedTable,
    paired_exact_envelope,
)


def calculate(*args, **kwargs):
    assert (
        importlib.util.find_spec("automated_phishing_detection.paired_power")
        is not None
    ), "paired power calculation is missing"
    return importlib.import_module(
        "automated_phishing_detection.paired_power"
    ).noninferiority_power(*args, **kwargs)


@pytest.mark.parametrize(
    "positive,negative",
    [(0.0, 0.0), (0.0, 0.2), (0.2, 0.0), (0.2, 0.3), (1.0, 0.0), (0.0, 1.0)],
)
def test_power_agrees_with_finite_multinomial_enumeration(positive, negative):
    expected = 0.0
    total, margin = 7, 0.6
    for favorable in range(total + 1):
        for adverse in range(total - favorable + 1):
            table = PairedTable(total - favorable - adverse, favorable, adverse, 0)
            if paired_exact_envelope(table).lower >= -margin:
                expected += multinomial.pmf(
                    [favorable, adverse, total - favorable - adverse],
                    total,
                    [positive, negative, 1 - positive - negative],
                )
    assert calculate(total, positive, negative, margin=margin) == pytest.approx(
        expected, abs=1e-12
    )


def test_deterministic_agreement_is_not_a_general_power_assumption():
    assert calculate(69, 0.0, 0.0) == 0.0
    assert calculate(300, 0.0, 0.0) == 1.0
    assert 0 < calculate(300, 0.05, 0.05) < 0.9


@pytest.mark.parametrize(
    "arguments",
    [
        (True, 0.1, 0.1),
        (0, 0.1, 0.1),
        (50, -0.1, 0.1),
        (50, 0.7, 0.7),
        (50, float("nan"), 0.1),
    ],
)
def test_invalid_design_is_rejected(arguments):
    with pytest.raises(ValueError):
        calculate(*arguments)
