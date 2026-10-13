"""Finite-sample power of the conservative paired interval under IID pairs.

Hypothetical discordance rates are design inputs, not estimates from a new cohort.
This does not account for dependence, uncertain labels or outcome-driven selection.
"""

import math
from functools import lru_cache

import numpy as np
from scipy.stats import beta, binom

from .paired_inference import PairedTable, paired_exact_envelope


def _validate(total, favorable, adverse, margin, confidence):
    if type(total) is not int or total < 1:
        raise ValueError("sample size must be a positive integer")
    if any(
        type(value) not in (int, float) or not math.isfinite(value)
        for value in (favorable, adverse, margin, confidence)
    ):
        raise ValueError("design parameters must be finite numbers")
    if min(favorable, adverse) < 0 or favorable + adverse > 1:
        raise ValueError("discordance probabilities must lie in the simplex")
    if not 0 < margin < 1 or not 0 < confidence < 1:
        raise ValueError("margin and confidence must lie strictly between zero and one")


@lru_cache(maxsize=32)
def _bounds(total, confidence):
    counts = np.arange(total + 1)
    alpha = (1 - confidence) / 4
    lower = np.zeros(total + 1)
    upper = np.ones(total + 1)
    lower[1:] = beta.ppf(alpha, counts[1:], total - counts[1:] + 1)
    upper[:-1] = beta.isf(alpha, counts[:-1] + 1, total - counts[:-1])
    lower.flags.writeable = upper.flags.writeable = False
    return lower, upper


def noninferiority_power(total, favorable, adverse, *, margin=0.02, confidence=0.95):
    """Sum the multinomial acceptance probability without simulated outcomes.

    Condition on favorable discordances, then sum the remaining adverse binomial
    counts accepted by the interval. Conditioning here computes power; inference
    itself does not condition away the unknown total discordance probability.
    """
    _validate(total, favorable, adverse, margin, confidence)
    if favorable == 1:
        return float(
            paired_exact_envelope(
                PairedTable(0, total, 0, 0), confidence=confidence
            ).lower
            >= -margin
        )
    lower, upper = _bounds(total, confidence)
    counts = np.arange(total + 1)
    maximum_adverse = np.minimum(
        np.searchsorted(upper, lower + margin, side="right") - 1, total - counts
    )
    conditional = binom.cdf(maximum_adverse, total - counts, adverse / (1 - favorable))
    power = np.dot(binom.pmf(counts, total, favorable), conditional)
    return max(0.0, min(1.0, float(power)))
