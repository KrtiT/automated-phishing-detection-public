"""Primary hypothesis gates from prepared saved evidence, without model execution.

The caller authenticates source/artifact identities and complete stream routing
before selecting outcome populations. This module validates alignment and gate
arithmetic, not provenance. Operational inputs are summaries of separately
verified measurements, not substitutes for a real HTTP experiment.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .paired_evaluation import (
    BinaryPrediction,
    EvaluationRecord,
    RecallDifference,
    paired_recall_difference,
)
from .saved_metrics import (
    DetectionMetrics,
    RateEstimate,
    control_alert_rate,
    detection_metrics,
)

MODELS = frozenset({"length_only", "logistic_l1", "cascade", "transformer", "policy"})
CONTRASTS = (
    ("internal", "logistic_l1", "length_only"),
    ("internal", "cascade", "logistic_l1"),
    ("gold", "logistic_l1", "length_only"),
    ("gold", "cascade", "logistic_l1"),
    ("gold", "policy", "cascade"),
    ("gold", "cascade", "transformer"),
)


class HypothesisEvaluationError(ValueError):
    """Evidence is malformed or has the wrong role in the primary evaluation."""


@dataclass(frozen=True)
class SavedPopulation:
    records: Sequence[EvaluationRecord]
    predictions: Mapping[str, Sequence[BinaryPrediction]]


@dataclass(frozen=True)
class SavedControls:
    """Tranco reference controls have no confirmatory binary outcome label."""

    record_ids: Sequence[str]
    predictions: Mapping[str, Sequence[BinaryPrediction]]


@dataclass(frozen=True)
class WindowCounts:
    alert_windows: int
    complete_windows: int


@dataclass(frozen=True)
class ReferenceInvocations:
    """Physical scorer counters for one full 10,000-record primary manifest."""

    request_count: int
    forward_attempts: int
    successful_scores: int
    completed_requests: int
    failed_requests: int


@dataclass(frozen=True)
class PrimaryHttpSummary:
    """Measured requests only; p95 must pool requests, not per-run quantiles.

    The producer enforces the frozen timeout, JSON/error construct, prevalence,
    warm-up exclusion and latency convention. No such execution is implied by
    constructing this value, and provenance is checked by the future runner.
    """

    run_request_counts: tuple[int, ...]
    concurrency: int
    request_errors: int
    pooled_p95_ms: float


@dataclass(frozen=True)
class GateResult:
    name: str
    status: str
    estimate: float | None
    threshold: float
    operator: str
    numerator: int | None = None
    denominator: int | None = None
    upper_95: float | None = None
    reason: str | None = None


@dataclass(frozen=True)
class HypothesisResult:
    decision: str
    complete: bool
    gates: tuple[GateResult, ...]


@dataclass(frozen=True)
class PrimaryEvaluation:
    metrics: dict[str, DetectionMetrics | RateEstimate]
    contrasts: dict[str, RecallDifference | None]
    hypotheses: dict[str, HypothesisResult]


def _counts(numerator: int, denominator: int) -> None:
    if (
        type(numerator) is not int
        or type(denominator) is not int
        or not 0 <= numerator <= denominator
    ):
        raise HypothesisEvaluationError("counts must be integers with 0 <= n <= d")


def _models(predictions: object, allowed: frozenset[str]) -> None:
    if not isinstance(predictions, Mapping) or not predictions:
        raise HypothesisEvaluationError("supplied populations need saved predictions")
    if not set(predictions) <= allowed:
        raise HypothesisEvaluationError("unknown model or model outside its population")


def _metrics(populations, controls):
    if not isinstance(populations, Mapping) or not set(populations) <= {
        "internal",
        "gold",
        "certified",
    }:
        raise HypothesisEvaluationError("unknown primary population")
    metrics = {}
    for role, population in populations.items():
        if type(population) is not SavedPopulation:
            raise HypothesisEvaluationError("use typed saved populations")
        allowed = MODELS - {"policy"} if role == "internal" else MODELS
        _models(population.predictions, allowed)
        for model, predictions in population.predictions.items():
            metrics[f"{role}.{model}"] = detection_metrics(
                population.records, predictions
            )
        # detection_metrics has already validated record types and binary labels.
        required_label = {"gold": 1, "certified": 0}.get(role)
        if required_label is not None and any(
            row.label != required_label for row in population.records
        ):
            raise HypothesisEvaluationError("outcome label does not match population")
    if controls is not None:
        if type(controls) is not SavedControls:
            raise HypothesisEvaluationError("use label-free saved controls")
        _models(controls.predictions, frozenset({"cascade", "transformer"}))
        for model, predictions in controls.predictions.items():
            metrics[f"tranco.{model}"] = control_alert_rate(
                controls.record_ids, predictions
            )
    external_ids: set[str] = set()
    strata = [
        {row.record_id for row in populations[role].records}
        for role in ("gold", "certified")
        if role in populations
    ]
    if controls is not None:
        strata.append(set(controls.record_ids))
    for identities in strata:
        if external_ids.intersection(identities):
            raise HypothesisEvaluationError("external outcome strata overlap")
        external_ids.update(identities)
    return metrics


def _contrasts(populations):
    contrasts = {}
    for role, candidate, reference in CONTRASTS:
        name = f"{role}.{candidate}_minus_{reference}"
        population = populations.get(role)
        if (
            population is None
            or not {candidate, reference} <= population.predictions.keys()
        ):
            contrasts[name] = None
            continue
        indices = [i for i, row in enumerate(population.records) if row.label == 1]
        contrasts[name] = paired_recall_difference(
            tuple(population.records[i] for i in indices),
            tuple(population.predictions[candidate][i] for i in indices),
            tuple(population.predictions[reference][i] for i in indices),
        )
    return contrasts


def _fraction_gate(
    name, counts, numerator_limit, denominator_limit, operator, upper=None
):
    threshold = numerator_limit / denominator_limit
    if counts is None:
        return GateResult(
            name, "pending", None, threshold, operator, reason="missing_evidence"
        )
    numerator, denominator = counts
    _counts(numerator, denominator)
    if not denominator:
        return GateResult(
            name,
            "not_estimable",
            None,
            threshold,
            operator,
            numerator,
            denominator,
            reason="zero_denominator",
        )
    left, right = denominator_limit * numerator, numerator_limit * denominator
    passed = {"<=": left <= right, ">=": left >= right, "<": left < right}[operator]
    return GateResult(
        name,
        "pass" if passed else "fail",
        numerator / denominator,
        threshold,
        operator,
        numerator,
        denominator,
        upper,
    )


def _fpr_gate(metrics, role, model):
    metric = metrics.get(f"{role}.{model}")
    rate = metric if role == "tranco" else metric.fpr if metric is not None else None
    name = f"{role}.{model}.{'alert_rate' if role == 'tranco' else 'fpr'}"
    return _fraction_gate(
        name,
        (rate.numerator, rate.denominator) if rate else None,
        1,
        100,
        "<=",
        rate.upper_95 if rate else None,
    )


def _recall_gate(name, difference, margin=0.0):
    operator = ">=" if margin == -0.02 else ">"
    if difference is None:
        return GateResult(
            name, "pending", None, margin, operator, reason="missing_evidence"
        )
    if difference.status == "not_estimable":
        return GateResult(
            name, "not_estimable", None, margin, operator, reason=difference.reason
        )
    lower = difference.lower
    passed = lower >= margin if operator == ">=" else lower > margin
    return GateResult(name, "pass" if passed else "fail", lower, margin, operator)


def _window_gate(name, counts, numerator_limit, denominator_limit, operator):
    if counts is not None and type(counts) is not WindowCounts:
        raise HypothesisEvaluationError("window evidence must use WindowCounts")
    return _fraction_gate(
        name,
        (counts.alert_windows, counts.complete_windows) if counts else None,
        numerator_limit,
        denominator_limit,
        operator,
    )


def _invocation_gate(reference):
    name = "reference_transformer_invocations"
    if reference is None:
        return _fraction_gate(name, None, 3, 10, "<=")
    if type(reference) is not ReferenceInvocations:
        raise HypothesisEvaluationError("invocation evidence must be physical counters")
    _counts(reference.forward_attempts, reference.request_count)
    _counts(reference.successful_scores, reference.forward_attempts)
    _counts(reference.completed_requests, reference.request_count)
    _counts(reference.failed_requests, reference.request_count)
    if (
        reference.request_count != 10000
        or reference.completed_requests + reference.failed_requests
        != reference.request_count
        or reference.forward_attempts - reference.successful_scores
        > reference.failed_requests
    ):
        raise HypothesisEvaluationError(
            "inconsistent primary reference execution counts"
        )
    if reference.failed_requests:
        return GateResult(
            name,
            "not_estimable",
            None,
            0.3,
            "<=",
            reason="reference_execution_incomplete",
        )
    return _fraction_gate(
        name, (reference.forward_attempts, reference.request_count), 3, 10, "<="
    )


def _http_gates(http):
    latency_name = "http_pooled_p95_ms"
    if http is None:
        return (
            GateResult(
                latency_name, "pending", None, 200.0, "<=", reason="missing_evidence"
            ),
            _fraction_gate("http_request_errors", None, 1, 1000, "<"),
        )
    if type(http) is not PrimaryHttpSummary:
        raise HypothesisEvaluationError(
            "HTTP evidence must be a primary measured summary"
        )
    if (
        type(http.run_request_counts) is not tuple
        or len(http.run_request_counts) != 5
        or any(type(n) is not int or n != 10000 for n in http.run_request_counts)
        or type(http.concurrency) is not int
        or http.concurrency != 64
    ):
        raise HypothesisEvaluationError(
            "HTTP primary evidence requires five measured c64 runs of 10000"
        )
    _counts(http.request_errors, 50000)
    latency = http.pooled_p95_ms
    if type(latency) not in (int, float) or not math.isfinite(latency) or latency < 0:
        raise HypothesisEvaluationError(
            "pooled latency must be finite nonnegative milliseconds"
        )
    return (
        GateResult(
            latency_name,
            "pass" if latency <= 200.0 else "fail",
            float(latency),
            200.0,
            "<=",
        ),
        _fraction_gate(
            "http_request_errors", (http.request_errors, 50000), 1, 1000, "<"
        ),
    )


def _hypothesis(gates):
    gates = tuple(gates)
    complete = all(gate.status in {"pass", "fail"} for gate in gates)
    decision = (
        "not_supported"
        if any(gate.status == "fail" for gate in gates)
        else "supported"
        if complete
        else "undecided"
    )
    return HypothesisResult(decision, complete, gates)


def evaluate_primary(
    *,
    populations: Mapping[str, SavedPopulation] | None = None,
    controls: SavedControls | None = None,
    external_windows: WindowCounts | None = None,
    audit_windows: WindowCounts | None = None,
    reference: ReferenceInvocations | None = None,
    http: PrimaryHttpSummary | None = None,
) -> PrimaryEvaluation:
    """Compute the six frozen contrasts and all primary conjunctive gates.

    Missing populations/models/summaries remain pending. Supplied empty strata
    are non-estimable; malformed evidence raises rather than being dropped.
    Final FPR gates use observed rates, not their reported confidence bounds.
    Route the entire external stream before supplying its gold/certified strata.
    No inference is performed and no historical result is implicitly loaded.
    """
    populations = {} if populations is None else populations
    metrics = _metrics(populations, controls)
    contrasts = _contrasts(populations)
    h1 = [
        _fpr_gate(metrics, role, model)
        for role in ("internal", "certified")
        for model in ("length_only", "logistic_l1", "cascade")
    ]
    h1.extend(_recall_gate(name, contrasts[name]) for name in tuple(contrasts)[:4])
    h2 = [
        _window_gate("external_window_alerts", external_windows, 4, 5, ">="),
        _window_gate("audit_window_alerts", audit_windows, 1, 20, "<="),
        _fpr_gate(metrics, "certified", "policy"),
        _recall_gate(
            "gold.policy_minus_cascade", contrasts["gold.policy_minus_cascade"]
        ),
    ]
    h3 = [
        _fpr_gate(metrics, role, model)
        for role in ("certified", "tranco")
        for model in ("cascade", "transformer")
    ]
    h3.extend(
        [
            _recall_gate(
                "gold.cascade_minus_transformer",
                contrasts["gold.cascade_minus_transformer"],
                -0.02,
            ),
            _invocation_gate(reference),
            *_http_gates(http),
        ]
    )
    return PrimaryEvaluation(
        metrics,
        contrasts,
        {"H1": _hypothesis(h1), "H2": _hypothesis(h2), "H3": _hypothesis(h3)},
    )
