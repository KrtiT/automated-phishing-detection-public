"""Reduce already verified saved components into the unchanged study gates.

These caller-constructible inputs receive structural checks only. This reducer
does not authenticate source provenance, measured execution, or authorization.
Missing components remain missing; no files or models are accessed.
"""

from dataclasses import dataclass

from . import hypothesis_evaluation, secondary_metrics
from .evaluation_stream import ExternalEvidence
from .hypothesis_evaluation import (
    PrimaryEvaluation,
    PrimaryHttpSummary,
    ReferenceInvocations,
    SavedControls,
    SavedPopulation,
    WindowCounts,
)
from .secondary_metrics import HolmFamily


class StudyEvidenceError(ValueError):
    """Supplied study evidence failed structural validation."""


@dataclass(frozen=True)
class StudyEvidence:
    primary: PrimaryEvaluation
    ablation_family: HolmFamily


def _populations(internal, external):
    populations = {}
    if internal is not None:
        if type(internal) is not SavedPopulation:
            raise StudyEvidenceError("invalid_internal_population")
        populations["internal"] = internal
    if external is not None:
        if (
            type(external) is not ExternalEvidence
            or type(external.populations) is not dict
            or set(external.populations) != {"gold", "certified"}
            or type(external.controls) is not SavedControls
            or type(external.external_windows) is not WindowCounts
        ):
            raise StudyEvidenceError("invalid_external_evidence")
        populations.update(external.populations)
    return populations


def _positive_mcnemar(population, candidate, reference):
    if (
        population is None
        or not {candidate, reference} <= population.predictions.keys()
    ):
        return None
    indices = tuple(
        index for index, row in enumerate(population.records) if row.label == 1
    )
    return secondary_metrics.exact_mcnemar(
        tuple(population.records[index] for index in indices),
        tuple(population.predictions[candidate][index] for index in indices),
        tuple(population.predictions[reference][index] for index in indices),
    )


def _ablation_family(populations):
    contrasts = {}
    for role, prefix in (("internal", "internal"), ("gold", "external_gold")):
        population = populations.get(role)
        contrasts[f"{prefix}_logistic_minus_length"] = _positive_mcnemar(
            population, "logistic_l1", "length_only"
        )
        contrasts[f"{prefix}_cascade_minus_logistic"] = _positive_mcnemar(
            population, "cascade", "logistic_l1"
        )
    return secondary_metrics.holm_ablation_family(contrasts)


def reduce_study_evidence(
    *,
    internal: SavedPopulation | None = None,
    external: ExternalEvidence | None = None,
    reference: ReferenceInvocations | None = None,
    http: PrimaryHttpSummary | None = None,
) -> StudyEvidence:
    """Combine verified components, retaining the fixed audit and all Holm slots."""
    try:
        populations = _populations(internal, external)
        primary = hypothesis_evaluation.evaluate_primary(
            populations=populations,
            controls=external.controls if external is not None else None,
            external_windows=external.external_windows
            if external is not None
            else None,
            audit_windows=WindowCounts(28, 252),
            reference=reference,
            http=http,
        )
        return StudyEvidence(primary, _ablation_family(populations))
    except StudyEvidenceError:
        raise
    except (ValueError, TypeError, KeyError, AttributeError, OverflowError):
        raise StudyEvidenceError("invalid_study_evidence") from None
