"""Replay the frozen routing policy from paired, previously computed scores."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from . import fixed_cascade, gmm_monitor
from .proposed_routing_policy import FIRST_WINDOW_END, drift_override_requests


class PolicyReplayError(ValueError):
    """Cached predictions cannot form a complete, exactly aligned replay."""


@dataclass(frozen=True)
class PairedProbabilities:
    """A saved model pair; callers establish its artifact and row provenance."""

    record_id: str
    stage1_probability: float
    transformer_probability: float


@dataclass(frozen=True)
class MonitorScore:
    record_id: str
    negative_log_likelihood: float


@dataclass(frozen=True)
class RoutingTrace:
    record_id: str
    fixed_decision: int
    policy_decision: int
    logical_band: bool
    drift_override: bool
    logical_stage2_mask: bool


@dataclass(frozen=True)
class WindowTrace:
    """One complete window; positions are inclusive and one-based."""

    start_position: int
    end_position: int
    score: float
    alert: bool


@dataclass(frozen=True)
class PolicyReplay:
    rows: tuple[RoutingTrace, ...]
    windows: tuple[WindowTrace, ...]
    window_alert_fraction: float | None


def _validate_inputs(
    probabilities: Sequence[PairedProbabilities],
    monitor_scores: Sequence[MonitorScore],
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    if not all(
        isinstance(value, Sequence) and not isinstance(value, (str, bytes))
        for value in (probabilities, monitor_scores)
    ):
        raise PolicyReplayError("inputs must be materialized ordered sequences")
    if len(probabilities) != len(monitor_scores):
        raise PolicyReplayError("prediction and monitor counts must match")

    seen: set[str] = set()
    stage1 = []
    transformer = []
    nll = []
    for row, monitor in zip(probabilities, monitor_scores, strict=True):
        if type(row) is not PairedProbabilities or type(monitor) is not MonitorScore:
            raise PolicyReplayError("inputs must use typed score records")
        identity = row.record_id
        if (
            type(identity) is not str
            or not identity
            or any(
                character.isspace() or not character.isprintable()
                for character in identity
            )
        ):
            raise PolicyReplayError("record ID must be a nonempty stable string")
        if identity in seen:
            raise PolicyReplayError("record IDs must be unique")
        seen.add(identity)
        if type(monitor.record_id) is not str or monitor.record_id != identity:
            raise PolicyReplayError("prediction and monitor record IDs or order differ")

        pair = []
        for field in ("stage1_probability", "transformer_probability"):
            probability = fixed_cascade._finite_number(getattr(row, field), field)
            if not 0.0 <= probability <= 1.0:
                raise PolicyReplayError(f"{field} must be a probability in [0, 1]")
            pair.append(probability)
        stage1.append(pair[0])
        transformer.append(pair[1])
        nll.append(
            fixed_cascade._finite_number(
                monitor.negative_log_likelihood, "negative_log_likelihood"
            )
        )
    return tuple(stage1), tuple(transformer), tuple(nll)


def replay_policy(
    probabilities: Sequence[PairedProbabilities],
    monitor_scores: Sequence[MonitorScore],
    *,
    stage1_threshold: float,
    transformer_threshold: float,
    half_width: float,
    monitor_boundary: float,
) -> PolicyReplay:
    """Route the original label-blind stream before any outcome stratification.

    Every row already has both model scores. Masks describe logical selection,
    not skipped computation or measured transformer calls. This function checks
    alignment and values, not the upstream artifact/stream provenance. It never
    reorders, repairs, filters, fits, or recalibrates its inputs.
    """
    try:
        stage1_threshold = fixed_cascade._threshold(
            stage1_threshold, "stage1_threshold"
        )
        transformer_threshold = fixed_cascade._threshold(
            transformer_threshold, "transformer_threshold"
        )
        half_width = fixed_cascade._finite_number(half_width, "half_width")
        if half_width < 0.0:
            raise PolicyReplayError("half_width must be nonnegative")
        monitor_boundary = fixed_cascade._finite_number(
            monitor_boundary, "monitor_boundary"
        )
        stage1, transformer, nll = _validate_inputs(probabilities, monitor_scores)
        if not probabilities:
            return PolicyReplay(rows=(), windows=(), window_alert_fraction=None)

        fixed = fixed_cascade.score_fixed_cascade(
            stage1,
            transformer,
            stage1_threshold=stage1_threshold,
            transformer_threshold=transformer_threshold,
            half_width=half_width,
        )
        ends, scores = (
            gmm_monitor.window_scores(nll) if len(nll) >= FIRST_WINDOW_END else ((), ())
        )
    except fixed_cascade.FixedCascadeError as exc:
        raise PolicyReplayError(str(exc)) from exc
    except (gmm_monitor.GMMMonitorError, FloatingPointError) as exc:
        raise PolicyReplayError(f"complete-window scoring failed: {exc}") from exc

    windows = tuple(
        WindowTrace(
            start_position=end - FIRST_WINDOW_END + 1,
            end_position=end,
            score=score,
            alert=score > monitor_boundary,
        )
        for end, score in zip(ends, scores, strict=True)
    )
    overrides = set(
        drift_override_requests(
            len(probabilities),
            [window.end_position for window in windows if window.alert],
        )
    )
    rows = []
    for index, row in enumerate(probabilities):
        override = index + 1 in overrides
        band = fixed.transformer_invoked[index]
        selected = band or override
        rows.append(
            RoutingTrace(
                record_id=row.record_id,
                fixed_decision=fixed.decisions[index],
                policy_decision=int(transformer[index] >= transformer_threshold)
                if selected
                else fixed.decisions[index],
                logical_band=band,
                drift_override=override,
                logical_stage2_mask=selected,
            )
        )
    return PolicyReplay(
        rows=tuple(rows),
        windows=windows,
        window_alert_fraction=sum(window.alert for window in windows) / len(windows)
        if windows
        else None,
    )
