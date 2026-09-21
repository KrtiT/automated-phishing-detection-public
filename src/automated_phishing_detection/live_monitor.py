"""Bounded, future-only live monitoring over an active singleton scorer.

The owner service supplies admission order, phase fences and complete traces.
This core does not authenticate request identities, authorize a warmup reset,
or turn a synthetic execution into research evidence.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from . import fixed_cascade, gmm_monitor
from .fixed_cascade import PortableLogisticL1
from .policy_replay import WindowTrace
from .proposed_routing_policy import FIRST_WINDOW_END, ROUTING_HORIZON, WINDOW_STRIDE
from .selective_inference import InferenceCounts, RequestScores, SelectiveCascade
from .url_features import extract_url_features


class LiveMonitorError(ValueError):
    """The live stream cannot continue without losing its frozen ordering."""


@dataclass(frozen=True)
class LiveRequestScores:
    scores: RequestScores
    monitor_nll: float
    position: int
    window: WindowTrace | None


class LiveMonitor:
    """Route using prior alerts, then score this row's singleton monitor feature.

    At most 256 NLLs and one activation endpoint are retained. A computation
    failure permanently poisons the stream; it cannot be reset or resumed.
    The supplied SelectiveCascade owns the numerical session and counters.
    """

    def __init__(
        self,
        active_scorer: SelectiveCascade,
        *,
        stage1_model: PortableLogisticL1,
        gmm: dict,
        boundary: float,
    ) -> None:
        active_scorer._require_owner()
        try:
            self._boundary = fixed_cascade._finite_number(boundary, "boundary")
        except fixed_cascade.FixedCascadeError as exc:
            raise LiveMonitorError(str(exc)) from exc
        self._scorer = active_scorer
        self._stage1_model = stage1_model
        self._gmm = gmm_monitor.load_gmm_artifact_bytes(
            gmm_monitor._canonical_json_bytes(gmm)
        )
        self._nlls: deque[float] = deque(maxlen=FIRST_WINDOW_END)
        self._position = 0
        self._override_through = 0
        self._poisoned = False

    @property
    def counts(self) -> InferenceCounts:
        return self._scorer.counts

    def _require_ready(self) -> None:
        self._scorer._require_owner()
        if self._poisoned:
            raise LiveMonitorError("live monitor is poisoned by a prior failure")

    def reset_monitor(self) -> None:
        """Clear only monitor state after the owner service authorizes a reset."""
        self._require_ready()
        self._nlls.clear()
        self._position = 0
        self._override_through = 0

    def scan_shift(self, raw_url: str) -> LiveRequestScores:
        """Complete one row before allowing its window to affect future rows."""
        self._require_ready()
        try:
            position = self._position + 1
            scores = self._scorer.scan(
                raw_url, drift_override=position <= self._override_through
            )
            # Preserve the original portable monitor feature, not the routing score.
            features = gmm_monitor._build_features(
                {
                    "features": np.asarray(
                        [extract_url_features(raw_url)], dtype=np.float64
                    ),
                    "raw_urls": (raw_url,),
                },
                self._stage1_model,
            )
            nll = float(
                gmm_monitor._finite_array(
                    gmm_monitor.score_feature_matrix(features, self._gmm),
                    (1,),
                    "singleton monitor NLL",
                )[0]
            )
            self._nlls.append(nll)
            window = None
            if (
                position >= FIRST_WINDOW_END
                and (position - FIRST_WINDOW_END) % WINDOW_STRIDE == 0
            ):
                # Reuse the exact float64 reduction, never a rolling sum.
                _, means = gmm_monitor.window_scores(tuple(self._nlls))
                score = means[0]
                window = WindowTrace(
                    position - FIRST_WINDOW_END + 1,
                    position,
                    score,
                    score > self._boundary,
                )
                if window.alert:
                    self._override_through = position + ROUTING_HORIZON
            self._position = position
            return LiveRequestScores(scores, nll, position, window)
        except BaseException:
            self._poisoned = True
            raise
