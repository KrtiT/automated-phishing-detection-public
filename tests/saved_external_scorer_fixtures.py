"""Invented transformer observations with unpatched authoritative stage-one math."""

import threading
from types import SimpleNamespace

from automated_phishing_detection import fixed_cascade
from automated_phishing_detection.selective_inference import (
    InferenceCounts,
    RequestScores,
)


class FixturePrimaryScorer:
    def __init__(self, cascade: SimpleNamespace) -> None:
        self.cascade = cascade
        self.urls: list[str] = []
        self.owner = threading.get_ident()

    def _require_owner(self) -> None:
        assert threading.get_ident() == self.owner

    @property
    def counts(self) -> InferenceCounts:
        count = len(self.urls)
        return InferenceCounts(count, count, count, 0)

    def score_all(self, raw_url: str) -> RequestScores:
        self._require_owner()
        probabilities, audit = fixed_cascade.score_logistic_l1_authoritative(
            self.cascade.stage1_model, (raw_url,)
        )
        transformer = (0.15, 0.85)[len(self.urls) % 2]
        scores = fixed_cascade.score_fixed_cascade(
            probabilities,
            (transformer,),
            stage1_threshold=self.cascade.stage1_threshold,
            transformer_threshold=self.cascade.transformer_threshold,
            half_width=self.cascade.half_width,
        )
        self.urls.append(raw_url)
        return RequestScores(
            probabilities[0],
            transformer,
            scores.decisions[0],
            scores.decisions[0],
            scores.transformer_invoked[0],
            False,
            scores.transformer_invoked[0],
            True,
            audit,
        )
