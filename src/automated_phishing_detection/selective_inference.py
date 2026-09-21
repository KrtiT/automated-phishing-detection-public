"""Singleton scoring shared by paired evaluation and future selective serving.

The singleton convention is adopted by singleton-inference-amendment-v1;
the original compatibility comparison remains not_equivalent.
This module neither reads research inputs nor supplies an HTTP measurement.
"""

from __future__ import annotations

import threading
from contextlib import ExitStack
from dataclasses import dataclass

import threadpoolctl
import torch

from . import fixed_cascade, gmm_monitor, transformer_scoring
from .transformer_inference import LoadedTransformerCascade, TransformerInferenceError

_PROCESS_SESSION = threading.Lock()


@dataclass(frozen=True)
class InferenceCounts:
    transformer_forward_attempts: int
    successful_transformer_scores: int
    completed_requests: int
    failed_requests: int


@dataclass(frozen=True)
class RequestScores:
    stage1_probability: float
    transformer_probability: float | None
    fixed_decision: int
    decision: int
    band_selected: bool
    drift_override: bool
    logical_stage2_selected: bool
    transformer_evaluated: bool
    stage1_scoring_audit: dict[str, object]


class SelectiveCascade:
    """One owner thread and one process-wide numerical context per session.

    A service must enqueue requests in its declared admission order for this
    owner; thread completion order is not a routing policy. CPU is allowed only
    through the explicit private fixture seam, not an official model loader.
    """

    def __init__(
        self, loaded: LoadedTransformerCascade, *, _fixture_cpu: bool = False
    ) -> None:
        self._loaded = loaded
        self._fixture_cpu = _fixture_cpu
        self._stack: ExitStack | None = None
        self._owner: int | None = None
        self._used = False
        self._forward_attempts = 0
        self._successful_scores = 0
        self._completed = 0
        self._failed = 0

    def __enter__(self) -> SelectiveCascade:
        if self._used or not _PROCESS_SESSION.acquire(blocking=False):
            raise TransformerInferenceError(
                "an inference session is already active or used"
            )
        stack = ExitStack()
        stack.callback(_PROCESS_SESSION.release)
        try:
            transformer_scoring._validate_model(self._loaded)
            if type(self._fixture_cpu) is not bool:
                raise TransformerInferenceError("fixture flag must be boolean")
            if torch.__version__.split("+", maxsplit=1)[0] != "2.7.1":
                raise TransformerInferenceError(
                    "singleton inference requires PyTorch 2.7.1"
                )
            if self._fixture_cpu:
                if self._loaded.device != torch.device("cpu"):
                    raise TransformerInferenceError(
                        "synthetic fixture device must be CPU"
                    )
            elif (
                self._loaded.device != torch.device("mps")
                or not torch.backends.mps.is_available()
            ):
                raise TransformerInferenceError(
                    "official singleton inference requires MPS"
                )
            try:
                gmm_monitor._require_runtime()
            except gmm_monitor.GMMMonitorError as exc:
                raise TransformerInferenceError(str(exc)) from exc
            # The first query initializes this owner's OpenMP state before limiting.
            torch.get_num_threads()
            stack.enter_context(threadpoolctl.threadpool_limits(limits=1))
            pools = threadpoolctl.threadpool_info()
            if (
                not pools
                or any(pool.get("num_threads") != 1 for pool in pools)
                or torch.get_num_threads() != 1
            ):
                raise TransformerInferenceError(
                    "runtime must honor one numerical thread"
                )
            stack.callback(
                torch.use_deterministic_algorithms,
                torch.are_deterministic_algorithms_enabled(),
                warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
            )
            torch.use_deterministic_algorithms(True, warn_only=False)
            self._owner = threading.get_ident()
            self._stack = stack
            self._used = True
            return self
        except BaseException:
            stack.close()
            raise

    def __exit__(self, exc_type, exc, traceback) -> bool:
        self._require_owner()
        stack, self._stack = self._stack, None
        self._owner = None
        return stack.__exit__(exc_type, exc, traceback)

    def _require_owner(self) -> None:
        if self._stack is None:
            raise TransformerInferenceError("scoring requires an open session")
        if threading.get_ident() != self._owner:
            raise TransformerInferenceError(
                "scoring must run on the session owner thread"
            )

    @property
    def counts(self) -> InferenceCounts:
        return InferenceCounts(
            self._forward_attempts,
            self._successful_scores,
            self._completed,
            self._failed,
        )

    def _score_transformer(self, raw_url: str) -> float:
        def record_attempt(model, inputs):
            self._forward_attempts += 1

        # Count entry to the actual forward, not selection or token preparation.
        hook = self._loaded._model.register_forward_pre_hook(record_attempt)
        try:
            values = transformer_scoring.score_transformer_urls(
                self._loaded, (raw_url,), batch_size=1
            )
        finally:
            hook.remove()
        self._successful_scores += 1
        return values[0]

    def _score(
        self, raw_url: str, *, drift_override: bool, force_transformer: bool
    ) -> RequestScores:
        self._require_owner()
        try:
            if type(drift_override) is not bool:
                raise TransformerInferenceError("drift_override must be boolean")
            transformer_scoring._validate_model(self._loaded)
            probabilities, audit = fixed_cascade.score_logistic_l1_authoritative(
                self._loaded.stage1_model, (raw_url,)
            )
            values = fixed_cascade._probability_vector(
                probabilities, "stage-one probabilities"
            )
            if len(values) != 1:
                raise TransformerInferenceError(
                    "singleton stage one must return one score"
                )
            stage1 = float(values[0])
            band = (
                abs(stage1 - self._loaded.stage1_threshold) <= self._loaded.half_width
            )
            selected = band or drift_override
            transformer = (
                self._score_transformer(raw_url)
                if selected or force_transformer
                else None
            )
            stage1_decision = int(stage1 >= self._loaded.stage1_threshold)
            stage2_decision = (
                int(transformer >= self._loaded.transformer_threshold)
                if transformer is not None
                else None
            )
            result = RequestScores(
                stage1_probability=stage1,
                transformer_probability=transformer,
                fixed_decision=stage2_decision if band else stage1_decision,
                decision=stage2_decision if selected else stage1_decision,
                band_selected=band,
                drift_override=drift_override,
                logical_stage2_selected=selected,
                transformer_evaluated=transformer is not None,
                stage1_scoring_audit=audit,
            )
        except BaseException:
            self._failed += 1
            raise
        self._completed += 1
        return result

    def scan(self, raw_url: str, *, drift_override: bool = False) -> RequestScores:
        """Invoke the transformer only for the inclusive band or a prior alert."""
        return self._score(
            raw_url, drift_override=drift_override, force_transformer=False
        )

    def score_all(self, raw_url: str) -> RequestScores:
        """Obtain both model scores through the same singleton inference path."""
        return self._score(raw_url, drift_override=False, force_transformer=True)
