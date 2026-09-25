"""Invented phase checkpoints for pure typed restoration tests."""

import importlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from external_composition_fixtures import _secondary_scorers
from external_producer_fixtures import prepared_external
from test_external_primary import external_session

from automated_phishing_detection import bound_secondary, evaluation_producer
from automated_phishing_detection.external_primary import (
    ExternalPrimaryScores,
    score_external_primary,
)
from automated_phishing_detection.external_producer import _all_scores
from automated_phishing_detection.external_secondary import (
    ProducedExternalSecondary,
    score_external_secondary,
)
from automated_phishing_detection.phishvn import PreparedExternal


def restore_module(suffix: str) -> ModuleType:
    name = f"automated_phishing_detection._saved_external_{suffix}"
    assert importlib.util.find_spec(name) is not None, "saved restoration missing"
    return importlib.import_module(name)


def _outputs(
    prepared: PreparedExternal,
    primary: ExternalPrimaryScores,
    secondary: ProducedExternalSecondary,
) -> dict[str, bytes]:
    return (
        prepared.private_outputs
        | {
            "preparation-summary.json": evaluation_producer._json_bytes(
                prepared.public_summary
            ),
            "primary-scores.jsonl": primary.checkpoint_bytes,
            "primary-completion.json": primary.receipt_bytes,
            "all-scores.jsonl": _all_scores(primary, secondary.scoring),
        }
        | secondary.private_outputs
    )


def phase_inputs(monkeypatch: pytest.MonkeyPatch, count: int = 5) -> SimpleNamespace:
    prepared = prepared_external(count, quarantine_count=1)
    session, unused = external_session(monkeypatch)
    primary = score_external_primary(prepared, session)
    _secondary_scorers(monkeypatch, [])
    bound = session.evaluation.secondary
    secondary = score_external_secondary(primary, bound)
    bindings = {
        "thresholds": dict(primary.thresholds),
        "secondary": evaluation_producer._secondary_binding(
            bound, bound.stage1_threshold
        ),
        "preparation": prepared.public_summary,
    }
    return SimpleNamespace(
        prepared=prepared,
        primary=primary,
        secondary=secondary.scoring,
        bound=bound,
        bindings=bindings,
        outputs=_outputs(prepared, primary, secondary),
    )


def rewrite(outputs: dict[str, bytes], name: str, path: tuple, value: object) -> None:
    payload = json.loads(outputs[name])
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    outputs[name] = evaluation_producer._json_bytes(payload)


def rewrite_row(
    outputs: dict[str, bytes], name: str, path: tuple, value: object
) -> None:
    lines = outputs[name].splitlines(keepends=True)
    temporary = {name: lines[0]}
    rewrite(temporary, name, path, value)
    outputs[name] = temporary[name] + b"".join(lines[1:])


def forbid_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("restoration attempted I/O or scoring")

    monkeypatch.setattr("builtins.open", forbidden)
    monkeypatch.setattr(Path, "read_bytes", forbidden)
    monkeypatch.setattr(Path, "read_text", forbidden)
    monkeypatch.setattr(bound_secondary, "score_bound_secondary", forbidden)
    monkeypatch.setattr(evaluation_producer, "score_primary_url", forbidden)
    monkeypatch.setattr(
        bound_secondary.secondary_transformer,
        "load_secondary_transformer_bytes",
        forbidden,
    )
