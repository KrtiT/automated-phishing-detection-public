"""Invented bound-secondary scorers for column retention tests."""

from functools import partial
from hashlib import sha256
from types import SimpleNamespace

import pytest
from test_bound_secondary import SEEDS, TABULAR_NAMES, _scoring_bound

from automated_phishing_detection import bound_secondary as secondary


def _values(state, phase, member, urls):
    state.events.append((phase, member))
    values = []
    for index, unused_url in enumerate(urls):
        state.singletons.append((phase, member, index))
        if state.fail_at == (phase, member) and index == 1:
            if state.invalid_output:
                return (0.2, float("nan"))
            raise ValueError("invented member failure")
        values.append((0.25, 0.75)[index])
    return tuple(values)


def _tabular_score(state, model, urls):
    return _values(state, "tabular", model.artifact_bytes.decode("ascii"), urls)


def _load_seed(state, weights, vocabulary, *, seed, device):
    state.events.append(("load", seed))
    return SimpleNamespace(
        seed=seed,
        weights_sha256=sha256(weights).hexdigest(),
        vocabulary_sha256=sha256(vocabulary).hexdigest(),
        device=device,
    )


def _score_seed(state, loaded, urls):
    return _values(state, "score", loaded.seed, urls)


def _cascade(state, original, *args, **kwargs):
    seed = next(
        member.seed
        for member in state.bound.seeds
        if member.transformer_threshold == kwargs["transformer_threshold"]
    )
    state.events.append(("cascade", seed))
    return original(*args, **kwargs)


def _install_scorers(state, monkeypatch):
    monkeypatch.setattr(
        secondary.SecondaryModel,
        "score_urls_singleton_ordered",
        lambda model, urls: _tabular_score(state, model, urls),
    )
    monkeypatch.setattr(
        secondary.secondary_transformer,
        "load_secondary_transformer_bytes",
        partial(_load_seed, state),
    )
    monkeypatch.setattr(
        secondary.secondary_transformer,
        "score_secondary_transformer_urls",
        partial(_score_seed, state),
    )
    monkeypatch.setattr(
        secondary.fixed_cascade,
        "score_fixed_cascade",
        partial(_cascade, state, secondary.fixed_cascade.score_fixed_cascade),
    )


@pytest.fixture
def scoring(monkeypatch):
    state = SimpleNamespace(
        bound=_scoring_bound(secondary),
        urls=("https://first.example/path", "https://second.example/path"),
        stage1=(0.5, 0.7),
        seed42=(0.41, 0.1),
        events=[],
        singletons=[],
        retained=[],
        fail_at=None,
        invalid_output=False,
    )
    _install_scorers(state, monkeypatch)
    return state


def identity(column):
    return column.name if hasattr(column, "name") else column.seed


def retain(state, column):
    state.events.append(("completed", identity(column)))
    state.retained.append(column)


def score(state, callback):
    return secondary.score_bound_secondary(
        state.bound,
        state.urls,
        state.stage1,
        state.seed42,
        on_completed_column=callback,
    )


def expected_events():
    events = []
    for name in TABULAR_NAMES:
        events.extend((("tabular", name), ("completed", name)))
    for seed in SEEDS:
        if seed != 42:
            events.extend((("load", seed), ("score", seed)))
        events.extend((("cascade", seed), ("completed", seed)))
    return events
