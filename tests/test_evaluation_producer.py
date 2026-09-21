"""Synthetic source bytes and no-fit composition; no research input is opened."""

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from automated_phishing_detection.bound_runtime import BoundSession
from automated_phishing_detection.phiusiil import canonicalize_url, record_id_for_row
from automated_phishing_detection.protocol_preflight import parse_suffix_rules
from automated_phishing_detection.selective_inference import (
    InferenceCounts,
    RequestScores,
    SelectiveCascade,
)
from automated_phishing_detection.transformer_inference import TransformerInferenceError

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "a" * 64
PSL = "b" * 64


@pytest.fixture
def producer():
    assert (
        ROOT / "src/automated_phishing_detection/evaluation_producer.py"
    ).is_file(), "missing single-pass evaluation composition"
    from automated_phishing_detection import evaluation_producer

    return evaluation_producer


def encoded(rows):
    return b"".join(
        (
            json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode("ascii")
        for row in rows
    )


def source_rows(count=4):
    rows = []
    for index in range(count):
        url = f"https://host{index}.example{index}.com/path"
        rows.append(
            {
                "record_id": record_id_for_row(SOURCE, index + 1),
                "raw_url": url,
                "canonical_url_sha256": sha256(
                    canonicalize_url(url).encode("utf-8")
                ).hexdigest(),
                "registrable_domain": f"example{index}.com",
                "is_phishing": index % 2,
                "split": "group_test",
            }
        )
    return rows


def parse(producer, rows=None, content=None, **overrides):
    rows = source_rows() if rows is None else rows
    content = encoded(rows) if content is None else content
    kwargs = {
        "expected_sha256": sha256(content).hexdigest(),
        "source_csv_sha256": SOURCE,
        "suffix_rules": parse_suffix_rules("com\nco.uk\n"),
        "suffix_rules_sha256": PSL,
        "expected_row_count": len(rows),
        "expected_domain_count": len({row["registrable_domain"] for row in rows}),
        "expected_class_counts": {
            "0": sum(row["is_phishing"] == 0 for row in rows),
            "1": sum(row["is_phishing"] == 1 for row in rows),
        },
    }
    kwargs.update(overrides)
    return producer.parse_internal_partition(content, **kwargs)


def test_parser_authenticates_exact_bytes_once_and_preserves_order(
    producer, monkeypatch
):
    content = encoded(source_rows())
    original = producer.sha256
    observed = []

    def digest(value):
        if value is content:
            observed.append(value)
        return original(value)

    monkeypatch.setattr(producer, "sha256", digest)
    prepared = parse(producer, content=content)
    assert observed == [content]
    assert [asdict(row) for row in prepared.records] == source_rows()
    assert prepared.partition_sha256 == sha256(content).hexdigest()
    assert prepared.source_csv_sha256 == SOURCE
    assert prepared.suffix_rules_sha256 == PSL


def test_bad_hash_precedes_decoding_and_any_scoring(producer, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("bad source reached parsing or scoring")

    monkeypatch.setattr(producer.json, "loads", forbidden)
    monkeypatch.setattr(
        producer.length_inference, "score_length_only_authoritative", forbidden
    )
    with pytest.raises(producer.EvaluationProducerError, match="SHA-256"):
        parse(producer, content=b"not JSON", expected_sha256="c" * 64)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda rows: rows.reverse(),
        lambda rows: rows.append(dict(rows[0])),
        lambda rows: rows[0].update(split="train"),
        lambda rows: rows[0].update(is_phishing=True),
        lambda rows: rows[0].update(record_id=record_id_for_row("c" * 64, 1)),
        lambda rows: rows[0].update(
            record_id=f"phiusiil-row-v1:{SOURCE}:0000000000000000"
        ),
        lambda rows: rows[0].update(canonical_url_sha256="d" * 64),
        lambda rows: rows[0].update(registrable_domain="com"),
        lambda rows: rows[0].update(extra="not allowed"),
        lambda rows: rows[1].update(
            raw_url=rows[0]["raw_url"],
            canonical_url_sha256=rows[0]["canonical_url_sha256"],
            registrable_domain=rows[0]["registrable_domain"],
        ),
    ],
)
def test_parser_rejects_metadata_not_repairs_it(producer, mutation):
    rows = source_rows()
    mutation(rows)
    with pytest.raises(producer.EvaluationProducerError):
        parse(producer, rows)


@pytest.mark.parametrize(
    "content",
    [
        b"",
        b"\xff\n",
        b"{}\n",
        encoded(source_rows()).rstrip(b"\n"),
        encoded(source_rows()) + b"\n",
        encoded(source_rows()).replace(
            b'"split":"group_test"', b'"split":"group_test","split":"group_test"'
        ),
        encoded(source_rows()).replace(b'"is_phishing":0', b'"is_phishing":NaN'),
        json.dumps(source_rows()[0]).encode() + b"\n",
    ],
)
def test_parser_requires_canonical_finite_exact_jsonl(producer, content):
    with pytest.raises(producer.EvaluationProducerError):
        parse(producer, content=content)


@pytest.mark.parametrize(
    "override",
    [
        {"expected_row_count": 3},
        {"expected_domain_count": 3},
        {"expected_class_counts": {"0": 3, "1": 1}},
        {"expected_row_count": True},
        {"suffix_rules_sha256": "not-a-hash"},
        {"source_csv_sha256": "A" * 64},
        {"suffix_rules": None},
    ],
)
def test_parser_rejects_inconsistent_declared_counts_and_pins(producer, override):
    with pytest.raises(producer.EvaluationProducerError):
        parse(producer, **override)


def test_parser_recomputes_psl_domain_not_only_hostname_relationship(producer):
    rows = source_rows()
    rows[0].update(
        raw_url="https://host.example.co.uk/path", registrable_domain="co.uk"
    )
    rows[0]["canonical_url_sha256"] = sha256(
        canonicalize_url(rows[0]["raw_url"]).encode("utf-8")
    ).hexdigest()
    with pytest.raises(producer.EvaluationProducerError, match="supplied PSL"):
        parse(producer, rows)


class SyntheticScorer:
    def __init__(self):
        self.urls = []
        self.fail_at = None
        self.extra_attempt = False
        self.owner = threading.get_ident()

    def _require_owner(self):
        if threading.get_ident() != self.owner:
            raise TransformerInferenceError("synthetic session owner differs")

    @property
    def counts(self):
        count = len(self.urls)
        return InferenceCounts(count + int(self.extra_attempt), count, count, 0)

    def score_all(self, raw_url):
        index = len(self.urls)
        self.urls.append(raw_url)
        if index == self.fail_at:
            raise RuntimeError("synthetic scorer failure")
        first = (0.1, 0.8, 0.6, 0.4)[index % 4]
        second = (0.9, 0.7, 0.2, 0.9)[index % 4]
        band = abs(first - 0.5) <= 0.11
        decision = int((second if band else first) >= 0.5)
        return RequestScores(
            first,
            second,
            decision,
            decision,
            band,
            False,
            band,
            True,
            {"synthetic_stage1_audit": True},
        )


def synthetic_session(producer, monkeypatch):
    length_calls, portable_calls, gmm_calls = [], [], []

    def length_score(model, urls):
        assert type(urls) is tuple and len(urls) == 1
        index = len(length_calls)
        length_calls.append(urls)
        return ((0.1, 0.2, 0.1, 0.9)[index % 4],), {"length_audit": True}

    def portable(urls):
        assert type(urls) is tuple and len(urls) == 1
        portable_calls.append(urls)
        return (0.33,)

    def gmm(matrix, artifact):
        assert len(matrix) == 1 and len(matrix[0]) == 26
        assert matrix[0][-1] == 0.33
        gmm_calls.append(tuple(matrix[0]))
        return (2.0,)

    monkeypatch.setattr(
        producer.length_inference, "score_length_only_authoritative", length_score
    )
    monkeypatch.setattr(producer.gmm_monitor, "score_feature_matrix", gmm)
    models = SimpleNamespace(
        length_only=SimpleNamespace(
            validation_threshold_record={"status": "selected", "threshold": 0.5}
        ),
        cascade=SimpleNamespace(
            stage1_model=SimpleNamespace(score_urls=portable),
            stage1_threshold=0.5,
            transformer_threshold=0.5,
            half_width=0.11,
        ),
        gmm={"synthetic": True},
        monitor_boundary=3.0,
        artifact_hashes=(("synthetic.json", "e" * 64),),
    )
    session = BoundSession(models, SyntheticScorer())
    return session, length_calls, portable_calls, gmm_calls


def test_singleton_composition_retains_all_rows_scores_and_monitor_path(
    producer, monkeypatch
):
    prepared = parse(producer)
    session, lengths, portable, gmm = synthetic_session(producer, monkeypatch)
    result = producer.produce_internal_evidence(prepared, session)
    assert len(result.rows) == len(lengths) == len(portable) == len(gmm) == 4
    assert session.scorer.urls == [row.raw_url for row in prepared.records]
    assert result.inference_counts == InferenceCounts(4, 4, 4, 0)
    assert [row.stage1_probability for row in result.rows] == [0.1, 0.8, 0.6, 0.4]
    assert [row.monitor_probability for row in result.rows] == [0.33] * 4
    assert [row.cascade_probability for row in result.rows] == [0.1, 0.8, 0.2, 0.9]
    assert [row.cascade_decision for row in result.rows] == [0, 1, 0, 1]
    assert all(len(row.features) == 25 for row in result.rows)
    assert all(row.negative_log_likelihood == 2.0 for row in result.rows)
    assert all(
        json.loads(row.length_scoring_audit_json) == {"length_audit": True}
        for row in result.rows
    )
    assert [row.record_id for row in result.population.records] == [
        row.record_id for row in prepared.records
    ]
    assert result.primary.metrics["internal.cascade"].true_positives == 2
    assert result.primary.metrics["internal.logistic_l1"].false_positives == 1
    assert result.primary.hypotheses["H2"].decision == "not_supported"
    assert result.primary.hypotheses["H2"].complete is False
    h2 = {gate.name: gate for gate in result.primary.hypotheses["H2"].gates}
    assert (
        h2["audit_window_alerts"].numerator,
        h2["audit_window_alerts"].denominator,
    ) == (28, 252)
    assert h2["external_window_alerts"].status == "pending"
    assert result.primary.hypotheses["H3"].decision == "undecided"
    assert set(result.manifests) == {10, 100, 500}
    assert all(
        value.status == "insufficient_capacity" for value in result.manifests.values()
    )


def test_private_serialization_is_deterministic_and_public_summary_is_aggregate_only(
    producer, monkeypatch
):
    prepared = parse(producer)
    session, *_ = synthetic_session(producer, monkeypatch)
    first = producer.produce_internal_evidence(prepared, session)
    session, *_ = synthetic_session(producer, monkeypatch)
    second = producer.produce_internal_evidence(prepared, session)
    assert first.private_outputs == second.private_outputs
    assert first.public_summary == second.public_summary
    evidence = [
        json.loads(line)
        for line in first.private_outputs["predictions.jsonl"].splitlines()
    ]
    assert len(evidence) == 4
    assert evidence[0]["record"]["raw_url"] == prepared.records[0].raw_url
    summary = json.dumps(first.public_summary)
    for row in prepared.records:
        assert row.raw_url not in summary
        assert row.registrable_domain not in summary
        assert row.record_id not in summary
    assert first.public_summary["protected_evaluation_authorized"] is False
    assert first.public_summary["source_binding"] == "caller_supplied_pins_only"
    assert "reference_invocations" not in first.public_summary
    assert all(
        sha256(content).hexdigest() == first.public_summary["private_sha256"][name]
        for name, content in first.private_outputs.items()
    )


def test_failure_returns_no_partially_accepted_output_or_reusable_session(
    producer, monkeypatch
):
    prepared = parse(producer)
    session, *_ = synthetic_session(producer, monkeypatch)
    session.scorer.fail_at = 2
    with pytest.raises(RuntimeError, match="synthetic scorer failure"):
        producer.produce_internal_evidence(prepared, session)
    assert len(session.scorer.urls) == 3
    with pytest.raises(producer.EvaluationProducerError, match="fresh"):
        producer.produce_internal_evidence(prepared, session)


def test_tampered_prepared_rows_and_nonfresh_counts_fail_before_scoring(
    producer, monkeypatch
):
    prepared = parse(producer)
    session, lengths, *_ = synthetic_session(producer, monkeypatch)
    with pytest.raises(producer.EvaluationProducerError):
        producer.produce_internal_evidence(
            replace(prepared, records=tuple(reversed(prepared.records))), session
        )
    assert session.scorer.urls == lengths == []
    session.scorer.extra_attempt = True
    with pytest.raises(producer.EvaluationProducerError, match="fresh"):
        producer.produce_internal_evidence(prepared, session)


@pytest.mark.parametrize("failure", ["nonfinite", "bad_counter", "wrong_decision"])
def test_invalid_scorer_outputs_fail_without_accepted_result(
    producer, monkeypatch, failure
):
    prepared = parse(producer)
    session, *_ = synthetic_session(producer, monkeypatch)
    original = session.scorer.score_all

    def invalid(url):
        scores = original(url)
        if failure == "nonfinite":
            return replace(scores, stage1_probability=float("nan"))
        if failure == "wrong_decision":
            return replace(scores, fixed_decision=1 - scores.fixed_decision)
        session.scorer.extra_attempt = True
        return scores

    monkeypatch.setattr(session.scorer, "score_all", invalid)
    with pytest.raises(producer.EvaluationProducerError):
        producer.produce_internal_evidence(prepared, session)


def test_real_gmm_output_dtype_is_accepted(producer, monkeypatch):
    prepared = parse(producer)
    session, *_ = synthetic_session(producer, monkeypatch)
    monkeypatch.setattr(
        producer.gmm_monitor,
        "score_feature_matrix",
        lambda matrix, artifact: np.array([2.0], dtype=np.float64),
    )
    result = producer.produce_internal_evidence(prepared, session)
    assert all(row.negative_log_likelihood == 2.0 for row in result.rows)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), True, "2.0"])
def test_malformed_gmm_scores_cannot_be_accepted(producer, monkeypatch, bad):
    prepared = parse(producer)
    session, *_ = synthetic_session(producer, monkeypatch)
    monkeypatch.setattr(
        producer.gmm_monitor, "score_feature_matrix", lambda matrix, artifact: (bad,)
    )
    with pytest.raises(producer.EvaluationProducerError):
        producer.produce_internal_evidence(prepared, session)


def test_all_three_full_manifests_are_serialized_with_frozen_counts(
    producer, monkeypatch
):
    rows = source_rows(10500)
    for index, row in enumerate(rows):
        row["is_phishing"] = int(index >= 10000)
    prepared = parse(producer, rows)
    session, *_ = synthetic_session(producer, monkeypatch)
    result = producer.produce_internal_evidence(prepared, session)
    saved = json.loads(result.private_outputs["manifests.json"])
    for prevalence in (10, 100, 500):
        outcome = result.manifests[prevalence]
        assert outcome.status == "prepared"
        assert len(outcome.manifest.records) == 10000
        assert len(outcome.manifest.warmup_records) == 1000
        assert sum(row.is_phishing for row in outcome.manifest.records) == prevalence
        assert len({row.record_id for row in outcome.manifest.records}) == 10000
        assert saved[str(prevalence)]["manifest"]["sha256"] == outcome.manifest.sha256
        summary = result.public_summary["manifests"][str(prevalence)]
        assert summary["measured_count"] == 10000
        assert "records" not in summary
    assert result.inference_counts == InferenceCounts(10500, 10500, 10500, 0)


@pytest.mark.parametrize("state", ["unopened", "closed", "wrong_thread"])
def test_actual_session_owner_guard_precedes_every_length_score(
    producer, monkeypatch, tmp_path, state
):
    from test_transformer_inference import _build_fixture, _load_fixture

    prepared = parse(producer)
    synthetic, lengths, *_ = synthetic_session(producer, monkeypatch)
    # Exercise real lifecycle/owner guards without asserting this test host's BLAS.
    monkeypatch.setattr(producer.gmm_monitor, "_require_runtime", lambda: None)
    scorer = SelectiveCascade(
        _load_fixture(_build_fixture(tmp_path)), _fixture_cpu=True
    )
    session = BoundSession(synthetic.models, scorer)

    def rejected():
        with pytest.raises(TransformerInferenceError):
            producer.produce_internal_evidence(prepared, session)
        assert lengths == []
        assert scorer.counts == InferenceCounts(0, 0, 0, 0)

    if state == "unopened":
        rejected()
    elif state == "closed":
        with scorer:
            pass
        rejected()
    else:
        with scorer, ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(rejected).result()
