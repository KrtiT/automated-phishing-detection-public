"""The shared primary scorer needs raw URLs, never labels for external controls."""

from dataclasses import asdict, fields

from test_evaluation_producer import parse, synthetic_session
from test_evaluation_producer import producer as producer


def test_url_only_scoring_requires_no_record_or_label(producer, monkeypatch):
    assert hasattr(producer, "score_primary_url"), "missing label-independent scoring"
    session, lengths, portable, gmm = synthetic_session(producer, monkeypatch)
    scores = producer.score_primary_url(
        "https://host0.example0.com/path",
        session.primary,
        producer._thresholds(session.primary),
        1,
    )
    assert not {"record", "is_phishing", "role"} & {
        field.name for field in fields(scores)
    }
    assert len(lengths) == len(portable) == len(gmm) == 1
    assert scores.length_probability == scores.stage1_probability == 0.1
    assert scores.transformer_probability == 0.9
    assert scores.monitor_probability == 0.33
    assert scores.negative_log_likelihood == 2.0
    assert asdict(scores.inference_counts) == asdict(
        producer.InferenceCounts(1, 1, 1, 0)
    )


def test_internal_adapter_preserves_every_existing_primary_field(producer, monkeypatch):
    assert hasattr(producer, "score_primary_url"), "missing label-independent scoring"
    records = parse(producer).records
    first, *_ = synthetic_session(producer, monkeypatch)
    expected = [
        asdict(
            producer._score_row(
                record, first.primary, producer._thresholds(first.primary), index
            )
        )
        for index, record in enumerate(records, 1)
    ]
    second, *_ = synthetic_session(producer, monkeypatch)
    actual = [
        asdict(
            producer.score_primary_url(
                record.raw_url,
                second.primary,
                producer._thresholds(second.primary),
                index,
            )
        )
        for index, record in enumerate(records, 1)
    ]
    for internal, shared in zip(expected, actual, strict=True):
        assert shared == {
            name: value
            for name, value in internal.items()
            if name not in ("record", "secondary_tabular", "secondary_seeds")
        }
