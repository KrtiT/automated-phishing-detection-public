"""Expose existing invented complete scoring columns through the production hook."""

from automated_phishing_detection.bound_secondary import (
    CompletedSeedColumn,
    CompletedTabularColumn,
)


def completed_columns(scoring, callback):
    if callback is None:
        return scoring
    for index, (name, count) in enumerate(scoring.counts.tabular_singleton_calls):
        callback(
            CompletedTabularColumn(
                name, tuple(row.tabular[index] for row in scoring.rows), count
            )
        )
    for index, (seed, count) in enumerate(scoring.counts.transformer_singleton_calls):
        callback(
            CompletedSeedColumn(
                seed,
                tuple(row.seeds[index] for row in scoring.rows),
                count,
                scoring.counts.reused_primary_transformer_scores if seed == 42 else 0,
            )
        )
    return scoring
