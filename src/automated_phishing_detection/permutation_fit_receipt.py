"""Private receipts for newly audited permutation fits, not historical proof."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np


class FitReceiptError(ValueError):
    def __init__(self, message: str, check_id: str):
        super().__init__(message)
        self.check_id = check_id


def _json_bytes(value: dict) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("ascii")


def _digest(array: np.ndarray, dtype: str) -> str:
    return hashlib.sha256(array.astype(dtype).tobytes(order="C")).hexdigest()


def _matrix_identity(matrix: np.ndarray) -> dict:
    return {
        "shape": list(matrix.shape),
        "dtype": "<f8",
        "order": "C",
        "sha256": _digest(matrix, "<f8"),
    }


@dataclass(frozen=True)
class PermutationFitAudit:
    record_ids: tuple[str, ...]
    original_labels: tuple[int, ...]
    seed: int
    source_features_json: str
    writer: Callable[[bytes], None]

    @classmethod
    def bind(cls, record_ids, original_labels, seed, source_features, writer):
        if not isinstance(record_ids, Sequence) or isinstance(record_ids, (str, bytes)):
            raise ValueError("training IDs must be an ordered sequence")
        if (
            len(record_ids) != len(original_labels)
            or any(
                type(identifier) is not str or not identifier.strip()
                for identifier in record_ids
            )
            or len(set(record_ids)) != len(record_ids)
        ):
            raise ValueError("training IDs must be aligned, nonempty and unique")
        if not callable(writer):
            raise ValueError("fit-input receipt writer is required")
        if type(seed) is not int or seed not in range(42, 47):
            raise ValueError("permutation seed must be 42 through 46")
        labels = np.asarray(original_labels)
        if labels.ndim != 1 or labels.dtype.kind not in "iu" or set(labels) != {0, 1}:
            raise ValueError(
                "original labels must be binary integers with both classes"
            )
        if source_features.shape != (len(labels), 25) or not np.all(
            np.isfinite(source_features)
        ):
            raise ValueError("source features must be finite and aligned")
        return cls(
            tuple(record_ids),
            tuple(map(int, labels)),
            seed,
            _json_bytes(_matrix_identity(source_features)).decode("ascii"),
            writer,
        )

    def _verify_and_copy(self, features, labels):
        permutation = np.random.Generator(np.random.PCG64(self.seed)).permutation(
            np.arange(len(self.original_labels), dtype=np.int64)
        )
        expected = np.asarray(self.original_labels, dtype=np.int64)[permutation]
        if labels.dtype != np.dtype("int64") or not np.array_equal(labels, expected):
            raise FitReceiptError(
                "permuted labels differ at estimator boundary", "fit_input_validation"
            )
        if (
            features.dtype != np.dtype("float64")
            or features.shape != (len(labels), 25)
            or not np.all(np.isfinite(features))
        ):
            raise FitReceiptError(
                "fit features must be finite float64 and aligned",
                "fit_input_validation",
            )
        feature_copy, label_copy = features.copy(order="C"), labels.copy(order="C")
        feature_copy.flags.writeable = label_copy.flags.writeable = False
        return feature_copy, label_copy, permutation

    def _prepared_receipt(self, estimator, features, labels, permutation):
        return _json_bytes(
            {
                "schema_version": 1,
                "analysis_stage": "post_review_new_fit_diagnostic",
                "event": "fit_call_prepared",
                "ordered_record_ids": list(self.record_ids),
                "consumed_labels": labels.tolist(),
                "consumed_labels_sha256": _digest(labels, "<i8"),
                "original_labels_sha256": _digest(
                    np.asarray(self.original_labels), "<i8"
                ),
                "label_class_counts": {
                    str(label): int(np.count_nonzero(labels == label))
                    for label in (0, 1)
                },
                "permutation_seed": self.seed,
                "bit_generator": "PCG64",
                "permutation_indices_sha256": _digest(permutation, "<i8"),
                "source_features": json.loads(self.source_features_json),
                "scaled_features": _matrix_identity(features),
                "parameters": estimator.get_params(deep=False),
            }
        )

    def _write(self, content):
        try:
            self.writer(content)
        except Exception as exc:
            raise FitReceiptError(
                "fit-input receipt write failed", "fit_input_receipt"
            ) from exc

    def fit(self, estimator, features, labels):
        feature_copy, label_copy, permutation = self._verify_and_copy(features, labels)
        prepared = self._prepared_receipt(
            estimator, feature_copy, label_copy, permutation
        )
        self._write(prepared)
        fitted = estimator.fit(feature_copy, label_copy)
        self._write(
            _json_bytes(
                {
                    "schema_version": 1,
                    "event": "fit_call_returned",
                    "prepared_receipt_sha256": hashlib.sha256(prepared).hexdigest(),
                }
            )
        )
        return fitted
