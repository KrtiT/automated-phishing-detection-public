"""Synthetic-only known answers for the frozen secondary drift procedures."""

from hashlib import sha256

import numpy as np
import pytest

from automated_phishing_detection import secondary_drift as drift


def _matrix(values):
    result = np.zeros((len(values), 26), dtype=np.float64)
    result[:, 0] = values
    return result


def _mmd_reference(values=None):
    if values is None:
        values = [0.0] * 128 + [1.0] * 128
    return drift.fit_mmd_reference(
        _matrix(values),
        [f"domain-{i}.example" for i in range(len(values))],
        [f"row-{i:04d}" for i in range(len(values))],
    )


def test_rbf_kernel_known_answer_float64_c_contiguous():
    first = np.asfortranarray(_matrix([0, 1]))
    second = np.asfortranarray(_matrix([0, 2]))
    actual = drift.rbf_kernel(first, second, bandwidth_squared=2.0)
    expected = np.exp(-np.array([[0, 4], [1, 1]], dtype=np.float64) / 4)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.float64
    assert actual.flags.c_contiguous


def test_mmd_reference_uses_smallest_stable_id_then_domain_hash_order():
    domains = [f"domain-{i}.example" for i in range(258)]
    ids = [f"b-{i:04d}" for i in range(258)]
    values = list(range(258))
    domains += ["DOMAIN-0.EXAMPLE."]
    ids += ["a-0000"]
    values += [-1]
    expected = sorted(
        set(domain.lower().removesuffix(".") for domain in domains),
        key=lambda domain: (
            sha256(("secondary-mmd-reference-v1:20260816:" + domain).encode()).digest(),
            domain,
        ),
    )[:256]
    reference = drift.fit_mmd_reference(_matrix(values), domains, ids)
    assert reference.reason is None
    assert reference.domains == tuple(expected)
    expected_id_by_domain = dict(zip(domains[:258], ids[:258], strict=True))
    expected_id_by_domain["domain-0.example"] = "a-0000"
    assert reference.stable_ids == tuple(expected_id_by_domain[d] for d in expected)
    reverse = drift.fit_mmd_reference(_matrix(values)[::-1], domains[::-1], ids[::-1])
    assert reference == reverse
    if "domain-0.example" in reference.domains:
        index = reference.domains.index("domain-0.example")
        assert reference.values[index][0] == -1


def test_reference_bandwidth_ignores_zero_pairwise_distances():
    reference = _mmd_reference()
    assert reference.bandwidth_squared == 1.0


def test_reference_bandwidth_is_linear_median_not_an_observed_quantile():
    reference = _mmd_reference([i**3 for i in range(256)])
    # Independent integer pair-distance ordering has these two middle values.
    assert reference.bandwidth_squared == (14208496981056 + 14210125415424) / 2


def test_mmd_biased_known_answer_includes_diagonals():
    reference = _mmd_reference()
    estimate = drift.mmd_squared(reference, _matrix([0] * 256))
    assert estimate.reason is None
    assert estimate.value == pytest.approx(0.5 * (1 - np.exp(-0.5)))
    same = drift.mmd_squared(reference, reference.values)
    assert same.value == pytest.approx(0.0, abs=1e-15)


@pytest.mark.parametrize("count", [0, 1, 255])
def test_too_few_reference_domains_are_not_estimable(count):
    reference = _mmd_reference(list(range(count)))
    assert reference.bandwidth_squared is None
    assert reference.reason == "fewer_than_256_training_domains"
    estimate = drift.mmd_squared(reference, _matrix([0] * 256))
    assert estimate.value is None
    assert estimate.reason == reference.reason


def test_constant_reference_has_no_positive_bandwidth():
    reference = _mmd_reference([1] * 256)
    assert reference.bandwidth_squared is None
    assert reference.reason == "no_positive_reference_distance"


def test_domain_count_not_row_count_controls_reference_estimability():
    reference = drift.fit_mmd_reference(
        _matrix(range(256)), ["one.example"] * 256, [str(i) for i in range(256)]
    )
    assert reference.reason == "fewer_than_256_training_domains"


def test_psi_decile_edges_deduplicate_and_use_right_assignment():
    reference = drift.fit_psi_reference(_matrix([0] * 5 + [1] * 5))
    feature = reference.features[0]
    assert feature.internal_edges == (0.0, 0.5, 1.0)
    assert feature.constant is None
    assert feature.training_counts == (0, 5, 0, 5)
    np.testing.assert_array_equal(
        feature.training_proportions, np.array([0.5, 5.5, 0.5, 5.5]) / 12
    )


def test_constant_feature_has_below_equal_above_bins():
    reference = drift.fit_psi_reference(_matrix([1] * 256))
    feature = reference.features[0]
    assert feature.constant == 1.0
    assert feature.internal_edges == ()
    assert feature.training_counts == (0, 256, 0)
    window = _matrix([0] * 64 + [1] * 128 + [2] * 64)
    result = drift.psi_window_score(reference, window)
    p = np.array([0.5, 256.5, 0.5]) / 257.5
    q = np.array([64.5, 128.5, 64.5]) / 257.5
    expected = float(np.sum((q - p) * np.log(q / p)))
    assert result.feature_scores == pytest.approx((expected,) + (0.0,) * 25)
    assert result.value == expected
    assert result.reason is None


def test_psi_uses_fixed_training_bins_and_retains_all_feature_scores():
    reference = drift.fit_psi_reference(_matrix(range(256)))
    frozen = reference
    window = _matrix([1000] * 256)
    window[:, 25] = 1
    result = drift.psi_window_score(reference, window)
    assert reference == frozen
    assert len(result.feature_scores) == 26
    assert result.value == max(result.feature_scores)
    assert result.feature_scores[25] > 0
    assert all(score == 0 for score in result.feature_scores[1:25])


def test_references_snapshot_training_values_without_mutating_inputs():
    training = _matrix(range(256))
    original = training.copy()
    mmd = drift.fit_mmd_reference(
        training, [f"d-{i}.example" for i in range(256)], [str(i) for i in range(256)]
    )
    psi = drift.fit_psi_reference(training)
    expected_values = mmd.values
    expected_edges = psi.features[0].internal_edges
    np.testing.assert_array_equal(training, original)
    training[:] = 999
    assert mmd.values == expected_values
    assert psi.features[0].internal_edges == expected_edges


def test_domain_breaks_reference_hash_ties(monkeypatch):
    class CollisionDigest:
        def digest(self):
            return b"\x00" * 32

    monkeypatch.setattr(drift, "sha256", lambda value: CollisionDigest())
    reference = _mmd_reference()
    assert reference.domains == tuple(sorted(reference.domains))


def test_empty_psi_training_is_not_estimable():
    reference = drift.fit_psi_reference(_matrix([]))
    assert reference.reason == "no_training_rows"
    result = drift.psi_window_score(reference, _matrix([0] * 256))
    assert result.value is None
    assert result.reason == reference.reason
    assert result.feature_scores == ()


@pytest.mark.parametrize("method", ["mmd", "psi"])
def test_complete_256_row_windows_at_stride_64_only(method):
    stream = _matrix([0] * 256 + [1] * 129)
    if method == "mmd":
        reference = _mmd_reference()
        score = drift.mmd_squared
        trace = drift.mmd_window_scores(reference, stream)
    else:
        reference = drift.fit_psi_reference(_matrix([0] * 256))
        score = drift.psi_window_score
        trace = drift.psi_window_scores(reference, stream)
    assert trace.reason is None
    assert trace.window_end_positions == (256, 320, 384)
    assert trace.scores == tuple(
        score(reference, stream[end - 256 : end]).value for end in (256, 320, 384)
    )
    if method == "psi":
        assert trace.feature_scores == tuple(
            score(reference, stream[end - 256 : end]).feature_scores
            for end in (256, 320, 384)
        )


@pytest.mark.parametrize("method", ["mmd", "psi"])
@pytest.mark.parametrize("count", [0, 255])
def test_incomplete_streams_have_explicit_nonestimability(method, count):
    if method == "mmd":
        reference = _mmd_reference()
        trace = drift.mmd_window_scores(reference, _matrix([0] * count))
    else:
        reference = drift.fit_psi_reference(_matrix([0]))
        trace = drift.psi_window_scores(reference, _matrix([0] * count))
    assert trace.window_end_positions == ()
    assert trace.scores == ()
    assert trace.reason == "no_complete_256_row_window"


def test_calibration_is_linear_95th_percentile_and_alert_boundary_is_strict():
    calibration = drift.calibrate_window_scores([0, 1, 2])
    assert calibration.threshold == 1.9
    assert calibration.calibration_window_count == 3
    audit = drift.audit_window_scores([1.9, np.nextafter(1.9, np.inf), 0], calibration)
    assert audit.alerts == (False, True, False)
    assert audit.alert_count == 1
    assert audit.window_count == 3
    assert audit.alert_fraction == 1 / 3
    assert audit.threshold == calibration.threshold
    assert audit.reason is None
    external = drift.audit_window_scores([100], calibration)
    assert external.threshold == 1.9
    assert external.alerts == (True,)


def test_empty_calibration_and_audit_have_null_results_with_reasons():
    unavailable = drift.calibrate_window_scores([])
    assert unavailable.threshold is None
    assert unavailable.reason == "no_calibration_windows"
    audit = drift.audit_window_scores([0], unavailable)
    assert audit.alert_count is None
    assert audit.alert_fraction is None
    assert audit.reason == "no_calibration_windows"
    empty = drift.audit_window_scores([], drift.calibrate_window_scores([0]))
    assert empty.alert_count is None
    assert empty.alert_fraction is None
    assert empty.reason == "no_audit_windows"


@pytest.mark.parametrize(
    "invalid",
    [
        np.zeros((3, 25)),
        [1, 2],
        [[0], [1, 2]],
        _matrix([np.nan]),
        _matrix([np.inf]),
        np.full((3, 26), True),
        np.full((3, 26), "1"),
    ],
)
def test_invalid_matrices_rejected_by_both_reference_builders(invalid):
    with pytest.raises(drift.SecondaryDriftError):
        drift.fit_psi_reference(invalid)
    with pytest.raises(drift.SecondaryDriftError):
        drift.fit_mmd_reference(invalid, ["a.example"] * 3, ["a", "b", "c"])


@pytest.mark.parametrize("bandwidth", [0, -1, np.nan, np.inf, True, "1"])
def test_invalid_bandwidth_rejected(bandwidth):
    with pytest.raises(drift.SecondaryDriftError):
        drift.rbf_kernel(_matrix([0]), _matrix([0]), bandwidth_squared=bandwidth)


@pytest.mark.parametrize("scores", [[np.nan], [np.inf], [[1]], [True], ["1"]])
def test_invalid_calibration_and_audit_scores_rejected(scores):
    with pytest.raises(drift.SecondaryDriftError):
        drift.calibrate_window_scores(scores)
    with pytest.raises(drift.SecondaryDriftError):
        drift.audit_window_scores(scores, drift.calibrate_window_scores([0]))


@pytest.mark.parametrize(
    ("domains", "ids"),
    [
        (["a.example"], []),
        (["bad domain"], ["id"]),
        (["a.example"], [""]),
        (["a.example", "b.example"], ["same", "same"]),
    ],
)
def test_misaligned_or_invalid_domain_and_stable_id_inputs_rejected(domains, ids):
    with pytest.raises(drift.SecondaryDriftError):
        drift.fit_mmd_reference(_matrix(range(len(domains))), domains, ids)


@pytest.mark.parametrize("count", [0, 255, 257])
def test_individual_window_scoring_requires_exactly_256_rows(count):
    with pytest.raises(drift.SecondaryDriftError):
        drift.mmd_squared(_mmd_reference(), _matrix([0] * count))
    with pytest.raises(drift.SecondaryDriftError):
        drift.psi_window_score(
            drift.fit_psi_reference(_matrix([0])), _matrix([0] * count)
        )


def test_invalid_window_is_not_hidden_by_unavailable_reference():
    with pytest.raises(drift.SecondaryDriftError):
        drift.mmd_squared(_mmd_reference([]), _matrix([np.nan] * 256))
    with pytest.raises(drift.SecondaryDriftError):
        drift.psi_window_score(
            drift.fit_psi_reference(_matrix([])), _matrix([np.inf] * 256)
        )


def test_finite_input_numerical_overflow_is_rejected_not_returned_as_nan():
    with pytest.raises(drift.SecondaryDriftError):
        drift.rbf_kernel(_matrix([1e308]), _matrix([-1e308]), bandwidth_squared=1)


@pytest.mark.parametrize("count", [0, 256])
def test_invalid_reference_types_are_rejected_for_all_stream_lengths(count):
    with pytest.raises(drift.SecondaryDriftError):
        drift.mmd_window_scores(None, _matrix([0] * count))
    with pytest.raises(drift.SecondaryDriftError):
        drift.psi_window_scores(None, _matrix([0] * count))
