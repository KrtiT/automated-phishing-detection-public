"""Invented URLs only; secondary estimators never access research files."""

import importlib.util
import json
from dataclasses import FrozenInstanceError
from itertools import islice, permutations

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from automated_phishing_detection import baselines


def test_secondary_tabular_module_is_implemented():
    assert (
        importlib.util.find_spec("automated_phishing_detection.secondary_tabular")
        is not None
    ), "missing secondary tabular module"


@pytest.fixture
def tabular():
    name = "automated_phishing_detection.secondary_tabular"
    assert importlib.util.find_spec(name) is not None, (
        "missing secondary tabular module"
    )
    return importlib.import_module(name)


def inputs():
    def url(index, label):
        if label:
            return f"HTTP://DANGER-{index % 8}.EXAMPLE:80/" + "B" * 40 + "?X=%aa"
        return f"https://safe-{index % 8}.example/" + "a" * (index % 3)

    train_labels = tuple(i % 2 for i in range(96))
    validation_labels = tuple(int(i % 10 == 0) for i in range(400))
    return (
        tuple(url(i, label) for i, label in enumerate(train_labels)),
        train_labels,
        tuple(url(i, label) for i, label in enumerate(validation_labels)),
        validation_labels,
    )


@pytest.mark.parametrize("kind", ["formatting", "permutation", "random_forest"])
def test_real_secondary_fit_roundtrip_and_frozen_validation_threshold(tabular, kind):
    args = inputs()
    if kind == "permutation":
        result = tabular.fit_label_permutation(*args, seed=42)
    else:
        result = getattr(tabular, f"fit_{kind}")(*args)
    model = tabular.load_secondary_model_bytes(result.artifact_bytes)
    assert model.score_urls(args[2]) == result.validation_scores
    assert result.validation_threshold == baselines.select_validation_threshold(
        result.validation_scores, args[3]
    )
    artifact = json.loads(result.artifact_bytes)
    assert artifact["artifact_type"] == "secondary-tabular-model"
    assert artifact["analysis_stage"] == "development_validation_only"
    assert artifact["protected_evaluation_authorized"] is False
    assert artifact["model_kind"] == kind
    assert artifact["classes"] == [0, 1]
    assert "https://" not in result.artifact_bytes.decode()
    assert "DANGER" not in result.artifact_bytes.decode()
    assert "train_labels" not in artifact
    assert "validation_labels" not in artifact
    with pytest.raises(FrozenInstanceError):
        result.artifact_bytes = b"replaced"


def test_frozen_v1_diagnostic_rf_leaf_renormalization(tabular, monkeypatch):
    # Characterize the known v1 rejection, not a fix or a research-failure diagnosis.
    urls = tuple(
        "https://invented.example/" + "".join(path)
        for path in islice(permutations("abcde"), 19)
    )
    assert len(set(urls)) == 19
    assert len(set(map(tuple, tabular._features(urls, "random_forest")))) == 1
    original = tabular._fit_forest
    observed = []

    def capture(train, labels, validation):
        result = original(train, labels, validation)
        observed.append((result[0], result[1].copy(), validation.copy()))
        return result

    monkeypatch.setattr(tabular, "_fit_forest", capture)
    with pytest.raises(
        tabular.SecondaryTabularError,
        match="^portable scores differ from fitted estimator$",
    ):
        tabular.fit_random_forest(urls[:17], (0,) * 11 + (1,) * 6, urls[17:], (0, 1))
    assert len(observed) == 1
    state, sklearn_scores, validation = observed[0]
    with tabular._numerical_runtime():
        direct = np.zeros((len(validation), 2), dtype=np.float64)
        for tree in state["trees"]:
            tabular._validate_tree(tree, validation.shape[1])
            assert tree["children_left"] == [-1]
            direct += np.asarray(tree["value"][0], dtype=np.float64)
        direct /= len(state["trees"])
        renormalized = tabular._forest_scores(state, validation)
    np.testing.assert_array_equal(direct[:, 1], sklearn_scores)
    assert not np.array_equal(renormalized, sklearn_scores)


def test_formatting_feature_order_and_baseline_settings_are_carried_forward(tabular):
    from automated_phishing_detection.secondary_probes import FORMATTING_FEATURE_NAMES

    artifact = json.loads(tabular.fit_formatting(*inputs()).artifact_bytes)
    assert artifact["features"] == list(FORMATTING_FEATURE_NAMES)
    assert artifact["parameters"]["classifier"]["random_state"] == 42
    assert artifact["parameters"]["classifier"]["solver"] == "saga"
    assert artifact["parameters"]["classifier"]["tol"] == 1e-4
    assert artifact["parameters"]["classifier"]["max_iter"] == 5000
    assert len(artifact["state"]["scaler"]["mean"]) == 5
    assert artifact["scoring"]["batch_size"] == 1
    expected_classifier = LogisticRegression(
        **{
            key: value
            for key, value in baselines._CLASSIFIER_CONFIG.items()
            if key != "class"
        }
    )
    assert artifact["parameters"]["classifier"] == expected_classifier.get_params(
        deep=False
    )
    assert artifact["parameters"]["scaler"] == StandardScaler().get_params(deep=False)


def test_each_permutation_starts_from_original_labels_and_keeps_solver_seed(
    tabular, monkeypatch
):
    original = tabular.LogisticRegression.fit
    observed = []

    def capture(self, features, labels, *args, **kwargs):
        observed.append((np.asarray(labels).copy(), self.random_state))
        return original(self, features, labels, *args, **kwargs)

    monkeypatch.setattr(tabular.LogisticRegression, "fit", capture)
    args = inputs()
    before = args[1], args[3]
    for seed in range(42, 47):
        result = tabular.fit_label_permutation(*args, seed=seed)
        expected = np.random.Generator(np.random.PCG64(seed)).permutation(args[1])
        np.testing.assert_array_equal(observed[-1][0], expected)
        assert observed[-1][1] == 42
        assert sum(observed[-1][0]) == sum(args[1])
        assert result.validation_threshold == baselines.select_validation_threshold(
            result.validation_scores, args[3]
        )
        artifact = json.loads(result.artifact_bytes)
        assert artifact["permutation"] == {"bit_generator": "PCG64", "seed": seed}
    assert before == (args[1], args[3])


def test_random_forest_records_every_effective_parameter_and_tree_order(tabular):
    artifact = json.loads(tabular.fit_random_forest(*inputs()).artifact_bytes)
    expected = RandomForestClassifier(
        n_estimators=100,
        criterion="gini",
        max_features="sqrt",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        bootstrap=True,
        class_weight="balanced",
        random_state=42,
        n_jobs=1,
        oob_score=False,
        warm_start=False,
    ).get_params(deep=False)
    assert artifact["parameters"] == expected
    assert len(artifact["state"]["trees"]) == 100
    assert artifact["scoring"]["input_dtype"] == "float32"
    assert artifact["scoring"]["tree_order"] == "fitted_estimator_order"


@pytest.mark.parametrize("seed", [True, 0, 41, 47, 42.0, "42"])
def test_unapproved_permutation_seed_is_rejected_before_fit(tabular, seed):
    with pytest.raises(tabular.SecondaryTabularError):
        tabular.fit_label_permutation(*inputs(), seed=seed)


@pytest.mark.parametrize("which", [1, 3])
@pytest.mark.parametrize("invalid", [[True] * 96, [0] * 96, [2] * 96, [], "0101"])
def test_label_contract_rejects_coercion_missing_classes_and_misalignment(
    tabular, which, invalid
):
    args = list(inputs())
    args[which] = invalid
    with pytest.raises(tabular.SecondaryTabularError):
        tabular.fit_formatting(*args)


def test_fit_warning_stops_without_return_or_retry(tabular, monkeypatch):
    import warnings

    calls = []

    def fail(*args, **kwargs):
        calls.append(1)
        warnings.warn("synthetic fit warning", RuntimeWarning)

    monkeypatch.setattr(tabular.LogisticRegression, "fit", fail)
    with pytest.raises(tabular.SecondaryTabularError):
        tabular.fit_formatting(*inputs())
    assert calls == [1]


@pytest.mark.parametrize("content", [b"{}", b"NaN", b'{"x":1,"x":2}', b"[]"])
def test_loader_rejects_noncanonical_or_wrong_schema(tabular, content):
    with pytest.raises(tabular.SecondaryTabularError):
        tabular.load_secondary_model_bytes(content)


def test_loaded_scoring_has_no_fit_dependency(tabular, monkeypatch):
    result = tabular.fit_formatting(*inputs())

    def no_fit(*args, **kwargs):
        pytest.fail("loaded model attempted to fit")

    monkeypatch.setattr(tabular.LogisticRegression, "fit", no_fit)
    model = tabular.load_secondary_model_bytes(result.artifact_bytes)
    assert model.score_urls(inputs()[2]) == result.validation_scores


def test_runtime_rejects_other_blas_before_extracting_any_input(tabular, monkeypatch):
    from automated_phishing_detection import gmm_monitor

    monkeypatch.setattr(
        gmm_monitor,
        "_numpy_build_configuration",
        lambda: {"name": "accelerate", "version": "unknown"},
    )

    def no_input(*args):
        pytest.fail("features accessed before runtime rejection")

    monkeypatch.setattr(tabular, "_features", no_input)
    with pytest.raises(tabular.SecondaryTabularError):
        tabular.fit_formatting(*inputs())


def test_secondary_numeric_context_does_not_ignore_underflow(tabular):
    with pytest.raises(FloatingPointError):
        with tabular._numerical_runtime():
            np.multiply(np.float64(1e-250), np.float64(1e-250))


def canonical(value):
    return (
        json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
        + "\n"
    ).encode("ascii")


@pytest.fixture
def formatting_artifact(tabular):
    return json.loads(tabular.fit_formatting(*inputs()).artifact_bytes)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda x: x.update(schema_version=True),
        lambda x: x.update(classes=[False, True]),
        lambda x: x.update(contract_id="rq1-baselines-v2"),
        lambda x: x["parameters"]["classifier"].update(random_state=43),
        lambda x: x["parameters"]["scaler"].update(copy=False),
        lambda x: x["state"]["scaler"]["mean"].append(0.0),
        lambda x: x["state"]["scaler"]["scale"].__setitem__(0, 0.0),
        lambda x: x["state"]["scaler"]["variance"].__setitem__(0, -1.0),
        lambda x: x["state"]["scaler"].update(n_samples_seen=True),
        lambda x: x["state"].update(n_iter=[5000]),
        lambda x: x["state"].update(n_iter=[False]),
        lambda x: x["state"]["coefficients"][0].__setitem__(0, float("nan")),
        lambda x: x["state"]["coefficients"][0].__setitem__(0, "0"),
    ],
)
def test_strict_logistic_model_schema_rejects_drift(
    tabular, formatting_artifact, mutation
):
    mutation(formatting_artifact)
    with pytest.raises(tabular.SecondaryTabularError):
        tabular.load_secondary_model_bytes(canonical(formatting_artifact))


@pytest.mark.parametrize(
    "mutation",
    [
        lambda x: x["state"]["trees"].pop(),
        lambda x: x["state"]["trees"][0]["children_left"].__setitem__(0, 0),
        lambda x: x["state"]["trees"][0]["feature"].__setitem__(0, 25),
        lambda x: x["state"]["trees"][0]["value"][0].__setitem__(0, -1.0),
        lambda x: x["state"]["trees"][0].update(random_state=True),
    ],
)
def test_strict_random_forest_rejects_bad_topology_and_state(tabular, mutation):
    artifact = json.loads(tabular.fit_random_forest(*inputs()).artifact_bytes)
    mutation(artifact)
    with pytest.raises(tabular.SecondaryTabularError):
        tabular.load_secondary_model_bytes(canonical(artifact))


def test_random_forest_traversal_compares_float32_value_to_double_threshold(tabular):
    artifact = json.loads(tabular.fit_random_forest(*inputs()).artifact_bytes)
    # Float64 2**24+1 rounds down to 2**24 when converted to the tree input dtype.
    tree = {
        "children_left": [1, -1, -1],
        "children_right": [2, -1, -1],
        "feature": [0, -2, -2],
        "threshold": [float(2**24), -2.0, -2.0],
        "value": [[0.5, 0.5], [1.0, 0.0], [0.0, 1.0]],
        "random_state": 1,
    }
    artifact["state"]["trees"] = [tree] * 100
    model = tabular.load_secondary_model_bytes(canonical(artifact))
    matrix = np.zeros((1, 25), dtype=np.float64)
    matrix[0, 0] = 2**24 + 1
    assert tabular._score_state(json.loads(model.artifact_bytes), matrix)[
        0
    ].tolist() == [0.0]


def test_forest_double_midpoint_does_not_round_up_to_float32_feature(tabular):
    lower = np.float32(1.0000001)
    upper = np.nextafter(lower, np.float32(2.0))
    training = np.zeros((2, 25), dtype=np.float32)
    training[:, 0] = (lower, upper)
    fitted = DecisionTreeClassifier(max_depth=1, random_state=42).fit(training, [0, 1])
    assert float(lower) < fitted.tree_.threshold[0] < float(upper)
    tree = {
        "children_left": fitted.tree_.children_left.tolist(),
        "children_right": fitted.tree_.children_right.tolist(),
        "feature": fitted.tree_.feature.tolist(),
        "threshold": fitted.tree_.threshold.tolist(),
        "value": fitted.tree_.value[:, 0, :].tolist(),
        "random_state": 42,
    }
    actual = tabular._forest_scores({"trees": [tree] * 100}, training)
    np.testing.assert_array_equal(actual, fitted.predict_proba(training)[:, 1])


@pytest.mark.parametrize("kind", ["formatting", "permutation", "random_forest"])
def test_validation_changes_cannot_change_fitted_model_state(tabular, kind):
    args = inputs()
    changed = (
        args[0],
        args[1],
        tuple(reversed(args[2])),
        tuple(1 - label for label in args[3]),
    )
    if kind == "permutation":
        first = tabular.fit_label_permutation(*args, seed=43)
        second = tabular.fit_label_permutation(*changed, seed=43)
    else:
        fit = getattr(tabular, f"fit_{kind}")
        first, second = fit(*args), fit(*changed)
    assert first.artifact_bytes == second.artifact_bytes
    assert first.validation_scores[::-1] == second.validation_scores
    assert second.validation_threshold == baselines.select_validation_threshold(
        second.validation_scores, changed[3]
    )


def test_permutation_result_retains_target_not_met_without_fallback(tabular):
    args = inputs()
    result = tabular.fit_label_permutation(args[0], args[1], args[0], args[1], seed=42)
    assert result.validation_threshold["status"] == "target_not_met"
    assert result.validation_threshold["threshold"] is None
    assert (
        result.scoring_audit["threshold_role"]
        == "secondary_descriptive_operating_point"
    )


def test_rf_v2_matches_sklearn_without_leaf_renormalization(tabular):
    urls = tuple(
        "https://invented.example/" + "".join(path)
        for path in islice(permutations("abcde"), 19)
    )
    checkpoints = []
    result = tabular.fit_random_forest_v2(
        urls[:17],
        (0,) * 11 + (1,) * 6,
        urls[17:],
        (0, 1),
        checkpoint=checkpoints.append,
    )
    model = tabular.load_secondary_model_bytes(result.artifact_bytes)
    assert model.score_urls(urls[17:]) == result.validation_scores
    artifact = json.loads(result.artifact_bytes)
    assert artifact["contract_id"] == "secondary-development-correction-v1"
    assert artifact["method_version"] == "secondary-rf-v2"
    assert result.scoring_audit["portable_exact_parity"] is True
    assert len(checkpoints) == 1
    checkpoint = tabular.validate_fit_checkpoint_bytes(
        checkpoints[0], model_bytes=result.artifact_bytes
    )
    assert checkpoint["artifact_type"] == "secondary-fit-checkpoint"
    assert checkpoint["classes"] == [0, 1]
    assert checkpoint["state"] == artifact["state"]
    with pytest.raises(tabular.SecondaryTabularError):
        tabular.load_secondary_model_bytes(checkpoints[0])
    legacy = dict(artifact)
    legacy.update(
        contract_id="secondary-development-v1",
        method_version="secondary-tabular-v1",
        scoring=tabular._RF_SCORING,
    )
    assert (
        tabular.load_secondary_model_bytes(canonical(legacy)).score_urls(urls[17:])
        != result.validation_scores
    )


@pytest.mark.parametrize("kind", ["formatting", "permutation", "random_forest_v2"])
def test_fit_checkpoint_precedes_validation_scoring(tabular, monkeypatch, kind):
    checkpoints = []

    def fail(*args, **kwargs):
        assert len(checkpoints) == 1
        raise FloatingPointError("private fixture details")

    monkeypatch.setattr(
        RandomForestClassifier if kind == "random_forest_v2" else LogisticRegression,
        "predict_proba",
        fail,
    )
    fit = getattr(
        tabular, "fit_label_permutation" if kind == "permutation" else f"fit_{kind}"
    )
    options = {"seed": 42} if kind == "permutation" else {}
    with pytest.raises(tabular.SecondaryTabularError) as error:
        fit(*inputs(), checkpoint=checkpoints.append, **options)
    assert error.value.check_id == "validation_score"
    assert len(checkpoints) == 1
    assert "private fixture" not in str(error.value)


def test_fit_checkpoint_failure_stops_before_scoring(tabular, monkeypatch):
    calls = []

    def fail(content):
        calls.append(content)
        raise OSError("/private/fixture/path")

    def no_score(*args, **kwargs):
        pytest.fail("scoring followed failed checkpoint")

    monkeypatch.setattr(RandomForestClassifier, "predict_proba", no_score)
    with pytest.raises(tabular.SecondaryTabularError) as error:
        tabular.fit_random_forest_v2(*inputs(), checkpoint=fail)
    assert error.value.check_id == "checkpoint_write"
    assert "/private/" not in str(error.value)
    assert len(calls) == 1


def test_fit_failure_does_not_fabricate_checkpoint(tabular, monkeypatch):
    checkpoints = []

    def fail(*args, **kwargs):
        raise FloatingPointError("invented fit failure")

    monkeypatch.setattr(RandomForestClassifier, "fit", fail)
    with pytest.raises(tabular.SecondaryTabularError) as error:
        tabular.fit_random_forest_v2(*inputs(), checkpoint=checkpoints.append)
    assert error.value.check_id == "fit"
    assert checkpoints == []


@pytest.mark.parametrize(
    "value,tag",
    [
        (float("nan"), "nan"),
        (float("inf"), "positive_infinity"),
        (-float("inf"), "negative_infinity"),
    ],
)
def test_checkpoint_retains_nonfinite_state_before_rejection(
    tabular, monkeypatch, value, tag
):
    original = LogisticRegression.fit
    checkpoints = []

    def corrupt(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        self.coef_[0, 0] = value
        return result

    monkeypatch.setattr(LogisticRegression, "fit", corrupt)
    with pytest.raises(tabular.SecondaryTabularError) as error:
        tabular.fit_formatting(*inputs(), checkpoint=checkpoints.append)
    assert error.value.check_id == "fitted_state"
    snapshot = tabular.validate_fit_checkpoint_bytes(checkpoints[0])
    assert snapshot["state"]["coefficients"][0][0] == {"nonfinite": tag}


def test_checkpoint_precedes_class_validation(tabular, monkeypatch):
    original = RandomForestClassifier.fit
    checkpoints = []

    def corrupt(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        self.classes_ = np.array([1, 0])
        return result

    monkeypatch.setattr(RandomForestClassifier, "fit", corrupt)
    with pytest.raises(tabular.SecondaryTabularError) as error:
        tabular.fit_random_forest_v2(*inputs(), checkpoint=checkpoints.append)
    assert error.value.check_id == "fitted_classes"
    assert json.loads(checkpoints[0])["classes"] == [1, 0]


@pytest.mark.parametrize("phase", ["portable_exact_parity", "threshold_selection"])
def test_late_failure_preserves_checkpoint_and_static_check_id(
    tabular, monkeypatch, phase
):
    checkpoints = []
    if phase == "portable_exact_parity":
        original = tabular._score_state

        def differ(*args):
            values, audit = original(*args)
            values[0] = np.nextafter(values[0], np.float64(2))
            return values, audit

        monkeypatch.setattr(tabular, "_score_state", differ)
    else:

        def fail(*args):
            raise ValueError("private threshold fixture")

        monkeypatch.setattr(baselines, "select_validation_threshold", fail)
    with pytest.raises(tabular.SecondaryTabularError) as error:
        tabular.fit_random_forest_v2(*inputs(), checkpoint=checkpoints.append)
    assert error.value.check_id == phase
    assert len(checkpoints) == 1


def test_check_id_rejects_unrecognized_or_private_strings(tabular):
    with pytest.raises(ValueError):
        tabular.SecondaryTabularError("safe", check_id="https://private.example")
    assert tabular.SecondaryTabularError("original message").check_id is None
    assert str(tabular.SecondaryTabularError("original message")) == "original message"
    assert tabular.SecondaryTabularError().args == ()
    assert tabular.SecondaryTabularError("first", "second").args == ("first", "second")


def test_checkpoint_callback_secondary_error_is_sanitized(tabular):
    def fail(content):
        raise tabular.SecondaryTabularError("https://private.example/model")

    with pytest.raises(tabular.SecondaryTabularError) as error:
        tabular.fit_random_forest_v2(*inputs(), checkpoint=fail)
    assert error.value.check_id == "checkpoint_write"
    assert "private" not in str(error.value)


def test_rf_checkpoint_precedes_nonfinite_tree_validation(tabular, monkeypatch):
    original = RandomForestClassifier.fit
    checkpoints = []

    def corrupt(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        self.estimators_[0].tree_.value[0, 0, 0] = float("nan")
        return result

    def no_score(*args, **kwargs):
        pytest.fail("validation scoring preceded tree-state validation")

    monkeypatch.setattr(RandomForestClassifier, "fit", corrupt)
    monkeypatch.setattr(RandomForestClassifier, "predict_proba", no_score)
    with pytest.raises(tabular.SecondaryTabularError) as error:
        tabular.fit_random_forest_v2(*inputs(), checkpoint=checkpoints.append)
    assert error.value.check_id == "fitted_state"
    assert len(checkpoints) == 1
    assert json.loads(checkpoints[0])["state"]["trees"][0]["value"][0][0] == {
        "nonfinite": "nan"
    }


@pytest.mark.parametrize(
    "mutation",
    [
        lambda x: x.update(contract_id="secondary-development-v1"),
        lambda x: x.update(method_version="secondary-tabular-v1"),
        lambda x: x.update(model_kind="formatting"),
        lambda x: x["scoring"].update(
            probability="sequential_float64_sum_of_normalized_leaf_values_divided_by_100"
        ),
    ],
)
def test_rf_v2_loader_rejects_mixed_version_identity(tabular, mutation):
    result = tabular.fit_random_forest_v2(*inputs(), checkpoint=lambda content: None)
    artifact = json.loads(result.artifact_bytes)
    mutation(artifact)
    with pytest.raises(tabular.SecondaryTabularError):
        tabular.load_secondary_model_bytes(canonical(artifact))


def test_checkpoint_verifier_rejects_model_difference_and_unrecognized_tag(tabular):
    checkpoints = []
    fitted = tabular.fit_formatting(*inputs(), checkpoint=checkpoints.append)
    snapshot = json.loads(checkpoints[0])
    snapshot["state"]["intercept"][0] += 0.1
    with pytest.raises(tabular.SecondaryTabularError, match="checkpoint differs"):
        tabular.validate_fit_checkpoint_bytes(
            canonical(snapshot), model_bytes=fitted.artifact_bytes
        )
    snapshot["state"]["intercept"][0] = {"nonfinite": "private"}
    with pytest.raises(tabular.SecondaryTabularError):
        tabular.validate_fit_checkpoint_bytes(canonical(snapshot))


def test_rf_v2_requires_checkpoint_before_fit(tabular, monkeypatch):
    def no_fit(*args, **kwargs):
        pytest.fail("fit started without checkpoint callback")

    monkeypatch.setattr(RandomForestClassifier, "fit", no_fit)
    with pytest.raises(TypeError):
        tabular.fit_random_forest_v2(*inputs())
    with pytest.raises(tabular.SecondaryTabularError) as error:
        tabular.fit_random_forest_v2(*inputs(), checkpoint=None)
    assert error.value.check_id == "input_validation"


def test_checkpoint_does_not_coerce_nonfinite_scaler_count(tabular, monkeypatch):
    original = StandardScaler.fit_transform
    checkpoints = []

    def corrupt(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        self.n_samples_seen_ = np.float64("nan")
        return result

    monkeypatch.setattr(StandardScaler, "fit_transform", corrupt)
    with pytest.raises(tabular.SecondaryTabularError) as error:
        tabular.fit_formatting(*inputs(), checkpoint=checkpoints.append)
    assert error.value.check_id == "fitted_state"
    assert json.loads(checkpoints[0])["state"]["scaler"]["n_samples_seen"] == {
        "nonfinite": "nan"
    }


def test_invalid_permutation_seed_has_input_check_id(tabular):
    with pytest.raises(tabular.SecondaryTabularError) as error:
        tabular.fit_label_permutation(*inputs(), seed=47)
    assert error.value.check_id == "input_validation"


@pytest.mark.parametrize("values", [[0.0, 1.0000000000005], [1.0000000000005, 0.0]])
def test_rf_v2_rejects_out_of_range_stored_leaf_probabilities_without_clamping(
    tabular, monkeypatch, values
):
    def forbidden(*args, **kwargs):
        pytest.fail("loaded synthetic state attempted a fit")

    monkeypatch.setattr(RandomForestClassifier, "fit", forbidden)
    tree = {
        "children_left": [-1],
        "children_right": [-1],
        "feature": [-2],
        "threshold": [-2.0],
        "value": [values],
        "random_state": 42,
    }
    artifact = tabular._artifact(
        "random_forest", 2, {"trees": [tree] * 100}, None, rf_v2=True
    )
    model = tabular.load_secondary_model_bytes(canonical(artifact))
    with pytest.raises(
        tabular.SecondaryTabularError, match="forest probabilities are invalid"
    ):
        model.score_urls(("https://invented.example/path",))
    artifact.update(
        contract_id="secondary-development-v1",
        method_version="secondary-tabular-v1",
        scoring=tabular._RF_SCORING,
    )
    legacy = tabular.load_secondary_model_bytes(canonical(artifact))
    assert legacy.score_urls(("https://invented.example/path",)) == (
        1.0 if values[1] else 0.0,
    )
