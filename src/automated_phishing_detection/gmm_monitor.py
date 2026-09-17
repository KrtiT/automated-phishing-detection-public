"""Frozen, train-only RQ2 mixture with disjoint validation calibration and audit."""

from __future__ import annotations

import io
import os
import platform
import shutil
import stat
import tempfile
import warnings
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from hashlib import sha256
from numbers import Real
from pathlib import Path

import numpy as np
import scipy
import sklearn
import threadpoolctl
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from . import baselines, fixed_cascade, protocol_preflight, transformer_pipeline
from .url_features import FEATURE_NAMES

CONTRACT_ID = "rq2-gmm-development-v1"
OFFICIAL_GMM_CONTRACT_SHA256 = (
    "22d32088b05e74432704f9671ab76ba28b4f573ead418846b23bc366315cb393"
)
_EXPECTED_CONTRACT_CANONICAL_SHA256 = (
    "90ac440578163a980ea5e1901bd6626fae06468514848d4f5dc42fa103679081"
)
GMM_FEATURE_NAMES = (*FEATURE_NAMES, "logistic_l1_phishing_probability")
_MIXTURE_PARAMETERS = {
    "covariance_type": "diag",
    "tol": 1e-3,
    "reg_covar": 1e-6,
    "max_iter": 500,
    "n_init": 5,
    "init_params": "kmeans",
    "weights_init": None,
    "means_init": None,
    "precisions_init": None,
    "random_state": 42,
    "warm_start": False,
    "verbose": 0,
    "verbose_interval": 10,
}
_VERSIONS = {
    "numpy": "2.2.6",
    "scipy": "1.15.3",
    "scikit-learn": "1.7.2",
    "threadpoolctl": "3.6.0",
}
_PRIVATE_FILES = ("gmm.json", "validation-audit.json")
_publish_path_without_replace = baselines._publish_path_without_replace
_write_file = transformer_pipeline._write_file
_write_summary_temp = transformer_pipeline._write_summary_temp
_fsync_directory = transformer_pipeline._fsync_directory


class GMMMonitorError(ValueError):
    """The frozen development method cannot complete without protocol drift."""


@dataclass(frozen=True)
class _InputHashPolicy:
    train_sha256: str
    validation_sha256: str
    preparation_summary_sha256: str
    baseline_contract_sha256: str
    logistic_l1_artifact_sha256: str
    gmm_contract_sha256: str


_OFFICIAL_INPUT_HASH_POLICY = _InputHashPolicy(
    train_sha256="575f2fb13a0766020e29d78bf8e633a185b381abde7060bdd1ed04cc4a5e38a0",
    validation_sha256="970c6568a6400a1fc265b7809ef7bd9d1c297632799cbb88801313bc34ac415a",
    preparation_summary_sha256="1a85a7eecc0f5baa7c59e03a0cbde63fd4595409feb918dc5ff916ead7cd5c9e",
    baseline_contract_sha256="05d6d0831def7d26448c8dbdc8117800ea2448cdfc2aca2ad95489f22d2d11ba",
    logistic_l1_artifact_sha256="71a3e24a0283a31ba188bc7dd60b18c1b708370b9ca275d5ab1a1004680c968a",
    gmm_contract_sha256=OFFICIAL_GMM_CONTRACT_SHA256,
)


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return transformer_pipeline._canonical_json_bytes(value)
    except transformer_pipeline.TransformerPipelineError as exc:
        raise GMMMonitorError(str(exc)) from exc


def _load_json(content: bytes, field: str) -> object:
    try:
        return transformer_pipeline._load_json(content, field)
    except transformer_pipeline.TransformerPipelineError as exc:
        raise GMMMonitorError(str(exc)) from exc


def _validate_contract(value: object) -> dict:
    canonical = _canonical_json_bytes(value).removesuffix(b"\n")
    if sha256(canonical).hexdigest() != _EXPECTED_CONTRACT_CANONICAL_SHA256:
        raise GMMMonitorError("GMM contract does not match the frozen method")
    return value


def _software_versions() -> dict[str, str]:
    return {
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
        "threadpoolctl": threadpoolctl.__version__,
    }


def _numpy_build_configuration() -> dict[str, str]:
    blas = np.__config__.CONFIG.get("Build Dependencies", {}).get("blas", {})
    return {
        "name": str(blas.get("name", "unknown")),
        "version": str(blas.get("version", "unknown")),
    }


def _require_runtime() -> None:
    if _software_versions() != _VERSIONS:
        raise GMMMonitorError("GMM runtime versions differ from the frozen contract")
    blas = _numpy_build_configuration()
    if blas["name"] != "scipy-openblas" or not (
        blas["version"] == "0.3.29"
        or blas["version"].startswith(("0.3.29.", "0.3.29-", "0.3.29+"))
    ):
        raise GMMMonitorError(
            "GMM requires NumPy OpenBLAS 0.3.29; Accelerate and other builds are unsupported"
        )


def _finite_array(value, shape, field):
    try:
        raw = np.asarray(value, dtype=object)
        if any(
            not isinstance(item, Real) or isinstance(item, (bool, np.bool_))
            for item in raw.flat
        ):
            raise GMMMonitorError(f"{field} must contain only numeric values")
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise GMMMonitorError(f"{field} must be finite numeric values") from exc
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise GMMMonitorError(f"{field} has invalid shape or nonfinite values")
    return array


def _matrix(value):
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise GMMMonitorError("features must be a finite 26-column matrix") from exc
    if (
        matrix.ndim != 2
        or matrix.shape[1] != 26
        or not matrix.shape[0]
        or not np.all(np.isfinite(matrix))
    ):
        raise GMMMonitorError("features must be a finite 26-column matrix")
    return matrix


def _normalize_domain(domain):
    try:
        return protocol_preflight._ascii_domain(domain)
    except protocol_preflight.PreflightError as exc:
        raise GMMMonitorError(f"invalid registrable domain: {exc}") from exc


def allocate_validation_domains(domains) -> dict[str, tuple[int, ...]]:
    """Allocate whole domains without labels; preserve the input stream order."""
    normalized = tuple(_normalize_domain(domain) for domain in domains)
    ordered = sorted(
        set(normalized),
        key=lambda domain: (
            sha256(
                ("rq2-gmm-validation-v1\0" + "20260816\0" + domain).encode("ascii")
            ).digest(),
            domain.encode("ascii"),
        ),
    )
    calibration_domains = set(ordered[: len(ordered) // 2])
    return {
        "calibration": tuple(
            i for i, d in enumerate(normalized) if d in calibration_domains
        ),
        "audit": tuple(
            i for i, d in enumerate(normalized) if d not in calibration_domains
        ),
    }


def window_scores(
    negative_log_likelihoods,
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """Return exclusive stream end positions and means for complete windows."""
    values = np.asarray(negative_log_likelihoods, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise GMMMonitorError("window input must be a finite vector")
    ends = tuple(range(256, len(values) + 1, 64))
    if not ends:
        raise GMMMonitorError("each stream requires at least one complete window")
    with np.errstate(all="raise"):
        scores = tuple(
            float(np.mean(values[end - 256 : end], dtype=np.float64)) for end in ends
        )
    if not all(np.isfinite(scores)):
        raise GMMMonitorError("window scores are nonfinite")
    return ends, scores


def calibrate_and_audit(calibration_scores, audit_scores) -> dict:
    calibration = np.asarray(calibration_scores, dtype=np.float64)
    audit = np.asarray(audit_scores, dtype=np.float64)
    for values in (calibration, audit):
        if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
            raise GMMMonitorError("calibration and audit require finite window scores")
    with np.errstate(all="raise"):
        threshold = float(np.quantile(calibration, 0.95, method="linear"))
    alerts = int(np.count_nonzero(audit > threshold))
    return {
        "threshold": threshold,
        "calibration_window_count": len(calibration),
        "audit_alert_count": alerts,
        "audit_window_count": len(audit),
        "audit_alert_fraction": alerts / len(audit),
        "false_alert_gate_met": 20 * alerts <= len(audit),
    }


def _mixture_state(model: GaussianMixture) -> dict:
    return {
        "components": int(model.n_components),
        "covariance_type": "diag",
        "weights": model.weights_.tolist(),
        "means": model.means_.tolist(),
        "variances": model.covariances_.tolist(),
        "precisions": model.precisions_.tolist(),
        "precisions_cholesky": model.precisions_cholesky_.tolist(),
        "converged": bool(model.converged_),
        "n_iter": int(model.n_iter_),
        "lower_bound": float(model.lower_bound_),
    }


def _validate_mixture(state):
    if type(state) is not dict or set(state) != {
        "components",
        "covariance_type",
        "weights",
        "means",
        "variances",
        "precisions",
        "precisions_cholesky",
        "converged",
        "n_iter",
        "lower_bound",
    }:
        raise GMMMonitorError("mixture fields are invalid")
    k = state["components"]
    if type(k) is not int or not 1 <= k <= 6 or state["covariance_type"] != "diag":
        raise GMMMonitorError("mixture must have 1..6 diagonal components")
    if (
        state["converged"] is not True
        or type(state["n_iter"]) is not int
        or not 1 <= state["n_iter"] <= 500
    ):
        raise GMMMonitorError(
            "every mixture must converge within the frozen iteration limit"
        )
    _finite_array(state["lower_bound"], (), "mixture lower bound")
    weights = _finite_array(state["weights"], (k,), "mixture weights")
    _finite_array(state["means"], (k, 26), "mixture means")
    variances = _finite_array(state["variances"], (k, 26), "mixture variances")
    precisions = _finite_array(state["precisions"], (k, 26), "mixture precisions")
    cholesky = _finite_array(
        state["precisions_cholesky"], (k, 26), "mixture precisions cholesky"
    )
    if np.any(weights <= 0) or not np.isclose(weights.sum(), 1.0, rtol=0, atol=1e-12):
        raise GMMMonitorError("mixture weights must be positive and normalized")
    if np.any(variances <= 0) or np.any(precisions <= 0) or np.any(cholesky <= 0):
        raise GMMMonitorError("mixture variances and precisions must be positive")
    with np.errstate(all="raise"):
        if not np.allclose(
            precisions, 1 / variances, rtol=1e-12, atol=0
        ) or not np.allclose(cholesky, 1 / np.sqrt(variances), rtol=1e-12, atol=0):
            raise GMMMonitorError("mixture precision state is inconsistent")


def load_gmm_artifact_bytes(content: bytes) -> dict:
    """Validate portable JSON state without pickle or estimator deserialization."""
    artifact = _load_json(content, "GMM artifact")
    if type(artifact) is not dict or set(artifact) != {
        "schema_version",
        "contract_id",
        "features",
        "dtype",
        "scaler",
        "mixture",
        "input_hashes",
    }:
        raise GMMMonitorError("GMM artifact fields are invalid")
    if (
        type(artifact["schema_version"]) is not int
        or artifact["schema_version"] != 1
        or artifact["contract_id"] != CONTRACT_ID
        or artifact["features"] != list(GMM_FEATURE_NAMES)
        or artifact["dtype"] != "float64"
    ):
        raise GMMMonitorError("GMM artifact identity or features are invalid")
    scaler = artifact["scaler"]
    if type(scaler) is not dict or set(scaler) != {
        "mean",
        "scale",
        "variance",
        "n_samples_seen",
        "n_features_in",
    }:
        raise GMMMonitorError("scaler fields are invalid")
    if (
        type(scaler["n_samples_seen"]) is not int
        or scaler["n_samples_seen"] < 6
        or type(scaler["n_features_in"]) is not int
        or scaler["n_features_in"] != 26
    ):
        raise GMMMonitorError("scaler sample or feature counts are invalid")
    _finite_array(scaler["mean"], (26,), "scaler mean")
    scale = _finite_array(scaler["scale"], (26,), "scaler scale")
    variance = _finite_array(scaler["variance"], (26,), "scaler variance")
    if np.any(scale <= 0) or np.any(variance < 0) or np.any(scale[variance == 0] != 1):
        raise GMMMonitorError("scaler scales or variances are invalid")
    if type(artifact["input_hashes"]) is not dict:
        raise GMMMonitorError("artifact input hashes are invalid")
    for name, digest in artifact["input_hashes"].items():
        try:
            baselines._lowercase_sha256(digest, name)
        except baselines.BaselineError as exc:
            raise GMMMonitorError(str(exc)) from exc
    _validate_mixture(artifact["mixture"])
    return artifact


def fit_training_mixture(features) -> tuple[dict, list[dict]]:
    """Fit the scaler and all six candidates on training features only."""
    matrix = _matrix(features)
    if len(matrix) < 6:
        raise GMMMonitorError("training requires at least six rows")
    _require_runtime()
    try:
        with (
            warnings.catch_warnings(),
            np.errstate(all="raise"),
            threadpoolctl.threadpool_limits(limits=1),
        ):
            warnings.simplefilter("error")
            if any(
                pool["num_threads"] != 1 for pool in threadpoolctl.threadpool_info()
            ):
                raise GMMMonitorError(
                    "GMM runtime requires one thread per numeric pool"
                )
            scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
            scaled = _matrix(scaler.fit_transform(matrix))
            candidates, selected, best_bic = [], None, float("inf")
            for k in range(1, 7):
                mixture = GaussianMixture(n_components=k, **_MIXTURE_PARAMETERS)
                # Vanishing component likelihoods/responsibilities are expected in EM.
                with np.errstate(under="ignore"):
                    mixture.fit(scaled)
                    bic = float(mixture.bic(scaled))
                state = _mixture_state(mixture)
                _validate_mixture(state)
                if not np.isfinite(bic):
                    raise GMMMonitorError("training BIC must be finite")
                candidates.append(
                    {
                        "components": k,
                        "bic": bic,
                        "converged": True,
                        "n_iter": state["n_iter"],
                    }
                )
                if bic < best_bic:
                    selected, best_bic = state, bic
            artifact = {
                "schema_version": 1,
                "contract_id": CONTRACT_ID,
                "features": list(GMM_FEATURE_NAMES),
                "dtype": "float64",
                "input_hashes": {},
                "mixture": selected,
                "scaler": {
                    "mean": scaler.mean_.tolist(),
                    "scale": scaler.scale_.tolist(),
                    "variance": scaler.var_.tolist(),
                    "n_samples_seen": int(scaler.n_samples_seen_),
                    "n_features_in": int(scaler.n_features_in_),
                },
            }
            return load_gmm_artifact_bytes(_canonical_json_bytes(artifact)), candidates
    except (Warning, FloatingPointError, ValueError, RuntimeError) as exc:
        if isinstance(exc, GMMMonitorError):
            raise
        raise GMMMonitorError(f"GMM training stopped: {exc}") from exc


def score_feature_matrix(features, artifact: dict) -> np.ndarray:
    """Restore validated portable state and use the frozen sklearn likelihood."""
    _require_runtime()
    artifact = load_gmm_artifact_bytes(_canonical_json_bytes(artifact))
    matrix = _matrix(features)
    scaler, mixture = artifact["scaler"], artifact["mixture"]
    restored = GaussianMixture(
        n_components=mixture["components"], **_MIXTURE_PARAMETERS
    )
    restored.weights_ = np.asarray(mixture["weights"], dtype=np.float64)
    restored.means_ = np.asarray(mixture["means"], dtype=np.float64)
    restored.covariances_ = np.asarray(mixture["variances"], dtype=np.float64)
    restored.precisions_ = np.asarray(mixture["precisions"], dtype=np.float64)
    restored.precisions_cholesky_ = np.asarray(
        mixture["precisions_cholesky"], dtype=np.float64
    )
    restored.n_features_in_ = 26
    try:
        with (
            warnings.catch_warnings(),
            np.errstate(all="raise"),
            threadpoolctl.threadpool_limits(limits=1),
        ):
            warnings.simplefilter("error")
            scaled = (matrix - np.asarray(scaler["mean"])) / np.asarray(scaler["scale"])
            with np.errstate(under="ignore"):
                nll = -restored.score_samples(scaled)
            return _finite_array(nll, (len(matrix),), "negative log likelihood")
    except (Warning, FloatingPointError, ValueError) as exc:
        if isinstance(exc, GMMMonitorError):
            raise
        raise GMMMonitorError(f"GMM likelihood scoring stopped: {exc}") from exc


def _load_inputs(input_paths, output_dir, summary_path, policy):
    if type(policy) is not _InputHashPolicy:
        raise GMMMonitorError("input hash policy is invalid")
    # Reject symlink components, including ancestors, before resolving destinations.
    for path in (*input_paths.values(), output_dir, summary_path):
        if any(part.is_symlink() for part in (path, *path.parents)):
            raise GMMMonitorError("input and output paths must not contain symlinks")
    transformer_pipeline._validate_paths(
        input_paths=input_paths, output_dir=output_dir, summary_path=summary_path
    )
    expected = {
        key.removesuffix("_sha256"): value for key, value in asdict(policy).items()
    }
    for role, digest in expected.items():
        baselines._lowercase_sha256(digest, role)
    with ExitStack() as stack:
        streams, stability = baselines._open_input_streams(input_paths, stack)
        snapshots = {role: stream.read() for role, stream in streams.items()}
        observed = {
            role: sha256(content).hexdigest() for role, content in snapshots.items()
        }
        for role, digest in expected.items():
            if observed[role] != digest:
                raise GMMMonitorError(
                    f"{role} SHA-256 mismatch: expected {digest}, observed {observed[role]}"
                )
        baselines._require_stable_streams(streams, stability)
    return snapshots, observed


def _partition(content, split, preparation):
    features, _, _, ordinals, counts = baselines._load_partition(
        io.BytesIO(content),
        split=split,
        declared=preparation["splits"][split],
        source_csv_sha256=preparation["source_csv_sha256"],
    )
    records = [_load_json(line, "partition record") for line in content.splitlines()]
    return {
        "features": features,
        "ordinals": ordinals,
        "class_counts": counts,
        "raw_urls": tuple(record["raw_url"] for record in records),
        "domains": tuple(
            _normalize_domain(record["registrable_domain"]) for record in records
        ),
        "record_ids": tuple(record["record_id"] for record in records),
    }


def _build_features(partition, stage1):
    probabilities = np.asarray(
        stage1.score_urls(partition["raw_urls"]), dtype=np.float64
    )
    if probabilities.shape != (len(partition["raw_urls"]),) or np.any(
        (probabilities < 0) | (probabilities > 1)
    ):
        raise GMMMonitorError("stage-1 probabilities are invalid")
    return _matrix(np.column_stack((partition["features"], probabilities)))


def _publish_artifacts(*, output_dir, summary_path, contents, summary):
    temporary_output = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    temporary_summary = None
    try:
        os.chmod(temporary_output, 0o700)
        for name, content in contents.items():
            _write_file(temporary_output / name, content, 0o600)
            if (temporary_output / name).read_bytes() != content or stat.S_IMODE(
                (temporary_output / name).stat().st_mode
            ) != 0o600:
                raise GMMMonitorError("private artifact write verification failed")
        if stat.S_IMODE(temporary_output.stat().st_mode) != 0o700:
            raise GMMMonitorError("private directory permissions are invalid")
        temporary_summary = _write_summary_temp(
            summary_path, _canonical_json_bytes(summary)
        )
        _fsync_directory(temporary_output)
        output_identity = baselines._path_identity(temporary_output)
        summary_identity = baselines._path_identity(temporary_summary)
        try:
            _publish_path_without_replace(temporary_output, output_dir)
            temporary_output = None
            _fsync_directory(output_dir.parent)
            _publish_path_without_replace(temporary_summary, summary_path)
            temporary_summary = None
            _fsync_directory(summary_path.parent)
            if not transformer_pipeline._has_path_identity(
                output_dir, output_identity
            ) or not transformer_pipeline._has_path_identity(
                summary_path, summary_identity
            ):
                raise GMMMonitorError(
                    "publication destination identity changed before completion"
                )
        except BaseException:
            for source, destination, identity in (
                (temporary_summary, summary_path, summary_identity),
                (temporary_output, output_dir, output_identity),
            ):
                if source is not None and os.path.lexists(source):
                    continue
                try:
                    baselines._remove_if_identity(destination, identity)
                except BaseException:
                    pass
            for parent in dict.fromkeys((summary_path.parent, output_dir.parent)):
                try:
                    _fsync_directory(parent)
                except BaseException:
                    pass
            raise
        return summary
    finally:
        if temporary_output is not None:
            shutil.rmtree(temporary_output, ignore_errors=True)
        if temporary_summary is not None:
            temporary_summary.unlink(missing_ok=True)


def _validate_public_summary(summary):
    expected = {
        "schema_version",
        "status",
        "analysis_stage",
        "contract",
        "hypothesis_status",
        "access",
        "input_hashes",
        "artifact_hashes",
        "input_counts",
        "candidates",
        "selected_component_count",
        "threshold",
        "calibration_window_count",
        "audit_alert_count",
        "audit_window_count",
        "audit_alert_fraction",
        "false_alert_gate_met",
        "software_versions",
        "numpy_build_configuration",
        "configuration",
        "warnings",
    }
    if set(summary) != expected:
        raise GMMMonitorError("public summary fields are not allowlisted")
    forbidden = {
        "record_id",
        "record_ids",
        "raw_url",
        "registrable_domain",
        "domains",
        "weights",
        "means",
        "variances",
        "predictions",
        "window_scores",
        "input_row_positions",
    }

    def scan(value):
        if isinstance(value, dict):
            if set(value) & forbidden:
                raise GMMMonitorError("public summary contains private data")
            for child in value.values():
                scan(child)
        elif isinstance(value, list):
            for child in value:
                scan(child)

    scan(summary)
    for candidate in summary["candidates"]:
        if set(candidate) != {"components", "bic", "converged", "n_iter"}:
            raise GMMMonitorError("public summary candidate fields are not allowlisted")
    _canonical_json_bytes(summary)


def _fit_gmm_monitor(
    *,
    train_path,
    validation_path,
    preparation_summary_path,
    baseline_contract_path,
    logistic_l1_artifact_path,
    gmm_contract_path,
    output_dir,
    summary_path,
    _input_hash_policy,
):
    """Private fixture seam; the public entry point never overrides pinned hashes."""
    try:
        _require_runtime()
        input_paths = {
            "train": Path(train_path),
            "validation": Path(validation_path),
            "preparation_summary": Path(preparation_summary_path),
            "baseline_contract": Path(baseline_contract_path),
            "logistic_l1_artifact": Path(logistic_l1_artifact_path),
            "gmm_contract": Path(gmm_contract_path),
        }
        output_dir, summary_path = Path(output_dir), Path(summary_path)
        snapshots, hashes = _load_inputs(
            input_paths, output_dir, summary_path, _input_hash_policy
        )
        contract = _validate_contract(
            _load_json(snapshots["gmm_contract"], "GMM contract")
        )
        preparation = baselines._validate_preparation_summary(
            _load_json(snapshots["preparation_summary"], "preparation summary")
        )
        baseline = baselines._validate_contract(
            _load_json(snapshots["baseline_contract"], "baseline contract")
        )
        if baseline["contract_id"] != "rq1-baselines-v2":
            raise GMMMonitorError("baseline contract identity is invalid")
        if _input_hash_policy == _OFFICIAL_INPUT_HASH_POLICY:
            expected_roles = {
                key: hashes[key]
                for key in (
                    "train",
                    "validation",
                    "preparation_summary",
                    "logistic_l1_artifact",
                )
            }
            expected_roles["contract"] = hashes["baseline_contract"]
            if expected_roles != contract["inputs"]["accepted_roles"]:
                raise GMMMonitorError("official hashes differ from the GMM contract")
        for split in ("train", "validation"):
            if preparation["output_hashes"][split + ".jsonl"] != hashes[split]:
                raise GMMMonitorError(f"{split} hash differs from preparation summary")
        train, validation = (
            _partition(snapshots[split], split, preparation)
            for split in ("train", "validation")
        )
        if set(train["domains"]) & set(validation["domains"]):
            raise GMMMonitorError("registrable domain crosses train and validation")
        if train["ordinals"] & validation["ordinals"]:
            raise GMMMonitorError("record identifier crosses train and validation")
        stage1 = fixed_cascade._load_logistic_l1_artifact_bytes(
            snapshots["logistic_l1_artifact"],
            expected_sha256=hashes["logistic_l1_artifact"],
            expected_contract_sha256=hashes["baseline_contract"],
        )
        del snapshots, preparation, baseline
        allocation = allocate_validation_domains(validation["domains"])
        if any(len(indices) < 256 for indices in allocation.values()):
            raise GMMMonitorError("each stream requires at least one complete window")
        with (
            warnings.catch_warnings(),
            np.errstate(all="raise"),
            threadpoolctl.threadpool_limits(limits=1),
        ):
            warnings.simplefilter("error")
            artifact, candidates = fit_training_mixture(_build_features(train, stage1))
            artifact["input_hashes"] = hashes
            artifact = load_gmm_artifact_bytes(_canonical_json_bytes(artifact))
            nll = score_feature_matrix(_build_features(validation, stage1), artifact)
            traces = {}
            for stream, indices in allocation.items():
                ends, scores = window_scores(nll[list(indices)])
                traces[stream] = {
                    "domains": sorted({validation["domains"][i] for i in indices}),
                    "record_ids": [validation["record_ids"][i] for i in indices],
                    "input_row_positions": list(indices),
                    "window_end_positions": ends,
                    "window_scores": scores,
                }
            audit = calibrate_and_audit(
                traces["calibration"]["window_scores"], traces["audit"]["window_scores"]
            )
        private_audit = {
            "schema_version": 1,
            "contract_id": CONTRACT_ID,
            "input_hashes": hashes,
            **traces,
            "result": audit,
        }
        contents = {
            "gmm.json": _canonical_json_bytes(artifact),
            "validation-audit.json": _canonical_json_bytes(private_audit),
        }
        artifact_hashes = {
            name: sha256(contents[name]).hexdigest() for name in _PRIVATE_FILES
        }
        contents["SHA256SUMS"] = "".join(
            f"{artifact_hashes[name]}  {name}\n" for name in sorted(artifact_hashes)
        ).encode("ascii")
        counts = {
            name: {
                "rows": len(partition["raw_urls"]),
                "domain_count": len(set(partition["domains"])),
                "class_counts": partition["class_counts"],
            }
            for name, partition in (("train", train), ("validation", validation))
        }
        counts.update(
            {
                name: {
                    "rows": len(trace["record_ids"]),
                    "domain_count": len(trace["domains"]),
                    "complete_windows": len(trace["window_scores"]),
                }
                for name, trace in traces.items()
            }
        )
        summary = {
            "schema_version": 1,
            "status": "completed_development_validation",
            "analysis_stage": "development_validation_only",
            "contract": {
                "id": CONTRACT_ID,
                "sha256": hashes["gmm_contract"],
                "protocol_version": "1.10",
            },
            "hypothesis_status": contract["execution"]["hypotheses"],
            "access": contract["artifacts"]["public_summary"]["access"],
            "input_hashes": hashes,
            "artifact_hashes": artifact_hashes,
            "input_counts": counts,
            "candidates": candidates,
            "selected_component_count": artifact["mixture"]["components"],
            **audit,
            "software_versions": {
                **_software_versions(),
                "python": platform.python_version(),
            },
            "numpy_build_configuration": _numpy_build_configuration(),
            "configuration": {
                key: contract[key]
                for key in (
                    "features",
                    "scaler",
                    "validation_allocation",
                    "mixture",
                    "runtime",
                    "windows",
                    "calibration",
                    "audit",
                )
            },
            "warnings": [],
        }
        _validate_public_summary(summary)
        return _publish_artifacts(
            output_dir=output_dir,
            summary_path=summary_path,
            contents=contents,
            summary=summary,
        )
    except (
        baselines.BaselineError,
        fixed_cascade.FixedCascadeError,
        transformer_pipeline.TransformerPipelineError,
        Warning,
        FloatingPointError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        if isinstance(exc, GMMMonitorError):
            raise
        raise GMMMonitorError(str(exc)) from exc


def fit_gmm_monitor(
    *,
    train_path: Path,
    validation_path: Path,
    preparation_summary_path: Path,
    baseline_contract_path: Path,
    logistic_l1_artifact_path: Path,
    gmm_contract_path: Path,
    output_dir: Path,
    summary_path: Path,
) -> dict:
    """Run the pinned development-only method; accept paths, never tuning values."""
    return _fit_gmm_monitor(
        train_path=train_path,
        validation_path=validation_path,
        preparation_summary_path=preparation_summary_path,
        baseline_contract_path=baseline_contract_path,
        logistic_l1_artifact_path=logistic_l1_artifact_path,
        gmm_contract_path=gmm_contract_path,
        output_dir=output_dir,
        summary_path=summary_path,
        _input_hash_policy=_OFFICIAL_INPUT_HASH_POLICY,
    )
