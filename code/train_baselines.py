#!/usr/bin/env python3
"""
Train baseline Logistic-L1 and class-weighted Random Forest models using the
Week 1 frozen snapshot. Outputs validation/test metrics, latency stats, and
persists the trained artifacts for Chapter 4 experiments.
"""

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Week 1 baseline models")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("Automated_Phishing_Detection_Praxis/data/configs/week1_snapshot.yml"),
        help="Path to the data snapshot config",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    if config_path and config_path.exists():
        with open(config_path, "r") as fp:
            return yaml.safe_load(fp) or {}
    logger.warning("Config %s not found. Using defaults.", config_path)
    return {}


def resolve_project_root(config_path: Path) -> Path:
    resolved = config_path.resolve()
    if len(resolved.parents) >= 2:
        # .../data/configs/<file> -> parents[0]=configs, parents[1]=data, parents[2]=repo root
        return resolved.parents[2]
    return Path.cwd()


def resolve_path(base_root: Path, relative_path: str) -> Path:
    path = Path(relative_path)
    if path.is_absolute():
        return path
    return (base_root / path).resolve()


def load_split(path: Path, feature_cols: list) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    X = df[feature_cols].copy().fillna(0)
    bool_cols = X.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        X[bool_cols] = X[bool_cols].astype(int)
    y = df["is_phishing"].astype(int)
    return X, y


def smote_resample(X: np.ndarray, y: np.ndarray, random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return X, y
    majority_class = classes[np.argmax(counts)]
    minority_class = classes[np.argmin(counts)]
    deficit = counts.max() - counts.min()
    if deficit <= 0:
        return X, y
    rng = np.random.default_rng(random_state)
    minority_indices = np.where(y == minority_class)[0]
    synthetic_samples = []
    for _ in range(deficit):
        i, j = rng.choice(minority_indices, size=2, replace=True)
        xi, xj = X[i], X[j]
        gap = rng.random()
        synthetic_samples.append(xi + gap * (xj - xi))
    X_syn = np.array(synthetic_samples)
    X_res = np.vstack([X, X_syn])
    y_res = np.concatenate([y, np.full(deficit, minority_class)])
    return X_res, y_res


def _to_array(X):
    if isinstance(X, pd.DataFrame):
        return X.to_numpy(dtype=float)
    return np.asarray(X, dtype=float)


def evaluate(model, X, y, scaler=None) -> dict:
    X_arr = _to_array(X)
    if scaler is not None:
        X_proc = scaler.transform(X_arr)
    else:
        X_proc = X_arr
    y_pred = model.predict(X_proc)
    metrics = {}
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, y_pred, average="binary", zero_division=0
    )
    metrics["accuracy"] = accuracy_score(y, y_pred)
    metrics["precision"] = precision
    metrics["recall"] = recall
    metrics["f1"] = f1
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_proc)[:, 1]
        try:
            metrics["roc_auc"] = roc_auc_score(y, y_proba)
        except ValueError:
            metrics["roc_auc"] = None
    metrics["confusion_matrix"] = confusion_matrix(y, y_pred).tolist()
    return metrics


def measure_latency(model, sample: np.ndarray, scaler=None, runs: int = 200) -> dict:
    latencies = []
    sample_arr = _to_array(sample)
    for _ in range(runs):
        start = time.perf_counter()
        if scaler is not None:
            data = scaler.transform(sample_arr)
        else:
            data = sample_arr
        _ = model.predict(data)
        latencies.append((time.perf_counter() - start) * 1000)
    return {
        "avg_ms": float(np.mean(latencies)),
        "p95_ms": float(np.percentile(latencies, 95)),
    }


def main():
    args = parse_args()
    config = load_config(args.config)
    project_root = resolve_project_root(args.config)

    training_cfg = config.get("training", {})
    use_smote = bool(training_cfg.get("use_smote", True))
    logistic_penalty = str(training_cfg.get("logistic_penalty", "l1")).lower()
    logistic_max_iter = int(training_cfg.get("logistic_max_iter", 4000))
    logistic_max_train_rows = training_cfg.get("logistic_max_train_rows")
    logistic_max_train_rows = int(logistic_max_train_rows) if logistic_max_train_rows else None
    rf_max_train_rows = training_cfg.get("rf_max_train_rows")
    rf_max_train_rows = int(rf_max_train_rows) if rf_max_train_rows else None

    output_cfg = config.get("output", {})
    processed_dir = resolve_path(project_root, output_cfg.get("processed_dir", "data/processed/v2025.08.10"))
    train_file = processed_dir / output_cfg.get("train_file", "train_data.csv")
    val_file = processed_dir / output_cfg.get("val_file", "val_data.csv")
    test_file = processed_dir / output_cfg.get("test_file", "test_data.csv")

    if not train_file.exists():
        raise FileNotFoundError(f"Training data not found at {train_file}")

    drop_cols = {"is_phishing", "timestamp", "source", "owasp_category", "atlas_tactic"}
    sample_df = pd.read_csv(train_file, nrows=1)
    feature_cols = [col for col in sample_df.columns if col not in drop_cols]
    logger.info("Using feature columns: %s", feature_cols)

    X_train, y_train = load_split(train_file, feature_cols)
    X_val, y_val = load_split(val_file, feature_cols)
    X_test, y_test = load_split(test_file, feature_cols)

    X_train_np = X_train.to_numpy(dtype=float)
    y_train_np = y_train.to_numpy()
    if use_smote:
        X_train_bal, y_train_bal = smote_resample(X_train_np, y_train_np, random_state=42)
        logger.info("SMOTE applied: %d -> %d samples", len(y_train), len(y_train_bal))
    else:
        X_train_bal, y_train_bal = X_train_np, y_train_np
        logger.info("SMOTE disabled; training on %d samples", len(y_train_bal))

    if logistic_max_train_rows and logistic_max_train_rows < len(y_train_bal):
        rng = np.random.default_rng(42)
        sample_idx = rng.choice(len(y_train_bal), size=logistic_max_train_rows, replace=False)
        X_train_bal = X_train_bal[sample_idx]
        y_train_bal = y_train_bal[sample_idx]
        logger.info("Downsampled Logistic training to %d rows", logistic_max_train_rows)

    models_dir = project_root / "code" / "models"
    if not models_dir.exists():
        models_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "timestamp": datetime.utcnow().isoformat(),
        "config": str(args.config),
        "processed_dir": str(processed_dir),
        "feature_columns": feature_cols,
        "training": {
            "use_smote": use_smote,
            "logistic_penalty": logistic_penalty,
            "logistic_max_iter": logistic_max_iter,
            "logistic_max_train_rows": logistic_max_train_rows,
            "rf_max_train_rows": rf_max_train_rows,
        },
        "models": {},
    }

    # Logistic L1
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_bal)
    if logistic_penalty not in {"l1", "l2"}:
        raise ValueError(f"Unsupported logistic_penalty: {logistic_penalty}")
    logistic = LogisticRegression(
        penalty=logistic_penalty,
        solver="saga",
        class_weight="balanced",
        max_iter=logistic_max_iter,
        n_jobs=-1,
        random_state=42,
    )
    logistic.fit(X_train_scaled, y_train_bal)
    logger.info("Logistic (%s) trained on %d samples", logistic_penalty, len(y_train_bal))

    logistic_metrics = {
        "validation": evaluate(logistic, X_val, y_val, scaler=scaler),
        "test": evaluate(logistic, X_test, y_test, scaler=scaler),
    }
    sample = X_val.iloc[[0]].to_numpy(dtype=float)
    logistic_metrics["latency_ms"] = measure_latency(logistic, sample, scaler=scaler)
    results["models"]["logistic_l1"] = logistic_metrics

    joblib.dump(
        {
            "model": logistic,
            "scaler": scaler,
            "feature_columns": feature_cols,
            "training": {"penalty": logistic_penalty, "max_iter": logistic_max_iter},
        },
        models_dir / "logistic_l1.pkl",
    )

    # Random Forest
    rf_train_X = X_train_bal
    rf_train_y = y_train_bal
    if rf_max_train_rows and rf_max_train_rows < len(rf_train_y):
        rng = np.random.default_rng(42)
        sample_idx = rng.choice(len(rf_train_y), size=rf_max_train_rows, replace=False)
        rf_train_X = rf_train_X[sample_idx]
        rf_train_y = rf_train_y[sample_idx]
        logger.info("Downsampled RF training to %d rows", rf_max_train_rows)

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    rf.fit(rf_train_X, rf_train_y)
    logger.info("Random Forest trained on %d samples", len(rf_train_y))

    rf_metrics = {
        "validation": evaluate(rf, X_val, y_val),
        "test": evaluate(rf, X_test, y_test),
    }
    rf_metrics["latency_ms"] = measure_latency(rf, sample)
    results["models"]["random_forest"] = rf_metrics

    joblib.dump(
        {"model": rf, "feature_columns": feature_cols, "training": {"max_train_rows": rf_max_train_rows}},
        models_dir / "random_forest.pkl",
    )

    metrics_path = processed_dir / "baseline_metrics.json"
    with open(metrics_path, "w") as fp:
        json.dump(results, fp, indent=2)
    logger.info("Metrics saved to %s", metrics_path)


if __name__ == "__main__":
    main()
