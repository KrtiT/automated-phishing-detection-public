#!/usr/bin/env python3
"""
Runs feature-family ablation experiments for Week 3. Each scenario drops a
feature group, retrains the baseline models, and reports validation/test metrics.
"""

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             roc_auc_score)
from sklearn.preprocessing import StandardScaler


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


FEATURE_FAMILIES = {
    "length": ["url_length", "domain_length", "path_length"],
    "punctuation": ["num_dots", "num_hyphens", "num_underscores", "num_slashes", "num_params"],
    "security_flags": ["has_ip", "is_https", "has_port"],
    "ai_endpoint": ["is_ai_endpoint"],
}


def parse_args():
    parser = argparse.ArgumentParser(description="Feature ablation experiments")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("Automated_Phishing_Detection_Praxis/data/configs/week1_snapshot.yml"),
    )
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=None,
        help="Optional cap on training rows to keep runtime bounded.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with open(path, "r") as fp:
        return yaml.safe_load(fp) or {}


def resolve_root(config_path: Path) -> Path:
    resolved = config_path.resolve()
    return resolved.parents[2] if len(resolved.parents) >= 3 else resolved.parent


def load_split(path: Path, max_rows: int | None = None, seed: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    if max_rows and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=seed).reset_index(drop=True)
    drop_cols = {"is_phishing", "timestamp", "source", "owasp_category", "atlas_tactic"}
    X = df[[c for c in df.columns if c not in drop_cols]].copy().fillna(0)
    bool_cols = X.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        X[bool_cols] = X[bool_cols].astype(int)
    y = df["is_phishing"].astype(int)
    return X, y


def smote_resample(X: np.ndarray, y: np.ndarray, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2 or counts.min() == counts.max():
        return X, y
    minority = classes[np.argmin(counts)]
    deficit = counts.max() - counts.min()
    rng = np.random.default_rng(seed)
    minority_idx = np.where(y == minority)[0]
    synthetic = []
    for _ in range(deficit):
        i, j = rng.choice(minority_idx, size=2, replace=True)
        gap = rng.random()
        synthetic.append(X[i] + gap * (X[j] - X[i]))
    X_syn = np.array(synthetic)
    y_syn = np.full(deficit, minority, dtype=int)
    return np.vstack([X, X_syn]), np.concatenate([y, y_syn])


def train_models(X_train, y_train, X_val, y_val, X_test, y_test) -> dict:
    results = {}

    X_train_np = X_train.to_numpy(dtype=float)
    y_train_np = y_train.to_numpy()
    X_bal, y_bal = smote_resample(X_train_np, y_train_np)

    scaler = StandardScaler()
    X_bal_scaled = scaler.fit_transform(X_bal)

    logistic = LogisticRegression(
        penalty="l1",
        solver="saga",
        class_weight="balanced",
        max_iter=2000,
        random_state=42,
        n_jobs=-1,
    )
    logistic.fit(X_bal_scaled, y_bal)
    results["logistic_l1"] = {
        "validation": evaluate(logistic, scaler, X_val, y_val),
        "test": evaluate(logistic, scaler, X_test, y_test),
    }

    rf = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    rf.fit(X_bal, y_bal)
    results["random_forest"] = {
        "validation": evaluate(rf, None, X_val, y_val),
        "test": evaluate(rf, None, X_test, y_test),
    }
    return results


def evaluate(model, scaler, X, y) -> dict:
    arr = X.to_numpy(dtype=float)
    if scaler is not None:
        arr = scaler.transform(arr)
    preds = model.predict(arr)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, preds, average="binary", zero_division=0
    )
    metrics = {
        "accuracy": float(accuracy_score(y, preds)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(arr)[:, 1]
        metrics["roc_auc"] = float(roc_auc_score(y, probs))
    return metrics


def main():
    args = parse_args()
    config = load_config(args.config)
    project_root = resolve_root(args.config)
    output_cfg = config.get("output", {})
    processed_dir = (project_root / output_cfg.get("processed_dir", "data/processed/v2025.08.10")).resolve()
    train_file = processed_dir / output_cfg.get("train_file", "train_data.csv")
    val_file = processed_dir / output_cfg.get("val_file", "val_data.csv")
    test_file = processed_dir / output_cfg.get("test_file", "test_data.csv")

    X_train, y_train = load_split(train_file, max_rows=args.max_train_rows)
    X_val, y_val = load_split(val_file)
    X_test, y_test = load_split(test_file)

    scenarios = {"baseline": []}
    for family, cols in FEATURE_FAMILIES.items():
        scenarios[f"minus_{family}"] = cols

    all_results = {}
    for name, drop_cols in scenarios.items():
        logger.info("Running scenario %s (drop %s)", name, drop_cols or "none")
        Xt = X_train.drop(columns=drop_cols, errors="ignore")
        Xv = X_val.drop(columns=drop_cols, errors="ignore")
        Xs = X_test.drop(columns=drop_cols, errors="ignore")
        all_results[name] = train_models(Xt, y_train, Xv, y_val, Xs, y_test)

    out_path = processed_dir / "eval" / "feature_ablations.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fp:
        json.dump(all_results, fp, indent=2)
    logger.info("Ablation results saved to %s", out_path)


if __name__ == "__main__":
    main()
