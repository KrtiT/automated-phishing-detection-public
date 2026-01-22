#!/usr/bin/env python3
"""
Week 3 evaluation package: calibrates decision thresholds, exports ROC curves
and confusion matrices, and runs McNemar significance tests on the Week 1
baseline models.
"""

import argparse
import json
import logging
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (confusion_matrix, precision_recall_fscore_support,
                             roc_auc_score, roc_curve)


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate saved baseline models")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("Automated_Phishing_Detection_Praxis/data/configs/week1_snapshot.yml"),
        help="Path to snapshot config",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override evaluation output directory",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        logger.warning("Config %s not found; using defaults", config_path)
        return {}
    with open(config_path, "r") as fp:
        return yaml.safe_load(fp) or {}


def resolve_root(config_path: Path) -> Path:
    resolved = config_path.resolve()
    return resolved.parents[2] if len(resolved.parents) >= 3 else resolved.parent


def load_split(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    feature_cols = [c for c in df.columns if c not in {"is_phishing", "timestamp", "source", "owasp_category", "atlas_tactic"}]
    X = df[feature_cols].copy().fillna(0)
    bool_cols = X.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        X[bool_cols] = X[bool_cols].astype(int)
    y = df["is_phishing"].astype(int)
    return X, y


def predict_proba(model, X, scaler=None):
    arr = X.to_numpy(dtype=float)
    if scaler is not None:
        arr = scaler.transform(arr)
    probs = model.predict_proba(arr)[:, 1]
    return probs


def scan_threshold(probs, y):
    best = {"threshold": 0.5, "f1": -1}
    for t in np.linspace(0.01, 0.99, 99):
        preds = (probs >= t).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y, preds, average="binary", zero_division=0
        )
        if f1 > best["f1"]:
            best = {
                "threshold": float(t),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }
    return best


def metrics_at_threshold(probs, y, threshold):
    preds = (probs >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, preds, average="binary", zero_division=0
    )
    acc = float((preds == y).mean())
    roc = roc_auc_score(y, probs)
    cm = confusion_matrix(y, preds).tolist()
    return {
        "threshold": threshold,
        "accuracy": acc,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc),
        "confusion_matrix": cm,
    }


def roc_points(probs, y):
    fpr, tpr, thresh = roc_curve(y, probs)
    return pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresh})


def mcnemar_test(pred_a, pred_b, y):
    a_correct = pred_a == y
    b_correct = pred_b == y
    b_only = np.sum(a_correct & ~b_correct)
    c_only = np.sum(~a_correct & b_correct)
    total = b_only + c_only
    if total == 0:
        return {"b_only": int(b_only), "c_only": int(c_only), "chi2": 0.0, "p_value": 1.0}
    chi2 = (abs(b_only - c_only) - 1) ** 2 / total
    p_value = 1 - math.erf(math.sqrt(chi2 / 2))
    return {
        "b_only": int(b_only),
        "c_only": int(c_only),
        "chi2": float(chi2),
        "p_value": float(p_value),
    }


def main():
    args = parse_args()
    import yaml  # lazy import to avoid global dependency

    config = load_config(args.config)
    project_root = resolve_root(args.config)
    output_cfg = config.get("output", {})
    processed_dir = (project_root / output_cfg.get("processed_dir", "data/processed/v2025.08.10")).resolve()
    models_dir = (project_root / "code" / "models").resolve()
    eval_dir = args.output or (processed_dir / "eval")
    Path(eval_dir).mkdir(parents=True, exist_ok=True)

    train_file = processed_dir / output_cfg.get("train_file", "train_data.csv")
    val_file = processed_dir / output_cfg.get("val_file", "val_data.csv")
    test_file = processed_dir / output_cfg.get("test_file", "test_data.csv")

    _, _ = load_split(train_file)  # confirm accessible (unused but ensures same columns)
    X_val, y_val = load_split(val_file)
    X_test, y_test = load_split(test_file)

    logistic_bundle = joblib.load(models_dir / "logistic_l1.pkl")
    rf_bundle = joblib.load(models_dir / "random_forest.pkl")

    logistic = logistic_bundle["model"]
    logistic_scaler = logistic_bundle.get("scaler")
    rf = rf_bundle["model"]

    log_val_probs = predict_proba(logistic, X_val, logistic_scaler)
    log_test_probs = predict_proba(logistic, X_test, logistic_scaler)
    rf_val_probs = predict_proba(rf, X_val)
    rf_test_probs = predict_proba(rf, X_test)

    log_threshold = scan_threshold(log_val_probs, y_val)
    rf_threshold = scan_threshold(rf_val_probs, y_val)

    results = {
        "logistic_l1": {
            "validation": metrics_at_threshold(log_val_probs, y_val, log_threshold["threshold"]),
            "test": metrics_at_threshold(log_test_probs, y_test, log_threshold["threshold"]),
            "threshold_search": log_threshold,
        },
        "random_forest": {
            "validation": metrics_at_threshold(rf_val_probs, y_val, rf_threshold["threshold"]),
            "test": metrics_at_threshold(rf_test_probs, y_test, rf_threshold["threshold"]),
            "threshold_search": rf_threshold,
        },
    }

    # ROC exports
    roc_points(log_test_probs, y_test).to_csv(Path(eval_dir) / "roc_logistic.csv", index=False)
    roc_points(rf_test_probs, y_test).to_csv(Path(eval_dir) / "roc_random_forest.csv", index=False)

    # McNemar (test set, using calibrated thresholds)
    log_test_preds = (log_test_probs >= log_threshold["threshold"]).astype(int)
    rf_test_preds = (rf_test_probs >= rf_threshold["threshold"]).astype(int)
    mcnemar = mcnemar_test(log_test_preds, rf_test_preds, y_test.to_numpy())
    results["mcnemar_test"] = mcnemar

    with open(Path(eval_dir) / "evaluation_summary.json", "w") as fp:
        json.dump(results, fp, indent=2)
    logger.info("Evaluation artifacts written to %s", eval_dir)


if __name__ == "__main__":
    main()
