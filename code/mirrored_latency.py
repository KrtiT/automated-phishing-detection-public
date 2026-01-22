#!/usr/bin/env python3
"""Replays held-out requests through the trained models to approximate mirrored traffic latency."""

import argparse
import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Measure mirrored latency")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("Automated_Phishing_Detection_Praxis/data/configs/week1_snapshot.yml"),
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with open(path, "r") as fp:
        return yaml.safe_load(fp) or {}


def resolve_root(config_path: Path) -> Path:
    resolved = config_path.resolve()
    return resolved.parents[2] if len(resolved.parents) >= 3 else resolved.parent


def load_features(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    drop_cols = {"is_phishing", "timestamp", "source", "owasp_category", "atlas_tactic"}
    X = df[[c for c in df.columns if c not in drop_cols]].copy().fillna(0)
    bool_cols = X.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        X[bool_cols] = X[bool_cols].astype(int)
    return X


def measure(model, scaler, X: np.ndarray) -> dict:
    latencies = []
    for row in X:
        start = time.perf_counter()
        sample = row.reshape(1, -1)
        if scaler is not None:
            sample = scaler.transform(sample)
        _ = model.predict(sample)
        latencies.append((time.perf_counter() - start) * 1000)
    return {
        "avg_ms": float(np.mean(latencies)),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "max_ms": float(np.max(latencies)),
        "samples": len(latencies),
    }


def main():
    args = parse_args()
    config = load_config(args.config)
    project_root = resolve_root(args.config)
    output_cfg = config.get("output", {})
    processed_dir = (project_root / output_cfg.get("processed_dir", "data/processed/v2025.08.10")).resolve()
    test_file = processed_dir / output_cfg.get("test_file", "test_data.csv")
    X_test = load_features(test_file)

    models_dir = project_root / "code" / "models"
    logistic_bundle = joblib.load(models_dir / "logistic_l1.pkl")
    rf_bundle = joblib.load(models_dir / "random_forest.pkl")

    logistic = logistic_bundle["model"]
    logistic_scaler = logistic_bundle.get("scaler")
    rf = rf_bundle["model"]

    mirrored = {
        "logistic_l1": measure(logistic, logistic_scaler, X_test.to_numpy(dtype=float)),
        "random_forest": measure(rf, None, X_test.to_numpy(dtype=float)),
    }

    out_dir = processed_dir / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "mirrored_latency.json", "w") as fp:
        json.dump(mirrored, fp, indent=2)
    logger.info("Mirrored latency metrics saved to %s", out_dir / "mirrored_latency.json")


if __name__ == "__main__":
    main()
