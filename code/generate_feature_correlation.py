#!/usr/bin/env python3
"""Generate feature correlation matrix and heat map for the frozen snapshot."""

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


DROP_COLS = {"is_phishing", "timestamp", "source", "owasp_category", "atlas_tactic"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate correlation heat map from processed splits.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("Automated_Phishing_Detection_Praxis/data/configs/week1_snapshot.yml"),
        help="Path to YAML config used for preprocessing/training.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Automated_Phishing_Detection_Praxis/reports/Fig3-FeatureCorrelation.png"),
        help="Where to save the heat map image.",
    )
    parser.add_argument(
        "--matrix-output",
        type=Path,
        default=Path("Automated_Phishing_Detection_Praxis/data/processed/v2025.08.10/eval/correlation_matrix.json"),
        help="Where to save the numeric correlation matrix as JSON.",
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
    drop_cols = [col for col in DROP_COLS if col in df.columns]
    features = df.drop(columns=drop_cols, errors="ignore").copy()
    bool_cols = features.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        features[bool_cols] = features[bool_cols].astype(int)
    return features.apply(pd.to_numeric, errors="coerce").fillna(0)


def main():
    args = parse_args()
    config = load_config(args.config)
    project_root = resolve_root(args.config)

    output_cfg = config.get("output", {})
    processed_dir = (project_root / output_cfg.get("processed_dir", "data/processed/v2025.08.10")).resolve()
    train_file = processed_dir / output_cfg.get("train_file", "train_data.csv")
    val_file = processed_dir / output_cfg.get("val_file", "val_data.csv")
    test_file = processed_dir / output_cfg.get("test_file", "test_data.csv")

    frames = []
    for split_name, path in [("train", train_file), ("validation", val_file), ("test", test_file)]:
        if not path.exists():
            raise FileNotFoundError(f"Split file missing for {split_name}: {path}")
        logger.info("Loading %s split from %s", split_name, path)
        frames.append(load_features(path))

    features_df = pd.concat(frames, ignore_index=True)
    logger.info("Combined feature shape: %s", features_df.shape)

    corr_matrix = features_df.corr(numeric_only=True)
    corr_json = corr_matrix.round(6).to_dict()

    matrix_out = args.matrix_output if args.matrix_output.is_absolute() else (project_root / args.matrix_output)
    matrix_out.parent.mkdir(parents=True, exist_ok=True)
    with open(matrix_out, "w") as fp:
        json.dump(corr_json, fp, indent=2)
    logger.info("Correlation matrix saved to %s", matrix_out)

    heatmap_path = args.output if args.output.is_absolute() else (project_root / args.output)
    heatmap_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", vmin=-1, vmax=1, square=True)
    plt.title("Feature Correlation (Pearson)")
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=300)
    logger.info("Heat map saved to %s", heatmap_path)


if __name__ == "__main__":
    main()
