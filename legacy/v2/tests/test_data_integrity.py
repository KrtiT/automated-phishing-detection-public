from pathlib import Path
import sys

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = BASE_DIR / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.append(str(CODE_DIR))

import server  # noqa: E402


PROCESSED_DIR = BASE_DIR / "data" / "processed" / "v2025.08.10"


def test_processed_files_exist():
    for name in ["train_data.csv", "val_data.csv", "test_data.csv"]:
        path = PROCESSED_DIR / name
        assert path.exists(), f"Missing processed file: {path}"


def test_feature_columns_match_server_expectations():
    df = pd.read_csv(PROCESSED_DIR / "test_data.csv")
    for feature in server.FEATURES:
        assert feature in df.columns, f"Missing feature {feature} in processed data"

    sample = df[server.FEATURES].fillna(0).iloc[0]
    assert len(sample.tolist()) == len(server.FEATURES)
