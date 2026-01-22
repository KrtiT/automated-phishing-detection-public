#!/usr/bin/env python3
"""
Data Preprocessing Pipeline for AI Phishing Detection
Author: Krti Tallam
Date: August 14, 2025 (updated)
Description: Freezes data snapshots, generates feature tables, and produces
chronological splits for the AI inference phishing detector.
"""

import argparse
import json
import logging
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import yaml


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> Dict:
    if not config_path:
        return {}
    if not config_path.exists():
        logger.warning("Config file %s not found. Falling back to defaults.", config_path)
        return {}
    with open(config_path, "r") as fp:
        return yaml.safe_load(fp) or {}


class PhishingDataPreprocessor:
    """Preprocesses phishing data from multiple sources for AI endpoint detection."""

    DEFAULT_WINDOW = {
        "start": "2025-05-12T00:00:00Z",
        "end": "2025-08-03T23:59:59Z",
    }

    DEFAULT_SPLITS = {
        "train": {"start": "2025-05-12T00:00:00Z", "end": "2025-06-22T23:59:59Z"},
        "validation": {"start": "2025-06-23T00:00:00Z", "end": "2025-07-13T23:59:59Z"},
        "test": {"start": "2025-07-14T00:00:00Z", "end": "2025-08-03T23:59:59Z"},
    }

    DEFAULT_SOURCES = {
        "phishtank": {
            "type": "feed",
            "description": "Verified phishing URLs",
            "snapshot_file": None,
        },
        "openphish": {
            "type": "feed",
            "description": "OpenPhish daily feed",
            "snapshot_file": None,
        },
        "urlhaus": {
            "type": "feed",
            "description": "URLHaus recent CSV",
            "snapshot_file": None,
        },
    }

    AI_ENDPOINT_PATTERNS = (
        "/v1/chat/completions",
        "/v1/completions",
        "/v1/embeddings",
        "/v1/models",
        "/inference",
        "/predict",
    )

    def __init__(
        self,
        data_dir: Path,
        config: Dict,
        config_path: Path = None,
        max_rows: int = None,
        benign_multiplier: float = None,
    ):
        self.config = config or {}
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.config_path = config_path
        if self.config_path and len(self.config_path.parents) >= 3:
            self.project_root = self.config_path.parents[2]
        else:
            self.project_root = Path.cwd()

        output_cfg = self.config.get("output", {})
        processed_dir = output_cfg.get("processed_dir", self.data_dir / "processed")
        self.processed_dir = self._resolve_path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        stats_file = output_cfg.get("stats_file", self.processed_dir / "dataset_stats.json")
        self.stats_file = self._resolve_path(stats_file)

        self.sources = self.config.get("sources", self.DEFAULT_SOURCES)
        self.window_cfg = self.config.get("window", self.DEFAULT_WINDOW)
        self.splits_cfg = self.config.get("splits", self.DEFAULT_SPLITS)
        self.split_mode = str(self.config.get("split_mode", "time")).lower()
        if self.split_mode not in {"time", "random"}:
            raise ValueError(f"Unsupported split_mode: {self.split_mode}")
        self.split_ratios = self.config.get(
            "split_ratios",
            {"train": 0.7, "validation": 0.15, "test": 0.15},
        )
        self.label_policy = self.config.get("label_policy", {})
        self.feature_profile = str(self.config.get("feature_profile", "url_plus_telemetry")).lower()
        if self.feature_profile not in {"url_only", "url_plus_telemetry"}:
            raise ValueError(f"Unsupported feature_profile: {self.feature_profile}")
        self.random_state = self.config.get("seed", 42)
        self.max_rows = max_rows or self.config.get("max_rows")
        self.benign_multiplier = benign_multiplier
        if self.benign_multiplier is None:
            self.benign_multiplier = float(self.config.get("benign_multiplier", 1.0))

        self._dropped_outside_window = 0
        self._dropped_outside_splits = 0
        np.random.seed(self.random_state)
        random.seed(self.random_state)

    def _resolve_path(self, maybe_path):
        path = Path(maybe_path)
        if path.is_absolute():
            return path
        return (self.project_root / path).resolve()

    def _normalize_boundary(self, value) -> pd.Timestamp:
        ts = pd.to_datetime(value)
        if getattr(ts, "tzinfo", None) is not None:
            try:
                ts = ts.tz_convert("UTC")
            except TypeError:
                ts = ts.tz_localize("UTC")
            return ts.tz_localize(None)
        return ts

    def download_datasets(self) -> Dict[str, pd.DataFrame]:
        datasets = {}
        for name, meta in self.sources.items():
            df = self._load_snapshot(name, meta)
            df["source"] = name
            datasets[name] = df
            logger.info("%s: %d samples ready", name, len(df))
        return datasets

    def _load_snapshot(self, source_name: str, meta: Dict) -> pd.DataFrame:
        if meta.get("type") in {"jailbreak_reference", "adversarial_prompts"}:
            logger.info("Skipping %s (reference material)", source_name)
            return pd.DataFrame(columns=["url", "target", "verified", "timestamp", "is_phishing", "tokens_prompt", "tokens_completion", "latency_ms"])
        snapshot_file = meta.get("snapshot_file")
        if snapshot_file:
            path = self._resolve_path(snapshot_file)
            if path.exists():
                try:
                    if meta.get("streaming"):
                        df = self._read_large_snapshot(path, self.max_rows, meta)
                    elif path.suffix in {".csv", ".txt"}:
                        df = pd.read_csv(path)
                    elif path.suffix in {".json"}:
                        df = pd.read_json(path)
                    elif path.suffix in {".jsonl"}:
                        df = pd.read_json(path, lines=True)
                    else:
                        df = pd.read_csv(path)
                    return self._normalize_raw_df(df, source_name, meta)
                except Exception as exc:
                    logger.warning("Failed to load %s (%s): %s", source_name, path, exc)

        logger.info("%s snapshot not found. Generating mock dataset.", source_name)
        return self._generate_mock_dataset(source_name, meta.get("type", "feed"))

    def _read_large_snapshot(self, path: Path, limit: int, meta: Dict) -> pd.DataFrame:
        chunks = []
        rows_read = 0
        chunk_size = int(meta.get("chunk_size", 500_000))
        read_kwargs = {"chunksize": chunk_size, "comment": "#", "low_memory": False}
        column_names = meta.get("column_names")
        if column_names:
            read_kwargs.update({"names": column_names, "header": None})
        for chunk in pd.read_csv(path, **read_kwargs):
            if limit:
                remaining = limit - rows_read
                if remaining <= 0:
                    break
                chunk = chunk.iloc[:remaining]
            chunks.append(chunk)
            rows_read += len(chunk)
            if limit and rows_read >= limit:
                break
        if not chunks:
            return pd.DataFrame()
        df = pd.concat(chunks, ignore_index=True)
        logger.info("Loaded %d rows from large snapshot %s", len(df), path)
        return df

    def _normalize_raw_df(self, df: pd.DataFrame, source_name: str, meta: Dict) -> pd.DataFrame:
        df = df.copy()
        df.columns = [str(col).strip().lower() for col in df.columns]
        if "url" not in df.columns:
            domain_col = meta.get("domain_column")
            url_template = meta.get("url_template")
            if url_template and domain_col and domain_col in df.columns:
                df["url"] = df[domain_col].astype(str).map(lambda d: url_template.format(domain=d))
            else:
                for candidate in ["link", "phish_url", "domain"]:
                    if candidate in df.columns:
                        df["url"] = df[candidate]
                        break
        if "url" not in df.columns:
            raise ValueError(f"Dataset {source_name} missing URL column")

        if "timestamp" not in df.columns:
            for candidate in ["date", "dateadded", "submission_time"]:
                if candidate in df.columns:
                    df["timestamp"] = df[candidate]
                    break
        if "timestamp" not in df.columns:
            df["timestamp"] = pd.NaT

        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        if meta.get("fill_missing_timestamps", True):
            df["timestamp"] = df["timestamp"].fillna(pd.Timestamp(self.window_cfg["start"]))

        if "target" not in df.columns:
            df["target"] = source_name.replace("_", " ")

        if "verified" not in df.columns:
            df["verified"] = True

        default_label = meta.get("default_label", 1 if meta.get("type") != "telemetry" else 0)
        if "is_phishing" in df.columns:
            df["is_phishing"] = df["is_phishing"].astype(int)
        else:
            df["is_phishing"] = default_label

        optional_cols = ["tokens_prompt", "tokens_completion", "latency_ms"]
        for col in optional_cols:
            if col not in df.columns:
                df[col] = 0
            else:
                df[col] = df[col].fillna(0)

        return df[["url", "target", "verified", "timestamp", "is_phishing", "tokens_prompt", "tokens_completion", "latency_ms"]]

    def _generate_mock_dataset(self, source_name: str, source_type: str) -> pd.DataFrame:
        base_domains = [
            "api.fake-openai.com",
            "login.model-internal.net",
            "secure-llm.example.org",
            "edge.prompt-hijack.biz",
        ]
        paths = [
            "/v1/completions",
            "/inference",
            "/extract",
            "/models/download",
            "/prompt/override",
        ]
        timestamps = pd.date_range(
            start=self.window_cfg["start"],
            end=self.window_cfg["end"],
            periods=120,
        )
        rows = []
        for idx, ts in enumerate(timestamps):
            domain = random.choice(base_domains)
            path = random.choice(paths)
            url = f"https://{domain}{path}?campaign={source_name}{idx}"
            rows.append(
                {
                    "url": url,
                    "target": f"{source_type}-campaign",
                    "verified": True,
                    "timestamp": ts,
                    "is_phishing": 1,
                }
            )
        return pd.DataFrame(rows)

    def augment_for_ai_endpoints(self, df: pd.DataFrame) -> pd.DataFrame:
        ai_patterns = [
            "/v1/models",
            "/inference",
            "/predict",
            "/complete",
            "/embeddings",
            "/classify",
            "/generate",
            "/api/v1",
        ]

        df = df.copy()
        df["is_ai_endpoint"] = df["url"].apply(lambda x: any(patt in x for patt in ai_patterns))

        synthetic_urls = []
        synthetic_timestamps = []
        window_start = pd.to_datetime(self.window_cfg["start"])
        window_end = pd.to_datetime(self.window_cfg["end"])
        window_delta = (window_end - window_start).days or 1
        base_domains = [
            "gpt-secure-check.io",
            "anthropic-review.net",
            "vertex-ai-security.org",
            "huggingface-policy.com",
            "cohere-billing.net",
            "openai-updates.help",
        ]
        for _ in range(20):
            domain = random.choice(base_domains)
            pattern = random.choice(ai_patterns)
            scheme = random.choice(["http", "https"])
            synthetic_urls.append(f"{scheme}://{domain}{pattern}?id={random.randint(100,999)}")
            synthetic_timestamps.append(window_start + pd.to_timedelta(random.randint(0, window_delta), unit="D"))

        ai_examples = pd.DataFrame(
            {
                "url": synthetic_urls + [
                    "https://fake-gpt.malicious.com/v1/completions",
                    "http://phishing-bert.net/inference/classify",
                    "https://evil-ai-api.com/models/extract",
                    "http://malicious-llm.org/v1/chat/completions",
                ],
                "target": ["Synthetic"] * len(synthetic_urls) + ["ChatGPT", "BERT", "Model Extraction", "LLM Service"],
                "is_ai_endpoint": True,
                "verified": True,
                "timestamp": synthetic_timestamps + [pd.to_datetime("2025-07-25"), pd.to_datetime("2025-07-26"), pd.to_datetime("2025-07-27"), pd.to_datetime("2025-07-28")],
                "is_phishing": 1,
            }
        )

        return pd.concat([df, ai_examples], ignore_index=True)

    def extract_url_features(self, url: str) -> Dict:
        from urllib.parse import urlparse

        parsed = urlparse(url)
        features = {
            "url_length": len(url),
            "domain_length": len(parsed.netloc),
            "path_length": len(parsed.path),
            "num_dots": url.count("."),
            "num_hyphens": url.count("-"),
            "num_underscores": url.count("_"),
            "num_slashes": url.count("/"),
            "has_ip": bool(re.match(r"\d+\.\d+\.\d+\.\d+", parsed.netloc)),
            "is_https": parsed.scheme == "https",
            "has_port": bool(parsed.port),
            "num_params": len(parsed.query.split("&")) if parsed.query else 0,
        }
        return features

    def _url_feature_frame(self, urls: pd.Series) -> pd.DataFrame:
        urls = urls.fillna("").astype(str)
        netloc = urls.str.extract(r"^[a-zA-Z]+://([^/]+)", expand=False).fillna("")
        host = netloc.str.replace(r":\d+$", "", regex=True)
        path = urls.str.extract(r"^[a-zA-Z]+://[^/]+([^?#]*)", expand=False).fillna("")
        query = urls.str.extract(r"\?([^#]*)", expand=False).fillna("")

        return pd.DataFrame(
            {
                "url_length": urls.str.len().astype(int),
                "domain_length": netloc.str.len().astype(int),
                "path_length": path.str.len().astype(int),
                "num_dots": urls.str.count(r"\.").astype(int),
                "num_hyphens": urls.str.count("-").astype(int),
                "num_underscores": urls.str.count("_").astype(int),
                "num_slashes": urls.str.count("/").astype(int),
                "has_ip": host.str.match(r"^\d+\.\d+\.\d+\.\d+$", na=False),
                "is_https": urls.str.startswith("https://"),
                "has_port": netloc.str.contains(r":\d+$", regex=True),
                "num_params": (query.str.count("&") + (query != "").astype(int)).astype(int),
            }
        )

    def create_feature_matrix(self, df: pd.DataFrame, source_name: str) -> pd.DataFrame:
        df = df.copy()
        features = self._url_feature_frame(df["url"])

        if "is_ai_endpoint" in df.columns:
            is_ai_endpoint = df["is_ai_endpoint"].fillna(False).astype(bool)
        else:
            pattern = "|".join(re.escape(p) for p in self.AI_ENDPOINT_PATTERNS)
            is_ai_endpoint = df["url"].fillna("").astype(str).str.contains(pattern, regex=True)

        if self.feature_profile == "url_only":
            prompt_tokens = pd.Series(0, index=df.index)
            completion_tokens = pd.Series(0, index=df.index)
            latency_ms = pd.Series(0, index=df.index)
        else:
            prompt_tokens = df.get("tokens_prompt", 0).fillna(0)
            completion_tokens = df.get("tokens_completion", 0).fillna(0)
            latency_ms = df.get("latency_ms", 0).fillna(0)

        features["is_phishing"] = df.get("is_phishing", 1).fillna(1).astype(int)
        features["is_ai_endpoint"] = is_ai_endpoint
        features["timestamp"] = df.get("timestamp", pd.Timestamp.now())
        features["source"] = source_name
        features["prompt_tokens"] = prompt_tokens.astype(float)
        features["completion_tokens"] = completion_tokens.astype(float)
        features["latency_ms"] = latency_ms.astype(float)
        return features

    def generate_legitimate_samples(self, n_samples: int) -> pd.DataFrame:
        legitimate_domains = [
            "api.openai.com",
            "api.anthropic.com",
            "api.cohere.ai",
            "api.huggingface.co",
            "vertex.ai.google.com",
            "api.together.xyz",
        ]
        legitimate_paths = [
            "/v1/completions",
            "/v1/chat/completions",
            "/v1/embeddings",
            "/v1/models",
            "/inference",
            "/predict",
        ]
        if n_samples <= 0:
            return pd.DataFrame(columns=[
                "url_length",
                "domain_length",
                "path_length",
                "num_dots",
                "num_hyphens",
                "num_underscores",
                "num_slashes",
                "has_ip",
                "is_https",
                "has_port",
                "num_params",
                "is_phishing",
                "is_ai_endpoint",
                "timestamp",
                "source",
                "prompt_tokens",
                "completion_tokens",
                "latency_ms",
            ])

        rng = np.random.default_rng(self.random_state)
        timestamps = pd.date_range(
            start=self.window_cfg["start"],
            end=self.window_cfg["end"],
            periods=n_samples,
        )
        domains = rng.choice(legitimate_domains, size=n_samples, replace=True)
        paths = rng.choice(legitimate_paths, size=n_samples, replace=True)
        schemes = np.where(rng.random(n_samples) > 0.15, "https", "http")
        urls = pd.Series(schemes, dtype=str) + "://" + pd.Series(domains, dtype=str) + pd.Series(paths, dtype=str)

        if self.feature_profile == "url_only":
            prompt_tokens = np.zeros(n_samples, dtype=int)
            completion_tokens = np.zeros(n_samples, dtype=int)
            latency_ms = np.zeros(n_samples, dtype=int)
        else:
            prompt_tokens = rng.integers(50, 600, size=n_samples, dtype=int)
            completion_tokens = rng.integers(20, 400, size=n_samples, dtype=int)
            latency_ms = rng.integers(40, 210, size=n_samples, dtype=int)

        raw = pd.DataFrame(
            {
                "url": urls,
                "timestamp": timestamps,
                "is_phishing": 0,
                "is_ai_endpoint": True,
                "tokens_prompt": prompt_tokens,
                "tokens_completion": completion_tokens,
                "latency_ms": latency_ms,
            }
        )
        return self.create_feature_matrix(raw, "benign")

    def _ensure_timestamp(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "timestamp" not in df.columns:
            df["timestamp"] = pd.NaT

        timestamps = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        missing = timestamps.isna()
        if missing.any():
            fill_values = pd.date_range(
                start=self.window_cfg["start"],
                end=self.window_cfg["end"],
                periods=int(missing.sum()),
            )
            timestamps = timestamps.copy()
            timestamps.loc[missing] = fill_values
        if timestamps.dt.tz is not None:
            df["timestamp"] = timestamps.dt.tz_convert("UTC").dt.tz_localize(None)
        else:
            df["timestamp"] = timestamps
        return df.sort_values("timestamp").reset_index(drop=True)

    def _filter_to_window(self, df: pd.DataFrame) -> pd.DataFrame:
        start = self._normalize_boundary(self.window_cfg["start"])
        end = self._normalize_boundary(self.window_cfg["end"])
        mask = (df["timestamp"] >= start) & (df["timestamp"] <= end)
        filtered = df[mask].copy()
        dropped = len(df) - len(filtered)
        self._dropped_outside_window += dropped
        if dropped > 0:
            logger.info("Filtered %d rows outside window [%s, %s]", dropped, start, end)
        return filtered.reset_index(drop=True)

    def _assign_taxonomy(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        taxonomy = self.label_policy.get("taxonomy", {})
        owasp_raw = taxonomy.get("owasp", [])
        if isinstance(owasp_raw, str):
            owasp_values = [val.strip() for val in owasp_raw.split(",") if val.strip()]
        else:
            owasp_values = owasp_raw or ["LLM01"]
        atlas_values = taxonomy.get("mitre_atlas", ["TA0042"])

        phishing_mask = df["is_phishing"] == 1
        df.loc[phishing_mask, "owasp_category"] = np.random.choice(
            owasp_values, phishing_mask.sum()
        )
        df.loc[phishing_mask, "atlas_tactic"] = np.random.choice(
            atlas_values, phishing_mask.sum()
        )
        df.loc[~phishing_mask, "owasp_category"] = "BENIGN"
        df.loc[~phishing_mask, "atlas_tactic"] = "N/A"
        return df

    def _split_by_time(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        df = df.sort_values("timestamp").reset_index(drop=True)
        masks = {}
        for split, bounds in self.splits_cfg.items():
            start = self._normalize_boundary(bounds["start"])
            end = self._normalize_boundary(bounds["end"])
            masks[split] = (df["timestamp"] >= start) & (df["timestamp"] <= end)
        default_mask = pd.Series(False, index=df.index)
        train_mask = masks.get("train", default_mask)
        val_mask = masks.get("validation", default_mask)
        test_mask = masks.get("test", default_mask)

        train_df = df[train_mask]
        val_df = df[val_mask]
        test_df = df[test_mask]

        leftover = df[~(train_mask | val_mask | test_mask)]
        if not leftover.empty:
            self._dropped_outside_splits += len(leftover)
            logger.warning(
                "Dropping %d rows outside configured split ranges (check splits cover window)",
                len(leftover),
            )

        return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)

    def _split_random(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        ratios = {
            "train": float(self.split_ratios.get("train", 0.7)),
            "validation": float(self.split_ratios.get("validation", 0.15)),
            "test": float(self.split_ratios.get("test", 0.15)),
        }
        total = ratios["train"] + ratios["validation"] + ratios["test"]
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"split_ratios must sum to 1.0 (got {total})")

        y = df["is_phishing"].astype(int).to_numpy()
        idx0 = np.where(y == 0)[0]
        idx1 = np.where(y == 1)[0]
        rng = np.random.default_rng(self.random_state)
        rng.shuffle(idx0)
        rng.shuffle(idx1)

        def split_indices(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            n = len(indices)
            n_train = int(n * ratios["train"])
            n_val = int(n * ratios["validation"])
            train_i = indices[:n_train]
            val_i = indices[n_train : n_train + n_val]
            test_i = indices[n_train + n_val :]
            return train_i, val_i, test_i

        t0, v0, s0 = split_indices(idx0)
        t1, v1, s1 = split_indices(idx1)

        train_idx = np.concatenate([t0, t1])
        val_idx = np.concatenate([v0, v1])
        test_idx = np.concatenate([s0, s1])
        rng.shuffle(train_idx)
        rng.shuffle(val_idx)
        rng.shuffle(test_idx)
        return (
            df.iloc[train_idx].reset_index(drop=True),
            df.iloc[val_idx].reset_index(drop=True),
            df.iloc[test_idx].reset_index(drop=True),
        )

    def prepare_final_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        source_frames = self.download_datasets()
        all_features = []
        for name, df in source_frames.items():
            meta = self.sources.get(name, {})
            if meta.get("augment_ai_endpoints", True):
                df = self.augment_for_ai_endpoints(df)
            features_df = self.create_feature_matrix(df, name)
            all_features.append(features_df)

        combined = pd.concat(all_features, ignore_index=True)
        combined = self._ensure_timestamp(combined)
        combined = self._filter_to_window(combined)

        phishing_count = int((combined["is_phishing"] == 1).sum())
        n_benign = int(phishing_count * self.benign_multiplier)
        synthetic_benign = self.generate_legitimate_samples(n_benign)
        full_dataset = pd.concat([combined, synthetic_benign], ignore_index=True)
        full_dataset = self._ensure_timestamp(full_dataset)
        full_dataset = self._assign_taxonomy(full_dataset)

        if self.split_mode == "random":
            train_df, val_df, test_df = self._split_random(full_dataset)
        else:
            train_df, val_df, test_df = self._split_by_time(full_dataset)
        logger.info(
            "Dataset prepared - Train: %d | Val: %d | Test: %d",
            len(train_df),
            len(val_df),
            len(test_df),
        )
        return train_df, val_df, test_df

    def save_datasets(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
        train_path = self.processed_dir / "train_data.csv"
        val_path = self.processed_dir / "val_data.csv"
        test_path = self.processed_dir / "test_data.csv"
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)
        logger.info("Datasets saved to %s", self.processed_dir)

        stats = {
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "test_samples": len(test_df),
            "train_phishing_ratio": float(train_df["is_phishing"].mean()),
            "val_phishing_ratio": float(val_df["is_phishing"].mean()),
            "test_phishing_ratio": float(test_df["is_phishing"].mean()),
            "features": [col for col in train_df.columns if col not in {"timestamp", "source", "owasp_category", "atlas_tactic"}],
            "snapshot_window": {
                "start": str(self.window_cfg.get("start")),
                "end": str(self.window_cfg.get("end")),
            },
            "split_mode": self.split_mode,
            "split_ratios": self.split_ratios if self.split_mode == "random" else None,
            "max_rows": self.max_rows,
            "benign_multiplier": self.benign_multiplier,
            "label_provenance": self.config.get("label_provenance"),
            "dropped_outside_window": self._dropped_outside_window,
            "dropped_outside_splits": self._dropped_outside_splits,
            "preprocessing_date": datetime.utcnow().isoformat(),
            "config_path": str(self.config_path) if self.config_path else None,
        }
        self.stats_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.stats_file, "w") as fp:
            json.dump(stats, fp, indent=2, default=str)
        logger.info("Dataset stats written to %s", self.stats_file)


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare AI phishing datasets")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to YAML config (e.g., data/configs/week1_snapshot.yml)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./data"),
        help="Base data directory for processed outputs",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap on rows loaded from large feeds (e.g., URLhaus full).",
    )
    parser.add_argument(
        "--benign-multiplier",
        type=float,
        default=None,
        help="Number of generated benign rows per phishing row (default 1.0).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config) if args.config else {}
    preprocessor = PhishingDataPreprocessor(
        data_dir=args.data_dir,
        config=config,
        config_path=args.config,
        max_rows=args.max_rows,
        benign_multiplier=args.benign_multiplier,
    )

    logger.info("Preparing datasets...")
    train_df, val_df, test_df = preprocessor.prepare_final_dataset()

    logger.info("Training samples: %d | Phishing: %d | Legitimate: %d",
                len(train_df), int(train_df["is_phishing"].sum()), int((train_df["is_phishing"] == 0).sum()))
    logger.info("Validation samples: %d", len(val_df))
    logger.info("Test samples: %d", len(test_df))
    logger.debug("Sample features:\n%s", train_df.head())

    preprocessor.save_datasets(train_df, val_df, test_df)
    logger.info("Preprocessing complete!")
