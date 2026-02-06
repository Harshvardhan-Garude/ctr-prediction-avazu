"""
Preprocessing utilities for high-cardinality CTR logs.

The scripts are intentionally conservative:
- Parse `hour` into time features
- One-hot encode low-cardinality categories (`site_category`, `app_category`)
- Optionally frequency-encode some high-cardinality columns
- Drop extremely high-cardinality IDs for simpler baselines

Adjust flags to match your exact experimental setup.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd


@dataclass
class PreprocessConfig:
    onehot_categories: bool = True
    freq_encode: bool = True
    drop_high_card_ids: bool = True
    sample_n: Optional[int] = None
    random_state: int = 42


HIGH_CARD_ID_COLS = [
    "device_id",
    "device_ip",   # often extremely large
    "device_model",
    "site_id",
    "app_id",
    "hour",
    "site_domain",
    "app_domain",
    "C14",
]


def _parse_hour(df: pd.DataFrame) -> pd.DataFrame:
    """Parse YYMMDDHH -> year/month/day/hour_of_day."""
    if "hour" not in df.columns:
        return df

    hour_str = df["hour"].astype(str).str.zfill(8)
    df = df.copy()
    df["year"] = ("20" + hour_str.str[:2]).astype(int)
    df["month"] = hour_str.str[2:4].astype(int)
    df["day"] = hour_str.str[4:6].astype(int)
    df["hour_of_day"] = hour_str.str[6:8].astype(int)
    return df


def _freq_encode(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Replace category columns with their frequency (count) in the dataset."""
    df = df.copy()
    for c in cols:
        if c in df.columns:
            vc = df[c].value_counts(dropna=False)
            df[c + "_freq"] = df[c].map(vc).astype(np.int64)
    return df


def load_and_featurize(
    csv_path: str | Path,
    cfg: PreprocessConfig = PreprocessConfig(),
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load raw CSV and return (X, y).
    Expected label column: `click` (0/1).
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Optional sampling for quick iteration
    if cfg.sample_n is not None and cfg.sample_n < len(df):
        df = df.sample(n=cfg.sample_n, random_state=cfg.random_state).reset_index(drop=True)

    # Parse time features
    df = _parse_hour(df)

    # One-hot encode selected low-cardinality categories
    if cfg.onehot_categories:
        for col in ["site_category", "app_category"]:
            if col in df.columns:
                df = pd.get_dummies(df, columns=[col], drop_first=True)

    # Frequency encoding for large categorical columns
    if cfg.freq_encode:
        df = _freq_encode(df, cols=["site_domain", "app_domain", "device_ip"])

    # Drop ID-like columns for simpler models
    if cfg.drop_high_card_ids:
        drop_cols = [c for c in HIGH_CARD_ID_COLS if c in df.columns]
        df = df.drop(columns=drop_cols)

    # Drop `id` if present (pure identifier)
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    if "click" not in df.columns:
        raise ValueError("Expected label column `click` in the dataset.")

    y = df["click"].astype(int)
    X = df.drop(columns=["click"])

    return X, y


def time_split_by_day(
    X: pd.DataFrame,
    y: pd.Series,
    day_col: str = "day",
    train_frac: float = 0.8,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Deterministic time-based split:
    - Sort unique days
    - First train_frac days -> train
    - Remaining days -> test
    """
    if day_col not in X.columns:
        raise ValueError(f"Expected `{day_col}` column in features for time-based split.")

    unique_days = np.sort(X[day_col].unique())
    split_idx = int(train_frac * len(unique_days))
    train_days = set(unique_days[:split_idx])
    test_days = set(unique_days[split_idx:])

    train_mask = X[day_col].isin(train_days)
    test_mask = X[day_col].isin(test_days)

    X_train, y_train = X.loc[train_mask].copy(), y.loc[train_mask].copy()
    X_test, y_test = X.loc[test_mask].copy(), y.loc[test_mask].copy()

    return X_train, X_test, y_train, y_test
