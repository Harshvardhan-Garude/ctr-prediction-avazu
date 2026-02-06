"""
Train simple Deep CTR models using deepctr-torch.

Models included:
- DeepFM
- DCN

These models can handle sparse categorical features more naturally than one-hot tree baselines,
but they may require more careful tuning and GPU acceleration for large datasets.

This script is provided as a *clean, runnable baseline*.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models import DeepFM, DCN

# Note: We intentionally keep preprocessing light here and rely on deepctr's sparse feature handling.


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_csv", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="outputs/deep")
    p.add_argument("--sample_n", type=int, default=200000, help="Subsample for deep models (default 200k).")
    p.add_argument("--model", type=str, choices=["deepfm", "dcn"], default="deepfm")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=4096)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data_csv)

    if args.sample_n is not None and args.sample_n < len(df):
        df = df.sample(n=args.sample_n, random_state=args.seed).reset_index(drop=True)

    # Parse hour -> time features
    hour_str = df["hour"].astype(str).str.zfill(8)
    df["day"] = hour_str.str[4:6].astype(int)
    df["hour_of_day"] = hour_str.str[6:8].astype(int)

    # Choose a subset of columns to keep this example simple & fast.
    # You can expand this list as needed.
    sparse_features = [
        "C1", "banner_pos", "site_category", "app_category",
        "device_type", "device_conn_type",
        "C15", "C16", "C17", "C18", "C19", "C20", "C21",
    ]
    dense_features = ["day", "hour_of_day"]

    # Fill NAs for safe categorical encoding
    for c in sparse_features:
        df[c] = df[c].astype(str).fillna("NA")

    for c in dense_features:
        df[c] = df[c].fillna(0)

    target = "click"

    # Label encode sparse features to integer IDs
    for feat in sparse_features:
        df[feat] = df[feat].astype("category").cat.codes.astype(int)

    # Build deepctr feature columns
    fixlen_feature_columns = [
        SparseFeat(feat, vocabulary_size=int(df[feat].max()) + 1, embedding_dim=8)
        for feat in sparse_features
    ] + [
        DenseFeat(feat, 1) for feat in dense_features
    ]
    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    train, test = train_test_split(df, test_size=0.2, random_state=args.seed, shuffle=True)
    train_model_input = {name: train[name].values for name in feature_names}
    test_model_input = {name: test[name].values for name in feature_names}

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    if args.model == "deepfm":
        model = DeepFM(linear_feature_columns, dnn_feature_columns, task="binary", device=device)
    else:
        model = DCN(linear_feature_columns, dnn_feature_columns, task="binary", device=device)

    model.compile("adagrad", "binary_crossentropy", metrics=["binary_crossentropy"])
    model.fit(
        train_model_input,
        train[target].values,
        batch_size=args.batch_size,
        epochs=args.epochs,
        verbose=2,
        validation_split=0.1,
    )

    pred_ans = model.predict(test_model_input, batch_size=args.batch_size)
    auc = roc_auc_score(test[target].values, pred_ans)
    (out_dir / "metrics.txt").write_text(f"{args.model.upper()} ROC-AUC: {auc:.4f}\n")
    model.save(str(out_dir / f"{args.model}.pth"))

    print((out_dir / "metrics.txt").read_text())


if __name__ == "__main__":
    main()
