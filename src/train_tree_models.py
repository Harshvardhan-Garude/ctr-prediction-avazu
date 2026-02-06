"""
Train and evaluate tree-based CTR models:
- RandomForestClassifier (scikit-learn)
- XGBoost (xgboost.XGBClassifier)

Default hyperparameters are set to the best settings found in the project notebook/report.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier

from preprocess import PreprocessConfig, load_and_featurize, time_split_by_day


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_csv", type=str, required=True, help="Path to sampled_data2.csv (not included in repo).")
    p.add_argument("--out_dir", type=str, default="outputs/tree", help="Directory to write models and metrics.")
    p.add_argument("--sample_n", type=int, default=None, help="Optional row subsample for quick runs.")
    p.add_argument("--no_freq_encode", action="store_true", help="Disable frequency encoding.")
    p.add_argument("--no_onehot", action="store_true", help="Disable one-hot encoding for app/site category.")
    p.add_argument("--keep_ids", action="store_true", help="Do not drop high-cardinality ID columns (may explode).")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = PreprocessConfig(
        onehot_categories=not args.no_onehot,
        freq_encode=not args.no_freq_encode,
        drop_high_card_ids=not args.keep_ids,
        sample_n=args.sample_n,
        random_state=args.seed,
    )

    X, y = load_and_featurize(args.data_csv, cfg=cfg)
    X_train, X_test, y_train, y_test = time_split_by_day(X, y)

    # Random Forest (best config from notebook)
    rf = RandomForestClassifier(
        n_estimators=40,
        max_depth=7,
        class_weight={0: 1, 1: 2},
        min_samples_leaf=1,
        random_state=args.seed,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_proba = rf.predict_proba(X_test)[:, 1]
    rf_pred = (rf_proba >= 0.5).astype(int)

    rf_acc = accuracy_score(y_test, rf_pred)
    rf_auc = roc_auc_score(y_test, rf_proba)

    # XGBoost (best config from notebook)
    xgb = XGBClassifier(
        eval_metric="logloss",
        random_state=args.seed,
        n_estimators=40,
        max_depth=7,
        learning_rate=0.3,
        subsample=0.6,
        colsample_bytree=0.8,
        min_child_weight=3,
        n_jobs=-1,
    )
    xgb.fit(X_train, y_train)
    xgb_proba = xgb.predict_proba(X_test)[:, 1]
    xgb_pred = (xgb_proba >= 0.5).astype(int)

    xgb_acc = accuracy_score(y_test, xgb_pred)
    xgb_auc = roc_auc_score(y_test, xgb_proba)

    metrics_path = out_dir / "metrics.txt"
    metrics_path.write_text(
        f"RandomForest:  accuracy={rf_acc:.4f}  roc_auc={rf_auc:.4f}\n"
        f"XGBoost:       accuracy={xgb_acc:.4f}  roc_auc={xgb_auc:.4f}\n"
    )

    joblib.dump(rf, out_dir / "rf.joblib")
    joblib.dump(xgb, out_dir / "xgb.joblib")

    print(metrics_path.read_text())


if __name__ == "__main__":
    main()
