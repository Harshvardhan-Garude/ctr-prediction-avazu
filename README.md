# ctr-prediction-avazu
=======
# CTR Prediction on High-Cardinality Ad Logs (Tree Models + Deep CTR)

This repository contains an end-to-end **click-through rate (CTR) prediction** project on a high-cardinality ad click log
dataset (features like `site_id`, `app_id`, `device_ip`, `site_category`, etc.). The focus is on practical preprocessing,
feature engineering, and comparing multiple model families:

- **Random Forest** baseline
- **Gradient Boosted Decision Trees (XGBoost)**
- **Deep CTR models** (DeepFM / DCN) using `deepctr-torch`

> Note: the original dataset CSV is **not** included in this repo. You can point the scripts to your local `sampled_data2.csv`.

## Results (from project report)
Using a time-based split by day:
- **XGBoost** achieved **ROC-AUC ~0.710** (Accuracy ~0.839)
- **Random Forest** achieved **ROC-AUC ~0.674** (Accuracy ~0.835)

(See the report PDFs in `report/` for details.)

## Quickstart

### 1) Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Train/evaluate tree models
```bash
python src/train_tree_models.py \
  --data_csv /path/to/sampled_data2.csv \
  --out_dir outputs/tree
```

### 3) Train/evaluate deep CTR models (optional)
```bash
python src/train_deep_models.py \
  --data_csv /path/to/sampled_data2.csv \
  --out_dir outputs/deep
```

## Project Structure
```
.
├── src/                    # runnable scripts (CLI)
├── notebooks/              # cleaned notebooks (no cell outputs)
├── report/                 # original project reports (PDF)
├── configs/                # example config files
└── assets/                 # plots/screenshots you may add later
```

## Data & Features

### Time features
We parse the `hour` column (YYMMDDHH) into:
- `year`, `month`, `day`, `hour_of_day`

### Categorical handling
High-cardinality columns can be very large to one-hot encode, so the scripts support:
- **one-hot** for `site_category`, `app_category`
- **frequency encoding** for `site_domain`, `app_domain`, `device_ip`
- dropping extremely high-cardinality IDs (`device_id`, etc.) for tree baselines

You can adjust this behavior via CLI flags in `src/preprocess.py`.

## Reproducibility
- Default split is **time-based by `day`** (first 80% days train, remaining test), consistent with the report.
- All scripts set a random seed for repeatable results.
>>>>>>> Initial release: CTR prediction baselines (RF/XGBoost/DeepCTR)
