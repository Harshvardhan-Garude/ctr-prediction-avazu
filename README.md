# CTR Prediction on High-Cardinality Ad Logs (Tree Models + Deep CTR)

This repository contains an end-to-end **click-through rate (CTR) prediction** project on a high-cardinality ad click log
dataset (features like `site_id`, `app_id`, `device_ip`, `site_category`, etc.). The focus is on practical preprocessing,
feature engineering, and comparing multiple model families:

- **Random Forest** baseline
- **Gradient Boosted Decision Trees (XGBoost)**
- **Deep CTR models** (DeepFM / DCN) using `deepctr-torch`

Use a local labeled Avazu-style CSV, such as `sampled_data2.csv`, with `hour`, `click`, and the feature columns.

## Results (from project report)
Using a time-based split by day:
- **XGBoost** achieved **ROC-AUC ~0.710** (Accuracy ~0.839)
- **Random Forest** achieved **ROC-AUC ~0.674** (Accuracy ~0.835)

See the [project reports](report/) for the original experiment.

## Quickstart

Run the commands from the repository root. Create a virtual environment with
`python -m venv .venv`. Activate it with `source .venv/bin/activate` on Linux or
macOS, or `.\.venv\Scripts\Activate.ps1` in Windows PowerShell. Then install
the dependencies:

```bash
python -m pip install -r requirements.txt
```

### Train and evaluate tree models

```bash
python src/train_tree_models.py --data_csv path/to/sampled_data2.csv --out_dir outputs/tree
```

### Deep CTR model setup is incomplete

A fresh Windows/Python 3.12 installation of `requirements.txt` completed, but
`python src/train_deep_models.py --help` failed because `deepctr-torch` imports
`requests`, which is missing from the manifest. Resolve and pin the deep-model
dependencies before relying on this path. The script also calls `model.save`,
which is not provided by the installed deepctr-torch 0.3.0 model classes; use a
supported PyTorch checkpoint-saving path and verify it before a full run.
The tree-model CLI is independently usable. The following command is the
intended deep-model invocation:

```bash
python src/train_deep_models.py --data_csv path/to/sampled_data2.csv --out_dir outputs/deep --model dcn
```

Use `--model deepfm` to try DeepFM. Each command writes a `metrics.txt` file
to its output directory.

## Project Structure
```
.
├── src/                    # runnable scripts (CLI)
├── notebooks/              # cleaned notebooks (no cell outputs)
├── report/                 # original project reports (PDF)
└── configs/                # example commands
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

The tree training script exposes these options as CLI flags. Run it with `--help` for the defaults.

## Reproducibility
- Tree models train on the earliest 80% of unique days and test on the remaining days.
- The deep-model script uses a seeded random 80/20 split. Use a common evaluation split for direct comparisons.
- Both training scripts accept `--seed`.
