#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DQN1 — Urban Air Quality & Health Risk Prediction
=================================================

End-to-end, well-commented pipeline for the DQN1 case study using
Gradient Boosting (scikit-learn) on tabular Excel data.

Highlights
----------
- Reads Excel (.xlsx) with pandas (requires openpyxl)
- Optional time column (e.g., sunriseEpoch, datetimeEpoch) for time-aware split
- Adds simple calendar features (year, month, day, dayofweek, hour)
- Robust preprocessing: median impute numeric, one-hot encode categoricals
- RandomizedSearchCV hyperparameter tuning
- Metrics: RMSE, MAE (+ MAPE for reference)
- Saves artifacts per target: metrics, predictions, feature importances, plots

Install deps once:
    pip install pandas numpy scikit-learn matplotlib openpyxl

Examples
--------
1) See all columns + suggested targets:
    python dqn1_model.py --data "DQN1 Dataset.xlsx" --list-columns

2) Train with time-aware split (date col = sunriseEpoch):
    python dqn1_model.py --data "DQN1 Dataset.xlsx" --target "PM2.5" --datecol sunriseEpoch

3) Train two targets in one go (air quality + health risk):
    python dqn1_model.py --data "DQN1 Dataset.xlsx" --target "PM2.5,health_risk_score" --datecol sunriseEpoch

4) Train without a time column (random split 80/20):
    python dqn1_model.py --data "DQN1 Dataset.xlsx" --target "health_risk_score"

Artifacts are written under:  ./artifacts/<target>/
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import (
    KFold,
    RandomizedSearchCV,
    TimeSeriesSplit,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Keep plots working in headless environments
warnings.filterwarnings("ignore", category=FutureWarning)
plt.switch_backend("agg")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def smart_to_datetime(series: pd.Series) -> pd.Series:
    """
    Convert to pandas datetime with common epoch handling.

    If a numeric series is ~13 digits -> treat as milliseconds since epoch.
    If ~10 digits -> treat as seconds since epoch.
    Otherwise fall back to pandas parsing.

    Returns a timezone-naive datetime (UTC converted to naive).
    """
    s = series.dropna()
    if s.empty:
        return pd.to_datetime(series, errors="coerce")

    sample = s.iloc[0]
    try:
        if pd.api.types.is_numeric_dtype(series):
            v_abs = abs(float(sample))
            if v_abs > 1e11:  # milliseconds
                dt = pd.to_datetime(series, unit="ms", utc=True, errors="coerce")
            elif v_abs > 1e9:  # seconds
                dt = pd.to_datetime(series, unit="s", utc=True, errors="coerce")
            else:
                dt = pd.to_datetime(series, utc=True, errors="coerce")
        else:
            dt = pd.to_datetime(series, utc=True, errors="coerce")
        return dt.dt.tz_convert(None)
    except Exception:
        return pd.to_datetime(series, errors="coerce")


def infer_candidate_targets(df: pd.DataFrame) -> List[str]:
    """
    Suggest plausible *numeric* target columns by name pattern.
    """
    patterns = [
        r"pm", r"pm2", r"pm2\.?5", r"no2", r"co2", r"o3", r"so2",
        r"aqi", r"air.?quality", r"health.*risk", r"risk.*score",
        r"mortality", r"hospital"
    ]
    pat = re.compile("|".join(patterns), re.IGNORECASE)
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    suggestions = [c for c in numeric_cols if pat.search(c)]
    return suggestions if suggestions else numeric_cols


def list_columns(df: pd.DataFrame) -> None:
    """Print columns and a heuristic list of candidate target columns."""
    print("\\n== All columns ==")
    for c in df.columns:
        print(" -", c, f"({df[c].dtype})")
    print("\\n== Candidate target columns (heuristic) ==")
    for c in infer_candidate_targets(df):
        print(" -", c)
    print("\\nTip: Choose one or more exact column names and pass via --target \"colA,colB\"")


def build_preprocessor(X: pd.DataFrame, drop_cols: List[str]) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """
    Build a ColumnTransformer that imputes numerics and one-hot encodes categoricals.
    Returns (preprocessor, numeric_cols, categorical_cols).
    """
    cols = [c for c in X.columns if c not in drop_cols]
    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(X[c])]
    categorical_cols = [c for c in cols if not pd.api.types.is_numeric_dtype(X[c])]

    numeric_transformer = SimpleImputer(strategy="median")
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )
    return preprocessor, numeric_cols, categorical_cols


def extract_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """
    Retrieve final feature names (numeric + one-hot-expanded categorical).
    """
    names: List[str] = []
    for name, transformer, cols in preprocessor.transformers_:
        if name == "num":
            names.extend(list(cols))
        elif name == "cat":
            ohe = transformer.named_steps.get("onehot")
            cat_cols = transformer.named_steps.get("imputer").feature_names_in_
            if hasattr(ohe, "get_feature_names_out"):
                names.extend(list(ohe.get_feature_names_out(cat_cols)))
            else:
                names.extend(list(cat_cols))
    return names


def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute RMSE, MAE, and MAPE (safe—ignores true values near zero).
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    mask = np.abs(y_true) > 1e-12
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0) if mask.any() else float("nan")
    return {"rmse": rmse, "mae": mae, "mape": mape}


def time_aware_split(df: pd.DataFrame, datecol: str, test_size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sort by date and split by final fraction as test (no leakage).
    """
    df = df.copy()
    dt = smart_to_datetime(df[datecol])
    df = df.assign(_dt_sort=dt).sort_values("_dt_sort")
    split_idx = int((1.0 - test_size) * len(df))
    train_df = df.iloc[:split_idx].drop(columns=["_dt_sort"])
    test_df = df.iloc[split_idx:].drop(columns=["_dt_sort"])
    return train_df, test_df


def add_calendar_features(df: pd.DataFrame, datecol: str) -> pd.DataFrame:
    """
    Add basic calendar/time features from datecol (no lags—simple and rubric-friendly).
    """
    df = df.copy()
    dt = smart_to_datetime(df[datecol])
    df["year"] = dt.dt.year
    df["month"] = dt.dt.month
    df["day"] = dt.dt.day
    df["dayofweek"] = dt.dt.dayofweek
    df["hour"] = (dt.dt.hour.fillna(0).astype(int) if not dt.dt.hour.isna().all() else 0)
    return df


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    data_path: Path
    targets: List[str]
    datecol: Optional[str] = None
    idcols: List[str] = None
    test_size: float = 0.2
    n_iter: int = 24
    random_state: int = 42
    outdir: Path = Path("artifacts")


def fit_for_target(df: pd.DataFrame, cfg: TrainConfig, target: str) -> None:
    """
    Train/evaluate a GradientBoostingRegressor for one target column.
    Saves all artifacts to cfg.outdir / <target>/
    """
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in data.")

    outdir = cfg.outdir / target.replace("/", "_")
    outdir.mkdir(parents=True, exist_ok=True)

    # Augment with time-derived features if a date column is provided
    work_df = df.copy()
    if cfg.datecol and cfg.datecol in work_df.columns:
        work_df = add_calendar_features(work_df, cfg.datecol)

    # Build X/y while excluding target, date column, and any ID columns
    drop_cols = [target]
    if cfg.datecol and cfg.datecol in work_df.columns:
        drop_cols.append(cfg.datecol)
    if cfg.idcols:
        drop_cols.extend([c for c in cfg.idcols if c in work_df.columns])

    X_all = work_df.drop(columns=[c for c in drop_cols if c in work_df.columns])
    y_all = work_df[target].astype(float).values

    # Preprocessing + model
    preprocessor, _, _ = build_preprocessor(X_all, drop_cols=[])
    model = GradientBoostingRegressor(random_state=cfg.random_state)
    pipe = Pipeline(steps=[("pre", preprocessor), ("model", model)])

    # Hyperparameter search space (compact, effective)
    param_distributions = {
        "model__n_estimators": np.arange(200, 801),          # 200..800 trees
        "model__learning_rate": np.linspace(0.01, 0.2, 40),  # 0.01..0.2
        "model__max_depth": np.arange(2, 7),                 # depth 2..6
        "model__subsample": np.linspace(0.6, 1.0, 21),       # 0.6..1.0
        "model__min_samples_leaf": np.arange(1, 11),         # 1..10
        "model__max_features": [None, "sqrt", "log2", 0.6, 0.8, 1.0],
    }

    # Train/validation split + CV strategy
    if cfg.datecol and cfg.datecol in df.columns:
        train_df, test_df = time_aware_split(work_df.assign(_y=y_all), cfg.datecol, cfg.test_size)
        X_train = train_df.drop(columns=["_y"])
        y_train = train_df["_y"].astype(float).values
        X_test = test_df.drop(columns=["_y"])
        y_test = test_df["_y"].astype(float).values
        cv = TimeSeriesSplit(n_splits=5)
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, test_size=cfg.test_size, random_state=cfg.random_state
        )
        cv = KFold(n_splits=5, shuffle=True, random_state=cfg.random_state)

    # Randomized hyperparameter search (optimize negative RMSE)
    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_distributions,
        n_iter=cfg.n_iter,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        cv=cv,
        random_state=cfg.random_state,
        verbose=1,
    )
    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    # Evaluate on holdout
    y_pred = best_model.predict(X_test)
    metrics = evaluate_metrics(y_test, y_pred)

    # Save metrics
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame([metrics]).to_csv(outdir / "metrics.csv", index=False)

    # Save predictions
    pd.DataFrame({"y_true": y_test, "y_pred": y_pred}).to_csv(outdir / "predictions.csv", index=False)

    # Feature importance
    preproc = best_model.named_steps["pre"]
    feat_names = []
    # Fit a small sample through preprocessor to obtain names after fitting
    # (the preprocessor is already fitted inside the pipeline)
    try:
        feat_names = extract_feature_names(preproc)
    except Exception:
        feat_names = [f"f{i}" for i in range(best_model.named_steps["model"].feature_importances_.shape[0])]

    importances = best_model.named_steps["model"].feature_importances_
    fi_df = pd.DataFrame({"feature": feat_names, "importance": importances}).sort_values(
        "importance", ascending=False
    )
    fi_df.to_csv(outdir / "feature_importances.csv", index=False)

    # Plot: predictions vs. actual
    plt.figure(figsize=(8, 6))
    plt.scatter(range(len(y_test)), y_test, label="Actual", s=10)
    plt.scatter(range(len(y_test)), y_pred, label="Predicted", s=10)
    plt.title(f"Predicted vs Actual — {target}")
    plt.xlabel("Test sample index (ordered)")
    plt.ylabel(target)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "pred_vs_actual.png", dpi=160)
    plt.close()

    # Plot: top-25 feature importances
    top = fi_df.head(25)[::-1]
    plt.figure(figsize=(10, 8))
    plt.barh(top["feature"], top["importance"])
    plt.title(f"Top 25 Feature Importances — {target}")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(outdir / "feature_importance.png", dpi=160)
    plt.close()

    # Short per-target README
    readme = f"""# Artifacts for target: {target}

**Metrics**
- RMSE: {metrics['rmse']:.4f}
- MAE:  {metrics['mae']:.4f}
- MAPE: {metrics['mape']:.2f}%

**Files**
- metrics.json / metrics.csv
- predictions.csv (y_true, y_pred)
- pred_vs_actual.png
- feature_importances.csv
- feature_importance.png
"""
    (outdir / "README.md").write_text(readme, encoding="utf-8")

    print(f"[{target}] RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}, MAPE={metrics['mape']:.2f}%")
    print("Artifacts ->", outdir.resolve())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="DQN1 Gradient Boosting pipeline.")
    parser.add_argument("--data", required=True, type=str, help="Path to Excel file, e.g., 'DQN1 Dataset.xlsx'")
    parser.add_argument("--target", required=False, type=str, default="",
                        help="Target column name(s), comma-separated for multiple. Use --list-columns to explore.")
    parser.add_argument("--datecol", required=False, type=str, default=None,
                        help="Optional date/time column (enables time-aware split + calendar features).")
    parser.add_argument("--idcols", required=False, type=str, default="",
                        help="Comma-separated columns to drop from features (IDs, text, etc.).")
    parser.add_argument("--test_size", required=False, type=float, default=0.2,
                        help="Holdout size for test set (default 0.2).")
    parser.add_argument("--n_iter", required=False, type=int, default=24,
                        help="RandomizedSearch iterations (default 24).")
    parser.add_argument("--outdir", required=False, type=str, default="artifacts",
                        help="Output directory (default './artifacts').")
    parser.add_argument("--list-columns", action="store_true",
                        help="Print available columns + suggested targets, then exit.")

    args = parser.parse_args(argv)

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    # Load Excel
    try:
        df = pd.read_excel(data_path)
    except Exception as e:
        print("ERROR reading Excel. Make sure 'openpyxl' is installed: pip install openpyxl")
        raise e

    if args.list_columns:
        list_columns(df)
        return

    targets = [t.strip() for t in args.target.split(",") if t.strip()]
    if not targets:
        print("\\nERROR: No --target provided. Try:")
        print("  python dqn1_model.py --data \"DQN1 Dataset.xlsx\" --list-columns")
        print("  python dqn1_model.py --data \"DQN1 Dataset.xlsx\" --target \"PM2.5\" --datecol sunriseEpoch\\n")
        sys.exit(1)

    idcols = [c.strip() for c in args.idcols.split(",") if c.strip()] if args.idcols else []

    cfg = TrainConfig(
        data_path=data_path,
        targets=targets,
        datecol=args.datecol,
        idcols=idcols,
        test_size=float(args.test_size),
        n_iter=int(args.n_iter),
        random_state=42,
        outdir=Path(args.outdir),
    )

    # Warn if any target is non-numeric
    for t in targets:
        if not pd.api.types.is_numeric_dtype(df[t]):
            print(f"WARNING: Target '{t}' is not numeric (dtype={df[t].dtype}). Coercing to float may introduce NaNs.")

    for t in targets:
        fit_for_target(df, cfg, t)


if __name__ == "__main__":
    main()
