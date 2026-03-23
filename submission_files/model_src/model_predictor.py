"""
Mode 2: Load saved models, apply the exact same feature transformation pipeline
used during training, generate predictions, and write one CSV per date.

Output CSV columns: Date, Time, Id, Pred
  - Date: YYYYMMDD integer
  - Time: "15:30" (end-of-day prediction timestamp)
  - Id:   stock identifier
  - Pred: predicted residual return (Ens-5 diversity-weighted ensemble)
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.stats import norm as sp_norm
from tqdm import tqdm

from model_src.resmlp import ResidualMLP


# ── Transformation helpers ─────────────────────────────────────────────────

def _apply_clipping(df: pd.DataFrame, clip_params: dict, feature_cols: list) -> pd.DataFrame:
    """Clip each feature using the saved train-derived MAD bounds."""
    df = df.copy()
    for col in feature_cols:
        if col not in clip_params["features"]:
            continue
        p = clip_params["features"][col]
        df[col] = df[col].clip(lower=p["clip_lo"], upper=p["clip_hi"])
    return df


def _apply_stage1_zscore(df: pd.DataFrame, norm_params: dict, feature_cols: list) -> pd.DataFrame:
    """Subtract train mean and divide by train std (stored in norm_params)."""
    df = df.copy()
    stage1 = norm_params["stage1_params"]
    for col in feature_cols:
        mu  = stage1[col]["mean"]
        std = stage1[col]["std"]
        df[col] = (df[col] - mu) / std if std > 1e-12 else 0.0
    return df


def _apply_xsect_rank_norm(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Per-date rank → inverse-normal CDF (no stored params needed)."""
    df = df.copy()
    for col in feature_cols:
        ranked = df.groupby(df.index)[col].rank(method="average", na_option="keep")
        counts = df.groupby(df.index)[col].transform("count")
        df[col] = sp_norm.ppf((ranked - 0.5) / counts)
    return df


def _transform_features(df_raw: pd.DataFrame,
                         clip_params: dict,
                         norm_params: dict,
                         feature_cols: list) -> pd.DataFrame:
    """
    Full three-stage pipeline — must match training exactly:
      1. 5-MAD clipping     (train-derived bounds)
      2. Stage-1 z-score    (train mean / std)
      3. Cross-sectional rank-norm (per-date, applied to the incoming data itself)
    """
    df = _apply_clipping(df_raw, clip_params, feature_cols)
    df = _apply_stage1_zscore(df, norm_params, feature_cols)
    df = _apply_xsect_rank_norm(df, feature_cols)
    return df


# ── Prediction entry point ─────────────────────────────────────────────────

def run_mode2_predictions(
    features_dir: Path,
    output_dir: Path,
    model_dir: Path,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> None:
    """
    Load feature CSVs written by Mode 1, transform them with saved train params,
    run the Ens-5 ensemble, and write predictions_YYYYMMDD.csv per date.

    Parameters
    ----------
    features_dir : Path   Directory with feature_YYYYMMDD.csv files (Mode 1 output).
    output_dir   : Path   Where predictions_YYYYMMDD.csv files are written.
    model_dir    : Path   Directory containing saved model .joblib / .pt files.
    start_date   : pd.Timestamp
    end_date     : pd.Timestamp
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir  = Path(model_dir)

    # ── Load saved artefacts ──────────────────────────────────────────────
    print("Loading saved models and transformation params...")
    clip_params   = joblib.load(model_dir / "clip_params.joblib")
    norm_params   = joblib.load(model_dir / "norm_params.joblib")
    feature_info  = joblib.load(model_dir / "feature_info.joblib")
    ensemble_info = joblib.load(model_dir / "ensemble_weights.joblib")
    feature_cols  = feature_info["feature_cols"]

    ridge  = joblib.load(model_dir / "ridge.joblib")
    rf     = joblib.load(model_dir / "rf.joblib")
    xgb_m  = joblib.load(model_dir / "xgb.joblib")
    mlp    = joblib.load(model_dir / "mlp.joblib")

    resmlp_params = joblib.load(model_dir / "resmlp_params.joblib")
    device = "mps" if torch.backends.mps.is_available() else (
             "cuda" if torch.cuda.is_available() else "cpu")
    resmlp = ResidualMLP(
        in_dim  = len(feature_cols),
        hidden  = resmlp_params["hidden"],
        n_blocks= resmlp_params["n_blocks"],
        dropout = resmlp_params["dropout"],
    ).to(device)
    resmlp.load_state_dict(
        torch.load(model_dir / "resmlp_state.pt", map_location=device)
    )
    resmlp.eval()

    w5     = ensemble_info["w5"]       # diversity weights for Ens-5
    names5 = ensemble_info["names_5"]  # ["Ridge","RF","XGB","MLP","ResidMLP"]

    print(f"  Device: {device}")
    print(f"  Ens-5 weights: {dict(zip(names5, w5.round(3)))}")

    # ── Collect feature files in date range ───────────────────────────────
    feature_files = sorted(features_dir.glob("features_*.csv"))
    feature_files = [
        f for f in feature_files
        if start_date <= pd.to_datetime(f.stem.split("_")[1], format="%Y%m%d") <= end_date
    ]
    if not feature_files:
        raise FileNotFoundError(
            f"No feature_YYYYMMDD.csv files found in {features_dir} "
            f"between {start_date.date()} and {end_date.date()}"
        )
    print(f"  {len(feature_files)} date files to predict.")

    # ── Predict date by date ──────────────────────────────────────────────
    for fpath in tqdm(feature_files, desc="Mode 2 predictions"):
        date_str = fpath.stem.split("_")[1]          # "YYYYMMDD"
        date_ts  = pd.to_datetime(date_str, format="%Y%m%d")

        df_raw = pd.read_csv(fpath)

        # Set DatetimeIndex for cross-sectional rank norm groupby
        if "Date" in df_raw.columns:
            df_raw = df_raw.set_index(
                pd.to_datetime(df_raw["Date"])
            )
        else:
            df_raw.index = pd.DatetimeIndex([date_ts] * len(df_raw))

        # Guard: check all feature columns are present
        missing = [c for c in feature_cols if c not in df_raw.columns]
        if missing:
            print(f"  Warning: {date_str} missing columns {missing} — skipping.")
            continue

        # ── Apply transformation pipeline ─────────────────────────────────
        df_norm = _transform_features(df_raw, clip_params, norm_params, feature_cols)
        X = np.nan_to_num(
            df_norm[feature_cols].values.astype(np.float32), nan=0.0
        )

        # ── Per-model predictions ─────────────────────────────────────────
        p_ridge  = ridge.predict(X)
        p_rf     = rf.predict(X)
        p_xgb    = xgb_m.predict(X)
        p_mlp    = mlp.predict(X)
        with torch.no_grad():
            p_resmlp = resmlp(torch.tensor(X, device=device)).cpu().numpy()

        pred_ens = np.column_stack([p_ridge, p_rf, p_xgb, p_mlp, p_resmlp]) @ w5

        # ── Build output DataFrame ─────────────────────────────────────────
        id_col = df_raw["Id"].values if "Id" in df_raw.columns else np.arange(len(X))
        out = pd.DataFrame({
            "Date": int(date_str),
            "Time": "15:30",
            "Id":   id_col,
            "Pred": pred_ens,
        })
        out.to_csv(output_dir / f"predictions_{date_str}.csv", index=False)

    print(f"\n✓ Mode 2 complete — {len(feature_files)} prediction files written to {output_dir}")
