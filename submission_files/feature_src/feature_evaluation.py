import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _get_feature_columns(daily_df: pd.DataFrame, exclude: Optional[List[str]] = None) -> List[str]:
    exclude = exclude or []
    return [c for c in daily_df.columns if c not in exclude]


def compute_daily_ic(
    daily_df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    target_col: str = "target_variable",
    method: str = "pearson",
) -> pd.DataFrame:
    """
    Compute daily cross-sectional IC between each feature and the target.
    """
    if feature_cols is None:
        feature_cols = _get_feature_columns(daily_df, exclude=["Id", target_col])

    feature_cols = [f for f in feature_cols if f in daily_df.columns]
    if not feature_cols:
        return pd.DataFrame()

    def _ic_for_day(g: pd.DataFrame) -> pd.Series:
        y = g[target_col]
        x_df = g[feature_cols]
        if method == "spearman":
            x_df = x_df.rank()
            y = y.rank()
        with warnings.catch_warnings(action="ignore", category=RuntimeWarning):
            ic_vals = x_df.corrwith(y, method="pearson")
        return ic_vals.astype(float)

    with warnings.catch_warnings(action="ignore", category=RuntimeWarning):
        ic_df = daily_df.groupby(daily_df.index).apply(_ic_for_day)
    return ic_df


def summarize_ic(ic_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate daily IC into mean, std, IR, and simple t-stat.
    """
    summary_rows = []
    for col in ic_df.columns:
        series = ic_df[col].dropna()
        if series.empty:
            summary_rows.append(
                {"feature": col, "mean_ic": np.nan, "std_ic": np.nan, "ir": np.nan, "t_stat": np.nan}
            )
            continue
        mean_ic = float(series.mean())
        std_ic = float(series.std(ddof=1))
        ir = mean_ic / std_ic if std_ic > 0 else np.nan
        t_stat = mean_ic / (std_ic / np.sqrt(len(series))) if std_ic > 0 else np.nan
        summary_rows.append(
            {"feature": col, "mean_ic": mean_ic, "std_ic": std_ic, "ir": ir, "t_stat": t_stat}
        )
    return pd.DataFrame(summary_rows).set_index("feature").sort_values("mean_ic", ascending=False)


def _mad_equal_width_bins(series: pd.Series, n_bins: int, mad_scale: float = 5.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split a series into equal-width bins over median +/- mad_scale * MAD.
    Returns zero-based bin indices and bin edges.
    """
    med = float(series.median())
    mad = float((series - med).abs().median())
    if mad == 0 or np.isnan(mad):
        mad = np.finfo(float).eps if med != 0 else 1e-10
    lower = med - mad_scale * mad
    upper = med + mad_scale * mad
    if lower >= upper:
        lower, upper = series.min(), series.max()
        if lower >= upper:
            upper = lower + 1e-10
    bin_edges = np.linspace(lower, upper, n_bins + 1)
    clipped = series.clip(lower=lower, upper=upper)
    bins_arr = np.digitize(clipped.values, bin_edges[1:-1], right=False)
    bins_arr = np.clip(bins_arr, 0, n_bins - 1)
    return bins_arr, bin_edges


def _winsorize_5mad(series: pd.Series) -> pd.Series:
    med = series.median()
    mad = (series - med).abs().median()
    if mad == 0 or np.isnan(mad):
        return series
    lower = med - 5 * mad
    upper = med + 5 * mad
    return series.clip(lower=lower, upper=upper)


def evaluate_feature_bins_detailed(
    daily_df: pd.DataFrame,
    feature_name: str,
    target_col: str = "target_variable",
    n_bins: int = 4,
    min_per_bin: int = 5,
    winsorize_5mad: bool = False,
) -> Tuple[Optional[pd.Series], Optional[pd.Series], Optional[np.ndarray]]:
    """
    Pool all observations and compute mean target, SE, and feature-bin centers.
    """
    df = daily_df[[feature_name, target_col]].copy().dropna()
    if len(df) < min_per_bin * n_bins:
        return None, None, None

    if winsorize_5mad:
        df[feature_name] = _winsorize_5mad(df[feature_name])
        df[target_col] = _winsorize_5mad(df[target_col])

    try:
        bins_arr, bin_edges = _mad_equal_width_bins(df[feature_name], n_bins=n_bins, mad_scale=5.0)
    except (ValueError, TypeError):
        return None, None, None

    df["_bin"] = bins_arr
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    grp = df.groupby("_bin")[target_col]
    means = grp.mean().reindex(range(n_bins))
    stds = grp.std().reindex(range(n_bins))
    counts = grp.count().reindex(range(n_bins))
    ses = stds / np.sqrt(counts)
    return means, ses, bin_centers
