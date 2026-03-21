"""
Feature Lab Evaluator
=====================
一次性对所有特征变体进行批量 IC 与分桶分析。
"""

import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

_SRC = str(Path(__file__).resolve().parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from feature_evaluation import (
    compute_daily_ic,
    summarize_ic,
    evaluate_feature_bins_detailed,
)


def evaluate_variants(
    eval_df: pd.DataFrame,
    variant_cols: List[str],
    target_col: str = "target_variable",
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    Evaluate all variants at once.

    Parameters
    ----------
    eval_df : DataFrame
        Index = Date, columns include Id, variant features, and *target_col*.
    variant_cols : list[str]
        Column names of feature variants.
    target_col : str
        Target column name.
    n_bins : int
        Number of MAD equal-width bins for bin analysis.

    Returns
    -------
    dict with keys:
        summary           – DataFrame (one row per variant, sorted by Pearson IR)
        pearson_ic_daily   – DataFrame (daily IC per variant)
        spearman_ic_daily  – DataFrame (daily IC per variant)
        bin_means          – dict[str, Series] (variant → mean target per bin)
    """
    pearson_ic = compute_daily_ic(
        eval_df, feature_cols=variant_cols,
        target_col=target_col, method="pearson",
    )
    pearson_summary = summarize_ic(pearson_ic)

    spearman_ic = compute_daily_ic(
        eval_df, feature_cols=variant_cols,
        target_col=target_col, method="spearman",
    )
    spearman_summary = summarize_ic(spearman_ic)

    bin_means_dict: Dict[str, pd.Series] = {}
    bin_ses_dict: Dict[str, pd.Series] = {}
    bin_centers_dict: Dict[str, np.ndarray] = {}
    bin_means_winsor_dict: Dict[str, pd.Series] = {}
    bin_ses_winsor_dict: Dict[str, pd.Series] = {}
    bin_centers_winsor_dict: Dict[str, np.ndarray] = {}
    for col in tqdm(variant_cols, desc="Bin analysis"):
        try:
            means, ses, centers = evaluate_feature_bins_detailed(
                eval_df, feature_name=col,
                target_col=target_col, n_bins=n_bins, winsorize_5mad=False,
            )
            bin_means_dict[col] = means if means is not None else pd.Series(dtype=float)
            bin_ses_dict[col] = ses if ses is not None else pd.Series(dtype=float)
            bin_centers_dict[col] = centers if centers is not None else np.array([])
            mw, sw, cw = evaluate_feature_bins_detailed(
                eval_df, feature_name=col,
                target_col=target_col, n_bins=n_bins, winsorize_5mad=True,
            )
            bin_means_winsor_dict[col] = mw if mw is not None else pd.Series(dtype=float)
            bin_ses_winsor_dict[col] = sw if sw is not None else pd.Series(dtype=float)
            bin_centers_winsor_dict[col] = cw if cw is not None else np.array([])
        except Exception:
            bin_means_dict[col] = pd.Series(dtype=float)
            bin_ses_dict[col] = pd.Series(dtype=float)
            bin_centers_dict[col] = np.array([])
            bin_means_winsor_dict[col] = pd.Series(dtype=float)
            bin_ses_winsor_dict[col] = pd.Series(dtype=float)
            bin_centers_winsor_dict[col] = np.array([])

    rows = []
    for col in variant_cols:
        row = {
            "variant": col,
            "pearson_mean_ic": _safe_get(pearson_summary, col, "mean_ic"),
            "pearson_std_ic":  _safe_get(pearson_summary, col, "std_ic"),
            "pearson_ir":      _safe_get(pearson_summary, col, "ir"),
            "pearson_t_stat":  _safe_get(pearson_summary, col, "t_stat"),
            "spearman_mean_ic": _safe_get(spearman_summary, col, "mean_ic"),
            "spearman_ir":      _safe_get(spearman_summary, col, "ir"),
        }
        rows.append(row)

    summary = (
        pd.DataFrame(rows)
        .set_index("variant")
        .sort_values("pearson_ir", ascending=False)
    )

    return {
        "summary": summary,
        "pearson_ic_daily": pearson_ic,
        "spearman_ic_daily": spearman_ic,
        "bin_means": bin_means_dict,
        "bin_ses": bin_ses_dict,
        "bin_centers": bin_centers_dict,
        "bin_means_winsor": bin_means_winsor_dict,
        "bin_ses_winsor": bin_ses_winsor_dict,
        "bin_centers_winsor": bin_centers_winsor_dict,
    }


def _safe_get(df: pd.DataFrame, idx, col) -> float:
    try:
        return float(df.loc[idx, col])
    except (KeyError, TypeError):
        return np.nan
