"""
nan_handler.py
==============
Deterministic, leakage-free NaN handling for the feature panel.

Apply the same pipeline to both train and test sets so that
preprocessing is identical in-sample and out-of-sample.

Usage
-----
    from feature_src.nan_handler import clean_feature_panel

    df_train_clean = clean_feature_panel(df_train)
    df_test_clean  = clean_feature_panel(df_test, warmup_cutoff=df_train_clean.index.min())
"""

from typing import List, Optional

import numpy as np
import pandas as pd


# ── Column groups (derived from NaN diagnosis) ─────────────────────────

# Features whose NaN means "zero volume / div-by-zero" → safe to fill 0
ZERO_FILL_COLS: List[str] = [
    "closing_auction_participation_ratio",
    "intraday_volume_concentration",
]

# Features whose NaN comes from rolling-lookback warmup or missing daily
# T-1 data → fill with same-day cross-sectional median
MEDIAN_FILL_COLS: List[str] = [
    "mean_reversion_ma",
    "bollinger_band",
    "raw_momentum",
    "short_term_reversal",
    "late_day_trend",
    "open_to_close_return",
    "close_position_daily_range",
]


def _get_feature_cols(df: pd.DataFrame) -> List[str]:
    """Return all feature column names (everything except Id and target)."""
    return [c for c in df.columns if c not in ("Id", "target_variable")]


def clean_feature_panel(
    df: pd.DataFrame,
    *,
    drop_target_nan: bool = True,
    warmup_cutoff: Optional[pd.Timestamp] = None,
    warmup_trading_days: int = 120,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Apply the full NaN-handling pipeline to a feature panel.

    Parameters
    ----------
    df : DataFrame
        Feature panel with DatetimeIndex, columns include ``Id``,
        ``target_variable``, and the 18 feature columns.
    drop_target_nan : bool, default True
        Drop rows where ``target_variable`` is NaN.  Set to False when
        cleaning a test set that has no target yet.
    warmup_cutoff : Timestamp or None
        If provided, drop all rows before this date.  Pass the training
        set's ``index.min()`` when cleaning the test set so both sets
        share the same cutoff logic.  If None, the cutoff is computed
        from the data (first ``warmup_trading_days`` trading days).
    warmup_trading_days : int, default 120
        Number of trading days to drop from the start when
        ``warmup_cutoff`` is not provided.  Covers the longest rolling
        window (SMA-120 in ``mean_reversion_ma``).
    verbose : bool, default True
        Print a step-by-step summary.

    Returns
    -------
    DataFrame
        Cleaned panel with zero NaNs in feature columns (and zero NaNs
        in ``target_variable`` if ``drop_target_nan`` is True).
    """
    out = df.copy()
    feature_cols = _get_feature_cols(out)
    n0 = len(out)

    def _log(msg: str) -> None:
        if verbose:
            print(msg)

    # ── Step 1: drop rows where target is NaN ──────────────────────────
    if drop_target_nan and "target_variable" in out.columns:
        out = out.dropna(subset=["target_variable"])
        _log(f"[1] Drop target NaN:          {n0:,} → {len(out):,}  "
             f"(−{n0 - len(out):,})")
    else:
        _log(f"[1] Target NaN drop:          skipped")

    # ── Step 2: drop rows where ALL features are NaN ───────────────────
    n1 = len(out)
    all_nan = out[feature_cols].isnull().all(axis=1)
    out = out[~all_nan]
    _log(f"[2] Drop all-NaN rows:        {n1:,} → {len(out):,}  "
         f"(−{n1 - len(out):,})")

    # ── Step 3: fill zero-volume edge cases with 0 ─────────────────────
    for c in ZERO_FILL_COLS:
        if c in out.columns:
            cnt = out[c].isnull().sum()
            out[c] = out[c].fillna(0.0)
            if cnt > 0:
                _log(f"[3] Fill {c}: {cnt:,} NaNs → 0")

    # ── Step 4: fill warmup NaNs with same-day cross-sectional median ──
    for c in MEDIAN_FILL_COLS:
        if c not in out.columns:
            continue
        before = out[c].isnull().sum()
        if before == 0:
            continue
        date_median = out.groupby(out.index)[c].transform("median")
        out[c] = out[c].fillna(date_median)
        after = out[c].isnull().sum()
        _log(f"[4] Median-fill {c}: "
             f"{before:,} → {after:,} NaNs")

    # ── Step 5: drop early warmup dates ────────────────────────────────
    if warmup_cutoff is None:
        trading_days = sorted(out.index.unique())
        if len(trading_days) > warmup_trading_days:
            warmup_cutoff = trading_days[warmup_trading_days]
            _log(f"[5] Warmup cutoff (auto):     {warmup_cutoff.date()}")
        else:
            _log(f"[5] Warmup cutoff:            skipped (not enough days)")
    else:
        _log(f"[5] Warmup cutoff (provided): {warmup_cutoff.date()}")

    if warmup_cutoff is not None:
        n_before = len(out)
        out = out[out.index >= warmup_cutoff]
        _log(f"    Rows after cutoff:        {n_before:,} → {len(out):,}  "
             f"(−{n_before - len(out):,})")

    # ── Step 6: fill any remaining stragglers with 0 ───────────────────
    remaining = out[feature_cols].isnull().sum().sum()
    if remaining > 0:
        _log(f"[6] Fill {remaining:,} remaining NaNs with 0")
        out[feature_cols] = out[feature_cols].fillna(0.0)
    else:
        _log(f"[6] No remaining NaNs ✓")

    # ── Assertions ─────────────────────────────────────────────────────
    assert out[feature_cols].isnull().sum().sum() == 0, \
        "Feature NaNs remain after cleaning"
    if drop_target_nan and "target_variable" in out.columns:
        assert out["target_variable"].isnull().sum() == 0, \
            "Target NaNs remain after cleaning"

    _log(f"\n Clean: {len(out):,} rows × {len(feature_cols)} features, "
         f"0 NaNs")
    return out
