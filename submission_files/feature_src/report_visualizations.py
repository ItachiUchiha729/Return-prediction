from pathlib import Path
from typing import List, Optional, Tuple

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .feature_evaluation import compute_daily_ic, summarize_ic


def _show_figure_if_interactive() -> None:
    """
    Show the current matplotlib figure only when running inside IPython/Jupyter.
    Avoid blocking CLI/script execution.
    """
    try:
        from IPython import get_ipython

        if get_ipython() is not None:
            plt.show()
    except Exception:
        pass


def resolve_submit_and_project_roots(start: Optional[Path] = None) -> Tuple[Path, Path]:
    """
    Resolve the submit directory and the project root from either the repo root
    or from inside submit/.
    """
    cwd = (start or Path.cwd()).resolve()
    for candidate in (cwd, cwd.parent):
        if (candidate / "feature_src").exists():
            return candidate, candidate.parent
        if (candidate / "submit" / "feature_src").exists():
            return candidate / "submit", candidate
    raise FileNotFoundError("Could not locate submit/feature_src from the current working directory.")


def get_feature_columns(df: pd.DataFrame, exclude: Optional[List[str]] = None) -> List[str]:
    exclude = set(exclude or [])
    return [c for c in df.columns if c not in exclude]


def load_training_feature_panel(project_root: Path) -> pd.DataFrame:
    """
    Load the 2010-2013 research panel used for feature-side ranking and bin plots.
    """
    train_path = project_root / "data" / "features_train.parquet"
    if not train_path.exists():
        raise FileNotFoundError(f"Training feature panel not found: {train_path}")

    train_df = pd.read_parquet(train_path).copy()
    train_df.index = pd.to_datetime(train_df.index)
    return train_df


def load_export_feature_panel(project_root: Path) -> Tuple[pd.DataFrame, str]:
    """
    Load the full feature panel used for monthly feature visualizations.
    Prefer exported Mode 1 CSV files; fall back to train/test parquet panels.
    """
    candidate_dirs = [
        project_root / "output_submit_full",
        project_root / "output_submit",
        project_root / "output_test",
    ]
    for directory in candidate_dirs:
        files = sorted(directory.glob("features_*.csv"))
        if files:
            frames = [pd.read_csv(path, parse_dates=["Date"]) for path in files]
            panel = pd.concat(frames, ignore_index=True)
            panel["Date"] = pd.to_datetime(panel["Date"])
            return panel, str(directory)

    train_path = project_root / "data" / "features_train.parquet"
    test_path = project_root / "data" / "features_test.parquet"
    if train_path.exists() and test_path.exists():
        train_df = pd.read_parquet(train_path).reset_index()
        test_df = pd.read_parquet(test_path).reset_index()
        panel = pd.concat([train_df, test_df], ignore_index=True)
        panel["Date"] = pd.to_datetime(panel["Date"])
        return panel, "data/features_train.parquet + data/features_test.parquet"

    raise FileNotFoundError("No exported feature CSV files or fallback parquet panels were found.")


def rank_features_by_ic(
    train_df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    target_col: str = "target_variable",
) -> pd.DataFrame:
    """
    Build a feature-side ranking table using daily cross-sectional IC.
    """
    if feature_cols is None:
        feature_cols = get_feature_columns(train_df, exclude=["Id", target_col])

    spearman_ic = compute_daily_ic(train_df, feature_cols=feature_cols, target_col=target_col, method="spearman")
    pearson_ic = compute_daily_ic(train_df, feature_cols=feature_cols, target_col=target_col, method="pearson")

    spearman_summary = summarize_ic(spearman_ic).add_prefix("spearman_")
    pearson_summary = summarize_ic(pearson_ic).add_prefix("pearson_")

    feature_rank = spearman_summary.join(pearson_summary, how="outer")
    feature_rank["abs_spearman_mean_ic"] = feature_rank["spearman_mean_ic"].abs()
    feature_rank = feature_rank.sort_values(
        ["abs_spearman_mean_ic", "spearman_ir"],
        ascending=False,
    )
    return feature_rank


def stable_qcut_bins(series: pd.Series, n_bins: int = 4) -> pd.Series:
    """
    Qcut-like equal-frequency binning that guarantees exactly n_bins when the
    sample size is sufficient, even if many values are tied after truncation.

    We break ties deterministically by ranking with method='first', then apply
    pandas.qcut to the unique ranks.
    """
    out = pd.Series(pd.NA, index=series.index, dtype="Int64")
    valid = series.dropna()
    if len(valid) < n_bins:
        return out

    ranks = valid.rank(method="first")
    bins = pd.qcut(ranks, q=n_bins, labels=False)
    valid_positions = np.flatnonzero(series.notna().to_numpy())
    out.iloc[valid_positions] = bins.to_numpy(dtype="int64")
    return out


def compute_feature_bin_summary(
    daily_df: pd.DataFrame,
    feature_name: str,
    target_col: str = "target_variable",
    n_bins: int = 4,
) -> pd.DataFrame:
    """
    Compute qcut-like equal-frequency feature bins and summarize target stats.
    """
    df = daily_df[[feature_name, target_col]].dropna().copy()
    if len(df) < n_bins:
        raise ValueError(f"Not enough observations to create {n_bins} bins for {feature_name}.")

    df["bin"] = stable_qcut_bins(df[feature_name], n_bins=n_bins)
    df = df.dropna(subset=["bin"]).copy()
    df["bin"] = df["bin"].astype(int)

    summary = (
        df.groupby("bin")
        .agg(
            feature_mean=(feature_name, "mean"),
            feature_min=(feature_name, "min"),
            feature_max=(feature_name, "max"),
            target_mean=(target_col, "mean"),
            target_std=(target_col, "std"),
            count=(target_col, "count"),
        )
        .reindex(range(n_bins))
    )
    summary["target_se"] = summary["target_std"] / np.sqrt(summary["count"])
    summary["bin_label"] = [f"Q{i + 1}" for i in range(n_bins)]
    return summary


def plot_top_feature_bin_plots(
    train_df: pd.DataFrame,
    feature_rank: pd.DataFrame,
    top_n: int = 6,
    n_bins: int = 4,
    output_path: Optional[Path] = None,
) -> Tuple[plt.Figure, List[str], Path]:
    """
    Plot 4-bin feature-vs-target charts for the top ranked features.
    """
    sns.set_style("whitegrid")
    plt.rcParams["figure.dpi"] = 130

    top_features = feature_rank.head(min(top_n, len(feature_rank))).index.tolist()
    ncols = 2
    nrows = math.ceil(len(top_features) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4.8 * nrows))
    axes = np.atleast_1d(axes).ravel()
    bar_colors = sns.color_palette("RdYlGn", n_colors=n_bins)

    for rank, feat in enumerate(top_features, start=1):
        ax = axes[rank - 1]
        summary = compute_feature_bin_summary(train_df, feat, n_bins=n_bins)

        x = np.arange(n_bins)
        y = summary["target_mean"].to_numpy() * 10000
        yerr = np.nan_to_num(summary["target_se"].to_numpy(), nan=0.0) * 10000

        ax.bar(x, y, yerr=yerr, capsize=3, color=bar_colors)
        ax.set_xticks(x)
        ax.set_xticklabels(summary["bin_label"].tolist(), fontsize=8)
        ax.axhline(0.0, color="gray", linewidth=0.8)
        ax.set_xlabel("Equal-frequency feature bin (qcut-like)", fontsize=9)
        ax.set_ylabel("Mean target +/- SE (bps)", fontsize=9)
        ax.set_title(
            (
                f"#{rank} {feat}\n"
                f"Spearman mean IC={feature_rank.loc[feat, 'spearman_mean_ic']:.4f}, "
                f"IR={feature_rank.loc[feat, 'spearman_ir']:.3f}"
            ),
            fontsize=10,
        )
        ax.tick_params(labelsize=8)

    for ax in axes[len(top_features):]:
        ax.set_visible(False)

    fig.suptitle(
        f"Most Significant Features vs Target (2010-2013 training sample, {n_bins} bins)",
        fontsize=14,
        y=1.02,
    )
    fig.tight_layout()

    if output_path is None:
        output_path = Path("top_feature_bin_plots_train_4bins.png")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    return fig, top_features, output_path


def compute_monthly_feature_summary(
    feature_panel: pd.DataFrame,
    feature_name: str,
) -> pd.DataFrame:
    """
    Compute monthly mean, 1st percentile, 99th percentile, min, and max.
    """
    if "Date" not in feature_panel.columns:
        raise KeyError("feature_panel must contain a Date column.")

    df = feature_panel[["Date", feature_name]].dropna().copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Month"] = df["Date"].dt.to_period("M").dt.to_timestamp()

    grp = df.groupby("Month")[feature_name]
    monthly = pd.DataFrame(
        {
            "mean": grp.mean(),
            "p01": grp.quantile(0.01),
            "p99": grp.quantile(0.99),
            "min": grp.min(),
            "max": grp.max(),
        }
    ).dropna(how="all")
    return monthly


def plot_monthly_feature_time_series(
    feature_panel: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    output_path: Optional[Path] = None,
    ncols: int = 3,
) -> Tuple[plt.Figure, Path]:
    """
    Plot each feature's monthly mean, 1st percentile, 99th percentile, min,
    and max as five explicit lines.
    """
    sns.set_style("whitegrid")
    plt.rcParams["figure.dpi"] = 130

    if feature_cols is None:
        feature_cols = get_feature_columns(feature_panel, exclude=["Date", "Id", "target_variable"])

    nrows = math.ceil(len(feature_cols) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 3.9 * nrows), sharex=False)
    axes = np.atleast_1d(axes).ravel()

    style_map = {
        "mean": {"color": "darkblue", "linewidth": 1.6, "linestyle": "-"},
        "p01": {"color": "steelblue", "linewidth": 1.0, "linestyle": "--"},
        "p99": {"color": "teal", "linewidth": 1.0, "linestyle": "-."},
        "min": {"color": "gray", "linewidth": 0.9, "linestyle": ":"},
        "max": {"color": "dimgray", "linewidth": 0.9, "linestyle": ":"},
    }

    for idx, feat in enumerate(feature_cols):
        ax = axes[idx]
        monthly = compute_monthly_feature_summary(feature_panel, feat)
        if monthly.empty:
            ax.set_visible(False)
            continue

        x = monthly.index
        for column in ["mean", "p01", "p99", "min", "max"]:
            ax.plot(x, monthly[column], label=column, **style_map[column])

        ax.set_title(feat, fontsize=10)
        ax.tick_params(axis="x", labelrotation=45, labelsize=7)
        ax.tick_params(axis="y", labelsize=8)
        if idx == 0:
            ax.legend(fontsize=7, loc="best")

    for ax in axes[len(feature_cols):]:
        ax.set_visible(False)

    fig.suptitle(
        "Monthly Feature Distribution Across the Full Five-Year Sample",
        fontsize=14,
        y=1.01,
    )
    fig.tight_layout()

    if output_path is None:
        output_path = Path("monthly_feature_distribution_all_features.png")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    return fig, output_path


def run_whitepaper_feature_bin_section(
    start: Optional[Path] = None,
    top_n: int = 6,
    n_bins: int = 4,
) -> dict:
    """
    One-call entry point for the white paper Section 7 figure.
    Resolves paths, loads data, ranks features, saves the figure, displays the
    ranking table when running in a notebook, and returns the artifacts.
    """
    submit_root, project_root = resolve_submit_and_project_roots(start)
    train_df = load_training_feature_panel(project_root)
    feature_rank = rank_features_by_ic(train_df)
    fig, top_features, output_path = plot_top_feature_bin_plots(
        train_df,
        feature_rank,
        top_n=top_n,
        n_bins=n_bins,
        output_path=submit_root / "figures" / "top_feature_bin_plots_train_4bins.png",
    )

    top_table = (
        feature_rank[
            [
                "spearman_mean_ic",
                "spearman_ir",
                "pearson_mean_ic",
                "pearson_ir",
            ]
        ]
        .head(10)
        .round(6)
    )
    try:
        from IPython.display import display

        display(top_table)
    except Exception:
        pass

    print(f"Training observations: {len(train_df):,}")
    print(f"Top features by |mean daily Spearman IC|: {top_features}")
    print(f"Saved bin-plot figure to: {output_path}")
    _show_figure_if_interactive()

    return {
        "figure": fig,
        "output_path": output_path,
        "top_features": top_features,
        "feature_rank": feature_rank,
        "top_table": top_table,
    }


def run_whitepaper_monthly_feature_appendix(
    start: Optional[Path] = None,
    ncols: int = 3,
) -> dict:
    """
    One-call entry point for the white paper appendix monthly feature figure.
    Resolves paths, loads the full feature panel, saves the figure, and returns
    the artifacts.
    """
    submit_root, project_root = resolve_submit_and_project_roots(start)
    feature_panel, feature_source = load_export_feature_panel(project_root)
    fig, output_path = plot_monthly_feature_time_series(
        feature_panel,
        output_path=submit_root / "figures" / "monthly_feature_distribution_all_features.png",
        ncols=ncols,
    )

    print(f"Feature source: {feature_source}")
    print(f"Rows loaded: {len(feature_panel):,}")
    print(f"Saved monthly appendix figure to: {output_path}")
    _show_figure_if_interactive()

    return {
        "figure": fig,
        "output_path": output_path,
        "feature_source": feature_source,
        "feature_panel_rows": len(feature_panel),
    }
