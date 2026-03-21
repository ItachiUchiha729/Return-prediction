"""
Feature Lab Reporter
====================
生成带内嵌图表的自包含 HTML 评估报告。
除 matplotlib 外无其他外部依赖。
"""

import base64
import io
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 图表辅助
# ---------------------------------------------------------------------------

def _fig_to_base64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


# ---------------------------------------------------------------------------
# HTML 构建块
# ---------------------------------------------------------------------------

_CSS = """
body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
       margin: 32px auto; max-width: 1400px; background: #f8f9fa; color: #212529; }
h1   { border-bottom: 3px solid #0d6efd; padding-bottom: 10px; }
h2   { color: #0d6efd; margin-top: 44px; }
table { border-collapse: collapse; width: 100%; margin: 16px 0; font-size: 0.92em; }
th, td { border: 1px solid #dee2e6; padding: 7px 12px; text-align: right; }
th   { background: #0d6efd; color: white; position: sticky; top: 0; }
tr:nth-child(even) { background: #e9ecef; }
tr:hover { background: #cfe2ff; }
.best { background: #d1e7dd !important; font-weight: bold; }
.metric { font-size: 0.88em; color: #6c757d; }
img  { max-width: 100%; height: auto; margin: 8px 0; }
"""


def _html_head(title: str, description: str) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return (
        "<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8'>"
        f"<title>{title}</title><style>{_CSS}</style></head><body>"
        f"<h1>{title}</h1>"
        f"<p class='metric'>{description} &mdash; Generated: {now}</p>"
    )


def _html_tail() -> str:
    return "</body></html>"


# ---------------------------------------------------------------------------
# 汇总表
# ---------------------------------------------------------------------------

def _summary_table(summary: pd.DataFrame, labels: Dict[str, str]) -> str:
    best_ir = summary["pearson_ir"].idxmax() if not summary.empty else None
    html = "<h2>Summary (sorted by Pearson IR)</h2>\n<table>\n"
    html += (
        "<tr><th>#</th><th style='text-align:left'>Variant</th>"
        "<th>Pearson IC</th><th>Std IC</th><th>IR</th><th>t-stat</th>"
        "<th>Spearman IC</th><th>Spearman IR</th></tr>\n"
    )
    for i, (idx, row) in enumerate(summary.iterrows(), 1):
        cls = ' class="best"' if idx == best_ir else ""
        lbl = labels.get(idx, idx)
        html += (
            f"<tr{cls}>"
            f"<td>{i}</td>"
            f"<td style='text-align:left'>{lbl}</td>"
            f"<td>{row['pearson_mean_ic']:.4f}</td>"
            f"<td>{row['pearson_std_ic']:.4f}</td>"
            f"<td>{row['pearson_ir']:.3f}</td>"
            f"<td>{row['pearson_t_stat']:.2f}</td>"
            f"<td>{row['spearman_mean_ic']:.4f}</td>"
            f"<td>{row['spearman_ir']:.3f}</td>"
            f"</tr>\n"
        )
    html += "</table>\n"
    return html


# ---------------------------------------------------------------------------
# IC 时间序列图
# ---------------------------------------------------------------------------

def _ic_timeseries(
    ic_daily: pd.DataFrame,
    variants: List[str],
    labels: Dict[str, str],
    top_n: int = 8,
    rolling: int = 20,
) -> str:
    fig, ax = plt.subplots(figsize=(14, 5))
    for col in variants[:top_n]:
        s = ic_daily[col].dropna()
        if s.empty:
            continue
        rm = s.rolling(rolling, min_periods=1).mean()
        ax.plot(rm.index, rm.values, linewidth=1.2, label=labels.get(col, col))
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_title(f"Pearson IC ({rolling}-day rolling mean) — top {min(top_n, len(variants))} variants")
    ax.set_xlabel("Date")
    ax.set_ylabel("IC")
    ax.legend(fontsize=7, loc="best", ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    b64 = _fig_to_base64(fig)
    return f'<h2>IC Time Series</h2>\n<img src="data:image/png;base64,{b64}" />\n'


# ---------------------------------------------------------------------------
# 分桶图
# ---------------------------------------------------------------------------

def _bin_plots(
    bin_means: Dict[str, pd.Series],
    bin_ses: Dict[str, pd.Series],
    bin_centers: Dict[str, np.ndarray],
    variants: List[str],
    labels: Dict[str, str],
    title_suffix: str = "",
) -> str:
    n = len(variants)
    ncols = min(4, n)
    nrows = max(1, (n + ncols - 1) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)

    for i, col in enumerate(variants):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        bm = bin_means.get(col, pd.Series(dtype=float))
        bse = bin_ses.get(col, pd.Series(dtype=float))
        centers = bin_centers.get(col, np.array([]))
        if bm.empty or len(centers) == 0:
            ax.set_visible(False)
            continue
        valid = bm.notna()
        if not valid.any():
            ax.set_visible(False)
            continue
        x = np.asarray(centers)[valid.values]
        y = bm.loc[valid].values
        yerr = None
        if not bse.empty:
            ye = bse.reindex(bm.index).loc[valid].values
            yerr = np.where(np.isnan(ye), 0.0, ye)
        colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in y]
        if len(x) > 1:
            min_gap = np.diff(np.sort(x)).min()
            width = min_gap * 0.75
        else:
            width = max(abs(x[0]) * 0.2, 0.01) if x[0] != 0 else 0.1
        ax.bar(x, y, width=width, color=colors, edgecolor="white", yerr=yerr, capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{v:.3g}" for v in x], fontsize=8, rotation=45, ha="right")
        ax.axhline(0, color="grey", linewidth=0.6)
        ax.set_title(labels.get(col, col), fontsize=8)
        ax.set_ylabel("mean target ± SE")
        ax.tick_params(labelsize=7)

    for i in range(n, nrows * ncols):
        r, c = divmod(i, ncols)
        axes[r][c].set_visible(False)

    fig.suptitle(f"Bin Analysis{title_suffix} (mean ± SE target per MAD equal-width bin; x-axis = feature value)", fontsize=12)
    fig.tight_layout()
    b64 = _fig_to_base64(fig)
    h2 = "Bin Plots (5MAD winsorized)" if "5MAD" in title_suffix else "Bin Plots"
    return f'<h2>{h2}</h2>\n<img src="data:image/png;base64,{b64}" />\n'


# ---------------------------------------------------------------------------
# 参数热力图（仅当网格恰好有 2 个键时）
# ---------------------------------------------------------------------------

def _heatmap(
    summary: pd.DataFrame,
    definition: Dict[str, Any],
    metric: str = "pearson_ir",
) -> str:
    grid = definition.get("param_grid", {})
    if len(grid) != 2:
        return ""

    keys = sorted(grid.keys())
    k1, k2 = keys
    vals1 = grid[k1] if isinstance(grid[k1], list) else [grid[k1]]
    vals2 = grid[k2] if isinstance(grid[k2], list) else [grid[k2]]

    base = definition["name"]
    heat = np.full((len(vals1), len(vals2)), np.nan)
    for i, v1 in enumerate(vals1):
        for j, v2 in enumerate(vals2):
            col = f"{base}__{k1}={v1}__{k2}={v2}"
            if col in summary.index:
                heat[i, j] = summary.loc[col, metric]

    fig, ax = plt.subplots(
        figsize=(max(6, len(vals2) * 1.3), max(4, len(vals1) * 0.9)),
    )
    im = ax.imshow(heat, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(vals2)))
    ax.set_xticklabels([str(v) for v in vals2], fontsize=9)
    ax.set_yticks(range(len(vals1)))
    ax.set_yticklabels([str(v) for v in vals1], fontsize=9)
    ax.set_xlabel(k2)
    ax.set_ylabel(k1)
    ax.set_title(f"Parameter Heatmap — {metric}")

    vmax = np.nanmax(np.abs(heat)) if np.any(~np.isnan(heat)) else 1.0
    for i in range(len(vals1)):
        for j in range(len(vals2)):
            val = heat[i, j]
            if not np.isnan(val):
                clr = "white" if abs(val) > vmax * 0.55 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=8, color=clr)

    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    b64 = _fig_to_base64(fig)
    return f'<h2>Parameter Heatmap ({metric})</h2>\n<img src="data:image/png;base64,{b64}" />\n'


# ---------------------------------------------------------------------------
# 边际参数敏感性分析
# ---------------------------------------------------------------------------

def _extract_param_from_variant(variant_name: str, param_key: str):
    """从变体列名中提取 *param_key* 的值。"""
    for segment in variant_name.split("__"):
        if segment.startswith(f"{param_key}="):
            return segment[len(f"{param_key}="):]
    return None


def _marginal_analysis(
    summary: pd.DataFrame,
    definition: Dict[str, Any],
    metrics: Optional[List[str]] = None,
) -> str:
    """
    对每个参数键，绘制各指标随该参数取值的变化（对其他参数取平均）。

    显示均值 ± 1 标准差及散点，便于区分稳健趋势与噪声。
    """
    grid = definition.get("param_grid", {})
    if not grid:
        return ""

    if metrics is None:
        metrics = ["pearson_ir", "pearson_mean_ic", "pearson_t_stat"]
    metrics = [m for m in metrics if m in summary.columns]
    if not metrics:
        return ""

    keys = sorted(grid.keys())
    n_keys = len(keys)
    n_metrics = len(metrics)

    fig, axes = plt.subplots(
        n_keys, n_metrics,
        figsize=(5 * n_metrics, 4 * n_keys),
        squeeze=False,
    )

    for ki, key in enumerate(keys):
        raw_values = grid[key] if isinstance(grid[key], list) else [grid[key]]
        str_values = [str(v) for v in raw_values]

        for mi, metric in enumerate(metrics):
            ax = axes[ki][mi]
            means, stds, all_pts = [], [], []

            for sv in str_values:
                pts = []
                for idx in summary.index:
                    extracted = _extract_param_from_variant(idx, key)
                    if extracted == sv and not np.isnan(summary.loc[idx, metric]):
                        pts.append(summary.loc[idx, metric])
                means.append(np.mean(pts) if pts else np.nan)
                stds.append(np.std(pts) if len(pts) > 1 else 0.0)
                all_pts.append(pts)

            x = np.arange(len(str_values))
            ax.errorbar(x, means, yerr=stds, fmt="o-", capsize=5,
                        linewidth=1.8, markersize=7, color="#0d6efd",
                        ecolor="#6c757d", zorder=3)
            for xi, pts in enumerate(all_pts):
                ax.scatter([xi] * len(pts), pts, alpha=0.35, s=22,
                           color="#adb5bd", zorder=2)

            ax.set_xticks(x)
            ax.set_xticklabels(str_values, fontsize=9)
            ax.set_xlabel(key, fontsize=10)
            ax.set_ylabel(metric, fontsize=10)
            ax.set_title(f"{key} → {metric}", fontsize=10)
            ax.grid(True, alpha=0.3)

    fig.suptitle("Marginal Parameter Sensitivity", fontsize=13, y=1.01)
    fig.tight_layout()
    b64 = _fig_to_base64(fig)
    return (
        '<h2>Marginal Parameter Sensitivity</h2>'
        '<p class="metric">Blue line = mean across other parameters; '
        'grey dots = individual variants; error bars = ±1 std.  '
        'Smooth monotonic trends indicate a robust parameter effect; '
        'noisy / flat curves suggest the parameter matters little.</p>'
        f'\n<img src="data:image/png;base64,{b64}" />\n'
    )


# ---------------------------------------------------------------------------
# 时间分割稳健性（前半段 vs 后半段 IC）
# ---------------------------------------------------------------------------

def _time_split_robustness(
    ic_daily: pd.DataFrame,
    variants: List[str],
    labels: Dict[str, str],
) -> str:
    """
    Scatter first-half mean IC vs second-half mean IC for each variant.
    Points near the diagonal are temporally robust.
    """
    n = len(ic_daily)
    if n < 20:
        return ""
    mid = n // 2
    ic_first = ic_daily.iloc[:mid][variants].mean()
    ic_second = ic_daily.iloc[mid:][variants].mean()

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(ic_first.values, ic_second.values, s=50, alpha=0.7,
               color="#0d6efd", edgecolors="white", linewidths=0.5)

    for col in variants:
        short_label = labels.get(col, col).split("(")[-1].rstrip(")")
        if short_label == labels.get(col, col):
            short_label = col.split("__", 1)[-1] if "__" in col else col
        ax.annotate(short_label, (ic_first[col], ic_second[col]),
                    fontsize=6.5, alpha=0.8,
                    xytext=(4, 4), textcoords="offset points")

    all_vals = list(ic_first.values) + list(ic_second.values)
    vmin = min(all_vals) - abs(min(all_vals)) * 0.15
    vmax = max(all_vals) + abs(max(all_vals)) * 0.15
    ax.plot([vmin, vmax], [vmin, vmax], "k--", alpha=0.25, linewidth=1)

    ax.set_xlabel("Mean IC — first half", fontsize=11)
    ax.set_ylabel("Mean IC — second half", fontsize=11)
    ax.set_title("Time-Split Robustness", fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    b64 = _fig_to_base64(fig)
    return (
        '<h2>Time-Split Robustness</h2>'
        '<p class="metric">Each dot is one parameter variant. '
        'Dots near the diagonal have consistent IC across the first '
        'and second halves of the sample — a sign of genuine signal '
        'rather than overfitting.</p>'
        f'\n<img src="data:image/png;base64,{b64}" />\n'
    )


# ---------------------------------------------------------------------------
# 公开 API
# ---------------------------------------------------------------------------

def generate_report(
    definition: Dict[str, Any],
    eval_results: Dict[str, Any],
    output_path: str,
    variant_labels: Optional[Dict[str, str]] = None,
) -> str:
    """生成自包含 HTML 报告并返回文件路径。"""
    summary = eval_results["summary"]
    ic_daily = eval_results["pearson_ic_daily"]
    bin_means = eval_results["bin_means"]
    bin_ses = eval_results.get("bin_ses", {})
    bin_centers = eval_results.get("bin_centers", {})
    bin_means_winsor = eval_results.get("bin_means_winsor", {})
    bin_ses_winsor = eval_results.get("bin_ses_winsor", {})
    bin_centers_winsor = eval_results.get("bin_centers_winsor", {})

    sorted_variants = list(summary.index)
    if variant_labels is None:
        variant_labels = {v: v for v in sorted_variants}

    title = f"Feature Lab: {definition['name']}"
    desc = definition.get("description", "")

    html_parts = [
        _html_head(title, desc),
        _summary_table(summary, variant_labels),
        _heatmap(summary, definition),
        _marginal_analysis(summary, definition),
        _time_split_robustness(ic_daily, sorted_variants, variant_labels),
        _ic_timeseries(ic_daily, sorted_variants, variant_labels),
        _bin_plots(bin_means, bin_ses, bin_centers, sorted_variants, variant_labels, title_suffix=""),
        _bin_plots(bin_means_winsor, bin_ses_winsor, bin_centers_winsor, sorted_variants, variant_labels, title_suffix=" (5MAD winsorized)"),
        _html_tail(),
    ]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))

    return output_path
