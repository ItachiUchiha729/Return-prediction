"""
Feature Lab Runner
==================
批量特征参数网格搜索的 CLI 入口。

用法
-----
    # 处理 feature_lab/definitions/ 下所有 definition JSON
    python feature_lab_runner.py

    # 处理单个 definition
    python feature_lab_runner.py --definition intraday_momentum.json

    # 清除缓存并重新处理
    python feature_lab_runner.py --clear-cache

    # 生成后在浏览器中打开报告
    python feature_lab_runner.py --open
"""

import argparse
import copy
import shutil
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

_SRC = str(Path(__file__).resolve().parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from feature_lab_generator import (
    expand_grid,
    generate_variants,
    load_definition,
    variant_col_name,
    variant_label,
)
from feature_lab_evaluator import evaluate_variants
from feature_lab_reporter import generate_report

from feature_engineering import (
    build_intraday_panel,
    compute_target_variable,
    _extract_cumvol_at_time,
)
from main import load_intraday_data
from daily_data_loader import load_daily_data


# ---------------------------------------------------------------------------
# 数据加载 / 缓存
# ---------------------------------------------------------------------------

def _load_or_cache(cache_dir: Path, intraday_dir: Path, daily_dir: Path):
    """若缓存存在则从缓存加载，否则从原始 CSV 加载并缓存。"""
    cache_dir.mkdir(parents=True, exist_ok=True)
    all_data_path = cache_dir / "all_data.parquet"
    daily_path = cache_dir / "daily_data.parquet"

    if all_data_path.exists():
        print(f"Loading cached intraday data: {all_data_path}")
        all_data = pd.read_parquet(all_data_path)
    else:
        print("Loading raw intraday CSVs (first run — will cache for future runs) ...")
        all_data = load_intraday_data(intraday_dir)
        all_data.to_parquet(all_data_path)
        print(f"  → cached to {all_data_path}")

    if daily_path.exists():
        print(f"Loading cached daily data: {daily_path}")
        daily_data = pd.read_parquet(daily_path)
    else:
        print("Loading raw daily CSVs ...")
        daily_data = load_daily_data(daily_dir)
        daily_data.to_parquet(daily_path)
        print(f"  → cached to {daily_path}")

    return all_data, daily_data


# ---------------------------------------------------------------------------
# 单个 definition 处理
# ---------------------------------------------------------------------------

def _trim_grid_for_quick_test(definition: dict) -> dict:
    """缩减参数网格，最多产生 2 个变体以快速测试。"""
    d = copy.deepcopy(definition)

    if "param_combos" in d:
        if len(d["param_combos"]) > 2:
            d["param_combos"] = d["param_combos"][:2]
        return d

    grid = d.get("param_grid", {})
    if not grid:
        return d

    found_multi = False
    for key in sorted(grid.keys()):
        if isinstance(grid[key], list) and len(grid[key]) > 1:
            if not found_multi:
                grid[key] = grid[key][:2]
                found_multi = True
            else:
                grid[key] = grid[key][:1]

    return d


def run_single_definition(
    def_path: Path,
    intraday_df: pd.DataFrame,
    target_df: pd.DataFrame,
    daily_data: Optional[pd.DataFrame],
    intraday_cumvol_1530: Optional[pd.Series],
    daily_date_shift_map: dict,
    report_dir: Path,
    n_bins: int = 10,
    quick: bool = False,
) -> str:
    """处理一个 definition JSON → 生成 HTML 报告。返回报告路径。"""
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"Processing: {def_path.name}")
    print(sep)

    definition = load_definition(str(def_path))
    if quick:
        definition = _trim_grid_for_quick_test(definition)
    overrides_list = expand_grid(definition)
    print(f"  Feature : {definition['name']}")
    print(f"  Type    : {definition['type']}")
    print(f"  Category: {definition.get('category', 'intraday')}")
    print(f"  Variants: {len(overrides_list)}")

    # ---- 生成 ----
    print("  Generating variants ...")
    variants_df = generate_variants(
        definition,
        intraday_df,
        daily_data=daily_data,
        intraday_cumvol_1530=intraday_cumvol_1530,
    )

    # 日频特征：shift Date T-1 → T 以与 target 对齐
    category = definition.get("category", "intraday")
    if category == "daily":
        idx = variants_df.index
        old_dates = idx.get_level_values(0)
        new_dates = old_dates.map(daily_date_shift_map)
        valid = new_dates.notna()
        variants_df = variants_df.loc[valid]
        new_dates = new_dates[valid]
        variants_df.index = pd.MultiIndex.from_arrays(
            [new_dates, idx.get_level_values(1)[valid]],
            names=idx.names,
        )
        if definition["type"] == "volume_surge_ratio" and intraday_cumvol_1530 is not None:
            cv = intraday_cumvol_1530.reindex(variants_df.index)
            for col in variants_df.columns:
                variants_df[col] = variants_df[col] * cv

    # ---- 与 target 合并 ----
    eval_df = variants_df.join(target_df, how="inner")
    eval_df = eval_df.reset_index()
    eval_df = eval_df.set_index("Date")

    variant_cols = [
        c for c in eval_df.columns
        if c not in ("Id", "target_variable", "Date")
    ]
    print(f"  Evaluating {len(variant_cols)} variants × {eval_df.index.nunique()} days ...")

    # ---- 评估 ----
    results = evaluate_variants(eval_df, variant_cols, n_bins=n_bins)

    # ---- 构建标签映射 ----
    labels = {}
    for ov in overrides_list:
        col = variant_col_name(definition["name"], ov)
        labels[col] = variant_label(definition["name"], ov)

    # ---- 报告 ----
    report_path = report_dir / f"{definition['name']}_report.html"
    generate_report(definition, results, str(report_path), variant_labels=labels)
    print(f"  Report → {report_path}")
    return str(report_path)


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Feature Lab — 批量参数网格搜索与评估",
    )
    parser.add_argument(
        "--definition", type=str, default=None,
        help="要处理的单个 definition JSON（文件名或路径）",
    )
    parser.add_argument(
        "--n-bins", type=int, default=10,
        help="分桶分析的分位数桶数（默认: 10）",
    )
    parser.add_argument(
        "--clear-cache", action="store_true",
        help="运行前删除缓存数据",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="快速测试模式：每个网格最多 2 个变体",
    )
    parser.add_argument(
        "--open", action="store_true",
        help="生成后在浏览器中打开 HTML 报告",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    lab_root = Path(__file__).resolve().parent
    intraday_dir = project_root / "data" / "data_intraday"
    daily_dir = project_root / "data" / "data_daily"
    definitions_dir = lab_root / "definitions"
    report_dir = lab_root / "reports"
    cache_dir = lab_root / "cache"

    if args.clear_cache and cache_dir.exists():
        shutil.rmtree(cache_dir)
        print("Cache cleared.")

    # ---- 加载数据 ----
    all_data, daily_data = _load_or_cache(cache_dir, intraday_dir, daily_dir)

    print("Building intraday panel ...")
    intraday_df = build_intraday_panel(all_data)

    print("Computing target variable ...")
    target_df = compute_target_variable(intraday_df)

    intraday_cumvol_1530 = _extract_cumvol_at_time(intraday_df, "15:30")

    daily_dates = sorted(pd.to_datetime(daily_data["Date"]).unique())
    daily_date_shift_map = dict(zip(daily_dates[:-1], daily_dates[1:]))

    # ---- 发现 definitions ----
    if args.definition:
        p = Path(args.definition)
        if not p.exists():
            p = definitions_dir / args.definition
        if not p.exists():
            raise FileNotFoundError(f"Definition not found: {args.definition}")
        def_files = [p]
    else:
        def_files = sorted(definitions_dir.glob("*.json"))
        if not def_files:
            print(f"No definition files found in {definitions_dir}")
            return
    print(f"\nFound {len(def_files)} definition(s) to process.\n")

    # ---- 处理 ----
    report_paths = []
    for def_path in def_files:
        try:
            rp = run_single_definition(
                def_path=def_path,
                intraday_df=intraday_df,
                target_df=target_df,
                daily_data=daily_data,
                intraday_cumvol_1530=intraday_cumvol_1530,
                daily_date_shift_map=daily_date_shift_map,
                report_dir=report_dir,
                n_bins=args.n_bins,
                quick=args.quick,
            )
            report_paths.append(rp)
        except Exception as exc:
            print(f"  ERROR processing {def_path.name}: {exc}")
            import traceback
            traceback.print_exc()

    print(f"\nAll done.  {len(report_paths)} report(s) saved to {report_dir}")

    if args.open:
        import os
        for rp in report_paths:
            os.startfile(rp)


if __name__ == "__main__":
    main()
