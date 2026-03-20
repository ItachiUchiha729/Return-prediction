"""
main.py - PDF submission entry point. Mode 1: features. Mode 2: predictions.

Usage:
  Mode 1: python main.py -i <raw_data_dir> -o <output_dir> -s YYYYMMDD -e YYYYMMDD -m 1
  Mode 2: python main.py -i <features_dir> -o <output_dir> -p <model_dir> -s YYYYMMDD -e YYYYMMDD -m 2

-i: raw input (Mode 1) or feature CSV dir (Mode 2)
-o: output directory
-p: model directory (Mode 2 only)
-s, -e: start/end date YYYYMMDD
-m: 1=features, 2=predictions
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Resolve sibling packages `feature_src` and `model_src` under this directory.
_SUBMIT_ROOT = str(Path(__file__).resolve().parent)
if _SUBMIT_ROOT not in sys.path:
    sys.path.insert(0, _SUBMIT_ROOT)

import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from feature_src.schema_check import sanity_check_and_time_intergrity_check
from feature_src.missing_value_drop import (
    drop_id_with_nan_by_trading_day,
    convert_cum_to_increment,
)
from feature_src.daily_data_loader import load_daily_data
from feature_src.feature_engineering import (
    build_daily_dataset,
    summarize_feature_nans,
    leakage_smoke_test,
)
from model_src.model_predictor import run_mode2_predictions


def _load_one_intraday_file(args: tuple) -> pd.DataFrame:
    """Worker: load one intraday CSV file. For ProcessPoolExecutor."""
    (
        file_path,
        expected_columns,
        expected_dtypes,
        reserved_columns,
        data_freq,
        expected_lines,
    ) = args
    return sanity_check_and_time_intergrity_check(
        file_path,
        expected_columns,
        expected_dtypes,
        reserved_columns,
        data_freq,
        expected_lines,
    )


def load_intraday_data(
    intraday_dir: Path,
    train_test_split: float = 0.8,
    expected_lines: int = 13000,
    n_jobs: int = -1,
    include_test: bool = False,
) -> pd.DataFrame:
    """
    Load and clean intraday data from CSV files.

    include_test=False: use first train_test_split of files (train set).
    include_test=True: use all files (train + test), for lookback when computing test features.
    """
    expected_columns = ["Date", "Time", "Id", "CumReturnResid", "CumReturnRaw", "CumVolume"]

    expected_dtypes = pd.Series(
        {
            "Date": "int64",
            "Time": "object",
            "Id": "object",
            "CumReturnResid": "float64",
            "CumReturnRaw": "float64",
            "CumVolume": "int64",
        }
    )

    reserved_columns = ["Id", "CumReturnResid", "CumReturnRaw", "CumVolume", "Time"]
    data_freq = "15m"

    files = sorted(list(intraday_dir.glob("*.csv")))
    if include_test:
        files_to_load = files
        print(f"Total files: {len(files)}")
        print(f"Using all files (train + test): {len(files_to_load)}")
    else:
        files_to_load = files[: int(len(files) * train_test_split)]
        print(f"Total files: {len(files)}")
        print(f"Using training files: {len(files_to_load)}")

    worker_args = [
        (
            fp,
            expected_columns,
            expected_dtypes,
            reserved_columns,
            data_freq,
            expected_lines,
        )
        for fp in files_to_load
    ]

    max_workers = None if n_jobs == -1 else max(1, n_jobs)
    data_list = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_load_one_intraday_file, a): str(a[0]) for a in worker_args}
        for f in tqdm(as_completed(futures), total=len(futures), desc="Loading intraday data"):
            data_list.append((futures[f], f.result()))
    data_list = [d for _, d in sorted(data_list, key=lambda x: x[0])]

    if not data_list:
        raise RuntimeError("No intraday CSV files found to load.")

    print("Concatenating data...")
    all_data = pd.concat(data_list, axis=0)
    all_data = all_data.sort_index()

    print("Converting cumulative values to increments...")
    all_data = convert_cum_to_increment(all_data)

    print("Dropping Id with NaN within trading day...")
    all_data = drop_id_with_nan_by_trading_day(all_data)

    all_data = all_data[
        ["Id", "ResidReturn", "RawReturn", "Volume", "CumReturnResid", "CumReturnRaw", "CumVolume"]
    ].sort_index()

    print("Removing timestamps where all stocks have zero raw return...")
    all_zero_times = (all_data["RawReturn"] == 0).groupby(level=0).all()
    all_zero_times = all_zero_times[all_zero_times].index
    all_data = all_data.loc[~all_data.index.isin(all_zero_times)].copy()

    print("Finished processing intraday data.")
    print(f"Final intraday dataset shape: {all_data.shape}")
    return all_data


def run_feature_pipeline(
    all_data: pd.DataFrame,
    feature_config_path: str,
    daily_data: Optional[pd.DataFrame] = None,
    run_tests: bool = False,
    include_target: bool = True,
) -> pd.DataFrame:
    """
    Compute the daily feature matrix from intraday (and optional daily) data.
    Optionally include target_variable for research/training workflows.
    """
    daily_df = build_daily_dataset(
        all_data=all_data,
        feature_config_path=feature_config_path,
        daily_data=daily_data,
        include_target=include_target,
    )
    print("Daily dataset built.")
    print(daily_df.info())

    print("\nNaN summary by feature:")
    nan_summary = summarize_feature_nans(daily_df)
    print(nan_summary)

    if run_tests:
        print("\nRunning leakage smoke test...")
        ok, diffs = leakage_smoke_test(
            all_data=all_data,
            feature_config_path=feature_config_path,
            daily_data=daily_data,
            full_daily_df=daily_df,
        )
        if ok:
            print("Leakage smoke test: PASSED (no differences detected in early window).")
        else:
            print("Leakage smoke test: POTENTIAL LEAKAGE DETECTED.")
            print("Maximum absolute differences per column:")
            print(diffs)

    return daily_df


def run_mode1(
    input_dir: Path,
    output_dir: Path,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    feature_config_path: Path,
) -> None:
    """
    Mode 1: Create leakage-safe features for [start_date, end_date], one CSV per date.
    Same data loading and pipeline as main.py, output format: CSV per date.
    """
    intraday_dir = input_dir / "data_intraday"
    daily_dir = input_dir / "data_daily"
    if not intraday_dir.exists():
        intraday_dir = input_dir / "intraday"
    if not daily_dir.exists():
        daily_dir = input_dir / "daily"

    if not intraday_dir.exists():
        raise FileNotFoundError(f"Intraday dir not found: {intraday_dir}")
    if not daily_dir.exists():
        raise FileNotFoundError(f"Daily dir not found: {daily_dir}")

    print("Loading daily data...")
    daily_data = load_daily_data(daily_dir=daily_dir)
    print(f"Daily data shape: {daily_data.shape}")

    print("Loading intraday data (full, include_test=True for any date range)...")
    all_data = load_intraday_data(
        intraday_dir=intraday_dir,
        include_test=True,
    )

    print("Building daily dataset (same pipeline as main.py)...")
    daily_df = run_feature_pipeline(
        all_data=all_data,
        feature_config_path=str(feature_config_path),
        daily_data=daily_data,
        run_tests=False,
        include_target=False,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = daily_df.reset_index()
    if "target_variable" in df.columns:
        df = df.drop(columns=["target_variable"])
    dates_in_range = df["Date"].dropna().unique()
    dates_in_range = sorted([d for d in dates_in_range if start_date <= pd.to_datetime(d) <= end_date])

    print(f"Saving {len(dates_in_range)} CSV files to {output_dir} ...")
    for d in tqdm(dates_in_range, desc="Writing CSV"):
        sub = df[df["Date"] == d]
        if sub.empty:
            continue
        dt = pd.to_datetime(d)
        fname = output_dir / f"features_{dt.strftime('%Y%m%d')}.csv"
        sub.to_csv(fname, index=False)
    print("Mode 1 completed.")


def run_mode2(
    features_dir: Path,
    output_dir: Path,
    model_dir: Path,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> None:
    """Mode 2: call teammate's function (`model_src.model_predictor`)."""
    run_mode2_predictions(features_dir, output_dir, model_dir, start_date, end_date)
    print("Mode 2 completed.")


def main():
    parser = argparse.ArgumentParser(description="PDF submission: Mode 1=features, Mode 2=predictions")
    parser.add_argument("-i", "--input", required=True, help="Raw data dir (Mode 1) or feature CSV dir (Mode 2)")
    parser.add_argument("-o", "--output", required=True, help="Output directory")
    parser.add_argument("-p", "--model", type=Path, default=None, help="Model directory (Mode 2 only, pickle files)")
    parser.add_argument("-s", "--start", required=True, help="Start date YYYYMMDD")
    parser.add_argument("-e", "--end", required=True, help="End date YYYYMMDD")
    parser.add_argument("-m", "--mode", type=int, default=1, help="Mode: 1=features, 2=predictions")
    args = parser.parse_args()

    start_date = pd.to_datetime(args.start, format="%Y%m%d")
    end_date = pd.to_datetime(args.end, format="%Y%m%d")
    if start_date > end_date:
        raise ValueError("start date must be <= end date")

    submit_dir = Path(__file__).resolve().parent
    feature_config_path = submit_dir / "feature_src" / "features_config.json"

    if args.mode == 1:
        run_mode1(
            input_dir=Path(args.input),
            output_dir=Path(args.output),
            start_date=start_date,
            end_date=end_date,
            feature_config_path=feature_config_path,
        )
    elif args.mode == 2:
        if args.model is None:
            raise ValueError("Mode 2 requires -p/--model (model directory)")
        run_mode2(
            features_dir=Path(args.input),
            output_dir=Path(args.output),
            model_dir=Path(args.model),
            start_date=start_date,
            end_date=end_date,
        )
    else:
        raise NotImplementedError(f"Mode {args.mode} not implemented (use 1 or 2)")


if __name__ == "__main__":
    main()
