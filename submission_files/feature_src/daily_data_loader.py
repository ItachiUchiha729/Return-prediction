"""
Load and preprocess daily data from data/data_daily/dat.*.csv.

- Price adjustment: P_adj = P * PxAdjFactor (Open, High, Low, Close)
- Volume adjustment: Volume_adj = Volume * SharesAdjFactor
- Date to datetime
- ID -> Id (align with intraday)
- Drop SYMBOL, MIC
- Keep only adjusted price/volume columns
"""

from pathlib import Path
from typing import Optional

import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


def _load_one_daily_file(file_path: Path) -> tuple[str, pd.DataFrame]:
    """Worker: read one daily CSV, rename and drop columns. Returns (path, df)."""
    df = pd.read_csv(file_path)
    if "ID" in df.columns:
        df = df.rename(columns={"ID": "Id"})
    df = df.drop(columns=["SYMBOL", "MIC"], errors="ignore")
    return (str(file_path), df)


def load_daily_data(
    daily_dir: Path,
    file_pattern: str = "dat.*.csv",
    max_files: Optional[int] = None,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """
    Load all daily CSV files, preprocess, and return a panel.

    Parameters
    ----------
    daily_dir : Path
        Directory containing dat.YYYYMMDD.csv files.
    file_pattern : str
        Glob pattern for daily files.

    Returns
    -------
    DataFrame
        Columns: Date, Id, FREE_FLOAT_PERCENTAGE, EST_VOL, MDV_63,
                 OpenAdj, HighAdj, LowAdj, CloseAdj, VolumeAdj,
                 PxAdjFactor, SharesAdjFactor.
        Index: default RangeIndex (use set_index(['Date','Id']) for panel).
    """
    files = sorted(daily_dir.glob(file_pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {file_pattern} in {daily_dir}")
    if max_files is not None:
        files = files[:max_files]

    max_workers = None if n_jobs == -1 else max(1, n_jobs)
    data_list = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_load_one_daily_file, fp): str(fp) for fp in files}
        for f in tqdm(as_completed(futures), total=len(futures), desc="Loading daily data"):
            data_list.append(f.result())
    data_list = [d for _, d in sorted(data_list, key=lambda x: x[0])]
    out = pd.concat(data_list, axis=0, ignore_index=True)

    # Date 转为 datetime
    out["Date"] = pd.to_datetime(out["Date"].astype(str), format="%Y%m%d")

    # Adjusted prices (do not keep raw)
    out["OpenAdj"] = out["Open"] * out["PxAdjFactor"]
    out["HighAdj"] = out["High"] * out["PxAdjFactor"]
    out["LowAdj"] = out["Low"] * out["PxAdjFactor"]
    out["CloseAdj"] = out["Close"] * out["PxAdjFactor"]
    out["VolumeAdj"] = out["Volume"] * out["SharesAdjFactor"]

    # 删除原始价格/成交量列
    out = out.drop(columns=["Open", "High", "Low", "Close", "Volume"], errors="ignore")

    # Reorder columns for readability
    keep_cols = [
        "Date",
        "Id",
        "FREE_FLOAT_PERCENTAGE",
        "EST_VOL",
        "MDV_63",
        "OpenAdj",
        "HighAdj",
        "LowAdj",
        "CloseAdj",
        "VolumeAdj",
        "PxAdjFactor",
        "SharesAdjFactor",
    ]
    out = out[[c for c in keep_cols if c in out.columns]]

    return out
