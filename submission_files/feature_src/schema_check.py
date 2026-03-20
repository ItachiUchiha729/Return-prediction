import pandas as pd
from pathlib import Path
import warnings

def check_columns(data, expected_columns, file_path):
    realized_columns = list(data.columns)

    if realized_columns != expected_columns:
        warnings.warn(
            f"Unexpected columns in {file_path}. "
            f"Expected {expected_columns}, got {realized_columns}"
        )

def _normalize_dtype(dtype_str):
    """Treat 'object', 'str', 'string' as equivalent for pandas version compatibility."""
    s = str(dtype_str).lower()
    if s in ("object", "str", "string"):
        return "object"
    return s

def check_dtypes(data, expected_dtypes, file_path):
    actual_norm = data.dtypes.map(_normalize_dtype)
    expect_norm = expected_dtypes.map(_normalize_dtype)
    if not (actual_norm == expect_norm).all():
        warnings.warn(
            f"Unexpected dtypes in {file_path}. "
            f"Expected {expected_dtypes}, got {data.dtypes}"
        )

def create_time_index(data, reserved_columns, data_freq):

    if data_freq == "15m":
        data["Time"] = pd.to_datetime(
            data["Date"].astype(str) + " " + data["Time"]
        )

    elif data_freq == "daily":
        data["Time"] = pd.to_datetime(data["Date"].astype(str))

    else:
        warnings.warn(f"Unexpected data frequency {data_freq}")

    data = data[reserved_columns]
    data = data.set_index("Time")

    return data

def check_lines_and_time(data, file_path, expected_lines):

    # Check total row count
    realized_lines = data.shape[0]

    if realized_lines != expected_lines:
        warnings.warn(
            f"{file_path} has unexpected number of rows. "
            f"Expected {expected_lines}, got {realized_lines}"
        )

    # 检查每只股票的时间
    for stock_id, group in data.groupby("Id"):

        # Check row count
        if len(group) != 26:
            warnings.warn(
                f"{file_path} Id {stock_id} has {len(group)} rows instead of 26"
            )

        # Get time index
        time_index = group.index

        # Check all timestamps are same day
        days = time_index.date
        if len(set(days)) != 1:
            warnings.warn(
                f"{file_path} Id {stock_id} has multiple days: {set(days)}"
            )

        # Check start/end time
        start_time = time_index.min().time()
        end_time = time_index.max().time()

        if start_time.strftime("%H:%M") != "09:45" or end_time.strftime("%H:%M") != "16:00":
            warnings.warn(
                f"{file_path} Id {stock_id} has wrong time range "
                f"{start_time} -> {end_time}"
            )

def sanity_check_and_time_intergrity_check(
    file_path,
    expected_columns,
    expected_dtypes,
    reserved_columns,
    data_freq,
    expected_lines
):

    data = pd.read_csv(file_path)

    check_columns(data, expected_columns, file_path)
    check_dtypes(data, expected_dtypes, file_path)

    data = create_time_index(data, reserved_columns, data_freq)

    check_lines_and_time(data, file_path, expected_lines)

    return data