import numpy as np
import pandas as pd
from pathlib import Path
import warnings

def convert_cum_to_increment(all_data):
    # Ensure time order
    all_data = all_data.sort_index().copy()

    # Group key: Id + date
    date_key = all_data.index.normalize()
    group_key = [all_data["Id"], date_key]

    # 识别每组首行
    is_first = all_data.groupby(group_key).cumcount() == 0

    # Compute increments
    all_data["ResidReturn"] = (
        all_data.groupby(group_key)["CumReturnResid"].diff()
    )

    all_data["RawReturn"] = (
        all_data.groupby(group_key)["CumReturnRaw"].diff()
    )

    all_data["Volume"] = (
        all_data.groupby(group_key)["CumVolume"].diff()
    )

    # First row: fill with original cumulative values
    all_data.loc[is_first, "ResidReturn"] = all_data.loc[is_first, "CumReturnResid"]
    all_data.loc[is_first, "RawReturn"] = all_data.loc[is_first, "CumReturnRaw"]
    all_data.loc[is_first, "Volume"] = all_data.loc[is_first, "CumVolume"]

    return all_data

def drop_id_with_nan_by_trading_day(df):
    df = df.sort_index().copy()

    trading_day = df.index.normalize()

    has_nan = df.isna().any(axis=1).groupby([trading_day, df["Id"]]).transform("any")

    result = df.loc[~has_nan].copy()

    return result


# Keep old name as alias for backward compatibility
drop_id_with_nan_by_custom_day = drop_id_with_nan_by_trading_day
    