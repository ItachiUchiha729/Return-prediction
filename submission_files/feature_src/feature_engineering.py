import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm


@dataclass
class IntradaySettings:
    start_time: str = "09:45"
    close_time: str = "15:30"


def load_feature_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _time_only(series: pd.DatetimeIndex) -> pd.Index:
    """返回时间部分（HH:MM 字符串）便于筛选。"""
    return series.strftime("%H:%M")


def build_intraday_panel(all_data: pd.DataFrame) -> pd.DataFrame:
    """
    构建日内面板，交易日 = 文件日历日期。

    Parameters
    ----------
    all_data : DataFrame
        索引为无时区 Timestamp，列至少包含：
        ['Id', 'ResidReturn', 'RawReturn', 'Volume',
         'CumReturnResid', 'CumReturnRaw', 'CumVolume']。

    Returns
    -------
    DataFrame
        与输入相同列，外加 'Date'（交易日 = 日历日期）。
        索引仍为时间戳。
    """
    df = all_data.sort_index().copy()
    df["Date"] = df.index.normalize()
    return df


def _window_mask(
    idx: pd.DatetimeIndex, start_time: Optional[str], end_time: Optional[str]
) -> np.ndarray:
    times = _time_only(idx)
    mask = np.ones(len(idx), dtype=bool)
    if start_time is not None:
        mask &= times >= start_time
    if end_time is not None:
        mask &= times <= end_time
    return mask


def _safe_corr(x: pd.Series, y: pd.Series) -> float:
    if x.size < 2 or y.size < 2:
        return np.nan
    if x.std(ddof=0) == 0 or y.std(ddof=0) == 0:
        return np.nan
    return float(x.corr(y))


def _vectorized_window_mask(times: pd.Index, start_time: Optional[str], end_time: Optional[str]) -> np.ndarray:
    """对完整索引的向量化时间窗口掩码。"""
    t = times.astype(str)
    mask = np.ones(len(t), dtype=bool)
    if start_time is not None:
        mask &= (t >= start_time)
    if end_time is not None:
        mask &= (t <= end_time)
    return mask


def _compute_one_intraday_feature(
    df: pd.DataFrame, feat: Dict[str, Any]
) -> Optional[Tuple[str, pd.Series]]:
    """计算单个日内特征，供并行调用。返回 (name, series) 或 None。"""
    if not feat.get("enabled", True):
        return None
    if feat["type"] == "late_day_trend_deviation":
        return None
    name = feat["name"]
    ftype = feat["type"]
    params = feat.get("params", {})

    if ftype == "intraday_sum":
        col = feat["input"]
        st, et = params.get("start_time"), params.get("end_time")
        mask = _vectorized_window_mask(df["_time"], st, et)
        return (name, df.loc[mask].groupby(["Date", "Id"], sort=True)[col].sum())
    elif ftype == "intraday_diff_of_sums":
        col = feat["input"]
        first, second = params["first_window"], params["second_window"]
        m1 = _vectorized_window_mask(df["_time"], first.get("start_time"), first.get("end_time"))
        m2 = _vectorized_window_mask(df["_time"], second.get("start_time"), second.get("end_time"))
        s1 = df.loc[m1].groupby(["Date", "Id"], sort=True)[col].sum()
        s2 = df.loc[m2].groupby(["Date", "Id"], sort=True)[col].sum()
        idx = s1.index.union(s2.index)
        return (name, s1.reindex(idx).fillna(0) - s2.reindex(idx).fillna(0))
    elif ftype == "intraday_volatility":
        col = feat["input"]
        st, et = params.get("start_time"), params.get("end_time")
        mask = _vectorized_window_mask(df["_time"], st, et)
        return (name, df.loc[mask].groupby(["Date", "Id"], sort=True)[col].std(ddof=0))
    elif ftype == "intraday_skewness":
        col = feat["input"]
        st, et = params.get("start_time"), params.get("end_time")
        mask = _vectorized_window_mask(df["_time"], st, et)
        masked = df.loc[mask][["Date", "Id", col]].copy()
        masked["_mean"] = masked.groupby(["Date", "Id"], sort=True)[col].transform("mean")
        masked["_m3"] = (masked[col] - masked["_mean"]) ** 3
        m3_agg = masked.groupby(["Date", "Id"], sort=True)["_m3"].mean()
        std_agg = masked.groupby(["Date", "Id"], sort=True)[col].std(ddof=0)
        skew = m3_agg / (std_agg ** 3)
        skew = skew.where(std_agg > 0, np.nan)
        return (name, skew)
    elif ftype == "volume_weighted_momentum":
        ret_col = feat["inputs"]["returns"]
        vol_col = feat["inputs"]["volume"]
        st, et = params.get("start_time"), params.get("end_time")
        mask = _vectorized_window_mask(df["_time"], st, et)
        masked = df.loc[mask][["Date", "Id", ret_col, vol_col]].copy()
        masked["_vol"] = masked[vol_col].clip(lower=0)
        masked["_ret_vol"] = masked[ret_col] * masked["_vol"]
        num = masked.groupby(["Date", "Id"], sort=True)["_ret_vol"].sum()
        denom = masked.groupby(["Date", "Id"], sort=True)["_vol"].sum()
        vw = num / denom
        vw = vw.where(denom > 0, 0.0)
        return (name, vw)
    elif ftype == "early_late_volume_mix":
        cum_vol_col = feat["inputs"]["cum_volume"]
        start, split, end = params["start_time"], params["split_time"], params["end_time"]
        m_start, m_split, m_end = df["_time"] <= start, df["_time"] <= split, df["_time"] <= end
        cv_start = df.loc[m_start].groupby(["Date", "Id"], sort=True)[cum_vol_col].last()
        cv_split = df.loc[m_split].groupby(["Date", "Id"], sort=True)[cum_vol_col].last()
        cv_end = df.loc[m_end].groupby(["Date", "Id"], sort=True)[cum_vol_col].last()
        idx = cv_start.index.union(cv_split.index).union(cv_end.index)
        cv_start, cv_split, cv_end = cv_start.reindex(idx), cv_split.reindex(idx), cv_end.reindex(idx)
        v_early, v_late = cv_split - cv_start, cv_end - cv_split
        v_total = cv_end - cv_start
        mix = (v_late - v_early) / v_total
        mix = mix.where(v_total != 0, 0.0).where(cv_start.notna() & cv_split.notna() & cv_end.notna(), np.nan)
        return (name, mix)
    elif ftype == "intraday_corr":
        x_col, y_col = feat["inputs"]["x"], feat["inputs"]["y"]
        st, et = params.get("start_time"), params.get("end_time")
        mask = _vectorized_window_mask(df["_time"], st, et)
        masked = df.loc[mask][["Date", "Id", x_col, y_col]].copy()
        masked["_xy"], masked["_x2"], masked["_y2"] = masked[x_col] * masked[y_col], masked[x_col] ** 2, masked[y_col] ** 2
        agg = masked.groupby(["Date", "Id"], sort=True).agg(
            sum_x=(x_col, "sum"), sum_y=(y_col, "sum"), sum_xy=("_xy", "sum"),
            sum_x2=("_x2", "sum"), sum_y2=("_y2", "sum"), n=(x_col, "count"))
        cov = agg["sum_xy"] / agg["n"] - (agg["sum_x"] / agg["n"]) * (agg["sum_y"] / agg["n"])
        var_x = agg["sum_x2"] / agg["n"] - (agg["sum_x"] / agg["n"]) ** 2
        var_y = agg["sum_y2"] / agg["n"] - (agg["sum_y"] / agg["n"]) ** 2
        std_prod = np.sqrt(var_x * var_y)
        corr = cov / std_prod
        corr = corr.where((std_prod > 0) & (agg["n"] >= 2), np.nan)
        return (name, corr)
    elif ftype == "gap_vs_continuation":
        col = feat["input"]
        eps = float(params.get("epsilon", 1e-6))
        m_open, m_close = df["_time"] >= params["open_time"], df["_time"] <= params["close_time"]
        r_gap = df.loc[m_open].groupby(["Date", "Id"], sort=True)[col].first()
        r_close = df.loc[m_close].groupby(["Date", "Id"], sort=True)[col].last()
        idx = r_gap.index.union(r_close.index)
        r_gap, r_close = r_gap.reindex(idx), r_close.reindex(idx)
        gap_cont = (r_close - r_gap) / (np.abs(r_gap) + eps)
        gap_cont = gap_cont.where(r_gap.notna() & r_close.notna(), np.nan)
        return (name, gap_cont)
    elif ftype == "intraday_lag1_autocorr":
        col = feat["input"]
        st, et = params.get("start_time"), params.get("end_time")
        mask = _vectorized_window_mask(df["_time"], st, et)
        masked = df.loc[mask][["Date", "Id", col]].copy().sort_index()
        masked["_x_prev"] = masked.groupby(["Date", "Id"], sort=True)[col].shift(1)
        masked = masked.dropna(subset=["_x_prev"])
        masked["_xy"], masked["_x2"], masked["_y2"] = masked[col] * masked["_x_prev"], masked[col] ** 2, masked["_x_prev"] ** 2
        agg = masked.groupby(["Date", "Id"], sort=True).agg(
            n=(col, "count"), sum_x=(col, "sum"), sum_y=("_x_prev", "sum"),
            sum_xy=("_xy", "sum"), sum_x2=("_x2", "sum"), sum_y2=("_y2", "sum"))
        cov = agg["sum_xy"] / agg["n"] - (agg["sum_x"] / agg["n"]) * (agg["sum_y"] / agg["n"])
        var_x = agg["sum_x2"] / agg["n"] - (agg["sum_x"] / agg["n"]) ** 2
        var_y = agg["sum_y2"] / agg["n"] - (agg["sum_y"] / agg["n"]) ** 2
        std_prod = np.sqrt(var_x * var_y)
        lag1 = cov / std_prod
        lag1 = lag1.where((std_prod > 0) & (agg["n"] >= 3), np.nan)
        return (name, lag1)
    elif ftype == "closing_auction_participation":
        col = feat["input"]
        t1530, t1600 = params.get("time_1530", "15:30"), params.get("time_1600", "16:00")
        mask1530, mask1600 = df["_time"] == t1530, df["_time"] == t1600
        cv1530 = df.loc[mask1530].groupby(["Date", "Id"], sort=True)[col].first()
        cv1600 = df.loc[mask1600].groupby(["Date", "Id"], sort=True)[col].first()
        idx = cv1530.index.union(cv1600.index)
        cv1530, cv1600 = cv1530.reindex(idx), cv1600.reindex(idx)
        valid = (cv1530 != 0) & cv1530.notna()
        return (name, pd.Series(np.where(valid, (cv1600 - cv1530) / cv1530, np.nan), index=idx))
    elif ftype == "amihud_illiquidity":
        ret_col, vol_col = feat["inputs"]["returns"], feat["inputs"]["volume"]
        st, et = params.get("start_time"), params.get("end_time")
        mask = _vectorized_window_mask(df["_time"], st, et)
        masked = df.loc[mask][["Date", "Id", ret_col, vol_col]].copy()
        masked["_vol_safe"] = masked[vol_col].clip(lower=1)
        masked["_amihud"] = masked[ret_col].abs() / masked["_vol_safe"]
        return (name, masked.groupby(["Date", "Id"], sort=True)["_amihud"].mean())
    elif ftype == "intraday_range":
        col = feat.get("input", "CumReturnResid")
        st, et = params.get("start_time"), params.get("end_time")
        mask = _vectorized_window_mask(df["_time"], st, et)
        masked = df.loc[mask][["Date", "Id", col]].copy()
        agg = masked.groupby(["Date", "Id"], sort=True)[col].agg(["max", "min"])
        return (name, agg["max"] - agg["min"])
    elif ftype == "roll_measure":
        col = feat.get("input", "ResidReturn")
        st, et = params.get("start_time"), params.get("end_time")
        mask = _vectorized_window_mask(df["_time"], st, et)
        masked = df.loc[mask][["Date", "Id", col]].copy()
        masked["_lag"] = masked.groupby(["Date", "Id"], sort=True)[col].shift(1)
        masked["_prod"] = masked[col] * masked["_lag"]
        agg = masked.groupby(["Date", "Id"], sort=True).agg({col: "mean", "_lag": "mean", "_prod": "mean"})
        cov = agg["_prod"] - agg[col] * agg["_lag"]
        return (name, pd.Series(np.where(cov < 0, 2 * np.sqrt(-cov), np.nan), index=cov.index))
    elif ftype == "intraday_kurtosis":
        col = feat.get("input", "ResidReturn")
        st, et = params.get("start_time"), params.get("end_time")
        mask = _vectorized_window_mask(df["_time"], st, et)
        masked = df.loc[mask][["Date", "Id", col]].copy()
        masked["_mean"] = masked.groupby(["Date", "Id"], sort=True)[col].transform("mean")
        masked["_m4"] = (masked[col] - masked["_mean"]) ** 4
        m4_agg = masked.groupby(["Date", "Id"], sort=True)["_m4"].mean()
        std_agg = masked.groupby(["Date", "Id"], sort=True)[col].std(ddof=0)
        kurt = m4_agg / (std_agg ** 4) - 3.0
        return (name, kurt.where(std_agg > 0, np.nan))
    elif ftype == "intraday_vwap_deviation":
        ret_col = feat.get("input", "CumReturnResid")
        vol_col = feat.get("inputs", {}).get("volume", "Volume") if feat.get("inputs") else "Volume"
        st, et = params.get("start_time"), params.get("end_time")
        mask = _vectorized_window_mask(df["_time"], st, et)
        masked = df.loc[mask][["Date", "Id", ret_col, vol_col]].copy()
        masked["_vol"] = masked[vol_col].clip(lower=0)
        masked["_one_plus_r"], masked["_weight"] = 1.0 + masked[ret_col], (1.0 + masked[ret_col]) * masked["_vol"]
        agg = masked.groupby(["Date", "Id"], sort=True).agg(sum_weight=("_weight", "sum"), sum_vol=("_vol", "sum"))
        last = masked.groupby(["Date", "Id"], sort=True)[ret_col].last()
        vwap_proxy = agg["sum_weight"] / agg["sum_vol"]
        vwap_proxy = vwap_proxy.where(agg["sum_vol"] > 0, np.nan)
        deviation = (1.0 + last) / vwap_proxy - 1.0
        return (name, deviation.where(vwap_proxy > 0, np.nan))
    elif ftype == "intraday_volume_concentration":
        vol_col = feat.get("inputs", {}).get("volume", "Volume") if feat.get("inputs") else "Volume"
        st, et = params.get("start_time"), params.get("end_time")
        mask = _vectorized_window_mask(df["_time"], st, et)
        masked = df.loc[mask][["Date", "Id", vol_col]].copy()
        masked["_vol"] = masked[vol_col].clip(lower=0)
        v_total = masked.groupby(["Date", "Id"], sort=True)["_vol"].transform("sum")
        masked["_share"] = masked["_vol"] / v_total
        masked["_share2"] = masked["_share"] ** 2
        herfindahl = masked.groupby(["Date", "Id"], sort=True)["_share2"].sum()
        v_sum = masked.groupby(["Date", "Id"], sort=True)["_vol"].sum()
        return (name, herfindahl.where(v_sum > 0, np.nan))
    return None


def compute_intraday_features_from_config(
    intraday_df: pd.DataFrame, config: Dict[str, Any], n_jobs: int = 1,
) -> pd.DataFrame:
    """
    计算所有日内特征。均使用向量化 groupby（无 apply）。
    n_jobs: 并行进程数，1 为串行，>1 使用 joblib 并行（threading 后端共享内存）。
    """
    features_cfg: List[Dict[str, Any]] = config.get("intraday_features", [])

    df = intraday_df.copy()
    df["_time"] = _time_only(df.index)
    group = df.groupby(["Date", "Id"], sort=True)

    main_features = [
        f for f in features_cfg
        if f.get("enabled", True) and f["type"] != "late_day_trend_deviation"
    ]
    if n_jobs != 1:
        parallel_results = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(_compute_one_intraday_feature)(df, feat) for feat in main_features
        )
        results = {r[0]: r[1] for r in parallel_results if r is not None}
    else:
        results = {}
        for feat in main_features:
            r = _compute_one_intraday_feature(df, feat)
            if r is not None:
                results[r[0]] = r[1]

    # 午后趋势偏离
    for feat in features_cfg:
        if not feat.get("enabled", True) or feat["type"] != "late_day_trend_deviation":
            continue
        name = feat["name"]
        col = feat["input"]
        params = feat.get("params", {})
        start_time = params.get("start_time", "14:30")
        end_time = params.get("end_time", "15:30")
        lookback = int(params.get("lookback_days", 5))
        mask = _vectorized_window_mask(df["_time"], start_time, end_time)
        late_mom_series = df.loc[mask].groupby(["Date", "Id"], sort=True)[col].sum()
        # 每个 Id 过去 K 日的滚动均值，排除当日
        s = late_mom_series.unstack("Id")
        rolling_mean = s.shift(1).rolling(lookback, min_periods=1).mean()
        deviation = s - rolling_mean
        results[name] = deviation.stack()

    # 应用配置中的可选方向符号（默认 +1）。
    sign_map = {
        feat["name"]: float(feat.get("sign", 1.0))
        for feat in features_cfg
        if feat.get("enabled", True)
    }
    for name, sign in sign_map.items():
        if name in results and sign != 1.0:
            results[name] = results[name] * sign

    if not results:
        return pd.DataFrame(index=group.size().index)

    feature_df = pd.concat(results.values(), axis=1)
    feature_df.columns = list(results.keys())
    feature_df.index.names = ["Date", "Id"]
    return feature_df


def compute_target_variable(intraday_df: pd.DataFrame) -> pd.DataFrame:
    """
    按 project_2026.pdf 计算目标变量：

    target(T, Id) = ResidReturn(T, 16:00)           [T 15:30 -> T 16:00]
                  + CumReturnResid(T+1_trading, 15:30) [T 16:00 -> T+1 15:30]

    使用数据推导的交易日历，正确处理周末和节假日（如周五目标跨至周一）。
    """
    df = intraday_df.sort_index().copy()
    time_str = _time_only(df.index)

    resid_1530_1600 = (
        df.loc[time_str == "16:00"]
        .set_index(["Date", "Id"])["ResidReturn"]
    )

    next_1530 = (
        df.loc[time_str == "15:30"]
        .set_index(["Date", "Id"])["CumReturnResid"]
    )

    trading_days = sorted(df["Date"].unique())
    prev_td = dict(zip(trading_days[1:], trading_days[:-1]))

    idx = next_1530.index.to_frame(index=False)
    idx["Date"] = idx["Date"].map(prev_td)
    valid = idx["Date"].notna()
    next_1530 = next_1530[valid.values]
    idx = idx.loc[valid]
    next_1530.index = pd.MultiIndex.from_frame(idx)

    target = resid_1530_1600.add(next_1530)
    target = target.to_frame(name="target_variable")
    target.index.names = ["Date", "Id"]
    return target


def compute_daily_features_from_config(
    daily_df: pd.DataFrame,
    config: Dict[str, Any],
    intraday_cumvol_1530: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    计算日频特征。均使用 T-1 或更早数据（无截断/标准化）。
    输出索引为 (Date, Id)；调用方在合并时会将 Date 加 1，使 T-1 行特征与预测日 T 对齐。

    Parameters
    ----------
    daily_df : DataFrame
        日频数据，含 Date, Id, OpenAdj, HighAdj, LowAdj, CloseAdj, EST_VOL, MDV_63 等。
    config : dict
        解析后的 JSON 配置（使用 'daily_features' 节）。
    intraday_cumvol_1530 : Series, optional
        日内 15:30 的 CumVolume，按 (Date, Id)；volume_surge_ratio 需要。
    """
    daily_features_cfg: List[Dict[str, Any]] = config.get("daily_features", [])
    if not daily_features_cfg:
        return pd.DataFrame()

    df = daily_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Id", "Date"]).set_index(["Date", "Id"])

    results: Dict[str, pd.Series] = {}

    for feat in daily_features_cfg:
        if not feat.get("enabled", True):
            continue

        name = feat["name"]
        ftype = feat["type"]
        params = feat.get("params", {})

        if ftype == "realized_vs_implied_vol":
            ann_days = int(params.get("annualization_days", 252))
            denom = df["CloseAdj"] * df["EST_VOL"] * np.sqrt(1.0 / ann_days)
            denom = denom.replace(0, np.nan)
            s = (df["HighAdj"] - df["LowAdj"]) / denom
            results[name] = s

        elif ftype == "overnight_gap_return":
            lag = int(params.get("lag", 1))
            close_prev = df.groupby(level="Id")["CloseAdj"].shift(lag)
            s = df["OpenAdj"] / close_prev - 1.0
            results[name] = s

        elif ftype == "vol_scaled_momentum":
            K = int(params.get("K", 21))
            ret = df.groupby(level="Id")["CloseAdj"].pct_change()
            # R^(K) = 从 d=2 到 K+1 的 (1+r) 乘积减 1；排除 t-1（1 日反转）
            # 向量化：log(prod(1+r)) = sum(log(1+r))，故 prod(1+r)-1 = expm1(sum(log1p(r)))
            one_plus = (1 + ret).groupby(level="Id").shift(2)
            log1p_ret = np.log(one_plus.clip(lower=1e-16))
            cumsum_log = log1p_ret.groupby(level="Id").transform(
                lambda x: x.rolling(K, min_periods=1).sum()
            )
            cumret = np.expm1(cumsum_log)
            est_vol = df["EST_VOL"].replace(0, np.nan)
            s = cumret / est_vol
            results[name] = s

        elif ftype == "volume_surge_ratio":
            if intraday_cumvol_1530 is None:
                results[name] = pd.Series(index=df.index, dtype=float)
                continue
            # daily_df 行为 T-1：CloseAdj, MDV_63。需要 (T, Id) 的 CumVol_15:30。
            # 输出索引为 (Date, Id) = (T-1, Id)，调用方会 +1 对齐。
            # volume_surge_ratio = CumVol(T, Id) * CloseAdj(T-1) / MDV(T-1)。
            # 此处计算 CloseAdj/MDV；调用方在 build_daily_dataset 中 shift 后乘以 CumVol_15:30。
            ratio = df["CloseAdj"] / df["MDV_63"].replace(0, np.nan)
            results[name] = ratio  # 调用方 shift 后会乘以 CumVol_15:30

        elif ftype == "mean_reversion_ma":
            K = int(params.get("K", 20))
            close = df["CloseAdj"]
            ma = close.groupby(level="Id").transform(
                lambda x: x.rolling(K, min_periods=max(1, K // 2)).mean()
            )
            ma = ma.replace(0, np.nan)
            results[name] = close / ma - 1.0

        elif ftype == "short_term_reversal":
            lag = int(params.get("lag", 1))
            results[name] = df.groupby(level="Id")["CloseAdj"].pct_change(lag)

        elif ftype == "bollinger_band_position":
            K = int(params.get("K", 20))
            close = df["CloseAdj"]
            ma = close.groupby(level="Id").transform(
                lambda x: x.rolling(K, min_periods=max(1, K // 2)).mean()
            )
            rstd = close.groupby(level="Id").transform(
                lambda x: x.rolling(K, min_periods=max(1, K // 2)).std(ddof=1)
            )
            rstd = rstd.replace(0, np.nan)
            results[name] = (close - ma) / rstd

        elif ftype == "raw_momentum":
            K = int(params.get("K", 21))
            ret = df.groupby(level="Id")["CloseAdj"].pct_change()
            one_plus = (1 + ret).groupby(level="Id").shift(2)
            log1p_ret = np.log(one_plus.clip(lower=1e-16))
            cumsum_log = log1p_ret.groupby(level="Id").transform(
                lambda x: x.rolling(K, min_periods=1).sum()
            )
            cumret = np.expm1(cumsum_log)
            results[name] = cumret

        elif ftype == "turnover":
            vol = df["VolumeAdj"]
            mdv = df["MDV_63"].replace(0, np.nan)
            results[name] = vol / mdv

        elif ftype == "week52_high":
            K = int(params.get("K", 252))
            close = df["CloseAdj"]
            rmin = close.groupby(level="Id").transform(
                lambda x: x.rolling(K, min_periods=K // 2).min()
            )
            rmax = close.groupby(level="Id").transform(
                lambda x: x.rolling(K, min_periods=K // 2).max()
            )
            rng = rmax - rmin
            s = (close - rmin) / rng.where(rng > 0, np.nan)
            results[name] = s

        elif ftype == "max_daily_return":
            K = int(params.get("K", 20))
            ret = df.groupby(level="Id")["CloseAdj"].pct_change()
            ret_shifted = ret.groupby(level="Id").shift(1)
            rmax = ret_shifted.groupby(level="Id").transform(
                lambda x: x.rolling(K, min_periods=1).max()
            )
            results[name] = rmax

        elif ftype == "high_low_range":
            denom = df["CloseAdj"].replace(0, np.nan)
            results[name] = (df["HighAdj"] - df["LowAdj"]) / denom

        elif ftype == "volatility_ratio":
            K_short = int(params.get("K_short", 5))
            K_long = int(params.get("K_long", 21))
            ret = df.groupby(level="Id")["CloseAdj"].pct_change()
            std_short = ret.groupby(level="Id").transform(
                lambda x: x.rolling(K_short, min_periods=max(1, K_short // 2)).std(ddof=1)
            )
            std_long = ret.groupby(level="Id").transform(
                lambda x: x.rolling(K_long, min_periods=max(1, K_long // 2)).std(ddof=1)
            )
            ratio = std_short / std_long.where(std_long > 0, np.nan)
            results[name] = ratio

        elif ftype == "volume_trend":
            K = int(params.get("K", 20))
            vol = df["VolumeAdj"]
            ma = vol.groupby(level="Id").transform(
                lambda x: x.rolling(K, min_periods=max(1, K // 2)).mean()
            )
            ma = ma.replace(0, np.nan)
            results[name] = vol / ma - 1.0

        elif ftype == "open_to_close_return":
            denom = df["OpenAdj"].replace(0, np.nan)
            results[name] = (df["CloseAdj"] - df["OpenAdj"]) / denom

        elif ftype == "down_volume_ratio":
            K = int(params.get("K", 20))
            ret = df.groupby(level="Id")["CloseAdj"].pct_change()
            vol = df["VolumeAdj"]
            up_mask = ret > 0
            down_mask = ret < 0
            v_down = vol.where(down_mask, 0).groupby(level="Id").transform(
                lambda x: x.rolling(K, min_periods=1).sum()
            )
            v_total = vol.groupby(level="Id").transform(
                lambda x: x.rolling(K, min_periods=1).sum()
            )
            ratio = v_down / v_total.where(v_total > 0, np.nan)
            results[name] = ratio

        elif ftype == "close_position_daily_range":
            denom = (df["HighAdj"] - df["LowAdj"]).replace(0, np.nan)
            numer = 2 * df["CloseAdj"] - df["HighAdj"] - df["LowAdj"]
            results[name] = numer / denom

        elif ftype == "upper_shadow_ratio":
            hl = df["HighAdj"] - df["LowAdj"]
            top = df["HighAdj"] - np.maximum(df["OpenAdj"], df["CloseAdj"])
            denom = hl.replace(0, np.nan)
            results[name] = top / denom

        elif ftype == "up_to_down_volume_ratio":
            K = int(params.get("K", 20))
            ret = df.groupby(level="Id")["CloseAdj"].pct_change()
            vol = df["VolumeAdj"]
            v_up = vol.where(ret > 0, 0).groupby(level="Id").transform(
                lambda x: x.rolling(K, min_periods=1).sum()
            )
            v_down = vol.where(ret < 0, 0).groupby(level="Id").transform(
                lambda x: x.rolling(K, min_periods=1).sum()
            )
            ratio = v_up / v_down.where(v_down > 0, np.nan)
            results[name] = ratio

        elif ftype == "volume_autocorrelation":
            K = int(params.get("K", 20))
            df["_vol_lag"] = df["VolumeAdj"].groupby(level="Id").shift(1)
            corr = df.groupby(level="Id").apply(
                lambda g: g["VolumeAdj"].rolling(K, min_periods=max(2, K // 2)).corr(g["_vol_lag"])
            )
            df.drop(columns=["_vol_lag"], inplace=True, errors="ignore")
            if corr.index.nlevels == 3:
                corr = corr.droplevel(0)
            elif corr.index.nlevels == 2 and corr.index.names[0] == "Id":
                corr = corr.swaplevel(0, 1).sort_index()
            results[name] = corr

        elif ftype == "free_float_adjusted_turnover":
            turnover = df["VolumeAdj"] / df["MDV_63"].replace(0, np.nan)
            ff = (df["FREE_FLOAT_PERCENTAGE"] / 100.0).replace(0, np.nan)
            results[name] = turnover / ff

        elif ftype == "body_to_range_ratio":
            body = (df["CloseAdj"] - df["OpenAdj"]).abs()
            rng = (df["HighAdj"] - df["LowAdj"]).replace(0, np.nan)
            results[name] = body / rng

        elif ftype == "daily_price_volume_corr":
            K = int(params.get("K", 10))
            corr = df.groupby(level="Id").apply(
                lambda g: g["CloseAdj"].rolling(K, min_periods=max(2, K // 2)).corr(g["VolumeAdj"])
            )
            if corr.index.nlevels == 3:
                corr = corr.droplevel(0)
            elif corr.index.nlevels == 2 and corr.index.names[0] == "Id":
                corr = corr.swaplevel(0, 1).sort_index()
            results[name] = corr

        elif ftype == "rsv":
            K = int(params.get("K", 9))
            close = df["CloseAdj"]
            rmin = df["LowAdj"].groupby(level="Id").transform(
                lambda x: x.rolling(K, min_periods=max(1, K // 2)).min()
            )
            rmax = df["HighAdj"].groupby(level="Id").transform(
                lambda x: x.rolling(K, min_periods=max(1, K // 2)).max()
            )
            rng = rmax - rmin
            s = (close - rmin) / rng.where(rng > 0, np.nan)
            results[name] = s

        else:
            continue

    # 应用配置中的可选方向符号（默认 +1）。
    sign_map = {
        feat["name"]: float(feat.get("sign", 1.0))
        for feat in daily_features_cfg
        if feat.get("enabled", True)
    }
    for name, sign in sign_map.items():
        if name in results and sign != 1.0:
            results[name] = results[name] * sign

    if not results:
        return pd.DataFrame()

    out = pd.concat(results.values(), axis=1)
    out.columns = list(results.keys())
    out.index.names = ["Date", "Id"]
    return out


def _filter_per_id_mad(
    g: pd.Series,
    rule: str,
    mad_scale: float,
    update_freq: int,
    warmup: int,
) -> pd.Series:
    """
    单 Id 的 MAD 筛选逻辑（向量化 O(n) 实现）。
    使用 expanding median，按 update_freq 采样；MAD 在更新日用同一 med 计算，再前向填充。
    """
    g = g.sort_index()
    vals = g.values.astype(float)
    n = len(vals)
    if n == 0:
        return g.copy()

    # expanding median（向量化 O(n)）
    min_per = max(warmup + 1, 2)
    expand_med = pd.Series(vals).expanding(min_periods=min_per).median().values

    # MAD 必须在更新日用同一 med 对 hist 中心化，故仅在更新日计算
    update_indices = np.arange(warmup, n, update_freq)
    if len(update_indices) == 0:
        med_eff = np.full(n, np.nan)
        mad_eff = np.full(n, np.nan)
    else:
        med_at_updates = expand_med[update_indices]
        mad_at_updates = np.empty(len(update_indices))
        for k, idx in enumerate(update_indices):
            hist = vals[: idx + 1]
            hist = hist[~np.isnan(hist)]
            if len(hist) < 2:
                mad_at_updates[k] = np.nan
            else:
                med_k = med_at_updates[k]  # 与 expand_med 一致
                mad_k = float(np.median(np.abs(hist - med_k)))
                mad_at_updates[k] = 1e-10 if (mad_k == 0 or np.isnan(mad_k)) else mad_k

        k_arr = np.arange(n - warmup) // update_freq
        k_arr = np.minimum(k_arr, len(update_indices) - 1)
        med_eff = np.full(n, np.nan)
        mad_eff = np.full(n, np.nan)
        med_eff[warmup:] = med_at_updates[k_arr]
        mad_eff[warmup:] = mad_at_updates[k_arr]

    # 向量化应用规则
    invalid = np.isnan(med_eff) | np.isnan(mad_eff)
    if rule == "outside_both":
        keep = (vals < med_eff - mad_scale * mad_eff) | (vals > med_eff + mad_scale * mad_eff)
    elif rule == "left_tail":
        keep = vals < med_eff - mad_scale * mad_eff
    elif rule == "right_tail":
        keep = vals > med_eff + mad_scale * mad_eff
    elif rule == "center":
        keep = np.abs(vals - med_eff) <= mad_scale * mad_eff
    else:
        keep = np.ones(n, dtype=bool)
    keep = keep | invalid  # 无效位置不筛选（保留原值）

    out = g.copy()
    out.iloc[np.where(~keep)[0]] = 0.0
    return out


def _apply_mad_filter_to_panel(
    combined: pd.DataFrame,
    config: Dict[str, Any],
    n_jobs: int = -1,
) -> None:
    """
    对配置了 mad_filter 的特征列应用 MAD 筛选（原地修改 combined）。
    使用 expanding 窗口、按 update_freq_days 更新 median/MAD，warmup_days 内不筛选。
    n_jobs: 并行进程数，-1 表示 CPU 核心数。
    """
    all_features = (
        [f for f in config.get("intraday_features", []) if f.get("enabled", True)]
        + [f for f in config.get("daily_features", []) if f.get("enabled", True)]
    )
    for feat in all_features:
        mf = feat.get("mad_filter")
        if not mf or feat["name"] not in combined.columns:
            continue
        name = feat["name"]
        rule = mf.get("rule", "outside_both")
        mad_scale = float(mf.get("mad_scale", 4))
        update_freq = int(mf.get("update_freq_days", 21))
        warmup = int(mf.get("warmup_days", 21))

        idx = combined.index
        if not isinstance(idx, pd.MultiIndex):
            continue

        groups = [
            (id_val, grp)
            for id_val, grp in combined.groupby(level="Id", group_keys=False)[name]
        ]
        results = Parallel(n_jobs=n_jobs)(
            delayed(_filter_per_id_mad)(grp, rule, mad_scale, update_freq, warmup)
            for _, grp in groups
        )
        filtered = pd.concat(results).reindex(combined.index)
        combined[name] = filtered


def _extract_cumvol_at_time(intraday_df: pd.DataFrame, time_str: str) -> pd.Series:
    """提取指定时间 HH:MM 的 CumVolume，按 (Date, Id)。"""
    times = _time_only(intraday_df.index)
    mask = times == time_str
    if not mask.any():
        return pd.Series(dtype=float)
    s = (
        intraday_df.loc[mask]
        .set_index(["Date", "Id"])["CumVolume"]
        .copy()
    )
    s.index.names = ["Date", "Id"]
    return s


def build_daily_dataset(
    all_data: pd.DataFrame,
    feature_config_path: str,
    daily_data: Optional[pd.DataFrame] = None,
    include_target: bool = True,
) -> pd.DataFrame:
    """
    高层入口：从日内 all_data（及可选日频数据）构建 per-(Date, Id) DataFrame。

    日频特征仅用 T-1 数据（不含当日日频）。合并时保留日内和日频均存在的 (Date, Id)。

    Returns
    -------
    DataFrame
        Index: Date
        Columns: ['Id', <intraday_features>, <daily_features>]，若 include_target=True
        则额外包含 'target_variable'。
    """
    with tqdm(total=6, desc="Building daily dataset", unit="step") as pbar:
        config = load_feature_config(feature_config_path)
        intraday_df = build_intraday_panel(all_data)
        pbar.update(1)

        pbar.set_postfix_str("intraday features")
        intraday_features = compute_intraday_features_from_config(intraday_df, config, n_jobs=-1)
        pbar.update(1)

        pbar.set_postfix_str("target" if include_target else "features only")
        target_df = compute_target_variable(intraday_df) if include_target else None
        pbar.update(1)

        pbar.set_postfix_str("join")
        combined = (
            intraday_features.join(target_df, how="inner")
            if include_target
            else intraday_features.copy()
        )
        pbar.update(1)

        if daily_data is not None:
            pbar.set_postfix_str("daily features")
            intraday_cumvol_1530 = _extract_cumvol_at_time(intraday_df, "15:30")
            daily_features = compute_daily_features_from_config(
                daily_data, config, intraday_cumvol_1530=intraday_cumvol_1530
            )
            if not daily_features.empty:
                # 使用日内交易日历构建 next_td，确保截断时（如 strict_leakage_test）
                # 仍能正确映射 T-1 -> T，避免因 daily_data 截断丢失最后一天映射
                trading_days = sorted(pd.to_datetime(intraday_df["Date"]).unique())
                next_td = dict(zip(trading_days[:-1], trading_days[1:]))

                idx = daily_features.index
                old_dates = idx.get_level_values(0)
                new_dates = old_dates.map(next_td)
                valid = new_dates.notna()
                daily_features = daily_features.loc[valid]
                new_dates = new_dates[valid]
                daily_features.index = pd.MultiIndex.from_arrays(
                    [new_dates, idx.get_level_values(1)[valid]],
                    names=idx.names,
                )
                combined = combined.join(daily_features, how="left")
                if "volume_surge_ratio" in combined.columns:
                    cv = intraday_cumvol_1530.reindex(combined.index)
                    combined["volume_surge_ratio"] = (
                        combined["volume_surge_ratio"] * cv
                    )
            pbar.update(1)
        else:
            pbar.update(1)

        pbar.set_postfix_str("MAD filter")
        _apply_mad_filter_to_panel(combined, config)
        pbar.update(1)

    combined = combined.reset_index()
    combined = combined.sort_values(["Date", "Id"])
    combined = combined.set_index("Date")
    return combined


def summarize_feature_nans(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    汇总每个特征列的 NaN 数量和占比。

    Parameters
    ----------
    features_df : DataFrame
        build_daily_dataset 产出的日频特征 DataFrame。

    Returns
    -------
    DataFrame
        Index: 特征名
        Columns: ['non_null', 'null', 'null_ratio']。
    """
    total = len(features_df)
    summary = []
    for col in features_df.columns:
        if col == "Id":
            continue
        non_null = features_df[col].notna().sum()
        null = total - non_null
        ratio = null / total if total > 0 else np.nan
        summary.append({"feature": col, "non_null": non_null, "null": null, "null_ratio": ratio})

    summary_df = pd.DataFrame(summary).set_index("feature").sort_values("null_ratio", ascending=False)
    return summary_df


def leakage_smoke_test(
    all_data: pd.DataFrame,
    feature_config_path: str,
    daily_data: Optional[pd.DataFrame] = None,
    full_daily_df: Optional[pd.DataFrame] = None,
    max_lookback_days: int = 5,
) -> Tuple[bool, pd.Series]:
    """
    简单泄漏冒烟测试：

    1. 按 Date（交易日）将数据分为早/晚两段。
    2. 扰动晚段（打乱 ResidReturn 等）。
    3. 在有/无扰动下重新计算特征/target。
    4. 检查早段日期（距 split 足够远，滚动 lookback 不依赖晚段）的特征和 target 是否不变。

    Parameters
    ----------
    full_daily_df : DataFrame, optional
        若提供，则用其替代构建完整数据集（避免重复 build_daily_dataset）。

    Returns
    -------
    ok : bool
        安全早段内无差异则 True。
    diffs_per_column : Series
        检查窗口内每列的最大绝对差。
    """
    intraday_df = build_intraday_panel(all_data)
    dates = intraday_df["Date"].sort_values().unique()
    if len(dates) < 2 * max_lookback_days + 2:
        # 历史不足，无法进行有意义测试。
        return True, pd.Series(dtype=float)

    # 选择大致居中的分割日期。
    split_idx = len(dates) // 2
    split_date = dates[split_idx]

    # 定义“早段安全窗口”：距 split 至少 max_lookback_days，使滚动 lookback 不跨入扰动侧。
    safe_mask = intraday_df["Date"] <= dates[split_idx - max_lookback_days - 1]
    safe_dates = intraday_df.loc[safe_mask, "Date"].unique()

    with tqdm(total=4, desc="Leakage smoke test", unit="step") as pbar:
        # 完整（未扰动）数据集。
        pbar.set_postfix_str("building full dataset")
        if full_daily_df is not None:
            full_daily = full_daily_df
        else:
            full_daily = build_daily_dataset(all_data, feature_config_path, daily_data=daily_data)
        pbar.update(1)

        # 扰动未来段（dates > split_date）。
        perturbed = intraday_df.copy()
        future_mask = perturbed["Date"] > split_date

        def _shuffle_group(g: pd.DataFrame) -> pd.DataFrame:
            cols_to_shuffle = ["ResidReturn", "RawReturn", "Volume", "CumReturnResid", "CumReturnRaw", "CumVolume"]
            shuffled = g.copy()
            for c in cols_to_shuffle:
                if c in shuffled.columns:
                    shuffled[c] = shuffled[c].sample(frac=1.0, random_state=0).values
            return shuffled

        if future_mask.any():
            perturbed_future = (
                perturbed.loc[future_mask]
                .groupby("Id", group_keys=False)
                .apply(_shuffle_group)
            )
            perturbed.loc[future_mask] = perturbed_future

        pbar.set_postfix_str("perturbing")
        pbar.update(1)

        # 从扰动后的 intraday_df 重建 all_data 风格 frame
        all_data_perturbed = perturbed[
            ["Id", "ResidReturn", "RawReturn", "Volume", "CumReturnResid", "CumReturnRaw", "CumVolume"]
        ].copy()
        all_data_perturbed.index = perturbed.index

        pbar.set_postfix_str("building perturbed dataset")
        daily_perturbed = build_daily_dataset(all_data_perturbed, feature_config_path, daily_data=daily_data)
        pbar.update(1)

        pbar.set_postfix_str("comparing")
        # 仅在安全早段对比。
        if not len(safe_dates):
            pbar.update(1)
            return True, pd.Series(dtype=float)

        full_safe = full_daily.loc[full_daily.index.isin(safe_dates)].sort_index()
        pert_safe = daily_perturbed.loc[daily_perturbed.index.isin(safe_dates)].sort_index()

        # 按 (Date, Id) 精确对齐。
        key_cols = ["Id"]
        full_safe = full_safe.reset_index().set_index(["Date"] + key_cols)
        pert_safe = pert_safe.reset_index().set_index(["Date"] + key_cols)
        full_safe, pert_safe = full_safe.align(pert_safe, join="inner", axis=0)

        diffs = (full_safe - pert_safe).abs().max(axis=0)
        # 忽略 Id 列（若存在）。
        diffs = diffs.drop(labels=[c for c in diffs.index if c == "Id"], errors="ignore")
        ok = bool((diffs.fillna(0.0) == 0.0).all())
        pbar.update(1)
        return ok, diffs

