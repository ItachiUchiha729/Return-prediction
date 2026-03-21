"""
Feature Lab Generator
=====================
根据 JSON 中定义的参数网格展开，并利用项目现有特征工程设施批量计算所有特征变体。
"""

import copy
import itertools
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

_SRC = str(Path(__file__).resolve().parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from feature_engineering import (
    build_intraday_panel,
    compute_intraday_features_from_config,
    compute_daily_features_from_config,
    compute_target_variable,
    _extract_cumvol_at_time,
)


# ---------------------------------------------------------------------------
# JSON 辅助
# ---------------------------------------------------------------------------

def load_definition(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _deep_set(d: dict, dotted_key: str, value) -> None:
    """用点号键设置嵌套字典值（会修改 *d*）。"""
    keys = dotted_key.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def _merge_params(base: dict, overrides: dict) -> dict:
    """深拷贝 *base* 并应用 *overrides*（支持点号键）。"""
    result = copy.deepcopy(base)
    for key, val in overrides.items():
        if "." in key:
            _deep_set(result, key, val)
        else:
            result[key] = val
    return result


# ---------------------------------------------------------------------------
# 网格展开
# ---------------------------------------------------------------------------

def expand_grid(definition: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    返回参数覆盖 dict 的列表。

    支持两种模式（互斥）：
      * ``param_grid``  – 参数列表的笛卡尔积。
      * ``param_combos`` – 显式参数 dict 列表。

    若均不存在，返回 ``[{}]``（单变体，使用 base_params）。
    """
    if "param_combos" in definition:
        return definition["param_combos"]

    grid = definition.get("param_grid", {})
    if not grid:
        return [{}]

    keys = sorted(grid.keys())
    values = [grid[k] if isinstance(grid[k], list) else [grid[k]] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


# ---------------------------------------------------------------------------
# 命名辅助
# ---------------------------------------------------------------------------

def variant_label(base_name: str, overrides: dict) -> str:
    """人类可读标签（用于图表/表格）。"""
    if not overrides:
        return base_name
    parts = [f"{k}={v}" for k, v in sorted(overrides.items())]
    return f"{base_name} ({', '.join(parts)})"


def variant_col_name(base_name: str, overrides: dict) -> str:
    """Column-safe name used as the pandas column key."""
    if not overrides:
        return base_name
    parts = [f"{k}={v}" for k, v in sorted(overrides.items())]
    return f"{base_name}__{'__'.join(parts)}"


# ---------------------------------------------------------------------------
# 批量特征计算
# ---------------------------------------------------------------------------

def _build_feat_defs(
    definition: Dict[str, Any],
    overrides_list: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """为计算引擎构建合成特征定义 dict。"""
    feat_defs: List[Dict[str, Any]] = []
    for ov in overrides_list:
        col = variant_col_name(definition["name"], ov)
        fd: Dict[str, Any] = {
            "name": col,
            "enabled": True,
            "type": definition["type"],
            "params": _merge_params(definition.get("base_params", {}), ov),
        }
        if "input" in definition:
            fd["input"] = definition["input"]
        if "inputs" in definition:
            fd["inputs"] = copy.deepcopy(definition["inputs"])
        if "sign" in definition:
            fd["sign"] = definition["sign"]
        feat_defs.append(fd)
    return feat_defs


def generate_variants(
    definition: Dict[str, Any],
    intraday_df: pd.DataFrame,
    daily_data: Optional[pd.DataFrame] = None,
    intraday_cumvol_1530: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    在一次批量调用中计算所有参数网格变体。

    返回以 ``(Date, Id)`` 为索引的 DataFrame，每变体一列。
    """
    overrides_list = expand_grid(definition)
    feat_defs = _build_feat_defs(definition, overrides_list)
    category = definition.get("category", "intraday")

    if category == "intraday":
        cfg = {"intraday_features": feat_defs, "daily_features": []}
        return compute_intraday_features_from_config(intraday_df, cfg)
    elif category == "daily":
        if daily_data is None:
            raise ValueError("daily_data is required for daily feature variants")
        cfg = {"intraday_features": [], "daily_features": feat_defs}
        return compute_daily_features_from_config(
            daily_data, cfg, intraday_cumvol_1530=intraday_cumvol_1530,
        )
    else:
        raise ValueError(f"Unknown category: {category}")
