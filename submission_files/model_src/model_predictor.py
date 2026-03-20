"""
Mode 2: Teammate implements run_mode2_predictions().

Input:  features_dir (CSV from Mode 1), output_dir, model_dir, start_date, end_date
Output: predictions_YYYYMMDD.csv with Date, Time, Id, Pred (Time=15:30)
"""

from pathlib import Path
from typing import Any

import pandas as pd


def run_mode2_predictions(
    features_dir: Path,
    output_dir: Path,
    model_dir: Path,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> None:
    """
    Load features, apply model, save predictions.
    Teammate implements this function.
    """
    raise NotImplementedError("Teammate: implement run_mode2_predictions in model_predictor.py")
