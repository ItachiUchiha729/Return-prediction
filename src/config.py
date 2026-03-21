"""
config.py
=========
Project-wide constants and default hyper-parameters.
"""
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────
ROOT_DIR       = Path(__file__).resolve().parent.parent   # …/MTH9899_Project_Final
FEATURE_DATA_DIR       = ROOT_DIR / "feature_data"
RAW_DATA_DIR   = ROOT_DIR / "raw_data"
