"""
并行运行 15 个新 feature 的 lab 测试，生成报告。
适配 32GB 内存 / 24 核：使用 6 个并行进程，避免内存冲突。
"""

import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# 15 个新添加的 feature 定义
NEW_FEATURE_DEFS = [
    "intraday_kurtosis.json",
    "intraday_vwap_deviation.json",
    "intraday_volume_concentration.json",
    "volatility_ratio.json",
    "volume_trend.json",
    "open_to_close_return.json",
    "down_volume_ratio.json",
    "close_position_daily_range.json",
    "upper_shadow_ratio.json",
    "up_to_down_volume_ratio.json",
    "volume_autocorrelation.json",
    "free_float_adjusted_turnover.json",
    "body_to_range_ratio.json",
    "daily_price_volume_corr.json",
    "rsv.json",
]


def run_one(def_name: str) -> tuple[str, int, str]:
    """运行单个 definition，返回 (def_name, exit_code, stderr 摘要)。"""
    lab_dir = Path(__file__).resolve().parent
    cmd = [
        sys.executable,
        str(lab_dir / "feature_lab_runner.py"),
        "--definition", def_name,
    ]
    try:
        result = subprocess.run(
            cmd,
            cwd=str(lab_dir),
            capture_output=True,
            text=True,
            timeout=600,
        )
        err = (result.stderr or "")[-500:] if result.stderr else ""
        return (def_name, result.returncode, err)
    except subprocess.TimeoutExpired:
        return (def_name, -1, "Timeout after 600s")
    except Exception as e:
        return (def_name, -1, str(e))


def main():
    max_workers = 6
    print(f"Running {len(NEW_FEATURE_DEFS)} feature lab tests with {max_workers} parallel workers...")
    print("=" * 60)

    failed = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(run_one, d): d for d in NEW_FEATURE_DEFS}
        for i, fut in enumerate(as_completed(futures), 1):
            def_name, code, err = fut.result()
            status = "OK" if code == 0 else "FAILED"
            print(f"[{i:2}/{len(NEW_FEATURE_DEFS)}] {def_name}: {status}")
            if code != 0:
                failed.append((def_name, err))
                if err:
                    print(f"       Error: {err[:200]}...")

    print("=" * 60)
    if failed:
        print(f"Failed: {len(failed)}")
        for d, e in failed:
            print(f"  - {d}")
        sys.exit(1)
    print("All 15 feature lab reports generated successfully.")
    report_dir = Path(__file__).parent / "reports"
    print(f"Reports: {report_dir}")


if __name__ == "__main__":
    main()
