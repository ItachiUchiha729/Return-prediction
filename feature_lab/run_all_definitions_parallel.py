"""
清空缓存和报告，并行运行 feature_lab/definitions 下所有 feature，生成新报告。
使用 5 个并行进程。
"""

import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def clear_cache_and_reports():
    """清空缓存目录，删除所有 HTML 报告。"""
    lab_dir = Path(__file__).resolve().parent
    cache_dir = lab_dir / "cache"
    report_dir = lab_dir / "reports"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        print("Cache cleared.")
    if report_dir.exists():
        for f in report_dir.glob("*.html"):
            f.unlink()
        print(f"Deleted HTML reports in {report_dir}.")
    report_dir.mkdir(parents=True, exist_ok=True)


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
            timeout=900,
        )
        err = (result.stderr or "")[-500:] if result.stderr else ""
        return (def_name, result.returncode, err)
    except subprocess.TimeoutExpired:
        return (def_name, -1, "Timeout after 900s")
    except Exception as e:
        return (def_name, -1, str(e))


def main():
    lab_dir = Path(__file__).resolve().parent
    definitions_dir = lab_dir / "definitions"
    def_files = sorted(f.name for f in definitions_dir.glob("*.json"))
    if not def_files:
        print("No definition files found.")
        sys.exit(1)

    print("Clearing cache and reports...")
    clear_cache_and_reports()

    max_workers = 5
    print(f"\nRunning {len(def_files)} feature lab tests with {max_workers} parallel workers...")
    print("=" * 60)

    failed = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(run_one, d): d for d in def_files}
        for i, fut in enumerate(as_completed(futures), 1):
            def_name, code, err = fut.result()
            status = "OK" if code == 0 else "FAILED"
            print(f"[{i:2}/{len(def_files)}] {def_name}: {status}")
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
    print(f"All {len(def_files)} feature lab reports generated successfully.")
    print(f"Reports: {lab_dir / 'reports'}")


if __name__ == "__main__":
    main()
