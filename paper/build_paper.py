from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


PAPER_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = PAPER_DIR / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from download_data import default_data_paths, prepare_sp500_data
from make_figures import make_figures
from make_tables import make_latex_tables
from run_sp500_comparison import run_comparison


def _compile_latex() -> None:
    latexmk = shutil.which("latexmk")
    if latexmk is None:
        print("latexmk was not found; skipping LaTeX compilation.")
        return
    build_dir = PAPER_DIR / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            latexmk,
            "-pdf",
            "-interaction=nonstopmode",
            f"-outdir={build_dir}",
            str(PAPER_DIR / "main.tex"),
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the reproducible S&P 500 comparison paper.")
    parser.add_argument("--ticker", default="^GSPC")
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()

    for path in (
        PAPER_DIR / "data" / "raw",
        PAPER_DIR / "data" / "processed",
        PAPER_DIR / "figures",
        PAPER_DIR / "tables",
        PAPER_DIR / "outputs",
        PAPER_DIR / "build",
    ):
        path.mkdir(parents=True, exist_ok=True)

    paths = default_data_paths(ticker=args.ticker, start=args.start)
    prepare_sp500_data(
        ticker=args.ticker,
        start=args.start,
        end=args.end,
        interval=args.interval,
        force_download=args.force_download,
        skip_download=args.skip_download,
        raw_csv=paths.raw_csv,
        processed_csv=paths.processed_csv,
    )
    run_comparison(processed_csv=paths.processed_csv)
    make_latex_tables()
    make_figures()

    if args.no_compile:
        print("Skipping LaTeX compilation because --no-compile was passed.")
    else:
        _compile_latex()

    print("Paper pipeline completed.")


if __name__ == "__main__":
    main()
