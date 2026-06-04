from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


PAPER_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PAPER_DIR / "data" / "raw"
PROCESSED_DIR = PAPER_DIR / "data" / "processed"


@dataclass(frozen=True)
class DataPaths:
    raw_csv: Path
    processed_csv: Path


def default_data_paths(ticker: str = "^GSPC", start: str = "2010-01-01") -> DataPaths:
    if ticker == "^GSPC" and start == "2010-01-01":
        return DataPaths(
            raw_csv=RAW_DIR / "sp500_2010_present.csv",
            processed_csv=PROCESSED_DIR / "sp500_log_prices.csv",
        )
    safe_ticker = ticker.replace("^", "").replace("/", "_").replace("\\", "_").lower()
    safe_start = start.replace("-", "")
    return DataPaths(
        raw_csv=RAW_DIR / f"{safe_ticker}_{safe_start}_present.csv",
        processed_csv=PROCESSED_DIR / f"{safe_ticker}_{safe_start}_log_prices.csv",
    )


def _flatten_columns(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    if isinstance(data.columns, pd.MultiIndex):
        flattened = []
        for col in data.columns:
            parts = [str(part) for part in col if str(part) and str(part) != "nan"]
            if len(parts) > 1 and parts[-1].startswith("^"):
                parts = parts[:-1]
            flattened.append(" ".join(parts))
        data.columns = flattened
    return data


def _normalize_raw_data(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    data = _flatten_columns(data)
    if "Date" not in data.columns:
        data = data.reset_index()
    if "Datetime" in data.columns and "Date" not in data.columns:
        data = data.rename(columns={"Datetime": "Date"})
    if "index" in data.columns and "Date" not in data.columns:
        data = data.rename(columns={"index": "Date"})
    if "Date" not in data.columns:
        raise ValueError("Downloaded data do not contain a Date column.")
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data = data.dropna(subset=["Date"]).sort_values("Date")
    data = data.loc[:, ~data.columns.duplicated()]
    data["ticker"] = ticker
    return data


def _price_column(columns: list[str]) -> str:
    for candidate in ("Adj Close", "Close"):
        if candidate in columns:
            return candidate
    normalized = {col.lower().replace(" ", "").replace("_", ""): col for col in columns}
    for candidate in ("adjclose", "close"):
        if candidate in normalized:
            return normalized[candidate]
    raise ValueError("Expected an adjusted close or close column in the S&P 500 data.")


def _download_with_yfinance(ticker: str, start: str, end: str | None, interval: str) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError(
            "yfinance is required to download data. Install with "
            'python -m pip install -e ".[finance]".'
        ) from exc

    data = yf.download(
        tickers=ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if data.empty:
        raise RuntimeError(f"yfinance returned no rows for ticker={ticker!r}, start={start!r}.")
    return _normalize_raw_data(data, ticker)


def prepare_sp500_data(
    *,
    ticker: str = "^GSPC",
    start: str = "2010-01-01",
    end: str | None = None,
    interval: str = "1d",
    use_log: bool = True,
    force_download: bool = False,
    skip_download: bool = False,
    raw_csv: Path | None = None,
    processed_csv: Path | None = None,
) -> pd.DataFrame:
    paths = default_data_paths(ticker=ticker, start=start)
    raw_csv = Path(raw_csv) if raw_csv is not None else paths.raw_csv
    processed_csv = Path(processed_csv) if processed_csv is not None else paths.processed_csv
    raw_csv.parent.mkdir(parents=True, exist_ok=True)
    processed_csv.parent.mkdir(parents=True, exist_ok=True)

    if raw_csv.exists() and not force_download:
        raw = _normalize_raw_data(pd.read_csv(raw_csv), ticker)
    else:
        if skip_download:
            raise FileNotFoundError(
                f"No cached raw data found at {raw_csv}. Re-run without --skip-download "
                "when internet access is available."
            )
        try:
            raw = _download_with_yfinance(ticker=ticker, start=start, end=end, interval=interval)
        except Exception as exc:
            if raw_csv.exists():
                raw = _normalize_raw_data(pd.read_csv(raw_csv), ticker)
            else:
                raise RuntimeError(
                    "Could not download S&P 500 data and no cached raw CSV exists. "
                    f"Expected cache path: {raw_csv}"
                ) from exc
        raw.to_csv(raw_csv, index=False)

    price_col = _price_column(list(raw.columns))
    processed = raw[["Date", price_col]].copy()
    processed = processed.rename(columns={"Date": "date", price_col: "price"})
    processed["price"] = pd.to_numeric(processed["price"], errors="coerce")
    processed = processed.dropna(subset=["date", "price"])
    processed = processed[processed["price"] > 0.0]
    processed["ticker"] = ticker
    processed["source_column"] = price_col
    processed["log_price"] = np.log(processed["price"]) if use_log else processed["price"]
    processed = processed.sort_values("date").reset_index(drop=True)
    if processed.empty:
        raise RuntimeError("Processed price data are empty after cleaning.")
    processed.to_csv(processed_csv, index=False)
    return processed


def main() -> None:
    parser = argparse.ArgumentParser(description="Download or load cached S&P 500 data.")
    parser.add_argument("--ticker", default="^GSPC")
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()
    data = prepare_sp500_data(
        ticker=args.ticker,
        start=args.start,
        end=args.end,
        interval=args.interval,
        force_download=args.force_download,
        skip_download=args.skip_download,
    )
    print(f"Saved processed data with {len(data)} rows.")


if __name__ == "__main__":
    main()
