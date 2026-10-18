"""Download once, retain a local adjusted-price snapshot, and verify its hash."""

from pathlib import Path
from datetime import datetime, timezone
from importlib.metadata import version
import argparse
import hashlib
import json
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent
TICKERS = ["MSFT", "JPM", "TLT", "GLD", "BTC-USD", "GOOG", "XLE", "SPY"]
START = "2020-01-01"
END = "2025-01-01"  # exclusive; preserves the original historical date range
PRICE_PATH = ROOT / "data" / "adjusted-close.csv"
MANIFEST_PATH = ROOT / "results" / "data-manifest.json"


def fetch_snapshot(refresh=False):
    if PRICE_PATH.exists() and not refresh:
        return load_snapshot()
    yf.set_tz_cache_location(str(ROOT / ".yf-cache"))
    raw = yf.download(
        TICKERS, start=START, end=END, auto_adjust=True, back_adjust=False,
        actions=False, repair=False, threads=False, progress=False, ignore_tz=True,
        group_by="column", multi_level_index=True, timeout=30,
    )
    if raw.empty or "Close" not in raw.columns.get_level_values(0):
        raise RuntimeError("Download did not return adjusted Close prices.")
    prices = raw["Close"].reindex(columns=TICKERS).sort_index()
    missing = prices.columns[prices.isna().all()].tolist()
    if missing:
        raise RuntimeError(f"Download failed for {missing}; no partial run permitted.")
    prices.index.name = "Date"
    PRICE_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(PRICE_PATH, float_format="%.12g")
    manifest = {
        "source": "Yahoo Finance via yfinance",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "requested_start_inclusive": START,
        "requested_end_exclusive": END,
        "auto_adjust": True,
        "repair": False,
        "ignore_tz": True,
        "yfinance_version": version("yfinance"),
        "tickers": TICKERS,
        "price_file": "data/adjusted-close.csv",
        "sha256": hashlib.sha256(PRICE_PATH.read_bytes()).hexdigest(),
        "combined_calendar_rows": len(prices),
        "observations_by_ticker": prices.notna().sum().astype(int).to_dict(),
        "note": "Fresh retrieval of historical data, not the unarchived 2025 download. "
                "Adjusted-price daily timestamps are an approximate common observation clock.",
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    return load_snapshot()


def load_snapshot():
    if not PRICE_PATH.exists() or not MANIFEST_PATH.exists():
        raise FileNotFoundError("Run fetch_data.py once before offline analysis.")
    manifest = json.loads(MANIFEST_PATH.read_text())
    actual = hashlib.sha256(PRICE_PATH.read_bytes()).hexdigest()
    if actual != manifest["sha256"]:
        raise RuntimeError("Local price snapshot differs from its recorded hash.")
    return pd.read_csv(PRICE_PATH, index_col="Date", parse_dates=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true",
                        help="Explicitly replace the local price snapshot.")
    args = parser.parse_args()
    prices = fetch_snapshot(refresh=args.refresh)
    print(f"Verified snapshot: {len(prices)} rows, {len(prices.columns)} assets")
    print(PRICE_PATH)
