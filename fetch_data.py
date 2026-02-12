"""ローカル環境でBinanceから1分足データを取得してCSV保存するスクリプト

Usage:
    # 単一シンボル
    python fetch_data.py --symbol BTCUSDT --days 7

    # 複数シンボル
    python fetch_data.py --symbol BTCUSDT ETHUSDT SOLUSDT --days 14

    # デフォルト全シンボル
    python fetch_data.py --days 7

    # 保存先指定
    python fetch_data.py --symbol BTCUSDT --days 7 --outdir data/csv
"""

import os
import sys
import time
import argparse
import requests
import pandas as pd
from datetime import datetime, timedelta

BINANCE_BASE_URL = "https://api.binance.com"

DEFAULT_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "DOTUSDT", "MATICUSDT",
    "LINKUSDT", "LTCUSDT", "ATOMUSDT", "UNIUSDT", "NEARUSDT",
]


def fetch_klines(symbol, interval="1m", limit=1000, start_time=None, end_time=None):
    url = f"{BINANCE_BASE_URL}/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    if start_time:
        params["startTime"] = start_time
    if end_time:
        params["endTime"] = end_time

    for attempt in range(4):
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            if attempt < 3:
                wait = 2 ** (attempt + 1)
                print(f"  Retry in {wait}s... ({e})")
                time.sleep(wait)
            else:
                raise


def fetch_symbol(symbol, days):
    """指定シンボルの1分足データを取得"""
    end_ms = int(datetime.utcnow().timestamp() * 1000)
    start_ms = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)

    all_klines = []
    current_start = start_ms
    total_expected = days * 24 * 60

    while current_start < end_ms:
        klines = fetch_klines(symbol, start_time=current_start, end_time=end_ms)
        if not klines:
            break
        all_klines.extend(klines)
        current_start = klines[-1][0] + 1
        pct = min(100, len(all_klines) / total_expected * 100)
        print(f"\r  {symbol}: {len(all_klines):,} candles ({pct:.0f}%)", end="", flush=True)
        time.sleep(0.1)  # レートリミット対策

    print()

    if not all_klines:
        return pd.DataFrame()

    columns = [
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore"
    ]
    df = pd.DataFrame(all_klines, columns=columns)

    numeric_cols = ["open", "high", "low", "close", "volume",
                    "quote_volume", "taker_buy_base", "taker_buy_quote"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["trades"] = df["trades"].astype(int)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")

    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser(description="Fetch Binance 1m kline data to CSV")
    parser.add_argument("--symbol", nargs="+", default=None, help="Symbol(s) to fetch")
    parser.add_argument("--days", type=int, default=7, help="Number of days (default: 7)")
    parser.add_argument("--outdir", type=str, default="data/csv", help="Output directory")
    args = parser.parse_args()

    symbols = args.symbol or DEFAULT_SYMBOLS
    os.makedirs(args.outdir, exist_ok=True)

    print(f"=== Binance 1m Data Fetcher ===")
    print(f"Symbols: {len(symbols)}")
    print(f"Period: {args.days} days")
    print(f"Output: {args.outdir}/")
    print()

    for i, symbol in enumerate(symbols):
        print(f"[{i+1}/{len(symbols)}] Fetching {symbol}...")
        try:
            df = fetch_symbol(symbol, args.days)
            if df.empty:
                print(f"  No data for {symbol}, skipping")
                continue

            filename = f"{symbol}_{args.days}d.csv"
            path = os.path.join(args.outdir, filename)
            df.to_csv(path, index=False)
            print(f"  Saved: {path} ({len(df):,} rows)")
        except Exception as e:
            print(f"  ERROR: {symbol}: {e}")

    # メタ情報を保存
    meta = {
        "symbols": symbols,
        "days": args.days,
        "fetched_at": datetime.utcnow().isoformat(),
        "interval": "1m",
    }
    import json
    meta_path = os.path.join(args.outdir, "_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nMeta saved: {meta_path}")
    print("Done!")


if __name__ == "__main__":
    main()
