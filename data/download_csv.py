"""Binanceからデータをダウンロードしてdata/csv/に保存するスクリプト

使い方:
    python -m data.download_csv --symbol BTCUSDT --days 10
    python -m data.download_csv --symbol BTCUSDT --start 2026-02-05 --end 2026-02-13
"""

import argparse
import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.fetcher import fetch_klines_bulk, _klines_to_dataframe, fetch_klines

CSV_DIR = "data/csv"


def download_to_csv(symbol: str, days: int = 10, start: str = None, end: str = None):
    """Binance APIからOHLCVデータをダウンロードしてCSV保存"""
    os.makedirs(CSV_DIR, exist_ok=True)

    if start and end:
        start_dt = datetime.strptime(start, "%Y-%m-%d")
        end_dt = datetime.strptime(end, "%Y-%m-%d") + timedelta(days=1)
        days_calc = (end_dt - start_dt).days
        label = f"{start}_to_{end}"
    else:
        end_dt = datetime.utcnow()
        start_dt = end_dt - timedelta(days=days)
        days_calc = days
        label = f"{days}d"

    print(f"ダウンロード: {symbol} 1分足")
    print(f"期間: {start_dt.strftime('%Y-%m-%d')} ~ {end_dt.strftime('%Y-%m-%d')} ({days_calc}日)")
    print(f"予想バー数: ~{days_calc * 24 * 60}")
    print()

    # fetch_klines_bulk を使ってまとめてDL
    import time
    import requests

    end_ms = int(end_dt.timestamp() * 1000)
    start_ms = int(start_dt.timestamp() * 1000)

    all_klines = []
    current_start = start_ms
    request_count = 0

    while current_start < end_ms:
        klines = fetch_klines(
            symbol=symbol,
            interval="1m",
            limit=1000,
            start_time=current_start,
            end_time=end_ms,
        )
        if not klines:
            break

        all_klines.extend(klines)
        current_start = klines[-1][0] + 1
        request_count += 1

        if request_count % 50 == 0:
            print(f"  {request_count} requests, {len(all_klines)} bars...")

        if request_count % 10 == 0:
            time.sleep(1.0)
        else:
            time.sleep(0.15)

    df = _klines_to_dataframe(all_klines)

    if df.empty:
        print("ERROR: データが取得できませんでした")
        sys.exit(1)

    # CSV保存 (timestampはISO形式で保存)
    path = os.path.join(CSV_DIR, f"{symbol}.csv")
    df.to_csv(path, index=False)

    print()
    print(f"完了: {len(df)} bars")
    print(f"期間: {df['timestamp'].iloc[0]} ~ {df['timestamp'].iloc[-1]}")
    print(f"保存: {path}")


def main():
    parser = argparse.ArgumentParser(description="Binance OHLCV データダウンロード")
    parser.add_argument("--symbol", default="BTCUSDT", help="通貨ペア")
    parser.add_argument("--days", type=int, default=10, help="直近N日分")
    parser.add_argument("--start", default=None, help="開始日 (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="終了日 (YYYY-MM-DD)")
    args = parser.parse_args()
    download_to_csv(args.symbol, args.days, args.start, args.end)


if __name__ == "__main__":
    main()
