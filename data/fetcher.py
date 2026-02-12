"""Binance APIからOHLCVデータを取得・キャッシュするモジュール

1分足固定で年単位の長期データにも対応する。
"""

import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
import glob as globmod
from config.settings import BINANCE_BASE_URL, BINANCE_KLINES_ENDPOINT, DATA_DIR

CSV_DIR = "data/csv"


def ensure_cache_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def fetch_klines(symbol: str, interval: str = "1m", limit: int = 1000,
                 start_time: int = None, end_time: int = None) -> list:
    """Binance APIから指定シンボルのローソク足データを取得する。

    Args:
        symbol: 取引ペア (例: "BTCUSDT")
        interval: 時間足 (デフォルト "1m")
        limit: 取得本数 (最大1000)
        start_time: 開始時刻 (ミリ秒UNIXタイムスタンプ)
        end_time: 終了時刻 (ミリ秒UNIXタイムスタンプ)

    Returns:
        list of kline data
    """
    url = f"{BINANCE_BASE_URL}{BINANCE_KLINES_ENDPOINT}"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
    }
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
                time.sleep(wait)
            else:
                raise RuntimeError(f"Binance API error after 4 retries: {e}")


def fetch_klines_bulk(symbol: str, interval: str = "1m",
                      days: int = 365) -> pd.DataFrame:
    """指定日数分の1分足データをまとめて取得する。

    1分足 × 1年 = 525,600本 → 526回のAPIリクエスト。

    Args:
        symbol: 取引ペア
        interval: 時間足 (デフォルト "1m")
        days: 取得日数 (365=1年, 730=2年, etc.)

    Returns:
        pd.DataFrame with columns:
            timestamp, open, high, low, close, volume, ...
    """
    ensure_cache_dir()

    end_ms = int(datetime.utcnow().timestamp() * 1000)
    start_ms = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)

    all_klines = []
    current_start = start_ms
    request_count = 0

    while current_start < end_ms:
        klines = fetch_klines(
            symbol=symbol,
            interval=interval,
            limit=1000,
            start_time=current_start,
            end_time=end_ms,
        )
        if not klines:
            break

        all_klines.extend(klines)
        # 次のリクエストの開始を最後のローソク足の次に
        current_start = klines[-1][0] + 1
        request_count += 1

        # レートリミット対策 (長期取得時は多めに待つ)
        if request_count % 10 == 0:
            time.sleep(1.0)
        else:
            time.sleep(0.15)

    df = _klines_to_dataframe(all_klines)
    return df


def _klines_to_dataframe(klines: list) -> pd.DataFrame:
    """APIレスポンスをDataFrameに変換"""
    if not klines:
        return pd.DataFrame()

    columns = [
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore"
    ]
    df = pd.DataFrame(klines, columns=columns)

    # 型変換
    numeric_cols = ["open", "high", "low", "close", "volume",
                    "quote_volume", "taker_buy_base", "taker_buy_quote"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["trades"] = df["trades"].astype(int)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")

    # 重複排除
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    return df


def save_cache(df: pd.DataFrame, symbol: str, days: int, interval: str = "1m"):
    """データをキャッシュとしてParquet形式で保存"""
    ensure_cache_dir()
    path = os.path.join(DATA_DIR, f"{symbol}_{days}d_{interval}.parquet")
    df.to_parquet(path, index=False)
    return path


def load_cache(symbol: str, days: int, interval: str = "1m",
               max_age_hours: int = 1) -> pd.DataFrame:
    """キャッシュがあれば読み込む。古ければNoneを返す。"""
    path = os.path.join(DATA_DIR, f"{symbol}_{days}d_{interval}.parquet")

    # 旧形式のキャッシュもフォールバック
    if not os.path.exists(path):
        old_path = os.path.join(DATA_DIR, f"{symbol}_{days}d.parquet")
        if os.path.exists(old_path):
            path = old_path
        else:
            return None

    mtime = os.path.getmtime(path)
    age_hours = (time.time() - mtime) / 3600
    if age_hours > max_age_hours:
        return None

    return pd.read_parquet(path)


def load_csv(symbol: str, days: int = None) -> pd.DataFrame:
    """CSVファイルからデータを読み込む。

    data/csv/ ディレクトリから該当シンボルのCSVを探す。
    ファイル名パターン: {SYMBOL}_{days}d.csv or {SYMBOL}.csv
    """
    candidates = []
    if days:
        candidates.append(os.path.join(CSV_DIR, f"{symbol}_{days}d.csv"))
    # daysなしでもマッチするファイルを探す
    pattern = os.path.join(CSV_DIR, f"{symbol}_*.csv")
    candidates.extend(sorted(globmod.glob(pattern), reverse=True))
    candidates.append(os.path.join(CSV_DIR, f"{symbol}.csv"))

    for path in candidates:
        if os.path.exists(path):
            df = pd.read_csv(path)
            # timestamp列をdatetimeに変換
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            if "close_time" in df.columns:
                df["close_time"] = pd.to_datetime(df["close_time"])
            return df

    return pd.DataFrame()


def list_csv_symbols() -> list:
    """data/csv/ にあるCSVファイルからシンボル一覧を取得"""
    if not os.path.exists(CSV_DIR):
        return []
    files = globmod.glob(os.path.join(CSV_DIR, "*.csv"))
    symbols = set()
    for f in files:
        name = os.path.basename(f).replace(".csv", "")
        # BTCUSDT_7d → BTCUSDT
        symbol = name.split("_")[0]
        if symbol != "_meta":
            symbols.add(symbol)
    return sorted(symbols)


def get_data(symbol: str, days: int = 365, interval: str = "1m",
             use_cache: bool = True) -> pd.DataFrame:
    """データ取得のメインエントリーポイント。

    Args:
        symbol: 取引ペア
        days: 取得日数 (365=1年, 730=2年, 1095=3年)
        interval: 時間足 (デフォルト "1m")
        use_cache: キャッシュを使うか

    優先順位:
    1. CSVファイル (data/csv/)
    2. リモートストレージからダウンロード (manifest.json)
    3. Parquetキャッシュ (data/cache/)
    4. Binance APIから取得
    """
    # 1. CSVファイルをチェック
    csv_data = load_csv(symbol, days)
    if not csv_data.empty:
        return csv_data

    # 2. リモートからダウンロード試行
    try:
        from data.remote import sync_data, list_remote_symbols
        if symbol in list_remote_symbols():
            result = sync_data(symbols=[symbol])
            if result["downloaded"]:
                csv_data = load_csv(symbol, days)
                if not csv_data.empty:
                    return csv_data
    except Exception:
        pass  # マニフェストがなければスキップ

    # 3. キャッシュをチェック
    if use_cache:
        cached = load_cache(symbol, days, interval)
        if cached is not None:
            return cached

    # 4. APIから取得
    df = fetch_klines_bulk(symbol, interval=interval, days=days)
    if not df.empty:
        save_cache(df, symbol, days, interval)
    return df


def get_available_symbols() -> list:
    """Binanceで取引可能なUSDTペアのリストを取得"""
    url = f"{BINANCE_BASE_URL}/api/v3/exchangeInfo"
    for attempt in range(4):
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            symbols = [
                s["symbol"] for s in data["symbols"]
                if s["quoteAsset"] == "USDT"
                and s["status"] == "TRADING"
                and s["isSpotTradingAllowed"]
            ]
            return sorted(symbols)
        except requests.RequestException:
            if attempt < 3:
                time.sleep(2 ** (attempt + 1))
            else:
                raise
