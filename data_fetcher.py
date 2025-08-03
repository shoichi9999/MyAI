"""
ビットコイン価格データ取得モジュール
"""

import pandas as pd
import yfinance as yf
import requests
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import os


class BitcoinDataFetcher:
    """ビットコイン価格データの取得クラス"""
    
    def __init__(self):
        self.cache_dir = "cache"
        self._create_cache_dir()
    
    def _create_cache_dir(self):
        """キャッシュディレクトリを作成"""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
    
    def fetch_yahoo_finance(self, 
                           symbol: str = "BTC-USD",
                           start_date: str = "2020-01-01",
                           end_date: Optional[str] = None,
                           interval: str = "1d") -> pd.DataFrame:
        """
        Yahoo Financeからビットコインデータを取得
        
        Args:
            symbol: 取得するシンボル（デフォルト: BTC-USD）
            start_date: 開始日（YYYY-MM-DD形式）
            end_date: 終了日（Noneの場合は現在日時）
            interval: データ間隔（1d, 1h, 5m等）
            
        Returns:
            pd.DataFrame: OHLCV価格データ
        """
        try:
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')
            
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date, interval=interval)
            
            if data.empty:
                raise ValueError(f"データが取得できませんでした: {symbol}")
            
            # カラム名を標準化
            data.columns = [col.lower().replace(" ", "_") for col in data.columns]
            
            # タイムゾーン情報を削除
            if data.index.tz is not None:
                data.index = data.index.tz_localize(None)
            
            return data
            
        except Exception as e:
            print(f"Yahoo Financeからのデータ取得エラー: {e}")
            raise
    
    def fetch_binance_data(self,
                          symbol: str = "BTCUSDT",
                          interval: str = "1d",
                          limit: int = 1000) -> pd.DataFrame:
        """
        Binance APIからビットコインデータを取得
        
        Args:
            symbol: 取引ペア（デフォルト: BTCUSDT）
            interval: データ間隔（1d, 1h, 5m等）
            limit: 取得件数（最大1000）
            
        Returns:
            pd.DataFrame: OHLCV価格データ
        """
        try:
            url = "https://api.binance.com/api/v3/klines"
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': limit
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # DataFrameに変換
            columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
                      'close_time', 'quote_asset_volume', 'number_of_trades',
                      'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
            
            df = pd.DataFrame(data, columns=columns)
            
            # データ型変換
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # 必要なカラムのみ抽出し、数値型に変換
            ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
            df = df[ohlcv_cols].astype(float)
            
            return df
            
        except Exception as e:
            print(f"Binanceからのデータ取得エラー: {e}")
            raise
    
    def save_to_cache(self, data: pd.DataFrame, filename: str):
        """データをキャッシュファイルに保存"""
        filepath = os.path.join(self.cache_dir, f"{filename}.csv")
        data.to_csv(filepath)
        print(f"データをキャッシュに保存しました: {filepath}")
    
    def load_from_cache(self, filename: str) -> Optional[pd.DataFrame]:
        """キャッシュファイルからデータを読み込み"""
        filepath = os.path.join(self.cache_dir, f"{filename}.csv")
        
        if os.path.exists(filepath):
            try:
                data = pd.read_csv(filepath, index_col=0, parse_dates=True)
                print(f"キャッシュからデータを読み込みました: {filepath}")
                return data
            except Exception as e:
                print(f"キャッシュファイル読み込みエラー: {e}")
                return None
        return None
    
    def get_bitcoin_data(self,
                        source: str = "yahoo",
                        start_date: str = "2020-01-01",
                        end_date: Optional[str] = None,
                        use_cache: bool = True) -> pd.DataFrame:
        """
        ビットコインデータを取得（キャッシュ機能付き）
        
        Args:
            source: データソース（'yahoo' または 'binance'）
            start_date: 開始日
            end_date: 終了日
            use_cache: キャッシュを使用するか
            
        Returns:
            pd.DataFrame: ビットコイン価格データ
        """
        cache_filename = f"btc_{source}_{start_date}_{end_date}"
        
        # キャッシュから読み込み試行
        if use_cache:
            cached_data = self.load_from_cache(cache_filename)
            if cached_data is not None:
                return cached_data
        
        # データ取得
        if source.lower() == "yahoo":
            data = self.fetch_yahoo_finance(start_date=start_date, end_date=end_date)
        elif source.lower() == "binance":
            data = self.fetch_binance_data()
        else:
            raise ValueError(f"未対応のデータソース: {source}")
        
        # キャッシュに保存
        if use_cache:
            self.save_to_cache(data, cache_filename)
        
        return data


def main():
    """動作テスト"""
    fetcher = BitcoinDataFetcher()
    
    print("=== Yahoo Financeからビットコインデータ取得 ===")
    btc_data = fetcher.get_bitcoin_data(
        source="yahoo",
        start_date="2023-01-01",
        end_date="2024-01-01"
    )
    
    print(f"データ形状: {btc_data.shape}")
    print("\n最初の5行:")
    print(btc_data.head())
    
    print("\n最後の5行:")
    print(btc_data.tail())
    
    print("\n基本統計:")
    print(btc_data.describe())


if __name__ == "__main__":
    main()