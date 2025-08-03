"""
Bybit Altcoin/BTC Futures Data Fetcher
Bybitアルトコイン/BTC先物データ取得モジュール
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
import os
from typing import List, Dict, Optional


class BybitDataFetcher:
    def __init__(self, cache_dir="bybit_cache"):
        self.base_url = "https://api.bybit.com"
        self.cache_dir = cache_dir
        self.session = requests.Session()
        
        # Create cache directory
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
    
    def get_symbols(self) -> List[Dict]:
        """Get all available symbols from Bybit"""
        try:
            url = f"{self.base_url}/v5/market/instruments-info"
            params = {
                'category': 'linear',  # USDT perpetual futures
            }
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if data['retCode'] == 0:
                symbols = []
                for item in data['result']['list']:
                    symbol = item['symbol']
                    # Filter for altcoin/USDT pairs (ending with USDT)
                    if symbol.endswith('USDT') and not symbol.startswith('BTC'):
                        symbols.append({
                            'symbol': symbol,
                            'baseCoin': item['baseCoin'],
                            'quoteCoin': item['quoteCoin'],
                            'status': item['status'],
                            'contractType': item['contractType']
                        })
                
                return symbols
            else:
                print(f"Error getting symbols: {data['retMsg']}")
                return []
                
        except Exception as e:
            print(f"Error fetching symbols: {e}")
            return []
    
    def get_kline_data(self, symbol: str, interval: str = "1h", 
                      start_time: Optional[int] = None, 
                      end_time: Optional[int] = None, 
                      limit: int = 200) -> pd.DataFrame:
        """Get kline/candlestick data for a symbol"""
        try:
            url = f"{self.base_url}/v5/market/kline"
            params = {
                'category': 'linear',
                'symbol': symbol,
                'interval': interval,
                'limit': limit
            }
            
            if start_time:
                params['start'] = start_time
            if end_time:
                params['end'] = end_time
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if data['retCode'] == 0:
                klines = data['result']['list']
                
                # Convert to DataFrame
                df = pd.DataFrame(klines, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
                ])
                
                # Convert data types
                df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='ms')
                numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'turnover']
                df[numeric_cols] = df[numeric_cols].astype(float)
                
                # Sort by timestamp
                df = df.sort_values('timestamp').reset_index(drop=True)
                df.set_index('timestamp', inplace=True)
                
                return df
            else:
                print(f"Error getting kline data for {symbol}: {data['retMsg']}")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Error fetching kline data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_historical_data(self, symbol: str, days: int = 30, interval: str = "15") -> pd.DataFrame:
        """Get historical data for specified number of days"""
        
        # Check cache first
        cache_file = os.path.join(self.cache_dir, f"{symbol}_{days}d_{interval}.csv")
        if os.path.exists(cache_file):
            # Check if cache is less than 1 hour old
            cache_time = os.path.getmtime(cache_file)
            if time.time() - cache_time < 3600:  # 1 hour
                try:
                    df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                    print(f"Using cached data for {symbol}")
                    return df
                except:
                    pass
        
        # Calculate time range (without specifying end_time to get latest data)
        # end_time = int(datetime.now().timestamp() * 1000)
        # start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
        
        all_data = []
        total_needed = days * 96  # 15分間隔なので1日96レコード（24時間 × 4）
        
        # Fetch data in chunks (Bybit limits to 200 records per request)
        while len(all_data) * 200 < total_needed:
            print(f"Fetching data for {symbol}...")
            
            df_chunk = self.get_kline_data(
                symbol=symbol,
                interval=interval,
                limit=200
            )
            
            if df_chunk.empty:
                print(f"No data returned for {symbol}")
                break
            
            all_data.append(df_chunk)
            print(f"Got {len(df_chunk)} records for {symbol}")
            
            # Rate limiting
            time.sleep(0.1)
            
            # Stop if we have enough data or less than requested was returned
            total_records = sum(len(chunk) for chunk in all_data)
            if total_records >= total_needed or len(df_chunk) < 200:
                break
        
        if all_data:
            # Combine all chunks
            full_df = pd.concat(all_data).sort_index()
            # Remove duplicates
            full_df = full_df[~full_df.index.duplicated(keep='first')]
            
            # Keep only the most recent N days worth of data
            if len(full_df) > total_needed:
                full_df = full_df.tail(total_needed)
            
            # Save to cache
            try:
                full_df.to_csv(cache_file)
                print(f"Cached data for {symbol}")
            except:
                pass
            
            return full_df
        
        return pd.DataFrame()
    
    def get_multiple_symbols_data(self, symbols: List[str], days: int = 30, 
                                 interval: str = "15", max_symbols: int = 20) -> Dict[str, pd.DataFrame]:
        """Get historical data for multiple symbols"""
        
        # Limit number of symbols to avoid rate limits
        symbols = symbols[:max_symbols]
        
        results = {}
        
        for i, symbol in enumerate(symbols, 1):
            print(f"[{i}/{len(symbols)}] Fetching {symbol}...")
            
            try:
                df = self.get_historical_data(symbol, days, interval)
                if not df.empty:
                    results[symbol] = df
                else:
                    print(f"No data available for {symbol}")
            except Exception as e:
                print(f"Error fetching {symbol}: {e}")
            
            # Rate limiting
            time.sleep(0.2)
        
        return results


def calculate_short_performance(data: pd.DataFrame, leverage: float = 1.0, 
                               entry_fee: float = 0.0006, exit_fee: float = 0.0006) -> Dict:
    """
    Calculate performance of a leverage short position
    
    Args:
        data: DataFrame with OHLCV data
        leverage: Leverage multiplier
        entry_fee: Entry fee rate (0.06% for Bybit)
        exit_fee: Exit fee rate (0.06% for Bybit)
    
    Returns:
        Dictionary with performance metrics
    """
    if data.empty:
        return {}
    
    # Calculate returns (negative for short position)
    price_returns = -data['close'].pct_change().dropna()
    
    # Apply leverage
    leveraged_returns = price_returns * leverage
    
    # Account for fees (deducted from each trade)
    # Assume we're constantly rebalancing (simplified)
    fee_adjusted_returns = leveraged_returns - (entry_fee + exit_fee)
    
    # Calculate cumulative performance
    cumulative_returns = (1 + fee_adjusted_returns).cumprod()
    
    # Performance metrics
    total_return = cumulative_returns.iloc[-1] - 1
    total_return_pct = total_return * 100
    
    # Volatility (annualized)
    volatility = fee_adjusted_returns.std() * np.sqrt(24 * 365)  # Hourly to annual
    
    # Sharpe ratio (assuming 0% risk-free rate)
    if volatility > 0:
        sharpe_ratio = (fee_adjusted_returns.mean() * 24 * 365) / volatility
    else:
        sharpe_ratio = 0
    
    # Maximum drawdown
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    max_drawdown_pct = max_drawdown * 100
    
    # Win rate
    win_rate = (fee_adjusted_returns > 0).mean() * 100
    
    return {
        'total_return_pct': total_return_pct,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown_pct': max_drawdown_pct,
        'win_rate': win_rate,
        'num_periods': len(fee_adjusted_returns),
        'cumulative_returns': cumulative_returns,
        'daily_returns': fee_adjusted_returns
    }


def main():
    """Test the Bybit data fetcher"""
    print("=== Bybit Altcoin/BTC Futures Data Fetcher Test ===")
    
    fetcher = BybitDataFetcher()
    
    # Get available symbols
    print("Fetching available symbols...")
    symbols_info = fetcher.get_symbols()
    
    if symbols_info:
        print(f"Found {len(symbols_info)} altcoin/BTC pairs:")
        for symbol_info in symbols_info[:10]:  # Show first 10
            print(f"  - {symbol_info['symbol']} ({symbol_info['baseCoin']}/BTC)")
        
        # Test with a few symbols
        test_symbols = [s['symbol'] for s in symbols_info[:5]]
        print(f"\nTesting with symbols: {test_symbols}")
        
        # Get historical data
        historical_data = fetcher.get_multiple_symbols_data(test_symbols, days=7, interval="1h")
        
        print(f"\nSuccessfully fetched data for {len(historical_data)} symbols:")
        
        for symbol, df in historical_data.items():
            if not df.empty:
                # Calculate short performance
                performance = calculate_short_performance(df, leverage=1.0)
                
                print(f"\n{symbol}:")
                print(f"  Data points: {len(df)}")
                print(f"  Date range: {df.index[0]} to {df.index[-1]}")
                print(f"  Short performance (1x leverage): {performance.get('total_return_pct', 0):.2f}%")
                print(f"  Max drawdown: {performance.get('max_drawdown_pct', 0):.2f}%")
                print(f"  Sharpe ratio: {performance.get('sharpe_ratio', 0):.3f}")
    
    else:
        print("No symbols found")


if __name__ == "__main__":
    main()