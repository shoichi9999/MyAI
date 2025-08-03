"""
Correlation-based Trading Strategies
相関ベースのトレード戦略
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Optional


def multi_asset_correlation_strategy(data: pd.DataFrame, 
                                   correlation_window: int = 30,
                                   threshold: float = 0.3,
                                   rebalance_frequency: int = 5) -> pd.DataFrame:
    """
    Multiple Asset Correlation Strategy
    複数資産との相関を利用した戦略
    
    Args:
        data: Price data
        correlation_window: Correlation calculation window
        threshold: Correlation threshold for signals
        rebalance_frequency: Rebalancing frequency in days
    """
    data = data.copy()
    
    try:
        # Download additional asset data for correlation
        spy_data = yf.download('SPY', start=data.index[0], end=data.index[-1], progress=False)['Close']
        gold_data = yf.download('GLD', start=data.index[0], end=data.index[-1], progress=False)['Close']
        bond_data = yf.download('TLT', start=data.index[0], end=data.index[-1], progress=False)['Close']
        
        # Align data by common dates
        common_dates = data.index.intersection(spy_data.index).intersection(gold_data.index).intersection(bond_data.index)
        data = data.loc[common_dates]
        spy_data = spy_data.loc[common_dates]
        gold_data = gold_data.loc[common_dates]
        bond_data = bond_data.loc[common_dates]
        
        # Calculate returns
        btc_returns = data['close'].pct_change()
        spy_returns = spy_data.pct_change()
        gold_returns = gold_data.pct_change()
        bond_returns = bond_data.pct_change()
        
        # Rolling correlations
        data['corr_spy'] = btc_returns.rolling(window=correlation_window).corr(spy_returns)
        data['corr_gold'] = btc_returns.rolling(window=correlation_window).corr(gold_returns)
        data['corr_bonds'] = btc_returns.rolling(window=correlation_window).corr(bond_returns)
        
        # Correlation strength (average absolute correlation)
        data['corr_strength'] = (data['corr_spy'].abs() + data['corr_gold'].abs() + data['corr_bonds'].abs()) / 3
        
        # Signal generation
        data['signal'] = 0
        
        # Buy when correlation with traditional assets is low (Bitcoin is decorrelated)
        buy_condition = (
            (data['corr_strength'] < threshold) &
            (data.index.to_series().diff().dt.days >= rebalance_frequency)
        )
        
        # Sell when correlation becomes high (Bitcoin follows traditional markets)
        sell_condition = data['corr_strength'] > (threshold + 0.2)
        
        data.loc[buy_condition, 'signal'] = 1
        data.loc[sell_condition, 'signal'] = -1
        
        # Forward fill signals with rebalancing frequency
        data['signal'] = data['signal'].replace(0, np.nan).fillna(method='ffill').fillna(0)
        
    except Exception as e:
        print(f"Warning: Could not fetch external data for correlation strategy: {e}")
        # Fallback to simple momentum when external data unavailable
        data['signal'] = np.where(data['close'].pct_change(10) > 0, 1, -1)
    
    return data


def volatility_correlation_strategy(data: pd.DataFrame,
                                   vol_window: int = 20,
                                   correlation_window: int = 30,
                                   vol_threshold: float = 0.5) -> pd.DataFrame:
    """
    Volatility Correlation Strategy
    ボラティリティ相関戦略
    
    Buy when Bitcoin volatility correlates negatively with price (fear buying)
    Sell when volatility correlates positively with price (momentum selling)
    """
    data = data.copy()
    
    # Calculate volatility
    data['returns'] = data['close'].pct_change()
    data['volatility'] = data['returns'].rolling(window=vol_window).std()
    
    # Calculate correlation between price and volatility
    price_change = data['close'].pct_change(vol_window)
    vol_change = data['volatility'].pct_change(vol_window)
    
    data['price_vol_corr'] = price_change.rolling(window=correlation_window).corr(vol_change)
    
    # Signal generation
    data['signal'] = 0
    
    # Buy when price-volatility correlation is negative (contrarian signal)
    buy_condition = data['price_vol_corr'] < -vol_threshold
    
    # Sell when price-volatility correlation is positive (momentum signal)
    sell_condition = data['price_vol_corr'] > vol_threshold
    
    data.loc[buy_condition, 'signal'] = 1
    data.loc[sell_condition, 'signal'] = -1
    
    # Clean signals - only change when crossing threshold
    data['signal'] = data['signal'].replace(0, np.nan).fillna(method='ffill').fillna(0)
    
    return data


def volume_price_correlation_strategy(data: pd.DataFrame,
                                     correlation_window: int = 20,
                                     volume_threshold: float = 0.3) -> pd.DataFrame:
    """
    Volume-Price Correlation Strategy
    出来高-価格相関戦略
    
    Uses correlation between volume and price changes as signal
    """
    data = data.copy()
    
    # Calculate returns and volume changes
    data['price_change'] = data['close'].pct_change()
    data['volume_change'] = data['volume'].pct_change()
    
    # Rolling correlation between price and volume changes
    data['vol_price_corr'] = data['price_change'].rolling(window=correlation_window).corr(data['volume_change'])
    
    # Signal generation based on volume-price correlation
    data['signal'] = 0
    
    # Buy when volume-price correlation is strong positive (confirmed moves)
    buy_condition = (
        (data['vol_price_corr'] > volume_threshold) &
        (data['price_change'] > 0)
    )
    
    # Sell when volume-price correlation breaks down
    sell_condition = (
        (data['vol_price_corr'] < -volume_threshold) |
        ((data['vol_price_corr'] < volume_threshold) & (data['price_change'] < 0))
    )
    
    data.loc[buy_condition, 'signal'] = 1
    data.loc[sell_condition, 'signal'] = -1
    
    # Clean signals
    data['signal'] = data['signal'].replace(0, np.nan).fillna(method='ffill').fillna(0)
    
    return data


def cross_timeframe_correlation_strategy(data: pd.DataFrame,
                                       short_period: int = 5,
                                       long_period: int = 20,
                                       correlation_window: int = 30,
                                       threshold: float = 0.4) -> pd.DataFrame:
    """
    Cross-Timeframe Correlation Strategy
    クロスタイムフレーム相関戦略
    
    Analyzes correlation between different timeframe returns
    """
    data = data.copy()
    
    # Calculate returns for different timeframes
    data['returns_short'] = data['close'].pct_change(short_period)
    data['returns_long'] = data['close'].pct_change(long_period)
    
    # Rolling correlation between short and long term returns
    data['timeframe_corr'] = data['returns_short'].rolling(window=correlation_window).corr(data['returns_long'])
    
    # Signal generation
    data['signal'] = 0
    
    # Buy when short and long term returns are positively correlated (trend continuation)
    buy_condition = (
        (data['timeframe_corr'] > threshold) &
        (data['returns_short'] > 0) &
        (data['returns_long'] > 0)
    )
    
    # Sell when correlation breaks down or turns negative
    sell_condition = (
        (data['timeframe_corr'] < -threshold) |
        ((data['returns_short'] < 0) & (data['returns_long'] < 0))
    )
    
    data.loc[buy_condition, 'signal'] = 1
    data.loc[sell_condition, 'signal'] = -1
    
    # Clean signals
    data['signal'] = data['signal'].replace(0, np.nan).fillna(method='ffill').fillna(0)
    
    return data


def sentiment_correlation_strategy(data: pd.DataFrame,
                                 rsi_window: int = 14,
                                 macd_fast: int = 12,
                                 macd_slow: int = 26,
                                 correlation_window: int = 20,
                                 threshold: float = 0.6) -> pd.DataFrame:
    """
    Sentiment Correlation Strategy
    センチメント相関戦略
    
    Uses correlation between different technical indicators as sentiment proxy
    """
    data = data.copy()
    
    # Calculate RSI
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_window).mean()
    rs = gain / loss
    data['rsi'] = 100 - (100 / (1 + rs))
    
    # Calculate MACD
    ema_fast = data['close'].ewm(span=macd_fast).mean()
    ema_slow = data['close'].ewm(span=macd_slow).mean()
    data['macd'] = ema_fast - ema_slow
    
    # Calculate correlation between RSI and MACD (sentiment alignment)
    data['sentiment_corr'] = data['rsi'].rolling(window=correlation_window).corr(data['macd'])
    
    # Signal generation
    data['signal'] = 0
    
    # Buy when indicators are highly correlated and bullish
    buy_condition = (
        (data['sentiment_corr'] > threshold) &
        (data['rsi'] > 50) &
        (data['macd'] > 0)
    )
    
    # Sell when indicators diverge or turn bearish
    sell_condition = (
        (data['sentiment_corr'] < -threshold) |
        ((data['rsi'] < 30) & (data['macd'] < 0))
    )
    
    data.loc[buy_condition, 'signal'] = 1
    data.loc[sell_condition, 'signal'] = -1
    
    # Clean signals
    data['signal'] = data['signal'].replace(0, np.nan).fillna(method='ffill').fillna(0)
    
    return data


def adaptive_correlation_strategy(data: pd.DataFrame,
                                base_window: int = 20,
                                adaptive_factor: float = 0.5,
                                correlation_threshold: float = 0.4) -> pd.DataFrame:
    """
    Adaptive Correlation Strategy
    適応的相関戦略
    
    Adapts correlation window based on market volatility
    """
    data = data.copy()
    
    # Calculate base volatility
    data['returns'] = data['close'].pct_change()
    data['volatility'] = data['returns'].rolling(window=base_window).std()
    
    # Adaptive window calculation
    vol_percentile = data['volatility'].rolling(window=100).rank(pct=True)
    data['adaptive_window'] = (base_window * (1 + adaptive_factor * vol_percentile)).astype(int)
    
    # Calculate adaptive correlation with own lagged returns
    data['price_lag'] = data['close'].shift(5)
    data['adaptive_corr'] = 0.0
    
    for i in range(base_window, len(data)):
        window = int(data['adaptive_window'].iloc[i])
        if window > 5:
            start_idx = max(0, i - window)
            corr_data = data.iloc[start_idx:i+1]
            if len(corr_data) > 5:
                corr_val = corr_data['close'].corr(corr_data['price_lag'])
                data.iloc[i, data.columns.get_loc('adaptive_corr')] = corr_val if pd.notna(corr_val) else 0
    
    # Signal generation
    data['signal'] = 0
    
    # Buy when adaptive correlation suggests trend continuation
    buy_condition = (
        (data['adaptive_corr'] > correlation_threshold) &
        (data['returns'] > 0)
    )
    
    # Sell when correlation suggests trend reversal
    sell_condition = (
        (data['adaptive_corr'] < -correlation_threshold) |
        ((data['adaptive_corr'].abs() < 0.1) & (data['returns'] < 0))
    )
    
    data.loc[buy_condition, 'signal'] = 1
    data.loc[sell_condition, 'signal'] = -1
    
    # Clean signals
    data['signal'] = data['signal'].replace(0, np.nan).fillna(method='ffill').fillna(0)
    
    return data