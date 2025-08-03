"""
Market Microstructure Trading Strategies
マーケットマイクロストラクチャー戦略
"""

import pandas as pd
import numpy as np
from typing import Optional


def order_flow_imbalance_strategy(data: pd.DataFrame,
                                 window: int = 20,
                                 imbalance_threshold: float = 0.3) -> pd.DataFrame:
    """
    Order Flow Imbalance Strategy
    オーダーフロー不均衡戦略
    """
    data = data.copy()
    
    # Estimate buy/sell volume using price movement and volume
    data['price_change'] = data['close'].diff()
    data['volume_weighted_price'] = (data['high'] + data['low'] + data['close']) / 3
    
    # Approximate buy/sell volume based on price direction
    data['buy_volume'] = np.where(data['price_change'] > 0, 
                                  data['volume'] * (1 + data['price_change']), 
                                  data['volume'] * 0.3)
    data['sell_volume'] = data['volume'] - data['buy_volume']
    
    # Rolling order flow imbalance
    data['buy_volume_ma'] = data['buy_volume'].rolling(window=window).sum()
    data['sell_volume_ma'] = data['sell_volume'].rolling(window=window).sum()
    data['total_volume_ma'] = data['buy_volume_ma'] + data['sell_volume_ma']
    
    # Order flow imbalance ratio
    data['ofi'] = (data['buy_volume_ma'] - data['sell_volume_ma']) / data['total_volume_ma']
    
    # Volume-weighted price momentum
    data['vwap'] = (data['volume_weighted_price'] * data['volume']).rolling(window=window).sum() / data['volume'].rolling(window=window).sum()
    data['price_vs_vwap'] = (data['close'] - data['vwap']) / data['vwap']
    
    # Signal generation
    data['signal'] = 0
    
    # Buy when strong buy flow imbalance and price above VWAP
    buy_condition = (
        (data['ofi'] > imbalance_threshold) &
        (data['price_vs_vwap'] > 0.001)
    )
    
    # Sell when strong sell flow imbalance and price below VWAP
    sell_condition = (
        (data['ofi'] < -imbalance_threshold) &
        (data['price_vs_vwap'] < -0.001)
    )
    
    data.loc[buy_condition, 'signal'] = 1
    data.loc[sell_condition, 'signal'] = -1
    
    data['signal'] = data['signal'].replace(0, np.nan).fillna(method='ffill').fillna(0)
    return data


def volume_price_analysis_strategy(data: pd.DataFrame,
                                  volume_window: int = 15,
                                  price_window: int = 10) -> pd.DataFrame:
    """
    Volume Price Analysis Strategy
    出来高価格分析戦略
    """
    data = data.copy()
    
    # Volume analysis
    data['volume_ma'] = data['volume'].rolling(window=volume_window).mean()
    data['volume_ratio'] = data['volume'] / data['volume_ma']
    data['volume_momentum'] = data['volume'].pct_change(5)
    
    # Price analysis
    data['price_momentum'] = data['close'].pct_change(price_window)
    data['price_volatility'] = data['close'].pct_change().rolling(window=price_window).std()
    
    # Volume-Price Trend (VPT)
    data['vpt'] = (data['volume'] * data['close'].pct_change()).cumsum()
    data['vpt_ma'] = data['vpt'].rolling(window=volume_window).mean()
    data['vpt_signal'] = np.where(data['vpt'] > data['vpt_ma'], 1, -1)
    
    # Money Flow Index approximation
    data['typical_price'] = (data['high'] + data['low'] + data['close']) / 3
    data['money_flow'] = data['typical_price'] * data['volume']
    
    positive_flow = data['money_flow'].where(data['typical_price'].diff() > 0, 0)
    negative_flow = data['money_flow'].where(data['typical_price'].diff() < 0, 0)
    
    data['positive_flow_sum'] = positive_flow.rolling(window=volume_window).sum()
    data['negative_flow_sum'] = negative_flow.rolling(window=volume_window).sum()
    
    data['money_flow_ratio'] = data['positive_flow_sum'] / (data['negative_flow_sum'] + 1e-10)
    data['mfi'] = 100 - (100 / (1 + data['money_flow_ratio']))
    
    # Accumulation/Distribution Line
    data['clv'] = ((data['close'] - data['low']) - (data['high'] - data['close'])) / (data['high'] - data['low'] + 1e-10)
    data['ad_line'] = (data['clv'] * data['volume']).cumsum()
    data['ad_line_ma'] = data['ad_line'].rolling(window=volume_window).mean()
    
    # Signal generation
    data['signal'] = 0
    
    # Buy conditions: Strong volume with upward price movement
    buy_condition = (
        (data['volume_ratio'] > 1.5) &
        (data['price_momentum'] > 0.02) &
        (data['vpt_signal'] == 1) &
        (data['mfi'] > 50) &
        (data['ad_line'] > data['ad_line_ma'])
    )
    
    # Sell conditions: High volume with downward price movement
    sell_condition = (
        (data['volume_ratio'] > 1.3) &
        (data['price_momentum'] < -0.02) &
        (data['vpt_signal'] == -1) &
        (data['mfi'] < 50)
    )
    
    data.loc[buy_condition, 'signal'] = 1
    data.loc[sell_condition, 'signal'] = -1
    
    data['signal'] = data['signal'].replace(0, np.nan).fillna(method='ffill').fillna(0)
    return data


def liquidity_provision_strategy(data: pd.DataFrame,
                                spread_window: int = 20,
                                liquidity_threshold: float = 0.01) -> pd.DataFrame:
    """
    Liquidity Provision Strategy
    流動性提供戦略
    """
    data = data.copy()
    
    # Estimate bid-ask spread using high-low range
    data['spread'] = (data['high'] - data['low']) / data['close']
    data['spread_ma'] = data['spread'].rolling(window=spread_window).mean()
    data['spread_normalized'] = (data['spread'] - data['spread_ma']) / data['spread_ma']
    
    # Price impact estimation
    data['price_impact'] = abs(data['close'].pct_change()) / (data['volume'] / data['volume'].rolling(window=20).mean())
    data['price_impact_ma'] = data['price_impact'].rolling(window=spread_window).mean()
    
    # Market depth proxy (inverse of volatility)
    data['volatility'] = data['close'].pct_change().rolling(window=spread_window).std()
    data['market_depth'] = 1 / (data['volatility'] + 1e-6)
    data['depth_percentile'] = data['market_depth'].rolling(window=50).rank(pct=True)
    
    # Liquidity conditions
    data['high_liquidity'] = (
        (data['spread_normalized'] < -0.2) &  # Narrow spreads
        (data['depth_percentile'] > 0.7) &    # High market depth
        (data['price_impact'] < data['price_impact_ma'])  # Low price impact
    )
    
    data['low_liquidity'] = (
        (data['spread_normalized'] > 0.3) |   # Wide spreads
        (data['depth_percentile'] < 0.3) |    # Low market depth
        (data['price_impact'] > data['price_impact_ma'] * 1.5)  # High price impact
    )
    
    # Signal generation
    data['signal'] = 0
    
    # In high liquidity: momentum strategy
    buy_high_liq = (
        data['high_liquidity'] &
        (data['close'].pct_change(5) > 0.01)
    )
    
    sell_high_liq = (
        data['high_liquidity'] &
        (data['close'].pct_change(5) < -0.01)
    )
    
    # In low liquidity: mean reversion strategy
    buy_low_liq = (
        data['low_liquidity'] &
        (data['close'] < data['close'].rolling(window=20).mean() * 0.98)
    )
    
    sell_low_liq = (
        data['low_liquidity'] &
        (data['close'] > data['close'].rolling(window=20).mean() * 1.02)
    )
    
    data.loc[buy_high_liq | buy_low_liq, 'signal'] = 1
    data.loc[sell_high_liq | sell_low_liq, 'signal'] = -1
    
    data['signal'] = data['signal'].replace(0, np.nan).fillna(method='ffill').fillna(0)
    return data


def tick_analysis_strategy(data: pd.DataFrame,
                          tick_window: int = 10,
                          uptick_threshold: float = 0.6) -> pd.DataFrame:
    """
    Tick Analysis Strategy
    ティック分析戦略
    """
    data = data.copy()
    
    # Tick direction (simplified - using close price changes)
    data['tick_direction'] = np.sign(data['close'].diff())
    data['tick_direction'] = data['tick_direction'].replace(0, np.nan).fillna(method='ffill')
    
    # Rolling tick statistics
    data['uptick_ratio'] = (data['tick_direction'] == 1).rolling(window=tick_window).mean()
    data['downtick_ratio'] = (data['tick_direction'] == -1).rolling(window=tick_window).mean()
    
    # Tick momentum
    data['tick_momentum'] = data['uptick_ratio'] - data['downtick_ratio']
    data['tick_momentum_ma'] = data['tick_momentum'].rolling(window=5).mean()
    
    # Price acceleration (second derivative)
    data['price_velocity'] = data['close'].diff()
    data['price_acceleration'] = data['price_velocity'].diff()
    
    # Volume-weighted tick analysis
    data['volume_weighted_ticks'] = data['tick_direction'] * data['volume']
    data['vw_tick_sum'] = data['volume_weighted_ticks'].rolling(window=tick_window).sum()
    data['total_volume'] = data['volume'].rolling(window=tick_window).sum()
    data['vw_tick_ratio'] = data['vw_tick_sum'] / (data['total_volume'] + 1e-10)
    
    # Signal generation
    data['signal'] = 0
    
    # Buy when strong uptick momentum and positive acceleration
    buy_condition = (
        (data['uptick_ratio'] > uptick_threshold) &
        (data['tick_momentum'] > 0.2) &
        (data['price_acceleration'] > 0) &
        (data['vw_tick_ratio'] > 0.1)
    )
    
    # Sell when strong downtick momentum and negative acceleration
    sell_condition = (
        (data['downtick_ratio'] > uptick_threshold) &
        (data['tick_momentum'] < -0.2) &
        (data['price_acceleration'] < 0) &
        (data['vw_tick_ratio'] < -0.1)
    )
    
    data.loc[buy_condition, 'signal'] = 1
    data.loc[sell_condition, 'signal'] = -1
    
    data['signal'] = data['signal'].replace(0, np.nan).fillna(method='ffill').fillna(0)
    return data


def market_making_strategy(data: pd.DataFrame,
                          inventory_target: float = 0.5,
                          spread_multiple: float = 2.0) -> pd.DataFrame:
    """
    Market Making Strategy
    マーケットメイキング戦略
    """
    data = data.copy()
    
    # Estimate fair value using various methods
    data['ema_fast'] = data['close'].ewm(span=12).mean()
    data['ema_slow'] = data['close'].ewm(span=26).mean()
    data['vwap'] = (data['close'] * data['volume']).rolling(window=20).sum() / data['volume'].rolling(window=20).sum()
    
    # Fair value as average of multiple estimates
    data['fair_value'] = (data['ema_fast'] + data['ema_slow'] + data['vwap']) / 3
    
    # Spread estimation
    data['volatility'] = data['close'].pct_change().rolling(window=20).std()
    data['estimated_spread'] = data['volatility'] * data['close'] * spread_multiple
    
    # Inventory management
    data['inventory_signal'] = 0
    data['inventory_level'] = 0  # Simplified inventory tracking
    
    # Market making logic
    data['bid_price'] = data['fair_value'] - data['estimated_spread'] / 2
    data['ask_price'] = data['fair_value'] + data['estimated_spread'] / 2
    
    # Signal based on where current price is relative to our quotes
    data['signal'] = 0
    
    # Buy when price hits our bid (we're providing liquidity to sellers)
    buy_condition = (
        (data['close'] <= data['bid_price']) &
        (data['inventory_level'] < inventory_target)
    )
    
    # Sell when price hits our ask (we're providing liquidity to buyers) 
    sell_condition = (
        (data['close'] >= data['ask_price']) &
        (data['inventory_level'] > -inventory_target)
    )
    
    # Update inventory (simplified)
    for i in range(1, len(data)):
        prev_inventory = data['inventory_level'].iloc[i-1]
        
        if data.loc[data.index[i-1], 'signal'] == 1:  # We bought
            data.iloc[i, data.columns.get_loc('inventory_level')] = min(1.0, prev_inventory + 0.1)
        elif data.loc[data.index[i-1], 'signal'] == -1:  # We sold
            data.iloc[i, data.columns.get_loc('inventory_level')] = max(-1.0, prev_inventory - 0.1)
        else:
            data.iloc[i, data.columns.get_loc('inventory_level')] = prev_inventory * 0.99  # Decay
    
    data.loc[buy_condition, 'signal'] = 1
    data.loc[sell_condition, 'signal'] = -1
    
    return data


def high_frequency_momentum_strategy(data: pd.DataFrame,
                                    momentum_window: int = 3,
                                    volume_factor: float = 1.5) -> pd.DataFrame:
    """
    High Frequency Momentum Strategy
    高頻度モメンタム戦略
    """
    data = data.copy()
    
    # Short-term momentum indicators
    data['micro_momentum'] = data['close'].pct_change(momentum_window)
    data['micro_volume_momentum'] = data['volume'].pct_change(momentum_window)
    data['price_volume_correlation'] = data['close'].pct_change().rolling(window=momentum_window).corr(
        data['volume'].pct_change()
    )
    
    # Intrabar analysis (using OHLC)
    data['intrabar_momentum'] = (data['close'] - data['open']) / data['open']
    data['intrabar_volume_ratio'] = data['volume'] / data['volume'].rolling(window=5).mean()
    
    # Microstructure noise filtering
    data['price_filter'] = data['close'].rolling(window=3).median()  # Median filter
    data['filtered_momentum'] = data['price_filter'].pct_change(momentum_window)
    
    # Volume burst detection
    data['volume_burst'] = (
        (data['volume'] > data['volume'].rolling(window=10).mean() * volume_factor) &
        (data['volume'] > data['volume'].shift(1) * 1.2)
    )
    
    # Signal generation
    data['signal'] = 0
    
    # Buy on strong positive momentum with volume confirmation
    buy_condition = (
        (data['filtered_momentum'] > 0.005) &
        (data['intrabar_momentum'] > 0.002) &
        (data['volume_burst'] | (data['intrabar_volume_ratio'] > 1.3)) &
        (data['price_volume_correlation'] > 0.3)
    )
    
    # Sell on strong negative momentum with volume confirmation
    sell_condition = (
        (data['filtered_momentum'] < -0.005) &
        (data['intrabar_momentum'] < -0.002) &
        (data['volume_burst'] | (data['intrabar_volume_ratio'] > 1.3)) &
        (data['price_volume_correlation'] < -0.3)
    )
    
    data.loc[buy_condition, 'signal'] = 1
    data.loc[sell_condition, 'signal'] = -1
    
    # Hold signals for very short duration (high frequency)
    data['signal'] = data['signal'].replace(0, np.nan).fillna(method='ffill', limit=2).fillna(0)
    
    return data