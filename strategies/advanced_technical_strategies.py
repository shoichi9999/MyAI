"""
Advanced Technical Analysis Strategies
高度なテクニカル分析戦略
"""

import pandas as pd
import numpy as np
from typing import Optional


def ichimoku_cloud_strategy(data: pd.DataFrame,
                           tenkan_period: int = 9,
                           kijun_period: int = 26,
                           senkou_span_b_period: int = 52) -> pd.DataFrame:
    """
    Ichimoku Cloud Strategy
    一目均衡表戦略
    """
    data = data.copy()
    
    # Tenkan-sen (Conversion Line)
    high_tenkan = data['high'].rolling(window=tenkan_period).max()
    low_tenkan = data['low'].rolling(window=tenkan_period).min()
    data['tenkan_sen'] = (high_tenkan + low_tenkan) / 2
    
    # Kijun-sen (Base Line)
    high_kijun = data['high'].rolling(window=kijun_period).max()
    low_kijun = data['low'].rolling(window=kijun_period).min()
    data['kijun_sen'] = (high_kijun + low_kijun) / 2
    
    # Senkou Span A (Leading Span A)
    data['senkou_span_a'] = ((data['tenkan_sen'] + data['kijun_sen']) / 2).shift(kijun_period)
    
    # Senkou Span B (Leading Span B)
    high_senkou = data['high'].rolling(window=senkou_span_b_period).max()
    low_senkou = data['low'].rolling(window=senkou_span_b_period).min()
    data['senkou_span_b'] = ((high_senkou + low_senkou) / 2).shift(kijun_period)
    
    # Chikou Span (Lagging Span)
    data['chikou_span'] = data['close'].shift(-kijun_period)
    
    # Cloud top and bottom
    data['cloud_top'] = np.maximum(data['senkou_span_a'], data['senkou_span_b'])
    data['cloud_bottom'] = np.minimum(data['senkou_span_a'], data['senkou_span_b'])
    
    # Signal generation
    data['signal'] = 0
    
    # Buy conditions: 
    # 1. Price above cloud
    # 2. Tenkan above Kijun
    # 3. Chikou span above price from 26 periods ago
    buy_condition = (
        (data['close'] > data['cloud_top']) &
        (data['tenkan_sen'] > data['kijun_sen']) &
        (data['chikou_span'] > data['close'].shift(kijun_period))
    )
    
    # Sell conditions:
    # 1. Price below cloud
    # 2. Tenkan below Kijun
    sell_condition = (
        (data['close'] < data['cloud_bottom']) &
        (data['tenkan_sen'] < data['kijun_sen'])
    )
    
    data.loc[buy_condition, 'signal'] = 1
    data.loc[sell_condition, 'signal'] = -1
    
    data['signal'] = data['signal'].replace(0, np.nan).fillna(method='ffill').fillna(0)
    return data


def elliott_wave_strategy(data: pd.DataFrame,
                         wave_window: int = 50,
                         fibonacci_levels: list = [0.236, 0.382, 0.618]) -> pd.DataFrame:
    """
    Elliott Wave Strategy (Simplified)
    エリオット波動戦略（簡略版）
    """
    data = data.copy()
    
    # Find local extrema (peaks and troughs)
    data['local_max'] = data['high'] == data['high'].rolling(window=5, center=True).max()
    data['local_min'] = data['low'] == data['low'].rolling(window=5, center=True).min()
    
    # Calculate wave patterns
    data['wave_direction'] = 0
    data['wave_strength'] = 0
    
    for i in range(wave_window, len(data)):
        window_data = data.iloc[i-wave_window:i]
        
        # Count peaks and troughs
        peaks = window_data[window_data['local_max']]['high'].values
        troughs = window_data[window_data['local_min']]['low'].values
        
        if len(peaks) >= 3 and len(troughs) >= 2:
            # Simple wave analysis
            recent_peak = peaks[-1]
            recent_trough = troughs[-1]
            prev_peak = peaks[-2] if len(peaks) >= 2 else recent_peak
            prev_trough = troughs[-2] if len(troughs) >= 2 else recent_trough
            
            # Wave direction
            if recent_peak > prev_peak and recent_trough > prev_trough:
                data.iloc[i, data.columns.get_loc('wave_direction')] = 1  # Uptrend
            elif recent_peak < prev_peak and recent_trough < prev_trough:
                data.iloc[i, data.columns.get_loc('wave_direction')] = -1  # Downtrend
            
            # Wave strength based on Fibonacci retracements
            if recent_peak != recent_trough:
                current_price = data['close'].iloc[i]
                retracement = (recent_peak - current_price) / (recent_peak - recent_trough)
                
                # Check if price is at key Fibonacci levels
                for fib_level in fibonacci_levels:
                    if abs(retracement - fib_level) < 0.05:  # Within 5% of Fibonacci level
                        data.iloc[i, data.columns.get_loc('wave_strength')] = 1
                        break
    
    # Signal generation
    data['signal'] = 0
    
    # Buy when in uptrend and at Fibonacci support
    buy_condition = (
        (data['wave_direction'] == 1) &
        (data['wave_strength'] == 1) &
        (data['close'] < data['close'].shift(1))  # Price pullback
    )
    
    # Sell when in downtrend or at resistance
    sell_condition = (
        (data['wave_direction'] == -1) |
        ((data['wave_strength'] == 1) & (data['close'] > data['close'].shift(1)))
    )
    
    data.loc[buy_condition, 'signal'] = 1
    data.loc[sell_condition, 'signal'] = -1
    
    data['signal'] = data['signal'].replace(0, np.nan).fillna(method='ffill').fillna(0)
    return data


def harmonic_pattern_strategy(data: pd.DataFrame,
                             pattern_window: int = 40,
                             tolerance: float = 0.05) -> pd.DataFrame:
    """
    Harmonic Pattern Strategy (Gartley, Butterfly, etc.)
    ハーモニックパターン戦略
    """
    data = data.copy()
    
    # Find swing points
    data['swing_high'] = data['high'] == data['high'].rolling(window=7, center=True).max()
    data['swing_low'] = data['low'] == data['low'].rolling(window=7, center=True).min()
    
    data['harmonic_signal'] = 0
    
    for i in range(pattern_window, len(data)):
        window_data = data.iloc[i-pattern_window:i]
        
        # Get swing points
        highs = window_data[window_data['swing_high']].copy()
        lows = window_data[window_data['swing_low']].copy()
        
        if len(highs) >= 2 and len(lows) >= 2:
            # Combine and sort by index
            swings = pd.concat([
                highs[['high']].rename(columns={'high': 'price'}),
                lows[['low']].rename(columns={'low': 'price'})
            ]).sort_index()
            
            if len(swings) >= 5:  # Need at least 5 points for harmonic pattern
                points = swings['price'].values[-5:]  # Last 5 swing points
                
                # Calculate ratios for Gartley pattern
                # X-A, A-B, B-C, C-D
                if len(points) == 5:
                    X, A, B, C, D = points
                    
                    # Gartley ratios
                    AB_XA = abs(B - A) / abs(X - A) if X != A else 0
                    BC_AB = abs(C - B) / abs(A - B) if A != B else 0
                    CD_BC = abs(D - C) / abs(B - C) if B != C else 0
                    
                    # Check if ratios match Gartley pattern (approximately)
                    gartley_match = (
                        abs(AB_XA - 0.618) < tolerance and
                        abs(BC_AB - 0.382) < tolerance and
                        abs(CD_BC - 1.272) < tolerance
                    )
                    
                    if gartley_match:
                        # Determine bullish or bearish pattern
                        if D < C:  # Bullish Gartley
                            data.iloc[i, data.columns.get_loc('harmonic_signal')] = 1
                        else:  # Bearish Gartley
                            data.iloc[i, data.columns.get_loc('harmonic_signal')] = -1
    
    # Signal generation
    data['signal'] = data['harmonic_signal']
    data['signal'] = data['signal'].replace(0, np.nan).fillna(method='ffill').fillna(0)
    
    return data


def market_profile_strategy(data: pd.DataFrame,
                           profile_window: int = 20,
                           value_area_percent: float = 0.7) -> pd.DataFrame:
    """
    Market Profile Strategy
    マーケットプロファイル戦略
    """
    data = data.copy()
    
    data['poc'] = np.nan  # Point of Control
    data['value_area_high'] = np.nan
    data['value_area_low'] = np.nan
    
    for i in range(profile_window, len(data)):
        window_data = data.iloc[i-profile_window:i]
        
        # Create price levels
        price_min = window_data['low'].min()
        price_max = window_data['high'].max()
        
        if price_max > price_min:
            # Create price bins
            n_bins = 20
            price_bins = np.linspace(price_min, price_max, n_bins + 1)
            
            # Calculate volume at each price level
            volume_profile = np.zeros(n_bins)
            
            for _, row in window_data.iterrows():
                # Distribute volume across price range
                low_bin = np.searchsorted(price_bins, row['low'], side='right') - 1
                high_bin = np.searchsorted(price_bins, row['high'], side='left')
                
                low_bin = max(0, min(n_bins - 1, low_bin))
                high_bin = max(0, min(n_bins - 1, high_bin))
                
                if high_bin >= low_bin:
                    bins_span = high_bin - low_bin + 1
                    volume_per_bin = row['volume'] / bins_span
                    volume_profile[low_bin:high_bin + 1] += volume_per_bin
            
            # Find Point of Control (highest volume)
            poc_bin = np.argmax(volume_profile)
            poc_price = (price_bins[poc_bin] + price_bins[poc_bin + 1]) / 2
            data.iloc[i, data.columns.get_loc('poc')] = poc_price
            
            # Calculate Value Area (70% of volume)
            total_volume = np.sum(volume_profile)
            target_volume = total_volume * value_area_percent
            
            # Expand from POC
            cumulative_volume = volume_profile[poc_bin]
            va_low_bin = poc_bin
            va_high_bin = poc_bin
            
            while cumulative_volume < target_volume and (va_low_bin > 0 or va_high_bin < n_bins - 1):
                # Add the bin with higher volume
                low_vol = volume_profile[va_low_bin - 1] if va_low_bin > 0 else 0
                high_vol = volume_profile[va_high_bin + 1] if va_high_bin < n_bins - 1 else 0
                
                if low_vol >= high_vol:
                    va_low_bin -= 1
                    cumulative_volume += low_vol
                elif high_vol > 0:
                    va_high_bin += 1
                    cumulative_volume += high_vol
                else:
                    break
            
            va_low_price = price_bins[va_low_bin]
            va_high_price = price_bins[va_high_bin + 1]
            
            data.iloc[i, data.columns.get_loc('value_area_low')] = va_low_price
            data.iloc[i, data.columns.get_loc('value_area_high')] = va_high_price
    
    # Signal generation
    data['signal'] = 0
    
    current_price = data['close']
    
    # Buy when price is below value area (discount)
    buy_condition = (
        (current_price < data['value_area_low']) &
        (current_price > data['value_area_low'] * 0.95)  # Close to value area
    )
    
    # Sell when price is above value area (premium)
    sell_condition = current_price > data['value_area_high']
    
    data.loc[buy_condition, 'signal'] = 1
    data.loc[sell_condition, 'signal'] = -1
    
    data['signal'] = data['signal'].replace(0, np.nan).fillna(method='ffill').fillna(0)
    return data


def wyckoff_method_strategy(data: pd.DataFrame,
                           accumulation_window: int = 30,
                           volume_threshold: float = 1.5) -> pd.DataFrame:
    """
    Wyckoff Method Strategy
    ワイコフ法戦略
    """
    data = data.copy()
    
    # Calculate relative volume
    data['volume_ma'] = data['volume'].rolling(window=20).mean()
    data['relative_volume'] = data['volume'] / data['volume_ma']
    
    # Price spread (range)
    data['price_spread'] = data['high'] - data['low']
    data['spread_ma'] = data['price_spread'].rolling(window=20).mean()
    data['relative_spread'] = data['price_spread'] / data['spread_ma']
    
    # Volume Spread Analysis (VSA)
    data['vsa_signal'] = 0
    
    # Wyckoff phases detection
    data['wyckoff_phase'] = 0  # 0: neutral, 1: accumulation, 2: markup, 3: distribution, 4: markdown
    
    for i in range(accumulation_window, len(data)):
        window_data = data.iloc[i-accumulation_window:i]
        
        # Check for accumulation phase
        # High volume, narrow spread, little price movement
        recent_volume = window_data['relative_volume'].tail(10).mean()
        recent_spread = window_data['relative_spread'].tail(10).mean()
        price_change = abs(window_data['close'].iloc[-1] - window_data['close'].iloc[0]) / window_data['close'].iloc[0]
        
        if recent_volume > volume_threshold and recent_spread < 0.8 and price_change < 0.05:
            data.iloc[i, data.columns.get_loc('wyckoff_phase')] = 1  # Accumulation
        
        # Check for markup phase
        # Increasing price with good volume
        elif (window_data['close'].iloc[-1] > window_data['close'].iloc[0] and 
              recent_volume > 1.2 and price_change > 0.03):
            data.iloc[i, data.columns.get_loc('wyckoff_phase')] = 2  # Markup
        
        # Check for distribution phase
        # High volume, wider spreads, sideways movement
        elif recent_volume > volume_threshold and recent_spread > 1.2 and price_change < 0.05:
            data.iloc[i, data.columns.get_loc('wyckoff_phase')] = 3  # Distribution
        
        # Check for markdown phase
        # Declining price with volume
        elif (window_data['close'].iloc[-1] < window_data['close'].iloc[0] and 
              recent_volume > 1.1 and price_change > 0.03):
            data.iloc[i, data.columns.get_loc('wyckoff_phase')] = 4  # Markdown
    
    # Volume analysis signals
    current_vol = data['relative_volume']
    current_spread = data['relative_spread']
    price_change = data['close'].pct_change()
    
    # High volume, narrow spread, up close = buying climax (sell signal)
    data.loc[(current_vol > 2) & (current_spread < 0.5) & (price_change > 0.02), 'vsa_signal'] = -1
    
    # High volume, wide spread, down close = selling climax (buy signal)
    data.loc[(current_vol > 2) & (current_spread > 1.5) & (price_change < -0.02), 'vsa_signal'] = 1
    
    # Signal generation
    data['signal'] = 0
    
    # Buy signals
    buy_condition = (
        (data['wyckoff_phase'] == 1) |  # End of accumulation
        (data['vsa_signal'] == 1)       # VSA buy signal
    )
    
    # Sell signals
    sell_condition = (
        (data['wyckoff_phase'] == 3) |  # Start of distribution
        (data['vsa_signal'] == -1)      # VSA sell signal
    )
    
    data.loc[buy_condition, 'signal'] = 1
    data.loc[sell_condition, 'signal'] = -1
    
    data['signal'] = data['signal'].replace(0, np.nan).fillna(method='ffill').fillna(0)
    return data