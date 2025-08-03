"""
Statistical Trading Strategies
統計的トレード戦略
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Optional


def cointegration_pairs_strategy(data: pd.DataFrame,
                                lookback_window: int = 60,
                                zscore_threshold: float = 2.0,
                                half_life_threshold: int = 20) -> pd.DataFrame:
    """
    Cointegration Pairs Strategy (Bitcoin vs its own moving average)
    共和分ペア戦略（ビットコインと移動平均の関係）
    """
    data = data.copy()
    
    # Create synthetic pair with moving average
    data['ma_long'] = data['close'].rolling(window=lookback_window).mean()
    
    # Calculate spread
    data['spread'] = data['close'] - data['ma_long']
    
    # Calculate rolling z-score of spread
    data['spread_mean'] = data['spread'].rolling(window=lookback_window).mean()
    data['spread_std'] = data['spread'].rolling(window=lookback_window).std()
    data['zscore'] = (data['spread'] - data['spread_mean']) / data['spread_std']
    
    # Calculate half-life of mean reversion
    data['spread_lag'] = data['spread'].shift(1)
    data['spread_diff'] = data['spread'].diff()
    
    # Rolling regression for half-life calculation
    data['half_life'] = np.nan
    for i in range(lookback_window, len(data)):
        spread_window = data['spread'].iloc[i-lookback_window:i]
        spread_lag_window = data['spread_lag'].iloc[i-lookback_window:i]
        
        if len(spread_window.dropna()) > 10:
            try:
                slope, _, _, _, _ = stats.linregress(spread_lag_window.dropna(), spread_window.dropna())
                if slope < 0:
                    half_life = -np.log(2) / np.log(1 + slope)
                    data.iloc[i, data.columns.get_loc('half_life')] = min(half_life, 100)  # Cap at 100
            except:
                pass
    
    # Signal generation
    data['signal'] = 0
    
    # Buy when spread is below threshold and mean reversion is likely
    buy_condition = (
        (data['zscore'] < -zscore_threshold) &
        (data['half_life'] < half_life_threshold)
    )
    
    # Sell when spread is above threshold
    sell_condition = data['zscore'] > zscore_threshold
    
    data.loc[buy_condition, 'signal'] = 1
    data.loc[sell_condition, 'signal'] = -1
    
    # Clean signals
    data['signal'] = data['signal'].replace(0, np.nan).fillna(method='ffill').fillna(0)
    
    return data


def regime_detection_strategy(data: pd.DataFrame,
                            lookback_window: int = 50,
                            regime_threshold: float = 0.02) -> pd.DataFrame:
    """
    Regime Detection Strategy
    レジーム検出戦略
    
    Detects bull/bear market regimes using statistical tests
    """
    data = data.copy()
    
    # Calculate returns
    data['returns'] = data['close'].pct_change()
    
    # Rolling regime detection using variance ratio
    data['variance_ratio'] = np.nan
    data['regime'] = 0  # 0: neutral, 1: bull, -1: bear
    
    for i in range(lookback_window, len(data)):
        returns_window = data['returns'].iloc[i-lookback_window:i].dropna()
        
        if len(returns_window) > 20:
            # Calculate variance ratio test
            returns_2 = returns_window.rolling(window=2).sum().dropna()
            var_1 = returns_window.var()
            var_2 = returns_2.var() / 2
            
            if var_1 > 0:
                variance_ratio = var_2 / var_1
                data.iloc[i, data.columns.get_loc('variance_ratio')] = variance_ratio
                
                # Regime classification
                mean_return = returns_window.mean()
                if variance_ratio > 1.2 and mean_return > regime_threshold:
                    data.iloc[i, data.columns.get_loc('regime')] = 1  # Bull market
                elif variance_ratio > 1.2 and mean_return < -regime_threshold:
                    data.iloc[i, data.columns.get_loc('regime')] = -1  # Bear market
    
    # Signal generation based on regime
    data['signal'] = 0
    
    # Buy in bull regime
    buy_condition = data['regime'] == 1
    
    # Sell in bear regime
    sell_condition = data['regime'] == -1
    
    data.loc[buy_condition, 'signal'] = 1
    data.loc[sell_condition, 'signal'] = -1
    
    return data


def fractal_dimension_strategy(data: pd.DataFrame,
                              window: int = 30,
                              threshold: float = 1.5) -> pd.DataFrame:
    """
    Fractal Dimension Strategy
    フラクタル次元戦略
    
    Uses Hurst exponent to detect trending vs mean-reverting periods
    """
    data = data.copy()
    
    # Calculate rolling Hurst exponent (simplified version)
    data['hurst_exponent'] = np.nan
    
    for i in range(window, len(data)):
        prices = data['close'].iloc[i-window:i].values
        
        if len(prices) > 10:
            # Simple Hurst calculation using R/S analysis
            try:
                # Calculate returns
                returns = np.diff(np.log(prices))
                
                # Calculate cumulative deviations
                mean_return = np.mean(returns)
                cumulative_devs = np.cumsum(returns - mean_return)
                
                # Calculate range
                R = np.max(cumulative_devs) - np.min(cumulative_devs)
                
                # Calculate standard deviation
                S = np.std(returns)
                
                if S > 0:
                    # Rescaled range
                    rs = R / S
                    
                    # Hurst exponent approximation
                    n = len(returns)
                    hurst = np.log(rs) / np.log(n)
                    
                    # Bound Hurst exponent
                    hurst = max(0, min(1, hurst))
                    data.iloc[i, data.columns.get_loc('hurst_exponent')] = hurst
                    
            except:
                pass
    
    # Signal generation
    data['signal'] = 0
    
    # Buy when Hurst > 0.5 (trending market)
    buy_condition = (
        (data['hurst_exponent'] > 0.6) &
        (data['close'].pct_change() > 0)
    )
    
    # Sell when Hurst < 0.5 (mean-reverting market)
    sell_condition = (
        (data['hurst_exponent'] < 0.4) &
        (data['close'].pct_change() < 0)
    )
    
    data.loc[buy_condition, 'signal'] = 1
    data.loc[sell_condition, 'signal'] = -1
    
    # Clean signals
    data['signal'] = data['signal'].replace(0, np.nan).fillna(method='ffill').fillna(0)
    
    return data


def entropy_based_strategy(data: pd.DataFrame,
                          entropy_window: int = 20,
                          entropy_threshold: float = 0.7) -> pd.DataFrame:
    """
    Entropy-Based Strategy
    エントロピーベース戦略
    
    Uses Shannon entropy to measure market predictability
    """
    data = data.copy()
    
    # Calculate returns and discretize them
    data['returns'] = data['close'].pct_change()
    
    # Calculate rolling entropy
    data['entropy'] = np.nan
    
    for i in range(entropy_window, len(data)):
        returns_window = data['returns'].iloc[i-entropy_window:i].dropna()
        
        if len(returns_window) > 5:
            # Discretize returns into bins
            n_bins = min(10, len(returns_window) // 2)
            hist, _ = np.histogram(returns_window, bins=n_bins)
            
            # Calculate probability distribution
            probs = hist / np.sum(hist)
            probs = probs[probs > 0]  # Remove zero probabilities
            
            # Calculate Shannon entropy
            entropy = -np.sum(probs * np.log2(probs))
            data.iloc[i, data.columns.get_loc('entropy')] = entropy
    
    # Signal generation
    data['signal'] = 0
    
    # Buy when entropy is low (predictable market)
    buy_condition = (
        (data['entropy'] < entropy_threshold) &
        (data['returns'] > 0)
    )
    
    # Sell when entropy is high (unpredictable market)
    sell_condition = (
        (data['entropy'] > entropy_threshold + 0.5) |
        ((data['entropy'] < entropy_threshold) & (data['returns'] < 0))
    )
    
    data.loc[buy_condition, 'signal'] = 1
    data.loc[sell_condition, 'signal'] = -1
    
    # Clean signals
    data['signal'] = data['signal'].replace(0, np.nan).fillna(method='ffill').fillna(0)
    
    return data


def kalman_filter_strategy(data: pd.DataFrame,
                          process_variance: float = 1e-4,
                          observation_variance: float = 1e-2) -> pd.DataFrame:
    """
    Kalman Filter Strategy
    カルマンフィルター戦略
    
    Uses Kalman filter to estimate true price and generate signals
    """
    data = data.copy()
    
    # Simple Kalman filter implementation
    n = len(data)
    
    # State variables
    x_hat = np.zeros(n)  # State estimate
    P = np.zeros(n)      # Error covariance
    
    # Kalman filter parameters
    Q = process_variance      # Process variance
    R = observation_variance  # Observation variance
    
    # Initialize
    x_hat[0] = data['close'].iloc[0]
    P[0] = 1.0
    
    # Kalman filter loop
    for i in range(1, n):
        # Prediction
        x_hat_minus = x_hat[i-1]
        P_minus = P[i-1] + Q
        
        # Update
        K = P_minus / (P_minus + R)  # Kalman gain
        x_hat[i] = x_hat_minus + K * (data['close'].iloc[i] - x_hat_minus)
        P[i] = (1 - K) * P_minus
    
    data['kalman_price'] = x_hat
    data['price_deviation'] = (data['close'] - data['kalman_price']) / data['kalman_price']
    
    # Signal generation
    data['signal'] = 0
    
    # Buy when actual price is below Kalman estimate (undervalued)
    buy_condition = data['price_deviation'] < -0.02
    
    # Sell when actual price is above Kalman estimate (overvalued)
    sell_condition = data['price_deviation'] > 0.02
    
    data.loc[buy_condition, 'signal'] = 1
    data.loc[sell_condition, 'signal'] = -1
    
    # Clean signals
    data['signal'] = data['signal'].replace(0, np.nan).fillna(method='ffill').fillna(0)
    
    return data


def granger_causality_strategy(data: pd.DataFrame,
                              volume_lag: int = 5,
                              causality_window: int = 30,
                              significance_level: float = 0.05) -> pd.DataFrame:
    """
    Granger Causality Strategy
    グレンジャー因果性戦略
    
    Tests if volume Granger-causes price movements
    """
    data = data.copy()
    
    # Prepare data
    data['price_change'] = data['close'].pct_change()
    data['volume_change'] = data['volume'].pct_change()
    
    # Rolling Granger causality test (simplified)
    data['granger_p_value'] = np.nan
    
    for i in range(causality_window + volume_lag, len(data)):
        price_window = data['price_change'].iloc[i-causality_window:i].dropna()
        volume_window = data['volume_change'].iloc[i-causality_window-volume_lag:i-volume_lag].dropna()
        
        if len(price_window) > 10 and len(volume_window) > 10:
            # Align series
            min_len = min(len(price_window), len(volume_window))
            price_series = price_window.iloc[-min_len:].values
            volume_series = volume_window.iloc[-min_len:].values
            
            try:
                # Simple correlation-based causality proxy
                correlation = np.corrcoef(price_series[volume_lag:], volume_series[:-volume_lag])[0, 1]
                
                # Convert correlation to pseudo p-value
                if not np.isnan(correlation):
                    t_stat = correlation * np.sqrt((min_len - 2) / (1 - correlation**2))
                    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), min_len - 2))
                    data.iloc[i, data.columns.get_loc('granger_p_value')] = p_value
                    
            except:
                pass
    
    # Signal generation
    data['signal'] = 0
    
    # Buy when volume Granger-causes price (significant causality)
    buy_condition = (
        (data['granger_p_value'] < significance_level) &
        (data['volume_change'] > 0) &
        (data['price_change'] > 0)
    )
    
    # Sell when no causality or negative signals
    sell_condition = (
        (data['granger_p_value'] > significance_level) |
        ((data['granger_p_value'] < significance_level) & (data['volume_change'] < 0))
    )
    
    data.loc[buy_condition, 'signal'] = 1
    data.loc[sell_condition, 'signal'] = -1
    
    # Clean signals
    data['signal'] = data['signal'].replace(0, np.nan).fillna(method='ffill').fillna(0)
    
    return data