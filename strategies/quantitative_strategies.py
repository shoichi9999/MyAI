"""
Quantitative Trading Strategies
クオンタティブトレード戦略
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Optional


def factor_model_strategy(data: pd.DataFrame,
                         lookback_window: int = 60,
                         factor_threshold: float = 1.5) -> pd.DataFrame:
    """
    Multi-Factor Model Strategy
    マルチファクターモデル戦略
    """
    data = data.copy()
    
    # Create factors
    data['returns'] = data['close'].pct_change()
    
    # Factor 1: Momentum
    data['momentum_factor'] = data['returns'].rolling(window=20).sum()
    
    # Factor 2: Volatility
    data['volatility_factor'] = data['returns'].rolling(window=20).std()
    
    # Factor 3: Volume
    data['volume_factor'] = (data['volume'] / data['volume'].rolling(window=20).mean() - 1)
    
    # Factor 4: Mean Reversion
    data['price_ma'] = data['close'].rolling(window=20).mean()
    data['mean_reversion_factor'] = (data['close'] - data['price_ma']) / data['price_ma']
    
    # Factor 5: Skewness
    data['skewness_factor'] = data['returns'].rolling(window=20).skew()
    
    # Factor 6: Kurtosis
    data['kurtosis_factor'] = data['returns'].rolling(window=20).kurt()
    
    # Standardize factors (z-score)
    factor_cols = ['momentum_factor', 'volatility_factor', 'volume_factor', 
                   'mean_reversion_factor', 'skewness_factor', 'kurtosis_factor']
    
    for col in factor_cols:
        rolling_mean = data[col].rolling(window=lookback_window).mean()
        rolling_std = data[col].rolling(window=lookback_window).std()
        data[f'{col}_zscore'] = (data[col] - rolling_mean) / rolling_std
    
    # Factor combination score
    data['factor_score'] = (
        data['momentum_factor_zscore'] * 0.3 +
        data['volatility_factor_zscore'] * -0.2 +  # Negative weight for volatility
        data['volume_factor_zscore'] * 0.2 +
        data['mean_reversion_factor_zscore'] * -0.1 +  # Contrarian
        data['skewness_factor_zscore'] * 0.1 +
        data['kurtosis_factor_zscore'] * 0.1
    )
    
    # Signal generation
    data['signal'] = 0
    
    # Buy when factor score is strongly positive
    buy_condition = data['factor_score'] > factor_threshold
    
    # Sell when factor score is strongly negative
    sell_condition = data['factor_score'] < -factor_threshold
    
    data.loc[buy_condition, 'signal'] = 1
    data.loc[sell_condition, 'signal'] = -1
    
    data['signal'] = data['signal'].replace(0, np.nan).fillna(method='ffill').fillna(0)
    return data


def risk_parity_strategy(data: pd.DataFrame,
                        volatility_window: int = 30,
                        rebalance_frequency: int = 5) -> pd.DataFrame:
    """
    Risk Parity Strategy (Bitcoin vs Cash)
    リスクパリティ戦略
    """
    data = data.copy()
    
    # Calculate rolling volatility
    data['returns'] = data['close'].pct_change()
    data['volatility'] = data['returns'].rolling(window=volatility_window).std() * np.sqrt(252)  # Annualized
    
    # Risk-free rate proxy (assuming 2% annual)
    risk_free_rate = 0.02
    data['excess_returns'] = data['returns'] - risk_free_rate / 252
    
    # Sharpe ratio
    data['sharpe_ratio'] = data['excess_returns'].rolling(window=volatility_window).mean() / data['returns'].rolling(window=volatility_window).std()
    
    # Risk parity weight calculation
    # Weight inversely proportional to volatility
    cash_volatility = 0.01  # Assume 1% volatility for cash
    
    data['btc_weight'] = cash_volatility / (data['volatility'] + cash_volatility)
    data['cash_weight'] = 1 - data['btc_weight']
    
    # Adjust weights based on Sharpe ratio
    data['adjusted_btc_weight'] = data['btc_weight'] * (1 + np.tanh(data['sharpe_ratio']))
    data['adjusted_btc_weight'] = np.clip(data['adjusted_btc_weight'], 0.1, 0.9)
    
    # Signal generation based on weight changes
    data['signal'] = 0
    data['weight_change'] = data['adjusted_btc_weight'].diff()
    
    # Rebalance periodically
    rebalance_mask = data.index.to_series().diff().dt.days >= rebalance_frequency
    
    # Buy when weight should increase significantly
    buy_condition = (data['weight_change'] > 0.05) & rebalance_mask
    
    # Sell when weight should decrease significantly
    sell_condition = (data['weight_change'] < -0.05) & rebalance_mask
    
    data.loc[buy_condition, 'signal'] = 1
    data.loc[sell_condition, 'signal'] = -1
    
    data['signal'] = data['signal'].replace(0, np.nan).fillna(method='ffill').fillna(0)
    return data


def black_litterman_strategy(data: pd.DataFrame,
                           confidence_window: int = 40,
                           view_strength: float = 0.5) -> pd.DataFrame:
    """
    Black-Litterman Inspired Strategy
    ブラック・リッターマン戦略
    """
    data = data.copy()
    
    # Market implied returns (simplified)
    data['returns'] = data['close'].pct_change()
    data['market_return'] = data['returns'].rolling(window=confidence_window).mean()
    data['market_volatility'] = data['returns'].rolling(window=confidence_window).std()
    
    # Create "views" based on technical indicators
    # View 1: Momentum view
    data['momentum_view'] = data['close'].pct_change(10)
    
    # View 2: Mean reversion view
    data['price_ma'] = data['close'].rolling(window=20).mean()
    data['mean_reversion_view'] = -(data['close'] - data['price_ma']) / data['price_ma']
    
    # View 3: Volatility view
    data['vol_percentile'] = data['market_volatility'].rolling(window=60).rank(pct=True)
    data['volatility_view'] = np.where(data['vol_percentile'] > 0.8, -0.02, 0.01)  # Negative when high vol
    
    # Confidence in views (based on historical accuracy)
    data['momentum_confidence'] = 1 - abs(data['momentum_view'].rolling(window=20).corr(data['returns'].shift(-1)))
    data['mean_reversion_confidence'] = 1 - abs(data['mean_reversion_view'].rolling(window=20).corr(data['returns'].shift(-1)))
    data['volatility_confidence'] = 0.6  # Fixed confidence
    
    # Combine views with market equilibrium
    data['combined_view'] = (
        data['market_return'] * 0.4 +  # Market equilibrium weight
        data['momentum_view'] * data['momentum_confidence'] * view_strength +
        data['mean_reversion_view'] * data['mean_reversion_confidence'] * view_strength +
        data['volatility_view'] * data['volatility_confidence'] * view_strength
    )
    
    # Expected return estimate
    data['expected_return'] = data['combined_view']
    
    # Risk-adjusted signal
    data['risk_adjusted_signal'] = data['expected_return'] / (data['market_volatility'] + 1e-6)
    
    # Signal generation
    data['signal'] = 0
    
    # Buy when risk-adjusted expected return is high
    buy_condition = data['risk_adjusted_signal'] > 0.5
    
    # Sell when risk-adjusted expected return is low
    sell_condition = data['risk_adjusted_signal'] < -0.5
    
    data.loc[buy_condition, 'signal'] = 1
    data.loc[sell_condition, 'signal'] = -1
    
    data['signal'] = data['signal'].replace(0, np.nan).fillna(method='ffill').fillna(0)
    return data


def copula_strategy(data: pd.DataFrame,
                   reference_window: int = 50,
                   quantile_threshold: float = 0.8) -> pd.DataFrame:
    """
    Copula-based Strategy
    コピュラベース戦略
    """
    data = data.copy()
    
    # Create synthetic correlated series (Bitcoin vs its own lagged version)
    data['returns'] = data['close'].pct_change()
    data['lagged_returns'] = data['returns'].shift(5)
    
    # Empirical CDFs
    def empirical_cdf(series, value):
        return (series <= value).mean()
    
    data['returns_percentile'] = np.nan
    data['lagged_percentile'] = np.nan
    
    for i in range(reference_window, len(data)):
        window_returns = data['returns'].iloc[i-reference_window:i]
        window_lagged = data['lagged_returns'].iloc[i-reference_window:i]
        
        current_return = data['returns'].iloc[i]
        current_lagged = data['lagged_returns'].iloc[i]
        
        if not np.isnan(current_return) and not np.isnan(current_lagged):
            data.iloc[i, data.columns.get_loc('returns_percentile')] = empirical_cdf(window_returns.dropna(), current_return)
            data.iloc[i, data.columns.get_loc('lagged_percentile')] = empirical_cdf(window_lagged.dropna(), current_lagged)
    
    # Copula-based dependence measure
    data['copula_correlation'] = data['returns_percentile'].rolling(window=20).corr(data['lagged_percentile'])
    
    # Tail dependence detection
    data['upper_tail'] = ((data['returns_percentile'] > quantile_threshold) & 
                         (data['lagged_percentile'] > quantile_threshold)).rolling(window=20).mean()
    
    data['lower_tail'] = ((data['returns_percentile'] < (1 - quantile_threshold)) & 
                         (data['lagged_percentile'] < (1 - quantile_threshold))).rolling(window=20).mean()
    
    # Signal generation
    data['signal'] = 0
    
    # Buy when strong positive tail dependence (momentum regime)
    buy_condition = (
        (data['upper_tail'] > 0.3) &
        (data['returns_percentile'] > 0.6)
    )
    
    # Sell when strong negative correlation or lower tail dependence
    sell_condition = (
        (data['lower_tail'] > 0.3) |
        (data['copula_correlation'] < -0.3)
    )
    
    data.loc[buy_condition, 'signal'] = 1
    data.loc[sell_condition, 'signal'] = -1
    
    data['signal'] = data['signal'].replace(0, np.nan).fillna(method='ffill').fillna(0)
    return data


def value_at_risk_strategy(data: pd.DataFrame,
                          var_window: int = 30,
                          confidence_level: float = 0.05,
                          var_threshold: float = 0.03) -> pd.DataFrame:
    """
    Value at Risk Strategy
    バリューアットリスク戦略
    """
    data = data.copy()
    
    # Calculate returns
    data['returns'] = data['close'].pct_change()
    
    # Rolling VaR calculation (Historical method)
    data['var_95'] = data['returns'].rolling(window=var_window).quantile(confidence_level)
    data['var_99'] = data['returns'].rolling(window=var_window).quantile(0.01)
    
    # Expected Shortfall (Conditional VaR)
    def calculate_es(returns_series, confidence_level):
        var = returns_series.quantile(confidence_level)
        return returns_series[returns_series <= var].mean()
    
    data['expected_shortfall'] = data['returns'].rolling(window=var_window).apply(
        lambda x: calculate_es(x, confidence_level), raw=False
    )
    
    # Parametric VaR (assuming normal distribution)
    data['returns_mean'] = data['returns'].rolling(window=var_window).mean()
    data['returns_std'] = data['returns'].rolling(window=var_window).std()
    data['parametric_var'] = data['returns_mean'] + stats.norm.ppf(confidence_level) * data['returns_std']
    
    # VaR-based position sizing
    data['var_position_size'] = np.where(
        data['var_95'].abs() > 0,
        var_threshold / data['var_95'].abs(),
        1.0
    )
    data['var_position_size'] = np.clip(data['var_position_size'], 0.1, 2.0)
    
    # Risk regime detection
    data['high_risk_regime'] = (
        (data['var_95'] < -0.05) |  # High VaR
        (data['expected_shortfall'] < -0.08) |  # High expected shortfall
        (data['returns_std'] > data['returns_std'].rolling(window=60).quantile(0.8))  # High volatility
    )
    
    # Signal generation
    data['signal'] = 0
    
    # In low risk regime: momentum strategy
    buy_low_risk = (
        (~data['high_risk_regime']) &
        (data['returns'].rolling(window=5).sum() > 0.02) &
        (data['var_position_size'] > 0.8)
    )
    
    sell_low_risk = (
        (~data['high_risk_regime']) &
        (data['returns'].rolling(window=5).sum() < -0.02)
    )
    
    # In high risk regime: defensive/contrarian
    buy_high_risk = (
        data['high_risk_regime'] &
        (data['returns'] < data['var_95']) &  # Extreme negative return
        (data['close'] < data['close'].rolling(window=20).mean() * 0.95)  # Oversold
    )
    
    sell_high_risk = (
        data['high_risk_regime'] &
        (data['returns'] > -data['var_95'])  # Extreme positive return in high risk
    )
    
    data.loc[buy_low_risk | buy_high_risk, 'signal'] = 1
    data.loc[sell_low_risk | sell_high_risk, 'signal'] = -1
    
    data['signal'] = data['signal'].replace(0, np.nan).fillna(method='ffill').fillna(0)
    return data


def maximum_diversification_strategy(data: pd.DataFrame,
                                   lookback_window: int = 40,
                                   rebalance_frequency: int = 10) -> pd.DataFrame:
    """
    Maximum Diversification Strategy
    最大多様化戦略
    """
    data = data.copy()
    
    # Create multiple synthetic assets from Bitcoin using different transformations
    data['returns'] = data['close'].pct_change()
    
    # Asset 1: Original Bitcoin
    data['asset1_returns'] = data['returns']
    
    # Asset 2: Bitcoin momentum
    data['asset2_returns'] = data['returns'].rolling(window=5).mean()
    
    # Asset 3: Bitcoin mean reversion
    data['price_ma'] = data['close'].rolling(window=20).mean()
    data['asset3_returns'] = -(data['close'] - data['price_ma']).pct_change()
    
    # Asset 4: Bitcoin volatility
    data['volatility'] = data['returns'].rolling(window=10).std()
    data['asset4_returns'] = -data['volatility'].pct_change()  # Inverse volatility
    
    # Asset 5: Bitcoin volume
    data['volume_normalized'] = data['volume'] / data['volume'].rolling(window=20).mean()
    data['asset5_returns'] = data['volume_normalized'].pct_change()
    
    asset_cols = ['asset1_returns', 'asset2_returns', 'asset3_returns', 'asset4_returns', 'asset5_returns']
    
    # Calculate diversification ratio
    data['diversification_ratio'] = np.nan
    data['optimal_weight'] = np.nan
    
    for i in range(lookback_window, len(data)):
        if i % rebalance_frequency == 0:  # Rebalance periodically
            # Get return matrix for window
            return_matrix = data[asset_cols].iloc[i-lookback_window:i].dropna()
            
            if len(return_matrix) > 10:
                # Calculate covariance matrix
                cov_matrix = return_matrix.cov().values
                volatilities = np.sqrt(np.diag(cov_matrix))
                
                # Maximum diversification optimization (simplified)
                # Weight = inverse volatility, normalized
                inv_vol_weights = 1 / (volatilities + 1e-6)
                inv_vol_weights = inv_vol_weights / np.sum(inv_vol_weights)
                
                # Portfolio volatility
                portfolio_vol = np.sqrt(np.dot(inv_vol_weights, np.dot(cov_matrix, inv_vol_weights)))
                
                # Weighted average of individual volatilities
                weighted_avg_vol = np.dot(inv_vol_weights, volatilities)
                
                # Diversification ratio
                div_ratio = weighted_avg_vol / portfolio_vol
                
                data.iloc[i, data.columns.get_loc('diversification_ratio')] = div_ratio
                data.iloc[i, data.columns.get_loc('optimal_weight')] = inv_vol_weights[0]  # Weight for original Bitcoin
        else:
            # Forward fill
            if i > 0:
                data.iloc[i, data.columns.get_loc('diversification_ratio')] = data.iloc[i-1, data.columns.get_loc('diversification_ratio')]
                data.iloc[i, data.columns.get_loc('optimal_weight')] = data.iloc[i-1, data.columns.get_loc('optimal_weight')]
    
    # Signal generation
    data['signal'] = 0
    data['weight_change'] = data['optimal_weight'].diff()
    
    # Buy when optimal weight increases significantly
    buy_condition = (
        (data['weight_change'] > 0.1) |
        ((data['diversification_ratio'] > 1.5) & (data['optimal_weight'] > 0.4))
    )
    
    # Sell when optimal weight decreases significantly
    sell_condition = (
        (data['weight_change'] < -0.1) |
        ((data['diversification_ratio'] < 1.2) & (data['optimal_weight'] < 0.3))
    )
    
    data.loc[buy_condition, 'signal'] = 1
    data.loc[sell_condition, 'signal'] = -1
    
    data['signal'] = data['signal'].replace(0, np.nan).fillna(method='ffill').fillna(0)
    return data