"""
モメンタムベースの戦略
"""

import pandas as pd
import numpy as np
from typing import Dict, Any


def rsi_strategy(data: pd.DataFrame,
                rsi_window: int = 14,
                oversold_threshold: float = 30,
                overbought_threshold: float = 70) -> pd.DataFrame:
    """
    RSI（相対力指数）戦略
    RSIが売られすぎ水準で買い、買われすぎ水準で売り
    
    Args:
        data: OHLCV価格データ
        rsi_window: RSI計算期間
        oversold_threshold: 売られすぎ閾値
        overbought_threshold: 買われすぎ閾値
        
    Returns:
        pd.DataFrame: シグナル付きデータ
    """
    data = data.copy()
    
    # RSI計算
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_window).mean()
    
    rs = gain / loss
    data['rsi'] = 100 - (100 / (1 + rs))
    
    # シグナル生成
    data['signal'] = 0
    
    # RSIが売られすぎ水準で買い
    buy_condition = data['rsi'] < oversold_threshold
    # RSIが買われすぎ水準で売り
    sell_condition = data['rsi'] > overbought_threshold
    
    data.loc[buy_condition, 'signal'] = 1
    data.loc[sell_condition, 'signal'] = -1
    
    # シグナル変化点のみを抽出
    data['position'] = data['signal'].diff()
    data['signal'] = np.where(data['position'] != 0, data['signal'], np.nan)
    
    return data


def macd_strategy(data: pd.DataFrame,
                 fast_period: int = 12,
                 slow_period: int = 26,
                 signal_period: int = 9) -> pd.DataFrame:
    """
    MACD戦略
    MACDラインがシグナルラインを上抜けで買い、下抜けで売り
    
    Args:
        data: OHLCV価格データ
        fast_period: 高速EMA期間
        slow_period: 低速EMA期間
        signal_period: シグナルライン期間
        
    Returns:
        pd.DataFrame: シグナル付きデータ
    """
    data = data.copy()
    
    # MACD計算
    ema_fast = data['close'].ewm(span=fast_period).mean()
    ema_slow = data['close'].ewm(span=slow_period).mean()
    
    data['macd'] = ema_fast - ema_slow
    data['macd_signal'] = data['macd'].ewm(span=signal_period).mean()
    data['macd_histogram'] = data['macd'] - data['macd_signal']
    
    # シグナル生成
    data['signal'] = 0
    data['signal'][slow_period:] = np.where(
        data['macd'][slow_period:] > data['macd_signal'][slow_period:], 1, -1
    )
    
    # シグナル変化点のみを抽出
    data['position'] = data['signal'].diff()
    data['signal'] = np.where(data['position'] != 0, data['signal'], np.nan)
    
    return data


def momentum_strategy(data: pd.DataFrame,
                     momentum_window: int = 10,
                     threshold: float = 0.02) -> pd.DataFrame:
    """
    プライスモメンタム戦略
    一定期間のリターンが閾値を超えたら同方向にポジション
    
    Args:
        data: OHLCV価格データ
        momentum_window: モメンタム計算期間
        threshold: エントリー閾値
        
    Returns:
        pd.DataFrame: シグナル付きデータ
    """
    data = data.copy()
    
    # モメンタム計算（期間リターン）
    data['momentum'] = data['close'].pct_change(periods=momentum_window)
    
    # シグナル生成
    data['signal'] = 0
    
    # 正のモメンタムが閾値を超えたら買い
    buy_condition = data['momentum'] > threshold
    # 負のモメンタムが閾値を下回ったら売り
    sell_condition = data['momentum'] < -threshold
    
    data.loc[buy_condition, 'signal'] = 1
    data.loc[sell_condition, 'signal'] = -1
    
    # シグナル変化点のみを抽出
    data['position'] = data['signal'].diff()
    data['signal'] = np.where(data['position'] != 0, data['signal'], np.nan)
    
    return data


def stochastic_strategy(data: pd.DataFrame,
                       k_window: int = 14,
                       d_window: int = 3,
                       oversold_threshold: float = 20,
                       overbought_threshold: float = 80) -> pd.DataFrame:
    """
    ストキャスティクス戦略
    %Kが売られすぎ水準で買い、買われすぎ水準で売り
    
    Args:
        data: OHLCV価格データ
        k_window: %K計算期間
        d_window: %D計算期間（%Kの移動平均）
        oversold_threshold: 売られすぎ閾値
        overbought_threshold: 買われすぎ閾値
        
    Returns:
        pd.DataFrame: シグナル付きデータ
    """
    data = data.copy()
    
    # ストキャスティクス計算
    lowest_low = data['low'].rolling(window=k_window).min()
    highest_high = data['high'].rolling(window=k_window).max()
    
    data['stoch_k'] = 100 * (data['close'] - lowest_low) / (highest_high - lowest_low)
    data['stoch_d'] = data['stoch_k'].rolling(window=d_window).mean()
    
    # シグナル生成
    data['signal'] = 0
    
    # %Kが売られすぎ水準で買い
    buy_condition = data['stoch_k'] < oversold_threshold
    # %Kが買われすぎ水準で売り
    sell_condition = data['stoch_k'] > overbought_threshold
    
    data.loc[buy_condition, 'signal'] = 1
    data.loc[sell_condition, 'signal'] = -1
    
    # シグナル変化点のみを抽出
    data['position'] = data['signal'].diff()
    data['signal'] = np.where(data['position'] != 0, data['signal'], np.nan)
    
    return data