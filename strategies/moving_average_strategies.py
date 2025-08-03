"""
移動平均ベースの戦略
"""

import pandas as pd
import numpy as np
from typing import Dict, Any


def simple_moving_average_crossover(data: pd.DataFrame, 
                                  short_window: int = 20, 
                                  long_window: int = 50) -> pd.DataFrame:
    """
    シンプル移動平均クロスオーバー戦略
    短期移動平均が長期移動平均を上抜けで買い、下抜けで売り
    
    Args:
        data: OHLCV価格データ
        short_window: 短期移動平均期間
        long_window: 長期移動平均期間
        
    Returns:
        pd.DataFrame: シグナル付きデータ
    """
    data = data.copy()
    
    # 移動平均計算
    data[f'sma_{short_window}'] = data['close'].rolling(window=short_window).mean()
    data[f'sma_{long_window}'] = data['close'].rolling(window=long_window).mean()
    
    # シグナル生成
    data['signal'] = 0
    data['signal'][long_window:] = np.where(
        data[f'sma_{short_window}'][long_window:] > data[f'sma_{long_window}'][long_window:], 1, -1
    )
    
    # シグナル変化点のみを抽出
    data['position'] = data['signal'].diff()
    data['signal'] = np.where(data['position'] != 0, data['signal'], np.nan)
    
    return data


def exponential_moving_average_crossover(data: pd.DataFrame,
                                       short_window: int = 12,
                                       long_window: int = 26) -> pd.DataFrame:
    """
    指数移動平均クロスオーバー戦略
    
    Args:
        data: OHLCV価格データ
        short_window: 短期EMA期間
        long_window: 長期EMA期間
        
    Returns:
        pd.DataFrame: シグナル付きデータ
    """
    data = data.copy()
    
    # EMA計算
    data[f'ema_{short_window}'] = data['close'].ewm(span=short_window).mean()
    data[f'ema_{long_window}'] = data['close'].ewm(span=long_window).mean()
    
    # シグナル生成
    data['signal'] = 0
    data['signal'][long_window:] = np.where(
        data[f'ema_{short_window}'][long_window:] > data[f'ema_{long_window}'][long_window:], 1, -1
    )
    
    # シグナル変化点のみを抽出
    data['position'] = data['signal'].diff()
    data['signal'] = np.where(data['position'] != 0, data['signal'], np.nan)
    
    return data


def triple_moving_average_strategy(data: pd.DataFrame,
                                 short_window: int = 10,
                                 medium_window: int = 20,
                                 long_window: int = 50) -> pd.DataFrame:
    """
    トリプル移動平均戦略
    短期 > 中期 > 長期の順序で買い、逆順で売り
    
    Args:
        data: OHLCV価格データ
        short_window: 短期移動平均期間
        medium_window: 中期移動平均期間
        long_window: 長期移動平均期間
        
    Returns:
        pd.DataFrame: シグナル付きデータ
    """
    data = data.copy()
    
    # 移動平均計算
    data[f'sma_{short_window}'] = data['close'].rolling(window=short_window).mean()
    data[f'sma_{medium_window}'] = data['close'].rolling(window=medium_window).mean()
    data[f'sma_{long_window}'] = data['close'].rolling(window=long_window).mean()
    
    # 強気条件: 短期 > 中期 > 長期
    bullish_condition = (
        (data[f'sma_{short_window}'] > data[f'sma_{medium_window}']) &
        (data[f'sma_{medium_window}'] > data[f'sma_{long_window}'])
    )
    
    # 弱気条件: 短期 < 中期 < 長期
    bearish_condition = (
        (data[f'sma_{short_window}'] < data[f'sma_{medium_window}']) &
        (data[f'sma_{medium_window}'] < data[f'sma_{long_window}'])
    )
    
    # シグナル生成
    data['signal'] = 0
    data.loc[bullish_condition, 'signal'] = 1
    data.loc[bearish_condition, 'signal'] = -1
    
    # シグナル変化点のみを抽出
    data['position'] = data['signal'].diff()
    data['signal'] = np.where(data['position'] != 0, data['signal'], np.nan)
    
    return data


def moving_average_envelope_strategy(data: pd.DataFrame,
                                   ma_window: int = 20,
                                   envelope_pct: float = 2.0) -> pd.DataFrame:
    """
    移動平均エンベロープ戦略
    価格が下部バンドを下回ったら買い、上部バンドを上回ったら売り
    
    Args:
        data: OHLCV価格データ
        ma_window: 移動平均期間
        envelope_pct: エンベロープ幅（%）
        
    Returns:
        pd.DataFrame: シグナル付きデータ
    """
    data = data.copy()
    
    # 移動平均計算
    data['sma'] = data['close'].rolling(window=ma_window).mean()
    
    # エンベロープ計算
    envelope_multiplier = envelope_pct / 100
    data['upper_band'] = data['sma'] * (1 + envelope_multiplier)
    data['lower_band'] = data['sma'] * (1 - envelope_multiplier)
    
    # シグナル生成
    data['signal'] = 0
    
    # 価格が下部バンドを下回ったら買い
    buy_condition = data['close'] < data['lower_band']
    # 価格が上部バンドを上回ったら売り
    sell_condition = data['close'] > data['upper_band']
    
    data.loc[buy_condition, 'signal'] = 1
    data.loc[sell_condition, 'signal'] = -1
    
    # シグナル変化点のみを抽出
    data['position'] = data['signal'].diff()
    data['signal'] = np.where(data['position'] != 0, data['signal'], np.nan)
    
    return data