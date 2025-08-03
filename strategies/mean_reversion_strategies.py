"""
平均回帰ベースの戦略
"""

import pandas as pd
import numpy as np
from typing import Dict, Any


def bollinger_bands_strategy(data: pd.DataFrame,
                           window: int = 20,
                           num_std: float = 2.0) -> pd.DataFrame:
    """
    ボリンジャーバンド戦略
    価格が下部バンドを下回ったら買い、上部バンドを上回ったら売り
    
    Args:
        data: OHLCV価格データ
        window: 移動平均およびボラティリティ計算期間
        num_std: 標準偏差の倍数
        
    Returns:
        pd.DataFrame: シグナル付きデータ
    """
    data = data.copy()
    
    # ボリンジャーバンド計算
    data['bb_middle'] = data['close'].rolling(window=window).mean()
    rolling_std = data['close'].rolling(window=window).std()
    
    data['bb_upper'] = data['bb_middle'] + (rolling_std * num_std)
    data['bb_lower'] = data['bb_middle'] - (rolling_std * num_std)
    
    # バンド幅の計算（ボラティリティ指標）
    data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
    
    # シグナル生成
    data['signal'] = 0
    
    # 価格が下部バンドを下回ったら買い
    buy_condition = data['close'] < data['bb_lower']
    # 価格が上部バンドを上回ったら売り
    sell_condition = data['close'] > data['bb_upper']
    
    data.loc[buy_condition, 'signal'] = 1
    data.loc[sell_condition, 'signal'] = -1
    
    # シグナル変化点のみを抽出
    data['position'] = data['signal'].diff()
    data['signal'] = np.where(data['position'] != 0, data['signal'], np.nan)
    
    return data


def mean_reversion_strategy(data: pd.DataFrame,
                          lookback_window: int = 20,
                          threshold_std: float = 1.5) -> pd.DataFrame:
    """
    シンプル平均回帰戦略
    価格が移動平均から一定の標準偏差を超えて乖離したら逆張り
    
    Args:
        data: OHLCV価格データ
        lookback_window: 移動平均計算期間
        threshold_std: エントリー閾値（標準偏差の倍数）
        
    Returns:
        pd.DataFrame: シグナル付きデータ
    """
    data = data.copy()
    
    # 移動平均と標準偏差計算
    data['sma'] = data['close'].rolling(window=lookback_window).mean()
    data['rolling_std'] = data['close'].rolling(window=lookback_window).std()
    
    # 価格の標準化（Zスコア）
    data['z_score'] = (data['close'] - data['sma']) / data['rolling_std']
    
    # シグナル生成
    data['signal'] = 0
    
    # 価格が平均を大きく下回ったら買い（平均回帰を期待）
    buy_condition = data['z_score'] < -threshold_std
    # 価格が平均を大きく上回ったら売り（平均回帰を期待）
    sell_condition = data['z_score'] > threshold_std
    
    data.loc[buy_condition, 'signal'] = 1
    data.loc[sell_condition, 'signal'] = -1
    
    # シグナル変化点のみを抽出
    data['position'] = data['signal'].diff()
    data['signal'] = np.where(data['position'] != 0, data['signal'], np.nan)
    
    return data


def pairs_trading_strategy(data: pd.DataFrame,
                         reference_data: pd.DataFrame,
                         lookback_window: int = 30,
                         entry_threshold: float = 2.0,
                         exit_threshold: float = 0.5) -> pd.DataFrame:
    """
    ペアトレーディング戦略（参考用）
    2つの資産の価格差の平均回帰を利用
    
    Args:
        data: メイン資産のOHLCV価格データ
        reference_data: 参照資産のOHLCV価格データ
        lookback_window: 統計計算期間
        entry_threshold: エントリー閾値
        exit_threshold: エグジット閾値
        
    Returns:
        pd.DataFrame: シグナル付きデータ
    """
    data = data.copy()
    
    # 共通の日付範囲に調整
    common_dates = data.index.intersection(reference_data.index)
    data = data.loc[common_dates]
    reference_data = reference_data.loc[common_dates]
    
    # 価格比率計算
    data['price_ratio'] = data['close'] / reference_data['close']
    
    # 移動平均と標準偏差
    data['ratio_sma'] = data['price_ratio'].rolling(window=lookback_window).mean()
    data['ratio_std'] = data['price_ratio'].rolling(window=lookback_window).std()
    
    # 標準化
    data['ratio_z_score'] = (data['price_ratio'] - data['ratio_sma']) / data['ratio_std']
    
    # シグナル生成
    data['signal'] = 0
    
    # 比率が大きく下がったら買い（メイン資産が相対的に安い）
    buy_condition = data['ratio_z_score'] < -entry_threshold
    # 比率が大きく上がったら売り（メイン資産が相対的に高い）
    sell_condition = data['ratio_z_score'] > entry_threshold
    # 中央値付近で決済
    exit_condition = abs(data['ratio_z_score']) < exit_threshold
    
    data.loc[buy_condition, 'signal'] = 1
    data.loc[sell_condition, 'signal'] = -1
    data.loc[exit_condition, 'signal'] = 0
    
    # シグナル変化点のみを抽出
    data['position'] = data['signal'].diff()
    data['signal'] = np.where(data['position'] != 0, data['signal'], np.nan)
    
    return data


def donchian_channel_strategy(data: pd.DataFrame,
                            channel_window: int = 20) -> pd.DataFrame:
    """
    ドンチャンチャネル戦略
    価格が下部チャネルを下回ったら買い、上部チャネルを上回ったら売り
    
    Args:
        data: OHLCV価格データ
        channel_window: チャネル計算期間
        
    Returns:
        pd.DataFrame: シグナル付きデータ
    """
    data = data.copy()
    
    # ドンチャンチャネル計算
    data['donchian_upper'] = data['high'].rolling(window=channel_window).max()
    data['donchian_lower'] = data['low'].rolling(window=channel_window).min()
    data['donchian_middle'] = (data['donchian_upper'] + data['donchian_lower']) / 2
    
    # シグナル生成
    data['signal'] = 0
    
    # 価格が下部チャネルを下回ったら買い
    buy_condition = data['close'] < data['donchian_lower']
    # 価格が上部チャネルを上回ったら売り
    sell_condition = data['close'] > data['donchian_upper']
    
    data.loc[buy_condition, 'signal'] = 1
    data.loc[sell_condition, 'signal'] = -1
    
    # シグナル変化点のみを抽出
    data['position'] = data['signal'].diff()
    data['signal'] = np.where(data['position'] != 0, data['signal'], np.nan)
    
    return data