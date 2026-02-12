"""戦略の基底クラスと共通インターフェース"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import pandas as pd
import numpy as np


@dataclass
class Signal:
    """トレードシグナル"""
    BUY = 1
    SELL = -1
    HOLD = 0


@dataclass
class Trade:
    """個別トレード記録"""
    entry_time: object
    exit_time: object
    direction: int          # 1=long, -1=short
    entry_price: float
    exit_price: float
    size: float             # ポジションサイズ(数量)
    pnl: float = 0.0       # 損益(手数料込み)
    pnl_pct: float = 0.0   # 損益率


@dataclass
class StrategyParams:
    """戦略パラメータの基底"""
    name: str = "base"


class BaseStrategy(ABC):
    """全戦略の基底クラス

    サブクラスは以下を実装する:
    - params_space(): Optunaのトライアルからパラメータを生成
    - generate_signals(): DataFrameにシグナル列を追加
    """

    def __init__(self, params: dict = None):
        self.params = params or {}
        self.name = self.__class__.__name__

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """OHLCVデータからシグナルを生成する。

        Args:
            df: timestamp, open, high, low, close, volume を含むDataFrame

        Returns:
            'signal' 列が追加されたDataFrame
            signal: 1=買い, -1=売り, 0=何もしない
        """
        pass

    @classmethod
    @abstractmethod
    def params_space(cls, trial) -> dict:
        """Optunaのトライアルオブジェクトからパラメータ空間を定義する。

        Args:
            trial: optuna.trial.Trial

        Returns:
            dict of parameters
        """
        pass

    @classmethod
    def default_params(cls) -> dict:
        """デフォルトパラメータを返す"""
        return {}

    def __repr__(self):
        return f"{self.name}(params={self.params})"


def add_indicator_sma(df: pd.DataFrame, period: int, column: str = "close") -> pd.Series:
    """単純移動平均"""
    return df[column].rolling(window=period).mean()


def add_indicator_ema(df: pd.DataFrame, period: int, column: str = "close") -> pd.Series:
    """指数移動平均"""
    return df[column].ewm(span=period, adjust=False).mean()


def add_indicator_rsi(df: pd.DataFrame, period: int = 14, column: str = "close") -> pd.Series:
    """RSI (Relative Strength Index)"""
    delta = df[column].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def add_indicator_bollinger(df: pd.DataFrame, period: int = 20,
                            std_dev: float = 2.0, column: str = "close"):
    """ボリンジャーバンド (middle, upper, lower) を返す"""
    middle = df[column].rolling(window=period).mean()
    std = df[column].rolling(window=period).std()
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    return middle, upper, lower


def add_indicator_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26,
                       signal: int = 9, column: str = "close"):
    """MACD (macd_line, signal_line, histogram) を返す"""
    ema_fast = df[column].ewm(span=fast, adjust=False).mean()
    ema_slow = df[column].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def add_indicator_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR (Average True Range)"""
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr


def add_indicator_stochastic(df: pd.DataFrame, k_period: int = 14,
                              d_period: int = 3):
    """%K と %D を返す"""
    low_min = df["low"].rolling(window=k_period).min()
    high_max = df["high"].rolling(window=k_period).max()
    k = 100 * (df["close"] - low_min) / (high_max - low_min)
    d = k.rolling(window=d_period).mean()
    return k, d
