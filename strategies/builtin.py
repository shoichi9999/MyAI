"""組み込み戦略セット

AI探索の初期戦略テンプレートとして使用される。
各戦略はOptunaパラメータ空間を定義し、自動最適化に対応する。
"""

import pandas as pd
import numpy as np
from strategies.base import (
    BaseStrategy, add_indicator_sma, add_indicator_ema,
    add_indicator_rsi, add_indicator_bollinger, add_indicator_macd,
    add_indicator_atr, add_indicator_stochastic,
)


class SMA_Cross(BaseStrategy):
    """SMAゴールデンクロス / デッドクロス戦略"""

    def __init__(self, params=None):
        super().__init__(params or self.default_params())
        self.name = "SMA_Cross"

    @classmethod
    def default_params(cls):
        return {"fast_period": 10, "slow_period": 50}

    @classmethod
    def params_space(cls, trial):
        fast = trial.suggest_int("fast_period", 3, 50)
        slow = trial.suggest_int("slow_period", 20, 200)
        return {"fast_period": fast, "slow_period": slow}

    def generate_signals(self, df):
        p = self.params
        df["sma_fast"] = add_indicator_sma(df, p["fast_period"])
        df["sma_slow"] = add_indicator_sma(df, p["slow_period"])
        df["signal"] = 0
        df.loc[df["sma_fast"] > df["sma_slow"], "signal"] = 1
        df.loc[df["sma_fast"] < df["sma_slow"], "signal"] = -1
        # クロスポイントのみシグナル
        df["signal"] = df["signal"].diff().clip(-1, 1).fillna(0).astype(int)
        return df


class EMA_Cross(BaseStrategy):
    """EMAクロスオーバー戦略"""

    def __init__(self, params=None):
        super().__init__(params or self.default_params())
        self.name = "EMA_Cross"

    @classmethod
    def default_params(cls):
        return {"fast_period": 9, "slow_period": 21}

    @classmethod
    def params_space(cls, trial):
        fast = trial.suggest_int("fast_period", 3, 50)
        slow = trial.suggest_int("slow_period", 15, 200)
        return {"fast_period": fast, "slow_period": slow}

    def generate_signals(self, df):
        p = self.params
        df["ema_fast"] = add_indicator_ema(df, p["fast_period"])
        df["ema_slow"] = add_indicator_ema(df, p["slow_period"])
        df["signal"] = 0
        df.loc[df["ema_fast"] > df["ema_slow"], "signal"] = 1
        df.loc[df["ema_fast"] < df["ema_slow"], "signal"] = -1
        df["signal"] = df["signal"].diff().clip(-1, 1).fillna(0).astype(int)
        return df


class RSI_MeanReversion(BaseStrategy):
    """RSI平均回帰戦略 - 買われすぎ/売られすぎで逆張り"""

    def __init__(self, params=None):
        super().__init__(params or self.default_params())
        self.name = "RSI_MeanReversion"

    @classmethod
    def default_params(cls):
        return {"rsi_period": 14, "oversold": 30, "overbought": 70}

    @classmethod
    def params_space(cls, trial):
        return {
            "rsi_period": trial.suggest_int("rsi_period", 5, 50),
            "oversold": trial.suggest_int("oversold", 10, 40),
            "overbought": trial.suggest_int("overbought", 60, 90),
        }

    def generate_signals(self, df):
        p = self.params
        df["rsi"] = add_indicator_rsi(df, p["rsi_period"])
        df["signal"] = 0
        df.loc[df["rsi"] < p["oversold"], "signal"] = 1   # 売られすぎ → 買い
        df.loc[df["rsi"] > p["overbought"], "signal"] = -1 # 買われすぎ → 売り
        return df


class BollingerBand_Breakout(BaseStrategy):
    """ボリンジャーバンドブレイクアウト戦略"""

    def __init__(self, params=None):
        super().__init__(params or self.default_params())
        self.name = "BollingerBand_Breakout"

    @classmethod
    def default_params(cls):
        return {"bb_period": 20, "bb_std": 2.0}

    @classmethod
    def params_space(cls, trial):
        return {
            "bb_period": trial.suggest_int("bb_period", 10, 50),
            "bb_std": trial.suggest_float("bb_std", 1.0, 3.0, step=0.1),
        }

    def generate_signals(self, df):
        p = self.params
        mid, upper, lower = add_indicator_bollinger(df, p["bb_period"], p["bb_std"])
        df["bb_mid"] = mid
        df["bb_upper"] = upper
        df["bb_lower"] = lower
        df["signal"] = 0
        df.loc[df["close"] > df["bb_upper"], "signal"] = 1   # 上抜け → ロング
        df.loc[df["close"] < df["bb_lower"], "signal"] = -1  # 下抜け → ショート
        return df


class BollingerBand_MeanReversion(BaseStrategy):
    """ボリンジャーバンド平均回帰戦略"""

    def __init__(self, params=None):
        super().__init__(params or self.default_params())
        self.name = "BollingerBand_MeanReversion"

    @classmethod
    def default_params(cls):
        return {"bb_period": 20, "bb_std": 2.0}

    @classmethod
    def params_space(cls, trial):
        return {
            "bb_period": trial.suggest_int("bb_period", 10, 50),
            "bb_std": trial.suggest_float("bb_std", 1.5, 3.5, step=0.1),
        }

    def generate_signals(self, df):
        p = self.params
        mid, upper, lower = add_indicator_bollinger(df, p["bb_period"], p["bb_std"])
        df["bb_mid"] = mid
        df["bb_upper"] = upper
        df["bb_lower"] = lower
        df["signal"] = 0
        # 下バンド到達 → 買い (反発期待)
        df.loc[df["close"] < df["bb_lower"], "signal"] = 1
        # 上バンド到達 → 売り (反落期待)
        df.loc[df["close"] > df["bb_upper"], "signal"] = -1
        return df


class MACD_Strategy(BaseStrategy):
    """MACDクロスオーバー戦略"""

    def __init__(self, params=None):
        super().__init__(params or self.default_params())
        self.name = "MACD_Strategy"

    @classmethod
    def default_params(cls):
        return {"fast": 12, "slow": 26, "signal": 9}

    @classmethod
    def params_space(cls, trial):
        return {
            "fast": trial.suggest_int("fast", 5, 20),
            "slow": trial.suggest_int("slow", 20, 50),
            "signal": trial.suggest_int("signal", 5, 15),
        }

    def generate_signals(self, df):
        p = self.params
        macd_line, signal_line, histogram = add_indicator_macd(
            df, p["fast"], p["slow"], p["signal"]
        )
        df["macd"] = macd_line
        df["macd_signal"] = signal_line
        df["macd_hist"] = histogram

        df["signal"] = 0
        # MACDがシグナルを上抜け → 買い
        prev_hist = histogram.shift(1)
        df.loc[(histogram > 0) & (prev_hist <= 0), "signal"] = 1
        # MACDがシグナルを下抜け → 売り
        df.loc[(histogram < 0) & (prev_hist >= 0), "signal"] = -1
        return df


class Stochastic_Strategy(BaseStrategy):
    """ストキャスティクス戦略"""

    def __init__(self, params=None):
        super().__init__(params or self.default_params())
        self.name = "Stochastic_Strategy"

    @classmethod
    def default_params(cls):
        return {"k_period": 14, "d_period": 3, "oversold": 20, "overbought": 80}

    @classmethod
    def params_space(cls, trial):
        return {
            "k_period": trial.suggest_int("k_period", 5, 30),
            "d_period": trial.suggest_int("d_period", 2, 7),
            "oversold": trial.suggest_int("oversold", 10, 30),
            "overbought": trial.suggest_int("overbought", 70, 90),
        }

    def generate_signals(self, df):
        p = self.params
        k, d = add_indicator_stochastic(df, p["k_period"], p["d_period"])
        df["stoch_k"] = k
        df["stoch_d"] = d
        df["signal"] = 0
        # %Kが売られすぎから%Dを上抜け → 買い
        df.loc[(k < p["oversold"]) & (k > d), "signal"] = 1
        # %Kが買われすぎから%Dを下抜け → 売り
        df.loc[(k > p["overbought"]) & (k < d), "signal"] = -1
        return df


class Triple_EMA(BaseStrategy):
    """トリプルEMA戦略 (短期・中期・長期の3本で判断)"""

    def __init__(self, params=None):
        super().__init__(params or self.default_params())
        self.name = "Triple_EMA"

    @classmethod
    def default_params(cls):
        return {"fast": 5, "mid": 21, "slow": 55}

    @classmethod
    def params_space(cls, trial):
        return {
            "fast": trial.suggest_int("fast", 3, 15),
            "mid": trial.suggest_int("mid", 15, 40),
            "slow": trial.suggest_int("slow", 40, 100),
        }

    def generate_signals(self, df):
        p = self.params
        df["ema_fast"] = add_indicator_ema(df, p["fast"])
        df["ema_mid"] = add_indicator_ema(df, p["mid"])
        df["ema_slow"] = add_indicator_ema(df, p["slow"])
        df["signal"] = 0
        # 全EMAが上昇順 → ロング
        long_cond = (df["ema_fast"] > df["ema_mid"]) & (df["ema_mid"] > df["ema_slow"])
        short_cond = (df["ema_fast"] < df["ema_mid"]) & (df["ema_mid"] < df["ema_slow"])
        df.loc[long_cond, "signal"] = 1
        df.loc[short_cond, "signal"] = -1
        # 変化点のみシグナル
        df["signal"] = df["signal"].diff().clip(-1, 1).fillna(0).astype(int)
        return df


class RSI_MACD_Combo(BaseStrategy):
    """RSI + MACD 組み合わせ戦略"""

    def __init__(self, params=None):
        super().__init__(params or self.default_params())
        self.name = "RSI_MACD_Combo"

    @classmethod
    def default_params(cls):
        return {
            "rsi_period": 14, "rsi_threshold": 50,
            "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
        }

    @classmethod
    def params_space(cls, trial):
        return {
            "rsi_period": trial.suggest_int("rsi_period", 7, 30),
            "rsi_threshold": trial.suggest_int("rsi_threshold", 40, 60),
            "macd_fast": trial.suggest_int("macd_fast", 5, 20),
            "macd_slow": trial.suggest_int("macd_slow", 20, 50),
            "macd_signal": trial.suggest_int("macd_signal", 5, 15),
        }

    def generate_signals(self, df):
        p = self.params
        df["rsi"] = add_indicator_rsi(df, p["rsi_period"])
        macd_line, signal_line, histogram = add_indicator_macd(
            df, p["macd_fast"], p["macd_slow"], p["macd_signal"]
        )
        df["macd_hist"] = histogram
        prev_hist = histogram.shift(1)

        df["signal"] = 0
        # RSIが閾値以上 かつ MACDクロスアップ → 買い
        buy = (df["rsi"] > p["rsi_threshold"]) & (histogram > 0) & (prev_hist <= 0)
        sell = (df["rsi"] < p["rsi_threshold"]) & (histogram < 0) & (prev_hist >= 0)
        df.loc[buy, "signal"] = 1
        df.loc[sell, "signal"] = -1
        return df


class ATR_Breakout(BaseStrategy):
    """ATRブレイクアウト戦略 - ボラティリティ拡大時にエントリー"""

    def __init__(self, params=None):
        super().__init__(params or self.default_params())
        self.name = "ATR_Breakout"

    @classmethod
    def default_params(cls):
        return {"atr_period": 14, "atr_multiplier": 1.5, "lookback": 20}

    @classmethod
    def params_space(cls, trial):
        return {
            "atr_period": trial.suggest_int("atr_period", 7, 30),
            "atr_multiplier": trial.suggest_float("atr_multiplier", 0.5, 3.0, step=0.1),
            "lookback": trial.suggest_int("lookback", 10, 50),
        }

    def generate_signals(self, df):
        p = self.params
        df["atr"] = add_indicator_atr(df, p["atr_period"])
        df["highest"] = df["high"].rolling(window=p["lookback"]).max()
        df["lowest"] = df["low"].rolling(window=p["lookback"]).min()
        threshold = df["atr"] * p["atr_multiplier"]

        df["signal"] = 0
        # 高値ブレイク → ロング
        df.loc[df["close"] > df["highest"].shift(1) + threshold, "signal"] = 1
        # 安値ブレイク → ショート
        df.loc[df["close"] < df["lowest"].shift(1) - threshold, "signal"] = -1
        return df


# CompositeStrategyも利用可能にする
from strategies.composer import CompositeStrategy

# 全戦略の登録
ALL_STRATEGIES = {
    "SMA_Cross": SMA_Cross,
    "EMA_Cross": EMA_Cross,
    "RSI_MeanReversion": RSI_MeanReversion,
    "BollingerBand_Breakout": BollingerBand_Breakout,
    "BollingerBand_MeanReversion": BollingerBand_MeanReversion,
    "MACD_Strategy": MACD_Strategy,
    "Stochastic_Strategy": Stochastic_Strategy,
    "Triple_EMA": Triple_EMA,
    "RSI_MACD_Combo": RSI_MACD_Combo,
    "ATR_Breakout": ATR_Breakout,
    "Composite": CompositeStrategy,
}
