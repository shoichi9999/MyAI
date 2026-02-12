"""シグナル合成エンジン — 戦略を自動生成する

固定戦略ではなく、インジケータ条件をブロックとして組み合わせ、
構造そのものを探索空間に含めて最適化する。

アーキテクチャ:
    Condition(原子条件) → Rule(AND結合) → CompositeStrategy(BUY/SELL各ルール)

探索空間の例:
    BUY: RSI(14) < 30 AND EMA(9) > EMA(21)
    SELL: RSI(14) > 70 AND MACD_hist < 0

    → インジケータの種類、パラメータ、閾値、比較演算子、
      条件数(1~4)、AND/OR結合をすべてOptunaが探索する。
"""

import pandas as pd
import numpy as np
from strategies.base import (
    BaseStrategy,
    add_indicator_sma, add_indicator_ema, add_indicator_rsi,
    add_indicator_bollinger, add_indicator_macd,
    add_indicator_atr, add_indicator_stochastic,
)


# ─── 条件ブロック定義 ───────────────────────────────

CONDITION_TYPES = [
    "rsi_threshold",        # RSI < or > 閾値
    "ema_cross",            # EMA短期 vs EMA長期
    "sma_cross",            # SMA短期 vs SMA長期
    "bb_position",          # close vs ボリンジャーバンド
    "macd_hist_sign",       # MACDヒストグラム正負
    "macd_cross",           # MACDライン vs シグナルライン
    "stoch_threshold",      # ストキャスティクス %K 閾値
    "stoch_cross",          # %K vs %D クロス
    "atr_breakout",         # ATRブレイクアウト
    "price_vs_sma",         # close vs SMA
    "price_vs_ema",         # close vs EMA
    "volume_spike",         # 出来高が平均の N倍以上
    "price_momentum",       # N期間のリターン > 閾値
    "candle_body",          # ローソク足実体サイズ (陽線/陰線)
]


def compute_condition(df: pd.DataFrame, cond_type: str, params: dict) -> pd.Series:
    """条件ブロックを評価し、True/False の Series を返す。

    Args:
        df: OHLCVデータ
        cond_type: CONDITION_TYPES の一つ
        params: 条件パラメータ (Optunaで決定)

    Returns:
        pd.Series[bool]
    """
    if cond_type == "rsi_threshold":
        rsi = add_indicator_rsi(df, params["period"])
        if params["direction"] == "below":
            return rsi < params["threshold"]
        else:
            return rsi > params["threshold"]

    elif cond_type == "ema_cross":
        fast = add_indicator_ema(df, params["fast_period"])
        slow = add_indicator_ema(df, params["slow_period"])
        if params["direction"] == "above":
            return fast > slow
        else:
            return fast < slow

    elif cond_type == "sma_cross":
        fast = add_indicator_sma(df, params["fast_period"])
        slow = add_indicator_sma(df, params["slow_period"])
        if params["direction"] == "above":
            return fast > slow
        else:
            return fast < slow

    elif cond_type == "bb_position":
        _, upper, lower = add_indicator_bollinger(
            df, params["period"], params["std_dev"]
        )
        if params["direction"] == "above_upper":
            return df["close"] > upper
        elif params["direction"] == "below_lower":
            return df["close"] < lower
        elif params["direction"] == "above_mid":
            mid, _, _ = add_indicator_bollinger(df, params["period"], params["std_dev"])
            return df["close"] > mid
        else:  # below_mid
            mid, _, _ = add_indicator_bollinger(df, params["period"], params["std_dev"])
            return df["close"] < mid

    elif cond_type == "macd_hist_sign":
        _, _, hist = add_indicator_macd(
            df, params["fast"], params["slow"], params["signal_period"]
        )
        if params["direction"] == "positive":
            return hist > 0
        else:
            return hist < 0

    elif cond_type == "macd_cross":
        macd_line, signal_line, _ = add_indicator_macd(
            df, params["fast"], params["slow"], params["signal_period"]
        )
        if params["direction"] == "above":
            return macd_line > signal_line
        else:
            return macd_line < signal_line

    elif cond_type == "stoch_threshold":
        k, _ = add_indicator_stochastic(df, params["k_period"], params.get("d_period", 3))
        if params["direction"] == "below":
            return k < params["threshold"]
        else:
            return k > params["threshold"]

    elif cond_type == "stoch_cross":
        k, d = add_indicator_stochastic(df, params["k_period"], params.get("d_period", 3))
        if params["direction"] == "above":
            return k > d
        else:
            return k < d

    elif cond_type == "atr_breakout":
        atr = add_indicator_atr(df, params["period"])
        lookback = params["lookback"]
        threshold = atr * params["multiplier"]
        if params["direction"] == "up":
            highest = df["high"].rolling(window=lookback).max().shift(1)
            return df["close"] > highest + threshold
        else:
            lowest = df["low"].rolling(window=lookback).min().shift(1)
            return df["close"] < lowest - threshold

    elif cond_type == "price_vs_sma":
        sma = add_indicator_sma(df, params["period"])
        if params["direction"] == "above":
            return df["close"] > sma
        else:
            return df["close"] < sma

    elif cond_type == "price_vs_ema":
        ema = add_indicator_ema(df, params["period"])
        if params["direction"] == "above":
            return df["close"] > ema
        else:
            return df["close"] < ema

    elif cond_type == "volume_spike":
        vol_ma = df["volume"].rolling(window=params["period"]).mean()
        return df["volume"] > vol_ma * params["multiplier"]

    elif cond_type == "price_momentum":
        ret = df["close"].pct_change(periods=params["period"])
        if params["direction"] == "positive":
            return ret > params["threshold"]
        else:
            return ret < -params["threshold"]

    elif cond_type == "candle_body":
        body = (df["close"] - df["open"]) / df["open"]
        if params["direction"] == "bullish":
            return body > params["threshold"]
        else:
            return body < -params["threshold"]

    # フォールバック: 全True
    return pd.Series(True, index=df.index)


def suggest_condition_params(trial, prefix: str, cond_type: str) -> dict:
    """Optunaで条件パラメータを提案する

    Note: パラメータ名に cond_type を含めることで、条件タイプが変わっても
    Optunaの分布互換性チェックに引っかからないようにする。
    """
    p = {}
    # 条件タイプごとにパラメータ名を一意にする
    pfx = f"{prefix}_{cond_type}"

    if cond_type == "rsi_threshold":
        p["period"] = trial.suggest_int(f"{pfx}_period", 5, 50)
        p["threshold"] = trial.suggest_int(f"{pfx}_threshold", 15, 85)
        p["direction"] = trial.suggest_categorical(f"{pfx}_dir", ["below", "above"])

    elif cond_type in ("ema_cross", "sma_cross"):
        p["fast_period"] = trial.suggest_int(f"{pfx}_fast", 3, 50)
        p["slow_period"] = trial.suggest_int(f"{pfx}_slow", 20, 200)
        p["direction"] = trial.suggest_categorical(f"{pfx}_dir", ["above", "below"])

    elif cond_type == "bb_position":
        p["period"] = trial.suggest_int(f"{pfx}_period", 10, 50)
        p["std_dev"] = trial.suggest_float(f"{pfx}_std", 1.0, 3.5, step=0.1)
        p["direction"] = trial.suggest_categorical(
            f"{pfx}_dir", ["above_upper", "below_lower", "above_mid", "below_mid"]
        )

    elif cond_type in ("macd_hist_sign", "macd_cross"):
        p["fast"] = trial.suggest_int(f"{pfx}_fast", 5, 20)
        p["slow"] = trial.suggest_int(f"{pfx}_slow", 20, 50)
        p["signal_period"] = trial.suggest_int(f"{pfx}_sig", 5, 15)
        if cond_type == "macd_hist_sign":
            p["direction"] = trial.suggest_categorical(f"{pfx}_dir", ["positive", "negative"])
        else:
            p["direction"] = trial.suggest_categorical(f"{pfx}_dir", ["above", "below"])

    elif cond_type == "stoch_threshold":
        p["k_period"] = trial.suggest_int(f"{pfx}_k", 5, 30)
        p["d_period"] = trial.suggest_int(f"{pfx}_d", 2, 7)
        p["threshold"] = trial.suggest_int(f"{pfx}_threshold", 10, 90)
        p["direction"] = trial.suggest_categorical(f"{pfx}_dir", ["below", "above"])

    elif cond_type == "stoch_cross":
        p["k_period"] = trial.suggest_int(f"{pfx}_k", 5, 30)
        p["d_period"] = trial.suggest_int(f"{pfx}_d", 2, 7)
        p["direction"] = trial.suggest_categorical(f"{pfx}_dir", ["above", "below"])

    elif cond_type == "atr_breakout":
        p["period"] = trial.suggest_int(f"{pfx}_period", 7, 30)
        p["lookback"] = trial.suggest_int(f"{pfx}_lookback", 10, 50)
        p["multiplier"] = trial.suggest_float(f"{pfx}_mult", 0.5, 3.0, step=0.1)
        p["direction"] = trial.suggest_categorical(f"{pfx}_dir", ["up", "down"])

    elif cond_type in ("price_vs_sma", "price_vs_ema"):
        p["period"] = trial.suggest_int(f"{pfx}_period", 5, 100)
        p["direction"] = trial.suggest_categorical(f"{pfx}_dir", ["above", "below"])

    elif cond_type == "volume_spike":
        p["period"] = trial.suggest_int(f"{pfx}_period", 10, 100)
        p["multiplier"] = trial.suggest_float(f"{pfx}_mult", 1.2, 5.0, step=0.1)

    elif cond_type == "price_momentum":
        p["period"] = trial.suggest_int(f"{pfx}_period", 5, 60)
        p["threshold"] = trial.suggest_float(f"{pfx}_threshold", 0.001, 0.05, step=0.001)
        p["direction"] = trial.suggest_categorical(f"{pfx}_dir", ["positive", "negative"])

    elif cond_type == "candle_body":
        p["threshold"] = trial.suggest_float(f"{pfx}_threshold", 0.001, 0.02, step=0.001)
        p["direction"] = trial.suggest_categorical(f"{pfx}_dir", ["bullish", "bearish"])

    return p


# ─── 合成戦略 ──────────────────────────────────────

class CompositeStrategy(BaseStrategy):
    """条件ブロックの組み合わせで構成される動的戦略

    params = {
        "buy_conditions": [
            {"type": "rsi_threshold", "params": {"period": 14, "threshold": 30, "direction": "below"}},
            {"type": "ema_cross", "params": {"fast_period": 9, "slow_period": 21, "direction": "above"}},
        ],
        "sell_conditions": [...],
        "buy_logic": "and",   # "and" or "or"
        "sell_logic": "and",
        "signal_mode": "state",  # "state" (状態ベース) or "cross" (変化点のみ)
    }
    """

    def __init__(self, params: dict = None):
        super().__init__(params or {})
        self.name = self._make_name()

    def _make_name(self):
        """条件から人間が読める名前を生成"""
        buy = self.params.get("buy_conditions", [])
        sell = self.params.get("sell_conditions", [])
        buy_names = [c["type"] for c in buy]
        sell_names = [c["type"] for c in sell]
        logic_b = self.params.get("buy_logic", "and").upper()
        logic_s = self.params.get("sell_logic", "and").upper()
        return (
            f"Composite[BUY:{logic_b}({','.join(buy_names)})"
            f"|SELL:{logic_s}({','.join(sell_names)})]"
        )

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        buy_logic = self.params.get("buy_logic", "and")
        sell_logic = self.params.get("sell_logic", "and")
        signal_mode = self.params.get("signal_mode", "state")

        # BUY条件を評価
        buy_conds = self.params.get("buy_conditions", [])
        if buy_conds:
            buy_mask = self._evaluate_conditions(df, buy_conds, buy_logic)
        else:
            buy_mask = pd.Series(False, index=df.index)

        # SELL条件を評価
        sell_conds = self.params.get("sell_conditions", [])
        if sell_conds:
            sell_mask = self._evaluate_conditions(df, sell_conds, sell_logic)
        else:
            sell_mask = pd.Series(False, index=df.index)

        # シグナル生成
        df["signal"] = 0
        df.loc[buy_mask, "signal"] = 1
        df.loc[sell_mask, "signal"] = -1

        # 両方Trueの場合はHOLD
        both = buy_mask & sell_mask
        df.loc[both, "signal"] = 0

        # cross モードなら変化点のみ
        if signal_mode == "cross":
            df["signal"] = df["signal"].diff().clip(-1, 1).fillna(0).astype(int)

        return df

    def _evaluate_conditions(self, df, conditions, logic):
        """複数条件をAND/ORで結合"""
        masks = []
        for cond in conditions:
            try:
                mask = compute_condition(df, cond["type"], cond["params"])
                masks.append(mask.fillna(False))
            except Exception:
                masks.append(pd.Series(False, index=df.index))

        if not masks:
            return pd.Series(False, index=df.index)

        if logic == "or":
            result = masks[0]
            for m in masks[1:]:
                result = result | m
            return result
        else:  # and
            result = masks[0]
            for m in masks[1:]:
                result = result & m
            return result

    @classmethod
    def params_space(cls, trial) -> dict:
        """Optunaが構造ごと探索する"""
        # 条件数 (1~4)
        n_buy = trial.suggest_int("n_buy_conditions", 1, 4)
        n_sell = trial.suggest_int("n_sell_conditions", 1, 4)

        buy_conditions = []
        for i in range(n_buy):
            ctype = trial.suggest_categorical(
                f"buy_{i}_type", CONDITION_TYPES
            )
            cparams = suggest_condition_params(trial, f"buy_{i}", ctype)
            buy_conditions.append({"type": ctype, "params": cparams})

        sell_conditions = []
        for i in range(n_sell):
            ctype = trial.suggest_categorical(
                f"sell_{i}_type", CONDITION_TYPES
            )
            cparams = suggest_condition_params(trial, f"sell_{i}", ctype)
            sell_conditions.append({"type": ctype, "params": cparams})

        buy_logic = trial.suggest_categorical("buy_logic", ["and", "or"])
        sell_logic = trial.suggest_categorical("sell_logic", ["and", "or"])
        signal_mode = trial.suggest_categorical("signal_mode", ["state", "cross"])

        return {
            "buy_conditions": buy_conditions,
            "sell_conditions": sell_conditions,
            "buy_logic": buy_logic,
            "sell_logic": sell_logic,
            "signal_mode": signal_mode,
        }

    @classmethod
    def default_params(cls) -> dict:
        return {
            "buy_conditions": [
                {"type": "rsi_threshold", "params": {"period": 14, "threshold": 30, "direction": "below"}},
            ],
            "sell_conditions": [
                {"type": "rsi_threshold", "params": {"period": 14, "threshold": 70, "direction": "above"}},
            ],
            "buy_logic": "and",
            "sell_logic": "and",
            "signal_mode": "state",
        }

    def describe(self) -> str:
        """戦略を人間が読める形で記述"""
        lines = [f"Strategy: {self.name}", ""]
        buy_logic = self.params.get("buy_logic", "and").upper()
        sell_logic = self.params.get("sell_logic", "and").upper()

        lines.append(f"BUY when ({buy_logic}):")
        for c in self.params.get("buy_conditions", []):
            lines.append(f"  - {c['type']}: {c['params']}")

        lines.append(f"SELL when ({sell_logic}):")
        for c in self.params.get("sell_conditions", []):
            lines.append(f"  - {c['type']}: {c['params']}")

        lines.append(f"Signal mode: {self.params.get('signal_mode', 'state')}")
        return "\n".join(lines)
