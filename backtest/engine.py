"""高速バックテストエンジン

1分足データに対してシグナルベースのバックテストを実行する。
指値注文（maker）での約定を前提としたシミュレーション。
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from config.settings import BACKTEST_DEFAULTS
from strategies.base import Trade


@dataclass
class BacktestResult:
    """バックテスト結果"""
    strategy_name: str
    symbol: str
    params: dict
    trades: list
    equity_curve: pd.Series
    metrics: dict

    def summary(self) -> dict:
        return {
            "strategy": self.strategy_name,
            "symbol": self.symbol,
            "params": self.params,
            **self.metrics,
        }


class BacktestEngine:
    """シグナルベースのバックテストエンジン"""

    def __init__(self, initial_capital: float = None, commission_rate: float = None,
                 leverage: int = None):
        defaults = BACKTEST_DEFAULTS
        self.initial_capital = initial_capital or defaults["initial_capital"]
        self.commission_rate = commission_rate or defaults["commission_rate"]
        self.leverage = leverage or defaults["leverage"]

    def run(self, strategy, df: pd.DataFrame, symbol: str = "") -> BacktestResult:
        """バックテストを実行する。

        Args:
            strategy: BaseStrategy のインスタンス
            df: OHLCVデータ (timestamp, open, high, low, close, volume)
            symbol: 通貨ペア名

        Returns:
            BacktestResult
        """
        # シグナル生成
        df = df.copy()
        df = strategy.generate_signals(df)

        if "signal" not in df.columns:
            raise ValueError("Strategy must generate 'signal' column")

        # シミュレーション実行
        trades, equity_curve = self._simulate(df)

        # メトリクス計算
        metrics = self._calculate_metrics(trades, equity_curve)

        return BacktestResult(
            strategy_name=strategy.name,
            symbol=symbol,
            params=strategy.params,
            trades=trades,
            equity_curve=equity_curve,
            metrics=metrics,
        )

    def _simulate(self, df: pd.DataFrame):
        """指値注文によるトレードシミュレーション

        シグナル足のclose価格で指値注文を発注し、次の足で約定判定する。
        - 買い指値: 次の足の low <= 指値価格 なら約定
        - 売り指値: 次の足の high >= 指値価格 なら約定
        - 約定しなければ注文はキャンセル (1足限りの有効期間)
        """
        capital = self.initial_capital
        position = 0        # 現在のポジション数量
        direction = 0       # 1=long, -1=short, 0=flat
        entry_price = 0.0
        entry_time = None
        trades = []
        equity = [capital]

        signals = df["signal"].values
        closes = df["close"].values
        highs = df["high"].values
        lows = df["low"].values
        timestamps = df["timestamp"].values

        # 保留中の指値注文
        pending_close_price = 0.0   # 指値価格 (前の足のclose)
        pending_signal = 0          # 保留シグナル

        for i in range(1, len(df)):
            prev_signal = signals[i - 1]
            limit_price = closes[i - 1]  # シグナル足のclose = 指値価格

            filled_close = False
            filled_open = False

            # ── ポジションクローズの指値約定判定 ──
            if direction != 0 and prev_signal != direction and prev_signal != 0:
                # 売り指値 (ロング決済): high >= limit_price で約定
                # 買い指値 (ショート決済): low <= limit_price で約定
                if direction == 1 and highs[i] >= limit_price:
                    filled_close = True
                elif direction == -1 and lows[i] <= limit_price:
                    filled_close = True

                if filled_close:
                    exit_price = limit_price
                    commission = abs(position) * exit_price * self.commission_rate
                    pnl = direction * position * (exit_price - entry_price) - commission
                    pnl_pct = pnl / (abs(position) * entry_price) if entry_price > 0 else 0

                    trades.append(Trade(
                        entry_time=entry_time,
                        exit_time=timestamps[i],
                        direction=direction,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        size=position,
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                    ))
                    capital += pnl
                    position = 0
                    direction = 0

            # ── ポジションオープンの指値約定判定 ──
            if direction == 0 and prev_signal != 0:
                # 買い指値: low <= limit_price で約定
                # 売り指値: high >= limit_price で約定
                if prev_signal == 1 and lows[i] <= limit_price:
                    filled_open = True
                elif prev_signal == -1 and highs[i] >= limit_price:
                    filled_open = True

                if filled_open:
                    direction = prev_signal
                    entry_price = limit_price
                    entry_time = timestamps[i]
                    position_value = capital * self.leverage
                    commission = position_value * self.commission_rate
                    position = (position_value - commission) / entry_price

            # 含み損益を反映したエクイティ
            if direction != 0:
                unrealized = direction * position * (closes[i] - entry_price)
                equity.append(capital + unrealized)
            else:
                equity.append(capital)

        # 最後にポジションが残っていればclose価格で決済
        if direction != 0 and len(df) > 0:
            exit_price = closes[-1]
            commission = abs(position) * exit_price * self.commission_rate
            pnl = direction * position * (exit_price - entry_price) - commission
            pnl_pct = pnl / (abs(position) * entry_price) if entry_price > 0 else 0
            trades.append(Trade(
                entry_time=entry_time,
                exit_time=timestamps[-1],
                direction=direction,
                entry_price=entry_price,
                exit_price=exit_price,
                size=position,
                pnl=pnl,
                pnl_pct=pnl_pct,
            ))
            capital += pnl

        equity_series = pd.Series(equity)
        return trades, equity_series

    def _calculate_metrics(self, trades: list, equity: pd.Series) -> dict:
        """パフォーマンスメトリクスを計算"""
        if not trades:
            return {
                "total_return": 0.0,
                "total_return_pct": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "max_drawdown_pct": 0.0,
                "win_rate": 0.0,
                "total_trades": 0,
                "profit_factor": 0.0,
                "avg_trade_pnl": 0.0,
                "avg_trade_pnl_pct": 0.0,
                "avg_holding_minutes": 0.0,
            }

        pnls = [t.pnl for t in trades]
        pnl_pcts = [t.pnl_pct for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        total_return = sum(pnls)
        total_return_pct = (equity.iloc[-1] / equity.iloc[0] - 1) * 100 if equity.iloc[0] > 0 else 0

        # シャープレシオ (1分足ベースのリターンから年率換算)
        returns = equity.pct_change().dropna()
        if len(returns) > 1 and returns.std() > 0:
            # 1分足 → 年率: sqrt(525600)
            sharpe = (returns.mean() / returns.std()) * np.sqrt(525600)
        else:
            sharpe = 0.0

        # 最大ドローダウン
        peak = equity.expanding().max()
        drawdown = equity - peak
        max_dd = drawdown.min()
        max_dd_pct = (drawdown / peak).min() * 100

        # 勝率
        win_rate = len(wins) / len(trades) * 100 if trades else 0

        # プロフィットファクター
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # 平均保有時間(分)
        holding_times = []
        for t in trades:
            try:
                delta = pd.Timestamp(t.exit_time) - pd.Timestamp(t.entry_time)
                holding_times.append(delta.total_seconds() / 60)
            except Exception:
                pass
        avg_holding = np.mean(holding_times) if holding_times else 0

        return {
            "total_return": round(total_return, 2),
            "total_return_pct": round(total_return_pct, 2),
            "sharpe_ratio": round(sharpe, 4),
            "max_drawdown": round(max_dd, 2),
            "max_drawdown_pct": round(max_dd_pct, 2),
            "win_rate": round(win_rate, 2),
            "total_trades": len(trades),
            "profit_factor": round(profit_factor, 4),
            "avg_trade_pnl": round(np.mean(pnls), 2),
            "avg_trade_pnl_pct": round(np.mean(pnl_pcts) * 100, 4),
            "avg_holding_minutes": round(avg_holding, 1),
        }
