"""高速バックテストエンジン

1分足データに対してシグナルベースのバックテストを実行する。
手数料・スリッページを考慮した現実的なシミュレーション。
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
                 slippage_rate: float = None, leverage: int = None):
        defaults = BACKTEST_DEFAULTS
        self.initial_capital = initial_capital or defaults["initial_capital"]
        self.commission_rate = commission_rate or defaults["commission_rate"]
        self.slippage_rate = slippage_rate or defaults["slippage_rate"]
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
        """シグナルに基づくトレードシミュレーション"""
        capital = self.initial_capital
        position = 0        # 現在のポジション数量
        direction = 0       # 1=long, -1=short, 0=flat
        entry_price = 0.0
        entry_time = None
        trades = []
        equity = [capital]

        signals = df["signal"].values
        opens = df["open"].values
        closes = df["close"].values
        timestamps = df["timestamp"].values

        for i in range(1, len(df)):
            signal = signals[i - 1]  # 前の足のシグナルで次の足のオープンで執行
            price = opens[i]         # 次の足の始値で執行

            # ポジションクローズ判定
            if direction != 0 and signal != direction and signal != 0:
                # クローズ
                exit_price = self._apply_slippage(price, -direction)
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

            # ポジションオープン判定
            if direction == 0 and signal != 0:
                direction = signal
                entry_price = self._apply_slippage(price, direction)
                entry_time = timestamps[i]
                # ポジションサイズ計算 (レバレッジ考慮)
                position_value = capital * self.leverage
                commission = position_value * self.commission_rate
                position = (position_value - commission) / entry_price

            # 含み損益を反映したエクイティ
            if direction != 0:
                unrealized = direction * position * (closes[i] - entry_price)
                equity.append(capital + unrealized)
            else:
                equity.append(capital)

        # 最後にポジションが残っていればクローズ
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

    def _apply_slippage(self, price: float, direction: int) -> float:
        """スリッページを適用。買いなら高くなり、売りなら安くなる。"""
        return price * (1 + direction * self.slippage_rate)

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
