"""戦略自動生成エンジン

CompositeStrategyの構造+パラメータをOptunaで同時探索する。
既存の固定10戦略とは別に、条件ブロックの組み合わせ空間を探索。

Usage:
    generator = StrategyGenerator(symbols=["BTCUSDT"], days=7)
    results = generator.run(n_trials=500)
    # → 数百〜数千の新しい戦略構造を試行し、ベストを返す
"""

import json
import os
import logging
from datetime import datetime

import optuna
import pandas as pd

from config.settings import EXPLORER_DEFAULTS, RESULTS_DIR
from backtest.engine import BacktestEngine
from strategies.composer import CompositeStrategy
from data.fetcher import get_data

optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)


class StrategyGenerator:
    """合成戦略の自動生成＆最適化

    固定戦略のパラメータ最適化とは根本的に異なる:
    - 条件の「種類」(RSI? EMA? BB?)
    - 条件の「数」(1〜4個)
    - 結合ロジック(AND/OR)
    - シグナルモード(状態/クロス)
    をすべて同時に探索する。
    """

    def __init__(self, symbols: list = None, days: int = 7,
                 metric: str = None):
        from config.settings import DEFAULT_SYMBOLS
        self.symbols = symbols or DEFAULT_SYMBOLS[:3]  # デフォルトは上位3銘柄
        self.days = days
        self.metric = metric or EXPLORER_DEFAULTS["metric"]
        self.engine = BacktestEngine()
        self.results = []
        self._stop_requested = False

    def _objective(self, trial, df: pd.DataFrame, symbol: str):
        """Optunaの目的関数 — 構造ごと探索"""
        params = CompositeStrategy.params_space(trial)

        # fast < slow 制約チェック
        for side in ["buy_conditions", "sell_conditions"]:
            for cond in params.get(side, []):
                cp = cond["params"]
                if "fast_period" in cp and "slow_period" in cp:
                    if cp["fast_period"] >= cp["slow_period"]:
                        return float("-inf")
                if "fast" in cp and "slow" in cp:
                    if cp["fast"] >= cp["slow"]:
                        return float("-inf")

        strategy = CompositeStrategy(params)

        try:
            result = self.engine.run(strategy, df, symbol)
        except Exception:
            return float("-inf")

        # 最低取引数
        if result.metrics["total_trades"] < EXPLORER_DEFAULTS["min_trades"]:
            return float("-inf")

        return result.metrics.get(self.metric, 0)

    def run(self, n_trials: int = 500, timeout: int = None,
            callback=None) -> list:
        """戦略生成を実行

        Args:
            n_trials: シンボルあたりの試行数 (構造×パラメータの総数)
            timeout: シンボルあたりのタイムアウト秒
            callback: 各結果のコールバック fn(result_dict)

        Returns:
            ベスト戦略のリスト (メトリクス順)
        """
        timeout = timeout or EXPLORER_DEFAULTS.get("timeout", 600)
        self._stop_requested = False
        all_results = []

        for symbol in self.symbols:
            if self._stop_requested:
                break

            logger.info(f"=== Generating strategies for {symbol} ===")
            try:
                df = get_data(symbol, days=self.days)
            except Exception as e:
                logger.warning(f"Failed to fetch {symbol}: {e}")
                continue

            if df.empty or len(df) < 100:
                logger.warning(f"Insufficient data for {symbol}")
                continue

            study = optuna.create_study(direction="maximize")

            def obj(trial):
                return self._objective(trial, df, symbol)

            study.optimize(obj, n_trials=n_trials, timeout=timeout)

            # 上位N件を記録
            top_trials = sorted(
                study.trials,
                key=lambda t: t.value if t.value is not None else float("-inf"),
                reverse=True,
            )[:20]

            for t in top_trials:
                if t.value is None or t.value == float("-inf"):
                    continue

                # パラメータを再構築して完全なバックテスト
                params = CompositeStrategy.params_space(t)
                strategy = CompositeStrategy(params)

                try:
                    result = self.engine.run(strategy, df, symbol)
                except Exception:
                    continue

                entry = {
                    "strategy": strategy.name,
                    "strategy_type": "composite",
                    "symbol": symbol,
                    "params": params,
                    "description": strategy.describe(),
                    "best_value": t.value,
                    "metrics": result.metrics,
                    "n_trials": n_trials,
                    "timestamp": datetime.utcnow().isoformat(),
                }
                all_results.append(entry)

                if callback:
                    callback(entry)

            logger.info(
                f"{symbol}: {len(study.trials)} trials, "
                f"best {self.metric}={study.best_value:.4f}"
            )

        # ランキング
        all_results.sort(key=lambda x: x.get("best_value", 0), reverse=True)
        self.results = all_results

        # 保存
        self._save_results(all_results)
        return all_results

    def stop(self):
        self._stop_requested = True

    def _save_results(self, results: list):
        os.makedirs(RESULTS_DIR, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(RESULTS_DIR, f"generated_{timestamp}.json")
        with open(path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Generated strategies saved to {path}")
        return path

    def get_best(self, n: int = 10) -> list:
        return self.results[:n]

    def describe_best(self, n: int = 5):
        """ベスト戦略を人間が読める形で出力"""
        for i, r in enumerate(self.results[:n]):
            print(f"\n{'='*60}")
            print(f"#{i+1} | {r['symbol']} | {self.metric}: {r['best_value']:.4f}")
            print(f"{'='*60}")
            print(r["description"])
            m = r["metrics"]
            print(f"\nReturn: {m.get('return_pct', 0):.2f}%")
            print(f"Sharpe: {m.get('sharpe_ratio', 0):.4f}")
            print(f"Max DD: {m.get('max_drawdown_pct', 0):.2f}%")
            print(f"Win Rate: {m.get('win_rate', 0):.1f}%")
            print(f"Trades: {m.get('total_trades', 0)}")
