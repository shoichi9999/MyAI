"""AI自動探索エンジン

Optunaを使って戦略パラメータを自動最適化する。
全戦略×全シンボルの組み合わせを探索し、ランキングする。
"""

import json
import os
import time
import logging
from datetime import datetime
from typing import Optional

import optuna
import pandas as pd

from config.settings import EXPLORER_DEFAULTS, RESULTS_DIR
from backtest.engine import BacktestEngine
from strategies.builtin import ALL_STRATEGIES
from data.fetcher import get_data

# Optunaのログを抑制
optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)


class StrategyOptimizer:
    """単一戦略のパラメータ最適化"""

    def __init__(self, strategy_class, symbol: str, df: pd.DataFrame,
                 metric: str = "sharpe_ratio", engine: BacktestEngine = None):
        self.strategy_class = strategy_class
        self.symbol = symbol
        self.df = df
        self.metric = metric
        self.engine = engine or BacktestEngine()

    def objective(self, trial):
        """Optunaの目的関数"""
        params = self.strategy_class.params_space(trial)

        # fast < slow のような制約チェック
        if "fast_period" in params and "slow_period" in params:
            if params["fast_period"] >= params["slow_period"]:
                return float("-inf")
        if "fast" in params and "slow" in params:
            if params["fast"] >= params["slow"]:
                return float("-inf")

        strategy = self.strategy_class(params)
        try:
            result = self.engine.run(strategy, self.df, self.symbol)
        except Exception:
            return float("-inf")

        # 最低取引数チェック
        if result.metrics["total_trades"] < EXPLORER_DEFAULTS["min_trades"]:
            return float("-inf")

        return result.metrics.get(self.metric, 0)

    def optimize(self, n_trials: int = None, timeout: int = None) -> dict:
        """最適化を実行"""
        n_trials = n_trials or EXPLORER_DEFAULTS["n_trials"]
        timeout = timeout or EXPLORER_DEFAULTS["timeout"]

        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=n_trials, timeout=timeout)

        if study.best_trial:
            best_params = study.best_trial.params
            # 最適パラメータでバックテスト再実行して完全な結果を取得
            strategy = self.strategy_class(best_params)
            result = self.engine.run(strategy, self.df, self.symbol)
            return {
                "strategy": self.strategy_class.__name__,
                "symbol": self.symbol,
                "best_params": best_params,
                "best_value": study.best_value,
                "metrics": result.metrics,
                "n_trials": len(study.trials),
            }
        return None


class AlgorithmExplorer:
    """全戦略×全シンボルの自動探索エンジン

    - 各戦略のパラメータを最適化
    - 結果をランキングして保存
    - 継続的に探索を実行可能
    """

    def __init__(self, symbols: list = None, days: int = 7,
                 n_trials: int = None, metric: str = None):
        from config.settings import DEFAULT_SYMBOLS
        self.symbols = symbols or DEFAULT_SYMBOLS
        self.days = days
        self.n_trials = n_trials or EXPLORER_DEFAULTS["n_trials"]
        self.metric = metric or EXPLORER_DEFAULTS["metric"]
        self.engine = BacktestEngine()
        self.results = []
        self.is_running = False
        self._stop_requested = False

    def run_exploration(self, strategy_names: list = None,
                        symbols: list = None,
                        callback=None) -> list:
        """探索を実行する。

        Args:
            strategy_names: 探索する戦略名リスト (Noneなら全戦略)
            symbols: 対象シンボル (Noneなら設定のデフォルト)
            callback: 各結果が出るたびに呼ばれるコールバック fn(result_dict)

        Returns:
            list of result dicts, sorted by metric
        """
        self.is_running = True
        self._stop_requested = False

        strategies = strategy_names or list(ALL_STRATEGIES.keys())
        target_symbols = symbols or self.symbols
        results = []

        total = len(strategies) * len(target_symbols)
        completed = 0

        for symbol in target_symbols:
            if self._stop_requested:
                break

            logger.info(f"Fetching data for {symbol}...")
            try:
                df = get_data(symbol, days=self.days)
            except Exception as e:
                logger.warning(f"Failed to fetch {symbol}: {e}")
                continue

            if df.empty or len(df) < 100:
                logger.warning(f"Insufficient data for {symbol}")
                continue

            for strat_name in strategies:
                if self._stop_requested:
                    break

                strat_class = ALL_STRATEGIES[strat_name]
                logger.info(f"Optimizing {strat_name} on {symbol}...")

                optimizer = StrategyOptimizer(
                    strategy_class=strat_class,
                    symbol=symbol,
                    df=df,
                    metric=self.metric,
                    engine=self.engine,
                )

                try:
                    result = optimizer.optimize(n_trials=self.n_trials)
                    if result:
                        result["timestamp"] = datetime.utcnow().isoformat()
                        results.append(result)
                        if callback:
                            callback(result)
                except Exception as e:
                    logger.warning(f"Error optimizing {strat_name}/{symbol}: {e}")

                completed += 1
                logger.info(f"Progress: {completed}/{total}")

        # ランキングソート
        results.sort(key=lambda x: x.get("best_value", 0), reverse=True)
        self.results = results
        self.is_running = False

        # 結果保存
        self._save_results(results)
        return results

    def stop(self):
        """探索を停止する"""
        self._stop_requested = True

    def _save_results(self, results: list):
        """結果をJSONファイルに保存"""
        os.makedirs(RESULTS_DIR, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(RESULTS_DIR, f"exploration_{timestamp}.json")
        with open(path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to {path}")
        return path

    def get_ranking(self, top_n: int = 20) -> list:
        """現在の結果からトップNを返す"""
        return self.results[:top_n]

    def load_latest_results(self) -> list:
        """最新の結果ファイルを読み込む"""
        if not os.path.exists(RESULTS_DIR):
            return []
        files = sorted(
            [f for f in os.listdir(RESULTS_DIR) if f.startswith("exploration_")],
            reverse=True
        )
        if not files:
            return []
        path = os.path.join(RESULTS_DIR, files[0])
        with open(path, "r") as f:
            self.results = json.load(f)
        return self.results


def quick_backtest(strategy_name: str, symbol: str, params: dict = None,
                   days: int = 7) -> dict:
    """単発のバックテストを実行するユーティリティ"""
    if strategy_name not in ALL_STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy_name}. Available: {list(ALL_STRATEGIES.keys())}")

    strat_class = ALL_STRATEGIES[strategy_name]
    strategy = strat_class(params or strat_class.default_params())

    df = get_data(symbol, days=days)
    if df.empty:
        raise ValueError(f"No data available for {symbol}")

    engine = BacktestEngine()
    result = engine.run(strategy, df, symbol)
    return result.summary()
