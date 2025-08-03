"""
ビットコインバックテスト環境 - メイン実行ファイル
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 自作モジュールのインポート
from data_fetcher import BitcoinDataFetcher
from backtest_engine import BacktestEngine
from performance import PerformanceAnalyzer
from visualization import BacktestVisualizer
from strategies.moving_average_strategies import (
    simple_moving_average_crossover,
    exponential_moving_average_crossover,
    triple_moving_average_strategy
)
from strategies.momentum_strategies import (
    rsi_strategy,
    macd_strategy,
    momentum_strategy
)
from strategies.mean_reversion_strategies import (
    bollinger_bands_strategy,
    mean_reversion_strategy
)


class BitcoinBacktester:
    """ビットコインバックテスター統合クラス"""
    
    def __init__(self, initial_capital: float = 1000000, commission_rate: float = 0.001):
        """
        Args:
            initial_capital: 初期資金
            commission_rate: 手数料率
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.data_fetcher = BitcoinDataFetcher()
        self.engine = BacktestEngine(initial_capital, commission_rate)
        
        # 利用可能な戦略辞書
        self.strategies = {
            'sma_crossover': simple_moving_average_crossover,
            'ema_crossover': exponential_moving_average_crossover,
            'triple_sma': triple_moving_average_strategy,
            'rsi': rsi_strategy,
            'macd': macd_strategy,
            'momentum': momentum_strategy,
            'bollinger_bands': bollinger_bands_strategy,
            'mean_reversion': mean_reversion_strategy
        }
    
    def run_single_backtest(self,
                          strategy_name: str,
                          start_date: str = "2022-01-01",
                          end_date: str = "2024-01-01",
                          data_source: str = "yahoo",
                          show_visualization: bool = True,
                          **strategy_params) -> dict:
        """
        単一戦略のバックテストを実行
        
        Args:
            strategy_name: 戦略名
            start_date: 開始日
            end_date: 終了日
            data_source: データソース
            show_visualization: 可視化表示フラグ
            **strategy_params: 戦略パラメータ
            
        Returns:
            dict: バックテスト結果
        """
        print(f"=== {strategy_name} 戦略バックテスト開始 ===")
        print(f"期間: {start_date} - {end_date}")
        print(f"初期資金: ¥{self.initial_capital:,.0f}")
        
        # データ取得
        print("ビットコインデータを取得中...")
        data = self.data_fetcher.get_bitcoin_data(
            source=data_source,
            start_date=start_date,
            end_date=end_date
        )
        print(f"データ取得完了: {len(data)} 件")
        
        # 戦略実行
        if strategy_name not in self.strategies:
            raise ValueError(f"未知の戦略: {strategy_name}. 利用可能: {list(self.strategies.keys())}")
        
        strategy_func = self.strategies[strategy_name]
        print(f"戦略実行中: {strategy_name}")
        
        results = self.engine.run_backtest(data, strategy_func, **strategy_params)
        
        # パフォーマンス分析
        analyzer = PerformanceAnalyzer(results)
        detailed_metrics = analyzer.calculate_comprehensive_metrics()
        
        # 結果表示
        self.engine.print_results()
        print("\n" + analyzer.generate_performance_report())
        
        # 可視化
        if show_visualization:
            data_with_signals = strategy_func(data.copy(), **strategy_params)
            visualizer = BacktestVisualizer(results, data_with_signals)
            
            try:
                import matplotlib.pyplot as plt
                fig = visualizer.plot_comprehensive_analysis()
                plt.show()
            except ImportError:
                print("matplotlib がインストールされていないため、グラフを表示できません。")
        
        return {
            'strategy_name': strategy_name,
            'parameters': strategy_params,
            'results': results,
            'detailed_metrics': detailed_metrics,
            'data': data
        }
    
    def run_strategy_comparison(self,
                              strategies_config: dict,
                              start_date: str = "2022-01-01",
                              end_date: str = "2024-01-01",
                              data_source: str = "yahoo") -> pd.DataFrame:
        """
        複数戦略の比較分析
        
        Args:
            strategies_config: 戦略設定辞書 {戦略名: パラメータ辞書}
            start_date: 開始日
            end_date: 終了日
            data_source: データソース
            
        Returns:
            pd.DataFrame: 比較結果
        """
        print("=== 戦略比較分析開始 ===")
        
        # データ取得（共通）
        data = self.data_fetcher.get_bitcoin_data(
            source=data_source,
            start_date=start_date,
            end_date=end_date
        )
        
        results_summary = []
        
        for strategy_name, params in strategies_config.items():
            print(f"\n戦略実行中: {strategy_name}")
            
            try:
                # バックテスト実行
                strategy_func = self.strategies[strategy_name]
                results = self.engine.run_backtest(data, strategy_func, **params)
                
                # 主要指標を抽出
                summary = {
                    '戦略名': strategy_name,
                    'パラメータ': str(params),
                    '総リターン(%)': results['total_return_pct'],
                    '最大DD(%)': results['max_drawdown_pct'],
                    'シャープレシオ': results['sharpe_ratio'],
                    '取引回数': results['num_trades'],
                    '最終資産価値': results['final_value'],
                    'Buy&Hold超過リターン(%)': results['excess_return_pct']
                }
                results_summary.append(summary)
                
            except Exception as e:
                print(f"戦略 {strategy_name} でエラー: {e}")
        
        # 結果をDataFrameに変換
        comparison_df = pd.DataFrame(results_summary)
        
        if not comparison_df.empty:
            # リターンでソート
            comparison_df = comparison_df.sort_values('総リターン(%)', ascending=False)
            
            print("\n" + "=" * 80)
            print("戦略比較結果")
            print("=" * 80)
            print(comparison_df.to_string(index=False))
            
            # トップ3戦略を表示
            print(f"\n【トップ3戦略】")
            for i, (_, row) in enumerate(comparison_df.head(3).iterrows(), 1):
                print(f"{i}. {row['戦略名']}: {row['総リターン(%)']:.2f}% リターン")
        
        return comparison_df
    
    def run_parameter_optimization(self,
                                 strategy_name: str,
                                 param_ranges: dict,
                                 start_date: str = "2022-01-01",
                                 end_date: str = "2024-01-01",
                                 optimization_metric: str = "sharpe_ratio") -> pd.DataFrame:
        """
        パラメータ最適化
        
        Args:
            strategy_name: 戦略名
            param_ranges: パラメータ範囲辞書
            start_date: 開始日
            end_date: 終了日
            optimization_metric: 最適化指標
            
        Returns:
            pd.DataFrame: 最適化結果
        """
        print(f"=== {strategy_name} パラメータ最適化開始 ===")
        
        # データ取得
        data = self.data_fetcher.get_bitcoin_data(
            source="yahoo",
            start_date=start_date,
            end_date=end_date
        )
        
        strategy_func = self.strategies[strategy_name]
        optimization_results = []
        
        # パラメータの組み合わせを生成
        from itertools import product
        
        param_names = list(param_ranges.keys())
        param_values = list(param_ranges.values())
        
        total_combinations = 1
        for values in param_values:
            total_combinations *= len(values)
        
        print(f"総組み合わせ数: {total_combinations}")
        
        for i, param_combination in enumerate(product(*param_values), 1):
            params = dict(zip(param_names, param_combination))
            
            try:
                results = self.engine.run_backtest(data, strategy_func, **params)
                
                optimization_results.append({
                    '組み合わせ': i,
                    **params,
                    '総リターン(%)': results['total_return_pct'],
                    'シャープレシオ': results['sharpe_ratio'],
                    '最大DD(%)': results['max_drawdown_pct'],
                    '取引回数': results['num_trades']
                })
                
                if i % 10 == 0:
                    print(f"進捗: {i}/{total_combinations} ({i/total_combinations*100:.1f}%)")
                    
            except Exception as e:
                print(f"パラメータ {params} でエラー: {e}")
        
        # 結果をDataFrameに変換
        optimization_df = pd.DataFrame(optimization_results)
        
        if not optimization_df.empty:
            # 最適化指標でソート
            optimization_df = optimization_df.sort_values(optimization_metric, ascending=False)
            
            print(f"\n最適パラメータ ({optimization_metric} 基準):")
            best_params = optimization_df.iloc[0]
            print(best_params.to_string())
        
        return optimization_df


def run_demo():
    """デモ実行"""
    print(">>> ビットコインバックテスト環境 デモ実行")
    
    # バックテスター初期化
    backtester = BitcoinBacktester(initial_capital=1000000, commission_rate=0.001)
    
    # 1. 単一戦略テスト
    print("\n" + "="*60)
    print("1. シンプル移動平均クロスオーバー戦略")
    print("="*60)
    
    result1 = backtester.run_single_backtest(
        strategy_name='sma_crossover',
        start_date='2023-01-01',
        end_date='2024-01-01',
        show_visualization=False,
        short_window=10,
        long_window=30
    )
    
    # 2. 戦略比較
    print("\n" + "="*60)
    print("2. 戦略比較分析")
    print("="*60)
    
    strategies_config = {
        'sma_crossover': {'short_window': 10, 'long_window': 30},
        'rsi': {'rsi_window': 14, 'oversold_threshold': 30, 'overbought_threshold': 70},
        'bollinger_bands': {'window': 20, 'num_std': 2.0}
    }
    
    comparison_result = backtester.run_strategy_comparison(
        strategies_config=strategies_config,
        start_date='2023-01-01',
        end_date='2024-01-01'
    )
    
    print("\n>>> デモ実行完了!")
    print("詳細な分析を行う場合は、個別の戦略を実行してください。")


if __name__ == "__main__":
    run_demo()