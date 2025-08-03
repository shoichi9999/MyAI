"""
ビットコインバックテスト環境 - 使用例
"""

from main import BitcoinBacktester
from data_fetcher import BitcoinDataFetcher
from strategies.moving_average_strategies import *
from strategies.momentum_strategies import *
from strategies.mean_reversion_strategies import *
import pandas as pd


def example_1_basic_backtest():
    """例1: 基本的なバックテスト"""
    print("【例1】基本的なSMAクロスオーバー戦略")
    print("=" * 50)
    
    backtester = BitcoinBacktester(initial_capital=1000000)
    
    result = backtester.run_single_backtest(
        strategy_name='sma_crossover',
        start_date='2023-01-01',
        end_date='2024-01-01',
        show_visualization=False,
        short_window=20,
        long_window=50
    )
    
    print(f"戦略: {result['strategy_name']}")
    print(f"パラメータ: {result['parameters']}")
    print(f"総リターン: {result['results']['total_return_pct']:.2f}%")


def example_2_strategy_comparison():
    """例2: 複数戦略の比較"""
    print("\n【例2】戦略比較分析")
    print("=" * 50)
    
    backtester = BitcoinBacktester(initial_capital=1000000)
    
    strategies_config = {
        'sma_crossover': {'short_window': 20, 'long_window': 50},
        'ema_crossover': {'short_window': 12, 'long_window': 26},
        'rsi': {'rsi_window': 14, 'oversold_threshold': 30, 'overbought_threshold': 70},
        'macd': {'fast_period': 12, 'slow_period': 26, 'signal_period': 9},
        'bollinger_bands': {'window': 20, 'num_std': 2.0}
    }
    
    comparison_df = backtester.run_strategy_comparison(
        strategies_config=strategies_config,
        start_date='2023-01-01',
        end_date='2024-01-01'
    )
    
    return comparison_df


def example_3_parameter_optimization():
    """例3: パラメータ最適化"""
    print("\n【例3】RSI戦略パラメータ最適化")
    print("=" * 50)
    
    backtester = BitcoinBacktester(initial_capital=1000000)
    
    param_ranges = {
        'rsi_window': [10, 14, 20],
        'oversold_threshold': [20, 30, 35],
        'overbought_threshold': [65, 70, 80]
    }
    
    optimization_df = backtester.run_parameter_optimization(
        strategy_name='rsi',
        param_ranges=param_ranges,
        start_date='2023-06-01',
        end_date='2024-01-01',
        optimization_metric='sharpe_ratio'
    )
    
    print(f"\n最適化結果 (上位5位):")
    print(optimization_df.head()[['rsi_window', 'oversold_threshold', 'overbought_threshold', 
                                 'シャープレシオ', '総リターン(%)']].to_string(index=False))


def example_4_custom_strategy():
    """例4: カスタム戦略の作成と実行"""
    print("\n【例4】カスタム戦略（複合指標）")
    print("=" * 50)
    
    def custom_multi_indicator_strategy(data, sma_short=20, sma_long=50, rsi_period=14):
        """SMAとRSIを組み合わせたカスタム戦略"""
        import numpy as np
        
        data = data.copy()
        
        # SMA計算
        data['sma_short'] = data['close'].rolling(window=sma_short).mean()
        data['sma_long'] = data['close'].rolling(window=sma_long).mean()
        
        # RSI計算
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # 複合シグナル: SMAクロスオーバー + RSI条件
        data['signal'] = 0
        
        # 買いシグナル: 短期SMA > 長期SMA かつ RSI < 70
        buy_condition = (
            (data['sma_short'] > data['sma_long']) & 
            (data['rsi'] < 70)
        )
        
        # 売りシグナル: 短期SMA < 長期SMA または RSI > 80
        sell_condition = (
            (data['sma_short'] < data['sma_long']) | 
            (data['rsi'] > 80)
        )
        
        data.loc[buy_condition, 'signal'] = 1
        data.loc[sell_condition, 'signal'] = -1
        
        # シグナル変化点のみを抽出
        data['position'] = data['signal'].diff()
        data['signal'] = np.where(data['position'] != 0, data['signal'], np.nan)
        
        return data
    
    # バックテストエンジンで直接実行
    from backtest_engine import BacktestEngine
    from data_fetcher import BitcoinDataFetcher
    
    fetcher = BitcoinDataFetcher()
    data = fetcher.get_bitcoin_data(
        source="yahoo",
        start_date="2023-01-01",
        end_date="2024-01-01"
    )
    
    engine = BacktestEngine(initial_capital=1000000)
    results = engine.run_backtest(
        data=data,
        strategy_func=custom_multi_indicator_strategy,
        sma_short=15,
        sma_long=35,
        rsi_period=14
    )
    
    engine.print_results()


def example_5_detailed_analysis():
    """例5: 詳細なパフォーマンス分析"""
    print("\n【例5】詳細パフォーマンス分析")
    print("=" * 50)
    
    backtester = BitcoinBacktester(initial_capital=1000000)
    
    result = backtester.run_single_backtest(
        strategy_name='bollinger_bands',
        start_date='2023-01-01',
        end_date='2024-01-01',
        show_visualization=False,
        window=20,
        num_std=2.0
    )
    
    # 詳細分析
    from performance import PerformanceAnalyzer
    
    analyzer = PerformanceAnalyzer(result['results'])
    detailed_metrics = analyzer.calculate_comprehensive_metrics()
    
    print("\n主要パフォーマンス指標:")
    key_metrics = {
        '総リターン': f"{detailed_metrics.get('total_return', 0):.2f}%",
        'シャープレシオ': f"{detailed_metrics.get('sharpe_ratio', 0):.3f}",
        'ソルティーノレシオ': f"{detailed_metrics.get('sortino_ratio', 0):.3f}",
        'カルマーレシオ': f"{detailed_metrics.get('calmar_ratio', 0):.3f}",
        '最大ドローダウン': f"{detailed_metrics.get('max_drawdown_pct', 0):.2f}%",
        '勝率': f"{detailed_metrics.get('win_rate', 0):.1f}%",
        '年率ボラティリティ': f"{detailed_metrics.get('volatility_annual', 0):.2f}%"
    }
    
    for metric, value in key_metrics.items():
        print(f"{metric}: {value}")


def example_6_live_data_analysis():
    """例6: 最新データでの分析"""
    print("\n🔄 例6: 最新データでの戦略テスト")
    print("=" * 50)
    
    from datetime import datetime, timedelta
    
    # 最近6ヶ月のデータで分析
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)
    
    backtester = BitcoinBacktester(initial_capital=1000000)
    
    try:
        result = backtester.run_single_backtest(
            strategy_name='triple_sma',
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            show_visualization=False,
            short_window=10,
            medium_window=20,
            long_window=50
        )
        
        print(f"分析期間: {start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')}")
        print(f"最終資産価値: ¥{result['results']['final_value']:,.0f}")
        print(f"Buy&Hold比較: {result['results']['excess_return_pct']:+.2f}%")
        
    except Exception as e:
        print(f"最新データ取得に失敗: {e}")
        print("インターネット接続を確認してください。")


def run_all_examples():
    """全ての例を実行"""
    print(">>> ビットコインバックテスト環境 - 使用例実行")
    print("=" * 80)
    
    examples = [
        example_1_basic_backtest,
        example_2_strategy_comparison,
        example_3_parameter_optimization,
        example_4_custom_strategy,
        example_5_detailed_analysis,
        example_6_live_data_analysis
    ]
    
    for i, example_func in enumerate(examples, 1):
        try:
            print(f"\n{'='*20} 実行中: 例{i} {'='*20}")
            example_func()
            print(f"OK: 例{i} 完了")
        except Exception as e:
            print(f"ERROR: 例{i} でエラー: {e}")
    
    print(f"\n>>> 全ての例の実行が完了しました！")
    print("\n次のステップ:")
    print("1. 独自の戦略を strategies/ ディレクトリに追加")
    print("2. パラメータを調整して最適化")
    print("3. より長期間のデータで検証")
    print("4. リスク管理機能の追加")


if __name__ == "__main__":
    run_all_examples()