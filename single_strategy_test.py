"""
単一戦略の詳細テスト
"""

from data_fetcher import BitcoinDataFetcher
from backtest_engine import BacktestEngine
from strategies.moving_average_strategies import simple_moving_average_crossover
from performance import PerformanceAnalyzer
import warnings
warnings.filterwarnings('ignore')

def test_single_strategy():
    """単一戦略の詳細テスト"""
    print("単一戦略詳細テスト")
    print("=" * 50)
    
    # データ取得
    print("ビットコインデータ取得中...")
    fetcher = BitcoinDataFetcher()
    data = fetcher.get_bitcoin_data(
        source="yahoo",
        start_date="2023-01-01",
        end_date="2024-01-01"
    )
    print(f"データ取得完了: {len(data)} 件")
    
    # バックテスト実行
    print("\nバックテスト実行中...")
    engine = BacktestEngine(initial_capital=1000000, commission_rate=0.001)
    results = engine.run_backtest(
        data=data,
        strategy_func=simple_moving_average_crossover,
        short_window=20,
        long_window=50
    )
    
    # 基本結果表示
    print("\n基本結果:")
    print(f"初期資金: {results['initial_capital']:,.0f} 円")
    print(f"最終資産価値: {results['final_value']:,.0f} 円")
    print(f"総リターン: {results['total_return_pct']:.2f}%")
    print(f"最大ドローダウン: {results['max_drawdown_pct']:.2f}%")
    print(f"シャープレシオ: {results['sharpe_ratio']:.3f}")
    print(f"取引回数: {results['num_trades']}")
    print(f"Buy&Hold比較: {results['excess_return_pct']:+.2f}%")
    
    # 詳細分析
    print("\n詳細パフォーマンス分析:")
    analyzer = PerformanceAnalyzer(results)
    detailed_metrics = analyzer.calculate_comprehensive_metrics()
    
    if detailed_metrics:
        print(f"ソルティーノレシオ: {detailed_metrics.get('sortino_ratio', 0):.3f}")
        print(f"カルマーレシオ: {detailed_metrics.get('calmar_ratio', 0):.3f}")
        print(f"年率ボラティリティ: {detailed_metrics.get('volatility_annual', 0):.2f}%")
        print(f"勝率: {detailed_metrics.get('win_rate', 0):.1f}%")
        print(f"プロフィットファクター: {detailed_metrics.get('profit_factor', 0):.2f}")
    
    # 取引履歴
    if results['trades']:
        print(f"\n取引履歴（最初の5件）:")
        for i, trade in enumerate(results['trades'][:5]):
            print(f"{i+1}. {trade.timestamp.date()} - {trade.action} {trade.quantity:.4f} BTC @ ${trade.price:,.0f}")
    
    # エクイティカーブの簡単な統計
    equity_curve = results['equity_curve']
    if not equity_curve.empty:
        print(f"\nエクイティカーブ統計:")
        print(f"開始価値: {equity_curve['portfolio_value'].iloc[0]:,.0f}")
        print(f"最高価値: {equity_curve['portfolio_value'].max():,.0f}")
        print(f"最低価値: {equity_curve['portfolio_value'].min():,.0f}")
        print(f"最終価値: {equity_curve['portfolio_value'].iloc[-1]:,.0f}")
    
    return results

if __name__ == "__main__":
    results = test_single_strategy()
    print("\n詳細テスト完了!")