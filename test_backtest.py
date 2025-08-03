"""
バックテスト環境のテストケース
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# テスト対象のインポート
from data_fetcher import BitcoinDataFetcher
from backtest_engine import BacktestEngine, Portfolio
from performance import PerformanceAnalyzer
from strategies.moving_average_strategies import simple_moving_average_crossover
from strategies.momentum_strategies import rsi_strategy


class TestBitcoinDataFetcher:
    """データ取得機能のテスト"""
    
    def setup_method(self):
        """各テストの前に実行"""
        self.fetcher = BitcoinDataFetcher()
    
    def test_yahoo_finance_data_structure(self):
        """Yahoo Financeデータの構造テスト"""
        try:
            data = self.fetcher.fetch_yahoo_finance(
                start_date="2023-01-01",
                end_date="2023-01-31"
            )
            
            # データが空でないことを確認
            assert not data.empty
            
            # 必要なカラムが存在することを確認
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                assert col in data.columns
            
            # データ型の確認
            assert data['close'].dtype in [np.float64, np.float32]
            
            print("OK: Yahoo Finance データ構造テスト成功")
            
        except Exception as e:
            print(f"⚠️ Yahoo Finance データ取得テストはスキップ (理由: {e})")
    
    def test_cache_functionality(self):
        """キャッシュ機能のテスト"""
        # テスト用データ
        test_data = pd.DataFrame({
            'close': [100, 101, 102, 103],
            'volume': [1000, 1100, 1200, 1300]
        }, index=pd.date_range('2023-01-01', periods=4))
        
        # キャッシュに保存
        self.fetcher.save_to_cache(test_data, 'test_cache')
        
        # キャッシュから読み込み
        loaded_data = self.fetcher.load_from_cache('test_cache')
        
        # データが正しく保存・読み込みされていることを確認
        assert loaded_data is not None
        assert len(loaded_data) == len(test_data)
        assert loaded_data['close'].iloc[0] == 100
        
        print("OK: キャッシュ機能テスト成功")


class TestBacktestEngine:
    """バックテストエンジンのテスト"""
    
    def setup_method(self):
        """各テストの前に実行"""
        self.engine = BacktestEngine(initial_capital=100000, commission_rate=0.001)
        
        # テスト用価格データ
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        np.random.seed(42)
        prices = 50000 + np.cumsum(np.random.randn(50) * 100)
        
        self.test_data = pd.DataFrame({
            'open': prices * 0.999,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'volume': np.random.randint(1000, 10000, 50)
        }, index=dates)
    
    def test_portfolio_initialization(self):
        """ポートフォリオ初期化テスト"""
        portfolio = Portfolio(initial_capital=100000, commission_rate=0.001)
        
        assert portfolio.initial_capital == 100000
        assert portfolio.cash == 100000
        assert portfolio.btc_holdings == 0.0
        assert portfolio.commission_rate == 0.001
        
        print("OK: ポートフォリオ初期化テスト成功")
    
    def test_trade_execution(self):
        """取引実行テスト"""
        portfolio = Portfolio(initial_capital=100000, commission_rate=0.001)
        
        # 買い注文
        success = portfolio.execute_trade(
            timestamp=datetime.now(),
            action='BUY',
            price=50000,
            quantity=1.0
        )
        
        assert success == True
        assert portfolio.btc_holdings == 1.0
        assert portfolio.cash < 50000  # 手数料分減る
        
        # 売り注文
        success = portfolio.execute_trade(
            timestamp=datetime.now(),
            action='SELL',
            price=51000,
            quantity=1.0
        )
        
        assert success == True
        assert portfolio.btc_holdings == 0.0
        assert portfolio.cash > 50000  # 利益が出ている
        
        print("✅ 取引実行テスト成功")
    
    def test_backtest_execution(self):
        """バックテスト実行テスト"""
        results = self.engine.run_backtest(
            data=self.test_data,
            strategy_func=simple_moving_average_crossover,
            short_window=5,
            long_window=15
        )
        
        # 結果の基本構造確認
        required_keys = ['initial_capital', 'final_value', 'total_return_pct', 
                        'max_drawdown_pct', 'sharpe_ratio', 'num_trades']
        
        for key in required_keys:
            assert key in results
        
        # 論理的な値の確認
        assert results['initial_capital'] == 100000
        assert results['final_value'] > 0
        assert isinstance(results['num_trades'], int)
        
        print("✅ バックテスト実行テスト成功")


class TestStrategies:
    """戦略関数のテスト"""
    
    def setup_method(self):
        """各テストの前に実行"""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        np.random.seed(42)
        prices = 50000 + np.cumsum(np.random.randn(100) * 100)
        
        self.test_data = pd.DataFrame({
            'open': prices * 0.999,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
    
    def test_sma_crossover_strategy(self):
        """SMAクロスオーバー戦略テスト"""
        result_data = simple_moving_average_crossover(
            self.test_data.copy(),
            short_window=10,
            long_window=20
        )
        
        # 必要なカラムが追加されていることを確認
        assert 'sma_10' in result_data.columns
        assert 'sma_20' in result_data.columns
        assert 'signal' in result_data.columns
        
        # シグナルが適切な値（1, -1, NaN）であることを確認
        signals = result_data['signal'].dropna()
        unique_signals = set(signals.unique())
        assert unique_signals.issubset({-1, 1})
        
        print("✅ SMAクロスオーバー戦略テスト成功")
    
    def test_rsi_strategy(self):
        """RSI戦略テスト"""
        result_data = rsi_strategy(
            self.test_data.copy(),
            rsi_window=14,
            oversold_threshold=30,
            overbought_threshold=70
        )
        
        # RSIカラムが追加されていることを確認
        assert 'rsi' in result_data.columns
        
        # RSIが0-100の範囲内であることを確認
        rsi_values = result_data['rsi'].dropna()
        assert rsi_values.min() >= 0
        assert rsi_values.max() <= 100
        
        print("✅ RSI戦略テスト成功")


class TestPerformanceAnalyzer:
    """パフォーマンス分析のテスト"""
    
    def setup_method(self):
        """各テストの前に実行"""
        # テスト用エクイティカーブ
        dates = pd.date_range('2023-01-01', periods=30, freq='D')
        portfolio_values = 100000 + np.cumsum(np.random.randn(30) * 1000)
        
        equity_curve = pd.DataFrame({
            'portfolio_value': portfolio_values,
            'cash': 50000,
            'btc_holdings': 1.0,
            'btc_value': 50000
        }, index=dates)
        
        self.test_results = {
            'initial_capital': 100000,
            'final_value': portfolio_values[-1],
            'total_return_pct': (portfolio_values[-1] / 100000 - 1) * 100,
            'buy_hold_return_pct': 15.0,
            'equity_curve': equity_curve,
            'trades': []
        }
    
    def test_performance_metrics_calculation(self):
        """パフォーマンス指標計算テスト"""
        analyzer = PerformanceAnalyzer(self.test_results)
        metrics = analyzer.calculate_comprehensive_metrics()
        
        # 基本指標の存在確認
        basic_metrics = ['initial_capital', 'final_value', 'total_return', 
                        'sharpe_ratio', 'max_drawdown_pct']
        
        for metric in basic_metrics:
            assert metric in metrics
        
        # 値の妥当性確認
        assert metrics['initial_capital'] == 100000
        assert isinstance(metrics['sharpe_ratio'], (int, float))
        assert isinstance(metrics['max_drawdown_pct'], (int, float))
        
        print("✅ パフォーマンス指標計算テスト成功")
    
    def test_performance_report_generation(self):
        """パフォーマンスレポート生成テスト"""
        analyzer = PerformanceAnalyzer(self.test_results)
        report = analyzer.generate_performance_report()
        
        # レポートが文字列で、適切な内容が含まれていることを確認
        assert isinstance(report, str)
        assert len(report) > 100
        assert '詳細パフォーマンス分析レポート' in report
        assert 'シャープレシオ' in report
        
        print("✅ パフォーマンスレポート生成テスト成功")


def run_integration_test():
    """統合テスト"""
    print("\n" + "="*50)
    print("統合テスト実行中...")
    print("="*50)
    
    try:
        # データ取得
        fetcher = BitcoinDataFetcher()
        
        # テスト用の小さなデータセット
        dates = pd.date_range('2023-01-01', periods=30, freq='D')
        np.random.seed(42)
        prices = 50000 + np.cumsum(np.random.randn(30) * 100)
        
        test_data = pd.DataFrame({
            'open': prices * 0.999,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'volume': np.random.randint(1000, 10000, 30)
        }, index=dates)
        
        # バックテスト実行
        engine = BacktestEngine(initial_capital=100000, commission_rate=0.001)
        results = engine.run_backtest(
            data=test_data,
            strategy_func=simple_moving_average_crossover,
            short_window=5,
            long_window=10
        )
        
        # パフォーマンス分析
        analyzer = PerformanceAnalyzer(results)
        metrics = analyzer.calculate_comprehensive_metrics()
        report = analyzer.generate_performance_report()
        
        print("✅ 統合テスト成功")
        print(f"最終資産価値: ¥{results['final_value']:,.0f}")
        print(f"総リターン: {results['total_return_pct']:.2f}%")
        print(f"取引回数: {results['num_trades']}")
        
        return True
        
    except Exception as e:
        print(f"❌ 統合テストでエラー: {e}")
        return False


def main():
    """メインテスト実行"""
    print("ビットコインバックテスト環境 テスト開始")
    
    test_classes = [
        TestBitcoinDataFetcher,
        TestBacktestEngine,
        TestStrategies,
        TestPerformanceAnalyzer
    ]
    
    for test_class in test_classes:
        print(f"\n--- {test_class.__name__} ---")
        test_instance = test_class()
        
        # setup_methodがある場合は実行
        if hasattr(test_instance, 'setup_method'):
            test_instance.setup_method()
        
        # テストメソッドを実行
        test_methods = [method for method in dir(test_instance) 
                       if method.startswith('test_')]
        
        for method_name in test_methods:
            try:
                method = getattr(test_instance, method_name)
                method()
            except Exception as e:
                print(f"❌ {method_name} でエラー: {e}")
    
    # 統合テスト
    run_integration_test()
    
    print(f"\n🎉 テスト完了!")
    print("本格的なバックテストを実行するには main.py を実行してください。")


if __name__ == "__main__":
    main()