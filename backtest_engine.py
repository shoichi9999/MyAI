"""
バックテストエンジン - トレード戦略の過去データでのテスト実行
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import copy


class Trade:
    """個別取引を表すクラス"""
    
    def __init__(self, 
                 timestamp: datetime,
                 action: str,  # 'BUY' or 'SELL'
                 price: float,
                 quantity: float,
                 commission: float = 0.0):
        self.timestamp = timestamp
        self.action = action
        self.price = price
        self.quantity = quantity
        self.commission = commission
        self.pnl = 0.0  # 損益（決済時に計算）


class Portfolio:
    """ポートフォリオ管理クラス"""
    
    def __init__(self, initial_capital: float = 100000, commission_rate: float = 0.001):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.btc_holdings = 0.0
        self.commission_rate = commission_rate
        self.trades: List[Trade] = []
        self.equity_curve = []
        
    def get_portfolio_value(self, current_price: float) -> float:
        """現在のポートフォリオ価値を計算"""
        return self.cash + (self.btc_holdings * current_price)
    
    def execute_trade(self, timestamp: datetime, action: str, price: float, quantity: float) -> bool:
        """取引を実行"""
        commission = quantity * price * self.commission_rate
        
        if action == 'BUY':
            total_cost = quantity * price + commission
            if self.cash >= total_cost:
                self.cash -= total_cost
                self.btc_holdings += quantity
                trade = Trade(timestamp, action, price, quantity, commission)
                self.trades.append(trade)
                return True
            else:
                return False  # 資金不足
                
        elif action == 'SELL':
            if self.btc_holdings >= quantity:
                proceeds = quantity * price - commission
                self.cash += proceeds
                self.btc_holdings -= quantity
                trade = Trade(timestamp, action, price, quantity, commission)
                self.trades.append(trade)
                return True
            else:
                return False  # 保有量不足
        
        return False
    
    def update_equity_curve(self, timestamp: datetime, price: float):
        """エクイティカーブを更新"""
        portfolio_value = self.get_portfolio_value(price)
        self.equity_curve.append({
            'timestamp': timestamp,
            'portfolio_value': portfolio_value,
            'cash': self.cash,
            'btc_holdings': self.btc_holdings,
            'btc_value': self.btc_holdings * price
        })


class BacktestEngine:
    """バックテストエンジンメインクラス"""
    
    def __init__(self, 
                 initial_capital: float = 100000,
                 commission_rate: float = 0.001):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.results = None
        
    def run_backtest(self, 
                    data: pd.DataFrame, 
                    strategy_func: callable,
                    **strategy_params) -> Dict[str, Any]:
        """
        バックテストを実行
        
        Args:
            data: OHLCV価格データ
            strategy_func: 戦略関数
            **strategy_params: 戦略パラメータ
            
        Returns:
            Dict: バックテスト結果
        """
        # ポートフォリオ初期化
        portfolio = Portfolio(self.initial_capital, self.commission_rate)
        
        # データをコピーして戦略指標を計算
        data_with_signals = data.copy()
        data_with_signals = strategy_func(data_with_signals, **strategy_params)
        
        # 各時点でのバックテスト実行
        for i, (timestamp, row) in enumerate(data_with_signals.iterrows()):
            current_price = row['close']
            
            # シグナルチェック
            if 'signal' in row and not pd.isna(row['signal']):
                signal = row['signal']
                
                if signal == 1:  # 買いシグナル
                    # 全資金でビットコインを購入
                    max_quantity = portfolio.cash / current_price * 0.98  # 手数料を考慮
                    if max_quantity > 0:
                        portfolio.execute_trade(timestamp, 'BUY', current_price, max_quantity)
                        
                elif signal == -1:  # 売りシグナル
                    # 全保有量を売却
                    if portfolio.btc_holdings > 0:
                        portfolio.execute_trade(timestamp, 'SELL', current_price, portfolio.btc_holdings)
            
            # エクイティカーブ更新
            portfolio.update_equity_curve(timestamp, current_price)
        
        # 結果をまとめる
        results = self._calculate_results(portfolio, data_with_signals)
        self.results = results
        
        return results
    
    def _calculate_results(self, portfolio: Portfolio, data: pd.DataFrame) -> Dict[str, Any]:
        """バックテスト結果を計算"""
        
        # エクイティカーブをDataFrameに変換
        equity_df = pd.DataFrame(portfolio.equity_curve)
        equity_df.set_index('timestamp', inplace=True)
        
        # 基本指標計算
        final_value = equity_df['portfolio_value'].iloc[-1]
        total_return = (final_value / self.initial_capital - 1) * 100
        
        # ドローダウン計算
        rolling_max = equity_df['portfolio_value'].expanding().max()
        drawdown = (equity_df['portfolio_value'] - rolling_max) / rolling_max * 100
        max_drawdown = drawdown.min()
        
        # リターン系列計算
        equity_df['daily_return'] = equity_df['portfolio_value'].pct_change()
        
        # シャープレシオ計算（年率換算）
        if len(equity_df['daily_return'].dropna()) > 1:
            daily_return_mean = equity_df['daily_return'].mean()
            daily_return_std = equity_df['daily_return'].std()
            sharpe_ratio = (daily_return_mean / daily_return_std * np.sqrt(252)) if daily_return_std > 0 else 0
        else:
            sharpe_ratio = 0
        
        # 取引統計
        num_trades = len(portfolio.trades)
        total_commission = sum(trade.commission for trade in portfolio.trades)
        
        # Buy & Holdとの比較
        buy_hold_return = (data['close'].iloc[-1] / data['close'].iloc[0] - 1) * 100
        
        results = {
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return_pct': total_return,
            'max_drawdown_pct': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'num_trades': num_trades,
            'total_commission': total_commission,
            'buy_hold_return_pct': buy_hold_return,
            'excess_return_pct': total_return - buy_hold_return,
            'equity_curve': equity_df,
            'trades': portfolio.trades,
            'final_cash': portfolio.cash,
            'final_btc_holdings': portfolio.btc_holdings
        }
        
        return results
    
    def print_results(self):
        """結果を表示"""
        if self.results is None:
            print("バックテストがまだ実行されていません。")
            return
        
        results = self.results
        
        print("=" * 50)
        print("バックテスト結果")
        print("=" * 50)
        print(f"初期資金: ¥{results['initial_capital']:,.0f}")
        print(f"最終資産価値: ¥{results['final_value']:,.0f}")
        print(f"総リターン: {results['total_return_pct']:.2f}%")
        print(f"最大ドローダウン: {results['max_drawdown_pct']:.2f}%")
        print(f"シャープレシオ: {results['sharpe_ratio']:.3f}")
        print(f"取引回数: {results['num_trades']}")
        print(f"総手数料: ¥{results['total_commission']:,.2f}")
        print()
        print("=" * 50)
        print("比較")
        print("=" * 50)
        print(f"Buy & Hold リターン: {results['buy_hold_return_pct']:.2f}%")
        print(f"戦略の超過リターン: {results['excess_return_pct']:.2f}%")
        print()
        print("=" * 50)
        print("最終ポジション")
        print("=" * 50)
        print(f"現金: ¥{results['final_cash']:,.2f}")
        print(f"BTC保有量: {results['final_btc_holdings']:.6f} BTC")


def simple_moving_average_strategy(data: pd.DataFrame, 
                                 short_window: int = 20, 
                                 long_window: int = 50) -> pd.DataFrame:
    """
    シンプル移動平均クロスオーバー戦略
    
    Args:
        data: OHLCV価格データ
        short_window: 短期移動平均期間
        long_window: 長期移動平均期間
        
    Returns:
        pd.DataFrame: シグナル付きデータ
    """
    data = data.copy()
    
    # 移動平均計算
    data[f'sma_{short_window}'] = data['close'].rolling(window=short_window).mean()
    data[f'sma_{long_window}'] = data['close'].rolling(window=long_window).mean()
    
    # シグナル生成
    data['signal'] = 0
    data['signal'][short_window:] = np.where(
        data[f'sma_{short_window}'][short_window:] > data[f'sma_{long_window}'][short_window:], 1, -1
    )
    
    # シグナル変化点のみを抽出（エントリー・エグジットポイント）
    data['position'] = data['signal'].diff()
    data['signal'] = np.where(data['position'] != 0, data['signal'], np.nan)
    
    return data


def main():
    """動作テスト"""
    from data_fetcher import BitcoinDataFetcher
    
    print("=== バックテストエンジン動作テスト ===")
    
    # データ取得
    fetcher = BitcoinDataFetcher()
    btc_data = fetcher.get_bitcoin_data(
        source="yahoo",
        start_date="2023-01-01",
        end_date="2024-01-01"
    )
    
    # バックテスト実行
    engine = BacktestEngine(initial_capital=1000000, commission_rate=0.001)
    results = engine.run_backtest(
        data=btc_data,
        strategy_func=simple_moving_average_strategy,
        short_window=10,
        long_window=30
    )
    
    # 結果表示
    engine.print_results()


if __name__ == "__main__":
    main()