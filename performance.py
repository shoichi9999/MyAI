"""
パフォーマンス評価指標計算モジュール
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class PerformanceAnalyzer:
    """パフォーマンス分析クラス"""
    
    def __init__(self, results: Dict[str, Any]):
        """
        Args:
            results: バックテスト結果辞書
        """
        self.results = results
        self.equity_curve = results.get('equity_curve', pd.DataFrame())
        self.trades = results.get('trades', [])
        self.initial_capital = results.get('initial_capital', 100000)
        
    def calculate_comprehensive_metrics(self) -> Dict[str, Any]:
        """包括的なパフォーマンス指標を計算"""
        
        if self.equity_curve.empty:
            return {}
        
        equity_curve = self.equity_curve.copy()
        
        # 基本的なリターン計算
        returns = self._calculate_returns(equity_curve)
        
        # 各種指標計算
        metrics = {}
        
        # === 基本指標 ===
        metrics.update(self._calculate_basic_metrics(equity_curve))
        
        # === リスク指標 ===
        metrics.update(self._calculate_risk_metrics(returns))
        
        # === ドローダウン指標 ===
        metrics.update(self._calculate_drawdown_metrics(equity_curve))
        
        # === 取引関連指標 ===
        metrics.update(self._calculate_trade_metrics())
        
        # === 時系列指標 ===
        metrics.update(self._calculate_time_series_metrics(returns))
        
        # === 比較指標 ===
        metrics.update(self._calculate_benchmark_metrics())
        
        return metrics
    
    def _calculate_returns(self, equity_curve: pd.DataFrame) -> pd.Series:
        """日次リターンを計算"""
        return equity_curve['portfolio_value'].pct_change().dropna()
    
    def _calculate_basic_metrics(self, equity_curve: pd.DataFrame) -> Dict[str, Any]:
        """基本指標を計算"""
        final_value = equity_curve['portfolio_value'].iloc[-1]
        
        return {
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return': (final_value / self.initial_capital - 1) * 100,
            'total_return_abs': final_value - self.initial_capital,
            'trading_days': len(equity_curve),
            'trading_period_years': len(equity_curve) / 252  # 年換算
        }
    
    def _calculate_risk_metrics(self, returns: pd.Series) -> Dict[str, Any]:
        """リスク指標を計算"""
        if len(returns) < 2:
            return {'sharpe_ratio': 0, 'sortino_ratio': 0, 'volatility_annual': 0}
        
        # 年率換算係数
        annual_factor = np.sqrt(252)
        
        # 基本統計
        mean_return = returns.mean()
        std_return = returns.std()
        
        # シャープレシオ
        sharpe_ratio = (mean_return / std_return * annual_factor) if std_return > 0 else 0
        
        # ソルティーノレシオ（下方偏差のみを考慮）
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 1 else std_return
        sortino_ratio = (mean_return / downside_std * annual_factor) if downside_std > 0 else 0
        
        # カルマーレシオは後でドローダウンと一緒に計算
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'volatility_annual': std_return * annual_factor * 100,
            'mean_return_daily': mean_return * 100,
            'std_return_daily': std_return * 100,
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis()
        }
    
    def _calculate_drawdown_metrics(self, equity_curve: pd.DataFrame) -> Dict[str, Any]:
        """ドローダウン指標を計算"""
        portfolio_values = equity_curve['portfolio_value']
        
        # ランニング最大値
        running_max = portfolio_values.expanding().max()
        
        # ドローダウン（金額ベース）
        drawdown_abs = portfolio_values - running_max
        
        # ドローダウン（パーセンテージベース）
        drawdown_pct = (drawdown_abs / running_max * 100)
        
        # 最大ドローダウン
        max_drawdown = drawdown_pct.min()
        max_drawdown_abs = drawdown_abs.min()
        
        # 最大ドローダウンの期間を特定
        max_dd_idx = drawdown_pct.idxmin()
        
        # カルマーレシオ（年率リターン / 最大ドローダウン）
        annual_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) ** (252 / len(portfolio_values)) - 1
        calmar_ratio = (annual_return * 100 / abs(max_drawdown)) if max_drawdown < 0 else 0
        
        # 水中時間（ドローダウン中の期間）
        underwater_periods = self._calculate_underwater_periods(portfolio_values, running_max)
        
        return {
            'max_drawdown_pct': max_drawdown,
            'max_drawdown_abs': max_drawdown_abs,
            'max_drawdown_date': max_dd_idx,
            'calmar_ratio': calmar_ratio,
            'avg_drawdown_pct': drawdown_pct[drawdown_pct < 0].mean() if len(drawdown_pct[drawdown_pct < 0]) > 0 else 0,
            'underwater_periods': underwater_periods,
            'current_drawdown_pct': drawdown_pct.iloc[-1]
        }
    
    def _calculate_underwater_periods(self, portfolio_values: pd.Series, running_max: pd.Series) -> Dict[str, Any]:
        """水中時間を計算"""
        is_underwater = portfolio_values < running_max
        
        if not is_underwater.any():
            return {'num_periods': 0, 'max_period_days': 0, 'total_days': 0}
        
        # 連続する水中期間を特定
        underwater_periods = []
        current_period = 0
        
        for underwater in is_underwater:
            if underwater:
                current_period += 1
            else:
                if current_period > 0:
                    underwater_periods.append(current_period)
                current_period = 0
        
        # 最後が水中で終わっている場合
        if current_period > 0:
            underwater_periods.append(current_period)
        
        return {
            'num_periods': len(underwater_periods),
            'max_period_days': max(underwater_periods) if underwater_periods else 0,
            'total_days': sum(underwater_periods) if underwater_periods else 0,
            'avg_period_days': np.mean(underwater_periods) if underwater_periods else 0
        }
    
    def _calculate_trade_metrics(self) -> Dict[str, Any]:
        """取引関連指標を計算"""
        if not self.trades:
            return {'num_trades': 0}
        
        # 取引をペアに分ける（買い→売りのペア）
        trade_pairs = self._pair_trades()
        
        if not trade_pairs:
            return {'num_trades': len(self.trades)}
        
        # 各ペアの損益計算
        trade_returns = []
        for buy_trade, sell_trade in trade_pairs:
            pnl = (sell_trade.price - buy_trade.price) * buy_trade.quantity - buy_trade.commission - sell_trade.commission
            trade_returns.append(pnl)
        
        winning_trades = [pnl for pnl in trade_returns if pnl > 0]
        losing_trades = [pnl for pnl in trade_returns if pnl < 0]
        
        return {
            'num_trades': len(self.trades),
            'num_complete_trades': len(trade_pairs),
            'num_winning_trades': len(winning_trades),
            'num_losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(trade_pairs) * 100 if trade_pairs else 0,
            'avg_winning_trade': np.mean(winning_trades) if winning_trades else 0,
            'avg_losing_trade': np.mean(losing_trades) if losing_trades else 0,
            'largest_winning_trade': max(winning_trades) if winning_trades else 0,
            'largest_losing_trade': min(losing_trades) if losing_trades else 0,
            'profit_factor': sum(winning_trades) / abs(sum(losing_trades)) if losing_trades else float('inf'),
            'total_commission': sum(trade.commission for trade in self.trades)
        }
    
    def _pair_trades(self) -> List[tuple]:
        """取引をペアに分ける"""
        pairs = []
        buy_trades = []
        
        for trade in self.trades:
            if trade.action == 'BUY':
                buy_trades.append(trade)
            elif trade.action == 'SELL' and buy_trades:
                # FIFO（先入先出）でペアリング
                buy_trade = buy_trades.pop(0)
                pairs.append((buy_trade, trade))
        
        return pairs
    
    def _calculate_time_series_metrics(self, returns: pd.Series) -> Dict[str, Any]:
        """時系列関連指標を計算"""
        if len(returns) < 2:
            return {}
        
        # 月次リターン（概算）
        monthly_returns = returns.groupby(returns.index.to_period('M')).apply(lambda x: (1 + x).prod() - 1)
        
        # 最大連続勝利・敗北日数
        win_streak, loss_streak = self._calculate_streaks(returns)
        
        return {
            'best_month_return': monthly_returns.max() * 100 if len(monthly_returns) > 0 else 0,
            'worst_month_return': monthly_returns.min() * 100 if len(monthly_returns) > 0 else 0,
            'positive_months': (monthly_returns > 0).sum() if len(monthly_returns) > 0 else 0,
            'negative_months': (monthly_returns < 0).sum() if len(monthly_returns) > 0 else 0,
            'max_consecutive_wins': win_streak,
            'max_consecutive_losses': loss_streak,
            'positive_days': (returns > 0).sum(),
            'negative_days': (returns < 0).sum()
        }
    
    def _calculate_streaks(self, returns: pd.Series) -> tuple:
        """連続勝敗を計算"""
        wins = returns > 0
        losses = returns < 0
        
        max_win_streak = 0
        max_loss_streak = 0
        current_win_streak = 0
        current_loss_streak = 0
        
        for is_win in wins:
            if is_win:
                current_win_streak += 1
                current_loss_streak = 0
                max_win_streak = max(max_win_streak, current_win_streak)
            else:
                current_loss_streak += 1
                current_win_streak = 0
                max_loss_streak = max(max_loss_streak, current_loss_streak)
        
        return max_win_streak, max_loss_streak
    
    def _calculate_benchmark_metrics(self) -> Dict[str, Any]:
        """ベンチマーク比較指標を計算"""
        buy_hold_return = self.results.get('buy_hold_return_pct', 0)
        strategy_return = self.results.get('total_return_pct', 0)
        
        return {
            'buy_hold_return_pct': buy_hold_return,
            'excess_return_pct': strategy_return - buy_hold_return,
            'information_ratio': 0  # 実装を簡略化
        }
    
    def generate_performance_report(self) -> str:
        """パフォーマンスレポートを生成"""
        metrics = self.calculate_comprehensive_metrics()
        
        if not metrics:
            return "パフォーマンス分析に十分なデータがありません。"
        
        report = []
        report.append("=" * 80)
        report.append("詳細パフォーマンス分析レポート")
        report.append("=" * 80)
        
        # 基本指標
        report.append("\n【基本指標】")
        report.append(f"初期資金: ¥{metrics.get('initial_capital', 0):,.0f}")
        report.append(f"最終資産価値: ¥{metrics.get('final_value', 0):,.0f}")
        report.append(f"総リターン: {metrics.get('total_return', 0):.2f}%")
        report.append(f"取引期間: {metrics.get('trading_days', 0)} 日 ({metrics.get('trading_period_years', 0):.2f} 年)")
        
        # リスク指標
        report.append(f"\n【リスク指標】")
        report.append(f"シャープレシオ: {metrics.get('sharpe_ratio', 0):.3f}")
        report.append(f"ソルティーノレシオ: {metrics.get('sortino_ratio', 0):.3f}")
        report.append(f"カルマーレシオ: {metrics.get('calmar_ratio', 0):.3f}")
        report.append(f"年率ボラティリティ: {metrics.get('volatility_annual', 0):.2f}%")
        report.append(f"歪度: {metrics.get('skewness', 0):.3f}")
        report.append(f"尖度: {metrics.get('kurtosis', 0):.3f}")
        
        # ドローダウン指標
        report.append(f"\n【ドローダウン指標】")
        report.append(f"最大ドローダウン: {metrics.get('max_drawdown_pct', 0):.2f}%")
        report.append(f"最大ドローダウン日: {metrics.get('max_drawdown_date', 'N/A')}")
        report.append(f"平均ドローダウン: {metrics.get('avg_drawdown_pct', 0):.2f}%")
        report.append(f"現在のドローダウン: {metrics.get('current_drawdown_pct', 0):.2f}%")
        
        # 取引指標
        report.append(f"\n【取引指標】")
        report.append(f"総取引数: {metrics.get('num_trades', 0)}")
        report.append(f"完了取引数: {metrics.get('num_complete_trades', 0)}")
        report.append(f"勝率: {metrics.get('win_rate', 0):.1f}%")
        report.append(f"プロフィットファクター: {metrics.get('profit_factor', 0):.2f}")
        report.append(f"平均勝利取引: ¥{metrics.get('avg_winning_trade', 0):,.2f}")
        report.append(f"平均敗北取引: ¥{metrics.get('avg_losing_trade', 0):,.2f}")
        
        # ベンチマーク比較
        report.append(f"\n【ベンチマーク比較】")
        report.append(f"Buy & Hold リターン: {metrics.get('buy_hold_return_pct', 0):.2f}%")
        report.append(f"超過リターン: {metrics.get('excess_return_pct', 0):.2f}%")
        
        # 月次統計
        if 'best_month_return' in metrics:
            report.append(f"\n【月次統計】")
            report.append(f"最良月リターン: {metrics.get('best_month_return', 0):.2f}%")
            report.append(f"最悪月リターン: {metrics.get('worst_month_return', 0):.2f}%")
            report.append(f"プラス月数: {metrics.get('positive_months', 0)}")
            report.append(f"マイナス月数: {metrics.get('negative_months', 0)}")
        
        report.append("=" * 80)
        
        return "\n".join(report)


def main():
    """動作テスト"""
    # サンプルデータでテスト
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    equity_curve = pd.DataFrame({
        'portfolio_value': np.random.walk(100, 100000),
        'cash': 50000,
        'btc_holdings': 1.0,
        'btc_value': 50000
    }, index=dates)
    
    sample_results = {
        'initial_capital': 100000,
        'final_value': 120000,
        'total_return_pct': 20.0,
        'buy_hold_return_pct': 15.0,
        'equity_curve': equity_curve,
        'trades': []
    }
    
    analyzer = PerformanceAnalyzer(sample_results)
    report = analyzer.generate_performance_report()
    print(report)


if __name__ == "__main__":
    main()