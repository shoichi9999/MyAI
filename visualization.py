"""
バックテスト結果可視化モジュール
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import seaborn as sns
from typing import Dict, Any, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class BacktestVisualizer:
    """バックテスト結果可視化クラス"""
    
    def __init__(self, results: Dict[str, Any], data: pd.DataFrame):
        """
        Args:
            results: バックテスト結果辞書
            data: 元の価格データ（シグナル付き）
        """
        self.results = results
        self.data = data
        self.equity_curve = results.get('equity_curve', pd.DataFrame())
        self.trades = results.get('trades', [])
        
        # スタイル設定
        plt.style.use('default')
        sns.set_palette("husl")
        self.colors = {
            'price': '#1f77b4',
            'buy_signal': '#2ca02c',
            'sell_signal': '#d62728',
            'equity': '#ff7f0e',
            'benchmark': '#9467bd',
            'drawdown': '#e377c2'
        }
    
    def plot_comprehensive_analysis(self, figsize: Tuple[int, int] = (16, 12)) -> plt.Figure:
        """包括的な分析チャートを作成"""
        
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(3, 2, height_ratios=[2, 1, 1], hspace=0.3, wspace=0.3)
        
        # 1. 価格チャートとシグナル
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_price_and_signals(ax1)
        
        # 2. エクイティカーブ
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_equity_curve(ax2)
        
        # 3. ドローダウン
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_drawdown(ax3)
        
        # 4. 月次リターン分布
        ax4 = fig.add_subplot(gs[2, 0])
        self._plot_monthly_returns(ax4)
        
        # 5. 取引統計
        ax5 = fig.add_subplot(gs[2, 1])
        self._plot_trade_statistics(ax5)
        
        plt.suptitle('ビットコイン戦略バックテスト分析', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        return fig
    
    def _plot_price_and_signals(self, ax):
        """価格チャートとエントリー・エグジットシグナルをプロット"""
        
        # 価格チャート
        ax.plot(self.data.index, self.data['close'], 
               color=self.colors['price'], linewidth=1, label='BTC価格')
        
        # 移動平均があれば表示
        if 'sma_20' in self.data.columns:
            ax.plot(self.data.index, self.data['sma_20'], 
                   color='orange', alpha=0.7, linewidth=1, label='SMA20')
        if 'sma_50' in self.data.columns:
            ax.plot(self.data.index, self.data['sma_50'], 
                   color='purple', alpha=0.7, linewidth=1, label='SMA50')
        
        # トレードシグナル
        buy_signals = []
        sell_signals = []
        
        for trade in self.trades:
            if trade.action == 'BUY':
                buy_signals.append((trade.timestamp, trade.price))
            elif trade.action == 'SELL':
                sell_signals.append((trade.timestamp, trade.price))
        
        if buy_signals:
            buy_dates, buy_prices = zip(*buy_signals)
            ax.scatter(buy_dates, buy_prices, color=self.colors['buy_signal'], 
                      marker='^', s=100, label='買いシグナル', zorder=5)
        
        if sell_signals:
            sell_dates, sell_prices = zip(*sell_signals)
            ax.scatter(sell_dates, sell_prices, color=self.colors['sell_signal'], 
                      marker='v', s=100, label='売りシグナル', zorder=5)
        
        ax.set_title('価格チャートと取引シグナル')
        ax.set_ylabel('価格 (USD)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # X軸の日付フォーマット
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    def _plot_equity_curve(self, ax):
        """エクイティカーブをプロット"""
        
        if self.equity_curve.empty:
            ax.text(0.5, 0.5, 'エクイティカーブデータなし', 
                   transform=ax.transAxes, ha='center', va='center')
            return
        
        # 戦略のエクイティカーブ
        ax.plot(self.equity_curve.index, self.equity_curve['portfolio_value'],
               color=self.colors['equity'], linewidth=2, label='戦略')
        
        # Buy & Hold比較
        initial_price = self.data['close'].iloc[0]
        final_price = self.data['close'].iloc[-1]
        buy_hold_values = (self.data['close'] / initial_price) * self.results['initial_capital']
        
        ax.plot(self.data.index, buy_hold_values,
               color=self.colors['benchmark'], linewidth=1, 
               alpha=0.7, label='Buy & Hold', linestyle='--')
        
        ax.set_title('エクイティカーブ比較')
        ax.set_ylabel('ポートフォリオ価値 (USD)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # パフォーマンス情報を追加
        final_value = self.equity_curve['portfolio_value'].iloc[-1]
        total_return = (final_value / self.results['initial_capital'] - 1) * 100
        ax.text(0.02, 0.98, f'総リターン: {total_return:.1f}%', 
               transform=ax.transAxes, va='top', 
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def _plot_drawdown(self, ax):
        """ドローダウンをプロット"""
        
        if self.equity_curve.empty:
            ax.text(0.5, 0.5, 'ドローダウンデータなし', 
                   transform=ax.transAxes, ha='center', va='center')
            return
        
        portfolio_values = self.equity_curve['portfolio_value']
        running_max = portfolio_values.expanding().max()
        drawdown = (portfolio_values - running_max) / running_max * 100
        
        ax.fill_between(drawdown.index, drawdown, 0, 
                       color=self.colors['drawdown'], alpha=0.7, label='ドローダウン')
        
        ax.set_title('ドローダウン')
        ax.set_ylabel('ドローダウン (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 最大ドローダウン情報
        max_dd = drawdown.min()
        ax.text(0.02, 0.02, f'最大DD: {max_dd:.1f}%', 
               transform=ax.transAxes, va='bottom',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def _plot_monthly_returns(self, ax):
        """月次リターン分布をプロット"""
        
        if self.equity_curve.empty:
            ax.text(0.5, 0.5, '月次リターンデータなし', 
                   transform=ax.transAxes, ha='center', va='center')
            return
        
        # 月次リターン計算
        portfolio_values = self.equity_curve['portfolio_value']
        monthly_values = portfolio_values.resample('M').last()
        monthly_returns = monthly_values.pct_change().dropna() * 100
        
        if len(monthly_returns) == 0:
            ax.text(0.5, 0.5, '月次リターンデータ不足', 
                   transform=ax.transAxes, ha='center', va='center')
            return
        
        # ヒストグラム
        n_bins = min(len(monthly_returns), 10)
        colors = ['red' if x < 0 else 'green' for x in monthly_returns]
        ax.hist(monthly_returns, bins=n_bins, alpha=0.7, color='skyblue', edgecolor='black')
        
        ax.set_title('月次リターン分布')
        ax.set_xlabel('月次リターン (%)')
        ax.set_ylabel('頻度')
        ax.grid(True, alpha=0.3)
        
        # 統計情報
        mean_return = monthly_returns.mean()
        std_return = monthly_returns.std()
        ax.axvline(mean_return, color='red', linestyle='--', alpha=0.8, label=f'平均: {mean_return:.1f}%')
        ax.legend()
    
    def _plot_trade_statistics(self, ax):
        """取引統計をプロット"""
        
        if not self.trades:
            ax.text(0.5, 0.5, '取引データなし', 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title('取引統計')
            return
        
        # 取引統計の計算
        num_trades = len(self.trades)
        buy_trades = len([t for t in self.trades if t.action == 'BUY'])
        sell_trades = len([t for t in self.trades if t.action == 'SELL'])
        total_commission = sum(t.commission for t in self.trades)
        
        # バーチャート
        categories = ['買い', '売り']
        values = [buy_trades, sell_trades]
        colors = [self.colors['buy_signal'], self.colors['sell_signal']]
        
        bars = ax.bar(categories, values, color=colors, alpha=0.7)
        
        # 値をバーの上に表示
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{int(value)}', ha='center', va='bottom')
        
        ax.set_title('取引統計')
        ax.set_ylabel('取引数')
        
        # 手数料情報を追加
        ax.text(0.02, 0.98, f'総手数料: ${total_commission:.2f}', 
               transform=ax.transAxes, va='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def plot_detailed_price_analysis(self, figsize: Tuple[int, int] = (14, 10)) -> plt.Figure:
        """詳細な価格分析チャート"""
        
        fig, axes = plt.subplots(3, 1, figsize=figsize, height_ratios=[3, 1, 1])
        fig.suptitle('詳細価格分析', fontsize=16, fontweight='bold')
        
        # 1. 価格とボリューム
        ax1 = axes[0]
        ax1.plot(self.data.index, self.data['close'], color=self.colors['price'], linewidth=1)
        ax1.set_title('ビットコイン価格')
        ax1.set_ylabel('価格 (USD)')
        ax1.grid(True, alpha=0.3)
        
        # 2. ボリューム
        ax2 = axes[1]
        ax2.bar(self.data.index, self.data['volume'], color='lightblue', alpha=0.7)
        ax2.set_title('取引ボリューム')
        ax2.set_ylabel('ボリューム')
        ax2.grid(True, alpha=0.3)
        
        # 3. 日次リターン
        ax3 = axes[2]
        daily_returns = self.data['close'].pct_change() * 100
        colors = ['red' if x < 0 else 'green' for x in daily_returns]
        ax3.bar(self.data.index, daily_returns, color=colors, alpha=0.7)
        ax3.set_title('日次リターン')
        ax3.set_ylabel('リターン (%)')
        ax3.set_xlabel('日付')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def save_all_charts(self, output_dir: str = 'charts'):
        """すべてのチャートをファイルに保存"""
        import os
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 包括的分析チャート
        fig1 = self.plot_comprehensive_analysis()
        fig1.savefig(f'{output_dir}/comprehensive_analysis.png', dpi=300, bbox_inches='tight')
        plt.close(fig1)
        
        # 詳細価格分析チャート
        fig2 = self.plot_detailed_price_analysis()
        fig2.savefig(f'{output_dir}/detailed_price_analysis.png', dpi=300, bbox_inches='tight')
        plt.close(fig2)
        
        print(f"チャートを {output_dir} ディレクトリに保存しました。")
    
    def show_interactive_dashboard(self):
        """インタラクティブダッシュボードを表示（Plotly使用）"""
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            # サブプロットを作成
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=('価格チャート', 'エクイティカーブ', 
                              'ボリューム', 'ドローダウン',
                              '日次リターン', '取引統計'),
                specs=[[{"colspan": 2}, None],
                      [{"secondary_y": False}, {"secondary_y": False}],
                      [{"secondary_y": False}, {"type": "bar"}]]
            )
            
            # 価格チャート
            fig.add_trace(
                go.Scatter(x=self.data.index, y=self.data['close'], 
                          name='BTC価格', line=dict(color='blue')),
                row=1, col=1
            )
            
            # エクイティカーブ
            if not self.equity_curve.empty:
                fig.add_trace(
                    go.Scatter(x=self.equity_curve.index, 
                              y=self.equity_curve['portfolio_value'],
                              name='戦略', line=dict(color='orange')),
                    row=2, col=1
                )
            
            # ボリューム
            fig.add_trace(
                go.Bar(x=self.data.index, y=self.data['volume'], 
                      name='ボリューム', marker_color='lightblue'),
                row=2, col=2
            )
            
            # レイアウト設定
            fig.update_layout(
                title='インタラクティブ バックテスト分析ダッシュボード',
                height=800,
                showlegend=True
            )
            
            fig.show()
            
        except ImportError:
            print("Plotlyがインストールされていません。pip install plotly でインストールしてください。")


def main():
    """動作テスト"""
    # サンプルデータでテスト
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    
    # サンプル価格データ
    np.random.seed(42)
    prices = 30000 + np.cumsum(np.random.randn(100) * 100)
    data = pd.DataFrame({
        'close': prices,
        'high': prices * 1.02,
        'low': prices * 0.98,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    # サンプルエクイティカーブ
    equity_curve = pd.DataFrame({
        'portfolio_value': 100000 + np.cumsum(np.random.randn(100) * 500),
        'cash': 50000,
        'btc_holdings': 1.0,
        'btc_value': 50000
    }, index=dates)
    
    # サンプル結果
    results = {
        'initial_capital': 100000,
        'final_value': 120000,
        'total_return_pct': 20.0,
        'equity_curve': equity_curve,
        'trades': []
    }
    
    # 可視化テスト
    visualizer = BacktestVisualizer(results, data)
    fig = visualizer.plot_comprehensive_analysis()
    plt.show()


if __name__ == "__main__":
    main()