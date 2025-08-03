"""
Web用のインタラクティブ可視化モジュール
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import json


class WebVisualizer:
    """Web用インタラクティブ可視化クラス"""
    
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
    
    def create_price_chart(self) -> str:
        """価格チャートとトレードシグナルのPlotlyグラフを作成"""
        
        fig = go.Figure()
        
        # 価格チャート（ローソク足）
        fig.add_trace(go.Candlestick(
            x=self.data.index,
            open=self.data['open'],
            high=self.data['high'],
            low=self.data['low'],
            close=self.data['close'],
            name='BTC価格',
            increasing=dict(line=dict(color='#00ff00')),
            decreasing=dict(line=dict(color='#ff0000'))
        ))
        
        # 移動平均線があれば追加
        if 'sma_20' in self.data.columns:
            fig.add_trace(go.Scatter(
                x=self.data.index,
                y=self.data['sma_20'],
                name='SMA20',
                line=dict(color='orange', width=2)
            ))
        
        if 'sma_50' in self.data.columns:
            fig.add_trace(go.Scatter(
                x=self.data.index,
                y=self.data['sma_50'],
                name='SMA50',
                line=dict(color='purple', width=2)
            ))
        
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
            fig.add_trace(go.Scatter(
                x=buy_dates,
                y=buy_prices,
                mode='markers',
                marker=dict(symbol='triangle-up', size=15, color='green'),
                name='買いシグナル'
            ))
        
        if sell_signals:
            sell_dates, sell_prices = zip(*sell_signals)
            fig.add_trace(go.Scatter(
                x=sell_dates,
                y=sell_prices,
                mode='markers',
                marker=dict(symbol='triangle-down', size=15, color='red'),
                name='売りシグナル'
            ))
        
        fig.update_layout(
            title='ビットコイン価格チャートと取引シグナル',
            xaxis_title='日付',
            yaxis_title='価格 (USD)',
            template='plotly_white',
            height=600,
            showlegend=True
        )
        
        return fig.to_html(include_plotlyjs='cdn', div_id='price-chart')
    
    def create_equity_curve_chart(self) -> str:
        """エクイティカーブチャートを作成"""
        
        if self.equity_curve.empty:
            return "<div>エクイティカーブデータがありません</div>"
        
        fig = go.Figure()
        
        # 戦略のエクイティカーブ
        fig.add_trace(go.Scatter(
            x=self.equity_curve.index,
            y=self.equity_curve['portfolio_value'],
            name='戦略',
            line=dict(color='blue', width=3)
        ))
        
        # Buy & Hold比較
        initial_price = self.data['close'].iloc[0]
        buy_hold_values = (self.data['close'] / initial_price) * self.results['initial_capital']
        
        fig.add_trace(go.Scatter(
            x=self.data.index,
            y=buy_hold_values,
            name='Buy & Hold',
            line=dict(color='gray', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title='エクイティカーブ比較',
            xaxis_title='日付',
            yaxis_title='ポートフォリオ価値 (円)',
            template='plotly_white',
            height=500
        )
        
        return fig.to_html(include_plotlyjs='cdn', div_id='equity-chart')
    
    def create_drawdown_chart(self) -> str:
        """ドローダウンチャートを作成"""
        
        if self.equity_curve.empty:
            return "<div>ドローダウンデータがありません</div>"
        
        portfolio_values = self.equity_curve['portfolio_value']
        running_max = portfolio_values.expanding().max()
        drawdown = (portfolio_values - running_max) / running_max * 100
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown,
            fill='tonexty',
            fillcolor='rgba(255, 0, 0, 0.3)',
            line=dict(color='red'),
            name='ドローダウン'
        ))
        
        fig.add_hline(y=0, line_dash="dash", line_color="black")
        
        fig.update_layout(
            title='ドローダウン',
            xaxis_title='日付',
            yaxis_title='ドローダウン (%)',
            template='plotly_white',
            height=400
        )
        
        return fig.to_html(include_plotlyjs='cdn', div_id='drawdown-chart')
    
    def create_monthly_returns_chart(self) -> str:
        """月次リターンチャートを作成"""
        
        if self.equity_curve.empty:
            return "<div>月次リターンデータがありません</div>"
        
        # 月次リターン計算
        portfolio_values = self.equity_curve['portfolio_value']
        monthly_values = portfolio_values.resample('M').last()
        monthly_returns = monthly_values.pct_change().dropna() * 100
        
        if len(monthly_returns) == 0:
            return "<div>月次リターンデータが不足しています</div>"
        
        # 色を設定（プラスは緑、マイナスは赤）
        colors = ['green' if x >= 0 else 'red' for x in monthly_returns]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=monthly_returns.index,
            y=monthly_returns.values,
            marker_color=colors,
            name='月次リターン'
        ))
        
        fig.add_hline(y=0, line_dash="dash", line_color="black")
        
        fig.update_layout(
            title='月次リターン',
            xaxis_title='月',
            yaxis_title='リターン (%)',
            template='plotly_white',
            height=400
        )
        
        return fig.to_html(include_plotlyjs='cdn', div_id='monthly-returns-chart')
    
    def create_trade_analysis_chart(self) -> str:
        """取引分析チャートを作成"""
        
        if not self.trades:
            return "<div>取引データがありません</div>"
        
        # 取引を買い・売りで分類
        buy_count = len([t for t in self.trades if t.action == 'BUY'])
        sell_count = len([t for t in self.trades if t.action == 'SELL'])
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=['買い注文', '売り注文'],
            y=[buy_count, sell_count],
            marker_color=['green', 'red'],
            name='取引数'
        ))
        
        fig.update_layout(
            title='取引統計',
            xaxis_title='取引タイプ',
            yaxis_title='回数',
            template='plotly_white',
            height=400
        )
        
        return fig.to_html(include_plotlyjs='cdn', div_id='trade-analysis-chart')
    
    def create_comprehensive_dashboard(self) -> str:
        """包括的なダッシュボードを作成"""
        
        # サブプロットを作成
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('価格チャート', 'エクイティカーブ', 
                          'ドローダウン', '月次リターン',
                          'ボリューム', '取引統計'),
            specs=[[{"colspan": 2}, None],
                  [{"secondary_y": False}, {"secondary_y": False}],
                  [{"secondary_y": False}, {"type": "bar"}]],
            vertical_spacing=0.08
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
        
        # ドローダウン
        if not self.equity_curve.empty:
            portfolio_values = self.equity_curve['portfolio_value']
            running_max = portfolio_values.expanding().max()
            drawdown = (portfolio_values - running_max) / running_max * 100
            
            fig.add_trace(
                go.Scatter(x=drawdown.index, y=drawdown,
                          fill='tonexty', name='ドローダウン',
                          line=dict(color='red')),
                row=2, col=2
            )
        
        # 月次リターン
        if not self.equity_curve.empty:
            portfolio_values = self.equity_curve['portfolio_value']
            monthly_values = portfolio_values.resample('M').last()
            monthly_returns = monthly_values.pct_change().dropna() * 100
            
            if len(monthly_returns) > 0:
                colors = ['green' if x >= 0 else 'red' for x in monthly_returns]
                fig.add_trace(
                    go.Bar(x=monthly_returns.index, y=monthly_returns.values,
                          marker_color=colors, name='月次リターン'),
                    row=3, col=1
                )
        
        # ボリューム
        fig.add_trace(
            go.Bar(x=self.data.index, y=self.data['volume'], 
                  name='ボリューム', marker_color='lightblue'),
            row=3, col=2
        )
        
        fig.update_layout(
            title='包括的バックテスト分析ダッシュボード',
            height=1000,
            showlegend=True,
            template='plotly_white'
        )
        
        return fig.to_html(include_plotlyjs='cdn', div_id='comprehensive-dashboard')
    
    def create_performance_summary_table(self) -> str:
        """パフォーマンス要約テーブルをHTML形式で作成"""
        
        from performance import PerformanceAnalyzer
        analyzer = PerformanceAnalyzer(self.results)
        metrics = analyzer.calculate_comprehensive_metrics()
        
        if not metrics:
            return "<div>パフォーマンス分析データがありません</div>"
        
        # 主要指標を選択
        key_metrics = [
            ('総リターン', f"{metrics.get('total_return', 0):.2f}%"),
            ('シャープレシオ', f"{metrics.get('sharpe_ratio', 0):.3f}"),
            ('ソルティーノレシオ', f"{metrics.get('sortino_ratio', 0):.3f}"),
            ('カルマーレシオ', f"{metrics.get('calmar_ratio', 0):.3f}"),
            ('最大ドローダウン', f"{metrics.get('max_drawdown_pct', 0):.2f}%"),
            ('年率ボラティリティ', f"{metrics.get('volatility_annual', 0):.2f}%"),
            ('取引回数', f"{metrics.get('num_trades', 0)}"),
            ('勝率', f"{metrics.get('win_rate', 0):.1f}%"),
            ('プロフィットファクター', f"{metrics.get('profit_factor', 0):.2f}"),
            ('Buy&Hold超過リターン', f"{metrics.get('excess_return_pct', 0):+.2f}%")
        ]
        
        html = """
        <div class="performance-table">
            <h3>パフォーマンス要約</h3>
            <table class="table table-striped">
                <thead>
                    <tr>
                        <th>指標</th>
                        <th>値</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for metric, value in key_metrics:
            html += f"""
                    <tr>
                        <td>{metric}</td>
                        <td><strong>{value}</strong></td>
                    </tr>
            """
        
        html += """
                </tbody>
            </table>
        </div>
        """
        
        return html


def create_sample_visualization():
    """サンプル可視化を作成（テスト用）"""
    
    # サンプルデータ作成
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    np.random.seed(42)
    prices = 50000 + np.cumsum(np.random.randn(100) * 100)
    
    data = pd.DataFrame({
        'open': prices * 0.999,
        'high': prices * 1.01,
        'low': prices * 0.99,
        'close': prices,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    equity_curve = pd.DataFrame({
        'portfolio_value': 100000 + np.cumsum(np.random.randn(100) * 500),
        'cash': 50000,
        'btc_holdings': 1.0,
        'btc_value': 50000
    }, index=dates)
    
    results = {
        'initial_capital': 100000,
        'final_value': 120000,
        'total_return_pct': 20.0,
        'equity_curve': equity_curve,
        'trades': []
    }
    
    visualizer = WebVisualizer(results, data)
    return visualizer.create_price_chart()


if __name__ == "__main__":
    # テスト実行
    chart_html = create_sample_visualization()
    print("Web可視化モジュールのテスト完了")