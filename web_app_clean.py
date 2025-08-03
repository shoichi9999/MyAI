"""
Bitcoin Backtest Web Application (Clean Version)
"""

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from datetime import datetime, timedelta
import json
import traceback
import warnings
import pandas as pd
warnings.filterwarnings('ignore')

# Import custom modules directly (without main.py)
from data_fetcher import BitcoinDataFetcher
from backtest_engine import BacktestEngine
from performance import PerformanceAnalyzer
from web_visualizer import WebVisualizer
from strategies.moving_average_strategies import *
from strategies.momentum_strategies import *
from strategies.mean_reversion_strategies import *
from strategies.correlation_strategies import *
from strategies.statistical_strategies import *
from strategies.ml_strategies import *
from strategies.advanced_technical_strategies import *
from strategies.market_microstructure_strategies import *
from strategies.quantitative_strategies import *
from leaderboard_clean import StrategyLeaderboard

app = Flask(__name__)
app.secret_key = 'bitcoin_backtest_secret_key_2024'

# Available strategies dictionary
STRATEGIES = {
    # Basic Strategies
    'sma_crossover': {
        'name': 'シンプル移動平均クロスオーバー',
        'function': simple_moving_average_crossover,
        'params': ['short_window', 'long_window']
    },
    'ema_crossover': {
        'name': 'Exponential Moving Average Crossover',
        'function': exponential_moving_average_crossover,
        'params': ['short_window', 'long_window']
    },
    'triple_sma': {
        'name': 'Triple Moving Average',
        'function': triple_moving_average_strategy,
        'params': ['short_window', 'medium_window', 'long_window']
    },
    'rsi': {
        'name': 'RSI Strategy',
        'function': rsi_strategy,
        'params': ['rsi_window', 'oversold_threshold', 'overbought_threshold']
    },
    'macd': {
        'name': 'MACD Strategy',
        'function': macd_strategy,
        'params': ['fast_period', 'slow_period', 'signal_period']
    },
    'momentum': {
        'name': 'Price Momentum',
        'function': momentum_strategy,
        'params': ['momentum_window', 'threshold']
    },
    'bollinger_bands': {
        'name': 'Bollinger Bands',
        'function': bollinger_bands_strategy,
        'params': ['window', 'num_std']
    },
    'mean_reversion': {
        'name': 'Mean Reversion Strategy',
        'function': mean_reversion_strategy,
        'params': ['lookback_window', 'threshold_std']
    },
    
    # Correlation Strategies
    'multi_asset_correlation': {
        'name': 'Multi-Asset Correlation Strategy',
        'function': multi_asset_correlation_strategy,
        'params': ['correlation_window', 'threshold', 'rebalance_frequency']
    },
    'volatility_correlation': {
        'name': 'Volatility Correlation Strategy',
        'function': volatility_correlation_strategy,
        'params': ['vol_window', 'correlation_window', 'vol_threshold']
    },
    'volume_price_correlation': {
        'name': 'Volume-Price Correlation Strategy',
        'function': volume_price_correlation_strategy,
        'params': ['correlation_window', 'volume_threshold']
    },
    'cross_timeframe_correlation': {
        'name': 'Cross-Timeframe Correlation Strategy',
        'function': cross_timeframe_correlation_strategy,
        'params': ['short_period', 'long_period', 'correlation_window', 'threshold']
    },
    'sentiment_correlation': {
        'name': 'Sentiment Correlation Strategy',
        'function': sentiment_correlation_strategy,
        'params': ['rsi_window', 'macd_fast', 'macd_slow', 'correlation_window', 'threshold']
    },
    'adaptive_correlation': {
        'name': 'Adaptive Correlation Strategy',
        'function': adaptive_correlation_strategy,
        'params': ['base_window', 'adaptive_factor', 'correlation_threshold']
    },
    
    # Statistical Strategies
    'cointegration_pairs': {
        'name': 'Cointegration Pairs Strategy',
        'function': cointegration_pairs_strategy,
        'params': ['lookback_window', 'zscore_threshold', 'half_life_threshold']
    },
    'regime_detection': {
        'name': 'Regime Detection Strategy',
        'function': regime_detection_strategy,
        'params': ['lookback_window', 'regime_threshold']
    },
    'fractal_dimension': {
        'name': 'Fractal Dimension Strategy',
        'function': fractal_dimension_strategy,
        'params': ['window', 'threshold']
    },
    'entropy_based': {
        'name': 'Entropy-Based Strategy',
        'function': entropy_based_strategy,
        'params': ['entropy_window', 'entropy_threshold']
    },
    'kalman_filter': {
        'name': 'Kalman Filter Strategy',
        'function': kalman_filter_strategy,
        'params': ['process_variance', 'observation_variance']
    },
    'granger_causality': {
        'name': 'Granger Causality Strategy',
        'function': granger_causality_strategy,
        'params': ['volume_lag', 'causality_window', 'significance_level']
    },
    
    # Machine Learning Strategies
    'linear_regression': {
        'name': 'Linear Regression Strategy',
        'function': linear_regression_strategy,
        'params': ['lookback_window', 'feature_lag', 'threshold']
    },
    'knn_strategy': {
        'name': 'K-Nearest Neighbors Strategy',
        'function': knn_strategy,
        'params': ['k_neighbors', 'lookback_window', 'feature_window']
    },
    'ensemble_ml': {
        'name': 'Ensemble ML Strategy',
        'function': ensemble_ml_strategy,
        'params': ['lookback_window', 'ensemble_size']
    },
    'neural_network_simple': {
        'name': 'Simple Neural Network Strategy',
        'function': neural_network_simple_strategy,
        'params': ['lookback_window', 'hidden_size', 'learning_rate']
    },
    
    # Advanced Technical Strategies
    'ichimoku_cloud': {
        'name': 'Ichimoku Cloud Strategy',
        'function': ichimoku_cloud_strategy,
        'params': ['tenkan_period', 'kijun_period', 'senkou_span_b_period']
    },
    'elliott_wave': {
        'name': 'Elliott Wave Strategy',
        'function': elliott_wave_strategy,
        'params': ['wave_window', 'fibonacci_levels']
    },
    'harmonic_pattern': {
        'name': 'Harmonic Pattern Strategy',
        'function': harmonic_pattern_strategy,
        'params': ['pattern_window', 'tolerance']
    },
    'market_profile': {
        'name': 'Market Profile Strategy',
        'function': market_profile_strategy,
        'params': ['profile_window', 'value_area_percent']
    },
    'wyckoff_method': {
        'name': 'Wyckoff Method Strategy',
        'function': wyckoff_method_strategy,
        'params': ['accumulation_window', 'volume_threshold']
    },
    
    # Market Microstructure Strategies
    'order_flow_imbalance': {
        'name': 'Order Flow Imbalance Strategy',
        'function': order_flow_imbalance_strategy,
        'params': ['window', 'imbalance_threshold']
    },
    'volume_price_analysis': {
        'name': 'Volume Price Analysis Strategy',
        'function': volume_price_analysis_strategy,
        'params': ['volume_window', 'price_window']
    },
    'liquidity_provision': {
        'name': 'Liquidity Provision Strategy',
        'function': liquidity_provision_strategy,
        'params': ['spread_window', 'liquidity_threshold']
    },
    'tick_analysis': {
        'name': 'Tick Analysis Strategy',
        'function': tick_analysis_strategy,
        'params': ['tick_window', 'uptick_threshold']
    },
    'market_making': {
        'name': 'Market Making Strategy',
        'function': market_making_strategy,
        'params': ['inventory_target', 'spread_multiple']
    },
    'high_frequency_momentum': {
        'name': 'High Frequency Momentum Strategy',
        'function': high_frequency_momentum_strategy,
        'params': ['momentum_window', 'volume_factor']
    },
    
    # Quantitative Strategies
    'factor_model': {
        'name': 'Multi-Factor Model Strategy',
        'function': factor_model_strategy,
        'params': ['lookback_window', 'factor_threshold']
    },
    'risk_parity': {
        'name': 'Risk Parity Strategy',
        'function': risk_parity_strategy,
        'params': ['volatility_window', 'rebalance_frequency']
    },
    'black_litterman': {
        'name': 'Black-Litterman Strategy',
        'function': black_litterman_strategy,
        'params': ['confidence_window', 'view_strength']
    },
    'copula_strategy': {
        'name': 'Copula-based Strategy',
        'function': copula_strategy,
        'params': ['reference_window', 'quantile_threshold']
    },
    'value_at_risk': {
        'name': 'Value at Risk Strategy',
        'function': value_at_risk_strategy,
        'params': ['var_window', 'confidence_level', 'var_threshold']
    },
    'maximum_diversification': {
        'name': 'Maximum Diversification Strategy',
        'function': maximum_diversification_strategy,
        'params': ['lookback_window', 'rebalance_frequency']
    }
}


@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')


@app.route('/backtest', methods=['GET', 'POST'])
def run_backtest():
    """Backtest execution page"""
    
    if request.method == 'GET':
        return render_template('backtest_form.html')
    
    try:
        # Get form data
        strategy_name = request.form.get('strategy')
        initial_capital = float(request.form.get('initial_capital', 1000000))
        start_date = request.form.get('start_date')
        end_date = request.form.get('end_date')
        commission_rate = float(request.form.get('commission_rate', 0.001))
        data_source = request.form.get('data_source', 'yahoo')
        
        # Parameter validation
        if not strategy_name or strategy_name not in STRATEGIES:
            flash('Please select a valid strategy.', 'error')
            return redirect(url_for('run_backtest'))
        
        if not start_date or not end_date:
            flash('Please enter start and end dates.', 'error')
            return redirect(url_for('run_backtest'))
        
        # Get strategy parameters
        strategy_info = STRATEGIES[strategy_name]
        strategy_params = {}
        
        for param in strategy_info['params']:
            value = request.form.get(param)
            if value:
                try:
                    # Convert to number
                    strategy_params[param] = float(value) if '.' in value else int(value)
                except ValueError:
                    strategy_params[param] = value
        
        print("Backtest execution started: {}".format(strategy_name))
        print("Parameters: {}".format(strategy_params))
        
        # Get data and run backtest
        fetcher = BitcoinDataFetcher()
        data = fetcher.get_bitcoin_data(
            source=data_source,
            start_date=start_date,
            end_date=end_date
        )
        
        engine = BacktestEngine(initial_capital, commission_rate)
        strategy_func = strategy_info['function']
        
        results = engine.run_backtest(data, strategy_func, **strategy_params)
        
        # Add strategy indicators to data (for visualization)
        data_with_signals = strategy_func(data.copy(), **strategy_params)
        
        # Create visualizations
        visualizer = WebVisualizer(results, data_with_signals)
        
        price_chart = visualizer.create_price_chart()
        equity_chart = visualizer.create_equity_curve_chart()
        drawdown_chart = visualizer.create_drawdown_chart()
        monthly_returns_chart = visualizer.create_monthly_returns_chart()
        trade_analysis_chart = visualizer.create_trade_analysis_chart()
        performance_table = visualizer.create_performance_summary_table()
        
        # Recent trades (last 10)
        recent_trades = results['trades'][-10:] if results['trades'] else []
        
        # Execution time
        execution_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        return render_template('backtest_results.html',
                             strategy_name=strategy_info['name'],
                             strategy_params=strategy_params,
                             start_date=start_date,
                             end_date=end_date,
                             results=results,
                             price_chart=price_chart,
                             equity_chart=equity_chart,
                             drawdown_chart=drawdown_chart,
                             monthly_returns_chart=monthly_returns_chart,
                             trade_analysis_chart=trade_analysis_chart,
                             performance_table=performance_table,
                             recent_trades=recent_trades,
                             execution_time=execution_time,
                             data_points=len(data))
        
    except Exception as e:
        error_msg = "Error occurred during backtest execution: {}".format(str(e))
        print("Error: {}".format(error_msg))
        print(traceback.format_exc())
        flash(error_msg, 'error')
        return redirect(url_for('run_backtest'))


@app.route('/comparison', methods=['GET', 'POST'])
def strategy_comparison():
    """Strategy comparison page"""
    
    if request.method == 'GET':
        return render_template('strategy_comparison_form.html', strategies=STRATEGIES)
    
    try:
        # Get form data
        selected_strategies = request.form.getlist('strategies')
        start_date = request.form.get('start_date')
        end_date = request.form.get('end_date')
        initial_capital = float(request.form.get('initial_capital', 1000000))
        commission_rate = float(request.form.get('commission_rate', 0.001))
        
        if len(selected_strategies) < 2:
            flash('Please select at least 2 strategies for comparison.', 'error')
            return redirect(url_for('strategy_comparison'))
        
        print("Strategy comparison execution started")
        print("Selected strategies: {}".format(selected_strategies))
        
        # Run backtest for each strategy
        results_summary = []
        
        # Get data (common)
        fetcher = BitcoinDataFetcher()
        data = fetcher.get_bitcoin_data(
            source="yahoo",
            start_date=start_date,
            end_date=end_date
        )
        
        for strategy_name in selected_strategies:
            if strategy_name in STRATEGIES:
                try:
                    print("Executing strategy: {}".format(strategy_name))
                    
                    # Use default parameters
                    if strategy_name == 'sma_crossover':
                        params = {'short_window': 20, 'long_window': 50}
                    elif strategy_name == 'rsi':
                        params = {'rsi_window': 14, 'oversold_threshold': 30, 'overbought_threshold': 70}
                    elif strategy_name == 'bollinger_bands':
                        params = {'window': 20, 'num_std': 2.0}
                    elif strategy_name == 'macd':
                        params = {'fast_period': 12, 'slow_period': 26, 'signal_period': 9}
                    elif strategy_name == 'ema_crossover':
                        params = {'short_window': 12, 'long_window': 26}
                    elif strategy_name == 'triple_sma':
                        params = {'short_window': 10, 'medium_window': 20, 'long_window': 50}
                    elif strategy_name == 'momentum':
                        params = {'momentum_window': 10, 'threshold': 0.02}
                    elif strategy_name == 'mean_reversion':
                        params = {'lookback_window': 20, 'threshold_std': 1.5}
                    # Correlation strategies
                    elif strategy_name == 'multi_asset_correlation':
                        params = {'correlation_window': 30, 'threshold': 0.3, 'rebalance_frequency': 5}
                    elif strategy_name == 'volatility_correlation':
                        params = {'vol_window': 20, 'correlation_window': 30, 'vol_threshold': 0.5}
                    elif strategy_name == 'volume_price_correlation':
                        params = {'correlation_window': 20, 'volume_threshold': 0.3}
                    elif strategy_name == 'cross_timeframe_correlation':
                        params = {'short_period': 5, 'long_period': 20, 'correlation_window': 30, 'threshold': 0.4}
                    elif strategy_name == 'sentiment_correlation':
                        params = {'rsi_window': 14, 'macd_fast': 12, 'macd_slow': 26, 'correlation_window': 20, 'threshold': 0.6}
                    elif strategy_name == 'adaptive_correlation':
                        params = {'base_window': 20, 'adaptive_factor': 0.5, 'correlation_threshold': 0.4}
                    # Statistical strategies
                    elif strategy_name == 'cointegration_pairs':
                        params = {'lookback_window': 60, 'zscore_threshold': 2.0, 'half_life_threshold': 20}
                    elif strategy_name == 'regime_detection':
                        params = {'lookback_window': 50, 'regime_threshold': 0.02}
                    elif strategy_name == 'fractal_dimension':
                        params = {'window': 30, 'threshold': 1.5}
                    elif strategy_name == 'entropy_based':
                        params = {'entropy_window': 20, 'entropy_threshold': 0.7}
                    elif strategy_name == 'kalman_filter':
                        params = {'process_variance': 1e-4, 'observation_variance': 1e-2}
                    elif strategy_name == 'granger_causality':
                        params = {'volume_lag': 5, 'causality_window': 30, 'significance_level': 0.05}
                    # ML strategies
                    elif strategy_name == 'linear_regression':
                        params = {'lookback_window': 30, 'feature_lag': 1, 'threshold': 0.01}
                    elif strategy_name == 'knn_strategy':
                        params = {'k_neighbors': 5, 'lookback_window': 50, 'feature_window': 5}
                    elif strategy_name == 'ensemble_ml':
                        params = {'lookback_window': 40, 'ensemble_size': 3}
                    elif strategy_name == 'neural_network_simple':
                        params = {'lookback_window': 30, 'hidden_size': 5, 'learning_rate': 0.01}
                    # Advanced technical strategies
                    elif strategy_name == 'ichimoku_cloud':
                        params = {'tenkan_period': 9, 'kijun_period': 26, 'senkou_span_b_period': 52}
                    elif strategy_name == 'elliott_wave':
                        params = {'wave_window': 50, 'fibonacci_levels': [0.236, 0.382, 0.618]}
                    elif strategy_name == 'harmonic_pattern':
                        params = {'pattern_window': 40, 'tolerance': 0.05}
                    elif strategy_name == 'market_profile':
                        params = {'profile_window': 20, 'value_area_percent': 0.7}
                    elif strategy_name == 'wyckoff_method':
                        params = {'accumulation_window': 30, 'volume_threshold': 1.5}
                    # Market microstructure strategies
                    elif strategy_name == 'order_flow_imbalance':
                        params = {'window': 20, 'imbalance_threshold': 0.3}
                    elif strategy_name == 'volume_price_analysis':
                        params = {'volume_window': 15, 'price_window': 10}
                    elif strategy_name == 'liquidity_provision':
                        params = {'spread_window': 20, 'liquidity_threshold': 0.01}
                    elif strategy_name == 'tick_analysis':
                        params = {'tick_window': 10, 'uptick_threshold': 0.6}
                    elif strategy_name == 'market_making':
                        params = {'inventory_target': 0.5, 'spread_multiple': 2.0}
                    elif strategy_name == 'high_frequency_momentum':
                        params = {'momentum_window': 3, 'volume_factor': 1.5}
                    # Quantitative strategies
                    elif strategy_name == 'factor_model':
                        params = {'lookback_window': 60, 'factor_threshold': 1.5}
                    elif strategy_name == 'risk_parity':
                        params = {'volatility_window': 30, 'rebalance_frequency': 5}
                    elif strategy_name == 'black_litterman':
                        params = {'confidence_window': 40, 'view_strength': 0.5}
                    elif strategy_name == 'copula_strategy':
                        params = {'reference_window': 50, 'quantile_threshold': 0.8}
                    elif strategy_name == 'value_at_risk':
                        params = {'var_window': 30, 'confidence_level': 0.05, 'var_threshold': 0.03}
                    elif strategy_name == 'maximum_diversification':
                        params = {'lookback_window': 40, 'rebalance_frequency': 10}
                    else:
                        params = {}
                    
                    # Run backtest
                    engine = BacktestEngine(initial_capital, commission_rate)
                    strategy_func = STRATEGIES[strategy_name]['function']
                    results = engine.run_backtest(data, strategy_func, **params)
                    
                    # Extract key metrics
                    summary = {
                        'Strategy': STRATEGIES[strategy_name]['name'],
                        'Parameters': str(params),
                        'Total Return (%)': results['total_return_pct'],
                        'Max DD (%)': results['max_drawdown_pct'],
                        'Sharpe Ratio': results['sharpe_ratio'],
                        'Num Trades': results['num_trades'],
                        'Final Value': results['final_value'],
                        'Excess Return vs Buy&Hold (%)': results['excess_return_pct']
                    }
                    results_summary.append(summary)
                    
                except Exception as e:
                    print("Error in strategy {}: {}".format(strategy_name, e))
        
        # Convert results to DataFrame and create HTML table
        if results_summary:
            comparison_df = pd.DataFrame(results_summary)
            comparison_df = comparison_df.sort_values('Total Return (%)', ascending=False)
            
            comparison_html = comparison_df.to_html(
                classes='table table-striped table-hover', 
                table_id='comparison-table',
                escape=False,
                index=False
            )
        else:
            comparison_html = "<p>No comparison results available</p>"
        
        return render_template('strategy_comparison_results.html',
                             comparison_table=comparison_html,
                             start_date=start_date,
                             end_date=end_date,
                             num_strategies=len(selected_strategies))
        
    except Exception as e:
        error_msg = "Error occurred during strategy comparison: {}".format(str(e))
        print("Error: {}".format(error_msg))
        print(traceback.format_exc())
        flash(error_msg, 'error')
        return redirect(url_for('strategy_comparison'))


@app.route('/api/strategies')
def api_strategies():
    """Strategies list API"""
    strategies_list = []
    for key, info in STRATEGIES.items():
        strategies_list.append({
            'id': key,
            'name': info['name'],
            'params': info['params']
        })
    return jsonify(strategies_list)


@app.route('/leaderboard')
def strategy_leaderboard():
    """Strategy Performance Leaderboard"""
    try:
        leaderboard = StrategyLeaderboard(strategies_dict=STRATEGIES)
        
        # デフォルトで最近6ヶ月のデータを使用
        from datetime import datetime, timedelta
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)
        
        results = leaderboard.generate_leaderboard(
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            max_workers=4,
            use_cache=True
        )
        
        if results:
            return render_template('leaderboard.html', 
                                 leaderboard_data=results,
                                 top_performers=results['leaderboard'][:10],
                                 all_results=results['leaderboard'])
        else:
            flash('リーダーボードの生成に失敗しました', 'error')
            return redirect(url_for('index'))
            
    except Exception as e:
        flash(f'リーダーボード生成中にエラーが発生しました: {str(e)}', 'error')
        return redirect(url_for('index'))


@app.route('/api/leaderboard')
def api_leaderboard():
    """Leaderboard API endpoint"""
    try:
        leaderboard = StrategyLeaderboard(strategies_dict=STRATEGIES)
        
        # パラメータ取得
        days = request.args.get('days', 180, type=int)
        use_cache = request.args.get('cache', 'true').lower() == 'true'
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        results = leaderboard.generate_leaderboard(
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            max_workers=6,
            use_cache=use_cache
        )
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health_check():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })


@app.errorhandler(404)
def not_found(error):
    """404 error handler"""
    return render_template('error.html', 
                         error_code=404, 
                         error_message="Page not found"), 404


@app.errorhandler(500)
def internal_error(error):
    """500 error handler"""
    return render_template('error.html', 
                         error_code=500, 
                         error_message="Internal server error occurred"), 500


if __name__ == '__main__':
    import os
    import logging
    
    # Production settings (Flask 2.3+)
    os.environ['FLASK_DEBUG'] = '0'
    app.config['DEBUG'] = False
    
    # Suppress development server warning
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    
    print("=== Bitcoin Backtest Web Application ===")
    print(f"Strategies loaded: {len(STRATEGIES)}")
    print("Server: http://localhost:5000")
    print("Mode: Production")
    print("Status: Ready")
    print("=" * 50)
    
    try:
        app.run(
            host='127.0.0.1',
            port=5000,
            debug=False,
            use_reloader=False,
            threaded=True
        )
    except KeyboardInterrupt:
        print("\nServer stopped")