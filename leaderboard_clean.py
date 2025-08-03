"""
Strategy Performance Leaderboard - Clean Version
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from data_fetcher import BitcoinDataFetcher
from backtest_engine import BacktestEngine


class StrategyLeaderboard:
    def __init__(self, strategies_dict=None, cache_file="leaderboard_cache.json"):
        self.cache_file = cache_file
        self.fetcher = BitcoinDataFetcher()
        self.strategies = strategies_dict or self._get_strategies_dict()
    
    def _get_strategies_dict(self):
        """Get strategies dictionary dynamically"""
        try:
            import importlib
            web_app_module = importlib.import_module('web_app_clean')
            return web_app_module.STRATEGIES
        except ImportError:
            return {}
        
    def run_single_strategy_test(self, strategy_info, data, initial_capital=1000000, commission_rate=0.001):
        """Run single strategy test"""
        strategy_name, strategy_data = strategy_info
        
        try:
            engine = BacktestEngine(initial_capital, commission_rate)
            strategy_func = strategy_data['function']
            
            # Use default parameters
            params = self._get_default_params(strategy_name)
            
            # Run backtest
            results = engine.run_backtest(data, strategy_func, **params)
            
            return {
                'strategy_name': strategy_name,
                'display_name': strategy_data['name'],
                'total_return_pct': results['total_return_pct'],
                'sharpe_ratio': results['sharpe_ratio'],
                'max_drawdown_pct': results['max_drawdown_pct'],
                'num_trades': results['num_trades'],
                'final_value': results['final_value'],
                'excess_return_pct': results['excess_return_pct'],
                'win_rate': results.get('win_rate', 0),
                'profit_factor': results.get('profit_factor', 0),
                'params': params,
                'success': True,
                'error': None
            }
            
        except Exception as e:
            return {
                'strategy_name': strategy_name,
                'display_name': strategy_data['name'],
                'success': False,
                'error': str(e),
                'total_return_pct': 0,
                'sharpe_ratio': 0,
                'max_drawdown_pct': 0,
                'num_trades': 0,
                'final_value': initial_capital,
                'excess_return_pct': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'params': {}
            }
    
    def _get_default_params(self, strategy_name):
        """Get default parameters"""
        defaults = {
            'sma_crossover': {'short_window': 20, 'long_window': 50},
            'ema_crossover': {'short_window': 12, 'long_window': 26},
            'triple_sma': {'short_window': 10, 'medium_window': 20, 'long_window': 50},
            'rsi': {'rsi_window': 14, 'oversold_threshold': 30, 'overbought_threshold': 70},
            'macd': {'fast_period': 12, 'slow_period': 26, 'signal_period': 9},
            'momentum': {'momentum_window': 10, 'threshold': 0.02},
            'bollinger_bands': {'window': 20, 'num_std': 2.0},
            'mean_reversion': {'lookback_window': 20, 'threshold_std': 1.5},
            'multi_asset_correlation': {'correlation_window': 30, 'threshold': 0.3, 'rebalance_frequency': 5},
            'volatility_correlation': {'vol_window': 20, 'correlation_window': 30, 'vol_threshold': 0.5},
            'volume_price_correlation': {'correlation_window': 20, 'volume_threshold': 0.3},
            'cross_timeframe_correlation': {'short_period': 5, 'long_period': 20, 'correlation_window': 30, 'threshold': 0.4},
            'sentiment_correlation': {'rsi_window': 14, 'macd_fast': 12, 'macd_slow': 26, 'correlation_window': 20, 'threshold': 0.6},
            'adaptive_correlation': {'base_window': 20, 'adaptive_factor': 0.5, 'correlation_threshold': 0.4},
            'cointegration_pairs': {'lookback_window': 60, 'zscore_threshold': 2.0, 'half_life_threshold': 20},
            'regime_detection': {'lookback_window': 50, 'regime_threshold': 0.02},
            'fractal_dimension': {'window': 30, 'threshold': 1.5},
            'entropy_based': {'entropy_window': 20, 'entropy_threshold': 0.7},
            'kalman_filter': {'process_variance': 1e-4, 'observation_variance': 1e-2},
            'granger_causality': {'volume_lag': 5, 'causality_window': 30, 'significance_level': 0.05},
            'linear_regression': {'lookback_window': 30, 'feature_lag': 1, 'threshold': 0.01},
            'knn_strategy': {'k_neighbors': 5, 'lookback_window': 50, 'feature_window': 5},
            'ensemble_ml': {'lookback_window': 40, 'ensemble_size': 3},
            'neural_network_simple': {'lookback_window': 30, 'hidden_size': 5, 'learning_rate': 0.01},
            'ichimoku_cloud': {'tenkan_period': 9, 'kijun_period': 26, 'senkou_span_b_period': 52},
            'elliott_wave': {'wave_window': 50, 'fibonacci_levels': [0.236, 0.382, 0.618]},
            'harmonic_pattern': {'pattern_window': 40, 'tolerance': 0.05},
            'market_profile': {'profile_window': 20, 'value_area_percent': 0.7},
            'wyckoff_method': {'accumulation_window': 30, 'volume_threshold': 1.5},
            'order_flow_imbalance': {'window': 20, 'imbalance_threshold': 0.3},
            'volume_price_analysis': {'volume_window': 15, 'price_window': 10},
            'liquidity_provision': {'spread_window': 20, 'liquidity_threshold': 0.01},
            'tick_analysis': {'tick_window': 10, 'uptick_threshold': 0.6},
            'market_making': {'inventory_target': 0.5, 'spread_multiple': 2.0},
            'high_frequency_momentum': {'momentum_window': 3, 'volume_factor': 1.5},
            'factor_model': {'lookback_window': 60, 'factor_threshold': 1.5},
            'risk_parity': {'volatility_window': 30, 'rebalance_frequency': 5},
            'black_litterman': {'confidence_window': 40, 'view_strength': 0.5},
            'copula_strategy': {'reference_window': 50, 'quantile_threshold': 0.8},
            'value_at_risk': {'var_window': 30, 'confidence_level': 0.05, 'var_threshold': 0.03},
            'maximum_diversification': {'lookback_window': 40, 'rebalance_frequency': 10}
        }
        return defaults.get(strategy_name, {})
    
    def generate_leaderboard(self, start_date="2023-01-01", end_date="2024-01-01", 
                           max_workers=4, use_cache=True):
        """Generate leaderboard"""
        
        cache_key = f"{start_date}_{end_date}"
        
        # Check cache
        if use_cache and os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    if cache_key in cache_data:
                        print("Using cached leaderboard data...")
                        return cache_data[cache_key]
            except:
                pass
        
        print(f"Generating leaderboard for {start_date} to {end_date}...")
        print("This may take a few minutes...")
        
        # Load data
        try:
            data = self.fetcher.get_bitcoin_data(
                source="yahoo",
                start_date=start_date,
                end_date=end_date
            )
            print(f"Data loaded: {len(data)} records")
        except Exception as e:
            print(f"Error loading data: {e}")
            return []
        
        # Run all strategies in parallel
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all strategies for parallel execution
            future_to_strategy = {
                executor.submit(
                    self.run_single_strategy_test, 
                    (name, info), 
                    data
                ): name for name, info in self.strategies.items()
            }
            
            # Collect results
            completed = 0
            total = len(self.strategies)
            
            for future in as_completed(future_to_strategy):
                try:
                    result = future.result(timeout=120)  # 2 minute timeout
                    results.append(result)
                    completed += 1
                    
                    if result['success']:
                        print(f"[{completed}/{total}] OK {result['strategy_name']}: {result['total_return_pct']:.1f}%")
                    else:
                        print(f"[{completed}/{total}] ERROR {result['strategy_name']}: {result['error']}")
                        
                except Exception as e:
                    strategy_name = future_to_strategy[future]
                    print(f"[{completed+1}/{total}] ERROR {strategy_name}: Timeout or error")
                    results.append({
                        'strategy_name': strategy_name,
                        'display_name': self.strategies[strategy_name]['name'],
                        'success': False,
                        'error': 'Timeout or execution error',
                        'total_return_pct': 0,
                        'sharpe_ratio': 0,
                        'max_drawdown_pct': 0,
                        'num_trades': 0,
                        'final_value': 1000000,
                        'excess_return_pct': 0,
                        'win_rate': 0,
                        'profit_factor': 0,
                        'params': {}
                    })
                    completed += 1
        
        # Sort results
        successful_results = [r for r in results if r['success']]
        failed_results = [r for r in results if not r['success']]
        
        # Sort by performance
        leaderboard = sorted(successful_results, key=lambda x: x['total_return_pct'], reverse=True)
        leaderboard.extend(failed_results)  # Add failed strategies at the end
        
        # Add rankings
        for i, result in enumerate(leaderboard, 1):
            result['rank'] = i if result['success'] else '-'
        
        # Save to cache
        try:
            cache_data = {}
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
            
            cache_data[cache_key] = {
                'leaderboard': leaderboard,
                'generated_at': datetime.now().isoformat(),
                'period': f"{start_date} to {end_date}",
                'total_strategies': len(leaderboard),
                'successful_strategies': len(successful_results)
            }
            
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            print(f"Warning: Could not save cache: {e}")
        
        return {
            'leaderboard': leaderboard,
            'generated_at': datetime.now().isoformat(),
            'period': f"{start_date} to {end_date}",
            'total_strategies': len(leaderboard),
            'successful_strategies': len(successful_results)
        }


def main():
    """Main execution function"""
    leaderboard = StrategyLeaderboard()
    
    print("=== Strategy Performance Leaderboard ===")
    print("Generating comprehensive strategy comparison...")
    
    # Generate leaderboard for recent 6 months
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)
    
    results = leaderboard.generate_leaderboard(
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
        max_workers=6,
        use_cache=True
    )
    
    if results:
        print("\n" + "="*80)
        print("STRATEGY PERFORMANCE LEADERBOARD")
        print("="*80)
        print(f"Period: {results['period']}")
        print(f"Successful strategies: {results['successful_strategies']}/{results['total_strategies']}")
        print("="*80)
        
        print(f"{'Rank':<4} {'Strategy':<35} {'Return':<8} {'Sharpe':<7} {'MaxDD':<7} {'Trades':<7}")
        print("-"*80)
        
        for result in results['leaderboard'][:20]:  # Top 20
            if result['success']:
                print(f"{result['rank']:<4} {result['display_name'][:34]:<35} "
                      f"{result['total_return_pct']:>6.1f}% {result['sharpe_ratio']:>6.2f} "
                      f"{result['max_drawdown_pct']:>6.1f}% {result['num_trades']:>6}")
        
        print("="*80)
        print("Leaderboard saved to cache for web display")


if __name__ == "__main__":
    main()