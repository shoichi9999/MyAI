"""
Bybit Altcoin/USDT Futures Shorting Backtest Web Application
Bybit アルトコイン/USDT先物ショートバックテスト ウェブアプリケーション
"""

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from datetime import datetime, timedelta
import json
import traceback
import warnings
import pandas as pd
import numpy as np
import os
warnings.filterwarnings('ignore')

from bybit_data_fetcher import BybitDataFetcher, calculate_short_performance

app = Flask(__name__)
app.secret_key = 'bybit_futures_backtest_secret_key_2024'

# グローバル変数
bybit_fetcher = BybitDataFetcher()


@app.route('/')
def index():
    """ホームページ"""
    return render_template('bybit_index.html')


@app.route('/symbols')
def list_symbols():
    """利用可能なシンボル一覧"""
    try:
        print("シンボル一覧を取得中...")
        symbols_info = bybit_fetcher.get_symbols()
        
        if symbols_info:
            # ステータスでフィルタリング（アクティブなもののみ）
            active_symbols = [s for s in symbols_info if s['status'] == 'Trading']
            print(f"アクティブなシンボル: {len(active_symbols)}個")
            
            return render_template('bybit_symbols.html', 
                                 symbols=active_symbols,
                                 total_symbols=len(active_symbols))
        else:
            flash('シンボル情報の取得に失敗しました', 'error')
            return redirect(url_for('index'))
            
    except Exception as e:
        flash(f'エラーが発生しました: {str(e)}', 'error')
        return redirect(url_for('index'))


@app.route('/backtest', methods=['GET', 'POST'])
def run_backtest():
    """バックテスト実行"""
    
    if request.method == 'GET':
        # シンボル一覧を取得
        try:
            symbols_info = bybit_fetcher.get_symbols()
            active_symbols = [s for s in symbols_info if s['status'] == 'Trading']
            return render_template('bybit_backtest_form.html', symbols=active_symbols)
        except Exception as e:
            flash(f'シンボル情報の取得に失敗しました: {str(e)}', 'error')
            return render_template('bybit_backtest_form.html', symbols=[])
    
    try:
        # フォームデータを取得
        selected_symbols = request.form.getlist('symbols')
        days = int(request.form.get('days', 30))
        leverage = float(request.form.get('leverage', 1.0))
        max_symbols = int(request.form.get('max_symbols', 20))
        
        if not selected_symbols:
            flash('少なくとも1つのシンボルを選択してください', 'error')
            return redirect(url_for('run_backtest'))
        
        print(f"バックテスト開始: {len(selected_symbols)}シンボル, {days}日間, レバレッジ{leverage}x")
        
        # データ取得
        historical_data = bybit_fetcher.get_multiple_symbols_data(
            symbols=selected_symbols[:max_symbols],
            days=days,
            interval="1h"
        )
        
        if not historical_data:
            flash('データの取得に失敗しました', 'error')
            return redirect(url_for('run_backtest'))
        
        # 各シンボルのショートパフォーマンスを計算
        results = []
        
        for symbol, df in historical_data.items():
            if not df.empty:
                try:
                    performance = calculate_short_performance(df, leverage=leverage)
                    
                    if performance:
                        results.append({
                            'symbol': symbol,
                            'data_points': len(df),
                            'start_date': df.index[0].strftime('%Y-%m-%d'),
                            'end_date': df.index[-1].strftime('%Y-%m-%d'),
                            'total_return_pct': performance.get('total_return_pct', 0),
                            'volatility': performance.get('volatility', 0),
                            'sharpe_ratio': performance.get('sharpe_ratio', 0),
                            'max_drawdown_pct': performance.get('max_drawdown_pct', 0),
                            'win_rate': performance.get('win_rate', 0),
                            'num_periods': performance.get('num_periods', 0),
                            'cumulative_returns': performance.get('cumulative_returns', pd.Series()).tolist(),
                            'daily_returns': performance.get('daily_returns', pd.Series()).tolist()
                        })
                except Exception as e:
                    print(f"エラー {symbol}: {e}")
        
        # 結果をソート（リターン順）
        results.sort(key=lambda x: x['total_return_pct'], reverse=True)
        
        execution_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        return render_template('bybit_backtest_results.html',
                             results=results,
                             leverage=leverage,
                             days=days,
                             execution_time=execution_time,
                             total_symbols=len(results))
        
    except Exception as e:
        error_msg = f"バックテスト実行中にエラーが発生しました: {str(e)}"
        print(f"エラー: {error_msg}")
        print(traceback.format_exc())
        flash(error_msg, 'error')
        return redirect(url_for('run_backtest'))


@app.route('/leaderboard')
def futures_leaderboard():
    """先物ショートリーダーボード"""
    try:
        # キャッシュファイルチェック
        cache_file = "bybit_leaderboard_cache.json"
        use_cache = True
        
        if use_cache and os.path.exists(cache_file):
            try:
                cache_time = os.path.getmtime(cache_file)
                # 1時間以内のキャッシュを使用
                if (datetime.now().timestamp() - cache_time) < 3600:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        cached_data = json.load(f)
                        print("キャッシュされたリーダーボードデータを使用")
                        return render_template('bybit_leaderboard.html', **cached_data)
            except Exception as e:
                print(f"キャッシュ読み込みエラー: {e}")
        
        print("新しいリーダーボードを生成中...")
        
        # シンボル取得
        symbols_info = bybit_fetcher.get_symbols()
        active_symbols = [s['symbol'] for s in symbols_info if s['status'] == 'Trading']
        
        # 人気のあるシンボルを選択（メジャーなアルトコイン）
        popular_symbols = ['ETHUSDT', 'ADAUSDT', 'SOLUSDT', 'DOGEUSDT', 'DOTUSDT', 
                          'LINKUSDT', 'MATICUSDT', 'AVAXUSDT', 'UNIUSDT', 'LTCUSDT']
        
        # アクティブシンボルから人気シンボルをフィルタ
        test_symbols = [s for s in popular_symbols if s in active_symbols][:10]
        
        # 人気シンボルが足りない場合は、アクティブシンボルから補完
        if len(test_symbols) < 10:
            additional = [s for s in active_symbols[:20] if s not in test_symbols]
            test_symbols.extend(additional[:10-len(test_symbols)])
        days = 1
        leverage = 1.0
        
        print(f"テスト対象: {len(test_symbols)}シンボル")
        
        historical_data = bybit_fetcher.get_multiple_symbols_data(
            symbols=test_symbols,
            days=days,
            interval="15",
            max_symbols=10
        )
        
        # パフォーマンス計算
        leaderboard_results = []
        
        for symbol, df in historical_data.items():
            if not df.empty and len(df) > 10:  # 最低10データポイント
                try:
                    performance = calculate_short_performance(df, leverage=leverage)
                    
                    if performance:
                        leaderboard_results.append({
                            'rank': 0,  # 後で設定
                            'symbol': symbol,
                            'base_coin': symbol.replace('BTC', ''),
                            'total_return_pct': performance.get('total_return_pct', 0),
                            'sharpe_ratio': performance.get('sharpe_ratio', 0),
                            'max_drawdown_pct': performance.get('max_drawdown_pct', 0),
                            'win_rate': performance.get('win_rate', 0),
                            'volatility': performance.get('volatility', 0),
                            'num_periods': performance.get('num_periods', 0),
                            'data_quality': 'excellent' if len(df) > 150 else 'good' if len(df) > 50 else 'fair'
                        })
                except Exception as e:
                    print(f"計算エラー {symbol}: {e}")
                    import traceback
                    traceback.print_exc()
        
        # ランキング設定
        leaderboard_results.sort(key=lambda x: x['total_return_pct'], reverse=True)
        for i, result in enumerate(leaderboard_results, 1):
            result['rank'] = i
        
        # トップパフォーマー
        top_performers = leaderboard_results[:3]
        
        # 統計情報
        if leaderboard_results:
            avg_return = np.mean([r['total_return_pct'] for r in leaderboard_results])
            best_return = leaderboard_results[0]['total_return_pct'] if leaderboard_results else 0
            worst_return = leaderboard_results[-1]['total_return_pct'] if leaderboard_results else 0
        else:
            avg_return = best_return = worst_return = 0
        
        # レンダリングデータ
        render_data = {
            'leaderboard_results': leaderboard_results,
            'top_performers': top_performers,
            'total_symbols': len(leaderboard_results),
            'test_period': f"過去{days}日間",
            'leverage': leverage,
            'avg_return': avg_return,
            'best_return': best_return,
            'worst_return': worst_return,
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # キャッシュに保存
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(render_data, f, ensure_ascii=False, indent=2)
            print("リーダーボードデータをキャッシュに保存")
        except Exception as e:
            print(f"キャッシュ保存エラー: {e}")
        
        return render_template('bybit_leaderboard.html', **render_data)
        
    except Exception as e:
        error_msg = f"リーダーボード生成中にエラーが発生しました: {str(e)}"
        print(f"エラー: {error_msg}")
        print(traceback.format_exc())
        
        # フォールバックとして空のリーダーボードを表示
        fallback_data = {
            'leaderboard_results': [],
            'top_performers': [],
            'total_symbols': 0,
            'test_period': "データ取得失敗",
            'leverage': 1.0,
            'avg_return': 0,
            'best_return': 0,
            'worst_return': 0,
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        flash(f'リーダーボードの生成に失敗しました: {str(e)}', 'error')
        return render_template('bybit_leaderboard.html', **fallback_data)


@app.route('/api/symbols')
def api_symbols():
    """シンボル一覧API"""
    try:
        symbols_info = bybit_fetcher.get_symbols()
        active_symbols = [s for s in symbols_info if s['status'] == 'Trading']
        return jsonify({
            'success': True,
            'symbols': active_symbols,
            'count': len(active_symbols)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/performance/<symbol>')
def api_symbol_performance(symbol):
    """個別シンボルパフォーマンスAPI"""
    try:
        days = request.args.get('days', 7, type=int)
        leverage = request.args.get('leverage', 1.0, type=float)
        
        # データ取得
        df = bybit_fetcher.get_historical_data(symbol, days=days, interval="1h")
        
        if df.empty:
            return jsonify({'success': False, 'error': 'データが見つかりません'}), 404
        
        # パフォーマンス計算
        performance = calculate_short_performance(df, leverage=leverage)
        
        if not performance:
            return jsonify({'success': False, 'error': '計算に失敗しました'}), 500
        
        # 価格データ
        price_data = df['close'].iloc[-min(100, len(df)):].tolist()
        timestamps = df.index[-min(100, len(df)):].strftime('%Y-%m-%d %H:%M').tolist()
        
        return jsonify({
            'success': True,
            'symbol': symbol,
            'performance': {
                'total_return_pct': performance.get('total_return_pct', 0),
                'sharpe_ratio': performance.get('sharpe_ratio', 0),
                'max_drawdown_pct': performance.get('max_drawdown_pct', 0),
                'win_rate': performance.get('win_rate', 0),
                'volatility': performance.get('volatility', 0),
                'num_periods': performance.get('num_periods', 0)
            },
            'price_data': {
                'prices': price_data,
                'timestamps': timestamps
            },
            'data_points': len(df),
            'period_days': days,
            'leverage': leverage
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/health')
def health_check():
    """ヘルスチェック"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'Bybit Futures Shorting Backtest',
        'version': '1.0.0'
    })


@app.errorhandler(404)
def not_found(error):
    """404エラーハンドラー"""
    return render_template('bybit_error.html', 
                         error_code=404, 
                         error_message="ページが見つかりません"), 404


@app.errorhandler(500)
def internal_error(error):
    """500エラーハンドラー"""
    return render_template('bybit_error.html', 
                         error_code=500, 
                         error_message="内部サーバーエラーが発生しました"), 500


if __name__ == '__main__':
    import logging
    
    # 本番設定
    os.environ['FLASK_DEBUG'] = '0'
    app.config['DEBUG'] = False
    
    # 開発サーバー警告を抑制
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    
    print("=== Bybit Altcoin/USDT Futures Shorting Backtest ===")
    print("サーバー: http://localhost:5002")
    print("機能: レバレッジ1倍ショート戦略バックテスト")
    print("対象: Bybit アルトコイン/USDT先物ペア")
    print("ステータス: 準備完了")
    print("=" * 55)
    
    try:
        app.run(
            host='127.0.0.1',
            port=5002,
            debug=False,
            use_reloader=False,
            threaded=True
        )
    except KeyboardInterrupt:
        print("\nサーバーを停止しました")