#!/usr/bin/env python3
"""
Bybit データ取得デバッグスクリプト
"""

import requests
from datetime import datetime, timedelta

def test_kline_api():
    """Bybit kline APIの直接テスト"""
    print("=== Bybit Kline API テスト ===")
    
    # テスト用のシンボル
    symbol = "ETHUSDT"
    
    # APIエンドポイント
    url = "https://api.bybit.com/v5/market/kline"
    
    # パラメータ
    params = {
        'category': 'linear',
        'symbol': symbol,
        'interval': '1h',
        'limit': 10
    }
    
    try:
        print(f"シンボル: {symbol}")
        print(f"URL: {url}")
        print(f"パラメータ: {params}")
        
        response = requests.get(url, params=params, timeout=10)
        print(f"ステータスコード: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"レスポンス構造: {list(data.keys())}")
            print(f"retCode: {data.get('retCode')}")
            print(f"retMsg: {data.get('retMsg')}")
            
            if 'result' in data and 'list' in data['result']:
                klines = data['result']['list']
                print(f"取得したkline数: {len(klines)}")
                
                if klines:
                    print("最初のklineデータ:")
                    print(f"  {klines[0]}")
                    
                    # タイムスタンプの変換テスト
                    import pandas as pd
                    df = pd.DataFrame(klines, columns=[
                        'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
                    ])
                    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
                    print(f"変換後の最初の行:")
                    print(f"  時刻: {df['timestamp'].iloc[0]}")
                    print(f"  OHLC: {df['open'].iloc[0]}, {df['high'].iloc[0]}, {df['low'].iloc[0]}, {df['close'].iloc[0]}")
                    
                    return True
                else:
                    print("klineデータが空です")
                    return False
            else:
                print("データ構造に問題があります")
                print(f"レスポンス内容: {data}")
                return False
        else:
            print(f"HTTPエラー: {response.text}")
            return False
            
    except Exception as e:
        print(f"エラー: {str(e)}")
        return False

def test_multiple_symbols():
    """複数シンボルでテスト"""
    print("\n=== 複数シンボルテスト ===")
    
    symbols = ['ETHUSDT', 'BTCUSDT', 'SOLUSDT', 'ADAUSDT', 'DOGEUSDT']
    
    for symbol in symbols:
        print(f"\n{symbol}をテスト中...")
        
        url = "https://api.bybit.com/v5/market/kline"
        params = {
            'category': 'linear',
            'symbol': symbol,
            'interval': '1h',
            'limit': 5
        }
        
        try:
            response = requests.get(url, params=params, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get('retCode') == 0 and data.get('result', {}).get('list'):
                    klines = data['result']['list']
                    print(f"  成功: {len(klines)}個のkline")
                else:
                    print(f"  データなし: {data.get('retMsg', 'Unknown error')}")
            else:
                print(f"  HTTPエラー: {response.status_code}")
        except Exception as e:
            print(f"  エラー: {e}")

if __name__ == "__main__":
    success = test_kline_api()
    if success:
        test_multiple_symbols()
    else:
        print("基本的なAPIテストに失敗しました")