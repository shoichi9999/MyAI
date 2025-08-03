#!/usr/bin/env python3
"""
Bybit API接続テストスクリプト
"""

import requests
import json

def test_bybit_api():
    """Bybit APIの接続テスト"""
    print("=== Bybit API接続テスト ===")
    
    url = 'https://api.bybit.com/v5/market/instruments-info'
    params = {'category': 'linear'}
    
    try:
        print("APIにリクエスト送信中...")
        response = requests.get(url, params=params, timeout=10)
        print(f"ステータスコード: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"レスポンス構造: {list(data.keys())}")
            print(f"retCode: {data.get('retCode')}")
            print(f"retMsg: {data.get('retMsg')}")
            
            if 'result' in data and 'list' in data['result']:
                total = len(data['result']['list'])
                print(f"総シンボル数: {total}")
                
                # BTC関連のシンボルを検索
                btc_symbols = []
                for item in data['result']['list']:
                    symbol = item['symbol']
                    if symbol.endswith('BTC') and symbol != 'BTC':
                        btc_symbols.append({
                            'symbol': symbol,
                            'baseCoin': item.get('baseCoin', ''),
                            'status': item.get('status', ''),
                            'contractType': item.get('contractType', '')
                        })
                
                print(f"BTC建てシンボル数: {len(btc_symbols)}")
                if btc_symbols:
                    print("最初の5個:")
                    for i, s in enumerate(btc_symbols[:5], 1):
                        print(f"  {i}. {s['symbol']} ({s['baseCoin']}) - {s['status']}")
                
                return btc_symbols
            else:
                print("データ構造に問題があります")
                print(f"データサンプル: {str(data)[:500]}...")
                return []
        else:
            print(f"HTTPエラー: {response.text[:200]}")
            return []
            
    except Exception as e:
        print(f"エラー: {str(e)}")
        return []

if __name__ == "__main__":
    symbols = test_bybit_api()
    print(f"\n結果: {len(symbols)}個のBTC建てシンボルを発見")