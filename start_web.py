"""
Webアプリケーション起動スクリプト（Windows対応）
"""

import sys
import os

# 文字エンコーディング設定
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    # コンソールのコードページをUTF-8に設定
    os.system('chcp 65001 > nul')

print("=== ビットコインバックテスト Webアプリケーション ===")
print("起動準備中...")

try:
    import os
    import logging
    from web_app_clean import app, STRATEGIES
    
    # Production settings (Flask 2.3+)
    os.environ['FLASK_DEBUG'] = '0'
    app.config['DEBUG'] = False
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    
    print("OK: Full-featured application loaded")
    print("Strategies: {} advanced trading strategies".format(len(STRATEGIES)))
    print("Server: http://localhost:5000")
    print("Mode: Production (no warnings)")
    print("-" * 60)
    
    app.run(
        debug=False, 
        host='127.0.0.1', 
        port=5000,
        use_reloader=False,
        threaded=True
    )
    
except Exception as e:
    print("ERROR: Failed to start application: {}".format(str(e)))
    print("Please install required libraries:")
    print("   pip install flask plotly pandas numpy yfinance")