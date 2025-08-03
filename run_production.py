#!/usr/bin/env python3
"""
Production-ready Bitcoin Backtest Web Application
本番環境対応ビットコインバックテストWebアプリケーション
"""

import os
import sys
from web_app_clean import app, STRATEGIES

def main():
    print("=" * 60)
    print("Bitcoin Backtest Web Application")
    print("=" * 60)
    
    # Strategy count
    print(f"Loaded strategies: {len(STRATEGIES)}")
    
    # Environment settings (Flask 2.3+)
    os.environ['FLASK_DEBUG'] = '0'
    app.config['DEBUG'] = False
    app.config['TESTING'] = False
    
    # Suppress Flask development server warning
    import logging
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    
    # Server info
    host = '127.0.0.1'
    port = 5000
    
    print(f"Server: http://{host}:{port}")
    print("Status: Production mode")
    print("=" * 60)
    print("READY: Open your browser and access the URL above")
    print("STOP:  Press Ctrl+C to stop the server")
    print("=" * 60)
    
    try:
        # Start server with production settings
        app.run(
            host=host,
            port=port,
            debug=False,
            use_reloader=False,  # Disable auto-reload
            threaded=True       # Enable threading
        )
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()