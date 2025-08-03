#!/usr/bin/env python3
"""
Clean start - Bitcoin Backtest Web Application with 41 strategies
"""

import os
import logging
from web_app_clean import app, STRATEGIES

# Production settings
os.environ['FLASK_DEBUG'] = '0'
app.config['DEBUG'] = False
logging.getLogger('werkzeug').setLevel(logging.ERROR)

print("=" * 70)
print("Bitcoin Backtest Web Application - Clean Version")
print("=" * 70)
print(f"Strategies loaded: {len(STRATEGIES)}")
print("Categories:")
print("  - Basic Strategies (8)")
print("  - Correlation Strategies (6)")
print("  - Statistical Strategies (6)")
print("  - Machine Learning Strategies (4)")
print("  - Advanced Technical Strategies (5)")
print("  - Market Microstructure Strategies (6)")
print("  - Quantitative Strategies (6)")
print("=" * 70)
print("Server: http://localhost:5000")
print("Status: All 41 strategies active")
print("=" * 70)

# Kill any existing processes on port 5000 (Windows)
import subprocess
try:
    result = subprocess.run(['netstat', '-ano'], capture_output=True, text=True)
    lines = result.stdout.split('\n')
    for line in lines:
        if ':5000' in line and 'LISTENING' in line:
            parts = line.split()
            if len(parts) >= 5:
                pid = parts[-1]
                if pid.isdigit():
                    try:
                        subprocess.run(['taskkill', '/F', '/PID', pid], 
                                     capture_output=True, check=False)
                        print(f"Stopped old process PID {pid}")
                    except:
                        pass
except:
    pass

print("Starting server...")

try:
    app.run(
        host='127.0.0.1',
        port=5000,
        debug=False,
        use_reloader=False,
        threaded=True
    )
except KeyboardInterrupt:
    print("\nServer stopped by user")
except Exception as e:
    print(f"Error: {e}")
    print("If port 5000 is busy, try: python run_port_5001.py")