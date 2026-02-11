"""システム全体の設定"""

# Binance API
BINANCE_BASE_URL = "https://api.binance.com"
BINANCE_KLINES_ENDPOINT = "/api/v3/klines"

# デフォルト対象ペア
DEFAULT_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "DOTUSDT", "MATICUSDT",
    "LINKUSDT", "LTCUSDT", "ATOMUSDT", "UNIUSDT", "NEARUSDT",
]

# バックテスト設定
BACKTEST_DEFAULTS = {
    "initial_capital": 10000,       # 初期資金 (USDT)
    "commission_rate": 0.001,       # 手数料率 0.1% (maker/taker)
    "slippage_rate": 0.0005,        # スリッページ 0.05%
    "leverage": 1,                  # レバレッジ倍率
    "interval": "1m",               # 1分足
    "max_position_pct": 1.0,        # 資金の最大使用割合
}

# データ保存設定
DATA_DIR = "data/cache"
RESULTS_DIR = "results"

# AI探索設定
EXPLORER_DEFAULTS = {
    "n_trials": 100,                # Optunaのトライアル数
    "n_jobs": 1,                    # 並列数
    "timeout": 600,                 # タイムアウト(秒)
    "metric": "sharpe_ratio",       # 最適化対象メトリック
    "min_trades": 10,               # 最低取引回数
}

# Web UI設定
WEB_HOST = "0.0.0.0"
WEB_PORT = 5000
