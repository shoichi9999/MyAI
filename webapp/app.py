"""Web UI アプリケーション

バックテスト結果の閲覧、手動テスト実行、AI探索の制御を行うWeb UI。
"""

import json
import logging
import threading
from flask import Flask, render_template, request, jsonify

from config.settings import (
    WEB_HOST, WEB_PORT, DEFAULT_SYMBOLS, BACKTEST_DEFAULTS, EXPLORER_DEFAULTS,
)
from strategies.builtin import ALL_STRATEGIES
from explorer.optimizer import AlgorithmExplorer, quick_backtest
from backtest.engine import BacktestEngine
from data.fetcher import get_data, list_csv_symbols

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder="templates", static_folder="static")

# グローバル探索エンジンインスタンス
explorer = AlgorithmExplorer()
explorer_thread = None
exploration_log = []  # リアルタイムログ


def exploration_callback(result):
    """探索結果が出るたびに呼ばれるコールバック"""
    entry = {
        "strategy": result["strategy"],
        "symbol": result["symbol"],
        "best_value": round(result["best_value"], 4),
        "total_return_pct": result["metrics"]["total_return_pct"],
        "win_rate": result["metrics"]["win_rate"],
        "total_trades": result["metrics"]["total_trades"],
        "params": result["best_params"],
    }
    exploration_log.append(entry)
    logger.info(f"Result: {result['strategy']}/{result['symbol']} -> {result['best_value']:.4f}")


# ---- Routes ----

@app.route("/")
def index():
    # CSVデータがあればそのシンボルを優先表示
    csv_symbols = list_csv_symbols()
    symbols = csv_symbols if csv_symbols else DEFAULT_SYMBOLS
    return render_template("index.html",
                           strategies=list(ALL_STRATEGIES.keys()),
                           symbols=symbols,
                           has_csv=bool(csv_symbols))


@app.route("/api/backtest", methods=["POST"])
def api_backtest():
    """単発バックテスト実行"""
    data = request.json
    strategy_name = data.get("strategy")
    symbol = data.get("symbol", "BTCUSDT")
    days = data.get("days", 7)
    params = data.get("params")

    if strategy_name not in ALL_STRATEGIES:
        return jsonify({"error": f"Unknown strategy: {strategy_name}"}), 400

    try:
        result = quick_backtest(strategy_name, symbol, params, days)
        return jsonify({"success": True, "result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/explore/start", methods=["POST"])
def api_explore_start():
    """AI探索開始"""
    global explorer, explorer_thread, exploration_log

    if explorer.is_running:
        return jsonify({"error": "Exploration already running"}), 409

    data = request.json or {}
    symbols = data.get("symbols", DEFAULT_SYMBOLS)
    strategies = data.get("strategies")
    days = data.get("days", 7)
    n_trials = data.get("n_trials", EXPLORER_DEFAULTS["n_trials"])

    exploration_log = []
    explorer = AlgorithmExplorer(
        symbols=symbols, days=days, n_trials=n_trials,
    )

    def run():
        explorer.run_exploration(
            strategy_names=strategies,
            callback=exploration_callback,
        )

    explorer_thread = threading.Thread(target=run, daemon=True)
    explorer_thread.start()

    return jsonify({"success": True, "message": "Exploration started"})


@app.route("/api/explore/stop", methods=["POST"])
def api_explore_stop():
    """AI探索停止"""
    explorer.stop()
    return jsonify({"success": True, "message": "Stop requested"})


@app.route("/api/explore/status")
def api_explore_status():
    """探索状態の取得"""
    return jsonify({
        "is_running": explorer.is_running,
        "results_count": len(explorer.results),
        "log": exploration_log[-50:],  # 直近50件
    })


@app.route("/api/explore/results")
def api_explore_results():
    """探索結果のランキングを返す"""
    top_n = request.args.get("top_n", 20, type=int)
    results = explorer.get_ranking(top_n)
    if not results:
        # ファイルから読み込み
        results = explorer.load_latest_results()[:top_n]
    return jsonify({"results": results})


@app.route("/api/symbols")
def api_symbols():
    """利用可能なシンボルリスト (CSV優先)"""
    csv_symbols = list_csv_symbols()
    symbols = csv_symbols if csv_symbols else DEFAULT_SYMBOLS
    return jsonify({"symbols": symbols, "source": "csv" if csv_symbols else "default"})


@app.route("/api/strategies")
def api_strategies():
    """利用可能な戦略一覧"""
    strategies = []
    for name, cls in ALL_STRATEGIES.items():
        strategies.append({
            "name": name,
            "default_params": cls.default_params(),
        })
    return jsonify({"strategies": strategies})


if __name__ == "__main__":
    app.run(host=WEB_HOST, port=WEB_PORT, debug=True)
