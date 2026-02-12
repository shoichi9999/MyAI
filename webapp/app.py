"""Web UI アプリケーション

戦略自動生成の制御と結果閲覧を行うWeb UI。
"""

import json
import logging
import os
import threading
from flask import Flask, render_template, request, jsonify

from config.settings import (
    WEB_HOST, WEB_PORT, DEFAULT_SYMBOLS, RESULTS_DIR,
)
from strategies.composer import CONDITION_TYPES
from explorer.generator import StrategyGenerator
from data.fetcher import list_csv_symbols

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder="templates", static_folder="static")

# グローバル戦略生成インスタンス
generator_instance = None
generator_thread = None
generator_log = []
generator_running = False


def generation_callback(result):
    """戦略生成結果のコールバック"""
    entry = {
        "strategy": result["strategy"],
        "symbol": result["symbol"],
        "best_value": round(result["best_value"], 4),
        "return_pct": round(result["metrics"].get("return_pct", result["metrics"].get("total_return_pct", 0)), 2),
        "sharpe": round(result["metrics"].get("sharpe_ratio", 0), 4),
        "win_rate": round(result["metrics"].get("win_rate", 0), 1),
        "total_trades": result["metrics"].get("total_trades", 0),
        "max_dd": round(result["metrics"].get("max_drawdown_pct", 0), 2),
        "description": result.get("description", ""),
    }
    generator_log.append(entry)
    logger.info(f"Generated: {result['strategy']} -> {result['best_value']:.4f}")


# ---- Routes ----

@app.route("/")
def index():
    csv_symbols = list_csv_symbols()
    symbols = csv_symbols if csv_symbols else DEFAULT_SYMBOLS
    return render_template("index.html", symbols=symbols)


@app.route("/api/symbols")
def api_symbols():
    """利用可能なシンボルリスト (CSV優先)"""
    csv_symbols = list_csv_symbols()
    symbols = csv_symbols if csv_symbols else DEFAULT_SYMBOLS
    return jsonify({"symbols": symbols, "source": "csv" if csv_symbols else "default"})


# ---- Strategy Generation (Auto-Compose) ----

@app.route("/api/generate/start", methods=["POST"])
def api_generate_start():
    """戦略自動生成を開始"""
    global generator_instance, generator_thread, generator_log, generator_running

    if generator_running:
        return jsonify({"error": "Generation already running"}), 409

    data = request.json or {}
    symbol = data.get("symbol", "BTCUSDT")
    days = data.get("days", 7)
    n_trials = data.get("n_trials", 500)

    generator_log = []
    generator_running = True
    symbols = [symbol] if symbol != "ALL" else None
    generator_instance = StrategyGenerator(symbols=symbols, days=days)

    def run():
        global generator_running
        try:
            generator_instance.run(n_trials=n_trials, callback=generation_callback)
        finally:
            generator_running = False

    generator_thread = threading.Thread(target=run, daemon=True)
    generator_thread.start()

    return jsonify({"success": True, "message": "Generation started",
                    "condition_types": len(CONDITION_TYPES)})


@app.route("/api/generate/stop", methods=["POST"])
def api_generate_stop():
    """戦略自動生成を停止"""
    global generator_running
    if generator_instance:
        generator_instance.stop()
    generator_running = False
    return jsonify({"success": True, "message": "Stop requested"})


@app.route("/api/generate/status")
def api_generate_status():
    """生成状態の取得"""
    sorted_log = sorted(generator_log, key=lambda x: x["best_value"], reverse=True)
    return jsonify({
        "is_running": generator_running,
        "results_count": len(generator_log),
        "log": generator_log[-50:],
        "top": sorted_log[:20],
    })


# ---- Results File Browser ----

@app.route("/api/results/files")
def api_results_files():
    """保存済み結果ファイル一覧"""
    if not os.path.exists(RESULTS_DIR):
        return jsonify({"files": []})

    files = []
    for fname in sorted(os.listdir(RESULTS_DIR), reverse=True):
        if fname.endswith(".json"):
            fpath = os.path.join(RESULTS_DIR, fname)
            size = os.path.getsize(fpath)
            files.append({
                "name": fname,
                "size_kb": round(size / 1024, 1),
            })
    return jsonify({"files": files})


@app.route("/api/results/load")
def api_results_load():
    """結果ファイルの内容を読み込む"""
    fname = request.args.get("file", "")
    if not fname or ".." in fname or "/" in fname:
        return jsonify({"error": "Invalid filename"}), 400

    fpath = os.path.join(RESULTS_DIR, fname)
    if not os.path.exists(fpath):
        return jsonify({"error": "File not found"}), 404

    with open(fpath, "r") as f:
        data = json.load(f)
    return jsonify({"file": fname, "results": data})


if __name__ == "__main__":
    app.run(host=WEB_HOST, port=WEB_PORT, debug=True)
