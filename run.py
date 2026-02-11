"""アプリケーションのエントリーポイント

Usage:
    python run.py              # Web UIを起動
    python run.py --explore    # AI探索のみ実行(CLI)
    python run.py --backtest   # 単発バックテスト(CLI)
"""

import argparse
import json
import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def main():
    parser = argparse.ArgumentParser(description="Binance Algorithm Explorer")
    parser.add_argument("--explore", action="store_true", help="Run AI exploration (CLI mode)")
    parser.add_argument("--backtest", action="store_true", help="Run single backtest (CLI mode)")
    parser.add_argument("--strategy", type=str, help="Strategy name for --backtest")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Symbol")
    parser.add_argument("--days", type=int, default=7, help="Data period in days")
    parser.add_argument("--trials", type=int, default=50, help="Optuna trials per strategy")
    parser.add_argument("--port", type=int, default=5000, help="Web UI port")
    args = parser.parse_args()

    if args.explore:
        run_exploration(args)
    elif args.backtest:
        run_backtest(args)
    else:
        run_web(args)


def run_web(args):
    """Web UIを起動"""
    from webapp.app import app
    from config.settings import WEB_HOST
    print(f"\n  Binance Algorithm Explorer")
    print(f"  http://localhost:{args.port}\n")
    app.run(host=WEB_HOST, port=args.port, debug=True)


def run_exploration(args):
    """CLI探索モード"""
    from explorer.optimizer import AlgorithmExplorer

    print(f"\n=== AI Algorithm Exploration ===")
    print(f"Symbol: {args.symbol}")
    print(f"Days: {args.days}")
    print(f"Trials: {args.trials}\n")

    explorer = AlgorithmExplorer(
        symbols=[args.symbol] if args.symbol != "ALL" else None,
        days=args.days,
        n_trials=args.trials,
    )

    def on_result(result):
        print(f"  {result['strategy']:30s} | {result['symbol']:12s} | "
              f"Sharpe: {result['best_value']:8.4f} | "
              f"Return: {result['metrics']['total_return_pct']:8.2f}% | "
              f"WR: {result['metrics']['win_rate']:5.1f}% | "
              f"Trades: {result['metrics']['total_trades']}")

    results = explorer.run_exploration(callback=on_result)

    print(f"\n=== Top 10 Results ===")
    for i, r in enumerate(results[:10]):
        print(f"  #{i+1} {r['strategy']:25s} {r['symbol']:12s} "
              f"Sharpe={r['best_value']:.4f} Return={r['metrics']['total_return_pct']:.2f}%")

    print(f"\nTotal results: {len(results)}")


def run_backtest(args):
    """CLI単発バックテスト"""
    from explorer.optimizer import quick_backtest
    from strategies.builtin import ALL_STRATEGIES

    if not args.strategy:
        print("Available strategies:")
        for name in ALL_STRATEGIES:
            print(f"  - {name}")
        print("\nUsage: python run.py --backtest --strategy SMA_Cross --symbol BTCUSDT --days 7")
        return

    print(f"\n=== Backtest: {args.strategy} on {args.symbol} ({args.days} days) ===\n")

    try:
        result = quick_backtest(args.strategy, args.symbol, days=args.days)
        for key, value in result.items():
            if key not in ("strategy", "symbol", "params"):
                print(f"  {key:25s}: {value}")
        print(f"\n  Params: {json.dumps(result.get('params', {}))}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
