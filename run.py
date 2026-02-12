"""アプリケーションのエントリーポイント

Usage:
    python run.py              # Web UIを起動
    python run.py --generate   # 戦略自体を自動生成＆探索(CLI)
"""

import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def main():
    parser = argparse.ArgumentParser(description="Binance Algorithm Explorer")
    parser.add_argument("--generate", action="store_true", help="Auto-generate new strategies (CLI mode)")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Symbol (or ALL)")
    parser.add_argument("--days", type=int, default=7, help="Data period in days")
    parser.add_argument("--trials", type=int, default=500, help="Optuna trials (structure × params)")
    parser.add_argument("--port", type=int, default=5000, help="Web UI port")
    args = parser.parse_args()

    if args.generate:
        run_generation(args)
    else:
        run_web(args)


def run_web(args):
    """Web UIを起動"""
    from webapp.app import app
    from config.settings import WEB_HOST
    print(f"\n  Binance Algorithm Explorer")
    print(f"  http://localhost:{args.port}\n")
    app.run(host=WEB_HOST, port=args.port, debug=True)


def run_generation(args):
    """CLI戦略自動生成モード — 構造ごと探索"""
    from explorer.generator import StrategyGenerator

    print(f"\n=== Strategy Generation (Auto-Compose) ===")
    print(f"Symbol: {args.symbol}")
    print(f"Days: {args.days}")
    print(f"Trials: {args.trials} (structure × params combined)")
    print(f"Exploring: {len(__import__('strategies.composer', fromlist=['CONDITION_TYPES']).CONDITION_TYPES)} condition types")
    print(f"")

    generator = StrategyGenerator(
        symbols=[args.symbol] if args.symbol != "ALL" else None,
        days=args.days,
    )

    results = generator.run(n_trials=args.trials)

    print(f"\n=== Top Generated Strategies ===")
    generator.describe_best(5)
    print(f"\nTotal viable strategies found: {len(results)}")
    if results:
        from config.settings import RESULTS_DIR
        print(f"Results saved to: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
