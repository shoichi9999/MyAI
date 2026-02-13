"""単一ロジック検証ツール

PythonバックテストとPineScriptの結果を1対1で照合するための
シンプルなバックテスト。ロングオンリーで動作し、トレードごとの
詳細をCSV出力する。

使い方:
    python -m backtest.validate --symbol BTCUSDT --days 7

    # EMA cross (デフォルト)
    python -m backtest.validate --symbol BTCUSDT --days 7 --strategy ema_cross --fast 9 --slow 21

    # RSI threshold
    python -m backtest.validate --symbol BTCUSDT --days 7 --strategy rsi_threshold --period 14 --threshold 30
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import BACKTEST_DEFAULTS
from data.fetcher import get_data
from strategies.base import (
    add_indicator_ema, add_indicator_sma, add_indicator_rsi,
)


# ─── シグナル生成 ──────────────────────────────────

def generate_ema_cross_signals(df: pd.DataFrame, fast: int, slow: int) -> pd.DataFrame:
    """EMAクロス: fast > slow → buy(1), fast < slow → sell(-1)"""
    df = df.copy()
    df["ema_fast"] = add_indicator_ema(df, fast)
    df["ema_slow"] = add_indicator_ema(df, slow)
    df["signal"] = 0
    df.loc[df["ema_fast"] > df["ema_slow"], "signal"] = 1
    df.loc[df["ema_fast"] < df["ema_slow"], "signal"] = -1
    return df


def generate_rsi_threshold_signals(df: pd.DataFrame, period: int,
                                    buy_threshold: int, sell_threshold: int) -> pd.DataFrame:
    """RSI閾値: RSI < buy_threshold → buy(1), RSI > sell_threshold → sell(-1)"""
    df = df.copy()
    df["rsi"] = add_indicator_rsi(df, period)
    df["signal"] = 0
    df.loc[df["rsi"] < buy_threshold, "signal"] = 1
    df.loc[df["rsi"] > sell_threshold, "signal"] = -1
    return df


# ─── ロングオンリー バックテスト ─────────────────────

def simulate_long_only(df: pd.DataFrame, initial_capital: float = 10000,
                       commission_rate: float = 0.0002) -> tuple:
    """ロングオンリーの指値バックテスト

    PineScriptと完全一致させるためのシンプルな実装:
    - signal=1 の足のclose で買い指値を発注 → 次足で low <= limit なら約定
    - signal=-1 の足のclose で売り指値を発注 → 次足で high >= limit なら約定
    - ショートは一切行わない
    - 注文は1足限り有効

    Returns:
        (trades_list, equity_series)
    """
    capital = initial_capital
    position = 0.0       # 保有数量
    entry_price = 0.0
    entry_time = None
    in_position = False
    trades = []
    equity = [capital]

    signals = df["signal"].values
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    timestamps = df["timestamp"].values

    for i in range(1, len(df)):
        prev_signal = signals[i - 1]
        limit_price = closes[i - 1]

        filled_close = False
        filled_open = False

        # ── 決済判定: ポジションあり & 前足が sell signal ──
        if in_position and prev_signal == -1:
            if highs[i] >= limit_price:
                filled_close = True
                exit_price = limit_price
                commission = position * exit_price * commission_rate
                pnl = position * (exit_price - entry_price) - commission
                pnl_pct = (exit_price - entry_price) / entry_price - commission_rate

                trades.append({
                    "trade_no": len(trades) + 1,
                    "entry_time": str(entry_time),
                    "entry_bar": None,  # 後で設定
                    "entry_price": round(entry_price, 8),
                    "exit_time": str(timestamps[i]),
                    "exit_bar": i,
                    "exit_price": round(exit_price, 8),
                    "qty": round(position, 8),
                    "pnl": round(pnl, 4),
                    "pnl_pct": round(pnl_pct * 100, 4),
                    "signal_bar_close": round(limit_price, 8),
                    "fill_bar_high": round(highs[i], 8),
                })
                capital += pnl
                position = 0.0
                in_position = False

        # ── エントリー判定: ノーポジ & 前足が buy signal ──
        if not in_position and prev_signal == 1:
            if lows[i] <= limit_price:
                filled_open = True
                entry_price = limit_price
                entry_time = timestamps[i]
                position_value = capital
                commission = position_value * commission_rate
                position = (position_value - commission) / entry_price

                # entry_bar を記録
                if trades or True:
                    entry_bar_idx = i

        # エクイティ計算
        if in_position:
            unrealized = position * (closes[i] - entry_price)
            equity.append(capital + unrealized)
        else:
            equity.append(capital)

        if filled_open:
            in_position = True
            # entry_barを保存 (次のトレードのcloseで使う)
            entry_bar_global = i

        if filled_close and not in_position:
            pass

    # 未決済ポジションがあれば最終足で決済
    if in_position and len(df) > 0:
        exit_price = closes[-1]
        commission = position * exit_price * commission_rate
        pnl = position * (exit_price - entry_price) - commission
        pnl_pct = (exit_price - entry_price) / entry_price - commission_rate
        trades.append({
            "trade_no": len(trades) + 1,
            "entry_time": str(entry_time),
            "entry_bar": None,
            "entry_price": round(entry_price, 8),
            "exit_time": str(timestamps[-1]),
            "exit_bar": len(df) - 1,
            "exit_price": round(exit_price, 8),
            "qty": round(position, 8),
            "pnl": round(pnl, 4),
            "pnl_pct": round(pnl_pct * 100, 4),
            "signal_bar_close": round(closes[-1], 8),
            "fill_bar_high": round(highs[-1], 8),
        })
        capital += pnl

    return trades, pd.Series(equity)


# ─── PineScript 生成 ──────────────────────────────

def generate_pinescript_ema_cross(fast: int, slow: int,
                                  initial_capital: int = 10000) -> str:
    """EMAクロス用の検証PineScriptを生成"""
    return f"""//@version=5
strategy("Validate: EMA Cross ({fast}/{slow})", overlay=true, initial_capital={initial_capital}, default_qty_type=strategy.percent_of_equity, default_qty_value=100, commission_type=strategy.commission.percent, commission_value=0.02)

// ── Indicators ──
ema_fast = ta.ema(close, {fast})
ema_slow = ta.ema(close, {slow})

// ── Signals ──
buy_signal = ema_fast > ema_slow
sell_signal = ema_fast < ema_slow

// ── Entry / Exit ──
strategy.cancel_all()
if buy_signal and strategy.position_size == 0
    strategy.entry("Long", strategy.long, limit=close)
if sell_signal and strategy.position_size > 0
    strategy.order("Exit Long", strategy.short, qty=strategy.position_size, limit=close)

// ── Plot ──
plot(ema_fast, color=color.green, title="EMA Fast")
plot(ema_slow, color=color.red, title="EMA Slow")
plotshape(buy_signal and strategy.position_size == 0, style=shape.triangleup, location=location.belowbar, color=color.green, size=size.small, title="Buy")
plotshape(sell_signal and strategy.position_size > 0, style=shape.triangledown, location=location.abovebar, color=color.red, size=size.small, title="Sell")
"""


def generate_pinescript_rsi_threshold(period: int, buy_threshold: int,
                                      sell_threshold: int,
                                      initial_capital: int = 10000) -> str:
    """RSI閾値用の検証PineScriptを生成"""
    return f"""//@version=5
strategy("Validate: RSI ({period}, {buy_threshold}/{sell_threshold})", overlay=true, initial_capital={initial_capital}, default_qty_type=strategy.percent_of_equity, default_qty_value=100, commission_type=strategy.commission.percent, commission_value=0.02)

// ── Indicators ──
rsi_val = ta.rsi(close, {period})

// ── Signals ──
buy_signal = rsi_val < {buy_threshold}
sell_signal = rsi_val > {sell_threshold}

// ── Entry / Exit ──
strategy.cancel_all()
if buy_signal and strategy.position_size == 0
    strategy.entry("Long", strategy.long, limit=close)
if sell_signal and strategy.position_size > 0
    strategy.order("Exit Long", strategy.short, qty=strategy.position_size, limit=close)

// ── Plot ──
plotshape(buy_signal and strategy.position_size == 0, style=shape.triangleup, location=location.belowbar, color=color.green, size=size.small, title="Buy")
plotshape(sell_signal and strategy.position_size > 0, style=shape.triangledown, location=location.abovebar, color=color.red, size=size.small, title="Sell")
"""


# ─── メトリクス計算 ────────────────────────────────

def calculate_metrics(trades: list, equity: pd.Series,
                      initial_capital: float) -> dict:
    """バックテスト結果のメトリクスを計算"""
    if not trades:
        return {
            "total_return_pct": 0.0, "net_profit": 0.0,
            "total_trades": 0, "win_rate": 0.0,
            "gross_profit": 0.0, "gross_loss": 0.0,
            "profit_factor": 0.0, "max_drawdown_pct": 0.0,
        }

    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    net_profit = sum(pnls)
    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # 最大ドローダウン
    peak = equity.expanding().max()
    dd_pct = ((equity - peak) / peak).min() * 100

    return {
        "net_profit": round(net_profit, 2),
        "total_return_pct": round(net_profit / initial_capital * 100, 2),
        "total_trades": len(trades),
        "win_rate": round(len(wins) / len(trades) * 100, 1),
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
        "profit_factor": round(profit_factor, 3),
        "max_drawdown_pct": round(dd_pct, 2),
    }


# ─── 出力 ─────────────────────────────────────────

def save_trades_csv(trades: list, path: str):
    """トレード一覧をCSV出力"""
    df = pd.DataFrame(trades)
    df.to_csv(path, index=False)
    return path


def save_signals_csv(df: pd.DataFrame, path: str, indicator_cols: list):
    """足ごとのシグナルとインジケータ値をCSV出力"""
    cols = ["timestamp", "open", "high", "low", "close", "volume"] + indicator_cols + ["signal"]
    out = df[cols].copy()
    out.to_csv(path, index=False)
    return path


# ─── メイン ────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="単一ロジック検証バックテスト")
    parser.add_argument("--symbol", default="BTCUSDT", help="通貨ペア")
    parser.add_argument("--days", type=int, default=7, help="データ期間(日)")
    parser.add_argument("--strategy", default="ema_cross",
                        choices=["ema_cross", "rsi_threshold"],
                        help="検証するロジック")
    # EMA cross params
    parser.add_argument("--fast", type=int, default=9, help="EMA fast period")
    parser.add_argument("--slow", type=int, default=21, help="EMA slow period")
    # RSI params
    parser.add_argument("--period", type=int, default=14, help="RSI period")
    parser.add_argument("--buy-threshold", type=int, default=30, help="RSI buy threshold")
    parser.add_argument("--sell-threshold", type=int, default=70, help="RSI sell threshold")
    # Backtest params
    parser.add_argument("--capital", type=float, default=10000, help="初期資金")
    parser.add_argument("--commission", type=float, default=0.0002, help="手数料率")
    parser.add_argument("--output-dir", default="results/validate", help="出力先ディレクトリ")

    args = parser.parse_args()

    print(f"=== 単一ロジック検証バックテスト ===")
    print(f"Symbol: {args.symbol}")
    print(f"Period: {args.days} days")
    print(f"Strategy: {args.strategy}")
    print()

    # データ取得
    print("データ取得中...")
    df = get_data(args.symbol, days=args.days)
    if df.empty:
        print("ERROR: データが取得できませんでした")
        sys.exit(1)
    print(f"データ: {len(df)} bars ({df['timestamp'].iloc[0]} ~ {df['timestamp'].iloc[-1]})")
    print()

    # シグナル生成
    indicator_cols = []
    pine_code = ""

    if args.strategy == "ema_cross":
        print(f"EMA Cross: fast={args.fast}, slow={args.slow}")
        df = generate_ema_cross_signals(df, args.fast, args.slow)
        indicator_cols = ["ema_fast", "ema_slow"]
        pine_code = generate_pinescript_ema_cross(args.fast, args.slow, int(args.capital))
    elif args.strategy == "rsi_threshold":
        print(f"RSI Threshold: period={args.period}, buy<{args.buy_threshold}, sell>{args.sell_threshold}")
        df = generate_rsi_threshold_signals(df, args.period, args.buy_threshold, args.sell_threshold)
        indicator_cols = ["rsi"]
        pine_code = generate_pinescript_rsi_threshold(
            args.period, args.buy_threshold, args.sell_threshold, int(args.capital))

    # バックテスト実行
    print()
    print("バックテスト実行中 (ロングオンリー)...")
    trades, equity = simulate_long_only(df, args.capital, args.commission)
    metrics = calculate_metrics(trades, equity, args.capital)

    # 結果表示
    print()
    print("=" * 60)
    print("  PYTHON BACKTEST RESULTS")
    print("=" * 60)
    print(f"  Net Profit:      {metrics['net_profit']:>10.2f} USDT ({metrics['total_return_pct']:>+.2f}%)")
    print(f"  Total Trades:    {metrics['total_trades']:>10d}")
    print(f"  Win Rate:        {metrics['win_rate']:>10.1f}%")
    print(f"  Gross Profit:    {metrics['gross_profit']:>10.2f} USDT")
    print(f"  Gross Loss:      {metrics['gross_loss']:>10.2f} USDT")
    print(f"  Profit Factor:   {metrics['profit_factor']:>10.3f}")
    print(f"  Max Drawdown:    {metrics['max_drawdown_pct']:>10.2f}%")
    print("=" * 60)

    # トレード一覧表示
    if trades:
        print()
        print("  TRADE LIST")
        print("-" * 100)
        print(f"  {'#':>3}  {'Entry Time':>20}  {'Entry$':>12}  {'Exit Time':>20}  {'Exit$':>12}  {'PnL':>10}  {'PnL%':>8}")
        print("-" * 100)
        for t in trades:
            entry_t = t["entry_time"][:19] if len(t["entry_time"]) > 19 else t["entry_time"]
            exit_t = t["exit_time"][:19] if len(t["exit_time"]) > 19 else t["exit_time"]
            print(f"  {t['trade_no']:>3}  {entry_t:>20}  {t['entry_price']:>12.2f}  "
                  f"{exit_t:>20}  {t['exit_price']:>12.2f}  {t['pnl']:>+10.2f}  {t['pnl_pct']:>+7.2f}%")
        print("-" * 100)

    # ファイル出力
    os.makedirs(args.output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    trades_path = os.path.join(args.output_dir, f"trades_{args.strategy}_{args.symbol}_{ts}.csv")
    signals_path = os.path.join(args.output_dir, f"signals_{args.strategy}_{args.symbol}_{ts}.csv")
    pine_path = os.path.join(args.output_dir, f"pinescript_{args.strategy}_{args.symbol}_{ts}.pine")

    save_trades_csv(trades, trades_path)
    save_signals_csv(df, signals_path, indicator_cols)
    with open(pine_path, "w") as f:
        f.write(pine_code)

    print()
    print(f"  Output files:")
    print(f"    Trades CSV:  {trades_path}")
    print(f"    Signals CSV: {signals_path}")
    print(f"    PineScript:  {pine_path}")
    print()
    print("  PineScript (TradingViewにコピー):")
    print("  " + "─" * 58)
    for line in pine_code.strip().split("\n"):
        print(f"  {line}")
    print("  " + "─" * 58)
    print()
    print("  照合手順:")
    print("  1. 上のPineScriptをTradingViewに貼り付け")
    print(f"  2. {args.symbol} の1分足チャートに適用")
    print(f"  3. テスター期間を {df['timestamp'].iloc[0]} ~ {df['timestamp'].iloc[-1]} に合わせる")
    print("  4. TradingViewの「トレード一覧」とTrades CSVを比較")
    print("  5. 差異があれば Signals CSV でインジケータ値を照合")


if __name__ == "__main__":
    main()
