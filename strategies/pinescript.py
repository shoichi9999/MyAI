"""CompositeStrategy → TradingView PineScript v5 変換モジュール"""


def to_pinescript(params: dict, title: str = "Generated Strategy") -> str:
    """CompositeStrategyのparamsをPineScript v5コードに変換する。

    Args:
        params: CompositeStrategyのparams dict
        title: インジケーター/ストラテジー名

    Returns:
        PineScript v5コード文字列
    """
    buy_conditions = params.get("buy_conditions", [])
    sell_conditions = params.get("sell_conditions", [])
    buy_logic = params.get("buy_logic", "and")
    sell_logic = params.get("sell_logic", "and")
    signal_mode = params.get("signal_mode", "state")

    lines = []
    lines.append('//@version=5')
    lines.append(f'strategy("{_escape(title)}", overlay=true, initial_capital=10000, default_qty_type=strategy.percent_of_equity, default_qty_value=100, commission_type=strategy.commission.percent, commission_value=0.02)')
    lines.append('')

    # 各条件の変数宣言
    buy_var_names = []
    for i, cond in enumerate(buy_conditions):
        var_name = f"buy_cond_{i}"
        buy_var_names.append(var_name)
        code = _condition_to_pine(cond["type"], cond["params"], var_name)
        lines.append(code)

    sell_var_names = []
    for i, cond in enumerate(sell_conditions):
        var_name = f"sell_cond_{i}"
        sell_var_names.append(var_name)
        code = _condition_to_pine(cond["type"], cond["params"], var_name)
        lines.append(code)

    # 条件の結合
    if buy_var_names:
        joiner = " and " if buy_logic == "and" else " or "
        buy_expr = joiner.join(buy_var_names)
    else:
        buy_expr = "false"

    if sell_var_names:
        joiner = " and " if sell_logic == "and" else " or "
        sell_expr = joiner.join(sell_var_names)
    else:
        sell_expr = "false"

    lines.append('// ── Signal Logic ──')

    if signal_mode == "cross":
        lines.append(f'buy_raw = {buy_expr}')
        lines.append(f'sell_raw = {sell_expr}')
        lines.append('buy_signal = buy_raw and not buy_raw[1]')
        lines.append('sell_signal = sell_raw and not sell_raw[1]')
    else:
        lines.append(f'buy_signal = {buy_expr}')
        lines.append(f'sell_signal = {sell_expr}')

    lines.append('')
    lines.append('// ── Entry / Exit ──')
    lines.append('// 毎足、未約定の指値注文をキャンセル (1足限りの有効期間)')
    lines.append('strategy.cancel_all()')
    lines.append('if buy_signal and strategy.position_size == 0')
    lines.append('    strategy.entry("Long", strategy.long, limit=close)')
    lines.append('if sell_signal and strategy.position_size > 0')
    lines.append('    strategy.close("Long")')
    lines.append('')
    lines.append('// ── Plot ──')
    lines.append('plotshape(buy_signal, style=shape.triangleup, location=location.belowbar, color=color.green, size=size.small, title="Buy")')
    lines.append('plotshape(sell_signal, style=shape.triangledown, location=location.abovebar, color=color.red, size=size.small, title="Sell")')

    return "\n".join(lines)


def _escape(s: str) -> str:
    return s.replace('"', '\\"')


def _condition_to_pine(cond_type: str, params: dict, var_name: str) -> str:
    """条件タイプごとにPineScriptの変数宣言コードを生成する。"""
    generators = {
        "rsi_threshold": _pine_rsi_threshold,
        "ema_cross": _pine_ema_cross,
        "sma_cross": _pine_sma_cross,
        "bb_position": _pine_bb_position,
        "macd_hist_sign": _pine_macd_hist_sign,
        "macd_cross": _pine_macd_cross,
        "stoch_threshold": _pine_stoch_threshold,
        "stoch_cross": _pine_stoch_cross,
        "atr_breakout": _pine_atr_breakout,
        "price_vs_sma": _pine_price_vs_sma,
        "price_vs_ema": _pine_price_vs_ema,
        "volume_spike": _pine_volume_spike,
        "price_momentum": _pine_price_momentum,
        "candle_body": _pine_candle_body,
    }
    gen = generators.get(cond_type)
    if gen is None:
        return f"{var_name} = true  // unknown condition: {cond_type}"
    return gen(params, var_name)


# ─── 各条件タイプの PineScript 生成 ───────────────────

def _pine_rsi_threshold(p: dict, var: str) -> str:
    period = p["period"]
    threshold = p["threshold"]
    direction = p["direction"]
    uid = var
    op = "<" if direction == "below" else ">"
    return (
        f"// RSI threshold ({direction} {threshold})\n"
        f"{uid}_rsi = ta.rsi(close, {period})\n"
        f"{var} = {uid}_rsi {op} {threshold}\n"
    )


def _pine_ema_cross(p: dict, var: str) -> str:
    fast = p["fast_period"]
    slow = p["slow_period"]
    direction = p["direction"]
    uid = var
    op = ">" if direction == "above" else "<"
    return (
        f"// EMA cross (fast={fast}, slow={slow}, {direction})\n"
        f"{uid}_fast = ta.ema(close, {fast})\n"
        f"{uid}_slow = ta.ema(close, {slow})\n"
        f"{var} = {uid}_fast {op} {uid}_slow\n"
    )


def _pine_sma_cross(p: dict, var: str) -> str:
    fast = p["fast_period"]
    slow = p["slow_period"]
    direction = p["direction"]
    uid = var
    op = ">" if direction == "above" else "<"
    return (
        f"// SMA cross (fast={fast}, slow={slow}, {direction})\n"
        f"{uid}_fast = ta.sma(close, {fast})\n"
        f"{uid}_slow = ta.sma(close, {slow})\n"
        f"{var} = {uid}_fast {op} {uid}_slow\n"
    )


def _pine_bb_position(p: dict, var: str) -> str:
    period = p["period"]
    std_dev = p["std_dev"]
    direction = p["direction"]
    uid = var
    lines = [
        f"// Bollinger Band position ({direction})",
        f"[{uid}_mid, {uid}_upper, {uid}_lower] = ta.bb(close, {period}, {std_dev})",
    ]
    if direction == "above_upper":
        lines.append(f"{var} = close > {uid}_upper")
    elif direction == "below_lower":
        lines.append(f"{var} = close < {uid}_lower")
    elif direction == "above_mid":
        lines.append(f"{var} = close > {uid}_mid")
    elif direction == "below_mid":
        lines.append(f"{var} = close < {uid}_mid")
    else:
        lines.append(f"{var} = true  // unknown bb direction: {direction}")
    return "\n".join(lines) + "\n"


def _pine_macd_hist_sign(p: dict, var: str) -> str:
    fast = p["fast"]
    slow = p["slow"]
    sig = p["signal_period"]
    direction = p["direction"]
    uid = var
    op = ">" if direction == "positive" else "<"
    return (
        f"// MACD histogram sign ({direction})\n"
        f"[{uid}_line, {uid}_sig, {uid}_hist] = ta.macd(close, {fast}, {slow}, {sig})\n"
        f"{var} = {uid}_hist {op} 0\n"
    )


def _pine_macd_cross(p: dict, var: str) -> str:
    fast = p["fast"]
    slow = p["slow"]
    sig = p["signal_period"]
    direction = p["direction"]
    uid = var
    op = ">" if direction == "above" else "<"
    return (
        f"// MACD cross ({direction})\n"
        f"[{uid}_line, {uid}_sig, {uid}_hist] = ta.macd(close, {fast}, {slow}, {sig})\n"
        f"{var} = {uid}_line {op} {uid}_sig\n"
    )


def _pine_stoch_threshold(p: dict, var: str) -> str:
    k = p["k_period"]
    d = p["d_period"]
    threshold = p["threshold"]
    direction = p["direction"]
    uid = var
    op = "<" if direction == "below" else ">"
    return (
        f"// Stochastic threshold ({direction} {threshold})\n"
        f"{uid}_k = ta.stoch(close, high, low, {k})\n"
        f"{uid}_d = ta.sma({uid}_k, {d})\n"
        f"{var} = {uid}_k {op} {threshold}\n"
    )


def _pine_stoch_cross(p: dict, var: str) -> str:
    k = p["k_period"]
    d = p["d_period"]
    direction = p["direction"]
    uid = var
    op = ">" if direction == "above" else "<"
    return (
        f"// Stochastic cross ({direction})\n"
        f"{uid}_k = ta.stoch(close, high, low, {k})\n"
        f"{uid}_d = ta.sma({uid}_k, {d})\n"
        f"{var} = {uid}_k {op} {uid}_d\n"
    )


def _pine_atr_breakout(p: dict, var: str) -> str:
    period = p["period"]
    lookback = p["lookback"]
    mult = p["multiplier"]
    direction = p["direction"]
    uid = var
    if direction == "up":
        return (
            f"// ATR breakout (up)\n"
            f"{uid}_atr = ta.atr({period})\n"
            f"{uid}_high = ta.highest(high, {lookback})\n"
            f"{var} = close > {uid}_high - {uid}_atr * {mult}\n"
        )
    else:
        return (
            f"// ATR breakout (down)\n"
            f"{uid}_atr = ta.atr({period})\n"
            f"{uid}_low = ta.lowest(low, {lookback})\n"
            f"{var} = close < {uid}_low + {uid}_atr * {mult}\n"
        )


def _pine_price_vs_sma(p: dict, var: str) -> str:
    period = p["period"]
    direction = p["direction"]
    uid = var
    op = ">" if direction == "above" else "<"
    return (
        f"// Price vs SMA({period}) ({direction})\n"
        f"{uid}_sma = ta.sma(close, {period})\n"
        f"{var} = close {op} {uid}_sma\n"
    )


def _pine_price_vs_ema(p: dict, var: str) -> str:
    period = p["period"]
    direction = p["direction"]
    uid = var
    op = ">" if direction == "above" else "<"
    return (
        f"// Price vs EMA({period}) ({direction})\n"
        f"{uid}_ema = ta.ema(close, {period})\n"
        f"{var} = close {op} {uid}_ema\n"
    )


def _pine_volume_spike(p: dict, var: str) -> str:
    period = p["period"]
    mult = p["multiplier"]
    uid = var
    return (
        f"// Volume spike (>{mult}x avg)\n"
        f"{uid}_avg_vol = ta.sma(volume, {period})\n"
        f"{var} = volume > {uid}_avg_vol * {mult}\n"
    )


def _pine_price_momentum(p: dict, var: str) -> str:
    period = p["period"]
    threshold = p["threshold"]
    direction = p["direction"]
    uid = var
    if direction == "positive":
        return (
            f"// Price momentum (positive > {threshold})\n"
            f"{uid}_mom = (close - close[{period}]) / close[{period}]\n"
            f"{var} = {uid}_mom > {threshold}\n"
        )
    else:
        return (
            f"// Price momentum (negative < -{threshold})\n"
            f"{uid}_mom = (close - close[{period}]) / close[{period}]\n"
            f"{var} = {uid}_mom < -{threshold}\n"
        )


def _pine_candle_body(p: dict, var: str) -> str:
    threshold = p["threshold"]
    direction = p["direction"]
    uid = var
    if direction == "bullish":
        return (
            f"// Candle body (bullish > {threshold})\n"
            f"{uid}_body = (close - open) / open\n"
            f"{var} = {uid}_body > {threshold}\n"
        )
    else:
        return (
            f"// Candle body (bearish < -{threshold})\n"
            f"{uid}_body = (close - open) / open\n"
            f"{var} = {uid}_body < -{threshold}\n"
        )
