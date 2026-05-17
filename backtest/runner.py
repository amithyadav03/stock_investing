"""
Walk-Forward Backtest Engine for Alpha Agent.

Design:
- Uses rule-based conviction scorer (no LLM) for speed and reproducibility.
- Walk-forward windows: 6-month in-sample (warmup), 3-month out-of-sample (test).
- Transaction costs: 0.4% round-trip + 0.1% slippage = 0.5% total per trade.
- Benchmark: Nifty 50 buy-and-hold.
- Output: per-trade log, equity curve, and summary statistics.

Usage:
    from backtest.runner import run_backtest
    results = run_backtest(
        symbols=['RELIANCE', 'INFY', 'HDFCBANK', 'TATAMOTORS', 'SUNPHARMA',
                 'AXISBANK', 'BAJFINANCE', 'WIPRO', 'CIPLA', 'MARUTI'],
        start_date='2019-01-01',
        end_date='2024-12-31',
        strategy_type='swing',
        initial_capital=500000,
    )
"""

import os
import json
import warnings
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings('ignore')

# ── Constants ──────────────────────────────────────────────────────────────────

TRANSACTION_COST_PCT = 0.002   # 0.2% one way (brokerage + STT + charges)
SLIPPAGE_PCT = 0.001           # 0.1% slippage per trade entry/exit
TOTAL_COST_ONE_WAY = TRANSACTION_COST_PCT + SLIPPAGE_PCT  # 0.3% per leg = 0.6% round-trip

STRATEGY_PARAMS = {
    "swing": {
        "holding_days_max": 30,
        "min_rr": 1.5,
        "atr_sl_mult": 2.0,
        "atr_tp_mult": 3.0,  # TP = entry + 3×ATR
        "conviction_threshold": 65,
        "risk_pct": 0.01,
    },
    "positional": {
        "holding_days_max": 180,
        "min_rr": 2.0,
        "atr_sl_mult": 2.5,
        "atr_tp_mult": 5.0,
        "conviction_threshold": 72,
        "risk_pct": 0.01,
    },
}


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class BacktestTrade:
    symbol: str
    entry_date: str
    exit_date: str
    direction: str          # BUY or SELL
    entry_price: float
    exit_price: float
    quantity: int
    stop_loss: float
    take_profit: float
    exit_reason: str        # TAKE_PROFIT, STOP_LOSS, MAX_HOLD, FORCED_CLOSE
    gross_pnl: float
    cost: float
    net_pnl: float
    net_pnl_pct: float
    holding_days: int
    conviction_score: int
    strategy_type: str


@dataclass
class BacktestResults:
    strategy_type: str
    start_date: str
    end_date: str
    initial_capital: float
    final_capital: float
    total_return_pct: float
    benchmark_return_pct: float     # Nifty 50 buy-and-hold
    alpha_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    total_trades: int
    win_rate_pct: float
    avg_win_pct: float
    avg_loss_pct: float
    profit_factor: float            # gross wins / gross losses
    avg_holding_days: float
    trades: List[BacktestTrade] = field(default_factory=list)
    equity_curve: List[dict] = field(default_factory=list)
    monte_carlo: dict = field(default_factory=dict)


# ── Indicator helpers (deterministic, no LLM) ─────────────────────────────────

def _compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all indicators needed for signal generation."""
    from tools.indicators import (
        atr, rsi, macd, bbands, adx, ema, stoch,
        vwap, vwap_deviation, obv, momentum_6m
    )
    df = df.copy()
    df['ATR'] = atr(df, 14)
    df['RSI'] = rsi(df, 14)
    df['EMA_20'] = ema(df, 20)
    df['EMA_50'] = ema(df, 50)
    df['EMA_200'] = ema(df, 200)
    macd_l, macd_s, macd_h = macd(df, 12, 26, 9)
    df['MACD_HIST'] = macd_h
    df['ADX'] = adx(df, 14)[0]
    stoch_k, _ = stoch(df, 14, 3)
    df['STOCH_K'] = stoch_k
    bb_u, _, bb_l, bb_p = bbands(df, 20, 2.0)
    df['BB_PCT'] = bb_p
    try:
        df['VWAP'] = vwap(df)
        df['VWAP_DEV'] = vwap_deviation(df)
        df['OBV'] = obv(df)
    except Exception:
        df['VWAP'] = df['EMA_20']
        df['VWAP_DEV'] = 0.0
        df['OBV'] = 0.0
    return df


def _rule_based_conviction(row: pd.Series, strategy_type: str = "swing") -> int:
    """
    Deterministic conviction score (0-100) using rule-based logic.
    Mirrors agents/conviction_scorer._rule_based_score but uses computed columns.
    No LLM calls.
    """
    score = 0

    rsi_val = row.get('RSI', 50)
    adx_val = row.get('ADX', 20)
    macd_h = row.get('MACD_HIST', 0)
    price = row.get('Close', 0)
    ema20 = row.get('EMA_20', 0)
    ema50 = row.get('EMA_50', 0)
    ema200 = row.get('EMA_200', 0)
    bb_pct = row.get('BB_PCT', 0.5)
    vwap_dev = row.get('VWAP_DEV', 0)

    # Technical score (0-40)
    if 40 <= rsi_val <= 65: score += 10
    elif 30 <= rsi_val < 40 or 65 < rsi_val <= 75: score += 5
    elif rsi_val > 80 or rsi_val < 25: score -= 5  # Overextended

    if adx_val > 25: score += 10
    elif adx_val > 18: score += 5

    if macd_h > 0: score += 8

    # Price above EMAs (trend alignment)
    if price > ema20 > ema50: score += 7
    elif price > ema20: score += 4

    if ema50 > ema200: score += 5  # Long-term bull trend

    # BB not overextended
    if 0.2 <= bb_pct <= 0.8: score += 3
    elif bb_pct > 0.95 or bb_pct < 0.05: score -= 3

    # VWAP confirmation
    if 0 < vwap_dev < 3: score += 5   # Slightly above VWAP = accumulation
    elif vwap_dev > 5: score -= 3     # Too far above = extended
    elif vwap_dev < -5: score -= 5    # Below VWAP = distribution

    # Macro/fundamental assumed NEUTRAL in pure technical backtest (add 12 pts baseline)
    score += 12

    # Strategy-specific adjustments
    if strategy_type == "positional":
        # Require stronger trend for positional
        if ema20 > ema50 > ema200:
            score += 5
        else:
            score -= 5

    return max(0, min(100, score))


def _generate_signal(df: pd.DataFrame, idx: int, strategy_type: str) -> Optional[dict]:
    """
    Generate a BUY signal at position idx if conviction passes threshold.
    Returns signal dict or None.
    NOTE: Only BUY signals for simplicity (long-only backtest).
    """
    if idx < 200:  # Need warmup period for EMA_200
        return None

    row = df.iloc[idx]
    params = STRATEGY_PARAMS[strategy_type]

    conviction = _rule_based_conviction(row, strategy_type)
    if conviction < params["conviction_threshold"]:
        return None

    price = float(row['Close'])
    atr_val = float(row['ATR'])

    if atr_val <= 0 or price <= 0:
        return None

    sl = price - (atr_val * params["atr_sl_mult"])
    tp = price + (atr_val * params["atr_tp_mult"])
    rr = (tp - price) / (price - sl) if price > sl else 0

    if rr < params["min_rr"]:
        return None

    # Apply transaction costs to entry price (slippage = buy slightly higher)
    entry_price = price * (1 + TOTAL_COST_ONE_WAY)

    return {
        "entry_price": round(entry_price, 2),
        "stop_loss": round(sl, 2),
        "take_profit": round(tp, 2),
        "atr": atr_val,
        "conviction": conviction,
        "rr": round(rr, 2),
    }


# ── Data loading ───────────────────────────────────────────────────────────────

def _load_data(symbols: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
    """Download historical data for all symbols. Uses yfinance."""
    print(f"[Backtest] Downloading data for {len(symbols)} symbols ({start_date} to {end_date})...")

    # Add buffer for indicator warmup
    start_dt = datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=250)
    start_buffered = start_dt.strftime("%Y-%m-%d")

    data = {}

    # Download Nifty 50 benchmark
    try:
        nifty_df = yf.Ticker("^NSEI").history(start=start_buffered, end=end_date)
        if not nifty_df.empty:
            data["_NIFTY50"] = _compute_indicators(nifty_df)
            print(f"[Backtest] Nifty 50 loaded: {len(nifty_df)} bars")
    except Exception as e:
        print(f"[Backtest] Warning: Nifty 50 data failed: {e}")

    # Download individual symbols in batch
    yf_symbols = [f"{s}.NS" for s in symbols]
    try:
        raw = yf.download(
            yf_symbols, start=start_buffered, end=end_date,
            group_by='ticker', threads=True, progress=False,
            auto_adjust=True,  # Adjust for splits/dividends
        )
    except Exception as e:
        print(f"[Backtest] Batch download failed: {e}. Trying individual downloads...")
        raw = None

    for s in symbols:
        try:
            if raw is not None and len(symbols) > 1:
                key = f"{s}.NS"
                if hasattr(raw.columns, 'get_level_values') and key in raw.columns.get_level_values(0):
                    df = raw[key].copy()
                else:
                    df = yf.Ticker(f"{s}.NS").history(start=start_buffered, end=end_date)
            else:
                df = yf.Ticker(f"{s}.NS").history(start=start_buffered, end=end_date)

            df = df.dropna(subset=['Close', 'Volume'])

            if len(df) < 250:
                print(f"[Backtest] Insufficient data for {s} ({len(df)} bars). Skipping.")
                continue

            data[s] = _compute_indicators(df)

        except Exception as e:
            print(f"[Backtest] Data loading failed for {s}: {e}")

    print(f"[Backtest] Loaded data for {len(data) - 1} symbols (plus Nifty).")
    return data


# ── Walk-forward simulation ────────────────────────────────────────────────────

def _simulate_symbol(
    symbol: str,
    df: pd.DataFrame,
    test_start: datetime,
    test_end: datetime,
    strategy_type: str,
    initial_capital: float,
    max_positions: int = 3,
) -> Tuple[List[BacktestTrade], float]:
    """
    Simulate trading for one symbol in one test window.
    Returns (trades, final_pnl).
    """
    params = STRATEGY_PARAMS[strategy_type]
    trades = []
    active_trade = None
    capital = initial_capital

    # Filter to test period
    test_df = df.loc[test_start:test_end].copy() if not df.empty else pd.DataFrame()
    if test_df.empty or len(test_df) < 10:
        return [], 0.0

    for i in range(len(test_df)):
        row = test_df.iloc[i]
        current_date = test_df.index[i]
        current_price = float(row['Close'])

        if active_trade:
            holding_days = (current_date - datetime.fromisoformat(active_trade['entry_date'])).days

            exit_reason = None
            exit_price = current_price

            # Check stop loss (use intraday low for realistic exit)
            if float(row['Low']) <= active_trade['stop_loss']:
                exit_reason = "STOP_LOSS"
                exit_price = active_trade['stop_loss']  # Assume fill at SL
            # Check take profit (use intraday high)
            elif float(row['High']) >= active_trade['take_profit']:
                exit_reason = "TAKE_PROFIT"
                exit_price = active_trade['take_profit']
            # Max holding period
            elif holding_days >= params["holding_days_max"]:
                exit_reason = "MAX_HOLD"

            if exit_reason:
                # Apply exit cost (slippage + transaction cost)
                exit_price_adj = exit_price * (1 - TOTAL_COST_ONE_WAY)
                qty = active_trade['quantity']
                gross_pnl = (exit_price_adj - active_trade['entry_price']) * qty
                cost_amt = exit_price * qty * TOTAL_COST_ONE_WAY + active_trade['entry_price'] * qty * TOTAL_COST_ONE_WAY
                net_pnl = gross_pnl - cost_amt / 2  # Cost already baked into prices
                net_pnl_pct = round((exit_price_adj - active_trade['entry_price']) / active_trade['entry_price'] * 100, 2)

                trade = BacktestTrade(
                    symbol=symbol,
                    entry_date=active_trade['entry_date'],
                    exit_date=str(current_date.date()),
                    direction="BUY",
                    entry_price=active_trade['entry_price'],
                    exit_price=round(exit_price_adj, 2),
                    quantity=qty,
                    stop_loss=active_trade['stop_loss'],
                    take_profit=active_trade['take_profit'],
                    exit_reason=exit_reason,
                    gross_pnl=round(gross_pnl, 2),
                    cost=round(cost_amt, 2),
                    net_pnl=round(net_pnl, 2),
                    net_pnl_pct=net_pnl_pct,
                    holding_days=holding_days,
                    conviction_score=active_trade['conviction'],
                    strategy_type=strategy_type,
                )
                trades.append(trade)
                capital += net_pnl
                active_trade = None

        # Look for new entry (only if no active trade)
        if not active_trade:
            full_idx = df.index.get_loc(current_date) if current_date in df.index else None
            if full_idx is None:
                continue

            signal = _generate_signal(df, full_idx, strategy_type)
            if signal:
                # Use next bar's Open as entry (can't trade at today's close when decision is made at close)
                next_idx = full_idx + 1
                if next_idx >= len(df):
                    continue
                next_open = float(df.iloc[next_idx]['Open'])
                actual_entry = round(next_open * (1 + TOTAL_COST_ONE_WAY), 2)

                # Recalculate SL/TP relative to actual entry (keep same ATR distances)
                atr_val = signal['atr']
                params_local = STRATEGY_PARAMS[strategy_type]
                actual_sl = round(actual_entry - atr_val * params_local["atr_sl_mult"], 2)
                actual_tp = round(actual_entry + atr_val * params_local["atr_tp_mult"], 2)
                risk_per_share = actual_entry - actual_sl
                if risk_per_share <= 0:
                    continue

                risk_amount = capital * params["risk_pct"]
                quantity = max(1, int(risk_amount / risk_per_share))
                max_qty = max(1, int((capital * 0.20) / actual_entry))
                quantity = min(quantity, max_qty)

                # Use next bar's date as actual entry date
                next_date = df.index[next_idx]
                active_trade = {
                    "entry_date": str(next_date.date()),
                    "entry_price": actual_entry,
                    "stop_loss": actual_sl,
                    "take_profit": actual_tp,
                    "quantity": quantity,
                    "conviction": signal['conviction'],
                }

    # Force close any open position at end of window
    if active_trade and len(test_df) > 0:
        exit_price = float(test_df.iloc[-1]['Close'])
        exit_price_adj = exit_price * (1 - TOTAL_COST_ONE_WAY)
        qty = active_trade['quantity']
        net_pnl = (exit_price_adj - active_trade['entry_price']) * qty
        holding_days = (test_df.index[-1] - datetime.fromisoformat(active_trade['entry_date'])).days
        trades.append(BacktestTrade(
            symbol=symbol,
            entry_date=active_trade['entry_date'],
            exit_date=str(test_df.index[-1].date()),
            direction="BUY",
            entry_price=active_trade['entry_price'],
            exit_price=round(exit_price_adj, 2),
            quantity=qty,
            stop_loss=active_trade['stop_loss'],
            take_profit=active_trade['take_profit'],
            exit_reason="FORCED_CLOSE",
            gross_pnl=round(net_pnl, 2),
            cost=0.0,
            net_pnl=round(net_pnl, 2),
            net_pnl_pct=round((exit_price_adj - active_trade['entry_price']) / active_trade['entry_price'] * 100, 2),
            holding_days=holding_days,
            conviction_score=active_trade['conviction'],
            strategy_type=strategy_type,
        ))

    return trades, sum(t.net_pnl for t in trades)


def _compute_sharpe(returns: pd.Series, risk_free_rate: float = 0.07) -> float:
    """Annualized Sharpe ratio. Uses daily returns, risk-free = 7% (India 10yr bond)."""
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    excess = returns - (risk_free_rate / 252)
    return round(float(np.sqrt(252) * excess.mean() / excess.std()), 2)


def _compute_max_drawdown(equity_curve: pd.Series) -> float:
    """Max peak-to-trough drawdown as a percentage."""
    if len(equity_curve) < 2:
        return 0.0
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    return round(float(drawdown.min() * 100), 2)


def run_monte_carlo(
    trades: list,
    initial_capital: float,
    n_simulations: int = 1000,
    risk_free_rate: float = 0.07,
) -> dict:
    """
    Monte Carlo robustness test: shuffle trade order 1000 times.
    Returns distribution of Sharpe ratios, max drawdowns, and final returns.
    Industry standard: 5th-percentile Sharpe > 0 indicates genuine robustness.
    """
    if len(trades) < 10:
        return {"error": "Insufficient trades for Monte Carlo (need >= 10)"}

    pnl_list = [t.net_pnl for t in trades]
    sharpes = []
    drawdowns = []
    final_returns = []

    rng = np.random.default_rng(42)
    for _ in range(n_simulations):
        shuffled = rng.permutation(pnl_list)
        equity = initial_capital + np.cumsum(shuffled)
        equity_series = pd.Series(np.concatenate([[initial_capital], equity]))
        daily_ret = equity_series.pct_change().dropna()
        sharpe = _compute_sharpe(daily_ret, risk_free_rate)
        dd = _compute_max_drawdown(equity_series)
        total_ret = (equity_series.iloc[-1] - initial_capital) / initial_capital * 100
        sharpes.append(sharpe)
        drawdowns.append(dd)
        final_returns.append(total_ret)

    sharpes_arr = np.array(sharpes)
    dd_arr = np.array(drawdowns)
    ret_arr = np.array(final_returns)

    p5_sharpe = float(np.percentile(sharpes_arr, 5))
    p95_sharpe = float(np.percentile(sharpes_arr, 95))
    p5_dd = float(np.percentile(dd_arr, 5))
    median_ret = float(np.median(ret_arr))

    is_robust = p5_sharpe > 0.0
    print(f"\n[Monte Carlo] {n_simulations} simulations on {len(trades)} trades:")
    print(f"  Sharpe — median: {np.median(sharpes_arr):.2f} | p5: {p5_sharpe:.2f} | p95: {p95_sharpe:.2f}")
    print(f"  Max DD — median: {np.median(dd_arr):.1f}% | worst 5%: {p5_dd:.1f}%")
    print(f"  Return — median: {median_ret:.1f}%")
    print(f"  Robustness: {'PASS (p5 Sharpe > 0)' if is_robust else 'FAIL (p5 Sharpe <= 0 — strategy fragile)'}")

    return {
        "n_simulations": n_simulations,
        "sharpe_median": round(float(np.median(sharpes_arr)), 2),
        "sharpe_p5": round(p5_sharpe, 2),
        "sharpe_p95": round(p95_sharpe, 2),
        "max_dd_median_pct": round(float(np.median(dd_arr)), 2),
        "max_dd_worst5_pct": round(p5_dd, 2),
        "return_median_pct": round(median_ret, 2),
        "is_robust": is_robust,
    }


def compute_walk_forward_efficiency(
    is_sharpe: float,
    oos_sharpe: float,
) -> float:
    """
    Walk-Forward Efficiency (WFE) = OOS annualized return / IS annualized return.
    WFE > 50% = robust; WFE < 30% = overfit.
    Here we use Sharpe ratio as proxy since it's risk-adjusted.
    """
    if is_sharpe <= 0:
        return 0.0
    wfe = round(oos_sharpe / is_sharpe * 100, 1)
    print(f"[WFE] Walk-Forward Efficiency: {wfe}% (IS Sharpe: {is_sharpe:.2f}, OOS Sharpe: {oos_sharpe:.2f})")
    if wfe >= 50:
        print(f"  PASS: WFE >= 50% — strategy performance carries to unseen data.")
    elif wfe >= 30:
        print(f"  MARGINAL: WFE 30-50% — some overfitting detected.")
    else:
        print(f"  FAIL: WFE < 30% — strategy is overfit to in-sample data.")
    return wfe


# ── Main backtest function ─────────────────────────────────────────────────────

def run_backtest(
    symbols: List[str],
    start_date: str,
    end_date: str = None,
    strategy_type: str = "swing",
    initial_capital: float = 500000,
    in_sample_months: int = 6,
    out_of_sample_months: int = 3,
    output_dir: str = "./backtest/results",
) -> BacktestResults:
    """
    Run walk-forward backtest over the specified date range.

    Args:
        symbols: List of NSE symbols to backtest
        start_date: Backtest start (YYYY-MM-DD)
        end_date: Backtest end (YYYY-MM-DD), defaults to today
        strategy_type: 'swing' or 'positional'
        initial_capital: Starting capital in INR
        in_sample_months: In-sample window for warmup (not traded)
        out_of_sample_months: Out-of-sample window (traded)
        output_dir: Directory to save results CSV

    Returns:
        BacktestResults with full trade log, equity curve, and statistics
    """
    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"WALK-FORWARD BACKTEST: {strategy_type.upper()}")
    print(f"Period: {start_date} -> {end_date}")
    print(f"Symbols: {len(symbols)} | Capital: Rs{initial_capital:,.0f}")
    print(f"Walk-forward: {in_sample_months}m in-sample, {out_of_sample_months}m out-of-sample")
    print(f"Transaction costs: {TOTAL_COST_ONE_WAY*200:.1f}% round-trip")
    print(f"{'='*60}\n")

    # Load all data
    all_data = _load_data(symbols, start_date, end_date)
    if not all_data:
        raise ValueError("No data loaded. Check symbols and date range.")

    # Generate walk-forward windows
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    windows = []
    window_start = start_dt
    while True:
        test_start = window_start + timedelta(days=in_sample_months * 30)
        test_end = test_start + timedelta(days=out_of_sample_months * 30)
        if test_end > end_dt:
            test_end = end_dt
        if test_start >= end_dt:
            break
        windows.append((window_start, test_start, test_end))
        window_start = test_start  # Roll forward by out_of_sample period
        if test_end >= end_dt:
            break

    print(f"[Backtest] {len(windows)} walk-forward windows generated.")

    # Run simulation across all windows and symbols
    all_trades: List[BacktestTrade] = []
    capital = initial_capital
    equity_history = [(start_dt.strftime("%Y-%m-%d"), capital)]

    for w_idx, (_, test_start, test_end) in enumerate(windows):
        print(f"[Backtest] Window {w_idx+1}/{len(windows)}: {test_start.strftime('%Y-%m-%d')} -> {test_end.strftime('%Y-%m-%d')}")

        window_trades = []
        for symbol in symbols:
            if symbol not in all_data:
                continue
            trades, _ = _simulate_symbol(
                symbol=symbol,
                df=all_data[symbol],
                test_start=test_start,
                test_end=test_end,
                strategy_type=strategy_type,
                initial_capital=capital / len(symbols),  # Allocate equally per symbol
            )
            window_trades.extend(trades)

        # Apply max open positions limit: sort by date, cap concurrent positions
        window_trades.sort(key=lambda t: t.entry_date)
        filtered_trades = []
        max_pos = 3  # Use 3 max concurrent positions per window

        for t in window_trades:
            # Simple concurrent position limit
            active = sum(1 for et in filtered_trades
                        if et.entry_date <= t.entry_date <= et.exit_date)
            if active < max_pos:
                filtered_trades.append(t)
                capital += t.net_pnl

        all_trades.extend(filtered_trades)
        equity_history.append((test_end.strftime("%Y-%m-%d"), capital))

        closed = len(filtered_trades)
        win = sum(1 for t in filtered_trades if t.net_pnl > 0)
        win_r = f"{win/closed*100:.0f}%" if closed > 0 else "N/A"
        print(f"  Trades: {closed} | Win rate: {win_r} | Capital: Rs{capital:,.0f}")

    # ── Statistics ─────────────────────────────────────────────────────────────

    if not all_trades:
        print("[Backtest] WARNING: No trades generated. Check conviction thresholds.")
        return BacktestResults(
            strategy_type=strategy_type, start_date=start_date, end_date=end_date,
            initial_capital=initial_capital, final_capital=initial_capital,
            total_return_pct=0, benchmark_return_pct=0, alpha_pct=0,
            sharpe_ratio=0, max_drawdown_pct=0, total_trades=0,
            win_rate_pct=0, avg_win_pct=0, avg_loss_pct=0, profit_factor=0,
            avg_holding_days=0, trades=[], equity_curve=[],
        )

    wins = [t for t in all_trades if t.net_pnl > 0]
    losses = [t for t in all_trades if t.net_pnl <= 0]
    win_rate = round(len(wins) / len(all_trades) * 100, 1)
    avg_win = round(sum(t.net_pnl_pct for t in wins) / len(wins), 2) if wins else 0
    avg_loss = round(sum(t.net_pnl_pct for t in losses) / len(losses), 2) if losses else 0
    gross_wins = sum(t.net_pnl for t in wins) if wins else 0
    gross_losses = abs(sum(t.net_pnl for t in losses)) if losses else 0
    profit_factor = round(gross_wins / gross_losses, 2) if gross_losses > 0 else 999.0
    avg_hold = round(sum(t.holding_days for t in all_trades) / len(all_trades), 1)
    total_return = round((capital - initial_capital) / initial_capital * 100, 2)

    # Equity curve for Sharpe
    equity_series = pd.Series(
        [e[1] for e in equity_history],
        index=pd.to_datetime([e[0] for e in equity_history])
    )
    daily_returns = equity_series.pct_change().dropna()
    sharpe = _compute_sharpe(daily_returns)
    max_dd = _compute_max_drawdown(equity_series)

    # Benchmark: Nifty 50 buy-and-hold
    benchmark_return = 0.0
    if "_NIFTY50" in all_data:
        nifty_df = all_data["_NIFTY50"]
        try:
            nifty_start_price = float(nifty_df.loc[nifty_df.index >= start_date].iloc[0]['Close'])
            nifty_end_price = float(nifty_df.iloc[-1]['Close'])
            benchmark_return = round((nifty_end_price - nifty_start_price) / nifty_start_price * 100, 2)
        except Exception:
            pass

    alpha = round(total_return - benchmark_return, 2)

    # ── Save results ────────────────────────────────────────────────────────────

    results = BacktestResults(
        strategy_type=strategy_type,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        final_capital=round(capital, 2),
        total_return_pct=total_return,
        benchmark_return_pct=benchmark_return,
        alpha_pct=alpha,
        sharpe_ratio=sharpe,
        max_drawdown_pct=max_dd,
        total_trades=len(all_trades),
        win_rate_pct=win_rate,
        avg_win_pct=avg_win,
        avg_loss_pct=avg_loss,
        profit_factor=profit_factor,
        avg_holding_days=avg_hold,
        trades=all_trades,
        equity_curve=[{"date": e[0], "capital": e[1]} for e in equity_history],
    )

    # Save trade log to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(output_dir, f"backtest_{strategy_type}_{timestamp}.csv")
    trades_df = pd.DataFrame([asdict(t) for t in all_trades])
    if not trades_df.empty:
        trades_df.to_csv(csv_path, index=False)
        print(f"\n[Backtest] Trade log saved to {csv_path}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"BACKTEST RESULTS SUMMARY ({strategy_type.upper()})")
    print(f"{'='*60}")
    print(f"Period:           {start_date} -> {end_date}")
    print(f"Total Trades:     {len(all_trades)}")
    print(f"Win Rate:         {win_rate}%")
    print(f"Avg Win:          {avg_win}%  |  Avg Loss: {avg_loss}%")
    print(f"Profit Factor:    {profit_factor}")
    print(f"Avg Hold:         {avg_hold} days")
    print(f"Total Return:     {total_return}%")
    print(f"Nifty 50 Return:  {benchmark_return}%")
    print(f"Alpha:            {alpha}%")
    print(f"Sharpe Ratio:     {sharpe}")
    print(f"Max Drawdown:     {max_dd}%")
    print(f"Final Capital:    Rs{capital:,.0f}")
    print(f"{'='*60}\n")

    # Deployment recommendation
    if sharpe >= 1.0 and len(all_trades) >= 50 and max_dd > -20:
        print("DEPLOYMENT READY: Sharpe >= 1.0, sufficient trade count, drawdown manageable.")
    elif sharpe >= 0.7 and len(all_trades) >= 30:
        print("CONDITIONAL: Run longer backtest and more symbols before deploying real money.")
    else:
        print("NOT READY: Sharpe < 0.7 or insufficient trades. Do not deploy with real money.")

    # Monte Carlo robustness test
    mc_results = run_monte_carlo(all_trades, initial_capital)
    if "error" not in mc_results:
        results.monte_carlo = mc_results

    # Walk-Forward Efficiency (simplified: compare first half IS vs second half OOS Sharpe)
    if len(all_trades) >= 20:
        mid = len(all_trades) // 2
        is_trades = all_trades[:mid]
        oos_trades = all_trades[mid:]
        is_pnls = pd.Series([t.net_pnl for t in is_trades])
        oos_pnls = pd.Series([t.net_pnl for t in oos_trades])
        is_rets = is_pnls / initial_capital
        oos_rets = oos_pnls / initial_capital
        is_sharpe_val = _compute_sharpe(is_rets)
        oos_sharpe_val = _compute_sharpe(oos_rets)
        wfe = compute_walk_forward_efficiency(is_sharpe_val, oos_sharpe_val)
        print(f"[Backtest] WFE: {wfe}%")

    return results


def run_full_backtest_suite(
    symbols: List[str] = None,
    start_date: str = "2019-01-01",
    end_date: str = None,
    initial_capital: float = 500000,
) -> Dict[str, BacktestResults]:
    """Run backtest for both swing and positional strategies. Returns dict of results."""
    if symbols is None:
        symbols = [
            "RELIANCE", "HDFCBANK", "INFY", "TCS", "ICICIBANK",
            "TATAMOTORS", "BAJFINANCE", "SUNPHARMA", "WIPRO", "AXISBANK",
            "MARUTI", "LTIM", "DRREDDY", "CIPLA", "SBIN",
            "TATASTEEL", "HINDALCO", "BHARTIARTL", "NESTLEIND", "HCLTECH",
        ]

    results = {}
    for strategy in ["swing", "positional"]:
        print(f"\n[BacktestSuite] Running {strategy} strategy...")
        try:
            results[strategy] = run_backtest(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                strategy_type=strategy,
                initial_capital=initial_capital,
            )
        except Exception as e:
            print(f"[BacktestSuite] {strategy} failed: {e}")

    return results


if __name__ == "__main__":
    results = run_full_backtest_suite(
        start_date="2020-01-01",
        end_date="2024-12-31",
        initial_capital=500000,
    )
