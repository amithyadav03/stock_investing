"""
End-to-End Backtest Engine v4.0
================================
1. Simulates the Nifty 500 pre-screener on historical Nifty 100 data (liquid, yfinance-covered).
2. Runs Claude AI analysis (claude-sonnet-4-6) on historical chart snapshots.
3. Evaluates outcomes: SL hit, TP hit, or time-based exit.
4. Reports: win rate, avg P&L, profit factor per conviction tier.

Run: python backtest.py
"""

import os
import sys
import base64
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import mplfinance as mpf
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from agents.state import RiskDecision
from core.config import settings
from core.claude_client import get_client, call_structured, call_text
from data.nifty500_symbols import NIFTY_100
from tools.indicators import add_all_indicators, ema as calc_ema

CHARTS_DIR = "./db/backtest_charts"
LOOKBACK_HISTORY = 700
TOP_N_CANDIDATES = 5
WINDOWS = [30, 90, 180, 360]

os.makedirs(CHARTS_DIR, exist_ok=True)


def encode_image(path: str) -> Optional[str]:
    if not path or not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def load_prompt(filename: str) -> tuple[str, str]:
    path = os.path.join(os.path.dirname(__file__), "prompts", filename)
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    parts = content.split("---", 1)
    return parts[0].strip(), parts[1].strip() if len(parts) > 1 else ("", parts[0].strip())


# ── Discovery (Simulated Pre-Screener) ──────────────────────────────────────────

def simulate_pre_scanner(cutoff_date: datetime) -> List[str]:
    """
    Simulates the 3-stage discovery logic on historical Nifty 100 data.
    Filters: Volume>500k, green candle, 0.5–10% change, price > EMA20, RV > 1.2.
    """
    print(f"  [Scanner] Discovering candidates from Nifty 100 @ {cutoff_date.strftime('%Y-%m-%d')}...")
    start = cutoff_date - timedelta(days=60)
    candidates = []

    for symbol in NIFTY_100:
        try:
            df = yf.Ticker(f"{symbol}.NS").history(
                start=start.strftime("%Y-%m-%d"),
                end=(cutoff_date + timedelta(days=1)).strftime("%Y-%m-%d"),
            )
            if df.empty or len(df) < 25:
                continue
            df = df[df.index <= pd.Timestamp(cutoff_date)]
            if len(df) < 21:
                continue

            signal = df.iloc[-1]
            prev = df.iloc[-2]
            pct = ((signal['Close'] - prev['Close']) / prev['Close']) * 100
            volume = signal['Volume']
            is_green = signal['Close'] > signal['Open']

            df['EMA_20'] = calc_ema(df, 20)
            ema20 = float(df['EMA_20'].iloc[-1]) if not pd.isna(df['EMA_20'].iloc[-1]) else None
            avg_vol = df['Volume'].iloc[-21:-1].mean()
            rv = volume / avg_vol if avg_vol > 0 else 0

            if (volume > 500_000 and is_green and 0.5 < pct < 10.0
                    and ema20 and signal['Close'] > ema20 and rv > 1.2):
                candidates.append({"symbol": symbol, "rv": rv, "pct": pct})
        except Exception:
            continue

    candidates.sort(key=lambda x: x['rv'], reverse=True)
    result = [c['symbol'] for c in candidates[:TOP_N_CANDIDATES]]
    print(f"  [Scanner] Found {len(result)} elite candidates: {result}")
    return result


# ── Structural Snapshot ──────────────────────────────────────────────────────────

def analyze_structural_snapshot(symbol: str, cutoff_date: datetime) -> Dict[str, Any]:
    """Mirrors production MarketDataTool analysis as of cutoff_date."""
    start = cutoff_date - timedelta(days=LOOKBACK_HISTORY)
    df = yf.Ticker(f"{symbol}.NS").history(
        start=start.strftime("%Y-%m-%d"),
        end=(cutoff_date + timedelta(days=1)).strftime("%Y-%m-%d"),
    )
    if df.empty or len(df) < 50:
        return {}

    df = df[df.index <= pd.Timestamp(cutoff_date)]
    if df.empty:
        return {}

    # Indicators
    df = add_all_indicators(df)

    # Weekly trend
    df_w = df.resample('W').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'})
    df_w['sma20'] = df_w['Close'].rolling(20).mean()
    weekly_trend = "UP" if df_w['Close'].iloc[-1] > df_w['sma20'].iloc[-1] else "DOWN"

    # S/R
    df['min_20'] = df['Low'].rolling(20, center=True).min()
    df['max_20'] = df['High'].rolling(20, center=True).max()
    price = df['Close'].iloc[-1]
    supp = sorted([round(x, 2) for x in df[df['Low'] == df['min_20']]['Low'].unique()
                   if 0.8 * price <= x < price], reverse=True)[:3]
    res = sorted([round(x, 2) for x in df[df['High'] == df['max_20']]['High'].unique()
                  if price < x <= 1.2 * price])[:3]

    # RS vs NIFTY
    try:
        nifty = yf.Ticker("^NSEI").history(
            start=(cutoff_date - timedelta(days=60)).strftime("%Y-%m-%d"),
            end=(cutoff_date + timedelta(days=1)).strftime("%Y-%m-%d"),
        )
        rs_score = round(
            (df['Close'].iloc[-1] - df['Close'].iloc[-30]) / df['Close'].iloc[-30]
            - (nifty['Close'].iloc[-1] - nifty['Close'].iloc[-30]) / nifty['Close'].iloc[-30],
            4,
        )
    except Exception:
        rs_score = 0.0

    # Candle table
    recent = df.tail(14)
    candles_table = "| Day | Open | High | Low | Close | Vol |\n| :--- | :--- | :--- | :--- | :--- | :--- |\n"
    for idx, row in recent.iterrows():
        candles_table += f"| {idx.strftime('%Y-%m-%d')} | {round(row['Open'],2)} | {round(row['High'],2)} | {round(row['Low'],2)} | {round(row['Close'],2)} | {int(row['Volume'])} |\n"

    # Chart
    chart_path = os.path.abspath(f"{CHARTS_DIR}/{symbol}_{cutoff_date.strftime('%Y%m%d')}.png")
    try:
        plot_df = df.tail(90)
        macd_col = next((c for c in df.columns if c.startswith('MACD_12')), None)
        macds_col = next((c for c in df.columns if c.startswith('MACDs_12')), None)
        apdict = []
        if macd_col and macds_col:
            apdict = [
                mpf.make_addplot(plot_df[macd_col], panel=1, color='fuchsia', ylabel='MACD'),
                mpf.make_addplot(plot_df[macds_col], panel=1, color='b'),
            ]
        mpf.plot(plot_df, type='candle', volume=True, style='charles',
                 title=f"{symbol} T-{cutoff_date.strftime('%Y-%m-%d')}",
                 addplot=apdict,
                 savefig=dict(fname=chart_path, dpi=100, bbox_inches='tight'))
    except Exception as e:
        print(f"  [Chart] Render failed for {symbol}: {e}")
        chart_path = None

    latest = df.iloc[-1]

    def _g(prefix, default=0.0):
        col = next((c for c in df.columns if c.startswith(prefix)), None)
        return round(float(latest[col]), 4) if col else default

    return {
        "symbol": symbol,
        "cutoff_price": round(float(price), 2),
        "weekly_trend": weekly_trend,
        "rs_score": rs_score,
        "supp_levels": supp,
        "res_levels": res,
        "atr": _g('ATRr_14'),
        "rsi": _g('RSI_14'),
        "macd_hist": _g('MACD_12') - _g('MACDs_12'),
        "adx": _g('ADX_14'),
        "ema_20": _g('EMA_20'),
        "ema_50": _g('EMA_50'),
        "recent_candles_table": candles_table,
        "chart_path": chart_path,
    }


# ── AI Decision ──────────────────────────────────────────────────────────────────

def get_ai_decision(snapshot: dict, snapshot_date: str) -> Optional[RiskDecision]:
    client = get_client()
    if not client:
        print("  [AI] No ANTHROPIC_API_KEY. Skipping AI analysis.")
        return None

    try:
        sys_prompt, user_template = load_prompt("backtester.txt")
        user_text = user_template.format(
            snapshot_date=snapshot_date,
            symbol=snapshot['symbol'],
            cutoff_price=snapshot['cutoff_price'],
            weekly_trend=snapshot['weekly_trend'],
            rs_score=snapshot['rs_score'],
            res_levels=snapshot['res_levels'],
            supp_levels=snapshot['supp_levels'],
            rsi=snapshot['rsi'],
            atr=snapshot['atr'],
            macd_hist=snapshot['macd_hist'],
            adx=snapshot.get('adx', 'N/A'),
            ema_20=snapshot.get('ema_20', 'N/A'),
            recent_candles_table=snapshot['recent_candles_table'],
        )
        image_b64 = encode_image(snapshot.get('chart_path'))

        result = call_structured(
            client=client,
            system_prompt=sys_prompt,
            user_text=user_text,
            tool_name="submit_backtest_decision",
            tool_description="Submit the historical trade decision for backtesting",
            tool_schema={
                "type": "object",
                "properties": {
                    "chain_of_thought_1_technicals": {"type": "string"},
                    "chain_of_thought_2_fundamentals": {"type": "string"},
                    "chain_of_thought_3_risk": {"type": "string"},
                    "proposed_action": {"type": "string", "enum": ["BUY", "SELL", "HOLD"]},
                    "proposed_entry": {"type": "number"},
                    "proposed_stop_loss": {"type": "number"},
                    "proposed_take_profit": {"type": "number"},
                    "conviction_tier": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]},
                    "win_probability_score": {"type": "integer", "minimum": 1, "maximum": 100},
                    "risk_percentage": {"type": "number"},
                    "expected_holding_days": {"type": "integer"},
                    "final_rationale": {"type": "string"},
                },
                "required": [
                    "chain_of_thought_1_technicals", "chain_of_thought_2_fundamentals",
                    "chain_of_thought_3_risk", "proposed_action", "proposed_entry",
                    "proposed_stop_loss", "proposed_take_profit", "conviction_tier",
                    "win_probability_score", "risk_percentage", "expected_holding_days",
                    "final_rationale",
                ],
            },
            image_base64=image_b64,
        )
        if result:
            return RiskDecision(**result)
    except Exception as e:
        print(f"  [AI] Decision failed for {snapshot['symbol']}: {e}")
    return None


# ── Outcome Evaluation ─────────────────────────────────────────────────────────

def evaluate_outcome(decision: RiskDecision, symbol: str, cutoff_date: datetime) -> dict:
    if decision.proposed_action != "BUY":
        return {"result": "SKIPPED", "pnl": 0.0, "days": 0}

    holding_days = max(decision.expected_holding_days, 5)
    end_date = min(cutoff_date + timedelta(days=holding_days + 10), datetime.now())

    outcome_df = yf.Ticker(f"{symbol}.NS").history(
        start=cutoff_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"),
    )
    if outcome_df.empty:
        return {"result": "NO_DATA", "pnl": 0.0, "days": 0}

    outcome_df = outcome_df[outcome_df.index > pd.Timestamp(cutoff_date)]
    if outcome_df.empty:
        return {"result": "NO_DATA", "pnl": 0.0, "days": 0}

    entry = decision.proposed_entry
    sl = decision.proposed_stop_loss
    tp = decision.proposed_take_profit
    exit_price = float(outcome_df['Close'].iloc[-1])
    hit_sl, hit_tp = False, False
    days_taken = len(outcome_df)

    for i, (_, row) in enumerate(outcome_df.iterrows()):
        if row['Low'] <= sl:
            hit_sl, exit_price, days_taken = True, sl, i + 1; break
        if row['High'] >= tp:
            hit_tp, exit_price, days_taken = True, tp, i + 1; break

    pnl = round(((exit_price - entry) / entry) * 100, 2) if entry > 0 else 0.0
    result = "HIT_TP" if hit_tp else ("HIT_SL" if hit_sl else "FORCE_EXIT")
    return {"result": result, "pnl": pnl, "days": days_taken}


# ── Main Runner ────────────────────────────────────────────────────────────────

def is_trading_day(d: datetime) -> bool:
    return d.weekday() < 5


def run_performance_audit():
    print("=" * 70)
    print("END-TO-END BACKTEST v4.0 — Claude AI + Nifty 100 Universe")
    print("=" * 70)

    if not settings.ANTHROPIC_API_KEY:
        print("WARNING: ANTHROPIC_API_KEY not set. AI decisions will be skipped.")

    results = []
    for window in WINDOWS:
        cutoff = datetime.now() - timedelta(days=window)
        while not is_trading_day(cutoff):
            cutoff -= timedelta(days=1)

        print(f"\n📅 T-{window} ({cutoff.strftime('%Y-%m-%d')})")
        candidates = simulate_pre_scanner(cutoff)

        for symbol in candidates:
            print(f"  Analyzing {symbol}...", end="", flush=True)
            snapshot = analyze_structural_snapshot(symbol, cutoff)
            if not snapshot:
                print(" SKIP (no data)")
                continue

            decision = get_ai_decision(snapshot, cutoff.strftime('%Y-%m-%d'))
            if not decision:
                print(" SKIP (no AI decision)")
                continue

            action = decision.proposed_action
            if action == "BUY":
                outcome = evaluate_outcome(decision, symbol, cutoff)
                print(f" {outcome['result']} ({outcome['pnl']:+.2f}%, {outcome['days']}d)")
                results.append({
                    "horizon": window, "symbol": symbol, "action": "BUY",
                    "conviction": decision.conviction_tier,
                    "win_prob": decision.win_probability_score,
                    "result": outcome['result'], "pnl": outcome['pnl'], "days": outcome['days'],
                })
            else:
                print(f" {action}")
                results.append({
                    "horizon": window, "symbol": symbol, "action": action,
                    "conviction": decision.conviction_tier,
                    "win_prob": decision.win_probability_score,
                    "result": "N/A", "pnl": 0.0, "days": 0,
                })

    if not results:
        print("\nNo signals generated. Ending.")
        return

    df = pd.DataFrame(results)
    df.to_csv("./db/backtest_report.csv", index=False)

    print("\n" + "=" * 70)
    print("BACKTEST SUMMARY")
    print("=" * 70)
    buys = df[df['action'] == "BUY"]
    if not buys.empty:
        win_rate = len(buys[buys['pnl'] > 0]) / len(buys) * 100
        avg_pnl = buys['pnl'].mean()
        winners_sum = buys[buys['pnl'] > 0]['pnl'].sum()
        losers_sum = abs(buys[buys['pnl'] < 0]['pnl'].sum())
        pf = winners_sum / losers_sum if losers_sum > 0 else float('inf')

        print(f"Universe screened       : Nifty 100 ({len(NIFTY_100)} stocks)")
        print(f"Total BUY signals       : {len(buys)}")
        print(f"Win Rate                : {win_rate:.1f}%")
        print(f"Avg P&L per trade       : {avg_pnl:+.2f}%")
        print(f"Profit Factor           : {pf:.2f}")
        print(f"Avg Holding Days        : {buys['days'].mean():.1f}")

        print("\nBy Conviction Tier:")
        for tier in ["HIGH", "MEDIUM", "LOW"]:
            t = buys[buys['conviction'] == tier]
            if not t.empty:
                wr = len(t[t['pnl'] > 0]) / len(t) * 100
                print(f"  {tier:6s}: {len(t):3d} trades | WR={wr:.0f}% | Avg={t['pnl'].mean():+.2f}%")
    else:
        print("No BUY signals generated.")

    print("=" * 70)
    print(f"Report: ./db/backtest_report.csv")


if __name__ == "__main__":
    run_performance_audit()
