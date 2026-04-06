"""
🚀 ULTIMATE END-TO-END BACKTEST ENGINE (v3.0)
==============================================
1. Discovers candidates from NIFTY 100 (Simulated Pre-Scanner).
2. Analyzes candidates via AI CMT Analyst (Institutional Logic).
3. Evaluates long-term performance (30, 90, 180, 360 days).

Run with: python backtest.py
"""

import os
import base64
import json
import pandas as pd
import pandas_ta as ta
import matplotlib
matplotlib.use('Agg')
import mplfinance as mpf
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from agents.state import RiskDecision
from core.config import settings

# ─── CONFIG ───────────────────────────────────────────────────────────────────
# We simulate the Pre-Screener on this universe to find the daily candidates
UNIVERSE_NIFTY_100 = [
    "ABB", "ADANIENSOL", "ADANIENT", "ADANIGREEN", "ADANIPORTS", "ADANIPOWER", "ATGL", "AMBUJACEM", "APOLLOHOSP", "ASIANPAINT",
    "DMART", "AXISBANK", "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV", "BAJAJHLDNG", "BANKBARODA", "BEL", "BPCL", "BHARTIARTL",
    "BIOCON", "BOSCHLTD", "BRITANNIA", "CANBK", "CHOLAFIN", "CIPLA", "COALINDIA", "COFORGE", "COLPAL", "DLF",
    "EICHERMOT", "GAIL", "GICRE", "GODREJCP", "GRASIM", "HCLTECH", "HDFCBANK", "HDFCLIFE", "HEROMOTOCO", "HINDALCO",
    "HAL", "HINDUNILVR", "ICICIBANK", "ICICIGI", "ICICIPRULI", "ITC", "IOC", "IRCTC", "IRFC", "INDUSINDBK",
    "INFY", "JSWSTEEL", "JINDALSTEL", "JIOFIN", "KOTAKBANK", "LTIM", "LT", "LICI", "M&M", "MARICO",
    "MARUTI", "NTPC", "NESTLEIND", "ONGC", "PIDILITIND", "PFC", "POWERGRID", "PNB", "RELIANCE", "SBICARD",
    "SBILIFE", "SRF", "SHREECEM", "SHRIRAMFIN", "SIEMENS", "SBIN", "SUNPHARMA", "TATACOMM", "TATAELXSI", "TATACONSUM",
    "TATAMOTORS", "TATAPOWER", "TATASTEEL", "TCS", "TECHM", "TITAN", "TRENT", "TVSMOTOR", "UNITDSPR", "VBL",
    "VEDL", "WIPRO", "ZOMATO", "ZYDUSLIFE"
]

WINDOWS = [30, 90, 180, 360]
CHARTS_DIR = "./db/backtest_charts"
LOOKBACK_HISTORY = 700 
TOP_N_CANDIDATES = 5 # How many discovered stocks to analyze per window

os.makedirs(CHARTS_DIR, exist_ok=True)

# ─── ENGINE CORE ──────────────────────────────────────────────────────────────

def get_llm():
    if settings.OPENAI_API_KEY:
        return ChatOpenAI(model="gpt-4o", temperature=0, api_key=settings.OPENAI_API_KEY)
    return None

def encode_image(path: str) -> str:
    if not os.path.exists(path): return ""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def load_prompt(filename: str) -> tuple[str, str]:
    path = os.path.join(os.path.dirname(__file__), "prompts", filename)
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    parts = content.split("---")
    return parts[0].strip(), parts[1].strip()

# ─── DISCOVERY (SIMULATED PRE-SCANNER) ────────────────────────────────────────

def simulate_pre_scanner(cutoff_date: datetime) -> List[str]:
    """
    Simulates the ELITE v4.0 Discovery Logic:
    1. Volume > 500k
    2. Green Candle
    3. Price > 20-EMA (Trend Confirmation)
    4. Relative Volume > 1.2 (Volume Surge)
    """
    print(f"  [Scanner] Discovering Elite Candidates from Nifty 100 for {cutoff_date.strftime('%Y-%m-%d')}...")
    candidates = []
    
    # We fetch a 50-day window before cutoff to calculate EMA and Relative Volume
    start_search = cutoff_date - timedelta(days=60)
    
    for symbol in UNIVERSE_NIFTY_100:
        try:
            df = yf.Ticker(f"{symbol}.NS").history(start=start_search.strftime("%Y-%m-%d"), end=(cutoff_date + timedelta(days=1)).strftime("%Y-%m-%d"))
            if df.empty or len(df) < 25: continue
            
            # Historical context as of cutoff
            df = df[df.index <= cutoff_date.strftime("%Y-%m-%d")]
            if len(df) < 21: continue
            
            signal_day = df.iloc[-1]
            prev_day = df.iloc[-2]
            
            # 1. Bhavcopy Heuristics
            pct_change = ((signal_day['Close'] - prev_day['Close']) / prev_day['Close']) * 100
            volume = signal_day['Volume']
            is_green = signal_day['Close'] > signal_day['Open']
            
            # 2. Multi-Factor (RV & EMA)
            df.ta.ema(length=20, append=True)
            ema_20 = df['EMA_20'].iloc[-1]
            avg_vol_20 = df['Volume'].iloc[-21:-1].mean()
            rel_vol = volume / avg_vol_20 if avg_vol_20 > 0 else 0
            
            # FILTERS: Vol > 500k, Green, 0.5% < Change < 8%, Price > 20-EMA, RV > 1.2
            if (volume > 500000 and is_green and 0.5 < pct_change < 10.0 and signal_day['Close'] > ema_20 and rel_vol > 1.2):
                candidates.append({
                    "symbol": symbol,
                    "pct_change": pct_change,
                    "rel_vol": rel_vol
                })
        except: continue

    # Rank by Relative Volume (Institutional Footprint)
    sorted_candidates = sorted(candidates, key=lambda x: x['rel_vol'], reverse=True)
    discovery = [c['symbol'] for c in sorted_candidates[:TOP_N_CANDIDATES]]
    print(f"  [Scanner] Found {len(discovery)} Elite v4.0 Candidates: {discovery}")
    return discovery

# ─── ANALYSIS (CMT AGENT) ─────────────────────────────────────────────────────

def analyze_structural_snapshot(symbol: str, cutoff_date: datetime) -> Dict[str, Any]:
    """Mirror production MarketDataTool structural analysis."""
    start_history = cutoff_date - timedelta(days=LOOKBACK_HISTORY)
    df = yf.Ticker(f"{symbol}.NS").history(start=start_history.strftime("%Y-%m-%d"), end=(cutoff_date + timedelta(days=1)).strftime("%Y-%m-%d"))
    
    if df.empty or len(df) < 50: return {}
    
    # Trim to exactly cutoff
    df = df[df.index <= cutoff_date.strftime("%Y-%m-%d")]
    if df.empty: return {}

    # INDICATORS
    df.ta.atr(length=14, append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)

    # WEEKLY TREND
    df_weekly = df.resample('W').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'})
    df_weekly['sma20'] = df_weekly['Close'].rolling(window=20).mean()
    weekly_trend = "UP" if df_weekly['Close'].iloc[-1] > df_weekly['sma20'].iloc[-1] else "DOWN"

    # SUPPORT / RESISTANCE
    df['min_20'] = df['Low'].rolling(window=20, center=True).min()
    df['max_20'] = df['High'].rolling(window=20, center=True).max()
    potential_supp = df[df['Low'] == df['min_20']]['Low'].unique()
    potential_res = df[df['High'] == df['max_20']]['High'].unique()
    current_price = df['Close'].iloc[-1]
    support_levels = sorted([round(x, 2) for x in potential_supp if 0.8 * current_price <= x < current_price], reverse=True)[:3]
    resistance_levels = sorted([round(x, 2) for x in potential_res if current_price < x <= 1.2 * current_price])[:3]

    # RELATIVE STRENGTH (vs NIFTY)
    try:
        nifty_df = yf.Ticker("^NSEI").history(start=(cutoff_date - timedelta(days=60)).strftime("%Y-%m-%d"), end=(cutoff_date + timedelta(days=1)).strftime("%Y-%m-%d"))
        stock_perf = (df['Close'].iloc[-1] - df['Close'].iloc[-30]) / df['Close'].iloc[-30]
        nifty_perf = (nifty_df['Close'].iloc[-1] - nifty_df['Close'].iloc[-30]) / nifty_df['Close'].iloc[-30]
        rs_score = round(stock_perf - nifty_perf, 4)
    except: rs_score = 0.0

    # SEQUENTIAL DATA TABLE (14 Days)
    recent_df = df.tail(14)
    candles_table = "| Day | Open | High | Low | Close | Vol |\n| :--- | :--- | :--- | :--- | :--- | :--- |\n"
    for idx, row in recent_df.iterrows():
        candles_table += f"| {idx.strftime('%Y-%m-%d')} | {round(row['Open'], 2)} | {round(row['High'], 2)} | {round(row['Low'], 2)} | {round(row['Close'], 2)} | {int(row['Volume'])} |\n"

    # CHART
    chart_path = os.path.abspath(f"{CHARTS_DIR}/{symbol}_{cutoff_date.strftime('%Y%m%d')}_chart.png")
    plot_df = df.tail(90)
    macd_col = [col for col in df.columns if col.startswith('MACD_')][0]
    macds_col = [col for col in df.columns if col.startswith('MACDs_')][0]
    apdict = [mpf.make_addplot(plot_df[macd_col], panel=1, color='fuchsia', ylabel='MACD'), mpf.make_addplot(plot_df[macds_col], panel=1, color='b')]
    mpf.plot(plot_df, type='candle', volume=True, style='charles', title=f"{symbol} - T-{cutoff_date.strftime('%Y-%m-%d')}", addplot=apdict, savefig=dict(fname=chart_path, dpi=100, bbox_inches='tight'))

    latest = df.iloc[-1]
    atr_col = [col for col in df.columns if col.startswith('ATRr_')][0]
    rsi_col = [col for col in df.columns if col.startswith('RSI_')][0]

    return {
        "symbol": symbol,
        "cutoff_price": round(latest['Close'], 2),
        "weekly_trend": weekly_trend,
        "rs_score": rs_score,
        "supp_levels": support_levels,
        "res_levels": resistance_levels,
        "atr": round(latest[atr_col], 2),
        "rsi": round(latest[rsi_col], 2),
        "macd_hist": round(latest[macd_col] - latest[macds_col], 2),
        "recent_candles_table": candles_table,
        "chart_path": chart_path
    }

def get_ai_decision(snapshot: dict, snapshot_date: str) -> RiskDecision | None:
    llm = get_llm()
    if not llm: return None
    try:
        structured_llm = llm.with_structured_output(RiskDecision)
        base64_image = encode_image(snapshot['chart_path'])
        sys_template, user_template = load_prompt("backtester.txt")
        
        user_prompt = user_template.format(
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
            recent_candles_table=snapshot['recent_candles_table']
        )
        
        decision = structured_llm.invoke([
            SystemMessage(content=sys_template),
            HumanMessage(content=[
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
            ])
        ])
        return decision
    except Exception as e:
        print(f"  [Error] LLM Exception for {snapshot['symbol']}: {e}")
        return None

def evaluate_outcome(decision: RiskDecision, symbol: str, cutoff_date: datetime) -> dict:
    if decision.proposed_action != "BUY": return {"result": "SKIPPED", "pnl": 0}
    
    # Exit criteria: SL, TP, or 30 days
    now = datetime.now()
    holding_days = decision.expected_holding_days if decision.expected_holding_days > 0 else 30
    end_date = min(cutoff_date + timedelta(days=holding_days + 10), now)
    
    outcome_df = yf.Ticker(f"{symbol}.NS").history(start=cutoff_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))
    if outcome_df.empty: return {"result": "NO_DATA", "pnl": 0}
    
    # We remove the first candle matching the entry to simulate 'entry at close or next open'
    outcome_df = outcome_df[outcome_df.index > cutoff_date.strftime("%Y-%m-%d")]
    if outcome_df.empty: return {"result": "NO_DATA", "pnl": 0}

    entry = decision.proposed_entry
    sl = decision.proposed_stop_loss
    tp = decision.proposed_take_profit
    
    exit_price = outcome_df['Close'].iloc[-1]
    hit_sl = False
    hit_tp = False
    
    for idx, row in outcome_df.iterrows():
        if row['Low'] <= sl:
            hit_sl = True; exit_price = sl; break
        if row['High'] >= tp:
            hit_tp = True; exit_price = tp; break
            
    pnl = round(((exit_price - entry) / entry) * 100, 2)
    return {"result": "HIT_TP" if hit_tp else ("HIT_SL" if hit_sl else "FORCE_EXIT"), "pnl": pnl}

# ─── RUNNER ───────────────────────────────────────────────────────────────────

def is_trading_day(d: datetime) -> bool:
    return d.weekday() < 5 # Basic Mon-Fri check

def run_performance_audit():
    print("=" * 75)
    print("🚀 END-TO-END DISCOVERY & ANALYSIS AUDIT (v3.0)")
    print("=" * 75)
    
    summary_results = []
    for window in WINDOWS:
        # Align cutoff_date to the nearest previous trading day
        cutoff_date = datetime.now() - timedelta(days=window)
        while not is_trading_day(cutoff_date):
            cutoff_date -= timedelta(days=1)
            
        print(f"\n📅 TESTING HORIZON: T-{window} days (Simulated Date: {cutoff_date.strftime('%Y-%m-%d')})")
        
        # 1. DISCOVERY (Simulated Screener)
        candidates = simulate_pre_scanner(cutoff_date)
        
        # 2. ANALYSIS
        for symbol in candidates:
            print(f"  🧠 AI Analyzing {symbol}...", end="", flush=True)
            snapshot = analyze_structural_snapshot(symbol, cutoff_date)
            if not snapshot: print(" SKIP (No snapshot)"); continue
            
            decision = get_ai_decision(snapshot, cutoff_date.strftime('%Y-%m-%d'))
            if not decision: print(" ERROR (No decision)"); continue
            
            # Map action properly
            action = decision.proposed_action if decision.proposed_action else "HOLD"
            
            if action == "BUY":
                outcome = evaluate_outcome(decision, symbol, cutoff_date)
                print(f" {outcome['result']} ({outcome['pnl']}%)")
                summary_results.append({
                    "horizon": window, "symbol": symbol, "action": "BUY",
                    "result": outcome['result'], "pnl": outcome['pnl'], "conviction": decision.conviction_tier
                })
            else:
                print(f" {action}")
                summary_results.append({
                    "horizon": window, "symbol": symbol, "action": action,
                    "result": "N/A", "pnl": 0, "conviction": decision.conviction_tier
                })

    # FINAL HARVEST
    if not summary_results:
        print("\n[Audit] No signals were generated across any horizons. Ending.")
        return

    df = pd.DataFrame(summary_results)
    df.to_csv("./db/end_to_end_backtest_report.csv", index=False)
    
    print("\n" + "=" * 75)
    print("🏁 FINAL PIPELINE SUMMARY")
    print("=" * 75)
    
    if "action" in df.columns:
        buys = df[df['action'] == "BUY"]
        if not buys.empty:
            win_rate = (len(buys[buys['pnl'] > 0]) / len(buys)) * 100
            avg_pnl = buys['pnl'].mean()
            print(f"Total Candidate Screened: {len(WINDOWS) * len(UNIVERSE_NIFTY_100)}")
            print(f"Total Actions Proposed  : {len(df)}")
            print(f"Total BUY Signals       : {len(buys)}")
            print(f"Aggregate Win Rate      : {win_rate:.1f}%")
            print(f"Avg PnL per Trade       : {avg_pnl:.2f}%")
            
            winners_pnl = buys[buys['pnl'] > 0]['pnl'].sum()
            losers_pnl = abs(buys[buys['pnl'] < 0]['pnl'].sum())
            profit_factor = (winners_pnl / losers_pnl) if losers_pnl > 0 else float('inf')
            print(f"Profit Factor           : {profit_factor:.2f}")
        else:
            print("No BUY signals were generated by the AI in this universe/window.")
    
    print("=" * 75)
    print("✅ Full Pipeline Report: ./db/end_to_end_backtest_report.csv")

if __name__ == "__main__":
    run_performance_audit()
