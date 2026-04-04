"""
BACKTESTING ENGINE
==================
Simulates what the AI would have decided 30 days ago,
then evaluates if the trade would have been profitable.

Run with: python backtest.py
"""

import os
import base64
import pandas as pd
import pandas_ta as ta
import matplotlib
matplotlib.use('Agg')
import mplfinance as mpf
import yfinance as yf
from datetime import datetime, timedelta
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from agents.state import RiskDecision
from core.config import settings

# ─── CONFIG ───────────────────────────────────────────────────────────────────

BACKTEST_STOCKS = [
    "RELIANCE", "ZOMATO", "IREDA", "TATASTEEL", "SUZLON",
    "HDFCBANK", "INFY", "PAYTM", "TRENT", "VEDL",
    "WIPRO", "BAJFINANCE", "NTPC", "ADANIENT", "SBIN"
]

LOOKBACK_DAYS = 180         # Simulate as if we are 6 months in the past
HISTORY_DAYS = 365          # 1 year of history fed to the AI for context
CHARTS_DIR = "./db/backtest_charts"

# ─── HELPERS ──────────────────────────────────────────────────────────────────

os.makedirs(CHARTS_DIR, exist_ok=True)

def get_llm():
    if settings.OPENAI_API_KEY:
        return ChatOpenAI(model="gpt-4o", temperature=0, api_key=settings.OPENAI_API_KEY)
    return None

def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def fetch_historical_slice(symbol: str, end_date: datetime, history_days: int) -> pd.DataFrame:
    """Fetch OHLCV data from (end_date - history_days) to end_date."""
    start_date = end_date - timedelta(days=history_days)
    ticker = yf.Ticker(f"{symbol}.NS")
    df = ticker.history(start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))
    return df

def fetch_outcome_data(symbol: str, from_date: datetime, to_date: datetime) -> pd.DataFrame:
    """Fetch the ACTUAL price data AFTER the signal was generated — used to evaluate the trade."""
    ticker = yf.Ticker(f"{symbol}.NS")
    df = ticker.history(start=from_date.strftime("%Y-%m-%d"), end=to_date.strftime("%Y-%m-%d"))
    return df

def compute_technicals(df: pd.DataFrame) -> dict:
    """Run pandas-ta math on the historical slice & render a chart."""
    if df.empty or len(df) < 30:
        return {}
    df.ta.atr(length=14, append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.bbands(length=20, std=2, append=True)
    return df

def render_chart(df: pd.DataFrame, symbol: str, label: str) -> str:
    """Render a candlestick chart and return its path."""
    chart_path = os.path.abspath(f"{CHARTS_DIR}/{symbol}_{label}_chart.png")
    plot_df = df.tail(60)
    
    macd_cols = [c for c in df.columns if c.startswith("MACD_")]
    macds_cols = [c for c in df.columns if c.startswith("MACDs_")]
    
    if not macd_cols or not macds_cols:
        mpf.plot(plot_df, type="candle", volume=True, style="charles",
                 title=f"{symbol} [{label}]",
                 savefig=dict(fname=chart_path, dpi=100, bbox_inches="tight"))
    else:
        apdict = [
            mpf.make_addplot(plot_df[macd_cols[0]], panel=1, color="fuchsia", ylabel="MACD"),
            mpf.make_addplot(plot_df[macds_cols[0]], panel=1, color="b"),
        ]
        mpf.plot(plot_df, type="candle", volume=True, style="charles",
                 title=f"{symbol} [{label}]",
                 addplot=apdict,
                 savefig=dict(fname=chart_path, dpi=100, bbox_inches="tight"))
    return chart_path

def load_prompt(filename: str) -> tuple[str, str]:
    """Loads a prompt file and splits it into system and user parts by '---'"""
    path = os.path.join(os.path.dirname(__file__), "prompts", filename)
    with open(path, "r") as f:
        content = f.read()
    parts = content.split("---")
    return parts[0].strip(), parts[1].strip()

def get_ai_decision(symbol: str, technicals: dict, chart_path: str, cutoff_price: float) -> RiskDecision | None:
    """Ask the LLM for a trade decision based on the historical snapshot."""
    llm = get_llm()
    if not llm:
        print(f"  [MOCK] No OpenAI key — generating mock decision for {symbol}")
        atr = technicals.get("atr", 2.0)
        return RiskDecision(
            chain_of_thought_1_technicals="Mock: RSI neutral, price in mid-band",
            chain_of_thought_2_fundamentals="Mock: Fair valuation",
            chain_of_thought_3_risk="Mock: ATR based SL applied",
            proposed_action="BUY",
            proposed_entry=cutoff_price,
            proposed_stop_loss=round(cutoff_price - (atr * 2), 2),
            proposed_take_profit=round(cutoff_price * 1.06, 2),
            risk_percentage=0.05,
            expected_holding_days=14,
            final_rationale="Mock backtest decision"
        )
    
    try:
        structured_llm = llm.with_structured_output(RiskDecision)
        base64_image = encode_image(chart_path)
        
        sys_template, user_template = load_prompt("backtester.txt")
        
        system_prompt = sys_template
        user_prompt = user_template.format(
            symbol=symbol,
            snapshot_date=(datetime.now() - timedelta(days=LOOKBACK_DAYS)).strftime('%Y-%m-%d'),
            cutoff_price=cutoff_price,
            atr=technicals.get('atr', 'N/A'),
            rsi=technicals.get('rsi', 'N/A'),
            macd_hist=technicals.get('macd_hist', 'N/A')
        )
        
        decision = structured_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=[
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
            ])
        ])
        return decision
    except Exception as e:
        print(f"  [LLM Error] {symbol}: {e}")
        return None

def evaluate_outcome(decision: RiskDecision, outcome_df: pd.DataFrame) -> dict:
    """
    Measure whether the trade hit Take Profit or Stop Loss over the next 30 days.
    Returns a results dict.
    """
    if outcome_df.empty or decision.proposed_action != "BUY":
        return {"result": "NO_DATA_OR_NOT_BUY", "pnl_pct": 0}

    entry = decision.proposed_entry
    sl = decision.proposed_stop_loss
    tp = decision.proposed_take_profit

    hit_tp = False
    hit_sl = False
    exit_price = outcome_df["Close"].iloc[-1]  # Default: close at end of window

    for _, row in outcome_df.iterrows():
        if row["Low"] <= sl:
            hit_sl = True
            exit_price = sl
            break
        if row["High"] >= tp:
            hit_tp = True
            exit_price = tp
            break

    pnl_pct = round(((exit_price - entry) / entry) * 100, 2)

    return {
        "result": "HIT_TP ✅" if hit_tp else ("HIT_SL ❌" if hit_sl else "OPEN_AT_EXPIRY 🔵"),
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "exit_price": exit_price,
        "pnl_pct": pnl_pct,
    }

# ─── MAIN BACKTEST LOOP ────────────────────────────────────────────────────────

def run_backtest():
    cutoff_date = datetime.now() - timedelta(days=LOOKBACK_DAYS)
    today = datetime.now()

    print("=" * 65)
    print(f"  AI SWING TRADE BACKTEST — Simulating from {cutoff_date.strftime('%Y-%m-%d')}")
    print(f"  Evaluating outcomes up to   {today.strftime('%Y-%m-%d')}")
    print("=" * 65)

    results = []

    for symbol in BACKTEST_STOCKS:
        print(f"\n📊 Analyzing {symbol}...")

        # Step 1: Fetch historical data as of cutoff
        hist_df = fetch_historical_slice(symbol, end_date=cutoff_date, history_days=HISTORY_DAYS)
        if hist_df.empty or len(hist_df) < 30:
            print(f"  ⚠️  Not enough data for {symbol}. Skipping.")
            continue

        # Step 2: Compute python math indicators
        hist_df = compute_technicals(hist_df)
        latest = hist_df.iloc[-1]
        cutoff_price = round(latest["Close"], 2)

        atr_col = next((c for c in hist_df.columns if c.startswith("ATRr_")), None)
        rsi_col = next((c for c in hist_df.columns if c.startswith("RSI_")), None)
        macd_col = next((c for c in hist_df.columns if c.startswith("MACD_")), None)
        macds_col = next((c for c in hist_df.columns if c.startswith("MACDs_")), None)

        technicals_snapshot = {
            "atr": round(latest[atr_col], 2) if atr_col else None,
            "rsi": round(latest[rsi_col], 2) if rsi_col else None,
            "macd_hist": round(latest[macd_col] - latest[macds_col], 2) if macd_col and macds_col else None,
        }

        # Step 3: Render chart as of cutoff
        chart_path = render_chart(hist_df, symbol, label="30d_ago")

        # Step 4: Ask LLM for decision based on historical snapshot
        print(f"  Price on {cutoff_date.strftime('%Y-%m-%d')}: ₹{cutoff_price} | ATR: {technicals_snapshot['atr']} | RSI: {technicals_snapshot['rsi']}")
        decision = get_ai_decision(symbol, technicals_snapshot, chart_path, cutoff_price)

        if not decision:
            print(f"  ⚠️  No decision generated.")
            continue

        print(f"  🤖 AI Decision: {decision.proposed_action} | Entry: {decision.proposed_entry} | SL: {decision.proposed_stop_loss} | TP: {decision.proposed_take_profit}")

        # Step 5: Fetch ACTUAL outcome data using the AI's OWN expected timeframe
        outcome_window_days = decision.expected_holding_days if decision.expected_holding_days > 0 else LOOKBACK_DAYS
        outcome_end = cutoff_date + timedelta(days=outcome_window_days)
        # Cap at today if the AI's window extends beyond today
        outcome_end = min(outcome_end, today)
        outcome_df = fetch_outcome_data(symbol, from_date=cutoff_date, to_date=outcome_end)

        # Step 6: Evaluate
        outcome = evaluate_outcome(decision, outcome_df)
        result_row = {
            "Symbol": symbol,
            "AI Decision": decision.proposed_action,
            "Timeframe (days)": decision.expected_holding_days,
            "Entry Price": decision.proposed_entry,
            "Stop Loss": decision.proposed_stop_loss,
            "Take Profit": decision.proposed_take_profit,
            "Outcome": outcome["result"],
            "Exit Price": outcome.get("exit_price", "-"),
            "PnL %": outcome.get("pnl_pct", 0),
        }
        results.append(result_row)
        print(f"  📈 Outcome: {outcome['result']} | Exit: {outcome.get('exit_price')} | PnL: {outcome.get('pnl_pct')}%")

    # ─── SUMMARY ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  BACKTEST SUMMARY")
    print("=" * 65)

    if not results:
        print("No results to summarize.")
        return

    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))

    # Aggregate stats
    buy_trades = df_results[df_results["AI Decision"] == "BUY"]
    if not buy_trades.empty:
        wins = buy_trades[buy_trades["Outcome"].str.contains("HIT_TP")]
        losses = buy_trades[buy_trades["Outcome"].str.contains("HIT_SL")]
        avg_pnl = buy_trades["PnL %"].mean()
        win_rate = len(wins) / len(buy_trades) * 100

        print(f"\n  Total BUY signals : {len(buy_trades)}")
        print(f"  Winners (Hit TP)  : {len(wins)}")
        print(f"  Losers  (Hit SL)  : {len(losses)}")
        print(f"  Win Rate          : {win_rate:.1f}%")
        print(f"  Avg PnL per trade : {avg_pnl:.2f}%")

    # Save to CSV for further analysis
    df_results.to_csv("./db/backtest_results.csv", index=False)
    print("\n  ✅ Full results saved to ./db/backtest_results.csv")
    print("=" * 65)

if __name__ == "__main__":
    run_backtest()
