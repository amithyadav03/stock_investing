"""
Nifty 500 Pre-Screener v5.0
3-stage elite discovery pipeline:
  Stage 1: NSE Bhavcopy (Volume > 500k, green, 0.5–10% change, price > ₹20)
           Falls back to Nifty 500 static list if Bhavcopy unavailable.
  Stage 2: Technical confirmation (Relative Volume > 1.2, price > EMA20)
  Stage 3: Sentiment guard (exclude fraud/negative news)
Output: db/daily_scan_list.json
"""

import os
import sys
import json
import glob
import time
import pandas as pd
from datetime import date, timedelta

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.nifty500_symbols import NIFTY_500

DOWNLOAD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'db'))
OUTPUT_JSON = os.path.join(DOWNLOAD_DIR, 'daily_scan_list.json')

# Negative keywords for Stage 3 sentiment guard
NEG_KEYWORDS = [
    "fraud", "default", "scandal", "investigation", "raid", "ed probe",
    "sebi penalty", "bankruptcy", "insolvency", "promoter pledge", "downgrade",
    "earnings miss", "profit warning", "write-off",
]


def get_candidate_dates(n=5):
    candidates = []
    d = date.today() - timedelta(days=1)
    while len(candidates) < n:
        if d.weekday() < 5:
            candidates.append(d)
        d -= timedelta(days=1)
    return candidates


def find_or_download_bhavcopy():
    existing = glob.glob(os.path.join(DOWNLOAD_DIR, 'cm*bhav.csv'))
    if existing:
        latest = max(existing, key=os.path.getmtime)
        # Only use if it's recent (within 5 trading days)
        print(f"[Bhavcopy] Found: {os.path.basename(latest)}")
        return latest

    try:
        from jugaad_data.nse import bhavcopy_save
        for target_date in get_candidate_dates():
            print(f"[Bhavcopy] Downloading for {target_date}...")
            try:
                bhavcopy_save(target_date, DOWNLOAD_DIR)
                time.sleep(1)
                downloaded = glob.glob(os.path.join(DOWNLOAD_DIR, 'cm*bhav.csv'))
                if downloaded:
                    return max(downloaded, key=os.path.getmtime)
            except Exception as e:
                print(f"  Could not fetch {target_date}: {e}")
    except ImportError:
        print("[Bhavcopy] jugaad-data not installed. Using static Nifty 500 list.")
    return None


def stage1_bhavcopy(file_path: str) -> list[str]:
    """Filter Bhavcopy for high-volume, green, trending stocks."""
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()

    col_map = {
        'OPEN_PRICE': 'OPEN', 'CLOSE_PRICE': 'CLOSE',
        'TTL_TRD_QNTY': 'TOTTRDQTY', 'PREV_CLOSE': 'PREVCLOSE',
    }
    df.rename(columns=col_map, inplace=True)

    if 'SERIES' in df.columns:
        df = df[df['SERIES'] == 'EQ']

    required = {'TOTTRDQTY', 'CLOSE', 'OPEN', 'PREVCLOSE', 'SYMBOL'}
    if not required.issubset(df.columns):
        print(f"[Stage 1] Missing columns. Available: {df.columns.tolist()}")
        return []

    s1 = df[
        (df['TOTTRDQTY'] > 500_000) &
        (df['CLOSE'] > df['OPEN']) &
        (df['CLOSE'] > 20.0)
    ].copy()

    s1['pct_change'] = (s1['CLOSE'] - s1['PREVCLOSE']) / s1['PREVCLOSE'] * 100
    s1 = s1[(s1['pct_change'] > 0.5) & (s1['pct_change'] < 10.0)]

    # Filter to Nifty 500 universe
    nifty500_set = set(NIFTY_500)
    s1 = s1[s1['SYMBOL'].isin(nifty500_set)]

    symbols = s1['SYMBOL'].tolist()
    print(f"[Stage 1] {len(symbols)} Nifty 500 momentum candidates from Bhavcopy.")
    return symbols


def stage1_static_fallback() -> list[str]:
    """When Bhavcopy unavailable, use the full Nifty 500 static list."""
    print(f"[Stage 1 Fallback] Using Nifty 500 static list ({len(NIFTY_500)} symbols).")
    return list(NIFTY_500)


def stage2_technical_confirmation(symbols: list[str]) -> list[dict]:
    """Relative Volume > 1.2 and price > EMA20 confirmation."""
    import yfinance as yf
    from tools.indicators import ema as calc_ema

    print(f"[Stage 2] Validating RV and EMA20 for {len(symbols)} symbols...")

    yf_symbols = [f"{s}.NS" for s in symbols]
    try:
        data = yf.download(yf_symbols, period="40d", group_by='ticker', threads=True, progress=False)
    except Exception as e:
        print(f"[Stage 2] Bulk download failed: {e}. Skipping.")
        return []

    confirmed = []
    for s in symbols:
        try:
            # Handle both single and multi-ticker download formats
            if len(symbols) == 1:
                df = data
            else:
                key = f"{s}.NS"
                if key not in data.columns.get_level_values(0):
                    continue
                df = data[key]

            if df.empty or len(df) < 25:
                continue

            df = df.dropna()
            df['EMA_20'] = calc_ema(df, 20)
            latest_price = float(df['Close'].iloc[-1])
            latest_ema = float(df['EMA_20'].iloc[-1])
            if pd.isna(latest_ema):
                continue
            avg_vol_20 = float(df['Volume'].iloc[-21:-1].mean())
            rel_vol = float(df['Volume'].iloc[-1]) / avg_vol_20 if avg_vol_20 > 0 else 0

            if rel_vol > 1.2 and latest_price > latest_ema:
                confirmed.append({"symbol": s, "rv": round(rel_vol, 2), "price": round(latest_price, 2)})
        except Exception:
            continue

    confirmed.sort(key=lambda x: x['rv'], reverse=True)
    print(f"[Stage 2] {len(confirmed)} passed technical confirmation.")
    return confirmed[:30]  # Top 30 by relative volume


def stage3_sentiment_guard(candidates: list[dict]) -> list[str]:
    """Exclude stocks with negative news flags."""
    from tools.fundamental_news import fundamental_news_tool

    print(f"[Stage 3] Sentiment guard for {len(candidates)} candidates...")
    clean = []
    for c in candidates:
        s = c['symbol']
        try:
            news = fundamental_news_tool.fetch_live_news_snippets(target_keyword=s)
            flagged = any(kw in h.lower() for h in news for kw in NEG_KEYWORDS)
            if flagged:
                print(f"  Excluded {s} — negative news detected.")
            else:
                clean.append(s)
        except Exception:
            clean.append(s)  # Include on fetch failure (fail open)
    return clean


def run_pre_screener() -> list[str]:
    print("=" * 60)
    print("NIFTY 500 PRE-SCREENER v5.0")
    print("=" * 60)

    # Stage 1
    file_path = find_or_download_bhavcopy()
    if file_path:
        symbols = stage1_bhavcopy(file_path)
    else:
        symbols = stage1_static_fallback()

    if not symbols:
        print("Stage 1 produced no candidates.")
        return []

    # Stage 2
    confirmed = stage2_technical_confirmation(symbols)
    if not confirmed:
        print("Stage 2 produced no confirmed candidates.")
        return []

    # Stage 3
    elite = stage3_sentiment_guard(confirmed)

    print(f"\nElite Candidates ({len(elite)}): {elite}")
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(elite, f, indent=4)
    print(f"Saved to {OUTPUT_JSON}")
    return elite


if __name__ == "__main__":
    run_pre_screener()
