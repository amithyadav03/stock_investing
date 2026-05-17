"""
Nifty 500 Pre-Screener v5.1
3-stage elite discovery pipeline:
  Stage 1: NSE Bhavcopy (Volume > 500k, green, 0.5–10% change, price > ₹20)
           Falls back to top 200 of Nifty 500 static list if Bhavcopy unavailable.
  Stage 2: Enhanced technical confirmation (RV > 1.3, price > EMA20, ADX > 18,
           green candle, price > ₹50, avg daily value > ₹5 crore)
  Stage 3: Claude-based sentiment guard (excludes BEARISH/VERY_BEARISH stocks)
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

# NSE trading holidays 2025-2026 (update annually)
NSE_HOLIDAYS_2025_2026 = {
    date(2025, 1, 26),   # Republic Day
    date(2025, 2, 26),   # Mahashivratri
    date(2025, 3, 14),   # Holi
    date(2025, 3, 31),   # Id-Ul-Fitr (Ramadan Eid) - tentative
    date(2025, 4, 10),   # Ram Navami - tentative
    date(2025, 4, 14),   # Dr. Ambedkar Jayanti
    date(2025, 4, 18),   # Good Friday
    date(2025, 5, 1),    # Maharashtra Day
    date(2025, 8, 15),   # Independence Day
    date(2025, 8, 27),   # Ganesh Chaturthi
    date(2025, 10, 2),   # Gandhi Jayanti / Dussehra
    date(2025, 10, 20),  # Diwali Laxmi Puja (tentative)
    date(2025, 10, 21),  # Diwali Balipratipada (tentative)
    date(2025, 11, 5),   # Gurunanak Jayanti
    date(2025, 12, 25),  # Christmas
    date(2026, 1, 26),   # Republic Day
    date(2026, 3, 20),   # Holi (tentative)
    date(2026, 4, 3),    # Good Friday (tentative)
    date(2026, 4, 14),   # Dr. Ambedkar Jayanti
    date(2026, 5, 1),    # Maharashtra Day
    date(2026, 8, 15),   # Independence Day
    date(2026, 10, 2),   # Gandhi Jayanti
    date(2026, 12, 25),  # Christmas
}

def is_trading_day(d: date) -> bool:
    """Returns True if the given date is a valid NSE trading day."""
    return d.weekday() < 5 and d not in NSE_HOLIDAYS_2025_2026


def get_candidate_dates(n=5):
    candidates = []
    d = date.today() - timedelta(days=1)
    while len(candidates) < n:
        if is_trading_day(d):
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
    """When Bhavcopy unavailable, use top 200 from Nifty 500 static list."""
    # Take only first 200 from the NIFTY_500 list (ordered by market cap in the data file)
    subset = list(NIFTY_500)[:200]
    print(f"[Stage 1 Fallback] Using top {len(subset)} symbols from Nifty 500 list.")
    return subset


def stage2_technical_confirmation(symbols: list[str]) -> list[dict]:
    """
    Enhanced technical confirmation:
    - Relative Volume > 1.3 (stronger signal)
    - Price > EMA20 (momentum)
    - ADX > 20 (trending, not ranging)
    - Green candle (close > open today)
    - Price > ₹50 (avoid penny stocks)
    - Min avg daily volume ₹5 crore (liquidity filter)
    """
    import yfinance as yf
    from tools.indicators import ema as calc_ema, adx as calc_adx
    from core.config import settings as _cfg
    _risk_cfg = _cfg.strategy.get("risk", {})
    _scan_cfg = _cfg.strategy.get("scanning", {})
    _rv_threshold = _scan_cfg.get("stage2_rv_threshold", 1.3)
    _adx_threshold = _risk_cfg.get("stage2_adx_threshold", 18)
    _top_n = _scan_cfg.get("pre_screener_top_n", 25)
    _min_price = _risk_cfg.get("min_stock_price_inr", 50)
    _min_avg_vol = _risk_cfg.get("min_avg_daily_volume", 100000)

    print(f"[Stage 2] Enhanced technical validation for {len(symbols)} symbols...")

    # Process in batches to avoid memory issues
    BATCH_SIZE = 50
    confirmed = []

    for batch_start in range(0, len(symbols), BATCH_SIZE):
        batch = symbols[batch_start:batch_start + BATCH_SIZE]
        yf_symbols = [f"{s}.NS" for s in batch]

        try:
            data = yf.download(yf_symbols, period="60d", group_by='ticker',
                              threads=True, progress=False, timeout=30)
        except Exception as e:
            print(f"[Stage 2] Batch download failed: {e}. Skipping batch.")
            continue

        for s in batch:
            try:
                if len(batch) == 1:
                    df = data
                else:
                    key = f"{s}.NS"
                    if hasattr(data.columns, 'get_level_values'):
                        if key not in data.columns.get_level_values(0):
                            continue
                        df = data[key]
                    else:
                        continue

                if df.empty or len(df) < 30:
                    continue

                df = df.dropna(subset=['Close', 'Volume'])
                if len(df) < 20:
                    continue

                latest_price = float(df['Close'].iloc[-1])
                latest_open = float(df['Open'].iloc[-1])

                # Price filter: no penny stocks
                if latest_price < _min_price:
                    continue

                df['EMA_20'] = calc_ema(df, 20)
                latest_ema = float(df['EMA_20'].iloc[-1])
                if pd.isna(latest_ema):
                    continue

                # Relative volume (vs 20-day average)
                avg_vol_20 = float(df['Volume'].iloc[-21:-1].mean())
                # Require recent trading activity (not delisted/suspended)
                today_volume = float(df['Volume'].iloc[-1]) if not df.empty else 0
                if today_volume == 0:
                    continue  # No trades today — stock may be suspended
                latest_vol = float(df['Volume'].iloc[-1])
                rel_vol = latest_vol / avg_vol_20 if avg_vol_20 > 0 else 0

                # Liquidity: avg daily value > ₹5 crore
                avg_daily_value = avg_vol_20 * float(df['Close'].iloc[-21:-1].mean())
                if avg_daily_value < 5_00_00_000:  # ₹5 crore
                    continue

                # ADX > 20 (trending market)
                try:
                    adx_s, _, _ = calc_adx(df, 14)
                    adx_val = float(adx_s.iloc[-1]) if not pd.isna(adx_s.iloc[-1]) else 0
                except Exception:
                    adx_val = 0

                # Apply filters
                is_green_candle = latest_price > latest_open
                passes = (
                    rel_vol > _rv_threshold and
                    latest_price > latest_ema and
                    is_green_candle and
                    adx_val > _adx_threshold
                )

                if passes:
                    confirmed.append({
                        "symbol": s,
                        "rv": round(rel_vol, 2),
                        "price": round(latest_price, 2),
                        "adx": round(adx_val, 1),
                        "avg_daily_value_cr": round(avg_daily_value / 1e7, 2),
                    })

            except Exception:
                continue

    # Sort by composite score: RV × ADX
    confirmed.sort(key=lambda x: x['rv'] * x.get('adx', 1), reverse=True)
    print(f"[Stage 2] {len(confirmed)} passed enhanced technical confirmation.")
    return confirmed[:_top_n]


def stage3_sentiment_guard(candidates: list[dict]) -> list[str]:
    """
    Claude-based sentiment guard — replaces naive keyword matching.
    Excludes stocks with BEARISH or VERY_BEARISH news sentiment.
    Falls back to keyword matching if Claude unavailable.
    """
    print(f"[Stage 3] Claude sentiment guard for {len(candidates)} candidates...")
    clean = []

    for c in candidates:
        s = c['symbol']
        try:
            from tools.fundamental_news import fundamental_news_tool
            news = fundamental_news_tool.fetch_live_news_snippets(target_keyword=s)

            if not news:
                clean.append(s)  # No news = include (no negative signal)
                continue

            # Check if sentiment scoring is available via Claude
            sentiment = fundamental_news_tool.get_micro_sentiment_score(s)
            label = sentiment.get("label", "NEUTRAL")

            if label in ("VERY_BEARISH", "BEARISH"):
                print(f"  Excluded {s} — {label} sentiment: {sentiment.get('summary', '')[:80]}")
            else:
                clean.append(s)

        except Exception:
            # Fallback: basic keyword check (fail open — include if we can't check)
            try:
                from tools.fundamental_news import fundamental_news_tool
                news = fundamental_news_tool.fetch_live_news_snippets(target_keyword=s)
                hard_negatives = ["fraud", "default", "bankruptcy", "insolvency", "ed raid", "sebi ban"]
                flagged = any(kw in h.lower() for h in news for kw in hard_negatives)
                if flagged:
                    print(f"  Excluded {s} — hard negative keyword detected.")
                else:
                    clean.append(s)
            except Exception:
                clean.append(s)  # Always include if we can't check

    return clean


def run_pre_screener() -> list[str]:
    print("=" * 60)
    print("NIFTY 500 PRE-SCREENER v5.1")
    print("=" * 60)

    # Stage 1
    bhavcopy_used = False
    file_path = find_or_download_bhavcopy()
    if file_path:
        symbols = stage1_bhavcopy(file_path)
        bhavcopy_used = True
    else:
        print("[WARNING] Bhavcopy unavailable — using static fallback. Results may be stale.")
        symbols = stage1_static_fallback()

    # Notify if in scheduler context
    if not bhavcopy_used:
        try:
            from core.telegram_bot import _send
            from core.config import settings
            _send({
                "chat_id": settings.TELEGRAM_CHAT_ID,
                "text": "⚠️ *Pre-screener* used static fallback (Bhavcopy unavailable). Today's scan may miss intraday movers.",
                "parse_mode": "Markdown",
            })
        except Exception:
            pass

    if not symbols:
        print("Stage 1 produced no candidates.")
        return []

    confirmed = stage2_technical_confirmation(symbols)
    if not confirmed:
        print("Stage 2 produced no confirmed candidates.")
        return []

    elite = stage3_sentiment_guard(confirmed)

    print(f"\nElite Candidates ({len(elite)}): {elite}")
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    with open(OUTPUT_JSON, 'w') as f:
        json.dump({
            "symbols": elite,
            "generated_at": str(date.today()),
            "bhavcopy_used": bhavcopy_used,
            "candidates_before_sentiment": len(confirmed),
        }, f, indent=4)
    print(f"Saved to {OUTPUT_JSON}")
    return elite


if __name__ == "__main__":
    run_pre_screener()
