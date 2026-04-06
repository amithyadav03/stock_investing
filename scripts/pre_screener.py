import os
import sys
import json
import glob
import time
import pandas as pd
from datetime import date, timedelta

# Add parent directory to path to allow importing core modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from jugaad_data.nse import bhavcopy_save
except ImportError:
    print("Please install jugaad-data: pip install jugaad-data")
    sys.exit(1)

DOWNLOAD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'db'))
OUTPUT_JSON = os.path.join(DOWNLOAD_DIR, 'daily_scan_list.json')


def get_candidate_dates(n=5):
    """Return up to n past week-days to try (most recent first)."""
    candidates = []
    d = date.today() - timedelta(days=1)
    while len(candidates) < n:
        if d.weekday() < 5:  # Mon-Fri only
            candidates.append(d)
        d -= timedelta(days=1)
    return candidates


def find_or_download_bhavcopy():
    """Try to find an existing bhavcopy, or download the most recent one."""
    # First check if any bhavcopy CSV already exists in db/
    existing = glob.glob(os.path.join(DOWNLOAD_DIR, 'cm*bhav.csv'))
    if existing:
        latest = max(existing, key=os.path.getmtime)
        print(f"Found existing Bhavcopy: {os.path.basename(latest)}")
        return latest

    # Otherwise try downloading for recent trading days
    for target_date in get_candidate_dates():
        print(f"Attempting to download Bhavcopy for {target_date}...")
        try:
            bhavcopy_save(target_date, DOWNLOAD_DIR)
            time.sleep(1)
            downloaded = glob.glob(os.path.join(DOWNLOAD_DIR, 'cm*bhav.csv'))
            if downloaded:
                return max(downloaded, key=os.path.getmtime)
        except Exception as e:
            print(f"  Could not fetch {target_date}: {e}")
            continue

    return None


def run_pre_screener():
    print("=" * 60)
    print("🚀 Running Tier-1 Multi-Factor Market Discovery (v4.0) 🚀")
    print("=" * 60)

    # --- STAGE 1: RAW MOMENTUM (Bhavcopy) ---
    file_path = find_or_download_bhavcopy()
    if not file_path:
        print("Fatal: Could not locate Bhavcopy CSV.")
        return []

    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    col_map = {'OPEN_PRICE': 'OPEN', 'CLOSE_PRICE': 'CLOSE', 'TTL_TRD_QNTY': 'TOTTRDQTY', 'PREV_CLOSE': 'PREVCLOSE'}
    df.rename(columns=col_map, inplace=True)
    
    if 'SERIES' in df.columns:
        df = df[df['SERIES'] == 'EQ']
    
    # HEURISTIC FILTERS (Liquidity & Penny Guard)
    s1_candidates = df[
        (df['TOTTRDQTY'] > 500000) & 
        (df['CLOSE'] > df['OPEN']) & 
        (df['CLOSE'] > 20.0)
    ].copy()
    
    s1_candidates['pct_change'] = (s1_candidates['CLOSE'] - s1_candidates['PREVCLOSE']) / s1_candidates['PREVCLOSE'] * 100
    s1_candidates = s1_candidates[(s1_candidates['pct_change'] > 0.5) & (s1_candidates['pct_change'] < 10.0)]
    
    symbols = s1_candidates['SYMBOL'].tolist()
    print(f"[Stage 1] Found {len(symbols)} Raw Momentum Candidates.")

    # --- STAGE 2: TECHNICAL CONFIRMATION (Relative Volume & Trend) ---
    if not symbols: return []
    
    print(f"[Stage 2] Validating RV and 20-EMA for {len(symbols)} equities...")
    import yfinance as yf
    final_candidates = []
    
    # Bulk Download for speed
    yf_symbols = [f"{s}.NS" for s in symbols]
    data = yf.download(yf_symbols, period="40d", group_by='ticker', threads=True, progress=False)
    
    for s in symbols:
        try:
            ticker_df = data[f"{s}.NS"]
            if ticker_df.empty or len(ticker_df) < 25: continue
            
            # Use pandas-ta for speed/consistency
            ticker_df.ta.ema(length=20, append=True)
            ema_col = "EMA_20"
            
            latest_price = ticker_df['Close'].iloc[-1]
            latest_ema = ticker_df[ema_col].iloc[-1]
            
            # Relative Volume (Current vs 20-day Average)
            avg_vol_20 = ticker_df['Volume'].iloc[-21:-1].mean()
            rel_vol = ticker_df['Volume'].iloc[-1] / avg_vol_20 if avg_vol_20 > 0 else 0
            
            # FILTERS: RV > 1.2 (Surge) & Price > EMA_20 (Uptrend)
            if rel_vol > 1.2 and latest_price > latest_ema:
                final_candidates.append({
                    "symbol": s,
                    "rv": round(rel_vol, 2),
                    "pct_change": s1_candidates[s1_candidates['SYMBOL'] == s]['pct_change'].iloc[0]
                })
        except: continue

    print(f"[Stage 2] {len(final_candidates)} passed Technical Confirmation.")
    
    # Sort by RV (Institutional footprint)
    final_candidates = sorted(final_candidates, key=lambda x: x['rv'], reverse=True)[:25]
    
    # --- STAGE 3: SENTIMENT GUARD (Headline Filter) ---
    print(f"[Stage 3] Running Headline Guard for Top {len(final_candidates)} candidates...")
    from tools.fundamental_news import fundamental_news_tool
    
    elite_symbols = []
    neg_keywords = ["fraud", "default", "scandal", "investigation", "raid", "ed", "sebi penalty", "bankruptcy"]
    
    for c in final_candidates:
        s = c['symbol']
        try:
            news = fundamental_news_tool.fetch_live_news_snippets(target_keyword=s)
            is_clean = True
            for headline in news:
                if any(k in headline.lower() for k in neg_keywords):
                    print(f"  ⚠️ Skipping {s} due to negative news: {headline[:50]}...")
                    is_clean = False
                    break
            if is_clean:
                elite_symbols.append(s)
        except:
            elite_symbols.append(s) # Default to include if news fetch fails

    print(f"\nFinal Elite Candidates: {elite_symbols}")
    
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(elite_symbols, f, indent=4)
        
    print(f"✅ Discovery List (v4.0) saved to {OUTPUT_JSON}")
    return elite_symbols

if __name__ == "__main__":
    run_pre_screener()
