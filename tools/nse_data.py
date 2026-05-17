"""
NSE official data — FII/DII flows, index performance, global market cues.
All free from public sources.
"""

import requests
import yfinance as yf
from datetime import datetime, timedelta
from core.cache import cache, TTL_GLOBAL_CUES
from core.config import settings


_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
    "Referer": "https://www.nseindia.com/",
}


def get_fii_dii_data() -> dict:
    """
    Fetches FII/DII net buy/sell data from NSE.
    Returns last 5 days of institutional flow data.
    """
    cached = cache.get("fii_dii_data")
    if cached:
        return cached

    result = {"fii_net": [], "dii_net": [], "summary": "", "error": None}

    try:
        session = requests.Session()
        # Prime the session with a cookie
        session.get("https://www.nseindia.com/", headers=_HEADERS, timeout=10)

        resp = session.get(
            "https://www.nseindia.com/api/fiidiiTradeReact",
            headers=_HEADERS,
            timeout=15,
        )
        if resp.status_code == 200:
            data = resp.json()
            fii_records = []
            dii_records = []
            for row in data[-5:]:  # Last 5 trading days
                date_str = row.get("date", "")
                fii_net = float(str(row.get("fiiNetBuy", "0")).replace(",", "") or 0)
                dii_net = float(str(row.get("diiNetBuy", "0")).replace(",", "") or 0)
                fii_records.append({"date": date_str, "net_crore": round(fii_net / 10_000_000, 2)})
                dii_records.append({"date": date_str, "net_crore": round(dii_net / 10_000_000, 2)})

            total_fii = sum(r["net_crore"] for r in fii_records)
            total_dii = sum(r["net_crore"] for r in dii_records)

            result = {
                "fii_net": fii_records,
                "dii_net": dii_records,
                "fii_5d_total_crore": round(total_fii, 2),
                "dii_5d_total_crore": round(total_dii, 2),
                "summary": (
                    f"FII: {'Buying' if total_fii > 0 else 'Selling'} ₹{abs(total_fii):.0f}Cr (5d) | "
                    f"DII: {'Buying' if total_dii > 0 else 'Selling'} ₹{abs(total_dii):.0f}Cr (5d)"
                ),
                "error": None,
            }
            cache.set("fii_dii_data", result, ttl_seconds=14400)  # 4 hrs
    except Exception as e:
        print(f"[NSEData] FII/DII fetch failed: {e}")
        result["error"] = str(e)
        result["summary"] = "FII/DII data unavailable."

    return result


def get_global_cues() -> list[dict]:
    """
    Fetches overnight global market data for the morning brief.
    Returns list of {name, ticker, price, change_pct, direction}.
    """
    cached = cache.get("global_cues")
    if cached:
        return cached

    indices = settings.strategy.get("global_indices", [
        {"ticker": "^GSPC", "name": "S&P 500"},
        {"ticker": "^IXIC", "name": "NASDAQ"},
        {"ticker": "^N225", "name": "Nikkei 225"},
        {"ticker": "GC=F",  "name": "Gold"},
        {"ticker": "CL=F",  "name": "Crude Oil"},
        {"ticker": "USDINR=X", "name": "USD/INR"},
        {"ticker": "^VIX",  "name": "VIX"},
    ])

    results = []
    for idx in indices:
        try:
            ticker = idx["ticker"]
            df = yf.Ticker(ticker).history(period="2d")
            if len(df) >= 2:
                prev = df['Close'].iloc[-2]
                curr = df['Close'].iloc[-1]
                change_pct = round((curr - prev) / prev * 100, 2)
                results.append({
                    "name": idx["name"],
                    "ticker": ticker,
                    "price": round(curr, 2),
                    "change_pct": change_pct,
                    "direction": "▲" if change_pct >= 0 else "▼",
                })
        except Exception:
            pass

    if results:
        cache.set("global_cues", results, TTL_GLOBAL_CUES)
    return results


def get_nifty_indices_performance() -> dict:
    """Nifty 50, Bank Nifty, Midcap, Smallcap performance today and 5-day."""
    cached = cache.get("nifty_performance")
    if cached:
        return cached

    indices = {
        "Nifty 50":      "^NSEI",
        "Bank Nifty":    "^NSEBANK",
        "Nifty Midcap":  "NIFTY_MIDCAP_100.NS",
        "Nifty IT":      "^CNXIT",
        "Nifty Pharma":  "^CNXPHARMA",
    }
    results = {}
    for name, ticker in indices.items():
        try:
            df = yf.Ticker(ticker).history(period="10d")
            if len(df) >= 2:
                day_chg = round((df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2] * 100, 2)
                week_chg = round((df['Close'].iloc[-1] - df['Close'].iloc[-5]) / df['Close'].iloc[-5] * 100, 2) if len(df) >= 5 else 0.0
                results[name] = {
                    "last_price": round(df['Close'].iloc[-1], 2),
                    "day_change_pct": day_chg,
                    "week_change_pct": week_chg,
                }
        except Exception:
            pass

    if results:
        cache.set("nifty_performance", results, ttl_seconds=1800)
    return results


def format_global_cues_text(cues: list[dict]) -> str:
    """Formats global cues into a Telegram-friendly string."""
    if not cues:
        return "_Global data unavailable_"
    lines = []
    for c in cues:
        sign = "+" if c["change_pct"] >= 0 else ""
        lines.append(f"{c['direction']} *{c['name']}*: {c['price']} ({sign}{c['change_pct']}%)")
    return "\n".join(lines)
