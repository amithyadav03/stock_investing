import os
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import matplotlib
matplotlib.use('Agg') # FIXES: "Starting a Matplotlib GUI outside of the main thread"
import mplfinance as mpf
from typing import Dict, Any
from core.config import settings
from kiteconnect import KiteConnect
from datetime import datetime, timedelta
from tenacity import retry, stop_after_attempt, wait_exponential

class MarketDataTool:
    def __init__(self):
        self.charts_dir = "./db/charts"
        os.makedirs(self.charts_dir, exist_ok=True)
        
        # Initialize Kite Connect
        self.kite = None
        if settings.KITE_API_KEY and settings.KITE_ACCESS_TOKEN:
            try:
                self.kite = KiteConnect(api_key=settings.KITE_API_KEY)
                self.kite.set_access_token(settings.KITE_ACCESS_TOKEN)
            except Exception as e:
                print(f"[Market Data] Kite initialization error: {e}")

    def get_kite_instrument_token(self, symbol: str) -> int:
        """
        Dynamically fetches the exact numeric instrument token from Zerodha.
        """
        try:
            instruments = self.kite.instruments("NSE")
            for item in instruments:
                if item['tradingsymbol'] == symbol:
                    return item['instrument_token']
        except Exception as e:
            print(f"[Market Data] Instrument lookup failed: {e}")
            
        print(f"[WARNING] Could not find NSE token for {symbol}. Falling back to default.")
        return 738561 # Fallback to Reliance

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def fetch_advanced_technicals(self, symbol: str, period_days: int = None) -> Dict[str, Any]:
        """
        Fetches 2+ years of OHLCV data to identify major structural levels.
        Calculates S/R zones, Market Correlation (Beta), and Weekly Trend.
        """
        if not period_days:
            period_days = settings.strategy.get("scanning", {}).get("lookback_days", 700)
            
        df = pd.DataFrame()
        
        # 1. LIVE KITE DATA
        if self.kite:
            try:
                instrument_token = self.get_kite_instrument_token(symbol)
                to_date = datetime.now()
                from_date = to_date - timedelta(days=period_days)
                
                records = self.kite.historical_data(
                    instrument_token=instrument_token,
                    from_date=from_date.strftime("%Y-%m-%d"),
                    to_date=to_date.strftime("%Y-%m-%d"),
                    interval="day"
                )
                
                df = pd.DataFrame(records)
                if not df.empty:
                    df = df.rename(columns={'date': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
                    df.set_index('Date', inplace=True)
            except Exception as e:
                print(f"[Market Data] Kite Historical API failed: {e}. Falling back to yfinance.")

        # 2. YFINANCE FALLBACK
        if df.empty:
            symbol_fetch = f"{symbol}.NS" if not (symbol.endswith(".NS") or symbol.endswith(".BO")) else symbol
            df = yf.Ticker(symbol_fetch).history(period=f"{period_days}d")
            
        if df.empty or len(df) < 50:
            return {"error": "Not enough data points to compute indicators."}

        # --- WEEKLY TREND GUARDRAIL ---
        df_weekly = df.resample('W').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'})
        df_weekly['sma20'] = df_weekly['Close'].rolling(window=20).mean()
        weekly_trend = "UP" if df_weekly['Close'].iloc[-1] > df_weekly['sma20'].iloc[-1] else "DOWN"

        # --- STRUCTURAL SUPPORT / RESISTANCE (2-Year Window) ---
        # Find local peaks/troughs using a rolling window
        df['min_20'] = df['Low'].rolling(window=20, center=True).min()
        df['max_20'] = df['High'].rolling(window=20, center=True).max()
        
        potential_support = df[df['Low'] == df['min_20']]['Low'].unique()
        potential_res = df[df['High'] == df['max_20']]['High'].unique()
        
        # Filter for recent relevant levels (within 20% of current price)
        current_price = df['Close'].iloc[-1]
        support_levels = sorted([round(x, 2) for x in potential_support if 0.8 * current_price <= x < current_price], reverse=True)[:3]
        resistance_levels = sorted([round(x, 2) for x in potential_res if current_price < x <= 1.2 * current_price])[:3]

        # --- RELATIVE STRENGTH (vs NIFTY 50) ---
        try:
            nifty_df = yf.Ticker("^NSEI").history(period="60d")
            stock_perf = (df['Close'].iloc[-1] - df['Close'].iloc[-30]) / df['Close'].iloc[-30]
            nifty_perf = (nifty_df['Close'].iloc[-1] - nifty_df['Close'].iloc[-30]) / nifty_df['Close'].iloc[-30]
            rs_score = round(stock_perf - nifty_perf, 4) # Positive means outperforming market
        except:
            rs_score = 0.0

        # --- INDICATORS ---
        df.ta.atr(length=14, append=True)
        df.ta.rsi(length=14, append=True)
        df.ta.macd(fast=12, slow=26, signal=9, append=True)
        
        # --- SEQUENTIAL DATA (Last 14 Days) ---
        recent_df = df.tail(14).copy()
        recent_candles = []
        for idx, row in recent_df.iterrows():
            recent_candles.append({
                "day": idx.strftime("%Y-%m-%d") if hasattr(idx, 'strftime') else str(idx),
                "open": round(row['Open'], 2),
                "high": round(row['High'], 2),
                "low": round(row['Low'], 2),
                "close": round(row['Close'], 2),
                "vol": int(row['Volume'])
            })

        # --- VISION RENDERING ---
        chart_path = os.path.abspath(f"{self.charts_dir}/{symbol}_chart.png")
        plot_df = df.tail(90)
        macd_col = [col for col in df.columns if col.startswith('MACD_')][0]
        macds_col = [col for col in df.columns if col.startswith('MACDs_')][0]
        
        apdict = [
            mpf.make_addplot(plot_df[macd_col], panel=1, color='fuchsia', ylabel='MACD'),
            mpf.make_addplot(plot_df[macds_col], panel=1, color='b')
        ]
        
        mpf.plot(plot_df, type='candle', volume=True, style='charles',
                 title=f"{symbol} - Daily Analysis",
                 addplot=apdict,
                 savefig=dict(fname=chart_path, dpi=100, bbox_inches='tight'))

        latest = df.iloc[-1]
        avg_vol_30d = df['Volume'].tail(30).mean()
        atr_col = [col for col in df.columns if col.startswith('ATRr_')][0]
        rsi_col = [col for col in df.columns if col.startswith('RSI_')][0]
        
        # --- SEQUENTIAL DATA TABLE ---
        candles_table = "| Day | Open | High | Low | Close | Vol |\n| :--- | :--- | :--- | :--- | :--- | :--- |\n"
        for c in recent_candles:
            candles_table += f"| {c['day']} | {c['open']} | {c['high']} | {c['low']} | {c['close']} | {c['vol']} |\n"

        return {
            "symbol": symbol,
            "source": "Kite Connect" if self.kite else "yfinance",
            "latest_price": round(latest['Close'], 2),
            "atr_14": round(latest[atr_col], 2), 
            "rsi_14": round(latest[rsi_col], 2),
            "macd_histogram": round(latest[macd_col] - latest[macds_col], 2),
            "chart_path": chart_path,
            "average_volume_30d": float(avg_vol_30d),
            "relative_strength_30d": rs_score,
            "weekly_trend": weekly_trend,
            "support_levels": support_levels,
            "resistance_levels": resistance_levels,
            "recent_candles": candles_table
        }

    def get_current_price(self, symbol: str) -> float:
        if self.kite:
            try:
                instrument_token = self.get_kite_instrument_token(symbol)
                quote = self.kite.quote(f"NSE:{symbol}")
                return quote[f"NSE:{symbol}"]["last_price"]
            except: pass
            
        symbol_fetch = f"{symbol}.NS"
        df = yf.Ticker(symbol_fetch).history(period="1d")
        if not df.empty:
            return round(df['Close'].iloc[-1], 2)
        return 0.0

market_data_tool = MarketDataTool()

market_data_tool = MarketDataTool()
