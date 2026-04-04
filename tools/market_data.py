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

    def fetch_advanced_technicals(self, symbol: str, period_days: int = 180) -> Dict[str, Any]:
        """
        Fetches EXACT OHLCV data from Zerodha Kite (or fallback), calculates pandas-ta math,
        and renders the Vision chart.
        """
        df = pd.DataFrame()
        
        # 1. LIVE KITE DATA (Zero Latency)
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
                    print(f"[Market Data] Successfully fetched {len(df)} candles for {symbol} natively from Zerodha Kite.")
            except Exception as e:
                print(f"[Market Data] Kite Historical API failed: {e}. Falling back to yfinance.")

        # 2. YFINANCE FALLBACK
        if df.empty:
            symbol_fetch = f"{symbol}.NS" if not (symbol.endswith(".NS") or symbol.endswith(".BO")) else symbol
            df = yf.Ticker(symbol_fetch).history(period=f"{period_days}d")
            print(f"[Market Data] Fetched candles via yfinance fallback.")
            
        if df.empty or len(df) < 50:
            return {"error": "Not enough data points to compute indicators."}

        # --- PYTHON MATH (pandas-ta) ---
        df.ta.atr(length=14, append=True)
        df.ta.rsi(length=14, append=True)
        df.ta.macd(fast=12, slow=26, signal=9, append=True)
        df.ta.bbands(length=20, std=2, append=True)
        
        # --- VISION RENDERING (mplfinance) ---
        chart_path = os.path.abspath(f"{self.charts_dir}/{symbol}_chart.png")
        plot_df = df.tail(90)
        
        macd_col = [col for col in df.columns if col.startswith('MACD_')][0]
        macds_col = [col for col in df.columns if col.startswith('MACDs_')][0]
        
        apdict = [
            mpf.make_addplot(plot_df[macd_col], panel=1, color='fuchsia', ylabel='MACD'),
            mpf.make_addplot(plot_df[macds_col], panel=1, color='b')
        ]
        
        mpf.plot(plot_df, type='candle', volume=True, style='charles',
                 title=f"{symbol} - Last 90 Days",
                 addplot=apdict,
                 savefig=dict(fname=chart_path, dpi=100, bbox_inches='tight'))

        # --- SYNTHESIS POST-CALCULATION ---
        latest = df.iloc[-1]
        atr_col = [col for col in df.columns if col.startswith('ATRr_')][0]
        rsi_col = [col for col in df.columns if col.startswith('RSI_')][0]
        
        return {
            "source": "Kite Connect" if self.kite else "yfinance",
            "latest_price": round(latest['Close'], 2),
            "atr_14": round(latest[atr_col], 2), 
            "rsi_14": round(latest[rsi_col], 2),
            "macd_histogram": round(latest[macd_col] - latest[macds_col], 2),
            "chart_path": chart_path 
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
