import os
import yfinance as yf
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import mplfinance as mpf
from typing import Dict, Any
from core.config import settings
from kiteconnect import KiteConnect
from datetime import datetime, timedelta
from tenacity import retry, stop_after_attempt, wait_exponential
from tools.indicators import add_all_indicators


class MarketDataTool:
    def __init__(self):
        self.charts_dir = "./db/charts"
        os.makedirs(self.charts_dir, exist_ok=True)

        self.kite = None
        if settings.KITE_API_KEY and settings.KITE_ACCESS_TOKEN:
            try:
                self.kite = KiteConnect(api_key=settings.KITE_API_KEY)
                self.kite.set_access_token(settings.KITE_ACCESS_TOKEN)
            except Exception as e:
                print(f"[Market Data] Kite init error: {e}")

    def get_kite_instrument_token(self, symbol: str) -> int | None:
        try:
            instruments = self.kite.instruments("NSE")
            for item in instruments:
                if item['tradingsymbol'] == symbol:
                    return item['instrument_token']
        except Exception as e:
            print(f"[Market Data] Instrument lookup failed for {symbol}: {e}")
        return None  # Explicit None — caller decides what to do

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def fetch_advanced_technicals(self, symbol: str, period_days: int = None) -> Dict[str, Any]:
        """
        2+ years of OHLCV → structural levels, multi-indicator suite, candlestick chart.
        Indicators: ATR, RSI, MACD, Bollinger Bands, ADX, Stochastic, EMA(20/50/200).
        """
        if not period_days:
            period_days = settings.strategy.get("scanning", {}).get("lookback_days", 700)

        df = pd.DataFrame()

        if self.kite:
            try:
                token = self.get_kite_instrument_token(symbol)
                if token:
                    to_date = datetime.now()
                    from_date = to_date - timedelta(days=period_days)
                    records = self.kite.historical_data(
                        instrument_token=token,
                        from_date=from_date.strftime("%Y-%m-%d"),
                        to_date=to_date.strftime("%Y-%m-%d"),
                        interval="day",
                    )
                    df = pd.DataFrame(records)
                    if not df.empty:
                        df = df.rename(columns={
                            'date': 'Date', 'open': 'Open', 'high': 'High',
                            'low': 'Low', 'close': 'Close', 'volume': 'Volume',
                        })
                        df.set_index('Date', inplace=True)
            except Exception as e:
                print(f"[Market Data] Kite historical failed for {symbol}: {e}. Using yfinance.")

        if df.empty:
            sym = f"{symbol}.NS" if not (symbol.endswith(".NS") or symbol.endswith(".BO")) else symbol
            df = yf.Ticker(sym).history(period=f"{period_days}d")

        if df.empty or len(df) < 50:
            return {"error": f"Insufficient data for {symbol}."}

        # ── Indicators ─────────────────────────────────────────────────────────
        df = add_all_indicators(df)

        # ── Weekly trend (Close vs 20-week SMA) ────────────────────────────────
        df_w = df.resample('W').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'})
        df_w['sma20'] = df_w['Close'].rolling(20).mean()
        weekly_trend = "UP" if df_w['Close'].iloc[-1] > df_w['sma20'].iloc[-1] else "DOWN"

        # ── Support / Resistance (2-year rolling min/max) ───────────────────────
        df['min_20'] = df['Low'].rolling(20, center=True).min()
        df['max_20'] = df['High'].rolling(20, center=True).max()
        price = df['Close'].iloc[-1]
        support_levels = sorted(
            [round(x, 2) for x in df[df['Low'] == df['min_20']]['Low'].unique()
             if 0.8 * price <= x < price], reverse=True
        )[:3]
        resistance_levels = sorted(
            [round(x, 2) for x in df[df['High'] == df['max_20']]['High'].unique()
             if price < x <= 1.2 * price]
        )[:3]

        # ── Relative strength vs NIFTY 50 (30-day) ─────────────────────────────
        try:
            nifty = yf.Ticker("^NSEI").history(period="60d")
            rs_score = round(
                (df['Close'].iloc[-1] - df['Close'].iloc[-30]) / df['Close'].iloc[-30]
                - (nifty['Close'].iloc[-1] - nifty['Close'].iloc[-30]) / nifty['Close'].iloc[-30],
                4,
            )
        except Exception:
            rs_score = 0.0

        # ── Chart (90-day candle + MACD) ────────────────────────────────────────
        chart_path = os.path.abspath(f"{self.charts_dir}/{symbol}_chart.png")
        plot_df = df.tail(90).copy()

        macd_col = next((c for c in df.columns if c.startswith('MACD_12')), None)
        macds_col = next((c for c in df.columns if c.startswith('MACDs_12')), None)
        bb_upper = next((c for c in df.columns if c.startswith('BBU_')), None)
        bb_lower = next((c for c in df.columns if c.startswith('BBL_')), None)

        apdict = []
        if macd_col and macds_col:
            apdict += [
                mpf.make_addplot(plot_df[macd_col], panel=1, color='fuchsia', ylabel='MACD'),
                mpf.make_addplot(plot_df[macds_col], panel=1, color='b'),
            ]
        if bb_upper and bb_lower:
            apdict += [
                mpf.make_addplot(plot_df[bb_upper], panel=0, color='gray', linestyle='--', width=0.7),
                mpf.make_addplot(plot_df[bb_lower], panel=0, color='gray', linestyle='--', width=0.7),
            ]

        try:
            mpf.plot(
                plot_df, type='candle', volume=True, style='charles',
                title=f"{symbol} — Daily",
                addplot=apdict,
                savefig=dict(fname=chart_path, dpi=100, bbox_inches='tight'),
            )
        except Exception as e:
            print(f"[Market Data] Chart render failed for {symbol}: {e}")
            chart_path = None

        # ── Latest values ───────────────────────────────────────────────────────
        latest = df.iloc[-1]
        avg_vol_30 = float(df['Volume'].tail(30).mean())

        def _get(col_prefix: str, default=0.0):
            col = next((c for c in df.columns if c.startswith(col_prefix)), None)
            return round(float(latest[col]), 4) if col else default

        atr_val    = _get('ATRr_14')
        rsi_val    = _get('RSI_14')
        macd_h_val = round(float(latest[macd_col] - latest[macds_col]), 4) if macd_col and macds_col else 0.0
        adx_val    = _get('ADX_14')
        stoch_k    = _get('STOCHk_14_3_3')
        bb_pct_b   = _get('BBP_20_2.0')  # % position within BB bands (0=lower, 1=upper)
        ema20      = _get('EMA_20')
        ema50      = _get('EMA_50')
        ema200     = _get('EMA_200')

        # ── Recent candle table (14 days) ───────────────────────────────────────
        recent = df.tail(14)
        candles_table = "| Day | Open | High | Low | Close | Vol |\n| :--- | :--- | :--- | :--- | :--- | :--- |\n"
        for idx, row in recent.iterrows():
            day_str = idx.strftime("%Y-%m-%d") if hasattr(idx, 'strftime') else str(idx)
            candles_table += f"| {day_str} | {round(row['Open'],2)} | {round(row['High'],2)} | {round(row['Low'],2)} | {round(row['Close'],2)} | {int(row['Volume'])} |\n"

        return {
            "symbol": symbol,
            "source": "Kite Connect" if self.kite else "yfinance",
            "latest_price": round(float(price), 2),
            "atr_14": atr_val,
            "rsi_14": rsi_val,
            "macd_histogram": macd_h_val,
            "adx_14": adx_val,
            "stoch_k": stoch_k,
            "bb_pct_b": bb_pct_b,
            "ema_20": ema20,
            "ema_50": ema50,
            "ema_200": ema200,
            "average_volume_30d": avg_vol_30,
            "relative_strength_30d": rs_score,
            "weekly_trend": weekly_trend,
            "support_levels": support_levels,
            "resistance_levels": resistance_levels,
            "recent_candles": candles_table,
            "chart_path": chart_path,
        }

    def get_current_price(self, symbol: str) -> float:
        if self.kite:
            try:
                quote = self.kite.quote(f"NSE:{symbol}")
                return float(quote[f"NSE:{symbol}"]["last_price"])
            except Exception:
                pass
        sym = f"{symbol}.NS"
        df = yf.Ticker(sym).history(period="1d")
        if not df.empty:
            return round(float(df['Close'].iloc[-1]), 2)
        return 0.0


market_data_tool = MarketDataTool()
