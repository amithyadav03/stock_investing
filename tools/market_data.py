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
from core.cache import cache, TTL_TECHNICALS, TTL_PRICE


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

        # Validate OHLC integrity
        invalid = (df['High'] < df['Low']).sum() + (df['Close'] < 0).sum()
        if invalid > 5:
            return {"error": f"Data integrity failure for {symbol}: {invalid} bad bars."}

        # Check for recent trading activity (delisted/suspended stocks have zero volume recently)
        recent_volume = df['Volume'].tail(5).sum()
        if recent_volume == 0:
            return {"error": f"No trading volume for {symbol} in last 5 days. Possibly delisted or suspended."}

        # Check if stock has reasonable minimum price (NSE stocks below ₹1 are penny/error)
        current_close = float(df['Close'].iloc[-1])
        if current_close < 1.0:
            return {"error": f"Suspiciously low price {current_close:.2f} for {symbol}. Possible data error or delisted."}

        # Detect gaps (missing trading days > 5 in a row = suspicious)
        if hasattr(df.index, 'to_series'):
            date_diffs = df.index.to_series().diff().dt.days.dropna()
            max_gap = date_diffs.max() if len(date_diffs) > 0 else 0
            if max_gap > 7:
                print(f"[MarketData] Warning: {symbol} has {max_gap}-day gap in data.")

        # ── Indicators ─────────────────────────────────────────────────────────
        df = add_all_indicators(df)

        # ── Weekly trend (Close vs 20-week SMA) ────────────────────────────────
        df_w = df.resample('W').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'})
        df_w['sma20'] = df_w['Close'].rolling(20).mean()
        weekly_trend = "UP" if df_w['Close'].iloc[-1] > df_w['sma20'].iloc[-1] else "DOWN"

        # ── Support / Resistance using pivot points ────────────────────────────
        from tools.indicators import pivot_support_resistance
        price = float(df['Close'].iloc[-1])
        support_levels, resistance_levels = pivot_support_resistance(df, n_pivots=3)

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

        # VWAP (last 20 days rolling)
        vwap_val = round(float(df['VWAP'].iloc[-1]), 2) if 'VWAP' in df.columns else 0.0
        vwap_dev = round(float(df['VWAP_DEV'].iloc[-1]), 2) if 'VWAP_DEV' in df.columns else 0.0

        # OBV trend (is OBV rising? Compare last 10 vs 20 bars back)
        obv_series = df['OBV'] if 'OBV' in df.columns else pd.Series(dtype=float)
        obv_trend = "RISING" if len(obv_series) >= 20 and float(obv_series.iloc[-1]) > float(obv_series.iloc[-20]) else "FALLING"

        # Divergence signals
        from tools.indicators import detect_rsi_divergence, detect_macd_divergence, momentum_6m, momentum_12m
        rsi_div = detect_rsi_divergence(df, lookback=20)
        macd_div = detect_macd_divergence(df, lookback=20)

        # Momentum factors (risk-adjusted)
        mom_6m = momentum_6m(df)
        mom_12m = momentum_12m(df)

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
            "vwap": vwap_val,
            "vwap_deviation_pct": vwap_dev,
            "obv_trend": obv_trend,
            "rsi_divergence": rsi_div,
            "macd_divergence": macd_div,
            "momentum_6m": mom_6m,
            "momentum_12m": mom_12m,
            "relative_strength_30d": rs_score,
            "weekly_trend": weekly_trend,
            "support_levels": support_levels,
            "resistance_levels": resistance_levels,
            "recent_candles": candles_table,
            "chart_path": chart_path,
        }

    def get_current_price(self, symbol: str) -> float:
        cache_key = f"price_{symbol}"
        cached = cache.get(cache_key)
        if cached:
            return cached
        price = 0.0
        if self.kite:
            try:
                quote = self.kite.quote(f"NSE:{symbol}")
                price = float(quote[f"NSE:{symbol}"]["last_price"])
            except Exception:
                pass
        if price <= 0:
            sym = f"{symbol}.NS"
            df = yf.Ticker(sym).history(period="1d")
            if not df.empty:
                price = round(float(df['Close'].iloc[-1]), 2)
        if price > 1.0:  # Minimum sanity check for NSE stocks
            cache.set(cache_key, price, TTL_PRICE)
        elif price > 0:
            print(f"[MarketData] Warning: {symbol} price {price} seems too low. Not caching.")
        return price

    def fetch_weekly_data(self, symbol: str) -> Dict[str, Any]:
        """Weekly OHLCV with key indicators for positional/value strategy confirmation."""
        cache_key = f"weekly_{symbol}"
        cached = cache.get(cache_key)
        if cached:
            return cached

        sym = f"{symbol}.NS" if not symbol.endswith(".NS") else symbol
        try:
            df = yf.Ticker(sym).history(period="2y", interval="1wk")
        except Exception as e:
            return {"error": str(e)}

        if df.empty or len(df) < 20:
            return {"error": f"Insufficient weekly data for {symbol}"}

        df = df.dropna()
        # EMA calculations on weekly
        df['EMA_20w'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['EMA_50w'] = df['Close'].ewm(span=50, adjust=False).mean()

        price = float(df['Close'].iloc[-1])
        ema20 = float(df['EMA_20w'].iloc[-1])
        ema50 = float(df['EMA_50w'].iloc[-1])

        # Weekly RSI
        delta = df['Close'].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        weekly_rsi = float(100 - 100 / (1 + rs.iloc[-1]))

        # Trend: price above both EMAs on weekly = strong uptrend
        if price > ema20 > ema50:
            weekly_structure = "STRONG_UP"
        elif price > ema50 and price <= ema20:
            weekly_structure = "PULLBACK_IN_UPTREND"   # Above 50w but below 20w — potential entry
        elif price > ema50:
            weekly_structure = "UP"
        elif price < ema20 < ema50:
            weekly_structure = "STRONG_DOWN"
        elif price < ema50 and price >= ema20:
            weekly_structure = "BOUNCE_IN_DOWNTREND"   # Below 50w but above 20w — watch for failure
        else:
            weekly_structure = "SIDEWAYS"

        # 52-week high/low
        high_52w = float(df['High'].tail(52).max())
        low_52w = float(df['Low'].tail(52).min())
        pct_from_high = round((price - high_52w) / high_52w * 100, 2)

        result = {
            "symbol": symbol,
            "weekly_price": round(price, 2),
            "weekly_ema_20": round(ema20, 2),
            "weekly_ema_50": round(ema50, 2),
            "weekly_rsi": round(weekly_rsi, 1),
            "weekly_structure": weekly_structure,
            "high_52w": round(high_52w, 2),
            "low_52w": round(low_52w, 2),
            "pct_from_52w_high": pct_from_high,
        }
        cache.set(cache_key, result, TTL_TECHNICALS)
        return result

    def fetch_monthly_data(self, symbol: str) -> Dict[str, Any]:
        """Monthly OHLCV for value and positional long-term trend assessment."""
        cache_key = f"monthly_{symbol}"
        cached = cache.get(cache_key)
        if cached:
            return cached

        sym = f"{symbol}.NS" if not symbol.endswith(".NS") else symbol
        try:
            df = yf.Ticker(sym).history(period="5y", interval="1mo")
        except Exception as e:
            return {"error": str(e)}

        if df.empty or len(df) < 12:
            return {"error": f"Insufficient monthly data for {symbol}"}

        df = df.dropna()
        df['EMA_12m'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_24m'] = df['Close'].ewm(span=24, adjust=False).mean()

        price = float(df['Close'].iloc[-1])
        ema12 = float(df['EMA_12m'].iloc[-1])
        ema24 = float(df['EMA_24m'].iloc[-1])

        # Monthly momentum: 3-month, 6-month, 12-month, and 24-month returns
        ret_3m = round((price - float(df['Close'].iloc[-3])) / float(df['Close'].iloc[-3]) * 100, 2) if len(df) >= 3 else 0.0
        ret_12m = round((price - float(df['Close'].iloc[-12])) / float(df['Close'].iloc[-12]) * 100, 2) if len(df) >= 12 else 0.0
        ret_6m = round((price - float(df['Close'].iloc[-6])) / float(df['Close'].iloc[-6]) * 100, 2) if len(df) >= 6 else 0.0
        ret_24m = round((price - float(df['Close'].iloc[-24])) / float(df['Close'].iloc[-24]) * 100, 2) if len(df) >= 24 else 0.0

        monthly_trend = "UP" if price > ema12 > ema24 else "DOWN" if price < ema24 else "NEUTRAL"

        result = {
            "symbol": symbol,
            "monthly_price": round(price, 2),
            "monthly_ema_12": round(ema12, 2),
            "monthly_ema_24": round(ema24, 2),
            "monthly_trend": monthly_trend,
            "return_3m_pct": ret_3m,
            "return_6m_pct": ret_6m,
            "return_12m_pct": ret_12m,
            "return_24m_pct": ret_24m,
        }
        cache.set(cache_key, result, TTL_TECHNICALS)
        return result

    def get_multi_timeframe_context(self, symbol: str) -> Dict[str, Any]:
        """Returns combined daily + weekly + monthly context for multi-timeframe analysis."""
        daily = self.fetch_advanced_technicals(symbol)
        weekly = self.fetch_weekly_data(symbol)
        monthly = self.fetch_monthly_data(symbol)

        # Confluence score: how many timeframes agree on direction
        timeframe_signals = []
        if daily.get("weekly_trend") == "UP":
            timeframe_signals.append(1)
        if weekly.get("weekly_structure") in ("UP", "STRONG_UP", "PULLBACK_IN_UPTREND"):
            timeframe_signals.append(1)
        if monthly.get("monthly_trend") == "UP":
            timeframe_signals.append(1)

        confluence_score = len(timeframe_signals)  # 0-3: 3 = all timeframes aligned

        return {
            "daily": daily,
            "weekly": weekly,
            "monthly": monthly,
            "timeframe_confluence": confluence_score,
            "all_timeframes_aligned": confluence_score == 3,
        }


market_data_tool = MarketDataTool()
