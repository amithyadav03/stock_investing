"""
Pure-pandas technical indicator calculations.
Drop-in replacement for pandas_ta — no C extensions needed.
All functions accept a DataFrame and return a Series with a consistent column name.
"""

import numpy as np
import pandas as pd


def ema(df: pd.DataFrame, length: int = 20, col: str = "Close") -> pd.Series:
    return df[col].ewm(span=length, adjust=False).mean()


def sma(df: pd.DataFrame, length: int = 20, col: str = "Close") -> pd.Series:
    return df[col].rolling(window=length).mean()


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False).mean()


def rsi(df: pd.DataFrame, length: int = 14, col: str = "Close") -> pd.Series:
    delta = df[col].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9, col: str = "Close"):
    """Returns (macd_line, signal_line, histogram) as three Series."""
    ema_fast = df[col].ewm(span=fast, adjust=False).mean()
    ema_slow = df[col].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def bbands(df: pd.DataFrame, length: int = 20, std: float = 2.0, col: str = "Close"):
    """Returns (upper, middle, lower, pct_b) as four Series."""
    middle = df[col].rolling(length).mean()
    std_dev = df[col].rolling(length).std()
    upper = middle + std * std_dev
    lower = middle - std * std_dev
    pct_b = (df[col] - lower) / (upper - lower + 1e-10)
    return upper, middle, lower, pct_b


def stoch(df: pd.DataFrame, k: int = 14, d: int = 3):
    """Returns (%K, %D) as two Series."""
    low_min = df["Low"].rolling(k).min()
    high_max = df["High"].rolling(k).max()
    k_pct = 100 * (df["Close"] - low_min) / (high_max - low_min + 1e-10)
    d_pct = k_pct.rolling(d).mean()
    return k_pct, d_pct


def adx(df: pd.DataFrame, length: int = 14):
    """Returns (ADX, +DI, -DI) as three Series."""
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)

    up_move = high - prev_high
    down_move = prev_low - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr_s = tr.ewm(alpha=1 / length, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / length, adjust=False).mean() / atr_s
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / length, adjust=False).mean() / atr_s
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx_s = dx.ewm(alpha=1 / length, adjust=False).mean()
    return adx_s, plus_di, minus_di


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Appends all standard indicator columns to a copy of the DataFrame.
    Column names match the keys used throughout the codebase.
    """
    df = df.copy()

    df["ATRr_14"] = atr(df, 14)
    df["RSI_14"] = rsi(df, 14)

    macd_l, macd_s, macd_h = macd(df, 12, 26, 9)
    df["MACD_12_26_9"] = macd_l
    df["MACDs_12_26_9"] = macd_s
    df["MACDh_12_26_9"] = macd_h

    bb_u, bb_m, bb_l, bb_p = bbands(df, 20, 2.0)
    df["BBU_20_2.0"] = bb_u
    df["BBM_20_2.0"] = bb_m
    df["BBL_20_2.0"] = bb_l
    df["BBP_20_2.0"] = bb_p

    adx_s, plus_di, minus_di = adx(df, 14)
    df["ADX_14"] = adx_s
    df["DMP_14"] = plus_di
    df["DMN_14"] = minus_di

    stoch_k, stoch_d = stoch(df, 14, 3)
    df["STOCHk_14_3_3"] = stoch_k
    df["STOCHd_14_3_3"] = stoch_d

    df["EMA_20"] = ema(df, 20)
    df["EMA_50"] = ema(df, 50)
    df["EMA_200"] = ema(df, 200)
    df["VWAP"] = vwap(df)
    df["VWAP_DEV"] = vwap_deviation(df)
    df["OBV"] = obv(df)
    return df


def vwap(df: pd.DataFrame) -> pd.Series:
    """Volume-Weighted Average Price — daily session VWAP."""
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    cum_tp_vol = (typical_price * df['Volume']).cumsum()
    cum_vol = df['Volume'].cumsum()
    return cum_tp_vol / (cum_vol + 1e-10)


def vwap_deviation(df: pd.DataFrame) -> pd.Series:
    """% deviation of close from VWAP. Positive = above VWAP (bullish bias)."""
    vwap_s = vwap(df)
    return ((df['Close'] - vwap_s) / vwap_s * 100).round(2)


def obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume — cumulative volume flow."""
    direction = np.sign(df['Close'].diff().fillna(0))
    return (direction * df['Volume']).cumsum()


def momentum_6m(df: pd.DataFrame) -> float:
    """6-month price momentum (risk-adjusted, like NSE Momentum Index)."""
    if len(df) < 126:
        return 0.0
    ret = (df['Close'].iloc[-1] - df['Close'].iloc[-126]) / df['Close'].iloc[-126]
    vol = df['Close'].pct_change().tail(126).std()
    return round(ret / (vol + 1e-10), 2)  # Risk-adjusted momentum


def momentum_12m(df: pd.DataFrame) -> float:
    """12-month price momentum (risk-adjusted)."""
    if len(df) < 252:
        return 0.0
    ret = (df['Close'].iloc[-1] - df['Close'].iloc[-252]) / df['Close'].iloc[-252]
    vol = df['Close'].pct_change().tail(252).std()
    return round(ret / (vol + 1e-10), 2)


def detect_rsi_divergence(df: pd.DataFrame, lookback: int = 20) -> str:
    """
    Detects RSI divergence over the last `lookback` bars.
    Returns: 'BULLISH_DIV', 'BEARISH_DIV', or 'NONE'

    Bullish divergence: price makes lower low but RSI makes higher low.
    Bearish divergence: price makes higher high but RSI makes lower high.
    """
    if 'RSI_14' not in df.columns or len(df) < lookback + 5:
        return 'NONE'

    recent = df.tail(lookback)
    price = recent['Close']
    rsi_s = recent['RSI_14']

    # Find price lows and highs
    price_min_idx = price.idxmin()
    price_max_idx = price.idxmax()

    # Check for bullish divergence: lower price low but higher RSI low
    # Compare last bar to the lowest point
    last_price = price.iloc[-1]
    last_rsi = rsi_s.iloc[-1]
    min_price = price.min()
    min_rsi_at_price_min = rsi_s.loc[price_min_idx] if price_min_idx in rsi_s.index else rsi_s.min()

    if last_price <= min_price * 1.02 and last_rsi > min_rsi_at_price_min + 3:
        return 'BULLISH_DIV'

    max_price = price.max()
    max_rsi_at_price_max = rsi_s.loc[price_max_idx] if price_max_idx in rsi_s.index else rsi_s.max()

    if last_price >= max_price * 0.98 and last_rsi < max_rsi_at_price_max - 3:
        return 'BEARISH_DIV'

    return 'NONE'


def detect_macd_divergence(df: pd.DataFrame, lookback: int = 20) -> str:
    """
    Detects MACD histogram divergence.
    Returns: 'BULLISH_DIV', 'BEARISH_DIV', or 'NONE'
    """
    macd_h_col = next((c for c in df.columns if 'MACDh' in c), None)
    if not macd_h_col or len(df) < lookback + 5:
        return 'NONE'

    recent = df.tail(lookback)
    price = recent['Close']
    hist = recent[macd_h_col]

    price_min_idx = price.idxmin()
    price_max_idx = price.idxmax()

    last_price = price.iloc[-1]
    last_hist = hist.iloc[-1]

    min_price = price.min()
    hist_at_price_min = hist.loc[price_min_idx] if price_min_idx in hist.index else hist.min()

    if last_price <= min_price * 1.02 and last_hist > hist_at_price_min + 0.01 and last_hist < 0:
        return 'BULLISH_DIV'

    max_price = price.max()
    hist_at_price_max = hist.loc[price_max_idx] if price_max_idx in hist.index else hist.max()

    if last_price >= max_price * 0.98 and last_hist < hist_at_price_max - 0.01 and last_hist > 0:
        return 'BEARISH_DIV'

    return 'NONE'


def pivot_support_resistance(df: pd.DataFrame, n_pivots: int = 3):
    """
    Proper pivot-based support and resistance levels.
    Uses local minima/maxima over the lookback period, weighted by volume.
    Returns (support_levels, resistance_levels) as sorted lists.
    """
    price = float(df['Close'].iloc[-1])

    # Find swing highs and lows using a simple local extrema approach
    highs = []
    lows = []
    window = 10  # bars each side

    for i in range(window, len(df) - window):
        bar_high = float(df['High'].iloc[i])
        bar_low = float(df['Low'].iloc[i])
        vol = float(df['Volume'].iloc[i])

        # Swing high: higher than surrounding bars
        if bar_high == df['High'].iloc[i-window:i+window+1].max():
            highs.append((bar_high, vol))
        # Swing low: lower than surrounding bars
        if bar_low == df['Low'].iloc[i-window:i+window+1].min():
            lows.append((bar_low, vol))

    # Also include 52-week high/low, round numbers
    high_52w = float(df['High'].tail(252).max())
    low_52w = float(df['Low'].tail(252).min())
    highs.append((high_52w, 1))
    lows.append((low_52w, 1))

    # Filter relevant levels (within 20% of current price)
    support = sorted(set([
        round(level, 2) for level, _ in lows
        if 0.80 * price <= level < price
    ]), reverse=True)[:n_pivots]

    resistance = sorted(set([
        round(level, 2) for level, _ in highs
        if price < level <= 1.20 * price
    ]))[:n_pivots]

    return support, resistance
