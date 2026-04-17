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

    return df
