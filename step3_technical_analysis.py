#!/usr/bin/env python3
"""
================================================================================
STEP 3: SPY Technical Analysis (Daily Only)
================================================================================

Pure technical analysis for SPY daily data - NO SCORING.
Calculates all technical indicators, patterns, and levels as RAW DATA ONLY.

Author: SignalsAlpha
Version: 1.0 (SPY Edition)
Date: 2026-04-29

================================================================================
PURPOSE
================================================================================

This module performs pure technical analysis on SPY daily data:
1. Calculates trend indicators (SMA, EMA, ADX)
2. Computes momentum oscillators (RSI, MACD, Stochastic, MFI, TRIX)
3. Generates volatility measures (Bollinger Bands, ATR)
4. Detects crossover events with strength scoring
5. Identifies candlestick patterns
6. Calculates support/resistance levels
7. Computes Fibonacci retracement levels
8. Hull Moving Average (HMA) with slope and turn detection
9. Elder Impulse System

Output: Raw technical data for SPY daily data
Next Step: Step 4 (Scoring & Ranking)

================================================================================
INPUT
================================================================================

Source: data/time_series/SPY/SPY_1d.csv
Columns: date, open, high, low, close, volume

================================================================================
OUTPUT
================================================================================

Destination: data/technical_analysis/SPY/SPY_1d_technical.csv

"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict
import json

# Configuration
TICKER = "SPY"
INTERVAL = "1d"
DATA_DIR = Path("data")
INPUT_FILE = DATA_DIR / "time_series" / TICKER / f"{TICKER}_{INTERVAL}.csv"
OUTPUT_DIR = DATA_DIR / "technical_analysis" / TICKER
OUTPUT_FILE = OUTPUT_DIR / f"{TICKER}_{INTERVAL}_technical.csv"


def calculate_sma(data: pd.Series, period: int) -> pd.Series:
    """Calculate Simple Moving Average."""
    return data.rolling(window=period).mean()


def calculate_ema(data: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average."""
    return data.ewm(span=period, adjust=False).mean()


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index."""
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
    """Calculate MACD, signal line, and histogram."""
    ema_fast = calculate_ema(close, fast)
    ema_slow = calculate_ema(close, slow)
    macd = ema_fast - ema_slow
    macd_signal = calculate_ema(macd, signal)
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist


def calculate_bollinger_bands(close: pd.Series, period: int = 20, std_dev: float = 2.0) -> tuple:
    """Calculate Bollinger Bands."""
    sma = calculate_sma(close, period)
    std = close.rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, sma, lower


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Average True Range."""
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    return atr


def calculate_hma(close: pd.Series, period: int = 13) -> pd.Series:
    """Calculate Hull Moving Average."""
    half_period = int(period / 2)
    sqrt_period = int(np.sqrt(period))
    
    wma_half = close.rolling(window=half_period).apply(
        lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1)), raw=True
    )
    wma_full = close.rolling(window=period).apply(
        lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1)), raw=True
    )
    
    hma = (2 * wma_half - wma_full).rolling(window=sqrt_period).apply(
        lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1)), raw=True
    )
    
    return hma


def calculate_rolling_high_low(high: pd.Series, low: pd.Series, period: int = 252) -> tuple:
    """Calculate rolling high/low (e.g., 52-week)."""
    rolling_high = high.rolling(window=period, min_periods=1).max()
    rolling_low = low.rolling(window=period, min_periods=1).min()
    return rolling_high, rolling_low


def calculate_elder_impulse(close: pd.Series, macd_hist: pd.Series, ema_period: int = 13) -> pd.Series:
    """Calculate Elder Impulse System."""
    ema = calculate_ema(close, ema_period)
    ema_slope = ema.diff()
    hist_slope = macd_hist.diff()
    
    impulse = pd.Series(index=close.index, dtype='object')
    impulse[:] = 'blue'  # Default
    
    # Green: rising EMA + rising histogram
    impulse[(ema_slope > 0) & (hist_slope > 0)] = 'green'
    # Red: falling EMA + falling histogram
    impulse[(ema_slope < 0) & (hist_slope < 0)] = 'red'
    
    return impulse


def generate_technical_analysis():
    """Generate technical analysis for SPY daily data."""
    print("=" * 70)
    print("STEP 3: SPY Technical Analysis (Daily)")
    print("=" * 70)
    
    # Check input file exists
    if not INPUT_FILE.exists():
        print(f"ERROR: Input file not found: {INPUT_FILE}")
        print("Run Step 1 first to collect time series data.")
        return
    
    # Load data
    print(f"\nLoading data from: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    print(f"Loaded {len(df)} rows")
    print(f"Date range: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")
    
    # Calculate indicators
    print("\nCalculating technical indicators...")
    
    # Trend indicators
    df['sma_20'] = calculate_sma(df['close'], 20)
    df['sma_50'] = calculate_sma(df['close'], 50)
    df['sma_200'] = calculate_sma(df['close'], 200)  # NEW: 200-day SMA
    df['ema_12'] = calculate_ema(df['close'], 12)
    df['ema_26'] = calculate_ema(df['close'], 26)
    
    # Momentum
    df['rsi'] = calculate_rsi(df['close'], 14)
    df['macd'], df['macd_signal'], df['macd_hist'] = calculate_macd(df['close'])
    
    # Volatility
    df['bb_upper'], df['bb_middle'], df['bb_lower'] = calculate_bollinger_bands(df['close'])
    df['atr'] = calculate_atr(df['high'], df['low'], df['close'], 14)
    
    # HMA and Elder Impulse
    df['hma_13'] = calculate_hma(df['close'], 13)
    df['elder_impulse'] = calculate_elder_impulse(df['close'], df['macd_hist'])
    
    # NEW: 52-week high/low (252 trading days)
    df['high_52w'], df['low_52w'] = calculate_rolling_high_low(df['high'], df['low'], 252)
    
    # NEW: Distance from 52-week high (as percentage)
    df['pct_from_52w_high'] = (df['close'] - df['high_52w']) / df['high_52w'] * 100
    df['pct_from_52w_low'] = (df['close'] - df['low_52w']) / df['low_52w'] * 100
    
    # Reset index to make date a column
    df.reset_index(inplace=True)
    
    # Save output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\nSaved to: {OUTPUT_FILE}")
    print(f"Total rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    
    # Print latest values
    latest = df.iloc[-1]
    print("\n" + "=" * 70)
    print("Latest Technical Values")
    print("=" * 70)
    print(f"Date: {latest['date'].strftime('%Y-%m-%d')}")
    print(f"Close: ${latest['close']:.2f}")
    print(f"RSI: {latest['rsi']:.1f}")
    print(f"MACD: {latest['macd']:.3f}")
    print(f"SMA 20: ${latest['sma_20']:.2f}")
    print(f"SMA 50: ${latest['sma_50']:.2f}")
    print(f"SMA 200: ${latest['sma_200']:.2f}")
    print(f"HMA 13: ${latest['hma_13']:.2f}")
    print(f"Elder Impulse: {latest['elder_impulse']}")
    print(f"\n52-Week Range:")
    print(f"  High: ${latest['high_52w']:.2f}")
    print(f"  Low: ${latest['low_52w']:.2f}")
    print(f"  From High: {latest['pct_from_52w_high']:.1f}%")
    print(f"  From Low: {latest['pct_from_52w_low']:.1f}%")
    print("=" * 70)


if __name__ == "__main__":
    generate_technical_analysis()
