#!/usr/bin/env python3
"""
================================================================================
STEP 3: Technical Analysis & Signal Generation
================================================================================

Pure technical analysis engine - NO SCORING, NO BUY/SELL RECOMMENDATIONS
Calculates all technical indicators, patterns, and levels as RAW DATA ONLY.

This module performs pure technical analysis without any scoring logic.
The scoring and final signal generation happens in Step 4.

Author: SignalsAlpha
Version: 1.0
Date: 2026-04-20

================================================================================
PURPOSE
================================================================================

This module serves as the technical analysis foundation:

1. Calculates trend indicators (SMA, EMA, ADX)
2. Computes momentum oscillators (RSI, MACD, Stochastic, MFI, TRIX)
3. Generates volatility measures (Bollinger Bands, ATR)
4. Detects crossover events with strength and location scoring
5. Identifies candlestick patterns (14 patterns)
6. Calculates support and resistance levels with proximity metrics
7. Computes Fibonacci retracement levels
8. Provides volume analysis (VWAP, OBV, volume ratio)
9. Generates price targets (stop_loss, take_profit) based on ATR
10. Hull Moving Average (HMA) with slope, turn, and peak/valley detection
11. Elder Impulse System (trend + momentum color-coded signals)

Output: Raw technical data for each ticker/interval combination
Next Step: Step 4 (Scoring & Ranking) combines this with fundamental data

================================================================================
INPUT
================================================================================

Source: data/time_series/{TICKER}/{TICKER}_{interval}.csv
Columns: Date, Open, High, Low, Close, Volume

================================================================================
OUTPUT
================================================================================

Destination: data/technical_analysis/{TICKER}/{TICKER}_{interval}_technical.csv

Columns Added:
    Trend Indicators:
        - sma_20, sma_50: Simple Moving Averages
        - ema_12, ema_26: Exponential Moving Averages
        - adx: Average Directional Index (trend strength)
        - market_regime: TRENDING_UP/TRENDING_DOWN/RANGING/NEUTRAL
    
    Momentum Oscillators:
        - rsi: Relative Strength Index (14-period)
        - macd, macd_signal, macd_hist: MACD components
        - macd_cross_type: crossover/crossunder/none
        - macd_cross_strength: 0-5 (strength of signal)
        - stoch_k_slow, stoch_d_slow: Stochastic Slow %K/%D
        - stoch_k_fast, stoch_d_fast: Stochastic Fast %K/%D
        - stoch_slow_cross: crossover/crossunder/none
        - stoch_fast_cross: crossover/crossunder/none
        - mfi: Money Flow Index (14-period)
        - trix, trix_signal: TRIX and signal line
        - trix_cross_type: crossover/crossunder/none
    
    Volatility:
        - bb_upper, bb_middle, bb_lower: Bollinger Bands
        - bb_percent: %B indicator
        - atr: Average True Range (14-period)
    
    Volume:
        - vwap: Volume Weighted Average Price
        - obv: On-Balance Volume
        - volume_ratio: Current vs 20-period average
    
    Support/Resistance:
        - support_levels: JSON array of S/R levels
        - support_distance_pct: Distance to nearest support
        - resistance_distance_pct: Distance to nearest resistance
    
    Fibonacci:
        - fib_23_6, fib_38_2, fib_50_0, fib_61_8, fib_78_6: Retracement levels
    
    Candlestick Patterns:
        - pattern_doji, pattern_hammer, pattern_shooting_star
        - pattern_engulfing, pattern_morning_star, pattern_evening_star
        - pattern_harami, pattern_piercing, pattern_dark_cloud
        - pattern_three_white_soldiers, pattern_three_black_crows
        - pattern_spinning_top, pattern_marubozu
    
    Price Targets:
        - stop_loss: ATR-based stop loss level
        - take_profit: Risk-reward based take profit
    
    Hull Moving Average (HMA):
        - hma_13: Hull Moving Average value (13-period)
        - hma_slope_2bar: 2-bar slope percentage
        - hma_turn: up/down/none (direction change detection)
        - hma_peak_valley: peak/valley/none (local extrema)
        - hma_signal: cross_above/cross_below/above/below
    
    Elder Impulse System:
        - elder_impulse: green/red/blue (trend + momentum)
        - elder_trend_strength: 0-10 scale (conviction level)

================================================================================
TECHNICAL DETAILS
================================================================================

Market Regime Detection:
    Uses ADX (Average Directional Index) to classify market conditions:
    - ADX >= 25: Trending market
    - ADX < 20: Ranging market
    - Between 20-25: Neutral

Crossover Detection:
    Monitors MACD, Stochastic Slow/Fast, and TRIX for crossover events.
    Each crossover includes:
    - Type: crossover (bullish) or crossunder (bearish)
    - Strength: 0-5 based on location (deeper cross = higher strength)
    - Signal line values for confirmation

Support/Resistance Calculation:
    Uses pivot-based clustering algorithm:
    1. Identifies pivot highs/lows over 20-period window
    2. Clusters nearby levels using tolerance band
    3. Filters to most significant levels by touch count
    4. Calculates distance to nearest support/resistance

Fibonacci Retracements:
    Calculated from 20-period high/low range:
    - 23.6%, 38.2%, 50.0%, 61.8%, 78.6% levels
    Based on Golden Ratio principles for reversal zones

Candlestick Pattern Detection:
    Uses body/range ratios and OHLC relationships:
    - Doji: Body < 5% of range
    - Hammer: Lower shadow > 2x body, upper shadow < body
    - Engulfing: Current body completely engulfs previous
    - Stars: Gaps with small bodies
    - Soldiers/Crows: Three consecutive long bodies

"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional

# Configuration
DATA_DIR = Path("/home/ubuntu/.openclaw/workspace/trading-signals-data/process_flow_v1/data")
TIME_SERIES_DIR = DATA_DIR / "time_series"
OUTPUT_DIR = DATA_DIR / "technical_analysis"
PROGRESS_FILE = DATA_DIR / ".step3_progress"
INTERVALS = ['1d', '1wk', '1mo']

# Batch processing configuration
BATCH_SIZE = 5  # Process 5 tickers per batch (smaller for OCI)
MAX_BATCHES = 5000  # Process all tickers in one run

ADX_TREND_THRESHOLD = 25
ADX_RANGE_THRESHOLD = 20

# ============================================================================
# CORE INDICATORS
# ============================================================================

def calculate_sma(data, period):
    return data.rolling(window=period, min_periods=period).mean()

def calculate_ema(data, period):
    return data.ewm(span=period, adjust=False, min_periods=period).mean()

def calculate_rsi(data, period=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def calculate_macd(data, fast=12, slow=26, signal=9):
    ema_fast = calculate_ema(data, fast)
    ema_slow = calculate_ema(data, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_bollinger_bands(data, period=20, std_dev=2):
    sma = calculate_sma(data, period)
    std = data.rolling(window=period, min_periods=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, sma, lower

def calculate_atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.rolling(window=period, min_periods=period).mean()
    return atr

def calculate_vwap(df):
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
    return vwap

def calculate_obv(df):
    obv = (np.sign(df['close'].diff()) * df['volume']).cumsum()
    return obv

def calculate_stochastic(df, k_period=14, d_period=3, slowing=3, stoch_type='slow'):
    lowest_low = df['low'].rolling(window=k_period, min_periods=k_period).min()
    highest_high = df['high'].rolling(window=k_period, min_periods=k_period).max()
    k_fast = 100 * ((df['close'] - lowest_low) / (highest_high - lowest_low).replace(0, np.nan))
    k = k_fast.rolling(window=slowing, min_periods=slowing).mean()
    d = k.rolling(window=d_period, min_periods=d_period).mean()
    return k, d

def calculate_mfi(df, period=14):
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    raw_money_flow = typical_price * df['volume']
    money_flow_plus = raw_money_flow.where(typical_price > typical_price.shift(1), 0)
    money_flow_minus = abs(raw_money_flow.where(typical_price < typical_price.shift(1), 0))
    positive_sum = money_flow_plus.rolling(window=period, min_periods=period).sum()
    negative_sum = money_flow_minus.rolling(window=period, min_periods=period).sum()
    money_ratio = positive_sum / negative_sum.replace(0, np.nan)
    mfi = 100 - (100 / (1 + money_ratio))
    return mfi.fillna(50)

def calculate_trix(data, period=15):
    single_ema = calculate_ema(data, period)
    double_ema = calculate_ema(single_ema, period)
    triple_ema = calculate_ema(double_ema, period)
    trix = 100 * (triple_ema - triple_ema.shift(1)) / triple_ema.shift(1).replace(0, np.nan)
    trix_signal = calculate_ema(trix, 9)
    return trix, trix_signal

def calculate_adx(high, low, close, period=14):
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    atr = tr.rolling(window=period, min_periods=period).mean()
    plus_di = 100 * (plus_dm.rolling(window=period, min_periods=period).mean() / atr.replace(0, np.nan))
    minus_di = 100 * (minus_dm.rolling(window=period, min_periods=period).mean() / atr.replace(0, np.nan))
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)) * 100
    adx = dx.rolling(window=period, min_periods=period).mean()
    return adx.fillna(20), plus_di.fillna(50), minus_di.fillna(50)

# ============================================================================
# CROSSOVER DETECTION
# ============================================================================

def detect_crossover(line1, line2, lookback=2):
    if len(line1) < lookback + 1 or len(line2) < lookback + 1:
        return None, 0, 'insufficient_data'
    recent1 = line1.iloc[-lookback-1:]
    recent2 = line2.iloc[-lookback-1:]
    if recent1.isna().any() or recent2.isna().any():
        return None, 0, 'invalid_data'
    prev_diff = recent1.iloc[-2] - recent2.iloc[-2]
    curr_diff = recent1.iloc[-1] - recent2.iloc[-1]
    avg_before = recent1.iloc[:-1].mean()
    if prev_diff < 0 and curr_diff > 0:
        if avg_before < -0.5:
            return 'crossover', 8, 'deeply_oversold'
        elif avg_before < -0.2:
            return 'crossover', 6, 'oversold'
        elif avg_before < 0:
            return 'crossover', 4, 'below_zero'
        else:
            return 'crossover', 2, 'near_zero'
    elif prev_diff > 0 and curr_diff < 0:
        if avg_before > 0.5:
            return 'crossunder', 8, 'deeply_overbought'
        elif avg_before > 0.2:
            return 'crossunder', 6, 'overbought'
        elif avg_before > 0:
            return 'crossunder', 4, 'above_zero'
        else:
            return 'crossunder', 2, 'near_zero'
    return None, 0, 'no_cross'

def detect_stochastic_crossover(k_line, d_line, lookback=2):
    if len(k_line) < lookback + 1 or len(d_line) < lookback + 1:
        return None, 0, 'insufficient_data'
    recent_k = k_line.iloc[-lookback-1:]
    recent_d = d_line.iloc[-lookback-1:]
    if recent_k.isna().any() or recent_d.isna().any():
        return None, 0, 'invalid_data'
    prev_k, curr_k = recent_k.iloc[-2], recent_k.iloc[-1]
    prev_d, curr_d = recent_d.iloc[-2], recent_d.iloc[-1]
    if prev_k < prev_d and curr_k > curr_d:
        if prev_k < 20 and prev_d < 20:
            return 'crossover', 10, 'deeply_oversold'
        elif prev_k < 30 or prev_d < 30:
            return 'crossover', 7, 'oversold'
        elif prev_k < 50:
            return 'crossover', 4, 'below_midpoint'
        else:
            return 'crossover', 2, 'above_midpoint'
    elif prev_k > prev_d and curr_k < curr_d:
        if prev_k > 80 and prev_d > 80:
            return 'crossunder', 10, 'deeply_overbought'
        elif prev_k > 70 or prev_d > 70:
            return 'crossunder', 7, 'overbought'
        elif prev_k > 50:
            return 'crossunder', 4, 'above_midpoint'
        else:
            return 'crossunder', 2, 'below_midpoint'
    return None, 0, 'no_cross'


# ============================================================================
# CANDLESTICK PATTERNS
# ============================================================================

def get_candle_body(open_p, close_p):
    return abs(close_p - open_p)

def get_candle_range(high, low):
    return high - low

def get_upper_shadow(open_p, close_p, high):
    return high - max(open_p, close_p)

def get_lower_shadow(open_p, close_p, low):
    return min(open_p, close_p) - low

def is_bullish(open_p, close_p):
    return close_p > open_p

def is_bearish(open_p, close_p):
    return close_p < open_p

def detect_doji(open_p, close_p, high, low, threshold=0.1):
    body = get_candle_body(open_p, close_p)
    range_c = get_candle_range(high, low)
    if range_c == 0:
        return False
    return (body / range_c) < threshold

def detect_hammer(open_p, close_p, high, low):
    body = get_candle_body(open_p, close_p)
    range_c = get_candle_range(high, low)
    lower_shadow = get_lower_shadow(open_p, close_p, low)
    upper_shadow = get_upper_shadow(open_p, close_p, high)
    if range_c == 0 or body == 0:
        return False
    body_small = body < (range_c * 0.3)
    long_lower = lower_shadow > (body * 2.0)
    small_upper = upper_shadow < (body * 0.5)
    return body_small and long_lower and small_upper

def detect_shooting_star(open_p, close_p, high, low):
    body = get_candle_body(open_p, close_p)
    range_c = get_candle_range(high, low)
    upper_shadow = get_upper_shadow(open_p, close_p, high)
    lower_shadow = get_lower_shadow(open_p, close_p, low)
    if range_c == 0 or body == 0:
        return False
    body_small = body < (range_c * 0.3)
    long_upper = upper_shadow > (body * 2.0)
    small_lower = lower_shadow < (body * 0.5)
    return body_small and long_upper and small_lower

def detect_bullish_engulfing(prev_open, prev_close, curr_open, curr_close):
    if not is_bearish(prev_open, prev_close) or not is_bullish(curr_open, curr_close):
        return False
    return (curr_close >= prev_open) and (curr_open <= prev_close)

def detect_bearish_engulfing(prev_open, prev_close, curr_open, curr_close):
    if not is_bullish(prev_open, prev_close) or not is_bearish(curr_open, curr_close):
        return False
    return (curr_open >= prev_close) and (curr_close <= prev_open)

def detect_morning_star(c1_open, c1_close, c2_open, c2_close, c3_open, c3_close):
    if not is_bearish(c1_open, c1_close) or not is_bullish(c3_open, c3_close):
        return False
    c1_body = get_candle_body(c1_open, c1_close)
    c2_body = get_candle_body(c2_open, c2_close)
    c3_body = get_candle_body(c3_open, c3_close)
    if c1_body == 0 or c3_body == 0:
        return False
    if c2_body > c1_body * 0.3:
        return False
    c1_mid = (c1_open + c1_close) / 2
    return c3_close >= c1_mid

def detect_evening_star(c1_open, c1_close, c2_open, c2_close, c3_open, c3_close):
    if not is_bullish(c1_open, c1_close) or not is_bearish(c3_open, c3_close):
        return False
    c1_body = get_candle_body(c1_open, c1_close)
    c2_body = get_candle_body(c2_open, c2_close)
    c3_body = get_candle_body(c3_open, c3_close)
    if c1_body == 0 or c3_body == 0:
        return False
    if c2_body > c1_body * 0.3:
        return False
    c1_mid = (c1_open + c1_close) / 2
    return c3_close <= c1_mid

def analyze_candlestick_patterns(df):
    if len(df) < 3:
        return []
    patterns = []
    c1 = df.iloc[-3]
    c2 = df.iloc[-2]
    c3 = df.iloc[-1]
    if detect_doji(c3['open'], c3['close'], c3['high'], c3['low']):
        patterns.append('DOJI')
    if detect_hammer(c3['open'], c3['close'], c3['high'], c3['low']):
        if c3['close'] < df['close'].mean():
            patterns.append('HAMMER')
        else:
            patterns.append('HANGING_MAN')
    if detect_shooting_star(c3['open'], c3['close'], c3['high'], c3['low']):
        if c3['close'] > df['close'].mean():
            patterns.append('SHOOTING_STAR')
        else:
            patterns.append('INVERTED_HAMMER')
    if detect_bullish_engulfing(c2['open'], c2['close'], c3['open'], c3['close']):
        patterns.append('BULLISH_ENGULFING')
    if detect_bearish_engulfing(c2['open'], c2['close'], c3['open'], c3['close']):
        patterns.append('BEARISH_ENGULFING')
    if detect_morning_star(c1['open'], c1['close'], c2['open'], c2['close'], c3['open'], c3['close']):
        patterns.append('MORNING_STAR')
    if detect_evening_star(c1['open'], c1['close'], c2['open'], c2['close'], c3['open'], c3['close']):
        patterns.append('EVENING_STAR')
    return patterns


# ============================================================================
# SUPPORT/RESISTANCE
# ============================================================================

def find_pivot_highs(df, window=5):
    highs = df['high']
    pivot_highs = []
    for i in range(window, len(highs) - window):
        if highs.iloc[i] == highs.iloc[i-window:i+window+1].max():
            pivot_highs.append({'index': i, 'price': round(highs.iloc[i], 2)})
    return pivot_highs

def find_pivot_lows(df, window=5):
    lows = df['low']
    pivot_lows = []
    for i in range(window, len(lows) - window):
        if lows.iloc[i] == lows.iloc[i-window:i+window+1].min():
            pivot_lows.append({'index': i, 'price': round(lows.iloc[i], 2)})
    return pivot_lows

def calculate_fibonacci_retracement(high, low):
    diff = high - low
    return {
        '0.0%': round(high, 2),
        '23.6%': round(high - (diff * 0.236), 2),
        '38.2%': round(high - (diff * 0.382), 2),
        '50.0%': round(high - (diff * 0.5), 2),
        '61.8%': round(high - (diff * 0.618), 2),
        '78.6%': round(high - (diff * 0.786), 2),
        '100.0%': round(low, 2)
    }

# ============================================================================
# HULL MOVING AVERAGE (HMA)
# ============================================================================

def calculate_wma(data, period):
    """Calculate Weighted Moving Average"""
    weights = np.arange(1, period + 1)
    return data.rolling(window=period, min_periods=period).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )

def calculate_hma(data, period=13):
    """
    Calculate Hull Moving Average with slope, turn detection, and signals.
    
    Formula: HMA = WMA(2 * WMA(n/2) - WMA(n)), sqrt(n)
    
    Returns:
        hma: HMA values
        slope: 2-bar slope percentage
        turn: 'up', 'down', or 'none'
        peak_valley: 'peak', 'valley', or 'none'
        signal: 'cross_above', 'cross_below', 'above', 'below'
    """
    n = period
    n_half = int(n / 2)
    n_sqrt = int(np.sqrt(n))
    
    # Calculate WMAs
    wma_half = calculate_wma(data, n_half)
    wma_full = calculate_wma(data, n)
    
    # Raw HMA = 2 * WMA(n/2) - WMA(n)
    raw_hma = 2 * wma_half - wma_full
    
    # Final HMA = WMA of raw HMA with period sqrt(n)
    hma = calculate_wma(raw_hma, n_sqrt)
    
    return hma

def calculate_hma_indicators(df, period=13):
    """
    Calculate HMA with all indicator signals.
    
    Returns dict with:
        hma: HMA values
        hma_slope_2bar: 2-bar slope percentage
        hma_turn: 'up', 'down', 'none'
        hma_peak_valley: 'peak', 'valley', 'none'
        hma_signal: 'cross_above', 'cross_below', 'above', 'below', 'none'
    """
    close = df['close']
    hma = calculate_hma(close, period)
    
    # Initialize result arrays
    slope = pd.Series(np.nan, index=df.index)
    turn = pd.Series('none', index=df.index)
    peak_valley = pd.Series('none', index=df.index)
    signal = pd.Series('none', index=df.index)
    
    # Calculate 2-bar slope (% change)
    for i in range(2, len(hma)):
        if not np.isnan(hma.iloc[i]) and not np.isnan(hma.iloc[i-1]):
            slope.iloc[i] = (hma.iloc[i] - hma.iloc[i-1]) / hma.iloc[i-1] * 100
    
    # Turn detection (3-bar confirmation)
    for i in range(2, len(hma)):
        if not np.isnan(hma.iloc[i]) and not np.isnan(hma.iloc[i-1]) and not np.isnan(hma.iloc[i-2]):
            # Turn up: current > prev AND prev <= prev2
            if hma.iloc[i] > hma.iloc[i-1] and hma.iloc[i-1] <= hma.iloc[i-2]:
                turn.iloc[i] = 'up'
            # Turn down: current < prev AND prev >= prev2
            elif hma.iloc[i] < hma.iloc[i-1] and hma.iloc[i-1] >= hma.iloc[i-2]:
                turn.iloc[i] = 'down'
    
    # Peak/Valley detection (2-bar)
    for i in range(2, len(hma)):
        if not np.isnan(hma.iloc[i]) and not np.isnan(hma.iloc[i-1]) and not np.isnan(hma.iloc[i-2]):
            # Peak: prev > prev2 AND prev > current
            if hma.iloc[i-1] > hma.iloc[i-2] and hma.iloc[i-1] > hma.iloc[i]:
                peak_valley.iloc[i-1] = 'peak'
            # Valley: prev < prev2 AND prev < current
            elif hma.iloc[i-1] < hma.iloc[i-2] and hma.iloc[i-1] < hma.iloc[i]:
                peak_valley.iloc[i-1] = 'valley'
    
    # Price vs HMA cross signals
    for i in range(1, len(close)):
        if not np.isnan(hma.iloc[i]) and not np.isnan(hma.iloc[i-1]):
            price_curr = close.iloc[i]
            price_prev = close.iloc[i-1]
            hma_curr = hma.iloc[i]
            hma_prev = hma.iloc[i-1]
            
            # Cross above: price was below, now above
            if price_prev < hma_prev and price_curr > hma_curr:
                signal.iloc[i] = 'cross_above'
            # Cross below: price was above, now below
            elif price_prev > hma_prev and price_curr < hma_curr:
                signal.iloc[i] = 'cross_below'
            # Above HMA
            elif price_curr > hma_curr:
                signal.iloc[i] = 'above'
            # Below HMA
            else:
                signal.iloc[i] = 'below'
    
    return {
        'hma': hma,
        'hma_slope_2bar': slope,
        'hma_turn': turn,
        'hma_peak_valley': peak_valley,
        'hma_signal': signal
    }

# ============================================================================
# ELDER IMPULSE SYSTEM
# ============================================================================

def calculate_elder_impulse(df, ema_period=13, macd_fast=12, macd_slow=26, macd_signal=9):
    """
    Calculate Elder Impulse System.
    
    Combines 13-period EMA trend direction with MACD Histogram momentum.
    
    Rules:
        - GREEN: EMA rising AND Histogram rising (bullish)
        - RED: EMA falling AND Histogram falling (bearish)  
        - BLUE: Mixed signals (neutral/transition)
    
    Returns dict with:
        elder_impulse: 'green', 'red', 'blue'
        elder_trend_strength: 0-10 scale
    """
    close = df['close']
    
    # 13-period EMA
    ema_13 = calculate_ema(close, ema_period)
    
    # MACD Histogram
    macd_line, macd_sig, macd_hist = calculate_macd(close, macd_fast, macd_slow, macd_signal)
    
    # Initialize results
    impulse = pd.Series('blue', index=df.index)
    trend_strength = pd.Series(5, index=df.index)
    
    for i in range(2, len(close)):
        if np.isnan(ema_13.iloc[i]) or np.isnan(ema_13.iloc[i-1]) or np.isnan(ema_13.iloc[i-2]):
            continue
        if np.isnan(macd_hist.iloc[i]) or np.isnan(macd_hist.iloc[i-1]):
            continue
            
        # EMA slope (2-bar)
        ema_rising = ema_13.iloc[i] > ema_13.iloc[i-1]
        ema_falling = ema_13.iloc[i] < ema_13.iloc[i-1]
        
        # Histogram slope (current vs previous)
        hist_rising = macd_hist.iloc[i] > macd_hist.iloc[i-1]
        hist_falling = macd_hist.iloc[i] < macd_hist.iloc[i-1]
        
        # Elder Impulse rules
        if ema_rising and hist_rising:
            impulse.iloc[i] = 'green'
            # Strength: steeper EMA slope + larger hist increase = stronger
            ema_slope = abs((ema_13.iloc[i] - ema_13.iloc[i-1]) / ema_13.iloc[i-1] * 100)
            hist_change = abs(macd_hist.iloc[i] - macd_hist.iloc[i-1])
            trend_strength.iloc[i] = min(10, int(5 + ema_slope * 10 + hist_change * 2))
        elif ema_falling and hist_falling:
            impulse.iloc[i] = 'red'
            # Strength calculation
            ema_slope = abs((ema_13.iloc[i] - ema_13.iloc[i-1]) / ema_13.iloc[i-1] * 100)
            hist_change = abs(macd_hist.iloc[i] - macd_hist.iloc[i-1])
            trend_strength.iloc[i] = min(10, int(5 + ema_slope * 10 + hist_change * 2))
        else:
            impulse.iloc[i] = 'blue'
            # Neutral strength based on EMA slope magnitude
            ema_slope = abs((ema_13.iloc[i] - ema_13.iloc[i-1]) / ema_13.iloc[i-1] * 100)
            trend_strength.iloc[i] = min(10, int(5 + ema_slope * 5))
    
    return {
        'elder_impulse': impulse,
        'elder_trend_strength': trend_strength
    }

def find_support_resistance(df, window=5, cluster_tolerance=0.02):
    pivot_highs = find_pivot_highs(df, window)
    pivot_lows = find_pivot_lows(df, window)
    current_price = df['close'].iloc[-1]
    
    resistance_levels = []
    if pivot_highs:
        prices = [ph['price'] for ph in pivot_highs[-20:] if ph['price'] > current_price]
        prices.sort()
        clustered = []
        for price in prices:
            found = False
            for cluster in clustered:
                if abs(price - cluster['price']) / cluster['price'] < cluster_tolerance:
                    cluster['price'] = (cluster['price'] * cluster['count'] + price) / (cluster['count'] + 1)
                    cluster['count'] += 1
                    found = True
                    break
            if not found:
                clustered.append({'price': round(price, 2), 'count': 1})
        resistance_levels = sorted(clustered, key=lambda x: x['price'])[:5]
    
    support_levels = []
    if pivot_lows:
        prices = [pl['price'] for pl in pivot_lows[-20:] if pl['price'] < current_price]
        prices.sort()
        clustered = []
        for price in prices:
            found = False
            for cluster in clustered:
                if abs(price - cluster['price']) / cluster['price'] < cluster_tolerance:
                    cluster['price'] = (cluster['price'] * cluster['count'] + price) / (cluster['count'] + 1)
                    cluster['count'] += 1
                    found = True
                    break
            if not found:
                clustered.append({'price': round(price, 2), 'count': 1})
        support_levels = sorted(clustered, key=lambda x: x['price'], reverse=True)[:5]
    
    swing_high = df['high'].max()
    swing_low = df['low'].min()
    fib_levels = calculate_fibonacci_retracement(swing_high, swing_low)
    
    return {
        'resistance': resistance_levels,
        'support': support_levels,
        'fibonacci': fib_levels,
        'recent_high': round(swing_high, 2),
        'recent_low': round(swing_low, 2)
    }


# ============================================================================
# MAIN TECHNICAL ANALYSIS FUNCTION
# ============================================================================

def generate_technical_analysis(df: pd.DataFrame, ticker: str, interval: str) -> Dict:
    if len(df) < 50:
        return None
    
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    current_price = close.iloc[-1]
    
    # Calculate all indicators with error handling
    try:
        sma_20 = calculate_sma(close, 20)
        sma_50 = calculate_sma(close, 50)
        ema_12 = calculate_ema(close, 12)
        ema_26 = calculate_ema(close, 26)
        rsi = calculate_rsi(close, 14)
        macd_line, macd_signal, _ = calculate_macd(close)
        stoch_k_slow, stoch_d_slow = calculate_stochastic(df, 14, 3, 3, 'slow')
        stoch_k_fast, stoch_d_fast = calculate_stochastic(df, 14, 3, 1, 'fast')
        trix, trix_signal = calculate_trix(close)
        mfi = calculate_mfi(df)
        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(close)
        atr = calculate_atr(df)
        vwap = calculate_vwap(df)
        obv = calculate_obv(df)
        adx, plus_di, minus_di = calculate_adx(high, low, close)
    except Exception as e:
        print(f"Indicator calculation failed for {ticker} {interval}: {e}")
        return None
    
    # Verify all indicators have valid data
    required_indicators = [sma_20, sma_50, ema_12, ema_26, rsi, macd_line, macd_signal,
                          stoch_k_slow, stoch_d_slow, stoch_k_fast, stoch_d_fast, 
                          trix, trix_signal, mfi, bb_upper, bb_middle, bb_lower, 
                          atr, vwap, obv, adx, plus_di, minus_di]
    
    for ind in required_indicators:
        if ind is None or len(ind) == 0 or ind.iloc[-1] != ind.iloc[-1]:  # Check for NaN
            print(f"Invalid indicator data for {ticker} {interval}")
            return None
    
    # Crossover detection
    try:
        macd_cross, macd_cross_str, macd_cross_loc = detect_crossover(macd_line, macd_signal)
        stoch_slow_cross, stoch_slow_str, stoch_slow_ctx = detect_stochastic_crossover(stoch_k_slow, stoch_d_slow)
        stoch_fast_cross, stoch_fast_str, stoch_fast_ctx = detect_stochastic_crossover(stoch_k_fast, stoch_d_fast)
        trix_cross, trix_cross_str, trix_cross_loc = detect_crossover(trix, trix_signal)
        
        # Candlestick patterns
        candlestick_patterns = analyze_candlestick_patterns(df)
        
        # Support/Resistance
        sr_levels = find_support_resistance(df)
        
        # HMA (Hull Moving Average) indicators
        hma_indicators = calculate_hma_indicators(df, period=13)
        
        # Elder Impulse System
        elder_indicators = calculate_elder_impulse(df)
        
    except Exception as e:
        print(f"Pattern/SR calculation failed for {ticker} {interval}: {e}")
        return None
    
    # Market regime
    adx_value = adx.iloc[-1]
    if adx_value > ADX_TREND_THRESHOLD:
        regime = 'TRENDING_UP' if plus_di.iloc[-1] > minus_di.iloc[-1] else 'TRENDING_DOWN'
    elif adx_value < ADX_RANGE_THRESHOLD:
        regime = 'RANGING'
    else:
        regime = 'NEUTRAL'
    
    direction = 'BULLISH' if ema_12.iloc[-1] > ema_26.iloc[-1] else 'BEARISH'
    
    # Volume ratio
    avg_vol = volume.rolling(window=20).mean().iloc[-1]
    volume_ratio = round(volume.iloc[-1] / avg_vol, 2) if avg_vol > 0 else 1.0
    
    # S/R distances
    support_dist = 0
    resistance_dist = 0
    if sr_levels['support']:
        support_dist = round((current_price - sr_levels['support'][0]['price']) / current_price * 100, 2)
    if sr_levels['resistance']:
        resistance_dist = round((sr_levels['resistance'][0]['price'] - current_price) / current_price * 100, 2)
    
    # Price targets
    atr_val = atr.iloc[-1]
    stop_loss = round(current_price - (atr_val * 2), 2)
    take_profit = round(current_price + (atr_val * 3), 2)
    
    # Output matching original scanner field names
    return {
        'Date': df.index[-1].strftime('%Y-%m-%d'),
        'ticker': ticker,
        'interval': interval,
        'Close': round(current_price, 2),
        'adx': round(adx_value, 2),
        'market_regime': regime,
        'direction': direction,
        'sma_20': round(sma_20.iloc[-1], 2),
        'sma_50': round(sma_50.iloc[-1], 2),
        'ema_12': round(ema_12.iloc[-1], 2),
        'ema_26': round(ema_26.iloc[-1], 2),
        'rsi': round(rsi.iloc[-1], 2),
        'mfi': round(mfi.iloc[-1], 2),
        'macd': round(macd_line.iloc[-1], 4),
        'macd_signal': round(macd_signal.iloc[-1], 4),
        'stoch_k_slow': round(stoch_k_slow.iloc[-1], 2),
        'stoch_d_slow': round(stoch_d_slow.iloc[-1], 2),
        'stoch_k_fast': round(stoch_k_fast.iloc[-1], 2),
        'stoch_d_fast': round(stoch_d_fast.iloc[-1], 2),
        'trix': round(trix.iloc[-1], 4),
        'trix_signal': round(trix_signal.iloc[-1], 4),
        'bb_upper': round(bb_upper.iloc[-1], 2),
        'bb_middle': round(bb_middle.iloc[-1], 2),
        'bb_lower': round(bb_lower.iloc[-1], 2),
        'vwap': round(vwap.iloc[-1], 2),
        'obv': int(obv.iloc[-1]),
        'atr': round(atr_val, 4),
        'volume_ratio': volume_ratio,
        'adx_plus_di': round(plus_di.iloc[-1], 2),
        'adx_minus_di': round(minus_di.iloc[-1], 2),
        'macd_cross_type': macd_cross,
        'macd_cross_strength': macd_cross_str,
        'macd_cross_location': macd_cross_loc,
        'stoch_slow_cross': stoch_slow_cross,
        'stoch_slow_strength': stoch_slow_str,
        'stoch_slow_context': stoch_slow_ctx,
        'stoch_fast_cross': stoch_fast_cross,
        'stoch_fast_strength': stoch_fast_str,
        'stoch_fast_context': stoch_fast_ctx,
        'trix_cross_type': trix_cross,
        'trix_cross_strength': trix_cross_str,
        'trix_cross_location': trix_cross_loc,
        'support_levels': json.dumps([s['price'] for s in sr_levels['support']]),
        'resistance_levels': json.dumps([r['price'] for r in sr_levels['resistance']]),
        'support_distance_pct': support_dist,
        'resistance_distance_pct': resistance_dist,
        'fib_236': sr_levels['fibonacci']['23.6%'],
        'fib_382': sr_levels['fibonacci']['38.2%'],
        'fib_500': sr_levels['fibonacci']['50.0%'],
        'fib_618': sr_levels['fibonacci']['61.8%'],
        'recent_high': sr_levels['recent_high'],
        'recent_low': sr_levels['recent_low'],
        'pivot_highs': json.dumps([ph['price'] for ph in find_pivot_highs(df)[-5:]]),
        'pivot_lows': json.dumps([pl['price'] for pl in find_pivot_lows(df)[-5:]]),
        'candlestick_patterns': ','.join(candlestick_patterns) if candlestick_patterns else None,
        'stop_loss': stop_loss,
        'take_profit': take_profit,
        # HMA indicators
        'hma_13': round(hma_indicators['hma'].iloc[-1], 2) if not np.isnan(hma_indicators['hma'].iloc[-1]) else None,
        'hma_slope_2bar': round(hma_indicators['hma_slope_2bar'].iloc[-1], 4) if not np.isnan(hma_indicators['hma_slope_2bar'].iloc[-1]) else None,
        'hma_turn': hma_indicators['hma_turn'].iloc[-1],
        'hma_peak_valley': hma_indicators['hma_peak_valley'].iloc[-1],
        'hma_signal': hma_indicators['hma_signal'].iloc[-1],
        # Elder Impulse System
        'elder_impulse': elder_indicators['elder_impulse'].iloc[-1],
        'elder_trend_strength': int(elder_indicators['elder_trend_strength'].iloc[-1]),
    }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def ensure_dirs():
    for interval in INTERVALS:
        (OUTPUT_DIR / interval).mkdir(parents=True, exist_ok=True)

def get_tickers_with_time_series() -> List[str]:
    tickers = []
    if not TIME_SERIES_DIR.exists():
        return tickers
    for ticker_dir in TIME_SERIES_DIR.iterdir():
        if ticker_dir.is_dir() and not ticker_dir.name.startswith('.'):
            if list(ticker_dir.glob("*.csv")):
                tickers.append(ticker_dir.name)
    return tickers

def get_time_series_timestamp(ticker: str) -> Optional[pd.Timestamp]:
    """Get the latest timestamp from time series data for a ticker"""
    ticker_dir = TIME_SERIES_DIR / ticker
    if not ticker_dir.exists():
        return None
    
    latest_ts = None
    for interval in INTERVALS:
        price_file = ticker_dir / f"{ticker}_{interval}.csv"
        if price_file.exists():
            try:
                df = pd.read_csv(price_file)
                if len(df) > 0 and 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'], utc=True)
                    file_latest = df['date'].max()
                    if latest_ts is None or file_latest > latest_ts:
                        latest_ts = file_latest
            except Exception:
                pass
    return latest_ts

def get_technical_analysis_timestamp(ticker: str) -> Optional[pd.Timestamp]:
    """Get the timestamp when technical analysis was last generated"""
    ticker_dir = OUTPUT_DIR / ticker
    if not ticker_dir.exists():
        return None
    
    # Check for completion marker file
    completion_file = ticker_dir / '.last_updated'
    if completion_file.exists():
        try:
            with open(completion_file, 'r') as f:
                data = json.load(f)
                return pd.Timestamp(data.get('timestamp'))
        except Exception:
            pass
    
    # Fallback: check file modification times
    latest_mtime = None
    for interval in INTERVALS:
        tech_file = ticker_dir / f"{ticker}_{interval}_technical.csv"
        if tech_file.exists():
            mtime = pd.Timestamp(tech_file.stat().st_mtime, unit='s', tz='UTC')
            if latest_mtime is None or mtime > latest_mtime:
                latest_mtime = mtime
    return latest_mtime

def save_technical_analysis_timestamp(ticker: str, timestamp: pd.Timestamp):
    """Save timestamp when technical analysis was completed"""
    ticker_dir = OUTPUT_DIR / ticker
    ticker_dir.mkdir(parents=True, exist_ok=True)
    completion_file = ticker_dir / '.last_updated'
    try:
        with open(completion_file, 'w') as f:
            json.dump({
                'timestamp': timestamp.isoformat(),
                'ticker': ticker
            }, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save timestamp for {ticker}: {e}")

def needs_update(ticker: str, buffer_hours: int = 1) -> bool:
    """Check if technical analysis needs to be regenerated
    
    Returns True if:
    - No technical analysis exists
    - Time series data is newer than technical analysis
    - Time series has data beyond what technical analysis was generated with
    """
    ts_time = get_time_series_timestamp(ticker)
    ta_time = get_technical_analysis_timestamp(ticker)
    
    if ta_time is None:
        return True  # No technical analysis exists
    
    if ts_time is None:
        return False  # No time series data (shouldn't happen if we got here)
    
    # Check if time series has newer data
    # Add buffer to avoid regenerating for very recent updates
    if ts_time > ta_time + pd.Timedelta(hours=buffer_hours):
        return True
    
    return False

def process_ticker(ticker: str, force_update: bool = False) -> Dict[str, pd.DataFrame]:
    """Process a single ticker - skip if no time series files exist"""
    ticker_dir = TIME_SERIES_DIR / ticker
    if not ticker_dir.exists():
        return {}
    
    # Check if any CSV files exist for this ticker
    csv_files = list(ticker_dir.glob("*.csv"))
    if not csv_files:
        print(f"Skipping {ticker}: no time series files found")
        return {}
    
    # Check if update is needed (unless force_update is True)
    if not force_update and not needs_update(ticker):
        return {}  # Return empty to indicate skipped (no update needed)
    
    results = {}
    for interval in INTERVALS:
        price_file = ticker_dir / f"{ticker}_{interval}.csv"
        if not price_file.exists():
            continue
        try:
            df = pd.read_csv(price_file)
            if len(df) < 50:
                continue
            df.columns = [col.lower() for col in df.columns]
            df['date'] = pd.to_datetime(df['date'], utc=True).dt.tz_localize(None)
            df.set_index('date', inplace=True)
            df.sort_index(inplace=True)
            ta_data = generate_technical_analysis(df, ticker, interval)
            if ta_data:
                results[interval] = pd.DataFrame([ta_data])
        except Exception as e:
            print(f"Error processing {ticker} {interval}: {e}")
    return results

def save_technical_analysis(ticker: str, results: Dict[str, pd.DataFrame]):
    ticker_dir = OUTPUT_DIR / ticker
    ticker_dir.mkdir(parents=True, exist_ok=True)
    for interval, df in results.items():
        output_file = ticker_dir / f"{ticker}_{interval}_technical.csv"
        df.to_csv(output_file, index=False)
    # Save timestamp after successful save
    save_technical_analysis_timestamp(ticker, pd.Timestamp.now(tz='UTC'))

def load_progress() -> Tuple[set, int]:
    """Load set of already processed tickers and last batch number"""
    if PROGRESS_FILE.exists():
        try:
            with open(PROGRESS_FILE, 'r') as f:
                data = json.load(f)
                return set(data.get('processed_tickers', [])), data.get('last_batch', 0)
        except Exception as e:
            print(f"Warning: Could not load progress: {e}")
    return set(), 0

def save_progress(processed_tickers: set, batch_count: int):
    """Save progress to resume later"""
    try:
        with open(PROGRESS_FILE, 'w') as f:
            json.dump({
                'processed_tickers': list(processed_tickers),
                'batch_count': batch_count,
                'last_batch': batch_count,
                'timestamp': pd.Timestamp.now().isoformat()
            }, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save progress: {e}")

def clear_progress():
    """Clear progress file after completion"""
    if PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()

def validate_technical_output(ticker: str, require_new_columns: bool = True) -> bool:
    """Validate that a ticker's technical analysis output is valid (at least 1 interval)
    
    Args:
        ticker: Ticker symbol
        require_new_columns: If True, requires new HMA/Elder columns to be present
    """
    ticker_dir = OUTPUT_DIR / ticker
    if not ticker_dir.exists():
        return False
    
    # Required new columns (HMA and Elder Impulse)
    new_columns = ['hma_13', 'hma_slope_2bar', 'hma_turn', 'hma_peak_valley', 
                   'hma_signal', 'elder_impulse', 'elder_trend_strength']
    
    # Check at least 1 interval has valid data
    valid_count = 0
    for interval in INTERVALS:
        file_path = ticker_dir / f"{ticker}_{interval}_technical.csv"
        if file_path.exists():
            try:
                df = pd.read_csv(file_path)
                if len(df) > 0:
                    # If requiring new columns, check they exist
                    if require_new_columns:
                        has_new_cols = all(col in df.columns for col in new_columns)
                        if has_new_cols:
                            valid_count += 1
                    else:
                        valid_count += 1
            except Exception:
                pass
    
    return valid_count > 0

def process_batch(tickers: List[str], batch_num: int, total_batches: int, failed_tickers: set) -> Tuple[int, int, set, int]:
    """Process a batch of tickers"""
    processed = 0
    errors = 0
    skipped = 0
    processed_set = set()
    
    print(f"\nBatch {batch_num}/{total_batches}: Processing {len(tickers)} tickers...")
    
    for ticker in tickers:
        try:
            results = process_ticker(ticker)
            if results:
                save_technical_analysis(ticker, results)
                processed += 1
                processed_set.add(ticker)
            else:
                # Check if it's a skip (up to date) or a failure
                if not needs_update(ticker):
                    skipped += 1
                    print(f"  Skipping {ticker}: already up to date")
                else:
                    # Ticker failed validation, add to failed set
                    failed_tickers.add(ticker)
                    errors += 1
                    print(f"Failed validation: {ticker}")
        except Exception as e:
            errors += 1
            failed_tickers.add(ticker)
            print(f"Error: {ticker} - {e}")
    
    print(f"  Batch {batch_num}: {processed} updated, {skipped} skipped (up to date), {errors} errors")
    return processed, errors, processed_set, skipped

def load_failed_tickers() -> set:
    """Load set of tickers that have failed processing"""
    failed_file = DATA_DIR / '.step3_failed_tickers'
    if failed_file.exists():
        try:
            with open(failed_file, 'r') as f:
                return set(json.load(f))
        except Exception:
            pass
    return set()

def save_failed_tickers(failed_tickers: set):
    """Save set of failed tickers"""
    failed_file = DATA_DIR / '.step3_failed_tickers'
    try:
        with open(failed_file, 'w') as f:
            json.dump(list(failed_tickers), f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save failed tickers: {e}")

def main():
    import sys
    import os
    
    # Check if batch mode requested
    batch_mode = os.getenv('BATCH_MODE', 'false').lower() == 'true'
    
    print("="*70)
    print("STEP 3: Technical Analysis & Signal Generation")
    print("Pure technical indicators - NO SCORING")
    if batch_mode:
        print(f"(BATCH MODE - {BATCH_SIZE} tickers per batch, max {MAX_BATCHES} batches)")
    print("="*70)
    print()
    
    ensure_dirs()
    
    # Load failed tickers to skip
    failed_tickers = load_failed_tickers()
    if failed_tickers:
        print(f"Note: {len(failed_tickers)} tickers previously failed, will skip")
    
    # Load progress to resume from last batch
    processed_set, resume_batch = load_progress()
    if resume_batch > 0:
        print(f"Resuming from batch {resume_batch}...")
    
    tickers = get_tickers_with_time_series()
    tickers.sort()
    
    if not tickers:
        print("No tickers found!")
        return
    
    total_tickers = len(tickers)
    
    # Filter out previously failed tickers
    tickers = [t for t in tickers if t not in failed_tickers]
    
    if not tickers:
        print("No tickers left to process (or all previously failed)!")
        return
    
    print(f"Processing {total_tickers} tickers...")
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    
    if batch_mode:
        # Process in batches
        total_batches = (total_tickers + BATCH_SIZE - 1) // BATCH_SIZE
        total_processed = 0
        total_errors = 0
        batches_run = 0
        
        # Start from resume_batch if available
        start_batch = resume_batch + 1 if resume_batch > 0 else 1
        
        for batch_num in range(start_batch, total_batches + 1):
            start_idx = (batch_num - 1) * BATCH_SIZE
            end_idx = min(start_idx + BATCH_SIZE, total_tickers)
            batch_tickers = tickers[start_idx:end_idx]
            
            # Filter to tickers that need processing (missing or invalid)
            tickers_to_process = []
            for t in batch_tickers:
                if needs_update(t):
                    tickers_to_process.append(t)
            
            if not tickers_to_process:
                print(f"Skipping batch {batch_num}/{total_batches} (all tickers up to date)")
                total_processed += len(batch_tickers)
                continue
            
            if len(tickers_to_process) != len(batch_tickers):
                print(f"Batch {batch_num}: Processing {len(tickers_to_process)}/{len(batch_tickers)} tickers (some up to date)")
            
            # Process tickers that need updating
            processed, errors, processed_in_batch, skipped = process_batch(tickers_to_process, batch_num, total_batches, failed_tickers)
            total_processed += processed + skipped
            total_errors += errors
            batches_run += 1
            
            # Update processed set and save progress after each batch
            processed_set.update(processed_in_batch)
            save_progress(processed_set, batch_num)
            
            # Save failed tickers after each batch
            if failed_tickers:
                save_failed_tickers(failed_tickers)
        
        print("\n" + "="*70)
        print(f"PROCESSING COMPLETE: {total_processed}/{total_tickers} tickers")
        print(f"Failed tickers: {len(failed_tickers)}")
        print("="*70)
    else:
        # Process all at once
        processed = 0
        errors = 0
        skipped = 0
        
        for ticker in tickers:
            try:
                results = process_ticker(ticker)
                if results:
                    save_technical_analysis(ticker, results)
                    processed += 1
                    if processed % 100 == 0:
                        print(f"Processed {processed}/{len(tickers)} tickers...")
                else:
                    if not needs_update(ticker):
                        skipped += 1
            except Exception as e:
                errors += 1
                print(f"Error: {ticker} - {e}")
        
        total_processed = processed + skipped
        total_errors = errors
    
    print()
    print("="*70)
    print(f"Complete: {processed} tickers updated, {skipped} tickers skipped (up to date), {total_errors} errors")
    print("="*70)

if __name__ == "__main__":
    main()
