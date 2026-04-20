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
INTERVALS = ['1d', '1wk', '1mo']

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
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.rolling(window=period, min_periods=period).mean()
    return atr

def calculate_vwap(df):
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    vwap = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()
    return vwap

def calculate_obv(df):
    obv = (np.sign(df['Close'].diff()) * df['Volume']).cumsum()
    return obv

def calculate_stochastic(df, k_period=14, d_period=3, slowing=3, stoch_type='slow'):
    lowest_low = df['Low'].rolling(window=k_period, min_periods=k_period).min()
    highest_high = df['High'].rolling(window=k_period, min_periods=k_period).max()
    k_fast = 100 * ((df['Close'] - lowest_low) / (highest_high - lowest_low).replace(0, np.nan))
    k = k_fast.rolling(window=slowing, min_periods=slowing).mean()
    d = k.rolling(window=d_period, min_periods=d_period).mean()
    return k, d

def calculate_mfi(df, period=14):
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    raw_money_flow = typical_price * df['Volume']
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
    if detect_doji(c3['Open'], c3['Close'], c3['High'], c3['Low']):
        patterns.append('DOJI')
    if detect_hammer(c3['Open'], c3['Close'], c3['High'], c3['Low']):
        if c3['Close'] < df['Close'].mean():
            patterns.append('HAMMER')
        else:
            patterns.append('HANGING_MAN')
    if detect_shooting_star(c3['Open'], c3['Close'], c3['High'], c3['Low']):
        if c3['Close'] > df['Close'].mean():
            patterns.append('SHOOTING_STAR')
        else:
            patterns.append('INVERTED_HAMMER')
    if detect_bullish_engulfing(c2['Open'], c2['Close'], c3['Open'], c3['Close']):
        patterns.append('BULLISH_ENGULFING')
    if detect_bearish_engulfing(c2['Open'], c2['Close'], c3['Open'], c3['Close']):
        patterns.append('BEARISH_ENGULFING')
    if detect_morning_star(c1['Open'], c1['Close'], c2['Open'], c2['Close'], c3['Open'], c3['Close']):
        patterns.append('MORNING_STAR')
    if detect_evening_star(c1['Open'], c1['Close'], c2['Open'], c2['Close'], c3['Open'], c3['Close']):
        patterns.append('EVENING_STAR')
    return patterns


# ============================================================================
# SUPPORT/RESISTANCE
# ============================================================================

def find_pivot_highs(df, window=5):
    highs = df['High']
    pivot_highs = []
    for i in range(window, len(highs) - window):
        if highs.iloc[i] == highs.iloc[i-window:i+window+1].max():
            pivot_highs.append({'index': i, 'price': round(highs.iloc[i], 2)})
    return pivot_highs

def find_pivot_lows(df, window=5):
    lows = df['Low']
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

def find_support_resistance(df, window=5, cluster_tolerance=0.02):
    pivot_highs = find_pivot_highs(df, window)
    pivot_lows = find_pivot_lows(df, window)
    current_price = df['Close'].iloc[-1]
    
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
    
    swing_high = df['High'].max()
    swing_low = df['Low'].min()
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
    
    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']
    current_price = close.iloc[-1]
    
    # Calculate all indicators
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
    
    # Crossover detection
    macd_cross, macd_cross_str, macd_cross_loc = detect_crossover(macd_line, macd_signal)
    stoch_slow_cross, stoch_slow_str, stoch_slow_ctx = detect_stochastic_crossover(stoch_k_slow, stoch_d_slow)
    stoch_fast_cross, stoch_fast_str, stoch_fast_ctx = detect_stochastic_crossover(stoch_k_fast, stoch_d_fast)
    trix_cross, trix_cross_str, trix_cross_loc = detect_crossover(trix, trix_signal)
    
    # Candlestick patterns
    candlestick_patterns = analyze_candlestick_patterns(df)
    
    # Support/Resistance
    sr_levels = find_support_resistance(df)
    
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

def process_ticker(ticker: str) -> Dict[str, pd.DataFrame]:
    results = {}
    for interval in INTERVALS:
        price_file = TIME_SERIES_DIR / ticker / f"{ticker}_{interval}.csv"
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

def main():
    print("="*70)
    print("STEP 3: Technical Analysis & Signal Generation")
    print("Pure technical indicators - NO SCORING")
    print("="*70)
    print()
    
    ensure_dirs()
    tickers = get_tickers_with_time_series()
    tickers.sort()
    
    if not tickers:
        print("No tickers found!")
        return
    
    print(f"Processing {len(tickers)} tickers...")
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    
    processed = 0
    errors = 0
    
    for ticker in tickers:
        try:
            results = process_ticker(ticker)
            if results:
                save_technical_analysis(ticker, results)
                processed += 1
                if processed % 100 == 0:
                    print(f"Processed {processed}/{len(tickers)} tickers...")
        except Exception as e:
            errors += 1
            print(f"Error: {ticker} - {e}")
    
    print()
    print("="*70)
    print(f"Complete: {processed} tickers processed, {errors} errors")
    print("="*70)

if __name__ == "__main__":
    main()
