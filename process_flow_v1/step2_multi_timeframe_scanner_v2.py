#!/usr/bin/env python3
"""
================================================================================
STEP 2: Multi-Timeframe Signal Scanner v2.0 - ADAPTIVE WEIGHTING
================================================================================

Enhanced technical analysis engine with ADX-based trend regime detection
and adaptive indicator weighting.

Author: SignalsAlpha
Version: 2.0
Date: 2026-04-19

================================================================================
NEW IN v2.0: ADAPTIVE INDICATOR WEIGHTING
================================================================================

MARKET REGIME DETECTION:
    - ADX (14-period) measures trend strength
    - ADX > 25: Trending market (strong directional movement)
    - ADX < 20: Ranging market (weak/no trend)
    - ADX 20-25: Transition zone (neutral)

ADAPTIVE WEIGHTING STRATEGY:

    Indicator        | Trending (ADX > 25) | Ranging (ADX < 20)
    -----------------|---------------------|--------------------
    SMA              | +1                  | +2
    EMA              | +2                  | +1
    MACD             | +3                  | +1
    TRIX             | +3                  | +1
    RSI              | +1                  | +3
    Stochastic       | +1                  | +3
    MFI              | +1                  | +3
    Bollinger Bands  | +1                  | +3
    Candlesticks     | Fixed               | Fixed

    Rationale:
    - SMA: Mean-reversion works better in ranges
    - EMA: Trend-following, stronger in trends
    - MACD/TRIX: Momentum works best in trending markets
    - RSI/Stochastic/MFI: Oscillators mean-revert, best in ranges
    - Bollinger Bands: Bounce trading works in ranges

================================================================================
OUTPUT ENHANCEMENTS
================================================================================

New columns added to signal output:
    - adx_value: Current ADX reading
    - market_regime: TRENDING_UP, TRENDING_DOWN, RANGING, NEUTRAL
    - sma_weight: Applied SMA weight for this signal
    - ema_weight: Applied EMA weight for this signal
    - macd_weight: Applied MACD weight
    - trix_weight: Applied TRIX weight
    - rsi_weight: Applied RSI weight
    - stoch_weight: Applied Stochastic weight
    - mfi_weight: Applied MFI weight
    - bb_weight: Applied Bollinger Bands weight
    - total_trend_weight: Sum of trend-following weights
    - total_range_weight: Sum of mean-reversion weights

================================================================================
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

# Configuration
DATA_DIR = Path("/home/ubuntu/.openclaw/workspace/trading-signals-data/process_flow_v1/data")
TIME_SERIES_DIR = DATA_DIR / "time_series"
OUTPUT_DIR = DATA_DIR / "signals_timeframe"
TICKERS_FILE = DATA_DIR / "tickers" / "stock_ticker_base.csv"

# ADX threshold for trend detection
ADX_TREND_THRESHOLD = 25
ADX_RANGE_THRESHOLD = 20

# Adaptive weights
WEIGHTS = {
    'trending': {
        'sma': 1,
        'ema': 2,
        'macd': 3,
        'trix': 3,
        'rsi': 1,
        'stoch': 1,
        'mfi': 1,
        'bb': 1
    },
    'ranging': {
        'sma': 2,
        'ema': 1,
        'macd': 1,
        'trix': 1,
        'rsi': 3,
        'stoch': 3,
        'mfi': 3,
        'bb': 3
    }
}


def ensure_dirs():
    """Create output directories"""
    for timeframe in ['1d', '1wk', '1mo']:
        (OUTPUT_DIR / timeframe).mkdir(parents=True, exist_ok=True)


def calculate_sma(data, period):
    """Calculate Simple Moving Average"""
    return data.rolling(window=period, min_periods=period).mean()


def calculate_ema(data, period):
    """Calculate Exponential Moving Average"""
    return data.ewm(span=period, adjust=False, min_periods=period).mean()


def calculate_rsi(data, period=14):
    """Calculate RSI"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def calculate_adx(high, low, close, period=14):
    """
    Calculate ADX (Average Directional Index)
    
    ADX measures trend strength (not direction)
    - 0-20: Weak trend (ranging)
    - 20-40: Strong trend
    - 40+: Very strong trend
    """
    # Calculate True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate +DM and -DM
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    # Smooth TR, +DM, -DM
    atr = tr.rolling(window=period, min_periods=period).mean()
    plus_di = 100 * (plus_dm.rolling(window=period, min_periods=period).mean() / atr.replace(0, np.nan))
    minus_di = 100 * (minus_dm.rolling(window=period, min_periods=period).mean() / atr.replace(0, np.nan))
    
    # Calculate DX and ADX
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)) * 100
    adx = dx.rolling(window=period, min_periods=period).mean()
    
    return adx.fillna(20), plus_di.fillna(50), minus_di.fillna(50)


def classify_market_regime(adx_value: float, plus_di: float, minus_di: float) -> Tuple[str, str]:
    """
    Classify market regime based on ADX and directional indicators
    
    Returns:
        regime: TRENDING_UP, TRENDING_DOWN, RANGING, or NEUTRAL
        direction: UP, DOWN, or SIDEWAYS
    """
    if pd.isna(adx_value):
        return 'NEUTRAL', 'SIDEWAYS'
    
    if adx_value >= ADX_TREND_THRESHOLD:
        # Strong trend
        if plus_di > minus_di:
            return 'TRENDING_UP', 'UP'
        else:
            return 'TRENDING_DOWN', 'DOWN'
    elif adx_value <= ADX_RANGE_THRESHOLD:
        # Ranging market
        return 'RANGING', 'SIDEWAYS'
    else:
        # Transition zone
        return 'NEUTRAL', 'SIDEWAYS'


def get_indicator_weights(regime: str) -> Dict[str, int]:
    """Get indicator weights based on market regime"""
    if regime in ['TRENDING_UP', 'TRENDING_DOWN']:
        return WEIGHTS['trending']
    elif regime == 'RANGING':
        return WEIGHTS['ranging']
    else:
        # Neutral - use average of both
        return {
            'sma': (WEIGHTS['trending']['sma'] + WEIGHTS['ranging']['sma']) // 2,
            'ema': (WEIGHTS['trending']['ema'] + WEIGHTS['ranging']['ema']) // 2,
            'macd': (WEIGHTS['trending']['macd'] + WEIGHTS['ranging']['macd']) // 2,
            'trix': (WEIGHTS['trending']['trix'] + WEIGHTS['ranging']['trix']) // 2,
            'rsi': (WEIGHTS['trending']['rsi'] + WEIGHTS['ranging']['rsi']) // 2,
            'stoch': (WEIGHTS['trending']['stoch'] + WEIGHTS['ranging']['stoch']) // 2,
            'mfi': (WEIGHTS['trending']['mfi'] + WEIGHTS['ranging']['mfi']) // 2,
            'bb': (WEIGHTS['trending']['bb'] + WEIGHTS['ranging']['bb']) // 2,
        }


def calculate_macd(data, fast=12, slow=26, signal=9):
    """Calculate MACD"""
    ema_fast = calculate_ema(data, fast)
    ema_slow = calculate_ema(data, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calculate_bollinger_bands(data, period=20, std_dev=2):
    """Calculate Bollinger Bands"""
    sma = calculate_sma(data, period)
    std = data.rolling(window=period, min_periods=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, sma, lower


def calculate_trix(data, period=15):
    """Calculate TRIX"""
    single_ema = calculate_ema(data, period)
    double_ema = calculate_ema(single_ema, period)
    triple_ema = calculate_ema(double_ema, period)
    trix = 100 * (triple_ema - triple_ema.shift(1)) / triple_ema.shift(1).replace(0, np.nan)
    trix_signal = calculate_ema(trix, 9)
    return trix, trix_signal


def calculate_stochastic(df, k_period=14, d_period=3, slowing=3, stoch_type='slow'):
    """Calculate Stochastic Oscillator"""
    lowest_low = df['Low'].rolling(window=k_period, min_periods=k_period).min()
    highest_high = df['High'].rolling(window=k_period, min_periods=k_period).max()
    
    if stoch_type == 'fast':
        k = 100 * ((df['Close'] - lowest_low) / (highest_high - lowest_low).replace(0, np.nan))
        d = calculate_sma(k, d_period)
        return k, d
    else:  # slow
        k_fast = 100 * ((df['Close'] - lowest_low) / (highest_high - lowest_low).replace(0, np.nan))
        k_slow = calculate_sma(k_fast, slowing)
        d_slow = calculate_sma(k_slow, d_period)
        return k_slow, d_slow


def calculate_mfi(df, period=14):
    """Calculate Money Flow Index"""
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    raw_money_flow = typical_price * df['Volume']
    
    money_flow = raw_money_flow.copy()
    money_flow_plus = money_flow.where(typical_price > typical_price.shift(1), 0)
    money_flow_minus = money_flow.where(typical_price < typical_price.shift(1), 0)
    
    positive_sum = money_flow_plus.rolling(window=period, min_periods=period).sum()
    negative_sum = money_flow_minus.rolling(window=period, min_periods=period).sum()
    
    money_ratio = positive_sum / negative_sum.replace(0, np.nan)
    mfi = 100 - (100 / (1 + money_ratio))
    return mfi.fillna(50)


def process_ticker(ticker: str) -> Dict[str, pd.DataFrame]:
    """
    Process single ticker and generate signals with adaptive weighting
    """
    results = {}
    
    for interval in ['1d', '1wk', '1mo']:
        # Load price data
        price_file = TIME_SERIES_DIR / ticker / f"{ticker}_{interval}.csv"
        if not price_file.exists():
            continue
        
        try:
            df = pd.read_csv(price_file)
            if len(df) < 50:
                continue
            
            # Normalize column names - lowercase first, then capitalize
            df.columns = [col.lower() for col in df.columns]
            df.columns = [col.capitalize() if col in ['open', 'high', 'low', 'close', 'volume'] else col for col in df.columns]
            df['Date'] = pd.to_datetime(df['date'], utc=True).dt.tz_localize(None)
            df = df.dropna(subset=['Date'])
            
            close = df['Close']
            high = df['High']
            low = df['Low']
            volume = df['Volume']
            
            # Calculate indicators
            sma_20 = calculate_sma(close, 20)
            sma_50 = calculate_sma(close, 50)
            ema_12 = calculate_ema(close, 12)
            ema_26 = calculate_ema(close, 26)
            rsi = calculate_rsi(close, 14)
            
            # Calculate ADX for trend detection
            adx, plus_di, minus_di = calculate_adx(high, low, close, 14)
            
            # Calculate MACD
            macd_line, signal_line, histogram = calculate_macd(close)
            
            # Calculate Bollinger Bands
            bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(close)
            
            # Calculate TRIX
            trix, trix_signal = calculate_trix(close)
            
            # Calculate Stochastic
            stoch_k_slow, stoch_d_slow = calculate_stochastic(df, stoch_type='slow')
            stoch_k_fast, stoch_d_fast = calculate_stochastic(df, stoch_type='fast')
            
            # Calculate MFI
            mfi = calculate_mfi(df)
            
            # Generate signals with adaptive weighting
            signals = []
            
            for i in range(len(df)):
                if i < 50:  # Skip initial rows
                    signals.append({
                        'Date': df['Date'].iloc[i],
                        'signal': 'HOLD',
                        'confidence': 50,
                        'adx': np.nan,
                        'market_regime': 'NEUTRAL',
                        'direction': 'SIDEWAYS'
                    })
                    continue
                
                # Current values
                current_close = close.iloc[i]
                current_sma_20 = sma_20.iloc[i]
                current_sma_50 = sma_50.iloc[i]
                current_ema_12 = ema_12.iloc[i]
                current_ema_26 = ema_26.iloc[i]
                current_rsi = rsi.iloc[i]
                current_adx = adx.iloc[i]
                current_plus_di = plus_di.iloc[i]
                current_minus_di = minus_di.iloc[i]
                current_macd = macd_line.iloc[i]
                current_signal = signal_line.iloc[i]
                current_bb_upper = bb_upper.iloc[i]
                current_bb_lower = bb_lower.iloc[i]
                current_trix = trix.iloc[i]
                current_stoch_k = stoch_k_slow.iloc[i]
                current_mfi = mfi.iloc[i]
                
                # Classify market regime
                regime, direction = classify_market_regime(current_adx, current_plus_di, current_minus_di)
                weights = get_indicator_weights(regime)
                
                # Calculate adaptive scores
                buy_score = 0
                sell_score = 0
                
                # SMA analysis
                if current_close > current_sma_20 > current_sma_50:
                    buy_score += weights['sma']
                elif current_close < current_sma_20 < current_sma_50:
                    sell_score += weights['sma']
                
                # EMA analysis
                if current_ema_12 > current_ema_26:
                    buy_score += weights['ema']
                else:
                    sell_score += weights['ema']
                
                # MACD analysis
                if current_macd > current_signal:
                    buy_score += weights['macd']
                else:
                    sell_score += weights['macd']
                
                # RSI analysis (adaptive weight)
                if current_rsi < 30:
                    buy_score += weights['rsi']
                elif current_rsi > 70:
                    sell_score += weights['rsi']
                elif current_rsi > 50:
                    buy_score += weights['rsi'] // 2
                else:
                    sell_score += weights['rsi'] // 2
                
                # Bollinger Bands (adaptive weight)
                if current_close < current_bb_lower:
                    buy_score += weights['bb']
                elif current_close > current_bb_upper:
                    sell_score += weights['bb']
                
                # TRIX analysis (adaptive weight)
                if current_trix > 0:
                    buy_score += weights['trix']
                else:
                    sell_score += weights['trix']
                
                # Stochastic (adaptive weight)
                if current_stoch_k < 20:
                    buy_score += weights['stoch']
                elif current_stoch_k > 80:
                    sell_score += weights['stoch']
                
                # MFI (adaptive weight)
                if current_mfi < 20:
                    buy_score += weights['mfi']
                elif current_mfi > 80:
                    sell_score += weights['mfi']
                
                # Determine signal
                if buy_score >= sell_score + 4:
                    signal = 'STRONG_BUY'
                    confidence = min(95, 50 + buy_score * 8)
                elif buy_score >= sell_score + 2:
                    signal = 'BUY'
                    confidence = min(85, 50 + buy_score * 7)
                elif sell_score >= buy_score + 4:
                    signal = 'STRONG_SELL'
                    confidence = min(95, 50 + sell_score * 8)
                elif sell_score >= buy_score + 2:
                    signal = 'SELL'
                    confidence = min(85, 50 + sell_score * 7)
                else:
                    signal = 'HOLD'
                    confidence = 50
                
                signals.append({
                    'Date': df['Date'].iloc[i],
                    'signal': signal,
                    'confidence': confidence,
                    'adx': current_adx,
                    'market_regime': regime,
                    'direction': direction,
                    'buy_score': buy_score,
                    'sell_score': sell_score,
                    'sma_weight': weights['sma'],
                    'ema_weight': weights['ema'],
                    'macd_weight': weights['macd'],
                    'trix_weight': weights['trix'],
                    'rsi_weight': weights['rsi'],
                    'stoch_weight': weights['stoch'],
                    'mfi_weight': weights['mfi'],
                    'bb_weight': weights['bb'],
                    'total_trend_weight': weights['ema'] + weights['macd'] + weights['trix'],
                    'total_range_weight': weights['sma'] + weights['rsi'] + weights['stoch'] + weights['mfi'] + weights['bb'],
                    'Close': current_close
                })
            
            results[interval] = pd.DataFrame(signals)
            
        except Exception as e:
            print(f"Error processing {ticker} {interval}: {e}")
            continue
    
    return results


def save_signals(ticker: str, results: Dict[str, pd.DataFrame]):
    """Save signals to CSV files"""
    ticker_dir = OUTPUT_DIR / ticker
    ticker_dir.mkdir(parents=True, exist_ok=True)
    
    for interval, df in results.items():
        output_file = ticker_dir / f"{ticker}_{interval}.csv"
        df.to_csv(output_file, index=False)


def main():
    """Main processing loop"""
    print("="*70)
    print("STEP 2: Multi-Timeframe Signal Scanner v2.0")
    print("Adaptive Indicator Weighting with ADX")
    print("="*70)
    print()
    
    ensure_dirs()
    
    # Get tickers
    if TICKERS_FILE.exists():
        df = pd.read_csv(TICKERS_FILE)
        tickers = df['symbol'].tolist() if 'symbol' in df.columns else []
    else:
        tickers = [d.name for d in TIME_SERIES_DIR.iterdir() if d.is_dir()]
    
    if not tickers:
        print("No tickers found!")
        return
    
    print(f"Processing {len(tickers)} tickers...")
    print(f"ADX Trend Threshold: {ADX_TREND_THRESHOLD}")
    print(f"ADX Range Threshold: {ADX_RANGE_THRESHOLD}")
    print()
    
    processed = 0
    errors = 0
    
    for ticker in tickers:
        try:
            results = process_ticker(ticker)
            if results:
                save_signals(ticker, results)
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
