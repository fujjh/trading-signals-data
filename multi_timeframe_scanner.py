#!/usr/bin/env python3
"""
Multi-Timeframe Signal Scanner
Runs technical analysis on each time series interval and generates signals
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuration
BASE_DIR = Path("/home/ubuntu/.openclaw/workspace/data")
TIME_SERIES_DIR = BASE_DIR / "time_series"
OUTPUT_DIR = BASE_DIR / "signals_timeframe"
INTERVALS = ['1d', '1wk', '1mo']  # Only 3 intervals from Daily Data Collector v2.0

def ensure_dir(path):
    """Create directory if it doesn't exist"""
    path.mkdir(parents=True, exist_ok=True)

def load_time_series(ticker, interval):
    """Load time series data for a specific ticker and interval"""
    file_path = TIME_SERIES_DIR / ticker / f"{ticker}_{interval}.csv"
    if not file_path.exists():
        return None
    
    try:
        df = pd.read_csv(file_path)
        # Handle different column names
        if 'date' in df.columns:
            date_col = 'date'
        elif 'Datetime' in df.columns:
            date_col = 'Datetime'
        else:
            date_col = df.columns[0]
        
        # Parse datetime with utc=True to handle mixed timezones
        df[date_col] = pd.to_datetime(df[date_col], utc=True)
        df.set_index(date_col, inplace=True)
        df.index = df.index.tz_localize(None)  # Remove timezone
        
        # Rename columns to standard format
        col_map = {}
        for c in df.columns:
            lower_c = c.lower()
            if lower_c in ['open', 'high', 'low', 'close', 'volume']:
                col_map[c] = c.capitalize()
        df.rename(columns=col_map, inplace=True)
        
        df.sort_index(inplace=True)
        return df
    except Exception as e:
        print(f"Error loading {ticker} {interval}: {e}")
        return None

def calculate_sma(data, period):
    """Calculate Simple Moving Average"""
    return data.rolling(window=period).mean()

def calculate_ema(data, period):
    """Calculate Exponential Moving Average"""
    return data.ewm(span=period, adjust=False).mean()

def calculate_rsi(data, period=14):
    """Calculate Relative Strength Index"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

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
    std = data.rolling(window=period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    return upper_band, sma, lower_band

def calculate_atr(df, period=14):
    """Calculate Average True Range"""
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.rolling(window=period).mean()
    return atr

def calculate_vwap(df):
    """Calculate Volume Weighted Average Price"""
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    vwap = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()
    return vwap

def calculate_obv(df):
    """Calculate On-Balance Volume"""
    obv = (np.sign(df['Close'].diff()) * df['Volume']).cumsum()
    return obv

def find_pivot_highs(df, window=5):
    """Find local pivot highs"""
    highs = df['High']
    pivot_highs = []
    
    for i in range(window, len(highs) - window):
        if highs.iloc[i] == highs.iloc[i-window:i+window+1].max():
            pivot_highs.append({
                'index': df.index[i],
                'price': round(highs.iloc[i], 2)
            })
    
    return pivot_highs

def find_pivot_lows(df, window=5):
    """Find local pivot lows"""
    lows = df['Low']
    pivot_lows = []
    
    for i in range(window, len(lows) - window):
        if lows.iloc[i] == lows.iloc[i-window:i+window+1].min():
            pivot_lows.append({
                'index': df.index[i],
                'price': round(lows.iloc[i], 2)
            })
    
    return pivot_lows

def calculate_fibonacci_retracement(high, low):
    """Calculate Fibonacci retracement levels"""
    diff = high - low
    levels = {
        '0.0%': round(high, 2),
        '23.6%': round(high - (diff * 0.236), 2),
        '38.2%': round(high - (diff * 0.382), 2),
        '50.0%': round(high - (diff * 0.5), 2),
        '61.8%': round(high - (diff * 0.618), 2),
        '78.6%': round(high - (diff * 0.786), 2),
        '100.0%': round(low, 2)
    }
    return levels

def find_support_resistance(df, window=5, cluster_tolerance=0.02):
    """Find support and resistance levels using pivot analysis"""
    pivot_highs = find_pivot_highs(df, window)
    pivot_lows = find_pivot_lows(df, window)
    
    current_price = df['Close'].iloc[-1]
    
    # Cluster resistance levels (pivot highs)
    resistance_levels = []
    if pivot_highs:
        prices = [ph['price'] for ph in pivot_highs[-20:]]  # Last 20 pivots
        prices.sort()
        
        clustered = []
        for price in prices:
            if price > current_price:  # Only above current price
                found_cluster = False
                for cluster in clustered:
                    if abs(price - cluster['price']) / cluster['price'] < cluster_tolerance:
                        cluster['price'] = (cluster['price'] * cluster['count'] + price) / (cluster['count'] + 1)
                        cluster['count'] += 1
                        found_cluster = True
                        break
                if not found_cluster:
                    clustered.append({'price': round(price, 2), 'count': 1})
        
        resistance_levels = sorted([c for c in clustered if c['price'] > current_price], 
                                   key=lambda x: x['price'])[:5]
    
    # Cluster support levels (pivot lows)
    support_levels = []
    if pivot_lows:
        prices = [pl['price'] for pl in pivot_lows[-20:]]  # Last 20 pivots
        prices.sort()
        
        clustered = []
        for price in prices:
            if price < current_price:  # Only below current price
                found_cluster = False
                for cluster in clustered:
                    if abs(price - cluster['price']) / cluster['price'] < cluster_tolerance:
                        cluster['price'] = (cluster['price'] * cluster['count'] + price) / (cluster['count'] + 1)
                        cluster['count'] += 1
                        found_cluster = True
                        break
                if not found_cluster:
                    clustered.append({'price': round(price, 2), 'count': 1})
        
        support_levels = sorted([c for c in clustered if c['price'] < current_price], 
                                key=lambda x: x['price'], reverse=True)[:5]
    
    # Get swing high/low for Fibonacci
    swing_high = df['High'].max() if len(df) > 0 else current_price
    swing_low = df['Low'].min() if len(df) > 0 else current_price
    fib_levels = calculate_fibonacci_retracement(swing_high, swing_low)
    
    return {
        'resistance': resistance_levels,
        'support': support_levels,
        'fibonacci': fib_levels,
        'recent_high': round(swing_high, 2),
        'recent_low': round(swing_low, 2)
    }

def generate_signal(df, ticker, interval):
    """Generate trading signal based on technical indicators"""
    if df is None or len(df) < 50:
        return None
    
    # Get latest data
    close = df['Close'].iloc[-1]
    prev_close = df['Close'].iloc[-2]
    
    # Calculate indicators
    sma_20 = calculate_sma(df['Close'], 20)
    sma_50 = calculate_sma(df['Close'], 50)
    ema_12 = calculate_ema(df['Close'], 12)
    ema_26 = calculate_ema(df['Close'], 26)
    rsi = calculate_rsi(df['Close'])
    macd_line, signal_line, histogram = calculate_macd(df['Close'])
    upper_band, middle_band, lower_band = calculate_bollinger_bands(df['Close'])
    atr = calculate_atr(df)
    
    # Get latest values
    latest_sma_20 = sma_20.iloc[-1]
    latest_sma_50 = sma_50.iloc[-1]
    latest_ema_12 = ema_12.iloc[-1]
    latest_ema_26 = ema_26.iloc[-1]
    latest_rsi = rsi.iloc[-1]
    latest_macd = macd_line.iloc[-1]
    latest_signal = signal_line.iloc[-1]
    latest_histogram = histogram.iloc[-1]
    latest_upper = upper_band.iloc[-1]
    latest_lower = lower_band.iloc[-1]
    latest_atr = atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else 0
    
    # NEW: Calculate VWAP and OBV
    vwap = calculate_vwap(df)
    obv = calculate_obv(df)
    latest_vwap = vwap.iloc[-1] if not pd.isna(vwap.iloc[-1]) else close
    latest_obv = obv.iloc[-1] if not pd.isna(obv.iloc[-1]) else 0
    
    # NEW: Find Support/Resistance levels
    sr_levels = find_support_resistance(df)
    
    # NEW: Find recent peaks and troughs
    pivot_highs = find_pivot_highs(df)
    pivot_lows = find_pivot_lows(df)
    
    # Calculate scores
    buy_score = 0
    sell_score = 0
    
    # VWAP analysis
    if close > latest_vwap:
        buy_score += 1
    elif close < latest_vwap:
        sell_score += 1
    
    # Trend analysis
    if close > latest_sma_20 > latest_sma_50:
        buy_score += 2
    elif close < latest_sma_20 < latest_sma_50:
        sell_score += 2
    
    if latest_ema_12 > latest_ema_26:
        buy_score += 1
    elif latest_ema_12 < latest_ema_26:
        sell_score += 1
    
    # RSI analysis
    if latest_rsi < 30:
        buy_score += 2  # Oversold
    elif latest_rsi > 70:
        sell_score += 2  # Overbought
    elif latest_rsi > 50:
        buy_score += 1
    else:
        sell_score += 1
    
    # MACD analysis
    if latest_macd > latest_signal and latest_histogram > 0:
        buy_score += 2
    elif latest_macd < latest_signal and latest_histogram < 0:
        sell_score += 2
    
    # Bollinger Bands
    if close < latest_lower:
        buy_score += 1  # Price below lower band
    elif close > latest_upper:
        sell_score += 1  # Price above upper band
    
    # Volume analysis (if available)
    volume_avg = df['Volume'].rolling(20).mean().iloc[-1]
    latest_volume = df['Volume'].iloc[-1]
    if not pd.isna(volume_avg) and volume_avg > 0:
        volume_ratio = latest_volume / volume_avg
        if volume_ratio > 1.5:
            if buy_score > sell_score:
                buy_score += 1
            elif sell_score > buy_score:
                sell_score += 1
    
    # Determine signal
    signal = "HOLD"
    confidence = 50
    
    if buy_score >= 7:
        signal = "STRONG_BUY"
        confidence = 50 + (buy_score * 7)
    elif buy_score >= 4:
        signal = "BUY"
        confidence = 50 + (buy_score * 10)
    elif sell_score >= 7:
        signal = "STRONG_SELL"
        confidence = 50 + (sell_score * 7)
    elif sell_score >= 4:
        signal = "SELL"
        confidence = 50 + (sell_score * 10)
    elif buy_score >= 2:
        signal = "WEAK_BUY"
        confidence = 50 + (buy_score * 8)
    elif sell_score >= 2:
        signal = "WEAK_SELL"
        confidence = 50 + (sell_score * 8)
    
    confidence = min(95, max(50, confidence))
    
    # Calculate price levels
    stop_loss = close - (latest_atr * 2) if latest_atr > 0 else close * 0.95
    take_profit = close + (latest_atr * 3) if latest_atr > 0 else close * 1.05
    
    return {
        'ticker': ticker,
        'interval': interval,
        'timestamp': df.index[-1].strftime('%Y-%m-%d %H:%M:%S'),
        'close': round(close, 2),
        'vwap': round(latest_vwap, 2) if not pd.isna(latest_vwap) else None,
        'obv': int(latest_obv) if not pd.isna(latest_obv) else None,
        'signal': signal,
        'confidence': confidence,
        'buy_score': buy_score,
        'sell_score': sell_score,
        'rsi': round(latest_rsi, 2) if not pd.isna(latest_rsi) else None,
        'macd': round(latest_macd, 4) if not pd.isna(latest_macd) else None,
        'sma_20': round(latest_sma_20, 2) if not pd.isna(latest_sma_20) else None,
        'sma_50': round(latest_sma_50, 2) if not pd.isna(latest_sma_50) else None,
        'bb_upper': round(latest_upper, 2) if not pd.isna(latest_upper) else None,
        'bb_lower': round(latest_lower, 2) if not pd.isna(latest_lower) else None,
        'atr': round(latest_atr, 2) if not pd.isna(latest_atr) else None,
        'volume_ratio': round(volume_ratio, 2) if 'volume_ratio' in locals() else None,
        'resistance_levels': json.dumps([r['price'] for r in sr_levels['resistance']]),
        'support_levels': json.dumps([s['price'] for s in sr_levels['support']]),
        'fib_236': sr_levels['fibonacci'].get('23.6%'),
        'fib_382': sr_levels['fibonacci'].get('38.2%'),
        'fib_500': sr_levels['fibonacci'].get('50.0%'),
        'fib_618': sr_levels['fibonacci'].get('61.8%'),
        'recent_high': sr_levels['recent_high'],
        'recent_low': sr_levels['recent_low'],
        'peaks': json.dumps([p['price'] for p in pivot_highs[-5:]]),
        'troughs': json.dumps([p['price'] for p in pivot_lows[-5:]]),
        'stop_loss': round(stop_loss, 2),
        'take_profit': round(take_profit, 2)
    }

def scan_ticker(ticker):
    """Scan all intervals for a single ticker"""
    results = []
    for interval in INTERVALS:
        df = load_time_series(ticker, interval)
        if df is not None:
            signal = generate_signal(df, ticker, interval)
            if signal:
                results.append(signal)
    return results

def save_signals(signals, ticker):
    """Save signals to CSV file"""
    if not signals:
        return
    
    ticker_dir = OUTPUT_DIR / ticker
    ensure_dir(ticker_dir)
    
    output_file = ticker_dir / f"{ticker}_signals.csv"
    
    df = pd.DataFrame(signals)
    df.to_csv(output_file, index=False)
    print(f"Saved {len(signals)} signals to {output_file}")

def main():
    """Main execution"""
    ensure_dir(OUTPUT_DIR)
    
    # Get list of tickers
    tickers = [d.name for d in TIME_SERIES_DIR.iterdir() if d.is_dir()]
    tickers.sort()
    
    print(f"Found {len(tickers)} tickers with time series data")
    print(f"Scanning intervals: {', '.join(INTERVALS)}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("-" * 60)
    
    total_signals = 0
    processed = 0
    
    for ticker in tickers:
        signals = scan_ticker(ticker)
        if signals:
            save_signals(signals, ticker)
            total_signals += len(signals)
            processed += 1
            
            if processed % 50 == 0:
                print(f"Progress: {processed}/{len(tickers)} tickers processed, {total_signals} signals generated")
    
    print("-" * 60)
    print(f"Complete! Processed {processed} tickers, generated {total_signals} signals")
    print(f"Signals saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()