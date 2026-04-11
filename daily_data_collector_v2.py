#!/usr/bin/env python3
"""
Daily Data Collector v2.0
Optimized for overnight refreshes - 3 key intervals only
Handles splits, validates data integrity
"""

import os
import sys
import time
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Configuration
OUTPUT_DIR = "data/daily_data_v2"
LOG_FILE = "data/daily_data_v2/collection_log.txt"

# 3 key intervals for daily refresh
INTERVALS = {
    '1d': {'period': 'max', 'description': 'Daily candles'},
    '1wk': {'period': 'max', 'description': 'Weekly candles'},
    '1mo': {'period': 'max', 'description': 'Monthly candles'}
}

# Rate limiting
MIN_DELAY = 1.0
last_call = 0


def rate_limit():
    """Ensure we don't hit rate limits"""
    global last_call
    elapsed = time.time() - last_call
    if elapsed < MIN_DELAY:
        time.sleep(MIN_DELAY - elapsed)
    last_call = time.time()


def log_message(msg: str):
    """Log with timestamp"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S ET')
    log_line = f"[{timestamp}] {msg}"
    print(log_line)
    with open(LOG_FILE, 'a') as f:
        f.write(log_line + '\n')


def validate_data(df: pd.DataFrame, ticker: str, interval: str) -> Tuple[bool, List[str]]:
    """
    Validate OHLCV data integrity
    Returns: (is_valid, list_of_issues)
    """
    issues = []
    
    if df.empty:
        return False, ["Empty dataframe"]
    
    # Required columns (yfinance returns capitalized)
    required_map = {'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}
    missing = [col for col, yf_col in required_map.items() if yf_col not in df.columns]
    if missing:
        issues.append(f"Missing columns: {missing}")
    
    # Check for nulls in required columns
    yf_cols = [required_map[col] for col in required_map if required_map[col] in df.columns]
    if yf_cols:
        null_counts = df[yf_cols].isnull().sum()
        if null_counts.any():
            issues.append(f"Null values: {null_counts.to_dict()}")
    
    # OHLC logic (allowing for split adjustments)
    if 'High' in df.columns and 'Low' in df.columns:
        if (df['High'] < df[['Open', 'Close']].max(axis=1)).any():
            violations = (df['High'] < df[['Open', 'Close']].max(axis=1)).sum()
            if violations > len(df) * 0.05:  # Allow 5% for adjustments
                issues.append(f"High < Open/Close in {violations} rows (possible splits)")
        
        if (df['Low'] > df[['Open', 'Close']].min(axis=1)).any():
            violations = (df['Low'] > df[['Open', 'Close']].min(axis=1)).sum()
            if violations > len(df) * 0.05:
                issues.append(f"Low > Open/Close in {violations} rows (possible splits)")
    
    # Volume should be non-negative
    if 'Volume' in df.columns:
        if (df['Volume'] < 0).any():
            issues.append("Negative volume detected")
    
    # Check for stock splits - just INFO, not an error
    if 'Stock Splits' in df.columns:
        splits = df[df['Stock Splits'] != 0]['Stock Splits']
        if len(splits) > 0:
            issues.append(f"INFO: {len(splits)} stock splits detected")
    
    # Only critical errors make it invalid
    critical_issues = [i for i in issues if not i.startswith("INFO:")]
    return len(critical_issues) == 0, issues


def fetch_and_validate(ticker: str, interval: str, period: str) -> Tuple[Optional[pd.DataFrame], Dict]:
    """
    Fetch data and validate
    Returns: (dataframe, metadata dict)
    """
    metadata = {
        'ticker': ticker,
        'interval': interval,
        'requested_at': datetime.now().isoformat(),
        'success': False,
        'rows': 0,
        'issues': [],
        'splits_detected': 0
    }
    
    try:
        rate_limit()
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval, prepost=False)
        
        if df.empty:
            metadata['issues'].append("No data returned")
            return None, metadata
        
        # Detect stock splits and calculate cumulative adjustment factor
        split_info = []
        if 'Stock Splits' in df.columns:
            splits = df[df['Stock Splits'] != 0]['Stock Splits']
            metadata['splits_detected'] = len(splits)
            if len(splits) > 0:
                metadata['split_dates'] = splits.index.strftime('%Y-%m-%d').tolist()
                metadata['split_ratios'] = splits.tolist()
                
                # Calculate cumulative adjustment factor (for reference)
                # If stock had 2:1 then 3:1 splits, total factor = 6
                cumulative_factor = 1.0
                for ratio in splits.tolist():
                    cumulative_factor *= ratio
                metadata['cumulative_split_factor'] = cumulative_factor
                
                # Split direction info
                forward_splits = sum(1 for r in splits.tolist() if r > 1)
                reverse_splits = sum(1 for r in splits.tolist() if r < 1 and r > 0)
                metadata['forward_splits'] = forward_splits
                metadata['reverse_splits'] = reverse_splits
        
        # Validate
        is_valid, issues = validate_data(df, ticker, interval)
        metadata['issues'].extend(issues)
        
        if not is_valid:
            return None, metadata
        
        # Format for output - yfinance prices are already split-adjusted
        df = df.reset_index()
        df.columns = [col.replace(' ', '_').lower() for col in df.columns]
        
        # Add metadata columns
        df['ticker'] = ticker
        df['interval'] = interval
        df['split_adjusted'] = True  # yfinance prices are split-adjusted by default
        df['collected_at'] = datetime.now().isoformat()
        
        metadata['success'] = True
        metadata['rows'] = len(df)
        metadata['start_date'] = str(df.iloc[0]['date']) if 'date' in df.columns else None
        metadata['end_date'] = str(df.iloc[-1]['date']) if 'date' in df.columns else None
        
        return df, metadata
        
    except Exception as e:
        metadata['issues'].append(str(e))
        return None, metadata


def save_data(ticker: str, interval: str, df: pd.DataFrame) -> str:
    """Save validated data to CSV"""
    ticker_dir = os.path.join(OUTPUT_DIR, ticker)
    os.makedirs(ticker_dir, exist_ok=True)
    
    filepath = os.path.join(ticker_dir, f"{ticker}_{interval}.csv")
    df.to_csv(filepath, index=False)
    return filepath


def process_ticker(ticker: str) -> Dict:
    """Process all 3 intervals for one ticker"""
    results = {
        'ticker': ticker,
        'intervals': {},
        'completed_at': None,
        'overall_success': True
    }
    
    log_message(f"Processing {ticker}...")
    
    for interval, config in INTERVALS.items():
        df, metadata = fetch_and_validate(ticker, interval, config['period'])
        results['intervals'][interval] = metadata
        
        if metadata['success'] and df is not None:
            filepath = save_data(ticker, interval, df)
            metadata['filepath'] = filepath
            log_message(f"  {interval}: ✓ {metadata['rows']:,} rows (splits: {metadata['splits_detected']})")
        else:
            log_message(f"  {interval}: ✗ {'; '.join(metadata['issues'][:2])}")
            results['overall_success'] = False
    
    results['completed_at'] = datetime.now().isoformat()
    return results


def test_single_stock(ticker: str = 'AAPL'):
    """Test with one stock and show detailed validation"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    log_message("=" * 80)
    log_message("DAILY DATA COLLECTOR v2.0 - SINGLE STOCK TEST")
    log_message("=" * 80)
    log_message(f"Testing: {ticker}")
    log_message(f"Intervals: {', '.join(INTERVALS.keys())}")
    log_message(f"Output: {OUTPUT_DIR}/{ticker}/")
    log_message("=" * 80)
    
    start = time.time()
    results = process_ticker(ticker)
    elapsed = time.time() - start
    
    # Detailed report
    log_message("")
    log_message("=" * 80)
    log_message("VALIDATION REPORT")
    log_message("=" * 80)
    
    for interval, meta in results['intervals'].items():
        log_message(f"\n{interval.upper()}:")
        log_message(f"  Success: {meta['success']}")
        log_message(f"  Rows: {meta['rows']:,}")
        log_message(f"  Splits detected: {meta['splits_detected']}")
        if meta.get('start_date'):
            log_message(f"  Date range: {meta['start_date'][:10]} to {meta['end_date'][:10]}")
        if meta['issues']:
            log_message(f"  Issues: {meta['issues']}")
    
    log_message(f"\nTime elapsed: {elapsed:.1f}s")
    log_message("=" * 80)
    
    return results


def process_all_stocks(symbols: List[str], batch_size: int = 50):
    """Process all stocks with checkpointing"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    log_message("=" * 80)
    log_message("DAILY DATA COLLECTOR v2.0 - FULL RUN")
    log_message("=" * 80)
    log_message(f"Total stocks: {len(symbols)}")
    log_message(f"Batch size: {batch_size}")
    log_message(f"Output: {OUTPUT_DIR}/")
    log_message("=" * 80)
    
    # Check already processed
    processed = []
    for symbol in symbols:
        sym_dir = os.path.join(OUTPUT_DIR, symbol)
        if os.path.exists(sym_dir):
            files = [f for f in os.listdir(sym_dir) if f.endswith('.csv')]
            if len(files) >= 3:  # All 3 intervals
                processed.append(symbol)
    
    if processed:
        log_message(f"Resuming: {len(processed)} already complete")
        symbols = [s for s in symbols if s not in processed]
    
    remaining = len(symbols)
    log_message(f"Remaining: {remaining}")
    
    start_time = time.time()
    
    for i, symbol in enumerate(symbols):
        try:
            process_ticker(symbol)
            
            # Progress update
            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (remaining - i - 1) / rate if rate > 0 else 0
                log_message(f"Progress: {i+1}/{remaining} | Rate: {rate:.2f}/s | ETA: {eta/3600:.1f}h")
                
        except Exception as e:
            log_message(f"ERROR processing {symbol}: {e}")
    
    log_message("")
    log_message("=" * 80)
    log_message("COLLECTION COMPLETE")
    log_message("=" * 80)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        ticker = sys.argv[2] if len(sys.argv) > 2 else 'AAPL'
        test_single_stock(ticker)
    else:
        # Full run
        df = pd.read_csv('data/signals_v3/all_signals_v3_final.csv')
        symbols = df['ticker'].tolist()
        process_all_stocks(symbols)
