#!/usr/bin/env python3
"""
Time Series Data Collector
Fetches OHLCV data for multiple intervals using yfinance
Saves each interval as separate CSV per ticker
"""

import os
import sys
import time
import yfinance as yf
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Configuration
OUTPUT_BASE_DIR = "data/time_series"
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

# Intervals to collect with their max periods
INTERVALS = {
    '1m': {'period': '7d', 'max_days': 7},      # 1 minute - 7 days max
    '2m': {'period': '60d', 'max_days': 60},     # 2 minutes - 60 days
    '5m': {'period': '60d', 'max_days': 60},     # 5 minutes - 60 days
    '15m': {'period': '60d', 'max_days': 60},    # 15 minutes - 60 days
    '30m': {'period': '60d', 'max_days': 60},    # 30 minutes - 60 days
    '60m': {'period': '730d', 'max_days': 730},  # 60 minutes - 730 days (2 years)
    '1h': {'period': '730d', 'max_days': 730},    # 1 hour - same as 60m
    '1d': {'period': 'max', 'max_days': None},    # 1 day - max available
    '1wk': {'period': 'max', 'max_days': None},   # 1 week - max available
    '1mo': {'period': 'max', 'max_days': None},  # 1 month - max available
}

# Rate limiting - be gentle with yfinance
MIN_DELAY = 1.0  # seconds between API calls
last_call = 0


def rate_limit():
    """Ensure we don't hit rate limits"""
    global last_call
    elapsed = time.time() - last_call
    if elapsed < MIN_DELAY:
        time.sleep(MIN_DELAY - elapsed)
    last_call = time.time()


def fetch_interval_data(ticker: str, interval: str, period: str) -> Optional[pd.DataFrame]:
    """Fetch data for a specific interval"""
    try:
        stock = yf.Ticker(ticker)
        
        # Download data
        df = stock.history(period=period, interval=interval, prepost=False)
        
        if df.empty:
            return None
        
        # Clean and format
        df = df.reset_index()
        
        # Rename columns to standard format
        df.columns = [col.replace(' ', '_').lower() for col in df.columns]
        
        # Add metadata
        df['ticker'] = ticker
        df['interval'] = interval
        
        return df
        
    except Exception as e:
        print(f"  Error fetching {ticker} {interval}: {e}")
        return None


def save_interval_data(ticker: str, interval: str, df: pd.DataFrame) -> str:
    """Save interval data to CSV"""
    
    # Create ticker directory
    ticker_dir = os.path.join(OUTPUT_BASE_DIR, ticker)
    os.makedirs(ticker_dir, exist_ok=True)
    
    # Save to CSV
    output_path = os.path.join(ticker_dir, f"{ticker}_{interval}.csv")
    df.to_csv(output_path, index=False)
    
    return output_path


def process_ticker(ticker: str) -> Dict:
    """Process all intervals for a single ticker"""
    
    results = {
        'ticker': ticker,
        'intervals_collected': [],
        'intervals_failed': [],
        'files_created': [],
        'total_rows': 0,
        'errors': []
    }
    
    print(f"\nProcessing {ticker}...")
    
    for interval, config in INTERVALS.items():
        print(f"  Fetching {interval}...", end='', flush=True)
        
        try:
            rate_limit()  # Respect rate limits
            
            df = fetch_interval_data(ticker, interval, config['period'])
            
            if df is not None and not df.empty:
                file_path = save_interval_data(ticker, interval, df)
                results['intervals_collected'].append(interval)
                results['files_created'].append(file_path)
                results['total_rows'] += len(df)
                print(f" ✓ {len(df)} rows")
            else:
                results['intervals_failed'].append(interval)
                print(f" ✗ No data")
                
        except Exception as e:
            results['intervals_failed'].append(interval)
            results['errors'].append(f"{interval}: {str(e)}")
            print(f" ✗ Error: {e}")
    
    results['completed_at'] = datetime.now().isoformat()
    return results


def test_single_stock(ticker: str = 'AAPL') -> Dict:
    """Test with a single stock"""
    
    print("=" * 80)
    print("TIME SERIES COLLECTOR - SINGLE STOCK TEST")
    print("=" * 80)
    print(f"Testing with: {ticker}")
    print(f"Intervals: {', '.join(INTERVALS.keys())}")
    print(f"Rate limit: {MIN_DELAY}s between calls")
    print(f"Pre/post market: False (regular hours only)")
    print("=" * 80)
    
    start_time = time.time()
    results = process_ticker(ticker)
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 80)
    print("TEST RESULTS")
    print("=" * 80)
    print(f"Ticker: {results['ticker']}")
    print(f"Time elapsed: {elapsed:.1f}s")
    print(f"\nIntervals collected: {len(results['intervals_collected'])}/{len(INTERVALS)}")
    
    if results['intervals_collected']:
        print("\n✓ Successful:")
        for interval in results['intervals_collected']:
            print(f"  - {interval}")
    
    if results['intervals_failed']:
        print("\n✗ Failed:")
        for interval in results['intervals_failed']:
            print(f"  - {interval}")
    
    if results['errors']:
        print("\nErrors:")
        for error in results['errors']:
            print(f"  - {error}")
    
    print(f"\nFiles created:")
    for f in results['files_created']:
        print(f"  {f}")
    
    print(f"\nTotal rows collected: {results['total_rows']:,}")
    print("=" * 80)
    
    return results


def process_all_stocks(symbols: List[str], batch_size: int = 20):
    """Process all stocks in batches"""
    
    print("=" * 80)
    print("TIME SERIES COLLECTOR - FULL RUN")
    print("=" * 80)
    print(f"Total stocks: {len(symbols)}")
    print(f"Batch size: {batch_size}")
    print(f"Estimated time: {len(symbols) * len(INTERVALS) * MIN_DELAY / 60:.1f} minutes")
    print("=" * 80)
    
    # Check for existing progress
    processed = []
    for symbol in symbols:
        ticker_dir = os.path.join(OUTPUT_BASE_DIR, symbol)
        if os.path.exists(ticker_dir):
            # Check if all intervals exist
            existing = [f for f in os.listdir(ticker_dir) if f.endswith('.csv')]
            if len(existing) >= len(INTERVALS) // 2:  # At least half done
                processed.append(symbol)
    
    if processed:
        print(f"\nResuming: {len(processed)} already complete")
        symbols = [s for s in symbols if s not in processed]
        print(f"Remaining: {len(symbols)}")
    
    start_time = time.time()
    total_processed = len(processed)
    
    for i, symbol in enumerate(symbols):
        print(f"\n[{i+1}/{len(symbols)}] Processing {symbol}...")
        
        try:
            results = process_ticker(symbol)
            total_processed += 1
            
            # Progress update every 10 stocks
            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                remaining = (len(symbols) - i - 1) / rate if rate > 0 else 0
                print(f"\nProgress: {total_processed}/{len(symbols) + len(processed)} "
                      f"| Rate: {rate:.2f} stocks/sec | ETA: {remaining/60:.1f}m")
            
        except Exception as e:
            print(f"  Failed to process {symbol}: {e}")
    
    # Summary
    print("\n" + "=" * 80)
    print("COLLECTION COMPLETE")
    print("=" * 80)
    print(f"Total processed: {total_processed}")
    print(f"Output directory: {OUTPUT_BASE_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        # Test mode - single stock
        ticker = sys.argv[2] if len(sys.argv) > 2 else 'AAPL'
        test_single_stock(ticker)
    else:
        # Full run - all stocks
        # Load from signals v3
        df = pd.read_csv('data/signals_v3/all_signals_v3_final.csv')
        symbols = df['ticker'].tolist()
        process_all_stocks(symbols)
