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


def get_last_date_from_csv(file_path: str) -> Optional[datetime]:
    """Get the last date from an existing CSV file"""
    try:
        if not os.path.exists(file_path):
            return None
        
        df = pd.read_csv(file_path)
        if df.empty:
            return None
        
        # Try common date column names
        date_cols = ['date', 'Date', 'datetime', 'Datetime', 'timestamp', 'Timestamp']
        date_col = None
        
        for col in date_cols:
            if col in df.columns:
                date_col = col
                break
        
        if date_col is None:
            date_col = df.columns[0]
        
        # Convert to datetime and get last
        df[date_col] = pd.to_datetime(df[date_col])
        last_date = df[date_col].max()
        
        return last_date
        
    except Exception as e:
        print(f"  Warning: Could not read last date from {file_path}: {e}")
        return None


def validate_data_coverage(file_path: str, min_days: int = 365) -> bool:
    """Validate that CSV has sufficient historical coverage"""
    try:
        if not os.path.exists(file_path):
            return False
        
        df = pd.read_csv(file_path)
        if df.empty:
            return False
        
        date_cols = ['date', 'Date', 'datetime', 'Datetime']
        date_col = None
        for col in date_cols:
            if col in df.columns:
                date_col = col
                break
        
        if date_col is None:
            date_col = df.columns[0]
        
        df[date_col] = pd.to_datetime(df[date_col])
        
        first_date = df[date_col].min()
        last_date = df[date_col].max()
        days_coverage = (last_date - first_date).days
        
        return days_coverage >= min_days
        
    except Exception as e:
        print(f"  Warning: Validation error for {file_path}: {e}")
        return False


def fetch_incremental_data(ticker: str, interval: str, period: str, existing_file: str) -> Optional[pd.DataFrame]:
    """
    Fetch only new data since last collection
    
    Args:
        ticker: Stock ticker symbol
        interval: Data interval (1d, 1wk, etc.)
        period: Yahoo Finance period string
        existing_file: Path to existing CSV file
    
    Returns:
        DataFrame with new data, or None if no update needed
    """
    try:
        # Check if file exists and get last date
        last_date = get_last_date_from_csv(existing_file)
        
        if last_date is None:
            # No existing data, fetch full history
            print("(full)", end=' ', flush=True)
            return fetch_interval_data(ticker, interval, period)
        
        # Calculate days since last update
        days_since = (datetime.now() - last_date).days
        
        if days_since <= 0:
            # Data is up to date
            return None
        
        # Fetch only new data
        if days_since <= 5:
            fetch_period = '5d'
        elif days_since <= 30:
            fetch_period = '1mo'
        elif days_since <= 90:
            fetch_period = '3mo'
        else:
            fetch_period = period  # Full period for large gaps
        
        print(f"(+{days_since}d)", end=' ', flush=True)
        
        new_df = fetch_interval_data(ticker, interval, fetch_period)
        
        if new_df is None or new_df.empty:
            return None
        
        # Combine with existing data
        existing_df = pd.read_csv(existing_file)
        
        # Ensure date columns match
        date_col_new = 'date' if 'date' in new_df.columns else new_df.columns[0]
        date_col_existing = 'date' if 'date' in existing_df.columns else existing_df.columns[0]
        
        new_df[date_col_new] = pd.to_datetime(new_df[date_col_new])
        existing_df[date_col_existing] = pd.to_datetime(existing_df[date_col_existing])
        
        # Remove overlapping dates from existing data
        existing_df = existing_df[existing_df[date_col_existing] < new_df[date_col_new].min()]
        
        # Concatenate and deduplicate
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=[date_col_existing], keep='last')
        
        return combined
        
    except Exception as e:
        print(f"(err: {e})", end=' ', flush=True)
        # Fall back to full fetch
        return fetch_interval_data(ticker, interval, period)


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


def process_ticker(ticker: str, incremental: bool = True, validate_coverage: bool = True) -> Dict:
    """
    Process all intervals for a single ticker
    
    Args:
        ticker: Stock ticker symbol
        incremental: If True, only fetch new data since last collection
        validate_coverage: If True, validate data has sufficient history
    
    Returns:
        Dictionary with results
    """
    
    results = {
        'ticker': ticker,
        'intervals_collected': [],
        'intervals_updated': [],
        'intervals_failed': [],
        'files_created': [],
        'total_rows': 0,
        'errors': []
    }
    
    print(f"\nProcessing {ticker}...")
    
    for interval, config in INTERVALS.items():
        print(f"  {interval}:", end=' ', flush=True)
        
        try:
            rate_limit()  # Respect rate limits
            
            # Check for existing file
            ticker_dir = os.path.join(OUTPUT_BASE_DIR, ticker)
            existing_file = os.path.join(ticker_dir, f"{ticker}_{interval}.csv")
            
            # Validate existing data coverage (for 1d, 1wk, 1mo only)
            if validate_coverage and interval in ['1d', '1wk', '1mo'] and os.path.exists(existing_file):
                min_days = {'1d': 365, '1wk': 52, '1mo': 12}[interval]
                has_coverage = validate_data_coverage(existing_file, min_days)
                
                if not has_coverage:
                    print("(re-fetching - insufficient history)", end=' ', flush=True)
                    # Force full fetch
                    df = fetch_interval_data(ticker, interval, config['period'])
                elif incremental:
                    # Use incremental fetch
                    df = fetch_incremental_data(ticker, interval, config['period'], existing_file)
                else:
                    # Skip if up to date
                    last_date = get_last_date_from_csv(existing_file)
                    if last_date and (datetime.now() - last_date).days <= 0:
                        print("(up to date)")
                        results['intervals_collected'].append(interval)
                        continue
                    else:
                        df = fetch_interval_data(ticker, interval, config['period'])
            elif incremental and os.path.exists(existing_file):
                # Use incremental fetch for intraday data too
                df = fetch_incremental_data(ticker, interval, config['period'], existing_file)
            else:
                # Full fetch
                print("(full)", end=' ', flush=True)
                df = fetch_interval_data(ticker, interval, config['period'])
            
            if df is not None and not df.empty:
                file_path = save_interval_data(ticker, interval, df)
                
                if os.path.exists(existing_file) and interval not in results['intervals_collected']:
                    results['intervals_updated'].append(interval)
                else:
                    results['intervals_collected'].append(interval)
                
                results['files_created'].append(file_path)
                results['total_rows'] += len(df)
                print(f"✓ {len(df)} rows")
            else:
                if os.path.exists(existing_file):
                    results['intervals_collected'].append(interval)
                    print("✓ (no update needed)")
                else:
                    results['intervals_failed'].append(interval)
                    print("✗ No data")
                
        except Exception as e:
            results['intervals_failed'].append(interval)
            results['errors'].append(f"{interval}: {str(e)}")
            print(f"✗ Error: {e}")
    
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
        # Load from ticker base (Step 0 output)
        ticker_file = '../../data/tickers/stock_ticker_base.csv'
        
        if not os.path.exists(ticker_file):
            print(f"Error: Ticker file not found: {ticker_file}")
            print("Please run Step 0 (ticker collector) first.")
            sys.exit(1)
        
        df = pd.read_csv(ticker_file)
        
        if 'symbol' in df.columns:
            symbols = df['symbol'].tolist()
        elif 'ticker' in df.columns:
            symbols = df['ticker'].tolist()
        else:
            print("Error: No symbol/ticker column found in ticker file")
            sys.exit(1)
        
        print(f"Loaded {len(symbols)} tickers from {ticker_file}")
        process_all_stocks(symbols)
