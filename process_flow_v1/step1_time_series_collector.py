#!/usr/bin/env python3
"""
================================================================================
STEP 1: Time Series Data Collector
================================================================================

An incremental data collection system that fetches OHLCV (Open, High, Low, 
Close, Volume) data from Yahoo Finance for 3,500+ tickers across multiple 
timeframes. Implements intelligent caching to only fetch new data since the 
last collection.

Author: SignalsAlpha
Version: 2.0 (with incremental collection)
Date: 2026-04-18

================================================================================
PURPOSE
================================================================================

This module serves as the data foundation for the SignalsAlpha trading system.
It:

1. Fetches historical and real-time price data from Yahoo Finance API
2. Collects data across 10 timeframes (1m to 1mo) for comprehensive analysis
3. Implements incremental updates - only fetches new data since last run
4. Validates data coverage to ensure sufficient historical depth
5. Organizes data hierarchically for efficient downstream processing

================================================================================
DATA INTERVALS COLLECTED
================================================================================

Daily/Weekly/Monthly (for swing trading analysis):
    - 1d: Daily bars (full available history, max ~20 years)
    - 1wk: Weekly bars (full available history)
    - 1mo: Monthly bars (full available history)

Note: Intraday intervals (1m, 5m, 1h, etc.) are not collected as they are
not needed for swing trading signals which focus on multi-day to multi-week
holding periods.

================================================================================
INCREMENTAL COLLECTION FEATURES
================================================================================

Smart Updates:
    - Checks existing CSV files for last recorded date
    - Calculates days since last update
    - Fetches only the gap period (not full history)
    - Merges new data with existing records
    - Removes duplicates automatically

Fetch Period Optimization:
    - 0-5 days missing: Fetches 5 days
    - 6-30 days missing: Fetches 1 month
    - 31-90 days missing: Fetches 3 months
    - 90+ days missing: Fetches full history

Validation:
    - Ensures minimum historical coverage (365 days for daily data)
    - Re-fetches if data is incomplete or corrupted
    - Handles delisted tickers gracefully

================================================================================
WORKFLOW
================================================================================

1. INITIALIZATION
   - Load ticker list from Step 0 (stock_ticker_base.csv)
   - Create output directory structure: data/time_series/{TICKER}/

2. PER-TICKER PROCESSING
   For each ticker:
   a. Check existing files for each interval
   b. Determine last recorded date
   c. Calculate days since update
   d. Fetch only missing data (or full history if new)
   e. Merge with existing data
   f. Save updated CSV

3. RATE LIMITING
   - Configurable delay between API calls (default: 1.0s)
   - Prevents Yahoo Finance rate limiting
   - Respects API terms of service

4. ERROR HANDLING
   - Retries on temporary failures
   - Skips delisted/unavailable tickers
   - Logs errors for manual review

================================================================================
OUTPUT FORMAT
================================================================================

Directory Structure:
    data/
    └── time_series/
        └── {TICKER}/
            ├── {TICKER}_1m.csv
            ├── {TICKER}_2m.csv
            ├── {TICKER}_5m.csv
            ├── {TICKER}_15m.csv
            ├── {TICKER}_30m.csv
            ├── {TICKER}_60m.csv
            ├── {TICKER}_1h.csv
            ├── {TICKER}_1d.csv
            ├── {TICKER}_1wk.csv
            └── {TICKER}_1mo.csv

CSV Schema (all intervals):
    date: Timestamp (ISO 8601 format)
    open: Opening price
    high: Highest price
    low: Lowest price
    close: Closing price
    volume: Trading volume
    dividends: Dividend payments (if any)
    stock_splits: Split ratios (if any)

================================================================================
USAGE
================================================================================

Full Collection (all tickers):
    python step1_time_series_collector.py

With Batch Restart (recommended for OCI):
    bash run_timeseries_batch.sh

Test Single Ticker:
    python step1_time_series_collector.py --test AAPL

================================================================================
CONFIGURATION
================================================================================

Rate Limiting:
    MIN_DELAY = 1.0  # Seconds between API calls

Retry Logic:
    MAX_RETRIES = 3    # Attempts per ticker
    INITIAL_BACKOFF = 2.0  # Seconds

Batch Processing:
    TICKERS_PER_BATCH = 10  # Process before restart

================================================================================
DEPENDENCIES
================================================================================

- yfinance: Yahoo Finance data access
- pandas: Data manipulation and CSV handling
- numpy: Numerical operations
- datetime: Date/time calculations
- time: Rate limiting
- typing: Type hints

================================================================================
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

# Intervals to collect - only daily, weekly, monthly for swing trading signals
INTERVALS = {
    '1d': {'period': 'max', 'max_days': None},    # 1 day - max available
    '1wk': {'period': 'max', 'max_days': None},   # 1 week - max available
    '1mo': {'period': 'max', 'max_days': None},  # 1 month - max available
}

# Rate limiting - be gentle with yfinance
MIN_DELAY = 1.0  # seconds between API calls
last_call = 0

# Progress tracking file
PROGRESS_FILE = ".step1_progress"

def load_progress():
    """Load list of already processed tickers from this session"""
    try:
        if os.path.exists(PROGRESS_FILE):
            with open(PROGRESS_FILE, 'r') as f:
                return set(line.strip() for line in f if line.strip())
    except Exception:
        pass
    return set()

def save_progress(ticker):
    """Save a ticker as processed"""
    try:
        with open(PROGRESS_FILE, 'a') as f:
            f.write(f"{ticker}\n")
    except Exception:
        pass


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
        
        # Convert to datetime and get last - handle mixed timezones
        df[date_col] = pd.to_datetime(df[date_col], utc=True)
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
        
        df[date_col] = pd.to_datetime(df[date_col], utc=True)
        
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
        
        # Calculate days since last update - handle timezone-aware dates
        now = datetime.now()
        if last_date.tzinfo is not None:
            # Make now timezone-aware to match last_date
            import pytz
            now = now.replace(tzinfo=pytz.UTC)
        
        days_since = (now - last_date).days
        
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
        
        # Ensure date columns match - use UTC to avoid timezone issues
        date_col_new = 'date' if 'date' in new_df.columns else new_df.columns[0]
        date_col_existing = 'date' if 'date' in existing_df.columns else existing_df.columns[0]
        
        new_df[date_col_new] = pd.to_datetime(new_df[date_col_new], utc=True)
        existing_df[date_col_existing] = pd.to_datetime(existing_df[date_col_existing], utc=True)
        
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


def process_all_stocks(symbols: List[str], batch_size: int = 50):
    """
    Process all stocks in batches with automatic restart capability.
    
    Args:
        symbols: List of stock symbols to process
        batch_size: Number of stocks to process before saving progress and exiting
                   (allows shell script to restart with fresh memory)
    """
    
    print("=" * 80)
    print("TIME SERIES COLLECTOR - BATCHED RUN")
    print("=" * 80)
    print(f"Total stocks: {len(symbols)}")
    print(f"Batch size: {batch_size}")
    print(f"Estimated time: {len(symbols) * len(INTERVALS) * MIN_DELAY / 60:.1f} minutes")
    print("=" * 80)
    
    # Load session progress (tickers already processed in this run)
    session_processed = load_progress()
    if session_processed:
        print(f"\nSession progress: {len(session_processed)} tickers already processed")
        symbols = [s for s in symbols if s not in session_processed]
        print(f"Remaining in this session: {len(symbols)}")
    
    # Check for existing progress - check data freshness, not just file existence
    from datetime import datetime
    import pytz
    
    processed = []
    needs_update = []
    
    for symbol in symbols:
        ticker_dir = os.path.join(OUTPUT_BASE_DIR, symbol)
        if os.path.exists(ticker_dir):
            # Check if all intervals exist AND are up to date
            existing = [f for f in os.listdir(ticker_dir) if f.endswith('.csv')]
            if len(existing) >= len(INTERVALS):
                # Check if data is fresh (within 1 day)
                all_fresh = True
                for interval in INTERVALS.keys():
                    file_path = os.path.join(ticker_dir, f"{symbol}_{interval}.csv")
                    if os.path.exists(file_path):
                        last_date = get_last_date_from_csv(file_path)
                        if last_date:
                            now = datetime.now().replace(tzinfo=pytz.UTC)
                            days_old = (now - last_date).days
                            if days_old > 1:  # Data is more than 1 day old
                                all_fresh = False
                                break
                
                if all_fresh:
                    processed.append(symbol)
                else:
                    needs_update.append(symbol)
    
    if processed:
        print(f"\nUp to date: {len(processed)} stocks")
    if needs_update:
        print(f"\nNeeds update: {len(needs_update)} stocks")
        symbols = needs_update + [s for s in symbols if s not in processed and s not in needs_update]
    else:
        symbols = [s for s in symbols if s not in processed]
    
    print(f"Remaining to process: {len(symbols)}")
    
    if not symbols:
        print("\n✓ All stocks already processed!")
        return
    
    # Process only up to batch_size stocks, then exit
    # Shell script will restart to clear memory
    symbols_to_process = symbols[:batch_size]
    remaining_after_batch = symbols[batch_size:]
    
    print(f"\nProcessing batch of {len(symbols_to_process)} stocks...")
    if remaining_after_batch:
        print(f"{len(remaining_after_batch)} stocks will be processed in next run")
    
    start_time = time.time()
    total_processed = len(processed)
    
    for i, symbol in enumerate(symbols_to_process):
        print(f"\n[{i+1}/{len(symbols_to_process)}] Processing {symbol}...")
        
        try:
            results = process_ticker(symbol)
            total_processed += 1
            
            # Save progress immediately after successful processing
            save_progress(symbol)
            
            # Progress update every 10 stocks
            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                print(f"\nBatch progress: {i+1}/{len(symbols_to_process)} "
                      f"| Rate: {rate:.2f} stocks/sec")
            
        except Exception as e:
            print(f"  Failed to process {symbol}: {e}")
    
    # Summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print("BATCH COMPLETE")
    print("=" * 80)
    print(f"Processed this batch: {len(symbols_to_process)}")
    print(f"Total complete: {total_processed}")
    print(f"Remaining: {len(remaining_after_batch)}")
    print(f"Batch time: {elapsed/60:.1f} minutes")
    print("=" * 80)
    
    # Save progress indicator for external monitoring
    progress_file = os.path.join(OUTPUT_BASE_DIR, ".progress")
    with open(progress_file, 'w') as f:
        f.write(f"{total_processed}/{len(symbols) + len(processed)}")
    
    # Exit with special code if more work remains
    # Shell script can check this and restart
    if remaining_after_batch:
        print(f"\n⚠ {len(remaining_after_batch)} stocks remaining - restart to continue")
        sys.exit(99)  # Special exit code for "more work needed"


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        # Test mode - single stock
        ticker = sys.argv[2] if len(sys.argv) > 2 else 'AAPL'
        test_single_stock(ticker)
    else:
        # Full run - all stocks
        # Load from ticker base (Step 0 output) - updated path
        ticker_file = '../process_flow_v0/data/tickers/stock_ticker_base.csv'
        
        if not os.path.exists(ticker_file):
            print(f"Error: Ticker file not found: {ticker_file}")
            print("Please run Step 0 (ticker collector) first.")
            sys.exit(1)
        
        df = pd.read_csv(ticker_file)
        
        # Prioritize yahoo_symbol (corrected format) over symbol (original)
        if 'yahoo_symbol' in df.columns:
            symbols = df['yahoo_symbol'].tolist()
            print(f"  Using yahoo_symbol column (corrected format)")
        elif 'symbol' in df.columns:
            symbols = df['symbol'].tolist()
            print(f"  Using symbol column (original format)")
        elif 'ticker' in df.columns:
            symbols = df['ticker'].tolist()
            print(f"  Using ticker column")
        else:
            print("Error: No symbol/ticker column found in ticker file")
            sys.exit(1)
        
        print(f"Loaded {len(symbols)} tickers from {ticker_file}")
        
        # Filter out invalid symbols (NaN, floats, empty strings)
        symbols = [str(s).strip() for s in symbols if pd.notna(s) and str(s).strip()]
        symbols = [s for s in symbols if s and s.lower() != 'nan']
        
        print(f"After filtering: {len(symbols)} valid tickers")
        process_all_stocks(symbols)
