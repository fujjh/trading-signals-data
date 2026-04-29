#!/usr/bin/env python3
"""
================================================================================
STEP 1: SPY Daily Time Series Collector (Incremental)
================================================================================

Fetches daily OHLCV data for SPY from Yahoo Finance with incremental updates.
Only collects daily interval data (no weekly/monthly) for focused SPY analysis.

Author: SignalsAlpha
Version: 1.0 (SPY Edition)
Date: 2026-04-29

================================================================================
PURPOSE
================================================================================

This module fetches daily price data for SPY (S&P 500 ETF) from Yahoo Finance.
It implements incremental collection - only fetching new data since the last
update, making it efficient for daily runs.

Key differences from full version:
- Single ticker (SPY only)
- Daily interval only (no weekly/monthly)
- Simplified logic, no batch processing needed

================================================================================
INCREMENTAL COLLECTION
================================================================================

Smart Updates:
    - Checks existing CSV file for last recorded date
    - Calculates days since last update
    - Fetches only the gap period (not full history)
    - Merges new data with existing records
    - Removes duplicates automatically

Fetch Period Optimization:
    - 0-5 days missing: Fetches 5 days
    - 6-30 days missing: Fetches 1 month
    - 31-90 days missing: Fetches 3 months
    - 90+ days missing: Fetches full history (max available)

================================================================================
OUTPUT FORMAT
================================================================================

Directory Structure:
    data/time_series/SPY/SPY_1d.csv

CSV Schema:
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

Full/Incremental Collection:
    python step1_time_series_collector.py

Force Full Refresh:
    python step1_time_series_collector.py --full

================================================================================
"""

import pandas as pd
import yfinance as yf
import pytz
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
import argparse


# Configuration
DATA_DIR = Path("data/time_series")
TICKER = "SPY"
INTERVAL = "1d"
RATE_LIMIT_DELAY = 1.0  # Seconds between API calls


def get_last_date_from_csv(csv_path: Path) -> Optional[datetime]:
    """Get the last date from existing CSV file."""
    if not csv_path.exists():
        return None
    
    try:
        df = pd.read_csv(csv_path)
        if 'date' not in df.columns or df.empty:
            return None
        
        # Parse last date with UTC timezone
        last_date = pd.to_datetime(df['date'].iloc[-1], utc=True)
        return last_date.to_pydatetime()
    except Exception as e:
        print(f"Warning: Could not read existing CSV: {e}")
        return None


def calculate_fetch_period(last_date: Optional[datetime]) -> str:
    """Determine appropriate fetch period based on last update date."""
    if last_date is None:
        print("No existing data found. Fetching full history...")
        return "max"
    
    # Use UTC for comparison
    now = datetime.now(pytz.UTC)
    days_missing = (now - last_date).days
    
    print(f"Last data date: {last_date.strftime('%Y-%m-%d')}")
    print(f"Days since last update: {days_missing}")
    
    # Determine fetch period
    if days_missing <= 0:
        print("Data is up to date. No fetch needed.")
        return None
    elif days_missing <= 5:
        print(f"Fetching last 5 days...")
        return "5d"
    elif days_missing <= 30:
        print(f"Fetching last 1 month...")
        return "1mo"
    elif days_missing <= 90:
        print(f"Fetching last 3 months...")
        return "3mo"
    else:
        print(f"Fetching full history...")
        return "max"


def fetch_spy_data(period: str) -> Optional[pd.DataFrame]:
    """Fetch SPY data from Yahoo Finance."""
    try:
        print(f"\nFetching SPY data (period: {period})...")
        
        ticker = yf.Ticker(TICKER)
        df = ticker.history(period=period, interval=INTERVAL)
        
        if df.empty:
            print("ERROR: No data received from Yahoo Finance")
            return None
        
        # Reset index to make date a column
        df.reset_index(inplace=True)
        
        # Standardize column names to lowercase
        df.columns = [col.lower().replace(' ', '_') for col in df.columns]
        
        # Ensure date column is datetime with UTC
        df['date'] = pd.to_datetime(df['date'], utc=True)
        
        # Sort by date
        df.sort_values('date', inplace=True)
        
        print(f"Fetched {len(df)} rows")
        print(f"Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
        
        return df
        
    except Exception as e:
        print(f"ERROR fetching SPY data: {e}")
        return None


def merge_with_existing(new_df: pd.DataFrame, csv_path: Path) -> pd.DataFrame:
    """Merge new data with existing CSV data."""
    if not csv_path.exists():
        return new_df
    
    try:
        # Read existing data
        existing_df = pd.read_csv(csv_path)
        existing_df['date'] = pd.to_datetime(existing_df['date'], utc=True)
        
        print(f"Existing data: {len(existing_df)} rows")
        
        # Combine and remove duplicates (keep last occurrence)
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        combined.drop_duplicates(subset=['date'], keep='last', inplace=True)
        combined.sort_values('date', inplace=True)
        
        print(f"Combined data: {len(combined)} rows")
        print(f"New rows added: {len(combined) - len(existing_df)}")
        
        return combined
        
    except Exception as e:
        print(f"Warning: Could not merge with existing data: {e}")
        print("Using new data only...")
        return new_df


def save_data(df: pd.DataFrame, csv_path: Path):
    """Save data to CSV file."""
    # Ensure directory exists
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    df.to_csv(csv_path, index=False)
    
    print(f"\nSaved to: {csv_path}")
    print(f"Total rows: {len(df)}")
    print(f"Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")


def collect_spy_data(full_refresh: bool = False):
    """Main collection function for SPY daily data."""
    
    print("=" * 70)
    print("STEP 1: SPY Daily Time Series Collection")
    print("=" * 70)
    print(f"Ticker: {TICKER}")
    print(f"Interval: {INTERVAL}")
    print(f"Mode: {'Full Refresh' if full_refresh else 'Incremental Update'}")
    print("=" * 70)
    
    # Setup paths
    ticker_dir = DATA_DIR / TICKER
    csv_path = ticker_dir / f"{TICKER}_{INTERVAL}.csv"
    
    # Determine fetch period
    if full_refresh:
        fetch_period = "max"
        print("\nFull refresh requested. Fetching maximum available history...")
    else:
        last_date = get_last_date_from_csv(csv_path)
        fetch_period = calculate_fetch_period(last_date)
    
    if fetch_period is None:
        print("\n" + "=" * 70)
        print("Data is already up to date. No fetch needed.")
        print("=" * 70)
        return
    
    # Fetch data
    df = fetch_spy_data(fetch_period)
    
    if df is None or df.empty:
        print("\nERROR: Failed to fetch data")
        return
    
    # Merge with existing if incremental
    if not full_refresh and csv_path.exists():
        df = merge_with_existing(df, csv_path)
    
    # Save data
    save_data(df, csv_path)
    
    # Summary
    print("\n" + "=" * 70)
    print("Collection Complete")
    print("=" * 70)
    print(f"Output: {csv_path}")
    print(f"Total records: {len(df)}")
    print(f"Latest close: ${df['close'].iloc[-1]:.2f}")
    print(f"Latest volume: {df['volume'].iloc[-1]:,}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SPY Daily Time Series Collector")
    parser.add_argument("--full", action="store_true", help="Force full refresh (ignore incremental)")
    
    args = parser.parse_args()
    
    collect_spy_data(full_refresh=args.full)
