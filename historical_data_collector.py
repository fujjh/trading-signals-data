#!/usr/bin/env python3
"""
Historical Stock Data Collector
Pulls 5 years of daily OHLCV data for all tickers from stock_ticker_base.csv
Saves each ticker as individual CSV in data/historical_stock_data/
"""

import os
import sys
import time
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List
import yfinance as yf

# Configuration
BATCH_SIZE = 100
DELAY_BETWEEN_CALLS = 0.5  # seconds - faster since just price data
DATA_DIR = "data/historical_stock_data"
CHECKPOINT_FILE = f"{DATA_DIR}/.collection_checkpoint.json"

def load_ticker_list() -> List[str]:
    """Load the comprehensive stock ticker list"""
    df = pd.read_csv("data/tickers/stock_ticker_base.csv")
    return df['symbol'].tolist()

def load_checkpoint() -> dict:
    """Load checkpoint if exists"""
    import json
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {'processed': [], 'last_index': -1, 'count': 0}

def save_checkpoint(processed: List[str], last_index: int, count: int):
    """Save collection checkpoint"""
    import json
    os.makedirs(DATA_DIR, exist_ok=True)
    checkpoint = {
        'processed': processed[-100:],
        'last_index': last_index,
        'count': count,
        'last_update': datetime.now().isoformat()
    }
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f)

def fetch_historical_data(symbol: str, max_retries: int = 3) -> Optional[pd.DataFrame]:
    """Fetch up to 5 years of historical daily OHLCV data"""
    
    for attempt in range(max_retries):
        try:
            time.sleep(DELAY_BETWEEN_CALLS)
            
            ticker = yf.Ticker(symbol)
            
            # Try to get 5 years of data
            df = ticker.history(period="5y", interval="1d")
            
            if len(df) < 10:  # Need at least 10 days of data
                return None
            
            # Ensure we have the required columns
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_cols:
                if col not in df.columns:
                    return None
            
            # Reset index to make Date a column
            df = df.reset_index()
            
            # Select only OHLCV columns
            df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
            
            # Round prices to 4 decimal places
            for col in ['Open', 'High', 'Low', 'Close']:
                df[col] = df[col].round(4)
            
            # Convert Volume to integer
            df['Volume'] = df['Volume'].astype(int)
            
            # Format date as string
            df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
            
            return df
            
        except Exception as e:
            if "Rate limited" in str(e) or "Too Many Requests" in str(e):
                wait_time = 60 * (attempt + 1)
                print(f"    Rate limited! Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                # Some tickers may not have historical data
                return None
    
    return None

def save_ticker_csv(symbol: str, df: pd.DataFrame) -> str:
    """Save ticker data as CSV"""
    os.makedirs(DATA_DIR, exist_ok=True)
    
    file_path = f"{DATA_DIR}/{symbol}.csv"
    df.to_csv(file_path, index=False)
    
    return file_path

def main():
    print("=" * 80)
    print("HISTORICAL STOCK DATA COLLECTOR")
    print("=" * 80)
    print(f"Target: Up to 5 years of daily OHLCV data")
    print(f"Output: Individual CSV files in {DATA_DIR}/")
    print(f"Columns: Date, Open, High, Low, Close, Volume")
    print("=" * 80)
    
    tickers = load_ticker_list()
    print(f"\nTotal tickers to process: {len(tickers)}")
    
    checkpoint = load_checkpoint()
    start_index = checkpoint['last_index'] + 1
    processed = checkpoint.get('processed', [])
    total_collected = checkpoint.get('count', 0)
    
    print(f"Resuming from index {start_index} ({len(processed)} already processed, {total_collected} with data)")
    
    batch_count = 0
    errors = 0
    
    for i in range(start_index, len(tickers), BATCH_SIZE):
        batch_count += 1
        batch = tickers[i:i + BATCH_SIZE]
        batch_collected = 0
        
        print(f"\n{'='*80}")
        print(f"Batch {batch_count}: Processing {len(batch)} tickers ({i+1} to {min(i+BATCH_SIZE, len(tickers))})")
        print(f"{'='*80}")
        
        for idx, symbol in enumerate(batch):
            current_idx = i + idx
            
            # Check if already exists
            expected_file = f"{DATA_DIR}/{symbol}.csv"
            if os.path.exists(expected_file):
                print(f"  [{current_idx+1}/{len(tickers)}] {symbol}... Already exists ✓")
                processed.append(symbol)
                total_collected += 1
                continue
            
            print(f"  [{current_idx+1}/{len(tickers)}] {symbol}... Fetching...", end=" ")
            
            try:
                df = fetch_historical_data(symbol)
                
                if df is not None and len(df) > 0:
                    file_path = save_ticker_csv(symbol, df)
                    processed.append(symbol)
                    total_collected += 1
                    batch_collected += 1
                    
                    days = len(df)
                    start_date = df['Date'].iloc[0]
                    end_date = df['Date'].iloc[-1]
                    
                    print(f"✓ Saved {days} days ({start_date} to {end_date})")
                else:
                    print(f"✗ No data")
                    
            except Exception as e:
                errors += 1
                print(f"✗ Error: {str(e)[:50]}")
            
            # Save checkpoint every 25 tickers
            if (current_idx + 1) % 25 == 0:
                save_checkpoint(processed, current_idx, total_collected)
        
        # Save checkpoint after each batch
        save_checkpoint(processed, min(i + BATCH_SIZE, len(tickers)) - 1, total_collected)
        
        print(f"\n  Batch {batch_count} complete: {batch_collected} tickers with data")
        print(f"  Running total: {total_collected}/{len(processed)} tickers with historical data")
        
        # Brief pause between batches
        if i + BATCH_SIZE < len(tickers):
            print(f"\n  Pausing briefly before next batch...")
            time.sleep(2)
    
    # Final summary
    print("\n" + "=" * 80)
    print("COLLECTION COMPLETE")
    print("=" * 80)
    print(f"Total tickers processed: {len(processed)}")
    print(f"Tickers with historical data: {total_collected}")
    print(f"Errors encountered: {errors}")
    print(f"\nData saved to: {DATA_DIR}/")
    print(f"Files: {total_collected} CSV files")
    
    # Show sample of files
    import glob
    files = glob.glob(f"{DATA_DIR}/*.csv")
    if files:
        print(f"\nSample files:")
        for f in sorted(files)[:5]:
            size = os.path.getsize(f) / 1024  # KB
            print(f"  - {os.path.basename(f)} ({size:.1f} KB)")

if __name__ == "__main__":
    main()
