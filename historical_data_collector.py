#!/usr/bin/env python3
"""
Historical Stock Data Collector
Pulls 5 years of daily OHLCV data for all tickers from stock_ticker_base.csv
Uses batch approach: 500 tickers/batch, 0.8s delays, checkpointing
Saves each ticker as individual CSV in data/historical_stock_data/
"""

import os
import sys
import json
import time
import pandas as pd
from datetime import datetime
from typing import Optional, List
import yfinance as yf

# Configuration - Same as comprehensive data collection
BATCH_SIZE = 500
DELAY_BETWEEN_CALLS = 0.8  # seconds
DELAY_BETWEEN_BATCHES = 60  # seconds
DATA_DIR = "data/historical_stock_data"
CHECKPOINT_FILE = f"{DATA_DIR}/collection_checkpoint.json"

def load_ticker_list() -> List[str]:
    """Load the comprehensive stock ticker list"""
    df = pd.read_csv("data/tickers/stock_ticker_base.csv")
    return df['symbol'].tolist()

def save_historical_csv(symbol: str, df: pd.DataFrame):
    """Save historical data as CSV for a single ticker"""
    os.makedirs(DATA_DIR, exist_ok=True)
    
    file_path = f"{DATA_DIR}/{symbol}.csv"
    df.to_csv(file_path, index=False)

def load_checkpoint() -> dict:
    """Load checkpoint if exists"""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {'processed': [], 'last_index': -1, 'count': 0}

def save_checkpoint(processed: List[str], last_index: int, count: int):
    """Save collection checkpoint"""
    checkpoint = {
        'processed': processed[-100:],  # Keep last 100 for memory
        'last_index': last_index,
        'count': count,
        'last_update': datetime.now().isoformat()
    }
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f)

def get_alternative_symbols(symbol: str) -> List[str]:
    """Generate alternative symbol formats to try"""
    alternatives = [symbol]  # Original first
    
    # Replace dash with dot (for preferred shares, class shares)
    if '-' in symbol:
        alternatives.append(symbol.replace('-', '.'))
    
    # Try .TO suffix for potential TSX stocks (short tickers without dots)
    if len(symbol) <= 3 and symbol.isalpha() and not '.' in symbol:
        alternatives.append(f"{symbol}.TO")
    
    return alternatives

def fetch_historical_data(symbol: str, max_retries: int = 2) -> Optional[pd.DataFrame]:
    """Fetch up to 5 years of historical daily OHLCV data"""
    
    # Try alternative symbol formats
    alternatives = get_alternative_symbols(symbol)
    
    for alt_symbol in alternatives:
        for attempt in range(max_retries):
            try:
                time.sleep(DELAY_BETWEEN_CALLS)
                
                ticker = yf.Ticker(alt_symbol)
                
                # Try to get maximum available historical data
                # First try 'max', then fall back to '10y', '5y', etc.
                df = None
                for period in ['max', '10y', '5y', '2y', '1y']:
                    try:
                        df = ticker.history(period=period, interval="1d")
                        if len(df) >= 10:
                            break
                    except:
                        continue
                
                if df is not None and len(df) >= 10:
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
                continue
    
    return None

def main():
    print("=" * 80)
    print("HISTORICAL STOCK DATA COLLECTOR")
    print("=" * 80)
    print(f"Target: Up to 5 years of daily OHLCV data")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Delay Between Calls: {DELAY_BETWEEN_CALLS}s")
    print(f"Delay Between Batches: {DELAY_BETWEEN_BATCHES}s")
    print(f"Output: {DATA_DIR}/{{TICKER}}.csv")
    print("=" * 80)
    
    tickers = load_ticker_list()
    print(f"\nTotal tickers to process: {len(tickers)}")
    
    checkpoint = load_checkpoint()
    start_index = checkpoint['last_index'] + 1
    processed = checkpoint.get('processed', [])
    total_collected = checkpoint.get('count', 0)
    
    print(f"Resuming from index {start_index}")
    print(f"Already processed: {len(processed)} tickers")
    print(f"With historical data: {total_collected} tickers")
    print("=" * 80)
    
    api_calls = 0
    batch_count = 0
    
    for i in range(start_index, len(tickers), BATCH_SIZE):
        batch_count += 1
        batch = tickers[i:i + BATCH_SIZE]
        batch_collected = []
        
        print(f"\n{'='*80}")
        print(f"BATCH {batch_count}: Processing {len(batch)} tickers ({i+1} to {min(i+BATCH_SIZE, len(tickers))})")
        print(f"{'='*80}")
        
        for symbol in batch:
            print(f"  Fetching {symbol}...", end=" ")
            
            try:
                df = fetch_historical_data(symbol)
                
                if df is not None and len(df) > 0:
                    save_historical_csv(symbol, df)
                    batch_collected.append(symbol)
                    processed.append(symbol)
                    total_collected += 1
                    
                    days = len(df)
                    start_date = df['Date'].iloc[0]
                    end_date = df['Date'].iloc[-1]
                    
                    print(f"✓ {days} days ({start_date} to {end_date})")
                else:
                    print(f"✗ No data")
                    processed.append(symbol)  # Still mark as processed
                    
            except Exception as e:
                print(f"✗ Error: {str(e)[:40]}")
                processed.append(symbol)  # Still mark as processed
            
            api_calls += 1
            
            # Save checkpoint every 50 tickers
            if len(processed) % 50 == 0:
                save_checkpoint(processed, i + batch.index(symbol) if symbol in batch else i, total_collected)
        
        # Save checkpoint after each batch
        save_checkpoint(processed, min(i + BATCH_SIZE, len(tickers)) - 1, total_collected)
        
        print(f"\n  Batch {batch_count} complete: {len(batch_collected)} tickers with data")
        print(f"  Running total: {total_collected} tickers with historical data")
        
        # Wait between batches (except last)
        if i + BATCH_SIZE < len(tickers):
            print(f"\n  Waiting {DELAY_BETWEEN_BATCHES}s before next batch...")
            time.sleep(DELAY_BETWEEN_BATCHES)
            api_calls = 0
    
    # Final summary
    print("\n" + "=" * 80)
    print("COLLECTION COMPLETE")
    print("=" * 80)
    print(f"Total processed: {len(processed)}")
    print(f"Tickers with historical data: {total_collected}")
    print(f"Data saved to: {DATA_DIR}/")
    
    # Show sample of files
    import glob
    files = sorted(glob.glob(f"{DATA_DIR}/*.csv"))
    if files:
        print(f"\nSample files ({len(files)} total):")
        for f in files[:10]:
            size = os.path.getsize(f) / 1024  # KB
            print(f"  {os.path.basename(f)} ({size:.1f} KB)")

if __name__ == "__main__":
    main()
