#!/usr/bin/env python3
"""
Step 1: Data Collection Pipeline
Pulls comprehensive yfinance data for all tickers in batches.
Saves raw data to disk without any signal logic.
"""

import os
import sys
import json
import time
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, List
import yfinance as yf

# Configuration
BATCH_SIZE = 500
DELAY_BETWEEN_CALLS = 0.8  # seconds
DELAY_BETWEEN_BATCHES = 60  # seconds
DATA_DIR = "data/raw_tickers"
CHECKPOINT_FILE = f"{DATA_DIR}/collection_checkpoint.json"

def load_ticker_list():
    """Load the comprehensive stock ticker list"""
    df = pd.read_csv("data/tickers/stock_ticker_base.csv")
    return df['symbol'].tolist()

def save_raw_data(symbol: str, data: Dict):
    """Save raw yfinance data for a single ticker"""
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Save as individual JSON file
    ticker_file = f"{DATA_DIR}/{symbol}.json"
    
    # Custom JSON encoder to handle Timestamp keys
    def convert_timestamps(obj):
        if isinstance(obj, dict):
            return {str(k) if not isinstance(k, (str, int, float, bool, type(None))) else k: 
                    convert_timestamps(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_timestamps(item) for item in obj]
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif hasattr(obj, 'isoformat'):  # datetime objects
            return obj.isoformat()
        return obj
    
    converted_data = convert_timestamps(data)
    
    with open(ticker_file, 'w') as f:
        json.dump(converted_data, f, indent=2)
    
    return ticker_file

def load_checkpoint():
    """Load checkpoint if exists"""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {'processed': [], 'last_index': -1}

def save_checkpoint(processed: List[str], last_index: int):
    """Save collection checkpoint"""
    checkpoint = {
        'processed': processed[-50:],  # Keep last 50 for memory
        'last_index': last_index,
        'last_update': datetime.now().isoformat(),
        'count': len(processed)
    }
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f)

def fetch_comprehensive_data(symbol: str, max_retries: int = 3) -> Optional[Dict]:
    """Fetch all yfinance data for a ticker"""
    
    for attempt in range(max_retries):
        try:
            time.sleep(DELAY_BETWEEN_CALLS)
            
            ticker = yf.Ticker(symbol)
            
            # Get historical prices (60 days)
            hist = ticker.history(period="60d", interval="1d")
            
            if len(hist) < 20:  # Need minimum data
                return None
            
            # Get info/fundamentals
            try:
                info = ticker.info
            except:
                info = {}
            
            # Get financials
            try:
                financials = ticker.financials
                quarterly_financials = ticker.quarterly_financials
            except:
                financials = None
                quarterly_financials = None
            
            # Get analyst recommendations
            try:
                recommendations = ticker.recommendations
            except:
                recommendations = None
            
            # Get institutional holders
            try:
                institutional_holders = ticker.institutional_holders
            except:
                institutional_holders = None
            
            # Get earnings
            try:
                earnings = ticker.earnings
                quarterly_earnings = ticker.quarterly_earnings
            except:
                earnings = None
                quarterly_earnings = None
            
            # Compile all data
            data = {
                'symbol': symbol,
                'collected_at': datetime.now().isoformat(),
                'historical_prices': hist.to_dict() if hist is not None else None,
                'info': info if info else None,
                'financials': financials.to_dict() if financials is not None else None,
                'quarterly_financials': quarterly_financials.to_dict() if quarterly_financials is not None else None,
                'recommendations': recommendations.to_dict() if recommendations is not None else None,
                'institutional_holders': institutional_holders.to_dict() if institutional_holders is not None else None,
                'earnings': earnings.to_dict() if earnings is not None else None,
                'quarterly_earnings': quarterly_earnings.to_dict() if quarterly_earnings is not None else None
            }
            
            return data
            
        except Exception as e:
            if "Rate limited" in str(e) or "Too Many Requests" in str(e):
                wait_time = 60 * (attempt + 1)
                print(f"    Rate limited! Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                return None
    
    return None

def main():
    print("=" * 80)
    print("STEP 1: DATA COLLECTION PIPELINE")
    print("=" * 80)
    
    tickers = load_ticker_list()
    print(f"Total tickers to process: {len(tickers)}")
    
    checkpoint = load_checkpoint()
    start_index = checkpoint['last_index'] + 1
    processed = checkpoint.get('processed', [])
    
    print(f"Resuming from index {start_index} ({len(processed)} already processed)")
    
    api_calls = 0
    batch_count = 0
    
    for i in range(start_index, len(tickers), BATCH_SIZE):
        batch_count += 1
        batch = tickers[i:i + BATCH_SIZE]
        batch_data = []
        
        print(f"\nBatch {batch_count}: Processing {len(batch)} tickers ({i+1} to {min(i+BATCH_SIZE, len(tickers))})")
        
        for symbol in batch:
            print(f"  Fetching {symbol}...")
            data = fetch_comprehensive_data(symbol)
            
            if data:
                save_raw_data(symbol, data)
                batch_data.append(symbol)
                processed.append(symbol)
                print(f"    ✓ Saved data for {symbol}")
            else:
                print(f"    ✗ No data for {symbol}")
            
            api_calls += 1
            
            # Save checkpoint every 50 tickers
            if len(processed) % 50 == 0:
                save_checkpoint(processed, i + batch.index(symbol) if symbol in batch else i)
        
        # Save checkpoint after each batch
        save_checkpoint(processed, min(i + BATCH_SIZE, len(tickers)) - 1)
        
        print(f"\n  Batch {batch_count} complete: {len(batch_data)} tickers with data")
        print(f"  Total collected: {len(processed)}")
        
        # Wait between batches
        if i + BATCH_SIZE < len(tickers):
            print(f"\n  Waiting {DELAY_BETWEEN_BATCHES}s before next batch...")
            time.sleep(DELAY_BETWEEN_BATCHES)
            api_calls = 0
    
    print("\n" + "=" * 80)
    print("DATA COLLECTION COMPLETE")
    print("=" * 80)
    print(f"Total tickers with data: {len(processed)}")
    print(f"Data saved to: {DATA_DIR}/")

if __name__ == "__main__":
    main()
