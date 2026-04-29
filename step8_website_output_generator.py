#!/usr/bin/env python3
"""
================================================================================
STEP 8: SPY Website Output Generator
================================================================================

Generates JSON output for SPY signals to be displayed on website.

Author: SignalsAlpha
Version: 1.0 (SPY Edition)
Date: 2026-04-29

================================================================================
INPUT
================================================================================

Source: data/signals_scored/spy_scored_*.csv

================================================================================
OUTPUT
================================================================================

Destination: data/website_output/spy_signals.json
             data/website_output/spy_latest.json

"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime

# Configuration
TICKER = "SPY"
SIGNALS_DIR = Path("data/signals_scored")
OUTPUT_DIR = Path("data/website_output")


def load_spy_signals():
    """Load the latest SPY signals."""
    signal_files = list(SIGNALS_DIR.glob("spy_scored_*.csv"))
    if not signal_files:
        print(f"No signal files found in {SIGNALS_DIR}")
        return None
    
    latest_file = max(signal_files, key=lambda p: p.stat().st_mtime)
    print(f"Loading signals from: {latest_file}")
    
    df = pd.read_csv(latest_file)
    df['date'] = pd.to_datetime(df['date'])
    return df


def generate_signals_json(df):
    """Generate complete SPY signals JSON."""
    # Convert to records format
    records = df.to_dict('records')
    
    # Format dates as strings
    for record in records:
        if 'date' in record:
            record['date'] = str(record['date'])
    
    output = {
        'ticker': TICKER,
        'generated_at': datetime.now().isoformat(),
        'total_records': len(records),
        'signals': records
    }
    
    return output


def generate_latest_json(df):
    """Generate latest signal JSON."""
    latest = df.iloc[-1].to_dict()
    
    # Format date
    if 'date' in latest:
        latest['date'] = str(latest['date'])
    
    output = {
        'ticker': TICKER,
        'generated_at': datetime.now().isoformat(),
        'latest': latest
    }
    
    return output


def generate_website_output():
    """Main function to generate website output."""
    print("=" * 70)
    print("STEP 8: SPY Website Output Generator")
    print("=" * 70)
    
    # Load signals
    df = load_spy_signals()
    if df is None or df.empty:
        print("No signals to export")
        return
    
    print(f"Loaded {len(df)} signal records")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Generate complete signals JSON
    signals_json = generate_signals_json(df)
    signals_file = OUTPUT_DIR / "spy_signals.json"
    with open(signals_file, 'w') as f:
        json.dump(signals_json, f, indent=2)
    print(f"✓ Saved: {signals_file}")
    
    # Generate latest signal JSON
    latest_json = generate_latest_json(df)
    latest_file = OUTPUT_DIR / "spy_latest.json"
    with open(latest_file, 'w') as f:
        json.dump(latest_json, f, indent=2)
    print(f"✓ Saved: {latest_file}")
    
    # Print summary
    latest = df.iloc[-1]
    print("\n" + "=" * 70)
    print("Website Output Summary")
    print("=" * 70)
    print(f"Total records: {len(df)}")
    print(f"Latest date: {latest['date']}")
    print(f"Latest price: ${latest['close']:.2f}")
    print(f"Latest signal: {latest['signal']}")
    print(f"Latest score: {latest['combined_score']:.1f}")
    print("=" * 70)


if __name__ == "__main__":
    generate_website_output()
