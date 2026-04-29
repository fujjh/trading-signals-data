#!/usr/bin/env python3
"""
================================================================================
STEP 7: SPY Signal History Tracker
================================================================================

Tracks SPY signal history over time for backtesting analysis.

Author: SignalsAlpha
Version: 1.0 (SPY Edition)
Date: 2026-04-29

================================================================================
PURPOSE
================================================================================

Tracks daily SPY signals to enable:
1. Historical signal analysis
2. Performance backtesting
3. Signal persistence tracking

================================================================================
INPUT
================================================================================

Source: data/signals_scored/spy_scored_*.csv

================================================================================
OUTPUT
================================================================================

Destination: data/signal_history/spy_signals_YYYY-MM-DD.csv
             data/signal_history/spy_summary_YYYY-MM-DD.json

"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime, timedelta
import glob

# Configuration
TICKER = "SPY"
SIGNALS_DIR = Path("data/signals_scored")
OUTPUT_DIR = Path("data/signal_history")


def load_latest_signals():
    """Load the latest SPY scored signals."""
    signal_files = list(SIGNALS_DIR.glob("spy_scored_*.csv"))
    if not signal_files:
        print(f"No signal files found in {SIGNALS_DIR}")
        return None
    
    latest_file = max(signal_files, key=lambda p: p.stat().st_mtime)
    print(f"Loading signals from: {latest_file}")
    
    df = pd.read_csv(latest_file)
    df['date'] = pd.to_datetime(df['date'])
    return df


def save_daily_snapshot(df):
    """Save today's SPY signals as a snapshot."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    today = datetime.now().strftime('%Y-%m-%d')
    snapshot_file = OUTPUT_DIR / f"spy_signals_{today}.csv"
    
    # Select key columns
    cols = ['date', 'close', 'technical_score', 'combined_score', 'signal',
            'rsi', 'macd', 'hma_13', 'elder_impulse']
    
    # Only save last 30 days for the snapshot
    recent = df.tail(30).copy()
    
    # Filter to columns that exist
    output_cols = [c for c in cols if c in recent.columns]
    
    recent[output_cols].to_csv(snapshot_file, index=False)
    print(f"✓ Saved daily snapshot: {snapshot_file}")
    
    return snapshot_file


def generate_summary(df):
    """Generate summary statistics for SPY signals."""
    today = datetime.now().strftime('%Y-%m-%d')
    
    # Get latest signal
    latest = df.iloc[-1]
    
    # Calculate signal distribution (last 100 days)
    recent = df.tail(100)
    signal_counts = recent['signal'].value_counts().to_dict()
    
    summary = {
        'date': today,
        'ticker': TICKER,
        'total_records': len(df),
        'latest_price': float(latest['close']),
        'latest_signal': latest['signal'],
        'latest_score': float(latest['combined_score']),
        'latest_rsi': float(latest['rsi']) if 'rsi' in latest else None,
        'signal_distribution_100d': signal_counts,
        'date_range': {
            'start': df['date'].min().strftime('%Y-%m-%d'),
            'end': df['date'].max().strftime('%Y-%m-%d')
        }
    }
    
    # Save summary
    summary_file = OUTPUT_DIR / f"spy_summary_{today}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Saved summary: {summary_file}")
    return summary


def track_spy_history():
    """Main function to track SPY signal history."""
    print("=" * 70)
    print("STEP 7: SPY Signal History Tracker")
    print("=" * 70)
    
    # Load signals
    df = load_latest_signals()
    if df is None or df.empty:
        print("No signals to track")
        return
    
    print(f"Loaded {len(df)} signal records")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Save snapshot
    save_daily_snapshot(df)
    
    # Generate summary
    summary = generate_summary(df)
    
    # Print summary
    print("\n" + "=" * 70)
    print("Signal History Summary")
    print("=" * 70)
    print(f"Total records: {summary['total_records']}")
    print(f"Latest price: ${summary['latest_price']:.2f}")
    print(f"Latest signal: {summary['latest_signal']}")
    print(f"Latest score: {summary['latest_score']:.1f}")
    print(f"Latest RSI: {summary['latest_rsi']:.1f}")
    print("\nSignal distribution (last 100 days):")
    for signal, count in summary['signal_distribution_100d'].items():
        print(f"  {signal}: {count}")
    print("=" * 70)


if __name__ == "__main__":
    track_spy_history()
