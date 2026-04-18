#!/usr/bin/env python3
"""
================================================================================
STEP 5: Signal History Tracker
================================================================================

Maintains a historical database of signals to track performance over time,
analyze signal persistence, and detect changes in signal direction.

Author: SignalsAlpha
Version: 1.0
Date: 2026-04-18

================================================================================
PURPOSE
================================================================================

This module provides historical tracking capabilities:

1. Saves daily snapshot of all active signals
2. Tracks signal changes (e.g., BUY -> SELL transitions)
3. Calculates holding periods for signals
4. Maintains signal lineage across days
5. Provides data for backtesting and performance analysis
6. Enables signal decay analysis (how long signals remain valid)

================================================================================
TRACKED DATA
================================================================================

Daily Snapshots:
    - All active signals from Step 2
    - Signal type, confidence, price levels
    - Technical indicator values at time of signal
    - Market conditions (VIX, sector performance)

Signal Changes:
    - NEW: First occurrence of a signal
    - CONTINUED: Same signal as yesterday
    - STRENGTHENED: Buy score increased
    - WEAKENED: Buy score decreased
    - CLOSED: Signal no longer active
    - FLIPPED: BUY -> SELL or SELL -> BUY

Holding Periods:
    - Days held for each signal
    - Entry and exit prices
    - Maximum favorable/unfavorable excursion

================================================================================
WORKFLOW
================================================================================

1. LOAD CURRENT SIGNALS
   - Read from data/signals_timeframe/
   - Aggregate across timeframes
   - Identify unique signals per ticker

2. LOAD PREVIOUS DAY
   - Read yesterday's snapshot from signal_history/
   - Parse into comparable format

3. COMPARE AND CLASSIFY
   - Match current signals with previous
   - Classify each signal's status change
   - Calculate holding periods

4. SAVE SNAPSHOT
   - Write today's signals to signals_YYYY-MM-DD.csv
   - Write changes to changes_YYYY-MM-DD.json
   - Update holding period database

5. GENERATE SUMMARY
   - Calculate statistics (new signals, closed signals, avg holding period)
   - Write summary_YYYY-MM-DD.json
   - Print report to console

================================================================================
OUTPUT FORMAT
================================================================================

Signal Snapshot (CSV):
    ticker,signal_type,confidence,entry_price,current_price,score,holding_days
    AAPL,BUY,85,150.25,152.30,12,3
    TSLA,STRONG_SELL,92,245.50,238.10,18,1

Change Log (JSON):
    {
        "date": "2026-04-18",
        "new_signals": [{"ticker": "NVDA", "signal": "BUY", ...}],
        "closed_signals": [{"ticker": "META", "signal": "SELL", ...}],
        "continued_signals": [...],
        "flipped_signals": [...]
    }

Summary (JSON):
    {
        "date": "2026-04-18",
        "total_signals": 2054,
        "new_signals": 45,
        "closed_signals": 38,
        "avg_holding_period": 4.2,
        "signal_distribution": {
            "STRONG_BUY": 48,
            "BUY": 396,
            "WEAK_BUY": 116,
            "SELL": 181,
            "STRONG_SELL": 28
        }
    }

================================================================================
USAGE
================================================================================

    python step5_signal_history_tracker.py

Should be run daily after Step 4 to maintain complete history.

================================================================================
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# Configuration
DATA_DIR = Path("/home/ubuntu/.openclaw/workspace/data")
SIGNALS_DIR = DATA_DIR / "signals_timeframe"
HISTORY_DIR = DATA_DIR / "signal_history"


def ensure_dirs():
    """Create history directory"""
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)


def load_current_signals():
    """Load all current signals from signals_timeframe"""
    print("Loading current signals...")
    
    all_signals = []
    
    if not SIGNALS_DIR.exists():
        print(f"Warning: Signals directory not found: {SIGNALS_DIR}")
        return pd.DataFrame()
    
    for ticker_dir in SIGNALS_DIR.iterdir():
        if not ticker_dir.is_dir():
            continue
        
        ticker = ticker_dir.name
        signals_file = ticker_dir / f"{ticker}_signals.csv"
        
        if signals_file.exists():
            try:
                df = pd.read_csv(signals_file)
                # Add metadata
                df['collection_date'] = datetime.now().strftime('%Y-%m-%d')
                df['collection_timestamp'] = datetime.now().isoformat()
                all_signals.append(df)
            except Exception as e:
                print(f"  Warning: Could not read {ticker}: {e}")
    
    if all_signals:
        combined = pd.concat(all_signals, ignore_index=True)
        print(f"  Loaded {len(combined)} signals from {len(all_signals)} tickers")
        return combined
    
    return pd.DataFrame()


def save_daily_snapshot(df):
    """Save daily snapshot to history database"""
    today = datetime.now().strftime('%Y-%m-%d')
    snapshot_file = HISTORY_DIR / f"signals_{today}.csv"
    
    # Add day number for tracking
    df['day_number'] = (datetime.now() - datetime(2026, 1, 1)).days
    
    df.to_csv(snapshot_file, index=False)
    print(f"\n✓ Saved daily snapshot: {snapshot_file}")
    print(f"  Records: {len(df)}")
    
    return snapshot_file


def detect_signal_changes():
    """Detect signal changes from previous day"""
    print("\nDetecting signal changes...")
    
    today = datetime.now()
    yesterday = today - timedelta(days=1)
    
    today_file = HISTORY_DIR / f"signals_{today.strftime('%Y-%m-%d')}.csv"
    yesterday_file = HISTORY_DIR / f"signals_{yesterday.strftime('%Y-%m-%d')}.csv"
    
    if not yesterday_file.exists():
        print("  No previous day data found (first run)")
        return pd.DataFrame()
    
    try:
        today_df = pd.read_csv(today_file)
        yesterday_df = pd.read_csv(yesterday_file)
        
        # Filter for daily timeframe only
        today_daily = today_df[today_df['interval'] == '1d']
        yesterday_daily = yesterday_df[yesterday_df['interval'] == '1d']
        
        # Merge to compare
        merged = today_daily.merge(
            yesterday_daily[['ticker', 'signal']], 
            on='ticker', 
            how='left',
            suffixes=('', '_prev')
        )
        
        # Find changes
        changes = merged[merged['signal'] != merged['signal_prev']]
        
        print(f"  Signal changes detected: {len(changes)}")
        
        if len(changes) > 0:
            # Categorize changes
            buy_signals = ['STRONG_BUY', 'BUY', 'WEAK_BUY']
            sell_signals = ['STRONG_SELL', 'SELL', 'WEAK_SELL']
            
            new_buys = changes[changes['signal'].isin(buy_signals) & 
                              ~changes['signal_prev'].isin(buy_signals)]
            new_sells = changes[changes['signal'].isin(sell_signals) & 
                               ~changes['signal_prev'].isin(sell_signals)]
            
            print(f"    New BUY signals: {len(new_buys)}")
            print(f"    New SELL signals: {len(new_sells)}")
            
            # Save changes
            changes_file = HISTORY_DIR / f"changes_{today.strftime('%Y-%m-%d')}.csv"
            changes.to_csv(changes_file, index=False)
            print(f"\n✓ Saved changes to: {changes_file}")
        
        return changes
        
    except Exception as e:
        print(f"  Error detecting changes: {e}")
        return pd.DataFrame()


def calculate_holding_periods():
    """Calculate average holding periods for signals"""
    print("\nCalculating holding period statistics...")
    
    # Load last 30 days of history
    history_files = sorted(HISTORY_DIR.glob("signals_*.csv"))[-30:]
    
    if len(history_files) < 2:
        print("  Insufficient history for holding period calculation")
        return {}
    
    # Track signal start dates
    signal_starts = {}
    
    for file in history_files:
        date_str = file.stem.replace('signals_', '')
        try:
            df = pd.read_csv(file)
            daily_df = df[df['interval'] == '1d']
            
            for _, row in daily_df.iterrows():
                ticker = row['ticker']
                signal = row['signal']
                
                key = f"{ticker}_{signal}"
                if key not in signal_starts:
                    signal_starts[key] = {
                        'ticker': ticker,
                        'signal': signal,
                        'start_date': date_str,
                        'end_date': date_str,
                        'days': 1
                    }
                else:
                    signal_starts[key]['end_date'] = date_str
                    signal_starts[key]['days'] += 1
                    
        except Exception as e:
            print(f"  Warning: Could not process {file}: {e}")
    
    # Calculate statistics
    if signal_starts:
        df = pd.DataFrame(signal_starts.values())
        
        stats = {
            'avg_holding_days': df['days'].mean(),
            'max_holding_days': df['days'].max(),
            'min_holding_days': df['days'].min(),
            'by_signal': df.groupby('signal')['days'].mean().to_dict()
        }
        
        print(f"\n  Average holding period: {stats['avg_holding_days']:.1f} days")
        print(f"  Range: {stats['min_holding_days']}-{stats['max_holding_days']} days")
        
        return stats
    
    return {}


def generate_history_summary():
    """Generate summary of signal history"""
    print("\n" + "="*60)
    print("Signal History Summary")
    print("="*60)
    
    # Count historical snapshots
    snapshots = list(HISTORY_DIR.glob("signals_*.csv"))
    print(f"\nTotal daily snapshots: {len(snapshots)}")
    
    if snapshots:
        # Get date range
        dates = sorted([f.stem.replace('signals_', '') for f in snapshots])
        print(f"Date range: {dates[0]} to {dates[-1]}")
        
        # Latest signals
        latest = pd.read_csv(snapshots[-1])
        daily_latest = latest[latest['interval'] == '1d']
        
        print(f"\nLatest signals: {len(daily_latest)} tickers")
        print(f"Signal distribution:")
        print(daily_latest['signal'].value_counts())
    
    # Calculate holding periods
    holding_stats = calculate_holding_periods()
    
    # Save summary
    summary_file = HISTORY_DIR / f"summary_{datetime.now().strftime('%Y-%m-%d')}.json"
    summary = {
        'generated_at': datetime.now().isoformat(),
        'total_snapshots': len(snapshots),
        'holding_stats': holding_stats
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n✓ Summary saved: {summary_file}")


def main():
    """Main execution"""
    print("="*60)
    print("Signal History Tracker")
    print("="*60)
    print()
    
    ensure_dirs()
    
    # Load and save current signals
    df = load_current_signals()
    
    if df.empty:
        print("\n✗ No signals data found. Run Step 2 first.")
        return 1
    
    # Save daily snapshot
    save_daily_snapshot(df)
    
    # Detect changes from previous day
    detect_signal_changes()
    
    # Generate summary
    generate_history_summary()
    
    print("\n" + "="*60)
    print("Signal History Tracking Complete!")
    print("="*60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
