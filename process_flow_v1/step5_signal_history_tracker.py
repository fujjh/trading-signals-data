#!/usr/bin/env python3
"""
================================================================================
STEP 5: Signal History Tracker
================================================================================
Tracks signal history over time to analyze performance
- Saves daily signals to historical database
- Tracks signal changes (BUY -> SELL, etc.)
- Calculates holding periods
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
