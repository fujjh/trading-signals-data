#!/usr/bin/env python3
"""
================================================================================
STEP 8: Signal History Tracker - Step 7
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
    
    # Technical Analysis Data:
    rsi,rsi_condition,macd,macd_signal,macd_cross_type,macd_cross_strength,stoch_k,stoch_d,stoch_cross_type,stoch_cross_strength
    
    # Crossover Rationale:
    trix,trix_signal,trix_cross_type,ema_12,ema_26,ema_trend,sma_20,sma_50,sma_trend
    
    # Price Levels & Targets:
    vwap,bb_upper,bb_lower,atr,stop_loss,take_profit,support_level_1,resistance_level_1
    
    # Signal Components (JSON):
    buy_score_breakdown,sell_score_breakdown
    
    Example:
    AAPL,BUY,85,150.25,152.30,12,3,45.2,oversold,0.45,0.38,crossover,8,18.5,22.3,crossover,9

Change Log (JSON):
    {
        "date": "2026-04-18",
        "new_signals": [{
            "ticker": "NVDA",
            "signal": "BUY",
            "confidence": 85,
            "factors": {
                "primary": "MACD_crossover_from_deeply_oversold",
                "secondary": ["Stochastic_oversold_cross", "RSI_45"],
                "trend": "SMA_above",
                "momentum": "TRIX_positive_cross"
            },
            "price_target": 165.30,
            "stop_loss": 142.50
        }],
        "closed_signals": [...],
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
        },
        "crossover_analysis": {
            "macd_crossovers": 156,
            "stoch_crossovers": 203,
            "trix_crossovers": 89,
            "avg_cross_strength": 6.8
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


def extract_signal_rationale(row):
    """
    Extract the primary rationale and contributing factors for a signal
    
    Returns a dictionary with:
    - primary_factor: The main reason for the signal
    - contributing_factors: List of secondary factors
    - price_targets: Dict with stop_loss and take_profit
    - technical_context: Summary of technical conditions
    """
    rationale = {
        'primary_factor': None,
        'contributing_factors': [],
        'technical_context': {},
        'price_targets': {}
    }
    
    # Determine primary factor based on crossover data
    if row.get('macd_cross_type') == 'crossover' and row.get('macd_cross_strength', 0) >= 6:
        rationale['primary_factor'] = 'MACD_strong_bullish_crossover'
        rationale['contributing_factors'].append(f"MACD_cross_from_{row.get('macd_cross_location', 'unknown')}")
    elif row.get('macd_cross_type') == 'crossunder' and row.get('macd_cross_strength', 0) >= 6:
        rationale['primary_factor'] = 'MACD_strong_bearish_crossunder'
        rationale['contributing_factors'].append(f"MACD_cross_from_{row.get('macd_cross_location', 'unknown')}")
    elif row.get('stoch_slow_cross') == 'crossover' and row.get('stoch_slow_strength', 0) >= 7:
        rationale['primary_factor'] = 'Stochastic_deeply_oversold_crossover'
        rationale['contributing_factors'].append(f"Stoch_context_{row.get('stoch_slow_context', 'unknown')}")
    elif row.get('stoch_slow_cross') == 'crossunder' and row.get('stoch_slow_strength', 0) >= 7:
        rationale['primary_factor'] = 'Stochastic_deeply_overbought_crossunder'
        rationale['contributing_factors'].append(f"Stoch_context_{row.get('stoch_slow_context', 'unknown')}")
    elif row.get('trix_cross_type') == 'crossover' and row.get('trix_cross_strength', 0) >= 6:
        rationale['primary_factor'] = 'TRIX_bullish_crossover'
    elif row.get('trix_cross_type') == 'crossunder' and row.get('trix_cross_strength', 0) >= 6:
        rationale['primary_factor'] = 'TRIX_bearish_crossunder'
    
    # Add trend analysis
    if pd.notna(row.get('sma_20')) and pd.notna(row.get('sma_50')) and pd.notna(row.get('close')):
        if row['close'] > row['sma_20'] > row['sma_50']:
            rationale['technical_context']['trend'] = 'strong_uptrend'
            rationale['contributing_factors'].append('SMA_bullish_alignment')
        elif row['close'] < row['sma_20'] < row['sma_50']:
            rationale['technical_context']['trend'] = 'strong_downtrend'
            rationale['contributing_factors'].append('SMA_bearish_alignment')
    
    # Add RSI context
    if pd.notna(row.get('rsi')):
        if row['rsi'] < 30:
            rationale['technical_context']['rsi'] = 'oversold'
            if not rationale['primary_factor']:
                rationale['primary_factor'] = 'RSI_oversold'
            rationale['contributing_factors'].append(f"RSI_{row['rsi']:.1f}")
        elif row['rsi'] > 70:
            rationale['technical_context']['rsi'] = 'overbought'
            if not rationale['primary_factor']:
                rationale['primary_factor'] = 'RSI_overbought'
            rationale['contributing_factors'].append(f"RSI_{row['rsi']:.1f}")
    
    # Add price targets if available
    if pd.notna(row.get('stop_loss')):
        rationale['price_targets']['stop_loss'] = row['stop_loss']
    if pd.notna(row.get('take_profit')):
        rationale['price_targets']['take_profit'] = row['take_profit']
    if pd.notna(row.get('close')):
        rationale['price_targets']['entry_price'] = row['close']
    
    # Calculate risk/reward ratio
    if rationale['price_targets'].get('stop_loss') and rationale['price_targets'].get('take_profit') and rationale['price_targets'].get('entry_price'):
        entry = rationale['price_targets']['entry_price']
        stop = rationale['price_targets']['stop_loss']
        target = rationale['price_targets']['take_profit']
        risk = abs(entry - stop)
        reward = abs(target - entry)
        if risk > 0:
            rationale['price_targets']['risk_reward_ratio'] = reward / risk
    
    return rationale


def load_current_signals():
    """Load all current signals from signals_timeframe with enhanced rationale tracking"""
    print("Loading current signals with rationale...")
    
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
                
                # Extract rationale for each signal
                rationales = []
                for _, row in df.iterrows():
                    rationale = extract_signal_rationale(row)
                    rationales.append(rationale)
                
                df['rationale_json'] = [json.dumps(r) for r in rationales]
                df['primary_factor'] = [r['primary_factor'] for r in rationales]
                df['contributing_factors'] = [', '.join(r['contributing_factors']) for r in rationales]
                
                all_signals.append(df)
            except Exception as e:
                print(f"  Warning: Could not read {ticker}: {e}")
    
    if all_signals:
        combined = pd.concat(all_signals, ignore_index=True)
        print(f"  Loaded {len(combined)} signals from {len(all_signals)} tickers")
        print(f"  With crossover/rationale data")
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
            
            # Print rationale for new signals
            if len(new_buys) > 0 and 'primary_factor' in new_buys.columns:
                print("\n    Top BUY rationales:")
                for _, row in new_buys.head(5).iterrows():
                    factor = row.get('primary_factor', 'N/A')
                    contrib = row.get('contributing_factors', 'N/A')
                    print(f"      {row['ticker']}: {factor} [{contrib}]")
            
            if len(new_sells) > 0 and 'primary_factor' in new_sells.columns:
                print("\n    Top SELL rationales:")
                for _, row in new_sells.head(5).iterrows():
                    factor = row.get('primary_factor', 'N/A')
                    contrib = row.get('contributing_factors', 'N/A')
                    print(f"      {row['ticker']}: {factor} [{contrib}]")
            
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
    
    # Calculate crossover statistics
    crossover_stats = calculate_crossover_stats()
    
    # Save summary
    summary_file = HISTORY_DIR / f"summary_{datetime.now().strftime('%Y-%m-%d')}.json"
    summary = {
        'generated_at': datetime.now().isoformat(),
        'total_snapshots': len(snapshots),
        'holding_stats': holding_stats,
        'crossover_stats': crossover_stats
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n✓ Summary saved: {summary_file}")


def calculate_crossover_stats():
    """Calculate crossover statistics from latest signals"""
    print("\nCalculating crossover statistics...")
    
    today = datetime.now().strftime('%Y-%m-%d')
    today_file = HISTORY_DIR / f"signals_{today}.csv"
    
    if not today_file.exists():
        return {}
    
    try:
        df = pd.read_csv(today_file)
        daily_df = df[df['interval'] == '1d']
        
        stats = {
            'total_signals': len(daily_df),
            'macd_crossovers': 0,
            'macd_crossunders': 0,
            'stoch_crossovers': 0,
            'stoch_crossunders': 0,
            'trix_crossovers': 0,
            'trix_crossunders': 0,
            'avg_macd_cross_strength': 0,
            'avg_stoch_cross_strength': 0,
            'avg_trix_cross_strength': 0
        }
        
        # Count crossovers
        if 'macd_cross_type' in daily_df.columns:
            stats['macd_crossovers'] = len(daily_df[daily_df['macd_cross_type'] == 'crossover'])
            stats['macd_crossunders'] = len(daily_df[daily_df['macd_cross_type'] == 'crossunder'])
            macd_strength = daily_df[daily_df['macd_cross_type'].notna()]['macd_cross_strength']
            if len(macd_strength) > 0:
                stats['avg_macd_cross_strength'] = macd_strength.mean()
        
        if 'stoch_slow_cross' in daily_df.columns:
            stats['stoch_crossovers'] = len(daily_df[daily_df['stoch_slow_cross'] == 'crossover'])
            stats['stoch_crossunders'] = len(daily_df[daily_df['stoch_slow_cross'] == 'crossunder'])
            stoch_strength = daily_df[daily_df['stoch_slow_cross'].notna()]['stoch_slow_strength']
            if len(stoch_strength) > 0:
                stats['avg_stoch_cross_strength'] = stoch_strength.mean()
        
        if 'trix_cross_type' in daily_df.columns:
            stats['trix_crossovers'] = len(daily_df[daily_df['trix_cross_type'] == 'crossover'])
            stats['trix_crossunders'] = len(daily_df[daily_df['trix_cross_type'] == 'crossunder'])
            trix_strength = daily_df[daily_df['trix_cross_type'].notna()]['trix_cross_strength']
            if len(trix_strength) > 0:
                stats['avg_trix_cross_strength'] = trix_strength.mean()
        
        print(f"  MACD crossovers: {stats['macd_crossovers']}")
        print(f"  Stochastic crossovers: {stats['stoch_crossovers']}")
        print(f"  TRIX crossovers: {stats['trix_crossovers']}")
        
        return stats
        
    except Exception as e:
        print(f"  Warning: Could not calculate crossover stats: {e}")
        return {}


def main():
    """Main execution"""
    print("="*60)
    print("Signal History Tracker - Step 7")
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
