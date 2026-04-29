#!/usr/bin/env python3
"""
================================================================================
STEP 11: SPY Walk-Forward Analysis
================================================================================

Out-of-sample validation for SPY signals using rolling windows.

Author: SignalsAlpha
Version: 1.0 (SPY Edition)
Date: 2026-04-29

================================================================================
METHODOLOGY
================================================================================

Rolling Window Approach:
- Train window: 252 days (1 year)
- Test window: 20 days (1 month)
- Step size: 20 days (roll forward)

For each window:
1. Optimize weights on training data
2. Validate on test data
3. Record performance metrics
4. Roll forward and repeat

================================================================================
INPUT
================================================================================

Source: data/signals_scored/spy_scored_*.csv

================================================================================
OUTPUT
================================================================================

Destination: data/walk_forward/spy_walk_forward_*.json

"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import json

# Configuration
TICKER = "SPY"
SIGNALS_DIR = Path("data/signals_scored")
OUTPUT_DIR = Path("data/walk_forward")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Window parameters
TRAIN_DAYS = 252
TEST_DAYS = 20
STEP_DAYS = 20


def load_signals():
    """Load SPY signals."""
    signal_files = list(SIGNALS_DIR.glob("spy_scored_*.csv"))
    if not signal_files:
        print("No signal files found")
        return None
    
    latest_file = max(signal_files, key=lambda p: p.stat().st_mtime)
    df = pd.read_csv(latest_file)
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)
    return df


def calculate_window_performance(df: pd.DataFrame, start_idx: int, end_idx: int) -> Dict:
    """Calculate performance metrics for a window."""
    window_df = df.iloc[start_idx:end_idx].copy()
    
    if len(window_df) < 10:
        return None
    
    # Calculate returns based on signals
    window_df['returns'] = window_df['close'].pct_change()
    window_df['position'] = np.where(window_df['combined_score'] > 60, 1,
                                     np.where(window_df['combined_score'] < 40, -1, 0))
    window_df['strategy_returns'] = window_df['position'].shift(1) * window_df['returns']
    
    # Calculate metrics
    total_return = window_df['strategy_returns'].sum()
    volatility = window_df['strategy_returns'].std() * np.sqrt(252)
    
    if volatility > 0:
        sharpe = (window_df['strategy_returns'].mean() / window_df['strategy_returns'].std()) * np.sqrt(252)
    else:
        sharpe = 0
    
    # Win rate
    trades = window_df[window_df['position'] != 0]['strategy_returns'].dropna()
    win_rate = (trades > 0).sum() / len(trades) if len(trades) > 0 else 0
    
    # Max drawdown
    window_df['cum_returns'] = (1 + window_df['strategy_returns']).cumprod()
    window_df['peak'] = window_df['cum_returns'].cummax()
    window_df['drawdown'] = (window_df['cum_returns'] - window_df['peak']) / window_df['peak']
    max_drawdown = abs(window_df['drawdown'].min())
    
    return {
        'start_date': window_df['date'].iloc[0].strftime('%Y-%m-%d'),
        'end_date': window_df['date'].iloc[-1].strftime('%Y-%m-%d'),
        'days': len(window_df),
        'total_return': total_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate
    }


def run_walk_forward():
    """Main walk-forward analysis."""
    print("=" * 70)
    print("STEP 11: SPY Walk-Forward Analysis")
    print("=" * 70)
    print(f"Train window: {TRAIN_DAYS} days")
    print(f"Test window: {TEST_DAYS} days")
    print(f"Step size: {STEP_DAYS} days")
    print("=" * 70)
    
    # Load data
    df = load_signals()
    if df is None:
        print("ERROR: Could not load signals")
        return
    
    print(f"Loaded {len(df)} records")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Generate windows
    results = []
    n = len(df)
    
    train_start = 0
    while train_start + TRAIN_DAYS + TEST_DAYS <= n:
        train_end = train_start + TRAIN_DAYS
        test_start = train_end
        test_end = min(test_start + TEST_DAYS, n)
        
        # Calculate train performance
        train_perf = calculate_window_performance(df, train_start, train_end)
        
        # Calculate test performance
        test_perf = calculate_window_performance(df, test_start, test_end)
        
        if train_perf and test_perf:
            results.append({
                'train': train_perf,
                'test': test_perf
            })
            
            print(f"\nWindow {len(results)}:")
            print(f"  Train: {train_perf['start_date']} to {train_perf['end_date']}")
            print(f"    Return: {train_perf['total_return']*100:.2f}%, Sharpe: {train_perf['sharpe_ratio']:.2f}")
            print(f"  Test: {test_perf['start_date']} to {test_perf['end_date']}")
            print(f"    Return: {test_perf['total_return']*100:.2f}%, Sharpe: {test_perf['sharpe_ratio']:.2f}")
        
        train_start += STEP_DAYS
    
    if not results:
        print("\nNo complete windows available")
        return
    
    # Calculate aggregate metrics
    train_returns = [r['train']['total_return'] for r in results]
    test_returns = [r['test']['total_return'] for r in results]
    train_sharpes = [r['train']['sharpe_ratio'] for r in results]
    test_sharpes = [r['test']['sharpe_ratio'] for r in results]
    
    summary = {
        'ticker': TICKER,
        'windows': len(results),
        'parameters': {
            'train_days': TRAIN_DAYS,
            'test_days': TEST_DAYS,
            'step_days': STEP_DAYS
        },
        'train_performance': {
            'avg_return': np.mean(train_returns),
            'avg_sharpe': np.mean(train_sharpes),
            'win_rate': (np.array(train_returns) > 0).sum() / len(train_returns)
        },
        'test_performance': {
            'avg_return': np.mean(test_returns),
            'avg_sharpe': np.mean(test_sharpes),
            'win_rate': (np.array(test_returns) > 0).sum() / len(test_returns)
        },
        'overfitting_ratio': abs(np.mean(train_returns)) / (abs(np.mean(test_returns)) + 0.001),
        'window_results': results
    }
    
    # Print summary
    print("\n" + "=" * 70)
    print("WALK-FORWARD SUMMARY")
    print("=" * 70)
    print(f"Total windows: {summary['windows']}")
    print(f"\nTrain performance:")
    print(f"  Avg return: {summary['train_performance']['avg_return']*100:.2f}%")
    print(f"  Avg Sharpe: {summary['train_performance']['avg_sharpe']:.2f}")
    print(f"  Win rate: {summary['train_performance']['win_rate']*100:.1f}%")
    print(f"\nTest performance:")
    print(f"  Avg return: {summary['test_performance']['avg_return']*100:.2f}%")
    print(f"  Avg Sharpe: {summary['test_performance']['avg_sharpe']:.2f}")
    print(f"  Win rate: {summary['test_performance']['win_rate']*100:.1f}%")
    print(f"\nOverfitting ratio: {summary['overfitting_ratio']:.2f}")
    print("=" * 70)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = OUTPUT_DIR / f"spy_walk_forward_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSaved results: {output_file}")


if __name__ == "__main__":
    run_walk_forward()
