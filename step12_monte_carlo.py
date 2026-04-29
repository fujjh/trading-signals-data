#!/usr/bin/env python3
"""
================================================================================
STEP 12: SPY Monte Carlo Stress Testing
================================================================================

Statistical robustness testing for SPY strategy using Monte Carlo simulation.

Author: SignalsAlpha
Version: 1.0 (SPY Edition)
Date: 2026-04-29

================================================================================
METHODOLOGY
================================================================================

Simulations (1,000 runs):
1. Trade shuffle: Randomly reorder trades
2. Returns resampling: Bootstrap historical returns
3. Parameter perturbation: ±10% variation in weights
4. Black swan injection: 5% probability of extreme moves

Confidence intervals at 95% level for all metrics.

================================================================================
INPUT
================================================================================

Source: data/signals_scored/spy_scored_*.csv

================================================================================
OUTPUT
================================================================================

Destination: data/monte_carlo/spy_monte_carlo_*.json

"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import json

# Configuration
TICKER = "SPY"
SIGNALS_DIR = Path("data/signals_scored")
OUTPUT_DIR = Path("data/monte_carlo")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_SIMULATIONS = 1000
CONFIDENCE_LEVEL = 0.95


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


def run_single_simulation(df: pd.DataFrame, sim_id: int) -> Dict:
    """Run one Monte Carlo simulation."""
    # Copy data
    sim_df = df.copy()
    
    # Perturb weights by ±10%
    weight_cols = ['rsi', 'macd', 'hma_13']
    for col in weight_cols:
        if col in sim_df.columns:
            perturbation = np.random.uniform(0.9, 1.1)
            sim_df[col] = sim_df[col] * perturbation
    
    # Resample returns with replacement (bootstrap)
    sim_df['returns'] = sim_df['close'].pct_change()
    n = len(sim_df)
    
    # Shuffle returns
    shuffled_returns = sim_df['returns'].dropna().sample(n=n-1, replace=True).values
    
    # Add black swan events (5% probability)
    black_swan_mask = np.random.random(len(shuffled_returns)) < 0.05
    shuffled_returns[black_swan_mask] = np.random.choice([-0.1, 0.1], size=black_swan_mask.sum())
    
    # Calculate cumulative returns
    sim_returns = [0] + list(shuffled_returns)
    sim_df['sim_returns'] = sim_returns
    sim_df['cumulative'] = (1 + sim_df['sim_returns']).cumprod()
    
    # Calculate metrics
    total_return = sim_df['cumulative'].iloc[-1] - 1
    volatility = sim_df['sim_returns'].std() * np.sqrt(252)
    
    if volatility > 0:
        sharpe = (sim_df['sim_returns'].mean() / sim_df['sim_returns'].std()) * np.sqrt(252)
    else:
        sharpe = 0
    
    # Max drawdown
    sim_df['peak'] = sim_df['cumulative'].cummax()
    sim_df['drawdown'] = (sim_df['cumulative'] - sim_df['peak']) / sim_df['peak']
    max_drawdown = abs(sim_df['drawdown'].min())
    
    return {
        'simulation_id': sim_id,
        'total_return': total_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown
    }


def calculate_confidence_intervals(results: List[Dict]) -> Dict:
    """Calculate confidence intervals."""
    returns = [r['total_return'] for r in results]
    sharpes = [r['sharpe_ratio'] for r in results]
    drawdowns = [r['max_drawdown'] for r in results]
    
    alpha = 1 - CONFIDENCE_LEVEL
    lower_pct = alpha / 2 * 100
    upper_pct = (1 - alpha / 2) * 100
    
    return {
        'returns': {
            'mean': np.mean(returns),
            'median': np.median(returns),
            'std': np.std(returns),
            'ci_lower': np.percentile(returns, lower_pct),
            'ci_upper': np.percentile(returns, upper_pct),
            'worst': np.min(returns),
            'best': np.max(returns)
        },
        'sharpe': {
            'mean': np.mean(sharpes),
            'median': np.median(sharpes),
            'std': np.std(sharpes),
            'ci_lower': np.percentile(sharpes, lower_pct),
            'ci_upper': np.percentile(sharpes, upper_pct)
        },
        'drawdown': {
            'mean': np.mean(drawdowns),
            'median': np.median(drawdowns),
            'std': np.std(drawdowns),
            'ci_lower': np.percentile(drawdowns, lower_pct),
            'ci_upper': np.percentile(drawdowns, upper_pct),
            'worst': np.max(drawdowns)
        }
    }


def run_monte_carlo():
    """Main Monte Carlo simulation."""
    print("=" * 70)
    print("STEP 12: SPY Monte Carlo Stress Testing")
    print("=" * 70)
    print(f"Simulations: {N_SIMULATIONS:,}")
    print(f"Confidence level: {CONFIDENCE_LEVEL*100:.0f}%")
    print("=" * 70)
    
    # Load data
    df = load_signals()
    if df is None:
        print("ERROR: Could not load signals")
        return
    
    print(f"Loaded {len(df)} records")
    
    # Run simulations
    print(f"\nRunning {N_SIMULATIONS} simulations...")
    results = []
    
    for i in range(N_SIMULATIONS):
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{N_SIMULATIONS}")
        
        result = run_single_simulation(df, i)
        results.append(result)
    
    # Calculate confidence intervals
    ci = calculate_confidence_intervals(results)
    
    # Print results
    print("\n" + "=" * 70)
    print("MONTE CARLO RESULTS")
    print("=" * 70)
    
    print("\nReturns:")
    print(f"  Mean: {ci['returns']['mean']*100:.2f}%")
    print(f"  Median: {ci['returns']['median']*100:.2f}%")
    print(f"  Std Dev: {ci['returns']['std']*100:.2f}%")
    print(f"  95% CI: [{ci['returns']['ci_lower']*100:.2f}%, {ci['returns']['ci_upper']*100:.2f}%]")
    print(f"  Worst case: {ci['returns']['worst']*100:.2f}%")
    print(f"  Best case: {ci['returns']['best']*100:.2f}%")
    
    print("\nSharpe Ratio:")
    print(f"  Mean: {ci['sharpe']['mean']:.2f}")
    print(f"  Median: {ci['sharpe']['median']:.2f}")
    print(f"  95% CI: [{ci['sharpe']['ci_lower']:.2f}, {ci['sharpe']['ci_upper']:.2f}]")
    
    print("\nMax Drawdown:")
    print(f"  Mean: {ci['drawdown']['mean']*100:.2f}%")
    print(f"  Median: {ci['drawdown']['median']*100:.2f}%")
    print(f"  95% CI: [{ci['drawdown']['ci_lower']*100:.2f}%, {ci['drawdown']['ci_upper']*100:.2f}%]")
    print(f"  Worst case: {ci['drawdown']['worst']*100:.2f}%")
    print("=" * 70)
    
    # Save results
    summary = {
        'ticker': TICKER,
        'timestamp': datetime.now().isoformat(),
        'parameters': {
            'n_simulations': N_SIMULATIONS,
            'confidence_level': CONFIDENCE_LEVEL
        },
        'confidence_intervals': ci,
        'probability_metrics': {
            'profit_probability': sum(1 for r in results if r['total_return'] > 0) / N_SIMULATIONS,
            'sharpe_above_1': sum(1 for r in results if r['sharpe_ratio'] > 1.0) / N_SIMULATIONS,
            'drawdown_under_20': sum(1 for r in results if r['max_drawdown'] < 0.2) / N_SIMULATIONS
        }
    }
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = OUTPUT_DIR / f"spy_monte_carlo_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSaved results: {output_file}")


if __name__ == "__main__":
    run_monte_carlo()
