#!/usr/bin/env python3
"""
================================================================================
STEP 9: SPY Focused Grid Search (Seed 93 Refinement)
================================================================================

Grid search around the best configuration from 100-seed GA (Seed 93).
Only searches the most sensitive parameters to avoid dimensionality explosion.

Author: SignalsAlpha
Version: 3.3 (Grid Search Edition)
Date: 2026-04-29
================================================================================
STRATEGY:
================================================================================

1. Fix weights from Seed 93 as base
2. Grid search thresholds (long_entry, exit, short_entry) - 5×4×4 = 80 combos
3. Fine-tune top 3 weights (MACD, SMA_200, RSI) with ±10% variation - 3×3×3 = 27 combos
4. Total: 107 evaluations, ~2-3 minutes
================================================================================
"""

import json
import random
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from itertools import product
from typing import Dict, Tuple

# Configuration
TICKER = "SPY"
TECHNICAL_DIR = Path("data/technical_analysis") / TICKER
OUTPUT_DIR = Path("data/optimizer")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Base configuration from Seed 93 (100-seed best)
BASE_CONFIG = {
    'macd_weight': 8.32,
    'hma_weight': 4.74,
    'rsi_weight': 5.41,
    'stoch_weight': 3.21,
    'sma_weight': 1.26,
    'ema_weight': 4.19,
    'mfi_weight': 4.01,
    'bb_weight': 1.65,
    'high_52w_weight': 0.94,
    'low_52w_weight': 2.55,
    'sma_200_weight': 8.02,
    'long_entry': 55,  # CORRECTED from actual seed 93
    'exit': 50,        # CORRECTED from actual seed 93
    'short_entry': 25, # CORRECTED from actual seed 93
}

# Grid search ranges - around actual seed 93 values
THRESHOLD_GRID = {
    'long_entry': [53, 54, 55, 56, 57],  # Around 55
    'exit': [48, 49, 50, 51, 52],       # Around 50
    'short_entry': [23, 24, 25, 26, 27], # Around 25
}

WEIGHT_VARIATIONS = {
    'macd_weight': [7.5, 8.32, 9.0],        # ±10%
    'sma_200_weight': [7.2, 8.02, 8.8],     # ±10%
    'rsi_weight': [4.9, 5.41, 6.0],         # ±10%
}


def load_spy_data():
    """Load SPY technical data."""
    tech_file = TECHNICAL_DIR / f"{TICKER}_1d_technical.csv"
    if not tech_file.exists():
        print(f"ERROR: Technical file not found: {tech_file}")
        return None
    
    df = pd.read_csv(tech_file)
    df['date'] = pd.to_datetime(df['date'])
    df['returns'] = df['close'].pct_change().fillna(0)
    return df


def calculate_signal_score(df: pd.DataFrame, individual: Dict) -> pd.Series:
    """Calculate technical score with all indicators."""
    score = pd.Series(50.0, index=df.index)
    
    orig_keys = ['macd', 'hma', 'rsi', 'stoch', 'sma', 'ema', 'mfi', 'bb']
    new_keys = ['high_52w', 'low_52w', 'sma_200']
    
    orig_weights = {k: max(0, individual.get(f'{k}_weight', 0)) for k in orig_keys}
    new_weights = {k: max(0, individual.get(f'{k}_weight', 0)) for k in new_keys}
    
    orig_total = sum(orig_weights.values()) or 1
    new_total = sum(new_weights.values()) or 1
    
    # Original indicators
    if 'rsi' in df.columns:
        score += (50 - df['rsi']) / 50 * orig_weights['rsi'] / orig_total * 30
    
    if 'macd' in df.columns and 'macd_signal' in df.columns:
        macd_diff = df['macd'] - df['macd_signal']
        score += np.clip(macd_diff * 5, -20, 20) * orig_weights['macd'] / orig_total
    
    if 'hma_13' in df.columns:
        hma_diff = (df['close'] - df['hma_13']) / df['hma_13'] * 100
        score += hma_diff * orig_weights['hma'] / orig_total
    
    if 'mfi' in df.columns:
        score += (50 - df['mfi']) / 50 * orig_weights['mfi'] / orig_total * 30
    
    if 'sma_20' in df.columns:
        trend = (df['close'] > df['sma_20']).astype(float) * 20
        score += trend * (orig_weights['sma'] + orig_weights['ema']) / orig_total
    
    # New indicators
    if 'pct_from_52w_high' in df.columns:
        high_signal = np.where(df['pct_from_52w_high'] > -5, 15,
                      np.where(df['pct_from_52w_high'] > -15, 5,
                      np.where(df['pct_from_52w_high'] > -30, -5, -15)))
        score += high_signal * new_weights['high_52w'] / new_total * 0.5
    
    if 'pct_from_52w_low' in df.columns:
        low_signal = np.clip(df['pct_from_52w_low'] / 50, -20, 20)
        score += low_signal * new_weights['low_52w'] / new_total * 0.5
    
    if 'sma_200' in df.columns:
        sma200_diff = (df['close'] - df['sma_200']) / df['sma_200'] * 100
        sma200_signal = np.clip(sma200_diff * 2, -20, 20)
        score += sma200_signal * new_weights['sma_200'] / new_total * 0.5
    
    return np.clip(score, 0, 100)


def calculate_fitness(df: pd.DataFrame, individual: Dict) -> Tuple[float, Dict]:
    """Calculate fitness for an individual."""
    df_copy = df.copy()
    df_copy['score'] = calculate_signal_score(df, individual)
    df_copy['returns'] = df_copy['close'].pct_change()
    
    n = len(df_copy)
    position = np.zeros(n)
    current_pos = 0.0
    
    long_entry = individual['long_entry']
    exit_level = individual['exit']
    short_entry = individual['short_entry']
    
    scores = df_copy['score'].values
    for i in range(n):
        s = scores[i]
        if s >= long_entry:
            current_pos = 1.0
        elif s <= short_entry:
            current_pos = -0.5
        elif s <= exit_level:
            current_pos = 0.0
        position[i] = current_pos
    
    returns = df_copy['returns'].values
    strategy_returns = np.zeros(n)
    strategy_returns[1:] = position[:-1] * returns[1:]
    
    total_return = strategy_returns.sum()
    trades = np.sum(np.abs(np.diff(position)) > 0.01)
    
    if trades < 20:
        return 0.05, {'reason': f'Too few trades: {trades}', 'trades': trades}
    if trades > 500:
        return 0.05, {'reason': f'Too many trades: {trades}', 'trades': trades}
    
    volatility = np.std(strategy_returns) * np.sqrt(252)
    sharpe = (np.mean(strategy_returns) * 252) / volatility if volatility > 0 else 0
    
    cumulative = np.cumprod(1 + strategy_returns)
    peak = np.maximum.accumulate(cumulative)
    drawdown = (peak - cumulative) / peak
    max_dd = np.max(drawdown)
    
    winning_days = np.sum(strategy_returns > 0)
    total_trading_days = np.sum(np.abs(strategy_returns) > 0.0001)
    win_rate = winning_days / total_trading_days if total_trading_days > 0 else 0
    
    return_score = min(abs(total_return) / 2.0, 1.0)
    dd_score = max(0, 1 - max_dd / 0.30)
    activity_score = min(trades / 100, 1.0)
    
    fitness = (
        return_score * 0.40 +
        dd_score * 0.30 +
        min(sharpe / 1.0, 1.0) * 0.15 +
        win_rate * 0.10 +
        activity_score * 0.05
    )
    
    details = {
        'total_return': total_return,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'win_rate': win_rate,
        'trades': trades,
        'volatility': volatility
    }
    
    return fitness, details


def run_focused_grid_search():
    """Run focused grid search around Seed 93 configuration."""
    print("=" * 70)
    print("STEP 9: SPY Focused Grid Search (Seed 93 Refinement)")
    print("=" * 70)
    
    df = load_spy_data()
    if df is None:
        return
    
    print(f"Loaded {len(df)} rows")
    print(f"\nBase configuration (Seed 93):")
    print(f"  Fitness: 0.7566")
    print(f"  Return: 335.08%, Sharpe: 0.86, MaxDD: 16.4%")
    print("=" * 70)
    
    # Test baseline first
    print("\nTesting baseline configuration...")
    base_fitness, base_details = calculate_fitness(df, BASE_CONFIG)
    print(f"Baseline fitness: {base_fitness:.4f}")
    
    results = []
    
    # Phase 1: Grid search thresholds
    print("\n" + "=" * 70)
    print("PHASE 1: Threshold Grid Search")
    print("=" * 70)
    
    threshold_combos = list(product(
        THRESHOLD_GRID['long_entry'],
        THRESHOLD_GRID['exit'],
        THRESHOLD_GRID['short_entry']
    ))
    
    print(f"Testing {len(threshold_combos)} threshold combinations...")
    
    for i, (long_e, exit_l, short_e) in enumerate(threshold_combos):
        config = BASE_CONFIG.copy()
        config['long_entry'] = long_e
        config['exit'] = exit_l
        config['short_entry'] = short_e
        
        fitness, details = calculate_fitness(df, config)
        results.append({
            'phase': 'threshold',
            'config': config.copy(),
            'fitness': fitness,
            'details': details
        })
        
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i+1}/{len(threshold_combos)}")
    
    # Phase 2: Weight fine-tuning with best thresholds
    print("\n" + "=" * 70)
    print("PHASE 2: Weight Fine-Tuning")
    print("=" * 70)
    
    # Find best thresholds from Phase 1
    threshold_results = [r for r in results if r['phase'] == 'threshold']
    best_threshold = max(threshold_results, key=lambda x: x['fitness'])
    
    print(f"Best thresholds found:")
    print(f"  long_entry: {best_threshold['config']['long_entry']}")
    print(f"  exit: {best_threshold['config']['exit']}")
    print(f"  short_entry: {best_threshold['config']['short_entry']}")
    print(f"  Fitness: {best_threshold['fitness']:.4f}")
    
    # Grid search top 3 weights
    weight_combos = list(product(
        WEIGHT_VARIATIONS['macd_weight'],
        WEIGHT_VARIATIONS['sma_200_weight'],
        WEIGHT_VARIATIONS['rsi_weight']
    ))
    
    print(f"\nTesting {len(weight_combos)} weight combinations...")
    
    for i, (macd_w, sma200_w, rsi_w) in enumerate(weight_combos):
        config = best_threshold['config'].copy()
        config['macd_weight'] = macd_w
        config['sma_200_weight'] = sma200_w
        config['rsi_weight'] = rsi_w
        
        fitness, details = calculate_fitness(df, config)
        results.append({
            'phase': 'weight',
            'config': config.copy(),
            'fitness': fitness,
            'details': details
        })
        
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(weight_combos)}")
    
    # Find best overall
    best = max(results, key=lambda x: x['fitness'])
    
    # Print results
    print("\n" + "=" * 70)
    print("GRID SEARCH RESULTS")
    print("=" * 70)
    
    # Top 10 threshold-only results
    print("\nTop 5 Threshold Combinations:")
    top_thresholds = sorted(threshold_results, key=lambda x: x['fitness'], reverse=True)[:5]
    for i, r in enumerate(top_thresholds, 1):
        c = r['config']
        d = r['details']
        print(f"{i}. long_entry={c['long_entry']}, exit={c['exit']}, short_entry={c['short_entry']} "
              f"→ Fitness: {r['fitness']:.4f}")
    
    # Top 10 overall results
    print("\nTop 10 Overall Results:")
    top_overall = sorted(results, key=lambda x: x['fitness'], reverse=True)[:10]
    for i, r in enumerate(top_overall, 1):
        c = r['config']
        d = r['details']
        print(f"{i}. Fitness: {r['fitness']:.4f} | "
              f"MACD={c['macd_weight']:.1f}, SMA200={c['sma_200_weight']:.1f}, "
              f"Entry={c['long_entry']}, Exit={c['exit']}")
    
    # Best result detailed
    print("\n" + "=" * 70)
    print("BEST CONFIGURATION")
    print("=" * 70)
    
    best_config = best['config']
    best_details = best['details']
    
    print(f"Fitness: {best['fitness']:.4f}")
    print(f"Return: {best_details.get('total_return', 0)*100:.2f}%")
    print(f"Sharpe: {best_details.get('sharpe', 0):.2f}")
    print(f"Max DD: {best_details.get('max_dd', 0)*100:.1f}%")
    print(f"Win Rate: {best_details.get('win_rate', 0)*100:.1f}%")
    print(f"Trades: {best_details.get('trades', 0)}")
    
    print(f"\nThresholds:")
    print(f"  long_entry: {best_config['long_entry']}")
    print(f"  exit: {best_config['exit']}")
    print(f"  short_entry: {best_config['short_entry']}")
    
    print(f"\nWeights:")
    for key in ['macd_weight', 'sma_200_weight', 'rsi_weight', 'hma_weight', 'ema_weight']:
        print(f"  {key}: {best_config[key]:.2f}")
    
    # Comparison with baseline
    print("\n" + "=" * 70)
    print("COMPARISON WITH SEED 93 BASELINE")
    print("=" * 70)
    print(f"Baseline:  {base_fitness:.4f}")
    print(f"Optimized: {best['fitness']:.4f}")
    improvement = (best['fitness'] - base_fitness) / base_fitness * 100
    print(f"Improvement: {improvement:+.2f}%")
    
    # Save results
    output_file = OUTPUT_DIR / f"spy_best_config_grid_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump({
            'ticker': TICKER,
            'base_config': BASE_CONFIG,
            'best_config': {k: float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else v for k, v in best_config.items()},
            'best_fitness': float(best['fitness']),
            'baseline_fitness': float(base_fitness),
            'improvement_pct': float(improvement),
            'top_10_results': [
                {
                    'rank': i+1,
                    'fitness': float(r['fitness']),
                    'config': {k: float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else v for k, v in r['config'].items()},
                    'details': {k: float(v) if isinstance(v, np.floating) else int(v) if isinstance(v, np.integer) else v for k, v in r['details'].items()}
                }
                for i, r in enumerate(top_overall)
            ],
            'generated_at': datetime.now().isoformat()
        }, f, indent=2)
    
    print(f"\nSaved to: {output_file}")
    print("=" * 70)


if __name__ == "__main__":
    run_focused_grid_search()
