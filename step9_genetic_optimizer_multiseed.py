#!/usr/bin/env python3
"""
================================================================================
STEP 9: SPY Multi-Seed Genetic Algorithm Optimizer v3.2
================================================================================

Multi-seed GA that runs optimization multiple times with different random seeds
and selects the best overall configuration to avoid local minima.

Author: SignalsAlpha
Version: 3.2 (Multi-Seed Edition)
Date: 2026-04-29

================================================================================
MULTI-SEED STRATEGY:
================================================================================

1. Run GA N times with different random seeds
2. Each run explores different regions of the search space
3. Compare final results across all seeds
4. Select best configuration based on fitness + robustness metrics

Benefits:
- Better exploration of global optimum
- Confidence that solution isn't a fluke
- Identifies stable high-performing regions
================================================================================
"""

import json
import random
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, List
import pickle

# Configuration
TICKER = "SPY"
TECHNICAL_DIR = Path("data/technical_analysis") / TICKER
OUTPUT_DIR = Path("data/optimizer")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# GA Parameters (per seed run)
POPULATION_SIZE = 30
GENERATIONS = 50
ELITISM_COUNT = 3
MUTATION_RATE = 0.15

# Multi-seed Parameters
NUM_SEEDS = 3  # Number of independent GA runs
CONVERGENCE_THRESHOLD = 0.01  # Stop early if fitness plateaus


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


def create_individual():
    """Create a random individual with new indicator weights."""
    return {
        # Original indicators
        'macd_weight': random.uniform(0, 10),
        'hma_weight': random.uniform(0, 10),
        'rsi_weight': random.uniform(0, 10),
        'stoch_weight': random.uniform(0, 10),
        'sma_weight': random.uniform(0, 10),
        'ema_weight': random.uniform(0, 10),
        'mfi_weight': random.uniform(0, 10),
        'bb_weight': random.uniform(0, 10),
        
        # NEW: 52-week indicators
        'high_52w_weight': random.uniform(0, 10),
        'low_52w_weight': random.uniform(0, 10),
        'sma_200_weight': random.uniform(0, 10),
        
        # Thresholds
        'long_entry': random.randint(55, 75),
        'exit': random.randint(35, 50),
        'short_entry': random.randint(25, 40),
    }


def calculate_signal_score(df: pd.DataFrame, individual: Dict) -> pd.Series:
    """Calculate technical score with all indicators."""
    score = pd.Series(50.0, index=df.index)
    
    # Separate original and new indicator weights
    orig_keys = ['macd', 'hma', 'rsi', 'stoch', 'sma', 'ema', 'mfi', 'bb']
    new_keys = ['high_52w', 'low_52w', 'sma_200']
    
    orig_weights = {k: max(0, individual.get(f'{k}_weight', 0)) for k in orig_keys}
    new_weights = {k: max(0, individual.get(f'{k}_weight', 0)) for k in new_keys}
    
    orig_total = sum(orig_weights.values()) or 1
    new_total = sum(new_weights.values()) or 1
    
    # ORIGINAL INDICATORS
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
    
    # NEW INDICATORS
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
    
    # Position sizing
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
    
    # Calculate strategy returns
    returns = df_copy['returns'].values
    strategy_returns = np.zeros(n)
    strategy_returns[1:] = position[:-1] * returns[1:]
    
    # Metrics
    total_return = strategy_returns.sum()
    trades = np.sum(np.abs(np.diff(position)) > 0.01)
    
    if trades < 20:
        return 0.05, {'reason': f'Too few trades: {trades}', 'trades': trades}
    if trades > 500:
        return 0.05, {'reason': f'Too many trades: {trades}', 'trades': trades}
    
    # Risk metrics
    volatility = np.std(strategy_returns) * np.sqrt(252)
    sharpe = (np.mean(strategy_returns) * 252) / volatility if volatility > 0 else 0
    
    cumulative = np.cumprod(1 + strategy_returns)
    peak = np.maximum.accumulate(cumulative)
    drawdown = (peak - cumulative) / peak
    max_dd = np.max(drawdown)
    
    winning_days = np.sum(strategy_returns > 0)
    total_trading_days = np.sum(np.abs(strategy_returns) > 0.0001)
    win_rate = winning_days / total_trading_days if total_trading_days > 0 else 0
    
    # Fitness calculation
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


def mutate(individual: Dict) -> Dict:
    """Mutate individual."""
    mutated = individual.copy()
    
    # Original weights
    for key in ['macd_weight', 'hma_weight', 'rsi_weight', 'stoch_weight', 
                'sma_weight', 'ema_weight', 'mfi_weight', 'bb_weight']:
        if random.random() < MUTATION_RATE:
            mutated[key] = max(0, min(10, individual[key] + random.uniform(-1, 1)))
    
    # New indicator weights
    for key in ['high_52w_weight', 'low_52w_weight', 'sma_200_weight']:
        if random.random() < MUTATION_RATE:
            mutated[key] = max(0, min(10, individual.get(key, 5) + random.uniform(-1, 1)))
    
    # Thresholds
    if random.random() < MUTATION_RATE:
        mutated['long_entry'] = max(55, min(75, individual['long_entry'] + random.randint(-3, 3)))
    if random.random() < MUTATION_RATE:
        mutated['exit'] = max(35, min(50, individual['exit'] + random.randint(-3, 3)))
    if random.random() < MUTATION_RATE:
        mutated['short_entry'] = max(25, min(40, individual['short_entry'] + random.randint(-3, 3)))
    
    return mutated


def crossover(p1: Dict, p2: Dict) -> Tuple[Dict, Dict]:
    """Crossover two parents."""
    c1, c2 = {}, {}
    for key in p1:
        if isinstance(p1[key], float):
            alpha = random.uniform(0, 1)
            c1[key] = alpha * p1[key] + (1 - alpha) * p2[key]
            c2[key] = (1 - alpha) * p1[key] + alpha * p2[key]
        else:
            c1[key] = random.choice([p1[key], p2[key]])
            c2[key] = random.choice([p1[key], p2[key]])
    return c1, c2


def run_single_optimization(df: pd.DataFrame, seed: int) -> Dict:
    """Run GA with a specific random seed."""
    random.seed(seed)
    np.random.seed(seed)
    
    print(f"\n{'='*70}")
    print(f"SEED {seed}: Starting optimization")
    print(f"{'='*70}")
    
    # Init population
    population = [create_individual() for _ in range(POPULATION_SIZE)]
    
    best_fitness = 0
    best_individual = None
    best_details = None
    fitness_history = []
    
    for gen in range(GENERATIONS):
        # Evaluate
        results = []
        for ind in population:
            fit, det = calculate_fitness(df, ind)
            results.append((fit, ind, det))
        
        results.sort(key=lambda x: x[0], reverse=True)
        
        # Update best
        if results[0][0] > best_fitness:
            best_fitness = results[0][0]
            best_individual = results[0][1].copy()
            best_details = results[0][2].copy()
            print(f"Gen {gen}: New best fitness = {best_fitness:.4f}")
        
        fitness_history.append(best_fitness)
        
        # Early stopping check (convergence)
        if gen > 20 and len(fitness_history) >= 10:
            recent = fitness_history[-10:]
            if max(recent) - min(recent) < CONVERGENCE_THRESHOLD:
                print(f"Early stopping at gen {gen} - converged")
                break
        
        if (gen + 1) % 10 == 0:
            avg_fit = sum([r[0] for r in results]) / len(results)
            print(f"Gen {gen+1}: Best={best_fitness:.4f}, Avg={avg_fit:.4f}")
        
        # Next generation
        new_pop = [results[i][1] for i in range(ELITISM_COUNT)]
        
        while len(new_pop) < POPULATION_SIZE:
            candidates = random.sample(results[:POPULATION_SIZE//2], 6)
            p1 = max(candidates[:3], key=lambda x: x[0])[1]
            p2 = max(candidates[3:], key=lambda x: x[0])[1]
            
            c1, c2 = crossover(p1, p2)
            new_pop.append(mutate(c1))
            if len(new_pop) < POPULATION_SIZE:
                new_pop.append(mutate(c2))
        
        population = new_pop
    
    return {
        'seed': seed,
        'fitness': best_fitness,
        'individual': best_individual,
        'details': best_details,
        'generations': len(fitness_history)
    }


def run_multi_seed_optimizer():
    """Run multiple GA optimizations with different seeds and pick best."""
    print("=" * 70)
    print("STEP 9: SPY Multi-Seed Genetic Optimizer v3.2")
    print("=" * 70)
    print(f"Number of seeds: {NUM_SEEDS}")
    print(f"Population per seed: {POPULATION_SIZE}")
    print(f"Max generations per seed: {GENERATIONS}")
    print("=" * 70)
    
    df = load_spy_data()
    if df is None:
        return
    
    print(f"Loaded {len(df)} rows")
    
    # Benchmark
    first = df['close'].iloc[0]
    last = df['close'].iloc[-1]
    bh_return = (last - first) / first
    print(f"Buy-and-Hold: {bh_return*100:.2f}% ({first:.2f} → {last:.2f})")
    print("=" * 70)
    
    # Run multiple seeds
    all_results = []
    for seed in range(NUM_SEEDS):
        result = run_single_optimization(df, seed)
        all_results.append(result)
    
    # Sort by fitness
    all_results.sort(key=lambda x: x['fitness'], reverse=True)
    
    # Print comparison
    print(f"\n{'='*70}")
    print("MULTI-SEED RESULTS COMPARISON")
    print(f"{'='*70}")
    print(f"{'Seed':<6} {'Fitness':<10} {'Return':<10} {'Sharpe':<8} {'MaxDD':<8} {'Trades':<8}")
    print("-" * 70)
    for r in all_results:
        det = r['details']
        print(f"{r['seed']:<6} {r['fitness']:<10.4f} {det.get('total_return', 0)*100:<10.2f}% "
              f"{det.get('sharpe', 0):<8.2f} {det.get('max_dd', 0)*100:<8.1f}% "
              f"{det.get('trades', 0):<8}")
    
    # Select best
    best = all_results[0]
    
    # Calculate robustness score (how consistent is it across seeds?)
    fitness_values = [r['fitness'] for r in all_results]
    fitness_std = np.std(fitness_values)
    fitness_mean = np.mean(fitness_values)
    
    print(f"\n{'='*70}")
    print("ROBUSTNESS ANALYSIS")
    print(f"{'='*70}")
    print(f"Best seed: {best['seed']}")
    print(f"Mean fitness across seeds: {fitness_mean:.4f}")
    print(f"Std dev across seeds: {fitness_std:.4f}")
    print(f"Range: {min(fitness_values):.4f} - {max(fitness_values):.4f}")
    
    if fitness_std < 0.05:
        print("✓ HIGH ROBUSTNESS: Results are consistent across seeds")
    elif fitness_std < 0.1:
        print("○ MODERATE ROBUSTNESS: Some variance across seeds")
    else:
        print("✗ LOW ROBUSTNESS: High variance - consider more seeds or parameter tuning")
    
    # Save best configuration
    output_file = OUTPUT_DIR / f"spy_best_config_multiseed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump({
            'ticker': TICKER,
            'num_seeds': NUM_SEEDS,
            'best_seed': best['seed'],
            'fitness': float(best['fitness']),
            'fitness_mean': float(fitness_mean),
            'fitness_std': float(fitness_std),
            'config': {k: float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else v for k, v in best['individual'].items()},
            'details': {k: float(v) if isinstance(v, np.floating) else int(v) if isinstance(v, np.integer) else v for k, v in best['details'].items()},
            'all_seeds_results': [
                {
                    'seed': r['seed'],
                    'fitness': float(r['fitness']),
                    'generations': r['generations'],
                    'details': {k: float(v) if isinstance(v, np.floating) else int(v) if isinstance(v, np.integer) else v for k, v in r['details'].items()}
                }
                for r in all_results
            ],
            'generated_at': datetime.now().isoformat()
        }, f, indent=2)
    
    # Print final summary
    print(f"\n{'='*70}")
    print("BEST CONFIGURATION (from multi-seed)")
    print(f"{'='*70}")
    print(f"Seed: {best['seed']}")
    print(f"Fitness: {best['fitness']:.4f}")
    print(f"Return: {best['details'].get('total_return', 0)*100:.2f}%")
    print(f"Sharpe: {best['details'].get('sharpe', 0):.2f}")
    print(f"Max DD: {best['details'].get('max_dd', 0)*100:.1f}%")
    print(f"Win Rate: {best['details'].get('win_rate', 0)*100:.1f}%")
    print(f"Trades: {best['details'].get('trades', 0)}")
    print(f"\n{'='*70}")
    print("Weights:")
    for key, value in sorted(best['individual'].items()):
        if 'weight' in key:
            print(f"  {key}: {value:.2f}")
    print(f"{'='*70}")
    print(f"Saved to: {output_file}")
    
    return best


if __name__ == "__main__":
    run_multi_seed_optimizer()
