#!/usr/bin/env python3
"""
================================================================================
STEP 9: SPY Genetic Algorithm Optimizer v3.1 (With 52w High/Low & SMA 200)
================================================================================

Genetic optimizer using 52-week high/low and 200-day SMA indicators.

Author: SignalsAlpha
Version: 3.1 (SPY Edition)
Date: 2026-04-29
================================================================================
"""

import json
import random
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple

# Configuration
TICKER = "SPY"
TECHNICAL_DIR = Path("data/technical_analysis") / TICKER
OUTPUT_DIR = Path("data/optimizer")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

POPULATION_SIZE = 30
GENERATIONS = 50
ELITISM_COUNT = 3
MUTATION_RATE = 0.15


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


def run_optimizer():
    """Main optimization."""
    print("=" * 70)
    print("STEP 9: SPY Genetic Optimizer v3.1 (With 52w High/Low & SMA 200)")
    print("=" * 70)
    print(f"Population: {POPULATION_SIZE}")
    print(f"Generations: {GENERATIONS}")
    print("=" * 70)
    
    df = load_spy_data()
    if df is None:
        return
    
    print(f"Loaded {len(df)} rows")
    print(f"New indicators available: 52w high/low, SMA 200")
    
    # Benchmark
    first = df['close'].iloc[0]
    last = df['close'].iloc[-1]
    bh_return = (last - first) / first
    print(f"Buy-and-Hold: {bh_return*100:.2f}% ({first:.2f} → {last:.2f})")
    print("=" * 70)
    
    # Init population
    population = [create_individual() for _ in range(POPULATION_SIZE)]
    
    best_fitness = 0
    best_individual = None
    best_details = None
    
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
            print(f"\nGen {gen}: New best fitness = {best_fitness:.4f}")
            if 'total_return' in best_details:
                print(f"  Return: {best_details['total_return']*100:.2f}%, Trades: {best_details['trades']}")
        
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
    
    # Save best
    if best_individual:
        output_file = OUTPUT_DIR / f"spy_best_config_v3_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump({
                'ticker': TICKER,
                'fitness': float(best_fitness),
                'config': {k: float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else v for k, v in best_individual.items()},
                'details': {k: float(v) if isinstance(v, np.floating) else int(v) if isinstance(v, np.integer) else v for k, v in best_details.items()},
                'generated_at': datetime.now().isoformat()
            }, f, indent=2)
        
        print(f"\n{'='*70}")
        print("OPTIMIZATION COMPLETE")
        print(f"{'='*70}")
        print(f"Best fitness: {best_fitness:.4f}")
        if best_details:
            print(f"Return: {best_details.get('total_return', 0)*100:.2f}%")
            print(f"Sharpe: {best_details.get('sharpe', 0):.2f}")
            print(f"Max DD: {best_details.get('max_dd', 0)*100:.1f}%")
            print(f"Win Rate: {best_details.get('win_rate', 0)*100:.1f}%")
            print(f"Trades: {best_details.get('trades', 0)}")
        print(f"\n{'='*70}")
        print("Best Configuration:")
        for key, value in sorted(best_individual.items()):
            if 'weight' in key:
                print(f"  {key}: {value:.2f}")
        print(f"{'='*70}")


if __name__ == "__main__":
    run_optimizer()
