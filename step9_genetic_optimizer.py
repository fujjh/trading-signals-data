#!/usr/bin/env python3
"""
================================================================================
STEP 9: SPY Genetic Algorithm Optimizer
================================================================================

Self-improving signal generator using genetic algorithms to optimize
indicator weights for SPY-only analysis.

Author: SignalsAlpha
Version: 1.0 (SPY Edition)
Date: 2026-04-29
================================================================================

GENOME STRUCTURE:
- Indicator weights (8 genes): macd, hma, rsi, stoch, sma, ema, mfi, bb
- Threshold parameters (4 genes): oversold, overbought, adx_strong, volume_confirm
- Position sizing (2 genes): max_position_pct, volatility_adj

FITNESS FUNCTION:
Fitness = (Profit_Factor × 0.35) + (Sharpe × 0.25) + (Expectancy × 0.20)
================================================================================
"""

import json
import random
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Configuration
TICKER = "SPY"
SIGNALS_DIR = Path("data/signals_scored")
TECHNICAL_DIR = Path("data/technical_analysis") / TICKER
OUTPUT_DIR = Path("data/optimizer")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

POPULATION_SIZE = 20
GENERATIONS = 30
ELITISM_COUNT = 3
MUTATION_RATE = 0.15
TARGET_FITNESS = 2.0


def load_spy_data():
    """Load SPY technical data and signals."""
    # Load technical data
    tech_file = TECHNICAL_DIR / f"{TICKER}_1d_technical.csv"
    if not tech_file.exists():
        print(f"ERROR: Technical file not found: {tech_file}")
        return None
    
    df = pd.read_csv(tech_file)
    df['date'] = pd.to_datetime(df['date'])
    return df


def create_individual():
    """Create a random individual (genome)."""
    individual = {
        # Indicator weights (0-5)
        'macd_weight': random.randint(0, 5),
        'hma_weight': random.randint(0, 5),
        'rsi_weight': random.randint(0, 5),
        'stoch_weight': random.randint(0, 5),
        'sma_weight': random.randint(0, 5),
        'ema_weight': random.randint(0, 5),
        'mfi_weight': random.randint(0, 5),
        'bb_weight': random.randint(0, 5),
        
        # Thresholds
        'oversold': random.randint(20, 35),
        'overbought': random.randint(65, 80),
        'adx_strong': random.randint(20, 35),
        'volume_confirm': round(random.uniform(1.0, 2.0), 2),
        
        # Position sizing
        'max_position_pct': round(random.uniform(0.10, 0.30), 2),
        'volatility_adj': random.choice([True, False])
    }
    return individual


def calculate_fitness(df: pd.DataFrame, individual: Dict) -> float:
    """Calculate fitness score for an individual."""
    # Simplified backtest using the genome's parameters
    
    # Calculate technical score based on weights
    score = 50  # Base
    
    # Trend (SMA/EMA weighted)
    if 'sma_20' in df.columns and 'sma_50' in df.columns:
        trend_score = np.where(
            (df['close'] > df['sma_20']) & (df['sma_20'] > df['sma_50']),
            individual['sma_weight'] + individual['ema_weight'],
            np.where(
                (df['close'] < df['sma_20']) & (df['sma_20'] < df['sma_50']),
                -(individual['sma_weight'] + individual['ema_weight']),
                0
            )
        )
        score += trend_score
    
    # RSI
    if 'rsi' in df.columns:
        rsi_score = np.where(
            df['rsi'] < individual['oversold'],
            individual['rsi_weight'],
            np.where(
                df['rsi'] > individual['overbought'],
                -individual['rsi_weight'],
                0
            )
        )
        score += rsi_score
    
    # MACD
    if 'macd' in df.columns and 'macd_signal' in df.columns:
        macd_score = np.where(
            df['macd'] > df['macd_signal'],
            individual['macd_weight'],
            -individual['macd_weight']
        )
        score += macd_score
    
    # HMA
    if 'hma_13' in df.columns:
        hma_score = np.where(
            df['close'] > df['hma_13'],
            individual['hma_weight'],
            -individual['hma_weight']
        )
        score += hma_score
    
    # Calculate returns based on score
    df_copy = df.copy()
    df_copy['score'] = score
    df_copy['position'] = np.where(df_copy['score'] > 60, 1, 
                                   np.where(df_copy['score'] < 40, -1, 0))
    
    # Simple strategy: go long when score > 60
    df_copy['returns'] = df_copy['close'].pct_change()
    df_copy['strategy_returns'] = df_copy['position'].shift(1) * df_copy['returns']
    
    # Calculate metrics
    total_return = df_copy['strategy_returns'].sum()
    volatility = df_copy['strategy_returns'].std() * np.sqrt(252)
    sharpe = total_return / volatility if volatility > 0 else 0
    
    # Calculate win rate
    trades = df_copy[df_copy['position'] != 0]['strategy_returns']
    if len(trades) > 0:
        win_rate = (trades > 0).sum() / len(trades)
    else:
        win_rate = 0.5
    
    # Fitness calculation
    fitness = (sharpe * 0.4) + (win_rate * 0.3) + (min(abs(total_return) * 10, 1.0) * 0.3)
    
    return max(0, fitness)


def mutate(individual: Dict) -> Dict:
    """Apply mutation to an individual."""
    mutated = individual.copy()
    
    # Mutate weights
    for key in ['macd_weight', 'hma_weight', 'rsi_weight', 'stoch_weight', 
                'sma_weight', 'ema_weight', 'mfi_weight', 'bb_weight']:
        if random.random() < MUTATION_RATE:
            mutated[key] = max(0, min(5, individual[key] + random.randint(-1, 1)))
    
    # Mutate thresholds
    if random.random() < MUTATION_RATE:
        mutated['oversold'] = max(20, min(35, individual['oversold'] + random.randint(-2, 2)))
    if random.random() < MUTATION_RATE:
        mutated['overbought'] = max(65, min(80, individual['overbought'] + random.randint(-2, 2)))
    
    return mutated


def crossover(parent1: Dict, parent2: Dict) -> Tuple[Dict, Dict]:
    """Create two children from two parents."""
    child1 = {}
    child2 = {}
    
    for key in parent1.keys():
        if isinstance(parent1[key], bool):
            # Boolean: randomly choose
            child1[key] = random.choice([parent1[key], parent2[key]])
            child2[key] = random.choice([parent1[key], parent2[key]])
        elif isinstance(parent1[key], int):
            # Integer: average and round
            avg = (parent1[key] + parent2[key]) // 2
            child1[key] = max(0, min(5, avg + random.randint(-1, 1)))
            child2[key] = max(0, min(5, avg + random.randint(-1, 1)))
        else:
            # Float: average
            avg = (parent1[key] + parent2[key]) / 2
            child1[key] = round(avg, 2)
            child2[key] = round(avg, 2)
    
    return child1, child2


def run_genetic_optimizer():
    """Main genetic algorithm."""
    print("=" * 70)
    print("STEP 9: SPY Genetic Algorithm Optimizer")
    print("=" * 70)
    print(f"Population: {POPULATION_SIZE}")
    print(f"Generations: {GENERATIONS}")
    print(f"Elitism: {ELITISM_COUNT}")
    print(f"Target fitness: {TARGET_FITNESS}")
    print("=" * 70)
    
    # Load data
    df = load_spy_data()
    if df is None:
        print("ERROR: Could not load SPY data")
        return
    
    print(f"Loaded {len(df)} rows of SPY data")
    
    # Initialize population
    population = [create_individual() for _ in range(POPULATION_SIZE)]
    
    best_fitness = 0
    best_individual = None
    no_improvement_count = 0
    
    for generation in range(GENERATIONS):
        # Evaluate fitness
        fitness_scores = []
        for i, individual in enumerate(population):
            fitness = calculate_fitness(df, individual)
            fitness_scores.append((fitness, individual))
        
        # Sort by fitness
        fitness_scores.sort(key=lambda x: x[0], reverse=True)
        
        # Update best
        if fitness_scores[0][0] > best_fitness:
            best_fitness = fitness_scores[0][0]
            best_individual = fitness_scores[0][1].copy()
            no_improvement_count = 0
            print(f"Generation {generation}: New best fitness = {best_fitness:.4f}")
        else:
            no_improvement_count += 1
        
        # Check early stopping
        if best_fitness >= TARGET_FITNESS:
            print(f"Target fitness reached!")
            break
        
        if no_improvement_count >= 10:
            print(f"No improvement for 10 generations, stopping early")
            break
        
        # Create next generation
        new_population = []
        
        # Elitism
        for i in range(ELITISM_COUNT):
            new_population.append(fitness_scores[i][1])
        
        # Crossover and mutation
        while len(new_population) < POPULATION_SIZE:
            parent1 = random.choice(fitness_scores[:POPULATION_SIZE//2])[1]
            parent2 = random.choice(fitness_scores[:POPULATION_SIZE//2])[1]
            
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1)
            child2 = mutate(child2)
            
            new_population.append(child1)
            if len(new_population) < POPULATION_SIZE:
                new_population.append(child2)
        
        population = new_population
    
    # Save best configuration
    if best_individual:
        output_file = OUTPUT_DIR / f"spy_best_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump({
                'ticker': TICKER,
                'fitness': best_fitness,
                'config': best_individual,
                'generated_at': datetime.now().isoformat()
            }, f, indent=2)
        
        print(f"\n{'='*70}")
        print("OPTIMIZATION COMPLETE")
        print(f"{'='*70}")
        print(f"Best fitness: {best_fitness:.4f}")
        print(f"\nBest configuration:")
        for key, value in best_individual.items():
            print(f"  {key}: {value}")
        print(f"\nSaved to: {output_file}")
        print(f"{'='*70}")


if __name__ == "__main__":
    run_genetic_optimizer()
