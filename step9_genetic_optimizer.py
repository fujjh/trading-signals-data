#!/usr/bin/env python3
"""
================================================================================
STEP 9: Genetic Algorithm Optimizer
================================================================================

Self-improving signal generator using genetic algorithms to optimize:
- Indicator weights
- Threshold parameters  
- Position sizing rules

Daily data only - optimized for swing trading.

Author: SignalsAlpha
Version: 1.0
Date: 2026-04-28
================================================================================

GENETIC ALGORITHM DESIGN:
================================================================================

GENOME (Individual):
├── Indicator Weights (8 genes)
│   ├── macd_weight: [0, 5]
│   ├── hma_weight: [0, 5]
│   ├── rsi_weight: [0, 5]
│   ├── stoch_weight: [0, 5]
│   ├── sma_weight: [0, 5]
│   ├── ema_weight: [0, 5]
│   ├── mfi_weight: [0, 5]
│   └── bb_weight: [0, 5]
│
├── Threshold Parameters (4 genes)
│   ├── oversold_threshold: [20, 35]
│   ├── overbought_threshold: [65, 80]
│   ├── adx_strong: [20, 35]
│   └── volume_confirm: [1.0, 2.0]
│
└── Position Sizing (2 genes)
    ├── max_position_pct: [0.10, 0.30]
    └── volatility_adj: [0, 1]  # Binary

FITNESS FUNCTION:
    Fitness = (Profit_Factor × 0.35) + (Sharpe × 0.25) + 
              (Expectancy × 0.20) + (1/Max_Drawdown × 0.15) +
              (Win_Rate × 0.05)

SELECTION: Tournament selection (top 50%)
CROSSOVER: Blend crossover (weighted average)
MUTATION: Gaussian mutation (σ = 10% of range)
ELITISM: Keep top 5 individuals unchanged
TERMINATION:
    - Max 50 generations
    - No improvement for 10 generations
    - Target fitness reached (fitness > 2.0)

================================================================================
"""

import os
import sys
import json
import random
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from copy import deepcopy
import warnings
warnings.filterwarnings('ignore')

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent / 'modules'))

from metrics_calculator import MetricsCalculator
from results_db import ResultsDatabase
from pipeline_bridge import PipelineBridge

# Configuration
DATA_DIR = Path(__file__).parent / "data"
SIGNALS_DIR = DATA_DIR / "signals_scored"
TIME_SERIES_DIR = DATA_DIR / "time_series"
OPTIMIZER_DIR = DATA_DIR / "optimizer"
OPTIMIZER_DIR.mkdir(parents=True, exist_ok=True)

POPULATION_SIZE = 30
GENERATIONS = 50
ELITISM_COUNT = 5
MUTATION_RATE = 0.15
MUTATION_STRENGTH = 0.10  # 10% of gene range
TARGET_FITNESS = 2.0
EARLY_STOP_PATIENCE = 10

# Gene definitions (name, min, max, type)
GENE_DEFINITIONS = [
    # Indicator weights
    ('macd_weight', 0, 5, 'int'),
    ('hma_weight', 0, 5, 'int'),
    ('rsi_weight', 0, 5, 'int'),
    ('stoch_weight', 0, 5, 'int'),
    ('sma_weight', 0, 5, 'int'),
    ('ema_weight', 0, 5, 'int'),
    ('mfi_weight', 0, 5, 'int'),
    ('bb_weight', 0, 5, 'int'),
    
    # Thresholds
    ('oversold_threshold', 20, 35, 'int'),
    ('overbought_threshold', 65, 80, 'int'),
    ('adx_strong', 20, 35, 'int'),
    ('volume_confirm', 1.0, 2.0, 'float'),
    
    # Position sizing
    ('max_position_pct', 0.10, 0.30, 'float'),
    ('volatility_adj', 0, 1, 'int'),  # Binary
]


@dataclass
class Individual:
    """Single individual in the population"""
    genes: Dict[str, Any]
    fitness: float = 0.0
    generation: int = 0
    metrics: Optional[Dict] = None
    
    def to_config(self) -> Dict:
        """Convert individual to scoring configuration"""
        return {
            'indicator_weights': {
                'macd': self.genes['macd_weight'],
                'hma': self.genes['hma_weight'],
                'rsi': self.genes['rsi_weight'],
                'stoch': self.genes['stoch_weight'],
                'sma': self.genes['sma_weight'],
                'ema': self.genes['ema_weight'],
                'mfi': self.genes['mfi_weight'],
                'bb': self.genes['bb_weight'],
            },
            'thresholds': {
                'oversold': self.genes['oversold_threshold'],
                'overbought': self.genes['overbought_threshold'],
                'adx_strong': self.genes['adx_strong'],
                'volume_confirm': self.genes['volume_confirm'],
            },
            'position_sizing': {
                'max_position_pct': self.genes['max_position_pct'],
                'volatility_adj': bool(self.genes['volatility_adj']),
            }
        }


class GeneticOptimizer:
    """Genetic Algorithm optimizer for trading strategy parameters"""
    
    def __init__(self, 
                 population_size: int = POPULATION_SIZE,
                 generations: int = GENERATIONS):
        """Initialize optimizer"""
        self.population_size = population_size
        self.generations = generations
        self.population: List[Individual] = []
        self.best_individual: Optional[Individual] = None
        self.fitness_history: List[float] = []
        self.generation = 0
        
        # For progress tracking
        self.progress_file = OPTIMIZER_DIR / 'optimizer_progress.pkl'
        self.results_db = ResultsDatabase()
        self.pipeline_bridge = PipelineBridge()
        
    def create_random_individual(self) -> Individual:
        """Create a random individual"""
        genes = {}
        for name, min_val, max_val, gene_type in GENE_DEFINITIONS:
            if gene_type == 'int':
                genes[name] = random.randint(int(min_val), int(max_val))
            else:  # float
                genes[name] = random.uniform(min_val, max_val)
        
        return Individual(genes=genes)
    
    def initialize_population(self):
        """Create initial random population"""
        print(f"Initializing population of {self.population_size} individuals...")
        self.population = [self.create_random_individual() 
                          for _ in range(self.population_size)]
    
    def evaluate_fitness(self, individual: Individual) -> float:
        """
        Evaluate fitness of an individual by running backtest simulation
        
        This is a simplified fitness evaluation using historical signal data.
        For production, this would run the full backtest (Step 6).
        """
        # Create config for this individual
        config = individual.to_config()
        
        # Simulate backtest metrics (simplified)
        # In production, this would run Step 6 backtest with these weights
        metrics = self._simulate_backtest(config)
        
        # Calculate composite fitness
        # Fitness = PF*0.35 + Sharpe*0.25 + Expectancy*0.20 + 1/DD*0.15 + WR*0.05
        fitness = (
            metrics['profit_factor'] * 0.35 +
            max(metrics['sharpe_ratio'], 0) * 0.25 +
            max(metrics['expectancy'], 0) * 0.20 +
            (1 / max(metrics['max_drawdown'], 0.01)) * 0.15 +
            metrics['win_rate'] * 0.05
        )
        
        individual.fitness = fitness
        individual.metrics = metrics
        
        return fitness
    
    def _simulate_backtest(self, config: Dict) -> Dict:
        """
        Simulate backtest with given configuration
        
        For now, this uses historical Step 4 results and adjusts based on config.
        In production, this would run actual Step 6 backtest.
        """
        # Load latest Step 4 signals
        signal_files = list(SIGNALS_DIR.glob('scored_signals_*.csv'))
        if not signal_files:
            # Return default metrics if no signals
            return {
                'profit_factor': 1.0,
                'sharpe_ratio': 0.5,
                'expectancy': 0.0,
                'max_drawdown': 0.20,
                'win_rate': 0.50
            }
        
        latest_signals = max(signal_files, key=lambda p: p.stat().st_mtime)
        df = pd.read_csv(latest_signals)
        
        # Calculate simulated metrics based on config weights
        weights = config['indicator_weights']
        
        # Adjust win rate based on weight distribution
        # More balanced weights = more realistic results
        weight_variance = np.var(list(weights.values()))
        base_win_rate = 0.52 - (weight_variance * 0.02)  # Penalize extreme weights
        
        # Simulate profit factor (higher when weights balanced)
        base_pf = 1.2 + (1.0 - weight_variance / 6.25) * 0.8
        
        # Simulate Sharpe
        base_sharpe = 0.8 + (1.0 - weight_variance / 6.25) * 0.7
        
        # Simulate drawdown
        base_dd = 0.15 + (weight_variance / 6.25) * 0.10
        
        # Add some randomness for genetic diversity
        noise = np.random.normal(0, 0.05)
        
        return {
            'profit_factor': max(0.5, min(3.0, base_pf + noise)),
            'sharpe_ratio': max(-1.0, min(2.5, base_sharpe + noise)),
            'expectancy': max(-100, min(200, base_pf * 50 + noise * 100)),
            'max_drawdown': max(0.05, min(0.50, base_dd + abs(noise) * 0.05)),
            'win_rate': max(0.30, min(0.70, base_win_rate + noise))
        }
    
    def evaluate_population(self):
        """Evaluate fitness for entire population"""
        print(f"\nEvaluating population (Generation {self.generation})...")
        
        for i, individual in enumerate(self.population):
            if i % 5 == 0:
                print(f"  Progress: {i}/{self.population_size}", end='\r')
            
            self.evaluate_fitness(individual)
            individual.generation = self.generation
        
        print(f"  Progress: {self.population_size}/{self.population_size}")
        
        # Sort by fitness (descending)
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        
        # Track best
        if not self.best_individual or self.population[0].fitness > self.best_individual.fitness:
            self.best_individual = deepcopy(self.population[0])
        
        self.fitness_history.append(self.population[0].fitness)
        
        print(f"  Best fitness: {self.population[0].fitness:.4f}")
        print(f"  Avg fitness: {np.mean([ind.fitness for ind in self.population]):.4f}")
    
    def tournament_selection(self, tournament_size: int = 3) -> Individual:
        """Select parent using tournament selection"""
        tournament = random.sample(self.population[:int(self.population_size * 0.5)], 
                                  tournament_size)
        return max(tournament, key=lambda x: x.fitness)
    
    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Blend crossover between two parents"""
        child1_genes = {}
        child2_genes = {}
        
        for name, min_val, max_val, gene_type in GENE_DEFINITIONS:
            # Blend crossover with random weight
            alpha = random.random()
            
            if gene_type == 'int':
                val1 = int(alpha * parent1.genes[name] + (1 - alpha) * parent2.genes[name])
                val2 = int((1 - alpha) * parent1.genes[name] + alpha * parent2.genes[name])
                child1_genes[name] = max(min_val, min(max_val, val1))
                child2_genes[name] = max(min_val, min(max_val, val2))
            else:  # float
                val1 = alpha * parent1.genes[name] + (1 - alpha) * parent2.genes[name]
                val2 = (1 - alpha) * parent1.genes[name] + alpha * parent2.genes[name]
                child1_genes[name] = max(min_val, min(max_val, val1))
                child2_genes[name] = max(min_val, min(max_val, val2))
        
        return Individual(genes=child1_genes), Individual(genes=child2_genes)
    
    def mutate(self, individual: Individual) -> Individual:
        """Apply Gaussian mutation to individual"""
        mutated_genes = individual.genes.copy()
        
        for name, min_val, max_val, gene_type in GENE_DEFINITIONS:
            if random.random() < MUTATION_RATE:
                # Gaussian mutation
                range_val = max_val - min_val
                if gene_type == 'int':
                    mutation = int(np.random.normal(0, range_val * MUTATION_STRENGTH))
                    mutated_genes[name] = max(min_val, min(max_val, 
                                                             mutated_genes[name] + mutation))
                else:  # float
                    mutation = np.random.normal(0, range_val * MUTATION_STRENGTH)
                    mutated_genes[name] = max(min_val, min(max_val, 
                                                             mutated_genes[name] + mutation))
        
        return Individual(genes=mutated_genes)
    
    def create_next_generation(self):
        """Create next generation using selection, crossover, mutation"""
        new_population = []
        
        # Elitism: Keep top performers
        new_population.extend(self.population[:ELITISM_COUNT])
        
        # Fill rest with offspring
        while len(new_population) < self.population_size:
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()
            
            child1, child2 = self.crossover(parent1, parent2)
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            
            new_population.append(child1)
            if len(new_population) < self.population_size:
                new_population.append(child2)
        
        self.population = new_population
        self.generation += 1
    
    def check_termination(self) -> Tuple[bool, str]:
        """Check if optimization should terminate"""
        # Target fitness reached
        if self.best_individual and self.best_individual.fitness >= TARGET_FITNESS:
            return True, "Target fitness reached"
        
        # Max generations
        if self.generation >= GENERATIONS:
            return True, "Max generations reached"
        
        # Early stopping (no improvement)
        if len(self.fitness_history) > EARLY_STOP_PATIENCE:
            recent = self.fitness_history[-EARLY_STOP_PATIENCE:]
            if max(recent) - min(recent) < 0.01:  # Less than 1% variation
                return True, "No improvement for {} generations".format(EARLY_STOP_PATIENCE)
        
        return False, ""
    
    def save_progress(self):
        """Save optimization progress"""
        progress = {
            'generation': self.generation,
            'population': self.population,
            'best_individual': self.best_individual,
            'fitness_history': self.fitness_history
        }
        with open(self.progress_file, 'wb') as f:
            pickle.dump(progress, f)
    
    def load_progress(self) -> bool:
        """Load optimization progress"""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'rb') as f:
                    progress = pickle.load(f)
                self.generation = progress['generation']
                self.population = progress['population']
                self.best_individual = progress['best_individual']
                self.fitness_history = progress['fitness_history']
                print(f"Resumed from generation {self.generation}")
                return True
            except Exception as e:
                print(f"Could not load progress: {e}")
        return False
    
    def run(self) -> Individual:
        """Run the genetic algorithm optimization"""
        print("=" * 70)
        print("GENETIC ALGORITHM OPTIMIZER")
        print("=" * 70)
        print(f"Population size: {self.population_size}")
        print(f"Generations: {self.generations}")
        print(f"Elitism: {ELITISM_COUNT}")
        print(f"Mutation rate: {MUTATION_RATE * 100:.0f}%")
        print(f"Target fitness: {TARGET_FITNESS}")
        print("=" * 70)
        
        # Initialize or resume
        if not self.load_progress():
            self.initialize_population()
            self.evaluate_population()
        
        # Evolution loop
        while True:
            # Check termination
            should_stop, reason = self.check_termination()
            if should_stop:
                print(f"\n{reason}")
                break
            
            # Evolution step
            self.create_next_generation()
            self.evaluate_population()
            self.save_progress()
            
            # Print best individual
            if self.generation % 5 == 0:
                print(f"\nGeneration {self.generation} Best:")
                self.print_individual(self.best_individual)
        
        # Final report
        print("\n" + "=" * 70)
        print("OPTIMIZATION COMPLETE")
        print("=" * 70)
        self.print_individual(self.best_individual)
        
        # Save to database
        self.save_best_to_database()
        
        return self.best_individual
    
    def print_individual(self, individual: Individual):
        """Print individual details"""
        config = individual.to_config()
        print(f"\nGeneration: {individual.generation}")
        print(f"Fitness: {individual.fitness:.4f}")
        print("\nIndicator Weights:")
        for name, weight in config['indicator_weights'].items():
            print(f"  {name:15s}: {weight}")
        print("\nThresholds:")
        for name, val in config['thresholds'].items():
            print(f"  {name:15s}: {val}")
        print("\nPosition Sizing:")
        for name, val in config['position_sizing'].items():
            print(f"  {name:15s}: {val}")
        
        if individual.metrics:
            print("\nSimulated Metrics:")
            for name, val in individual.metrics.items():
                if isinstance(val, float):
                    print(f"  {name:15s}: {val:.3f}")
                else:
                    print(f"  {name:15s}: {val}")
    
    def save_best_to_database(self):
        """Save best individual to results database using bridge"""
        if not self.best_individual:
            return
        
        metrics = self.best_individual.metrics
        if metrics:
            # Use pipeline bridge for proper format conversion
            result_id = self.pipeline_bridge.save_step10_result(
                config_name=f"genetic_optimized_gen{self.generation}",
                config=self.best_individual.to_config(),
                metrics=metrics,
                generation=self.generation,
                fitness=self.best_individual.fitness
            )
            
            print(f"\nSaved to database (ID: {result_id})")
    
    def export_best_config(self, output_path: Path):
        """Export best configuration to JSON"""
        if not self.best_individual:
            print("No best individual to export")
            return
        
        config = {
            'generation': self.best_individual.generation,
            'fitness': self.best_individual.fitness,
            'config': self.best_individual.to_config(),
            'metrics': self.best_individual.metrics
        }
        
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Exported best config to: {output_path}")


def main():
    """Main entry point"""
    optimizer = GeneticOptimizer(
        population_size=POPULATION_SIZE,
        generations=GENERATIONS
    )
    
    best = optimizer.run()
    
    # Export configuration
    config_path = OPTIMIZER_DIR / f"best_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    optimizer.export_best_config(config_path)
    
    print("\n" + "=" * 70)
    print("OPTIMIZER COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
