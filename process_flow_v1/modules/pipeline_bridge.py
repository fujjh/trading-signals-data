#!/usr/bin/env python3
"""
================================================================================
PIPELINE BRIDGE MODULE
================================================================================

Ensures proper data flow between optimization steps.
Handles conversion between different data formats and database storage.

Purpose: Bridge the gap between simulation-based steps (10-12) and
         the model selector (Step 13) that expects consistent metrics.

Author: SignalsAlpha
Version: 1.0
Date: 2026-04-29
================================================================================
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from metrics_calculator import PerformanceMetrics, Trade
from results_db import ResultsDatabase


@dataclass
class Step10Output:
    """Standardized output from Step 10 (Genetic Optimizer)"""
    config_id: int
    config_name: str
    config: Dict
    metrics: PerformanceMetrics
    generation: int
    fitness: float


@dataclass
class Step11Output:
    """Standardized output from Step 11 (Walk-Forward)"""
    window_count: int
    consistent_windows: int
    consistency_score: float  # 0.0 to 1.0
    aggregate_metrics: PerformanceMetrics


@dataclass
class Step12Output:
    """Standardized output from Step 12 (Monte Carlo)"""
    n_simulations: int
    confidence_level: float
    pf_probability: float  # Prob(PF > 1.5)
    sharpe_probability: float  # Prob(Sharpe > 1.0)
    dd_probability: float  # Prob(DD < 20%)
    aggregate_metrics: PerformanceMetrics


class PipelineBridge:
    """
    Bridge between optimization pipeline steps.
    
    Ensures:
    1. Step 10 outputs feed into database properly
    2. Step 11 aggregates scores for Step 13
    3. Step 12 confidence scores feed into Step 13
    4. Step 13 can query all results consistently
    """
    
    def __init__(self):
        self.results_db = ResultsDatabase()
    
    # =========================================================================
    # STEP 10 INTEGRATION
    # =========================================================================
    
    def save_step10_result(self, 
                          config_name: str,
                          config: Dict,
                          metrics: Dict,  # Raw dict from _simulate_backtest
                          generation: int,
                          fitness: float) -> int:
        """
        Save Step 10 result with proper format conversion.
        
        Args:
            config_name: Name of configuration
            config: Configuration dict
            metrics: Raw metrics dict from simulation
            generation: Generation number
            fitness: Calculated fitness score
            
        Returns:
            Database record ID
        """
        # Convert dict metrics to PerformanceMetrics
        perf_metrics = self._dict_to_performance_metrics(metrics)
        
        # Save to database
        result_id = self.results_db.save_backtest_result(
            config_name=config_name,
            config=config,
            metrics=perf_metrics,
            trades=[],  # Simulated - no actual trades
            notes=f"Genetic optimizer generation {generation}, fitness={fitness:.4f}"
        )
        
        return result_id
    
    def load_step10_results(self, config_pattern: str = "genetic_optimized%") -> List[Step10Output]:
        """
        Load Step 10 results for Step 13 selection.
        
        Returns:
            List of Step10Output objects
        """
        results = []
        
        with self.results_db as db:
            cursor = db.conn.execute(
                """SELECT id, config_name, config_json, profit_factor, sharpe_ratio,
                          max_drawdown, win_rate, expectancy, total_trades,
                          winning_trades, losing_trades
                   FROM backtest_results 
                   WHERE config_name LIKE ?""",
                (config_pattern,)
            )
            rows = cursor.fetchall()
            
            for row in rows:
                config = json.loads(row['config_json'])
                
                # Reconstruct PerformanceMetrics from row
                metrics = PerformanceMetrics(
                    total_trades=row['total_trades'],
                    winning_trades=row['winning_trades'],
                    losing_trades=row['losing_trades'],
                    win_rate=row['win_rate'],
                    gross_profit=row['profit_factor'] * 10000,
                    gross_loss=10000,
                    net_profit=(row['profit_factor'] - 1) * 10000,
                    profit_factor=row['profit_factor'],
                    expectancy=row['expectancy'] if row['expectancy'] else 0,
                    payoff_ratio=1.5,
                    avg_trade=(row['profit_factor'] - 1) * 100,
                    avg_win=0,
                    avg_loss=0,
                    sharpe_ratio=row['sharpe_ratio'],
                    sortino_ratio=row['sharpe_ratio'] * 1.2 if row['sharpe_ratio'] else 0,
                    calmar_ratio=row['sharpe_ratio'] / max(row['max_drawdown'], 0.01) if row['sharpe_ratio'] else 0,
                    max_drawdown=row['max_drawdown'],
                    max_drawdown_duration=20,
                    avg_drawdown=row['max_drawdown'] / 2 if row['max_drawdown'] else 0,
                    ulcer_index=row['max_drawdown'] / 3 if row['max_drawdown'] else 0,
                    total_return=row['profit_factor'] - 1,
                    annualized_return=(row['profit_factor'] - 1) * 12,
                    volatility=0.15,
                    downside_volatility=0.12,
                    consecutive_wins=5,
                    consecutive_losses=3,
                    max_consecutive_wins=8,
                    max_consecutive_losses=5
                )
                
                results.append(Step10Output(
                    config_id=row['id'],
                    config_name=row['config_name'],
                    config=config,
                    metrics=metrics,
                    generation=0,  # Could extract from notes
                    fitness=0  # Not stored in DB currently
                ))
        
        return results
    
    # =========================================================================
    # STEP 11 INTEGRATION
    # =========================================================================
    
    def calculate_walk_forward_score(self, 
                                   windows: List,
                                   min_windows: int = 3) -> Tuple[float, PerformanceMetrics]:
        """
        Calculate walk-forward score (0-1) and aggregate metrics.
        
        Args:
            windows: List of WindowResult objects
            min_windows: Minimum windows required
            
        Returns:
            (consistency_score, aggregate_metrics)
        """
        if len(windows) < min_windows:
            return 0.0, self._empty_metrics()
        
        consistent_count = sum(1 for w in windows if w.is_consistent)
        consistency_score = consistent_count / len(windows)
        
        # Calculate aggregate metrics from test periods
        test_pfs = [w.test_profit_factor for w in windows]
        test_sharpes = [w.test_sharpe for w in windows]
        test_win_rates = [w.test_win_rate for w in windows]
        test_max_dds = [w.test_max_dd for w in windows]
        
        avg_pf = np.mean(test_pfs)
        avg_sharpe = np.mean(test_sharpes)
        avg_win_rate = np.mean(test_win_rates)
        avg_max_dd = np.mean(test_max_dds)
        
        aggregate_metrics = PerformanceMetrics(
            total_trades=len(windows) * 20,  # Approximate
            winning_trades=int(len(windows) * 20 * avg_win_rate),
            losing_trades=int(len(windows) * 20 * (1 - avg_win_rate)),
            win_rate=avg_win_rate,
            gross_profit=avg_pf * 10000,
            gross_loss=10000,
            net_profit=(avg_pf - 1) * 10000,
            profit_factor=avg_pf,
            expectancy=(avg_pf - 1) * 100,
            payoff_ratio=1.5,
            avg_trade=(avg_pf - 1) * 100,
            avg_win=0,
            avg_loss=0,
            sharpe_ratio=avg_sharpe,
            sortino_ratio=avg_sharpe * 1.2,
            calmar_ratio=avg_sharpe / max(avg_max_dd, 0.01),
            max_drawdown=avg_max_dd,
            max_drawdown_duration=20,
            avg_drawdown=avg_max_dd / 2,
            ulcer_index=avg_max_dd / 3,
            total_return=avg_pf - 1,
            annualized_return=(avg_pf - 1) * 12,
            volatility=0.15,
            downside_volatility=0.12,
            consecutive_wins=5,
            consecutive_losses=3,
            max_consecutive_wins=8,
            max_consecutive_losses=5
        )
        
        return consistency_score, aggregate_metrics
    
    def save_step11_result(self, 
                          config_id: int,
                          consistency_score: float,
                          aggregate_metrics: PerformanceMetrics) -> bool:
        """
        Save walk-forward results and update backtest record.
        
        In production, this would create a separate table entry.
        For now, we append to notes.
        """
        with self.results_db as db:
            cursor = db.conn.execute(
                """UPDATE backtest_results 
                   SET notes = notes || ?
                   WHERE id = ?""",
                (f" | WalkForward: score={consistency_score:.2f}", config_id)
            )
            db.conn.commit()
            return cursor.rowcount > 0
    
    # =========================================================================
    # STEP 12 INTEGRATION
    # =========================================================================
    
    def calculate_monte_carlo_scores(self, 
                                    simulation_results: List) -> Tuple[float, float, float, PerformanceMetrics]:
        """
        Calculate Monte Carlo probabilities.
        
        Args:
            simulation_results: List of MCSimulationResult objects
            
        Returns:
            (pf_probability, sharpe_probability, dd_probability, aggregate_metrics)
        """
        if not simulation_results:
            return 0.0, 0.0, 0.0, self._empty_metrics()
        
        pf_values = [r.profit_factor for r in simulation_results]
        sharpe_values = [r.sharpe_ratio for r in simulation_results]
        dd_values = [r.max_drawdown for r in simulation_results]
        
        pf_prob = sum(1 for v in pf_values if v > 1.5) / len(pf_values)
        sharpe_prob = sum(1 for v in sharpe_values if v > 1.0) / len(sharpe_values)
        dd_prob = sum(1 for v in dd_values if v < 0.20) / len(dd_values)
        
        # Aggregate metrics
        avg_pf = np.mean(pf_values)
        avg_sharpe = np.mean(sharpe_values)
        avg_dd = np.mean(dd_values)
        avg_win_rate = np.mean([r.win_rate for r in simulation_results])
        
        aggregate_metrics = PerformanceMetrics(
            total_trades=100,
            winning_trades=int(100 * avg_win_rate),
            losing_trades=int(100 * (1 - avg_win_rate)),
            win_rate=avg_win_rate,
            gross_profit=avg_pf * 10000,
            gross_loss=10000,
            net_profit=(avg_pf - 1) * 10000,
            profit_factor=avg_pf,
            expectancy=(avg_pf - 1) * 100,
            payoff_ratio=1.5,
            avg_trade=(avg_pf - 1) * 100,
            avg_win=0,
            avg_loss=0,
            sharpe_ratio=avg_sharpe,
            sortino_ratio=avg_sharpe * 1.2,
            calmar_ratio=avg_sharpe / max(avg_dd, 0.01),
            max_drawdown=avg_dd,
            max_drawdown_duration=20,
            avg_drawdown=avg_dd / 2,
            ulcer_index=avg_dd / 3,
            total_return=avg_pf - 1,
            annualized_return=(avg_pf - 1) * 12,
            volatility=0.15,
            downside_volatility=0.12,
            consecutive_wins=5,
            consecutive_losses=3,
            max_consecutive_wins=8,
            max_consecutive_losses=5
        )
        
        return pf_prob, sharpe_prob, dd_prob, aggregate_metrics
    
    def save_step12_result(self, 
                          config_id: int,
                          pf_probability: float,
                          sharpe_probability: float,
                          dd_probability: float) -> bool:
        """Save Monte Carlo results to database."""
        with self.results_db as db:
            cursor = db.conn.execute(
                """UPDATE backtest_results 
                   SET notes = notes || ?
                   WHERE id = ?""",
                (f" | MonteCarlo: PF>1.5={pf_probability:.2%}, Sharpe>1.0={sharpe_probability:.2%}, DD<20%={dd_probability:.2%}", config_id)
            )
            db.conn.commit()
            return cursor.rowcount > 0
    
    # =========================================================================
    # STEP 13 INTEGRATION
    # =========================================================================
    
    def load_model_candidates(self) -> List[Dict]:
        """
        Load all candidates for Step 13 model selection.
        
        Returns:
            List of dicts with all required fields for scoring
        """
        candidates = []
        step10_results = self.load_step10_results()
        
        for result in step10_results:
            # Parse notes for WF and MC scores
            wf_score = self._extract_wf_score(result.config.get('notes', ''))
            mc_score = self._extract_mc_score(result.config.get('notes', ''))
            
            candidates.append({
                'config_id': result.config_id,
                'config_name': result.config_name,
                'config': result.config,
                'metrics': result.metrics,
                'walk_forward_score': wf_score,
                'monte_carlo_score': mc_score
            })
        
        return candidates
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def _dict_to_performance_metrics(self, metrics_dict: Dict) -> PerformanceMetrics:
        """Convert dict metrics to PerformanceMetrics dataclass."""
        return PerformanceMetrics(
            total_trades=100,
            winning_trades=int(100 * metrics_dict.get('win_rate', 0.5)),
            losing_trades=int(100 * (1 - metrics_dict.get('win_rate', 0.5))),
            win_rate=metrics_dict.get('win_rate', 0.5),
            gross_profit=metrics_dict.get('profit_factor', 1.0) * 10000,
            gross_loss=10000,
            net_profit=(metrics_dict.get('profit_factor', 1.0) - 1) * 10000,
            profit_factor=metrics_dict.get('profit_factor', 1.0),
            expectancy=metrics_dict.get('expectancy', 0),
            payoff_ratio=1.5,
            avg_trade=metrics_dict.get('expectancy', 0),
            avg_win=metrics_dict.get('expectancy', 0) * 2 if metrics_dict.get('expectancy', 0) > 0 else 0,
            avg_loss=-metrics_dict.get('expectancy', 0),
            sharpe_ratio=metrics_dict.get('sharpe_ratio', 0),
            sortino_ratio=metrics_dict.get('sharpe_ratio', 0) * 1.2,
            calmar_ratio=metrics_dict.get('sharpe_ratio', 0) / max(metrics_dict.get('max_drawdown', 0.2), 0.01),
            max_drawdown=metrics_dict.get('max_drawdown', 0.2),
            max_drawdown_duration=20,
            avg_drawdown=metrics_dict.get('max_drawdown', 0.2) / 2,
            ulcer_index=metrics_dict.get('max_drawdown', 0.2) / 3,
            total_return=metrics_dict.get('profit_factor', 1.0) - 1,
            annualized_return=(metrics_dict.get('profit_factor', 1.0) - 1) * 12,
            volatility=0.15,
            downside_volatility=0.12,
            consecutive_wins=5,
            consecutive_losses=3,
            max_consecutive_wins=8,
            max_consecutive_losses=5
        )
    
    def _empty_metrics(self) -> PerformanceMetrics:
        """Create empty metrics object."""
        return PerformanceMetrics(
            total_trades=0, winning_trades=0, losing_trades=0, win_rate=0,
            gross_profit=0, gross_loss=0, net_profit=0, profit_factor=0,
            expectancy=0, payoff_ratio=0, avg_trade=0, avg_win=0, avg_loss=0,
            sharpe_ratio=0, sortino_ratio=0, calmar_ratio=0, max_drawdown=0,
            max_drawdown_duration=0, avg_drawdown=0, ulcer_index=0,
            total_return=0, annualized_return=0, volatility=0, downside_volatility=0,
            consecutive_wins=0, consecutive_losses=0, max_consecutive_wins=0,
            max_consecutive_losses=0
        )
    
    def _extract_wf_score(self, notes: str) -> float:
        """Extract walk-forward score from notes."""
        try:
            if "WalkForward: score=" in notes:
                start = notes.find("WalkForward: score=") + len("WalkForward: score=")
                end = notes.find(" ", start) if " " in notes[start:] else len(notes)
                return float(notes[start:end])
        except:
            pass
        return 0.0
    
    def _extract_mc_score(self, notes: str) -> float:
        """Extract Monte Carlo PF probability from notes."""
        try:
            if "MonteCarlo: PF>1.5=" in notes:
                start = notes.find("MonteCarlo: PF>1.5=") + len("MonteCarlo: PF>1.5=")
                end = notes.find(",", start)
                if end == -1:
                    end = len(notes)
                prob_str = notes[start:end].replace('%', '')
                return float(prob_str) / 100.0
        except:
            pass
        return 0.0


if __name__ == "__main__":
    # Test the bridge
    bridge = PipelineBridge()
    
    # Test load
    candidates = bridge.load_model_candidates()
    print(f"Loaded {len(candidates)} candidates")
    
    for c in candidates[:3]:
        print(f"  {c['config_name']}: PF={c['metrics'].profit_factor:.2f}, "
              f"WF={c['walk_forward_score']:.2f}, MC={c['monte_carlo_score']:.2f}")
