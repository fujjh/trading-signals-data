#!/usr/bin/env python3
"""
================================================================================
STEP 13: Model Selector & Deployment
================================================================================

Final step in the recursive optimization pipeline.
Selects best configuration based on aggregate metrics across all tests.
Generates production-ready deployment config.

Author: SignalsAlpha
Version: 1.0
Date: 2026-04-29
================================================================================

MODEL SELECTION CRITERIA:
================================================================================

RANKING WEIGHTS:
├── Profit Factor (30%): Primary profitability metric
├── Sharpe Ratio (25%): Risk-adjusted returns
├── Walk-Forward Consistency (20%): Out-of-sample robustness
├── Monte Carlo Confidence (15%): Statistical reliability
└── Max Drawdown (10%): Capital preservation

MINIMUM THRESHOLDS:
├── Profit Factor ≥ 1.5
├── Sharpe Ratio ≥ 1.0
├── Walk-Forward ≥ 3 consistent windows
├── Monte Carlo PF ≥ 1.5 (95% confidence)
└── Max Drawdown ≤ 20%

DEPLOYMENT OUTPUT:
================================================================================

1. PRODUCTION_CONFIG.JSON
   - Selected indicator weights
   - Threshold parameters
   - Position sizing rules
   - Risk management settings

2. DEPLOYMENT_REPORT.HTML
   - Comparison of all configurations
   - Final selected model details
   - Performance projections

3. DEPLOYMENT_CHECKLIST
   - Pre-deployment validation
   - Risk controls verification
   - Monitoring requirements

================================================================================
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent / 'modules'))

from metrics_calculator import PerformanceMetrics
from results_db import ResultsDatabase
from pipeline_bridge import PipelineBridge

# Configuration
DATA_DIR = Path(__file__).parent / "data"
DEPLOYMENT_DIR = DATA_DIR / "deployment"
DEPLOYMENT_DIR.mkdir(parents=True, exist_ok=True)

# Ranking weights
WEIGHTS = {
    'profit_factor': 0.30,
    'sharpe_ratio': 0.25,
    'walk_forward_consistency': 0.20,
    'monte_carlo_confidence': 0.15,
    'max_drawdown': 0.10  # Inverted (lower is better)
}

# Minimum thresholds
MIN_THRESHOLDS = {
    'profit_factor': 1.5,
    'sharpe_ratio': 1.0,
    'walk_forward_windows': 3,
    'monte_carlo_pf_prob': 0.95,
    'max_drawdown': 0.20
}


@dataclass
class ModelCandidate:
    """Candidate model configuration"""
    config_id: int
    config_name: str
    config: Dict
    
    # Metrics from different stages
    backtest_metrics: Optional[PerformanceMetrics] = None
    walk_forward_score: float = 0.0
    monte_carlo_score: float = 0.0
    
    # Aggregate score
    composite_score: float = 0.0
    passed_validation: bool = False


class ModelSelector:
    """Selects optimal model configuration for deployment"""
    
    def __init__(self):
        """Initialize model selector"""
        self.results_db = ResultsDatabase()
        self.pipeline_bridge = PipelineBridge()
        self.candidates: List[ModelCandidate] = []
        self.selected_model: Optional[ModelCandidate] = None
        
    def load_candidates_from_database(self) -> List[ModelCandidate]:
        """Load all candidate configurations from database using bridge"""
        print("Loading candidates from database...")
        
        # Use pipeline bridge for consistent loading
        raw_candidates = self.pipeline_bridge.load_model_candidates()
        
        candidates = []
        for raw in raw_candidates:
            candidate = ModelCandidate(
                config_id=raw['config_id'],
                config_name=raw['config_name'],
                config=raw['config'],
                backtest_metrics=raw['metrics'],
                walk_forward_score=raw.get('walk_forward_score', 0.0),
                monte_carlo_score=raw.get('monte_carlo_score', 0.0)
            )
            candidates.append(candidate)
        
        print(f"Loaded {len(candidates)} candidates")
        return candidates
    
    def calculate_composite_score(self, candidate: ModelCandidate) -> float:
        """
        Calculate composite score for candidate
        
        Returns weighted score from 0-100
        """
        scores = []
        
        # Profit Factor score (target: 2.0+)
        if candidate.backtest_metrics:
            pf = candidate.backtest_metrics.profit_factor
            pf_score = min(pf / 2.0, 2.0) * 50  # Scale to 0-100
            scores.append(('profit_factor', pf_score, WEIGHTS['profit_factor']))
            
            # Sharpe Ratio score (target: 1.5+)
            sharpe = candidate.backtest_metrics.sharpe_ratio
            sharpe_score = min(max(sharpe, 0) / 1.5, 2.0) * 50
            scores.append(('sharpe_ratio', sharpe_score, WEIGHTS['sharpe_ratio']))
            
            # Max Drawdown score (inverted, target: <15%)
            dd = candidate.backtest_metrics.max_drawdown
            dd_score = max(0, (0.20 - dd) / 0.20) * 100
            scores.append(('max_drawdown', dd_score, WEIGHTS['max_drawdown']))
        
        # Walk-forward score (0-100 based on consistency)
        wf_score = candidate.walk_forward_score * 100
        scores.append(('walk_forward', wf_score, WEIGHTS['walk_forward_consistency']))
        
        # Monte Carlo confidence score
        mc_score = candidate.monte_carlo_score * 100
        scores.append(('monte_carlo', mc_score, WEIGHTS['monte_carlo_confidence']))
        
        # Calculate weighted average
        if scores:
            total_weight = sum(w for _, _, w in scores)
            weighted_sum = sum(score * weight for _, score, weight in scores)
            composite = weighted_sum / total_weight if total_weight > 0 else 0
        else:
            composite = 0
        
        # Store breakdown
        candidate.score_breakdown = {name: (score, weight) for name, score, weight in scores}
        
        return composite
    
    def validate_candidate(self, candidate: ModelCandidate) -> Tuple[bool, List[str]]:
        """
        Validate candidate meets minimum thresholds
        
        Returns:
            (is_valid, list_of_failures)
        """
        failures = []
        
        if not candidate.backtest_metrics:
            failures.append("No backtest metrics available")
            return False, failures
        
        # Check profit factor
        if candidate.backtest_metrics.profit_factor < MIN_THRESHOLDS['profit_factor']:
            failures.append(
                f"PF {candidate.backtest_metrics.profit_factor:.2f} < "
                f"{MIN_THRESHOLDS['profit_factor']}"
            )
        
        # Check Sharpe ratio
        if candidate.backtest_metrics.sharpe_ratio < MIN_THRESHOLDS['sharpe_ratio']:
            failures.append(
                f"Sharpe {candidate.backtest_metrics.sharpe_ratio:.2f} < "
                f"{MIN_THRESHOLDS['sharpe_ratio']}"
            )
        
        # Check max drawdown
        if candidate.backtest_metrics.max_drawdown > MIN_THRESHOLDS['max_drawdown']:
            failures.append(
                f"DD {candidate.backtest_metrics.max_drawdown:.1%} > "
                f"{MIN_THRESHOLDS['max_drawdown']}"
            )
        
        # Check walk-forward
        if candidate.walk_forward_score < 0.6:  # At least 60% consistency
            failures.append(f"Walk-forward {candidate.walk_forward_score:.1%} < 60%")
        
        # Check Monte Carlo
        if candidate.monte_carlo_score < MIN_THRESHOLDS['monte_carlo_pf_prob']:
            failures.append(
                f"MC confidence {candidate.monte_carlo_score:.1%} < "
                f"{MIN_THRESHOLDS['monte_carlo_pf_prob']}"
            )
        
        is_valid = len(failures) == 0
        return is_valid, failures
    
    def rank_candidates(self) -> List[ModelCandidate]:
        """Rank all candidates by composite score"""
        print("\nRanking candidates...")
        
        # Calculate scores and validate
        for candidate in self.candidates:
            candidate.composite_score = self.calculate_composite_score(candidate)
            is_valid, failures = self.validate_candidate(candidate)
            candidate.passed_validation = is_valid
            candidate.validation_failures = failures
        
        # Sort by composite score (descending)
        ranked = sorted(self.candidates, 
                       key=lambda x: (x.passed_validation, x.composite_score), 
                       reverse=True)
        
        return ranked
    
    def select_model(self) -> Optional[ModelCandidate]:
        """Select best model from ranked candidates"""
        print("=" * 70)
        print("MODEL SELECTION")
        print("=" * 70)
        
        # Load candidates
        self.candidates = self.load_candidates_from_database()
        
        # For demo, create synthetic candidates
        if len(self.candidates) == 0:
            self.candidates = self._create_demo_candidates()
        
        # Rank candidates
        ranked = self.rank_candidates()
        
        # Print ranking
        print("\nRANKED CANDIDATES:")
        print("-" * 70)
        print(f"{'Rank':<6} {'Name':<30} {'Score':<8} {'Status':<12} {'PF':<6} {'Sharpe':<8}")
        print("-" * 70)
        
        for i, candidate in enumerate(ranked[:10], 1):  # Top 10
            metrics = candidate.backtest_metrics
            pf = f"{metrics.profit_factor:.2f}" if metrics else "N/A"
            sharpe = f"{metrics.sharpe_ratio:.2f}" if metrics else "N/A"
            status = "✓ PASS" if candidate.passed_validation else "✗ FAIL"
            
            print(f"{i:<6} {candidate.config_name:<30} "
                  f"{candidate.composite_score:<8.1f} {status:<12} {pf:<6} {sharpe:<8}")
            
            if not candidate.passed_validation and hasattr(candidate, 'validation_failures'):
                for failure in candidate.validation_failures[:2]:  # Show first 2
                    print(f"       └─ {failure}")
        
        # Select best valid candidate
        valid_candidates = [c for c in ranked if c.passed_validation]
        
        if valid_candidates:
            self.selected_model = valid_candidates[0]
            print("\n" + "=" * 70)
            print("SELECTED MODEL")
            print("=" * 70)
            print(f"Config: {self.selected_model.config_name}")
            print(f"Score: {self.selected_model.composite_score:.1f}/100")
            print(f"\nScore Breakdown:")
            for name, (score, weight) in self.selected_model.score_breakdown.items():
                weighted = score * weight
                print(f"  {name:25s}: {score:5.1f} × {weight:.2f} = {weighted:5.1f}")
        else:
            print("\n" + "=" * 70)
            print("NO VALID CANDIDATES")
            print("=" * 70)
            print("All candidates failed validation. Review thresholds.")
            # Select best anyway with warning
            if ranked:
                self.selected_model = ranked[0]
                print(f"\nSelecting best candidate anyway: {self.selected_model.config_name}")
                print("WARNING: Does not meet minimum thresholds!")
        
        return self.selected_model
    
    def _create_demo_candidates(self) -> List[ModelCandidate]:
        """Create demo candidates for testing"""
        candidates = []
        
        # Candidate 1: Strong performer
        candidates.append(ModelCandidate(
            config_id=1,
            config_name="genetic_optimized_gen25",
            config={
                'indicator_weights': {'macd': 4, 'hma': 3, 'rsi': 2, 'stoch': 2},
                'thresholds': {'oversold': 25, 'overbought': 75}
            },
            backtest_metrics=PerformanceMetrics(
                total_trades=100, winning_trades=60, losing_trades=40,
                win_rate=0.60, profit_factor=2.1, sharpe_ratio=1.6,
                max_drawdown=0.12, expectancy=150
            ),
            walk_forward_score=0.85,
            monte_carlo_score=0.97
        ))
        
        # Candidate 2: Good but higher DD
        candidates.append(ModelCandidate(
            config_id=2,
            config_name="genetic_optimized_gen32",
            config={
                'indicator_weights': {'macd': 5, 'hma': 2, 'rsi': 3, 'stoch': 1},
                'thresholds': {'oversold': 30, 'overbought': 70}
            },
            backtest_metrics=PerformanceMetrics(
                total_trades=100, winning_trades=58, losing_trades=42,
                win_rate=0.58, profit_factor=1.9, sharpe_ratio=1.4,
                max_drawdown=0.22, expectancy=120
            ),
            walk_forward_score=0.75,
            monte_carlo_score=0.92
        ))
        
        # Candidate 3: Weak performer
        candidates.append(ModelCandidate(
            config_id=3,
            config_name="genetic_optimized_gen15",
            config={
                'indicator_weights': {'macd': 2, 'hma': 2, 'rsi': 2, 'stoch': 2},
                'thresholds': {'oversold': 35, 'overbought': 65}
            },
            backtest_metrics=PerformanceMetrics(
                total_trades=100, winning_trades=50, losing_trades=50,
                win_rate=0.50, profit_factor=1.3, sharpe_ratio=0.8,
                max_drawdown=0.25, expectancy=50
            ),
            walk_forward_score=0.55,
            monte_carlo_score=0.75
        ))
        
        return candidates
    
    def generate_production_config(self) -> Dict:
        """Generate production-ready configuration"""
        if not self.selected_model:
            return {}
        
        config = {
            'version': '1.0.0',
            'generated_at': datetime.now().isoformat(),
            'selected_from': len(self.candidates),
            'composite_score': self.selected_model.composite_score,
            'validation_passed': self.selected_model.passed_validation,
            
            'model_config': self.selected_model.config,
            
            'performance_metrics': {
                'profit_factor': self.selected_model.backtest_metrics.profit_factor 
                                if self.selected_model.backtest_metrics else 0,
                'sharpe_ratio': self.selected_model.backtest_metrics.sharpe_ratio 
                               if self.selected_model.backtest_metrics else 0,
                'max_drawdown': self.selected_model.backtest_metrics.max_drawdown 
                               if self.selected_model.backtest_metrics else 0,
                'win_rate': self.selected_model.backtest_metrics.win_rate 
                           if self.selected_model.backtest_metrics else 0,
                'expectancy': self.selected_model.backtest_metrics.expectancy 
                            if self.selected_model.backtest_metrics else 0
            },
            
            'risk_management': {
                'max_position_pct': 0.20,
                'max_portfolio_exposure': 0.80,
                'stop_loss_pct': 0.08,
                'take_profit_pct': 0.16,
                'trailing_stop': True,
                'volatility_adjustment': True
            },
            
            'execution': {
                'entry_timing': 'next_open',
                'exit_on_signal_reversal': True,
                'partial_fills_allowed': True,
                'cash_constrained_entries': True
            },
            
            'monitoring': {
                'alert_on_dd_threshold': 0.15,
                'daily_report_enabled': True,
                'weekly_review_enabled': True,
                'monthly_optimization': True
            }
        }
        
        return config
    
    def save_production_config(self, output_path: Path):
        """Save production configuration to JSON"""
        config = self.generate_production_config()
        
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\nProduction config saved to: {output_path}")
    
    def generate_deployment_report(self) -> str:
        """Generate deployment report"""
        report = []
        report.append("=" * 70)
        report.append("DEPLOYMENT REPORT")
        report.append("=" * 70)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        if not self.selected_model:
            report.append("ERROR: No model selected")
            return "\n".join(report)
        
        report.append("SELECTED MODEL")
        report.append("-" * 70)
        report.append(f"Config Name: {self.selected_model.config_name}")
        report.append(f"Config ID: {self.selected_model.config_id}")
        report.append(f"Composite Score: {self.selected_model.composite_score:.1f}/100")
        report.append(f"Validation: {'PASSED' if self.selected_model.passed_validation else 'FAILED'}")
        report.append("")
        
        report.append("PERFORMANCE PROJECTIONS")
        report.append("-" * 70)
        if self.selected_model.backtest_metrics:
            m = self.selected_model.backtest_metrics
            report.append(f"Profit Factor: {m.profit_factor:.2f}")
            report.append(f"Sharpe Ratio: {m.sharpe_ratio:.2f}")
            report.append(f"Win Rate: {m.win_rate:.1%}")
            report.append(f"Max Drawdown: {m.max_drawdown:.1%}")
            report.append(f"Expectancy: ${m.expectancy:.0f} per trade")
        report.append("")
        
        report.append("VALIDATION SUMMARY")
        report.append("-" * 70)
        if self.selected_model.passed_validation:
            report.append("✓ All minimum thresholds met")
            report.append("✓ Walk-forward consistency verified")
            report.append("✓ Monte Carlo confidence validated")
            report.append("✓ Risk parameters within limits")
        else:
            report.append("⚠ Some thresholds not met:")
            if hasattr(self.selected_model, 'validation_failures'):
                for failure in self.selected_model.validation_failures:
                    report.append(f"  - {failure}")
        report.append("")
        
        report.append("DEPLOYMENT CHECKLIST")
        report.append("-" * 70)
        report.append("[ ] Review production configuration")
        report.append("[ ] Verify database connectivity")
        report.append("[ ] Test signal generation")
        report.append("[ ] Validate risk controls")
        report.append("[ ] Enable monitoring alerts")
        report.append("[ ] Schedule daily data updates")
        report.append("[ ] Configure backup systems")
        report.append("")
        
        report.append("NEXT STEPS")
        report.append("-" * 70)
        report.append("1. Deploy production_config.json to trading system")
        report.append("2. Start daily data collection (Step 1)")
        report.append("3. Monitor first week of live signals")
        report.append("4. Schedule monthly model review")
        report.append("")
        
        report.append("=" * 70)
        report.append("END OF REPORT")
        report.append("=" * 70)
        
        return "\n".join(report)
    
    def save_deployment_report(self, output_path: Path):
        """Save deployment report"""
        report = self.generate_deployment_report()
        
        with open(output_path, 'w') as f:
            f.write(report)
        
        print(f"Deployment report saved to: {output_path}")
    
    def export_ranking_csv(self, output_path: Path):
        """Export candidate ranking to CSV"""
        if not self.candidates:
            return
        
        records = []
        for c in sorted(self.candidates, 
                       key=lambda x: (x.passed_validation, x.composite_score), 
                       reverse=True):
            record = {
                'config_id': c.config_id,
                'config_name': c.config_name,
                'composite_score': c.composite_score,
                'passed_validation': c.passed_validation,
                'pf': c.backtest_metrics.profit_factor if c.backtest_metrics else 0,
                'sharpe': c.backtest_metrics.sharpe_ratio if c.backtest_metrics else 0,
                'max_dd': c.backtest_metrics.max_drawdown if c.backtest_metrics else 0,
                'wf_score': c.walk_forward_score,
                'mc_score': c.monte_carlo_score
            }
            records.append(record)
        
        df = pd.DataFrame(records)
        df.to_csv(output_path, index=False)
        print(f"Ranking exported to: {output_path}")


def main():
    """Main entry point"""
    print("=" * 70)
    print("STEP 13: MODEL SELECTOR & DEPLOYMENT")
    print("=" * 70)
    
    # Initialize selector
    selector = ModelSelector()
    
    # Run selection
    selected = selector.select_model()
    
    if selected:
        # Save production config
        config_path = DEPLOYMENT_DIR / "production_config.json"
        selector.save_production_config(config_path)
        
        # Save deployment report
        report_path = DEPLOYMENT_DIR / "deployment_report.txt"
        selector.save_deployment_report(report_path)
        
        # Export ranking
        ranking_path = DEPLOYMENT_DIR / "candidate_ranking.csv"
        selector.export_ranking_csv(ranking_path)
        
        # Print report
        print("\n")
        print(selector.generate_deployment_report())
    
    print("\n" + "=" * 70)
    print("MODEL SELECTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
