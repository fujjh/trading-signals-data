#!/usr/bin/env python3
"""
================================================================================
STEP 13: SPY Model Selector
================================================================================

Production model selection for SPY-only strategy.

Author: SignalsAlpha
Version: 1.0 (SPY Edition)
Date: 2026-04-29

================================================================================
SELECTION CRITERIA
================================================================================

Composite Score:
- Profit Factor: 30%
- Sharpe Ratio: 25%
- Walk-Forward: 20%
- Monte Carlo: 15%
- Drawdown: 10%

Minimum Thresholds:
- Profit Factor >= 1.5
- Sharpe Ratio >= 1.0
- Max Drawdown < 20%

================================================================================
INPUT
================================================================================

Source: data/backtests/spy_backtest_*.json
        data/walk_forward/spy_walk_forward_*.json
        data/monte_carlo/spy_monte_carlo_*.json

================================================================================
OUTPUT
================================================================================

Destination: data/production_config/spy_production_config.json

"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import json
import glob

# Configuration
TICKER = "SPY"
BACKTEST_DIR = Path("data/backtests")
WALK_FORWARD_DIR = Path("data/walk_forward")
MONTE_CARLO_DIR = Path("data/monte_carlo")
OPTIMIZER_DIR = Path("data/optimizer")
OUTPUT_DIR = Path("data/production_config")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Selection weights
WEIGHTS = {
    'profit_factor': 0.30,
    'sharpe_ratio': 0.25,
    'walk_forward': 0.20,
    'monte_carlo': 0.15,
    'drawdown': 0.10
}

# Minimum thresholds
MIN_PROFIT_FACTOR = 1.5
MIN_SHARPE = 1.0
MAX_DRAWDOWN = 0.20


def load_latest_results():
    """Load latest results from all steps."""
    results = {}
    
    # Load backtest results
    backtest_files = list(BACKTEST_DIR.glob("spy_backtest_*.json"))
    if backtest_files:
        latest = max(backtest_files, key=lambda p: p.stat().st_mtime)
        with open(latest) as f:
            results['backtest'] = json.load(f)
    
    # Load walk-forward results
    wf_files = list(WALK_FORWARD_DIR.glob("spy_walk_forward_*.json"))
    if wf_files:
        latest = max(wf_files, key=lambda p: p.stat().st_mtime)
        with open(latest) as f:
            results['walk_forward'] = json.load(f)
    
    # Load Monte Carlo results
    mc_files = list(MONTE_CARLO_DIR.glob("spy_monte_carlo_*.json"))
    if mc_files:
        latest = max(mc_files, key=lambda p: p.stat().st_mtime)
        with open(latest) as f:
            results['monte_carlo'] = json.load(f)
    
    # Load optimizer results - prefer v3 configs, then v2, then v1
    opt_files = list(OPTIMIZER_DIR.glob("spy_best_config_v3_*.json"))
    if not opt_files:
        opt_files = list(OPTIMIZER_DIR.glob("spy_best_config_v2_*.json"))
    if not opt_files:
        # Fall back to v1 configs
        opt_files = list(OPTIMIZER_DIR.glob("spy_best_config_*.json"))
    
    if opt_files:
        latest = max(opt_files, key=lambda p: p.stat().st_mtime)
        with open(latest) as f:
            results['optimizer'] = json.load(f)
    
    return results


def calculate_composite_score(results: Dict) -> float:
    """Calculate composite score for model selection."""
    scores = {}
    
    # Profit Factor score (from backtest)
    if 'backtest' in results and 'metrics' in results['backtest']:
        pf = results['backtest']['metrics'].get('profit_factor', 0)
        scores['profit_factor'] = min(pf / 2.0, 1.0)  # Normalize: PF=2.0 = score 1.0
    
    # Sharpe Ratio score (from backtest)
    if 'backtest' in results and 'metrics' in results['backtest']:
        sharpe = results['backtest']['metrics'].get('sharpe_ratio', 0)
        scores['sharpe_ratio'] = min(sharpe / 2.0, 1.0)  # Normalize: Sharpe=2.0 = score 1.0
    
    # Walk-Forward score
    if 'walk_forward' in results:
        test_perf = results['walk_forward'].get('test_performance', {})
        wf_return = test_perf.get('avg_return', 0)
        wf_sharpe = test_perf.get('avg_sharpe', 0)
        # Combine return and sharpe
        scores['walk_forward'] = min(max(wf_return * 10, 0) + wf_sharpe * 0.3, 1.0)
    
    # Monte Carlo score
    if 'monte_carlo' in results:
        ci = results['monte_carlo'].get('confidence_intervals', {})
        prob_pf = results['monte_carlo'].get('probability_metrics', {}).get('profit_probability', 0)
        prob_dd = results['monte_carlo'].get('probability_metrics', {}).get('drawdown_under_20', 0)
        # Combine probabilities
        scores['monte_carlo'] = (prob_pf + prob_dd) / 2
    
    # Drawdown score (lower is better)
    if 'backtest' in results and 'metrics' in results['backtest']:
        dd = results['backtest']['metrics'].get('max_drawdown', 1)
        scores['drawdown'] = max(1 - dd / MAX_DRAWDOWN, 0)  # 0 drawdown = score 1.0
    
    # Calculate weighted composite
    composite = sum(scores.get(k, 0) * WEIGHTS.get(k, 0) for k in WEIGHTS.keys())
    
    return composite, scores


def check_thresholds(results: Dict) -> Dict:
    """Check if results meet minimum thresholds."""
    checks = {
        'passed': True,
        'failures': []
    }
    
    # Profit Factor
    if 'backtest' in results and 'metrics' in results['backtest']:
        pf = results['backtest']['metrics'].get('profit_factor', 0)
        if pf < MIN_PROFIT_FACTOR:
            checks['passed'] = False
            checks['failures'].append(f"Profit Factor {pf:.2f} < {MIN_PROFIT_FACTOR}")
    
    # Sharpe Ratio
    if 'backtest' in results and 'metrics' in results['backtest']:
        sharpe = results['backtest']['metrics'].get('sharpe_ratio', 0)
        if sharpe < MIN_SHARPE:
            checks['passed'] = False
            checks['failures'].append(f"Sharpe {sharpe:.2f} < {MIN_SHARPE}")
    
    # Max Drawdown
    if 'backtest' in results and 'metrics' in results['backtest']:
        dd = results['backtest']['metrics'].get('max_drawdown', 1)
        if dd > MAX_DRAWDOWN:
            checks['passed'] = False
            checks['failures'].append(f"Max DD {dd*100:.1f}% > {MAX_DRAWDOWN*100:.0f}%")
    
    return checks


def generate_production_config(results: Dict, composite_score: float, component_scores: Dict) -> Dict:
    """Generate production configuration."""
    
    # Get optimizer config if available
    opt_config = results.get('optimizer', {}).get('config', {})
    
    config = {
        'ticker': TICKER,
        'generated_at': datetime.now().isoformat(),
        'selection_score': {
            'composite': round(composite_score, 4),
            'components': {k: round(v, 4) for k, v in component_scores.items()}
        },
        'indicator_weights': {
            'macd': opt_config.get('macd_weight', 3),
            'hma': opt_config.get('hma_weight', 5),
            'rsi': opt_config.get('rsi_weight', 1),
            'stoch': opt_config.get('stoch_weight', 2),
            'sma': opt_config.get('sma_weight', 0),
            'ema': opt_config.get('ema_weight', 4),
            'mfi': opt_config.get('mfi_weight', 2),
            'bb': opt_config.get('bb_weight', 0)
        },
        'thresholds': {
            'oversold': opt_config.get('oversold', 30),
            'overbought': opt_config.get('overbought', 70),
            'strong_buy': 60,
            'strong_sell': 40
        },
        'position_sizing': {
            'max_position_pct': opt_config.get('max_position_pct', 0.20),
            'volatility_adjust': opt_config.get('volatility_adj', True)
        },
        'risk_management': {
            'stop_loss_pct': 0.05,
            'take_profit_pct': 0.10,
            'max_drawdown_limit': 0.20
        }
    }
    
    return config


def run_model_selection():
    """Main model selection function."""
    print("=" * 70)
    print("STEP 13: SPY Model Selector")
    print("=" * 70)
    
    # Load results
    results = load_latest_results()
    
    if not results:
        print("ERROR: No results found from previous steps")
        return
    
    print(f"Loaded results from {len(results)} sources:")
    for source in results.keys():
        print(f"  - {source}")
    
    # Check thresholds
    threshold_checks = check_thresholds(results)
    
    if not threshold_checks['passed']:
        print("\n" + "=" * 70)
        print("THRESHOLD CHECKS FAILED")
        print("=" * 70)
        for failure in threshold_checks['failures']:
            print(f"  ✗ {failure}")
        print("\nModel does not meet minimum requirements for production")
        print("=" * 70)
    
    # Calculate composite score
    composite_score, component_scores = calculate_composite_score(results)
    
    # Generate production config
    production_config = generate_production_config(results, composite_score, component_scores)
    
    # Print results
    print("\n" + "=" * 70)
    print("MODEL SELECTION RESULTS")
    print("=" * 70)
    print(f"\nComposite Score: {composite_score:.4f}")
    print("\nComponent Scores:")
    for component, score in component_scores.items():
        weight = WEIGHTS.get(component, 0) * 100
        print(f"  {component}: {score:.4f} (weight: {weight:.0f}%)")
    
    print("\nProduction Configuration:")
    print(f"  Max Position: {production_config['position_sizing']['max_position_pct']*100:.0f}%")
    print(f"  Stop Loss: {production_config['risk_management']['stop_loss_pct']*100:.0f}%")
    print(f"  Take Profit: {production_config['risk_management']['take_profit_pct']*100:.0f}%")
    
    if threshold_checks['passed']:
        print("\n✓ Model APPROVED for production")
    else:
        print("\n⚠ Model NOT APPROVED - review threshold failures")
    
    print("=" * 70)
    
    # Save production config
    output_file = OUTPUT_DIR / "spy_production_config.json"
    with open(output_file, 'w') as f:
        json.dump(production_config, f, indent=2)
    
    print(f"\nSaved production config: {output_file}")


if __name__ == "__main__":
    run_model_selection()
