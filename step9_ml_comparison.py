#!/usr/bin/env python3
"""
================================================================================
STEP 9: Machine Learning Comparison Dashboard
================================================================================

Compare GA, XGBoost, LSTM, and RL strategies side-by-side.
Generates comprehensive comparison report.

Author: SignalsAlpha
Version: 1.0 (Comparison Edition)
Date: 2026-04-29
================================================================================
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import subprocess

OUTPUT_DIR = Path("data/optimizer")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def check_dependencies():
    """Check which ML libraries are available."""
    deps = {
        'xgboost': False,
        'tensorflow': False,
        'torch': False
    }
    
    try:
        import xgboost
        deps['xgboost'] = True
    except ImportError:
        pass
    
    try:
        import tensorflow
        deps['tensorflow'] = True
    except ImportError:
        pass
    
    try:
        import torch
        deps['torch'] = True
    except ImportError:
        pass
    
    return deps


def run_ga_baseline():
    """Get GA baseline results (already computed)."""
    # Load the best config from multi-seed
    best_file = OUTPUT_DIR / "spy_best_config_grid_search_20260429_184201.json"
    if best_file.exists():
        with open(best_file) as f:
            data = json.load(f)
        return {
            'method': 'Genetic Algorithm (100-seed + Grid)',
            'fitness': data['best_fitness'],
            'return': data['best_config']['strategy_return'] * 100,
            'sharpe': data['best_config']['details']['sharpe'],
            'max_dd': data['best_config']['details']['max_dd'] * 100,
            'win_rate': data['best_config']['details']['win_rate'] * 100,
            'trades': data['best_config']['details']['trades']
        }
    return None


def run_comparison():
    """Run all available methods and compare."""
    print("=" * 80)
    print("STEP 9: Machine Learning Strategy Comparison")
    print("=" * 80)
    
    deps = check_dependencies()
    print("\nDependencies:")
    for lib, available in deps.items():
        status = "✓" if available else "✗"
        print(f"  {status} {lib}")
    
    results = []
    
    # Get GA baseline (always available)
    print("\n" + "=" * 80)
    print("Loading GA Baseline...")
    print("=" * 80)
    ga_result = run_ga_baseline()
    if ga_result:
        results.append(ga_result)
        print(f"✓ GA baseline loaded: fitness={ga_result['fitness']:.4f}")
    
    # Run XGBoost
    if deps['xgboost']:
        print("\n" + "=" * 80)
        print("Running XGBoost...")
        print("=" * 80)
        try:
            subprocess.run(['python3', 'step9_xgboost_trainer.py'], 
                          cwd='/home/ubuntu/.openclaw/workspace/trading-signals-data/process_flow_v1_SPY',
                          timeout=300, capture_output=True)
            
            # Load latest result
            xgb_files = sorted(OUTPUT_DIR.glob('spy_xgboost_config_*.json'))
            if xgb_files:
                with open(xgb_files[-1]) as f:
                    data = json.load(f)
                results.append({
                    'method': 'XGBoost',
                    'fitness': None,  # Not directly comparable
                    'return': data['strategy_return'] * 100,
                    'sharpe': data['strategy_sharpe'],
                    'max_dd': data['strategy_max_dd'] * 100,
                    'win_rate': data['strategy_win_rate'] * 100,
                    'trades': data['strategy_trades'],
                    'test_accuracy': data['test_accuracy'],
                    'test_f1': data['test_f1']
                })
                print(f"✓ XGBoost complete: return={data['strategy_return']*100:.2f}%")
        except Exception as e:
            print(f"✗ XGBoost failed: {e}")
    
    # Run LSTM
    if deps['tensorflow']:
        print("\n" + "=" * 80)
        print("Running LSTM...")
        print("=" * 80)
        try:
            subprocess.run(['python3', 'step9_lstm_trainer.py'],
                          cwd='/home/ubuntu/.openclaw/workspace/trading-signals-data/process_flow_v1_SPY',
                          timeout=600, capture_output=True)
            
            lstm_files = sorted(OUTPUT_DIR.glob('spy_lstm_config_*.json'))
            if lstm_files:
                with open(lstm_files[-1]) as f:
                    data = json.load(f)
                results.append({
                    'method': 'LSTM',
                    'fitness': None,
                    'return': data['strategy_return'] * 100,
                    'sharpe': data['strategy_sharpe'],
                    'max_dd': data['strategy_max_dd'] * 100,
                    'win_rate': data['strategy_win_rate'] * 100,
                    'trades': data['strategy_trades'],
                    'test_accuracy': data['test_accuracy'],
                    'test_f1': data['test_f1']
                })
                print(f"✓ LSTM complete: return={data['strategy_return']*100:.2f}%")
        except Exception as e:
            print(f"✗ LSTM failed: {e}")
    
    # Run RL
    if deps['torch']:
        print("\n" + "=" * 80)
        print("Running PPO (Reinforcement Learning)...")
        print("=" * 80)
        try:
            subprocess.run(['python3', 'step9_rl_trainer.py'],
                          cwd='/home/ubuntu/.openclaw/workspace/trading-signals-data/process_flow_v1_SPY',
                          timeout=900, capture_output=True)
            
            rl_files = sorted(OUTPUT_DIR.glob('spy_ppo_config_*.json'))
            if rl_files:
                with open(rl_files[-1]) as f:
                    data = json.load(f)
                results.append({
                    'method': 'PPO (RL)',
                    'fitness': None,
                    'return': data['final_return'] * 100,
                    'sharpe': None,  # Not calculated in RL
                    'max_dd': None,
                    'win_rate': None,
                    'trades': data['n_trades'],
                    'outperformance': data['outperformance'] * 100
                })
                print(f"✓ PPO complete: return={data['final_return']*100:.2f}%")
        except Exception as e:
            print(f"✗ PPO failed: {e}")
    
    # Generate comparison report
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    
    if results:
        print(f"\n{'Method':<35} {'Return':<10} {'Sharpe':<8} {'MaxDD':<8} {'Trades':<8}")
        print("-" * 80)
        
        for r in results:
            ret = f"{r['return']:.1f}%" if r['return'] is not None else "N/A"
            sharpe = f"{r['sharpe']:.2f}" if r['sharpe'] is not None else "N/A"
            maxdd = f"{r['max_dd']:.1f}%" if r['max_dd'] is not None else "N/A"
            trades = f"{r['trades']}" if r['trades'] is not None else "N/A"
            print(f"{r['method']:<35} {ret:<10} {sharpe:<8} {maxdd:<8} {trades:<8}")
        
        # Save comparison
        comparison_file = OUTPUT_DIR / f"ml_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(comparison_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'results': results,
                'dependencies': deps
            }, f, indent=2)
        
        print(f"\nSaved comparison to: {comparison_file}")
    else:
        print("\nNo results to compare.")
    
    print("=" * 80)


if __name__ == "__main__":
    run_comparison()
