#!/usr/bin/env python3
"""
================================================================================
STEP 4 VALIDATION TEST - STRONG BUY SIGNALS
================================================================================

Validates STRONG_BUY signals from Step 4 output to ensure they meet
criteria and have reasonable price targets.

Tests:
1. STRONG_BUY signals have combined_score >= 75
2. Technical scores are positive (bullish technicals)
3. Stop loss is below current price (valid risk management)
4. Take profit is above current price (positive expected return)
5. Risk/Reward ratio is reasonable (>= 1:1)
6. Key drivers are present and meaningful
7. Grade distribution is sensible

Author: SignalsAlpha
Date: 2026-04-21
================================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

DATA_DIR = Path("/home/ubuntu/.openclaw/workspace/trading-signals-data/process_flow_v1/data")
SIGNALS_FILE = DATA_DIR / "signals_scored" / "scored_signals_20260421.csv"

def test_strong_buy_signals():
    """Run validation tests on STRONG_BUY signals"""
    
    print("="*70)
    print("STEP 4 STRONG BUY SIGNAL VALIDATION TEST")
    print("="*70)
    print()
    
    # Load signals
    if not SIGNALS_FILE.exists():
        print(f"ERROR: Signals file not found: {SIGNALS_FILE}")
        return False
    
    df = pd.read_csv(SIGNALS_FILE)
    
    # Filter STRONG_BUY signals
    strong_buys = df[df['final_signal'] == 'STRONG_BUY'].copy()
    
    print(f"Total signals: {len(df):,}")
    print(f"STRONG_BUY signals: {len(strong_buys):,}")
    print()
    
    tests_passed = 0
    tests_failed = 0
    
    # Test 1: Score threshold
    print("Test 1: Combined score >= 75")
    below_threshold = strong_buys[strong_buys['combined_score'] < 75]
    if len(below_threshold) == 0:
        print(f"  ✓ PASS - All {len(strong_buys)} signals meet threshold")
        tests_passed += 1
    else:
        print(f"  ✗ FAIL - {len(below_threshold)} signals below threshold")
        tests_failed += 1
    
    # Test 2: Technical bullish
    print("\nTest 2: Technical scores are bullish")
    bearish_tech = strong_buys[strong_buys['technical_buy_score'] <= strong_buys['technical_sell_score']]
    if len(bearish_tech) == 0:
        print(f"  ✓ PASS - All signals have bullish technicals")
        tests_passed += 1
    else:
        print(f"  ✗ FAIL - {len(bearish_tech)} signals with bearish technicals")
        tests_failed += 1
    
    # Test 3: Stop loss below price
    print("\nTest 3: Stop loss below current price")
    invalid_stops = strong_buys[strong_buys['stop_loss'] >= strong_buys['close']]
    if len(invalid_stops) == 0:
        print(f"  ✓ PASS - All stop losses valid")
        tests_passed += 1
    else:
        print(f"  ✗ FAIL - {len(invalid_stops)} signals with stop >= price")
        tests_failed += 1
    
    # Test 4: Take profit above price
    print("\nTest 4: Take profit above current price")
    invalid_targets = strong_buys[strong_buys['take_profit'] <= strong_buys['close']]
    if len(invalid_targets) == 0:
        print(f"  ✓ PASS - All take profits valid")
        tests_passed += 1
    else:
        print(f"  ✗ FAIL - {len(invalid_targets)} signals with target <= price")
        tests_failed += 1
    
    # Test 5: Risk/Reward ratio
    print("\nTest 5: Risk/Reward ratio >= 1:1")
    strong_buys['risk'] = strong_buys['close'] - strong_buys['stop_loss']
    strong_buys['reward'] = strong_buys['take_profit'] - strong_buys['close']
    strong_buys['rr_ratio'] = strong_buys['reward'] / strong_buys['risk']
    
    poor_rr = strong_buys[strong_buys['rr_ratio'] < 1.0]
    if len(poor_rr) == 0:
        print(f"  ✓ PASS - All signals have R/R >= 1:1")
        print(f"         Average R/R: {strong_buys['rr_ratio'].mean():.2f}")
        tests_passed += 1
    else:
        print(f"  ✗ FAIL - {len(poor_rr)} signals with R/R < 1")
        tests_failed += 1
    
    # Test 6: Key drivers present
    print("\nTest 6: Key drivers are present")
    no_drivers = strong_buys[strong_buys['key_drivers'].isna() | (strong_buys['key_drivers'] == 'None')]
    if len(no_drivers) == 0:
        print(f"  ✓ PASS - All signals have key drivers")
        tests_passed += 1
    else:
        print(f"  ✗ FAIL - {len(no_drivers)} signals missing drivers")
        tests_failed += 1
    
    # Test 7: Fundamental grades
    print("\nTest 7: Grade distribution analysis")
    grade_dist = strong_buys['fundamental_grade'].value_counts().sort_index()
    print(f"  Grade distribution:")
    for grade in ['A', 'B', 'C', 'D', 'F']:
        count = grade_dist.get(grade, 0)
        pct = count / len(strong_buys) * 100 if len(strong_buys) > 0 else 0
        print(f"    {grade}: {count} ({pct:.1f}%)")
    
    # Most STRONG_BUY should have decent fundamentals
    good_grades = strong_buys[strong_buys['fundamental_grade'].isin(['A', 'B', 'C'])]
    if len(good_grades) >= len(strong_buys) * 0.5:
        print(f"  ✓ PASS - {len(good_grades)}/{len(strong_buys)} have grade C or better")
        tests_passed += 1
    else:
        print(f"  ⚠ WARNING - Only {len(good_grades)}/{len(strong_buys)} have grade C or better")
        tests_failed += 1
    
    # Summary statistics
    print("\n" + "="*70)
    print("STRONG BUY STATISTICS")
    print("="*70)
    print(f"Count: {len(strong_buys)}")
    print(f"Avg Combined Score: {strong_buys['combined_score'].mean():.1f}")
    print(f"Avg Technical Score: {strong_buys['technical_raw_score'].mean():.1f}")
    print(f"Avg Fundamental Score: {strong_buys['fundamental_score'].mean():.1f}")
    print(f"Avg Conviction: {strong_buys['conviction_pct'].mean():.1f}%")
    print(f"Avg Risk/Reward: {strong_buys['rr_ratio'].mean():.2f}")
    print(f"\nTop Intervals:")
    print(strong_buys['interval'].value_counts())
    
    # Top 10 by score
    print("\n" + "="*70)
    print("TOP 10 STRONG BUY SIGNALS")
    print("="*70)
    top_10 = strong_buys.nlargest(10, 'combined_score')
    for idx, row in top_10.iterrows():
        print(f"{row['ticker']:6s} | {row['interval']:3s} | Score: {row['combined_score']:2d} | "
              f"Tech: {row['technical_buy_score']}/{row['technical_sell_score']} | "
              f"Grade: {row['fundamental_grade']} | R/R: {row['rr_ratio']:.2f}")
    
    # Test summary
    print("\n" + "="*70)
    print(f"TEST SUMMARY: {tests_passed} passed, {tests_failed} failed")
    print("="*70)
    
    return tests_failed == 0


if __name__ == "__main__":
    success = test_strong_buy_signals()
    exit(0 if success else 1)
