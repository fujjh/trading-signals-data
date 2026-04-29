#!/usr/bin/env python3
"""
================================================================================
STEP 6: SPY Signal Validator
================================================================================

Validates SPY scored signals from Step 5.

Author: SignalsAlpha
Version: 1.0 (SPY Edition)
Date: 2026-04-29
================================================================================
"""

import pandas as pd
from pathlib import Path
import json
from datetime import datetime

SIGNALS_DIR = Path("data/signals_scored")
OUTPUT_DIR = Path("data/validated")
TICKER = "SPY"


def validate_signals():
    """Validate SPY scored signals."""
    print("=" * 70)
    print("STEP 6: SPY Signal Validation")
    print("=" * 70)
    
    # Find latest signals file
    signal_files = list(SIGNALS_DIR.glob("spy_scored_*.csv"))
    if not signal_files:
        print("ERROR: No signal files found")
        return False
    
    latest_file = max(signal_files, key=lambda p: p.stat().st_mtime)
    print(f"Validating: {latest_file}")
    
    df = pd.read_csv(latest_file)
    
    errors = []
    
    # Check score ranges
    if 'combined_score' in df.columns:
        invalid_scores = ((df['combined_score'] < 0) | (df['combined_score'] > 100)).sum()
        if invalid_scores > 0:
            errors.append(f"{invalid_scores} scores outside 0-100 range")
    
    # Check signal values
    valid_signals = ['STRONG_BUY', 'BUY', 'HOLD', 'SELL', 'STRONG_SELL']
    if 'signal' in df.columns:
        invalid = (~df['signal'].isin(valid_signals)).sum()
        if invalid > 0:
            errors.append(f"{invalid} invalid signal values")
    
    # Check for recent data
    if len(df) < 50:
        errors.append(f"Insufficient data: {len(df)} rows (min 50)")
    
    # Build report
    result = {
        'ticker': TICKER,
        'file': str(latest_file),
        'status': 'PASS' if not errors else 'FAIL',
        'errors': errors,
        'total_signals': len(df),
        'latest_score': float(df['combined_score'].iloc[-1]) if 'combined_score' in df.columns else None,
        'latest_signal': df['signal'].iloc[-1] if 'signal' in df.columns else None,
        'timestamp': datetime.now().isoformat()
    }
    
    # Save report
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_file = OUTPUT_DIR / f"validation_report_signals_spy_{datetime.now().strftime('%Y%m%d')}.json"
    with open(report_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\nStatus: {result['status']}")
    print(f"Total Signals: {result['total_signals']}")
    print(f"Latest Score: {result['latest_score']:.1f}" if result['latest_score'] else "N/A")
    print(f"Latest Signal: {result['latest_signal']}")
    
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for e in errors:
            print(f"  ✗ {e}")
    else:
        print("✓ All validations passed")
    
    print(f"\nReport saved: {report_file}")
    return len(errors) == 0


if __name__ == "__main__":
    success = validate_signals()
    exit(0 if success else 1)
