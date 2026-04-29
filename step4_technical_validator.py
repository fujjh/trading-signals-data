#!/usr/bin/env python3
"""
================================================================================
STEP 4: SPY Technical Validator - Step 4
================================================================================

Validates SPY technical analysis output from Step 3.

Author: SignalsAlpha
Version: 1.0 (SPY Edition)
Date: 2026-04-29
================================================================================
"""

import pandas as pd
from pathlib import Path
import json
from datetime import datetime

TICKER = "SPY"
INTERVAL = "1d"
TECHNICAL_FILE = Path("data/technical_analysis") / TICKER / f"{TICKER}_{INTERVAL}_technical.csv"
OUTPUT_DIR = Path("data/validated")


def validate_technical():
    """Validate SPY technical analysis data."""
    print("=" * 70)
    print("STEP 4: SPY Technical Validation")
    print("=" * 70)
    
    if not TECHNICAL_FILE.exists():
        print(f"ERROR: Technical file not found: {TECHNICAL_FILE}")
        return False
    
    df = pd.read_csv(TECHNICAL_FILE)
    
    errors = []
    
    # Check required columns
    required = ['date', 'close', 'sma_20', 'sma_50', 'rsi', 'macd', 'hma_13', 'elder_impulse']
    for col in required:
        if col not in df.columns:
            errors.append(f"Missing column: {col}")
    
    # Check for NaN in recent data
    recent = df.tail(20)
    for col in ['rsi', 'macd', 'sma_20']:
        if col in df.columns and recent[col].isnull().all():
            errors.append(f"Recent {col} values are NaN")
    
    # Build report
    result = {
        'ticker': TICKER,
        'interval': INTERVAL,
        'status': 'PASS' if not errors else 'FAIL',
        'errors': errors,
        'row_count': len(df),
        'columns': len(df.columns),
        'timestamp': datetime.now().isoformat()
    }
    
    # Save report
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_file = OUTPUT_DIR / f"validation_report_technical_spy_{datetime.now().strftime('%Y%m%d')}.json"
    with open(report_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"Status: {result['status']}")
    print(f"Rows: {result['row_count']}")
    print(f"Columns: {result['columns']}")
    
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for e in errors:
            print(f"  ✗ {e}")
    else:
        print("✓ All validations passed")
    
    print(f"\nReport saved: {report_file}")
    return len(errors) == 0


if __name__ == "__main__":
    success = validate_technical()
    exit(0 if success else 1)
