#!/usr/bin/env python3
"""
================================================================================
STEP 3: Data Validator
================================================================================
Validates data integrity across all pipeline outputs before website generation
================================================================================
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

# Configuration
DATA_DIR = Path("/home/ubuntu/.openclaw/workspace/data")
TIME_SERIES_DIR = DATA_DIR / "time_series"
SIGNALS_DIR = DATA_DIR / "signals_timeframe"
OUTPUT_DIR = DATA_DIR / "validated"

# Validation thresholds
MIN_SIGNALS_PER_TICKER = 1
MAX_SIGNALS_PER_TICKER = 3  # One per timeframe
MAX_CONFIDENCE = 95
MIN_CONFIDENCE = 50


def validate_time_series():
    """Validate time series data integrity"""
    print("Validating Time Series Data...")
    print("-" * 60)
    
    errors = []
    warnings = []
    stats = {"tickers": 0, "files": 0, "intervals": {}}
    
    if not TIME_SERIES_DIR.exists():
        errors.append(f"Time series directory not found: {TIME_SERIES_DIR}")
        return False, errors, warnings, stats
    
    for ticker_dir in TIME_SERIES_DIR.iterdir():
        if not ticker_dir.is_dir():
            continue
        
        stats["tickers"] += 1
        ticker = ticker_dir.name
        
        # Check for expected intervals
        expected_intervals = ['1d', '1wk', '1mo']
        for interval in expected_intervals:
            file_path = ticker_dir / f"{ticker}_{interval}.csv"
            if file_path.exists():
                stats["files"] += 1
                stats["intervals"][interval] = stats["intervals"].get(interval, 0) + 1
                
                # Validate CSV structure
                try:
                    df = pd.read_csv(file_path)
                    if df.empty:
                        warnings.append(f"{ticker} {interval}: Empty file")
                    elif 'Close' not in df.columns and 'close' not in df.columns:
                        errors.append(f"{ticker} {interval}: Missing Close column")
                except Exception as e:
                    errors.append(f"{ticker} {interval}: Parse error - {e}")
            else:
                warnings.append(f"{ticker}: Missing {interval} data")
    
    print(f"✓ Time Series: {stats['tickers']} tickers, {stats['files']} files")
    print(f"  Intervals: {stats['intervals']}")
    
    return len(errors) == 0, errors, warnings, stats


def validate_signals():
    """Validate signal data integrity"""
    print("\nValidating Signal Data...")
    print("-" * 60)
    
    errors = []
    warnings = []
    stats = {"tickers": 0, "signals": 0, "by_signal": {}}
    
    if not SIGNALS_DIR.exists():
        errors.append(f"Signals directory not found: {SIGNALS_DIR}")
        return False, errors, warnings, stats
    
    for ticker_dir in SIGNALS_DIR.iterdir():
        if not ticker_dir.is_dir():
            continue
        
        ticker = ticker_dir.name
        signals_file = ticker_dir / f"{ticker}_signals.csv"
        
        if not signals_file.exists():
            warnings.append(f"{ticker}: No signals file")
            continue
        
        try:
            df = pd.read_csv(signals_file)
            stats["tickers"] += 1
            stats["signals"] += len(df)
            
            # Validate required columns
            required_cols = ['ticker', 'signal', 'confidence', 'close', 'timestamp']
            missing_cols = [c for c in required_cols if c not in df.columns]
            if missing_cols:
                errors.append(f"{ticker}: Missing columns {missing_cols}")
                continue
            
            # Validate signal values
            valid_signals = ['STRONG_BUY', 'BUY', 'WEAK_BUY', 'HOLD', 
                           'WEAK_SELL', 'SELL', 'STRONG_SELL']
            invalid_signals = df[~df['signal'].isin(valid_signals)]
            if not invalid_signals.empty:
                errors.append(f"{ticker}: Invalid signal values found")
            
            # Validate confidence range
            out_of_range = df[(df['confidence'] < MIN_CONFIDENCE) | 
                             (df['confidence'] > MAX_CONFIDENCE)]
            if not out_of_range.empty:
                warnings.append(f"{ticker}: Confidence out of range")
            
            # Count by signal type
            for sig in df['signal']:
                stats["by_signal"][sig] = stats["by_signal"].get(sig, 0) + 1
            
        except Exception as e:
            errors.append(f"{ticker}: Error reading signals - {e}")
    
    print(f"✓ Signals: {stats['tickers']} tickers, {stats['signals']} signals")
    print(f"  Distribution: {stats['by_signal']}")
    
    return len(errors) == 0, errors, warnings, stats


def validate_freshness(max_age_hours=26):
    """Validate data is fresh (not stale)"""
    print("\nValidating Data Freshness...")
    print("-" * 60)
    
    errors = []
    warnings = []
    
    # Check latest signal timestamp
    latest_timestamp = None
    signal_count = 0
    
    for ticker_dir in SIGNALS_DIR.iterdir():
        if not ticker_dir.is_dir():
            continue
        
        ticker = ticker_dir.name
        signals_file = ticker_dir / f"{ticker}_signals.csv"
        
        if signals_file.exists():
            try:
                df = pd.read_csv(signals_file)
                if 'timestamp' in df.columns and not df.empty:
                    ts = pd.to_datetime(df['timestamp'].iloc[-1])
                    if latest_timestamp is None or ts > latest_timestamp:
                        latest_timestamp = ts
                    signal_count += 1
            except:
                pass
    
    if latest_timestamp:
        age_hours = (datetime.now() - latest_timestamp).total_seconds() / 3600
        print(f"  Latest signal timestamp: {latest_timestamp}")
        print(f"  Data age: {age_hours:.1f} hours")
        
        if age_hours > max_age_hours:
            warnings.append(f"Data is stale ({age_hours:.1f}h old, max {max_age_hours}h)")
        else:
            print(f"  ✓ Data is fresh")
    else:
        warnings.append("Could not determine data freshness")
    
    return len(errors) == 0, errors, warnings, {"latest": latest_timestamp}


def generate_validation_report(ts_valid, ts_errors, ts_warnings, ts_stats,
                               sig_valid, sig_errors, sig_warnings, sig_stats,
                               fresh_valid, fresh_errors, fresh_warnings, fresh_stats):
    """Generate comprehensive validation report"""
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_file = OUTPUT_DIR / f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "overall_status": "PASS" if all([ts_valid, sig_valid, fresh_valid]) else "FAIL",
        "time_series": {
            "valid": ts_valid,
            "tickers": ts_stats.get("tickers", 0),
            "files": ts_stats.get("files", 0),
            "intervals": ts_stats.get("intervals", {}),
            "errors": ts_errors,
            "warnings": ts_warnings
        },
        "signals": {
            "valid": sig_valid,
            "tickers": sig_stats.get("tickers", 0),
            "total_signals": sig_stats.get("signals", 0),
            "distribution": sig_stats.get("by_signal", {}),
            "errors": sig_errors,
            "warnings": sig_warnings
        },
        "freshness": {
            "valid": fresh_valid,
            "latest_timestamp": str(fresh_stats.get("latest")) if fresh_stats.get("latest") else None,
            "errors": fresh_errors,
            "warnings": fresh_warnings
        }
    }
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    return report_file


def main():
    """Main validation execution"""
    print("=" * 60)
    print("SignalsAlpha Data Validator")
    print("=" * 60)
    print()
    
    # Validate time series
    ts_valid, ts_errors, ts_warnings, ts_stats = validate_time_series()
    
    # Validate signals
    sig_valid, sig_errors, sig_warnings, sig_stats = validate_signals()
    
    # Validate freshness
    fresh_valid, fresh_errors, fresh_warnings, fresh_stats = validate_freshness()
    
    # Generate report
    report_file = generate_validation_report(
        ts_valid, ts_errors, ts_warnings, ts_stats,
        sig_valid, sig_errors, sig_warnings, sig_stats,
        fresh_valid, fresh_errors, fresh_warnings, fresh_stats
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Time Series: {'✓ PASS' if ts_valid else '✗ FAIL'}")
    print(f"Signals:     {'✓ PASS' if sig_valid else '✗ FAIL'}")
    print(f"Freshness:   {'✓ PASS' if fresh_valid else '✗ FAIL'}")
    print()
    
    total_errors = len(ts_errors) + len(sig_errors) + len(fresh_errors)
    total_warnings = len(ts_warnings) + len(sig_warnings) + len(fresh_warnings)
    
    if total_errors > 0:
        print(f"ERRORS: {total_errors}")
        for e in ts_errors + sig_errors + fresh_errors:
            print(f"  ✗ {e}")
    
    if total_warnings > 0:
        print(f"WARNINGS: {total_warnings}")
        for w in ts_warnings + sig_warnings + fresh_warnings:
            print(f"  ⚠ {w}")
    
    print(f"\nReport saved to: {report_file}")
    
    # Exit with appropriate code
    if ts_valid and sig_valid:
        print("\n✓ Validation PASSED - Ready for website output")
        return 0
    else:
        print("\n✗ Validation FAILED - Please fix errors before proceeding")
        return 1


if __name__ == "__main__":
    sys.exit(main())
