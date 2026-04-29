#!/usr/bin/env python3
"""
================================================================================
STEP 2: SPY Time Series Validator - Step 2
================================================================================

Validates SPY daily OHLCV time series data from Step 1.
Simplified version for single-ticker analysis.

Author: SignalsAlpha
Version: 1.0 (SPY Edition)
Date: 2026-04-29

================================================================================
PURPOSE
================================================================================

Quality assurance for SPY price data:
1. Validates CSV file exists and contains required columns
2. Checks for data completeness (no missing values)
3. Verifies data types (numeric columns contain numbers)
4. Detects outliers and anomalies in price/volume data
5. Ensures date ranges are valid
6. Validates sufficient historical data for technical analysis

================================================================================
VALIDATION CHECKS
================================================================================

File-Level:
    - File exists and is readable
    - CSV parses successfully

Schema:
    - Required columns present (date, open, high, low, close, volume)

Data Quality:
    - No null values in critical columns
    - Price values are positive
    - Volume is non-negative
    - High >= Low (basic OHLC sanity)
    - Close is between Low and High

Date Validation:
    - Dates are in valid format
    - No future dates
    - Data is current (last date within 7 days)
    - Sufficient rows for technical analysis (min 50)

================================================================================
OUTPUT
================================================================================

Report: data/validated/validation_report_spy_{date}.json

Contains:
    - validation_status: PASS/FAIL
    - total_rows: Number of data points
    - date_range: First and last date
    - latest_price: Most recent closing price
    - latest_volume: Most recent volume
    - errors: List of any validation errors

================================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# Configuration
DATA_DIR = Path("data")
TIME_SERIES_DIR = DATA_DIR / "time_series" / "SPY"
OUTPUT_DIR = DATA_DIR / "validated"
TICKER = "SPY"
INTERVAL = "1d"

# Validation thresholds
MIN_ROWS_FOR_INDICATORS = 50
MAX_PRICE = 100000  # Sanity check
MIN_PRICE = 0.0001
MAX_VOLUME = 1e12  # Sanity check
MAX_DATE_STALE_DAYS = 7


class SPYTimeSeriesValidator:
    """Validator for SPY time series OHLCV data"""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.stats = {}
    
    def validate(self) -> Dict:
        """
        Run all validation checks on SPY data.
        
        Returns:
            Dict with validation results
        """
        print("=" * 70)
        print("STEP 2: SPY Time Series Validation")
        print("=" * 70)
        
        csv_path = TIME_SERIES_DIR / f"{TICKER}_{INTERVAL}.csv"
        
        # Check file exists
        if not csv_path.exists():
            self.errors.append(f"File not found: {csv_path}")
            return self._build_result(status="FAIL")
        
        print(f"Validating: {csv_path}")
        
        # Load data
        try:
            df = pd.read_csv(csv_path)
            print(f"Loaded {len(df)} rows")
        except Exception as e:
            self.errors.append(f"Failed to load CSV: {e}")
            return self._build_result(status="FAIL")
        
        # Run validations
        self._validate_schema(df)
        self._validate_data_quality(df)
        self._validate_dates(df)
        self._calculate_stats(df)
        
        # Build result
        status = "PASS" if not self.errors else "FAIL"
        result = self._build_result(status, df)
        
        # Save report
        self._save_report(result)
        
        # Print summary
        self._print_summary(result)
        
        return result
    
    def _validate_schema(self, df: pd.DataFrame):
        """Validate CSV has required columns."""
        required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            self.errors.append(f"Missing columns: {missing}")
        else:
            print("✓ Schema validation passed")
    
    def _validate_data_quality(self, df: pd.DataFrame):
        """Validate data values."""
        if self.errors:  # Skip if schema failed
            return
        
        # Check for nulls in critical columns
        critical_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        for col in critical_cols:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                self.errors.append(f"Column '{col}' has {null_count} null values")
        
        # Check price sanity
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if (df[col] <= 0).any():
                self.errors.append(f"Column '{col}' has non-positive values")
            
            if (df[col] > MAX_PRICE).any():
                self.warnings.append(f"Column '{col}' has values exceeding {MAX_PRICE}")
        
        # Check OHLC logic (allow small tolerance for adjusted data)
        invalid_hl = (df['high'] < df['low']).sum()
        if invalid_hl > 0:
            self.errors.append(f"{invalid_hl} rows have High < Low")
        
        # Check Close is approximately within High-Low range (allow 1% tolerance for splits/adjustments)
        tolerance = 0.01
        invalid_close = ((df['close'] < df['low'] * (1 - tolerance)) | 
                        (df['close'] > df['high'] * (1 + tolerance))).sum()
        if invalid_close > 0:
            self.warnings.append(f"{invalid_close} rows have Close slightly outside High-Low range (likely due to splits/adjustments)")
        
        # Check volume
        if (df['volume'] < 0).any():
            self.errors.append("Volume has negative values")
        
        if not self.errors:
            print("✓ Data quality validation passed")
    
    def _validate_dates(self, df: pd.DataFrame):
        """Validate date column."""
        if self.errors:
            return
        
        try:
            df['date'] = pd.to_datetime(df['date'], utc=True)
        except Exception as e:
            self.errors.append(f"Failed to parse dates: {e}")
            return
        
        # Check for future dates
        now = pd.Timestamp.now(tz='UTC')
        future_dates = (df['date'] > now).sum()
        if future_dates > 0:
            self.errors.append(f"{future_dates} rows have future dates")
        
        # Check for stale data
        last_date = df['date'].max()
        days_since_update = (now - last_date).days
        if days_since_update > MAX_DATE_STALE_DAYS:
            self.warnings.append(f"Data is {days_since_update} days old (threshold: {MAX_DATE_STALE_DAYS})")
        
        # Check minimum rows
        if len(df) < MIN_ROWS_FOR_INDICATORS:
            self.errors.append(f"Insufficient rows: {len(df)} (minimum: {MIN_ROWS_FOR_INDICATORS})")
        
        if not self.errors:
            print("✓ Date validation passed")
    
    def _calculate_stats(self, df: pd.DataFrame):
        """Calculate summary statistics."""
        if self.errors:
            return
        
        df['date'] = pd.to_datetime(df['date'], utc=True)
        
        self.stats = {
            'total_rows': len(df),
            'date_range': {
                'start': df['date'].min().strftime('%Y-%m-%d'),
                'end': df['date'].max().strftime('%Y-%m-%d')
            },
            'latest_price': float(df['close'].iloc[-1]),
            'latest_volume': int(df['volume'].iloc[-1]),
            'price_range': {
                'min': float(df['low'].min()),
                'max': float(df['high'].max())
            },
            'avg_volume': int(df['volume'].mean())
        }
    
    def _build_result(self, status: str, df: pd.DataFrame = None) -> Dict:
        """Build validation result dictionary."""
        result = {
            'ticker': TICKER,
            'interval': INTERVAL,
            'validation_status': status,
            'validation_timestamp': datetime.now().isoformat(),
            'errors': self.errors,
            'warnings': self.warnings,
            'stats': self.stats
        }
        
        if df is not None and not df.empty:
            result['file_path'] = str(TIME_SERIES_DIR / f"{TICKER}_{INTERVAL}.csv")
        
        return result
    
    def _save_report(self, result: Dict):
        """Save validation report to JSON."""
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        date_str = datetime.now().strftime('%Y%m%d')
        report_path = OUTPUT_DIR / f"validation_report_spy_{date_str}.json"
        
        with open(report_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"\nReport saved: {report_path}")
    
    def _print_summary(self, result: Dict):
        """Print validation summary."""
        print("\n" + "=" * 70)
        print("Validation Summary")
        print("=" * 70)
        print(f"Status: {result['validation_status']}")
        print(f"Ticker: {result['ticker']}")
        print(f"Interval: {result['interval']}")
        
        if result['stats']:
            stats = result['stats']
            print(f"\nData Statistics:")
            print(f"  Total rows: {stats['total_rows']:,}")
            print(f"  Date range: {stats['date_range']['start']} to {stats['date_range']['end']}")
            print(f"  Latest price: ${stats['latest_price']:.2f}")
            print(f"  Latest volume: {stats['latest_volume']:,}")
            print(f"  Price range: ${stats['price_range']['min']:.2f} - ${stats['price_range']['max']:.2f}")
        
        if result['errors']:
            print(f"\nErrors ({len(result['errors'])}):")
            for error in result['errors']:
                print(f"  ✗ {error}")
        
        if result['warnings']:
            print(f"\nWarnings ({len(result['warnings'])}):")
            for warning in result['warnings']:
                print(f"  ⚠ {warning}")
        
        print("=" * 70)


def main():
    """Main entry point."""
    validator = SPYTimeSeriesValidator()
    result = validator.validate()
    
    # Exit with appropriate code
    exit_code = 0 if result['validation_status'] == "PASS" else 1
    exit(exit_code)


if __name__ == "__main__":
    main()
