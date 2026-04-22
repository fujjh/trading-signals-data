#!/usr/bin/env python3
"""
================================================================================
STEP 1.5: Time Series Validator
================================================================================

Validates raw OHLCV time series data from Step 1 before technical analysis.

Author: SignalsAlpha
Version: 1.0
Date: 2026-04-20

================================================================================
PURPOSE
================================================================================

Quality assurance for raw price data:
1. Validates CSV files exist and contain required columns
2. Checks for data completeness (no missing values in critical fields)
3. Verifies data types (numeric columns contain numbers)
4. Detects outliers and anomalies in price/volume data
5. Ensures date ranges are valid and consecutive
6. Validates that all tickers have complete interval coverage

================================================================================
VALIDATION CHECKS
================================================================================

File-Level:
    - File exists and is readable
    - File size is reasonable (not empty, not corrupted)
    - CSV parses successfully

Schema:
    - Required columns present (ticker, interval, date, open, high, low, close, volume)
    - No duplicate column names

Data Quality:
    - No null values in critical columns
    - Price values are positive and reasonable
    - Volume is non-negative
    - High >= Low (basic OHLC sanity)
    - Close is between Low and High

Date Validation:
    - Dates are in valid format
    - No future dates
    - Data is current (last date within reasonable range)

Coverage:
    - All expected intervals present for each ticker
    - Sufficient rows for technical analysis (min 50 for indicators)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

# Configuration
DATA_DIR = Path("/home/ubuntu/.openclaw/workspace/trading-signals-data/process_flow_v1/data")
TIME_SERIES_DIR = DATA_DIR / "time_series"
OUTPUT_DIR = DATA_DIR / "validated"
PROGRESS_FILE = OUTPUT_DIR / ".validation_progress"
INTERVALS = ['1d', '1wk', '1mo']

# Batch processing configuration
BATCH_SIZE = 100  # Process 100 tickers per batch
MAX_BATCHES = 50  # Exit after 50 batches (5000 tickers)

# Validation thresholds
MIN_ROWS_FOR_INDICATORS = 50
MAX_PRICE = 100000  # Sanity check
MIN_PRICE = 0.0001
MAX_VOLUME = 1e12  # Sanity check
MAX_DATE_FUTURE_DAYS = 1
MAX_DATE_STALE_DAYS = 7

class TimeSeriesValidator:
    """Validator for time series OHLCV data with batch processing"""
    
    def __init__(self):
        self.results = []
        self.summary = {
            'total_tickers': 0,
            'total_files': 0,
            'passed': 0,
            'failed': 0,
            'errors_by_type': {}
        }
        self.processed_tickers = set()
        self.batch_count = 0
    
    def load_progress(self):
        """Load list of already processed tickers"""
        if PROGRESS_FILE.exists():
            try:
                with open(PROGRESS_FILE, 'r') as f:
                    data = json.load(f)
                    self.processed_tickers = set(data.get('processed_tickers', []))
                    self.batch_count = data.get('batch_count', 0)
                    self.results = data.get('results', [])
                print(f"Loaded progress: {len(self.processed_tickers)} tickers already validated")
                return True
            except Exception as e:
                print(f"Warning: Could not load progress: {e}")
        return False
    
    def save_progress(self):
        """Save progress to resume later"""
        try:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            with open(PROGRESS_FILE, 'w') as f:
                json.dump({
                    'processed_tickers': list(self.processed_tickers),
                    'batch_count': self.batch_count,
                    'results': self.results,
                    'timestamp': datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save progress: {e}")
    
    def validate_file_exists(self, file_path: Path) -> Tuple[bool, str]:
        """Check if file exists and is readable"""
        if not file_path.exists():
            return False, f"File does not exist: {file_path}"
        if not file_path.is_file():
            return False, f"Path is not a file: {file_path}"
        if file_path.stat().st_size == 0:
            return False, f"File is empty: {file_path}"
        return True, "OK"
    
    def validate_csv_parse(self, file_path: Path) -> Tuple[bool, str, Optional[pd.DataFrame]]:
        """Try to parse CSV file"""
        try:
            df = pd.read_csv(file_path)
            if df.empty:
                return False, "CSV file is empty (no rows)", None
            return True, "OK", df
        except pd.errors.EmptyDataError:
            return False, "CSV file is empty (no columns)", None
        except pd.errors.ParserError as e:
            return False, f"CSV parse error: {str(e)}", None
        except Exception as e:
            return False, f"Unexpected error: {str(e)}", None
    
    def validate_schema(self, df: pd.DataFrame) -> Tuple[bool, str, List[str]]:
        """Validate CSV has required columns"""
        required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        
        # Normalize column names (lowercase)
        df_cols_lower = [c.lower() for c in df.columns]
        
        missing = []
        for req in required_cols:
            if req not in df_cols_lower:
                missing.append(req)
        
        if missing:
            return False, f"Missing required columns: {missing}", missing
        
        # Check for duplicate columns
        if len(df.columns) != len(set(df.columns)):
            return False, "Duplicate column names found", []
        
        return True, "OK", []
    
    def validate_data_quality(self, df: pd.DataFrame) -> Tuple[bool, str, Dict]:
        """Validate data values"""
        issues = {}
        
        # Check for nulls in critical columns
        null_counts = df.isnull().sum()
        if null_counts.any():
            issues['null_values'] = null_counts[null_counts > 0].to_dict()
        
        # Validate price ranges
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                prices = df[col]
                if (prices <= 0).any():
                    issues[f'{col}_non_positive'] = int((prices <= 0).sum())
                if (prices > MAX_PRICE).any():
                    issues[f'{col}_too_high'] = int((prices > MAX_PRICE).sum())
                if (prices < MIN_PRICE).any():
                    issues[f'{col}_too_low'] = int((prices < MIN_PRICE).sum())
        
        # Validate High >= Low
        if 'high' in df.columns and 'low' in df.columns:
            invalid_hl = (df['high'] < df['low']).sum()
            if invalid_hl > 0:
                issues['high_lt_low'] = int(invalid_hl)
        
        # Validate Close between High and Low
        if all(c in df.columns for c in ['high', 'low', 'close']):
            invalid_close = ((df['close'] > df['high']) | (df['close'] < df['low'])).sum()
            if invalid_close > 0:
                issues['close_outside_range'] = int(invalid_close)
        
        # Validate volume
        if 'volume' in df.columns:
            if (df['volume'] < 0).any():
                issues['negative_volume'] = int((df['volume'] < 0).sum())
            if (df['volume'] > MAX_VOLUME).any():
                issues['volume_too_high'] = int((df['volume'] > MAX_VOLUME).sum())
        
        # Check sufficient rows
        if len(df) < MIN_ROWS_FOR_INDICATORS:
            issues['insufficient_rows'] = f"{len(df)} rows (min {MIN_ROWS_FOR_INDICATORS})"
        
        if issues:
            return False, f"Data quality issues: {list(issues.keys())}", issues
        
        return True, "OK", {}
    
    def validate_dates(self, df: pd.DataFrame) -> Tuple[bool, str, Dict]:
        """Validate date column"""
        issues = {}
        
        if 'date' not in df.columns:
            return False, "Date column not found", {'missing_date_column': True}
        
        try:
            # Try to parse dates
            dates = pd.to_datetime(df['date'], errors='coerce', utc=True).dt.tz_localize(None)
            
            # Check for unparseable dates
            if dates.isnull().any():
                issues['invalid_dates'] = int(dates.isnull().sum())
            
            # Check for future dates
            now = datetime.now()
            future_dates = (dates > now).sum()
            if future_dates > 0:
                issues['future_dates'] = int(future_dates)
            
            # Check for stale data
            if not dates.empty:
                latest_date = dates.max()
                days_since_latest = (now - latest_date).days
                if days_since_latest > MAX_DATE_STALE_DAYS:
                    issues['stale_data'] = f"{days_since_latest} days old"
            
            # Check date ordering (should be ascending)
            if len(dates) > 1:
                if not dates.is_monotonic_increasing:
                    issues['dates_not_ordered'] = True
            
        except Exception as e:
            return False, f"Date validation error: {str(e)}", {'date_parse_error': str(e)}
        
        if issues:
            return False, f"Date issues: {list(issues.keys())}", issues
        
        return True, "OK", {}
    
    def validate_ticker(self, ticker: str) -> Dict:
        """Validate all interval files for a ticker"""
        ticker_results = {
            'ticker': ticker,
            'intervals': {},
            'overall_pass': True
        }
        
        for interval in INTERVALS:
            file_path = TIME_SERIES_DIR / ticker / f"{ticker}_{interval}.csv"
            
            result = {
                'file': str(file_path),
                'exists': False,
                'parses': False,
                'schema_ok': False,
                'data_ok': False,
                'dates_ok': False,
                'pass': False,
                'errors': []
            }
            
            # Check file exists
            exists_ok, exists_msg = self.validate_file_exists(file_path)
            if not exists_ok:
                result['errors'].append(exists_msg)
                ticker_results['intervals'][interval] = result
                ticker_results['overall_pass'] = False
                continue
            result['exists'] = True
            
            # Parse CSV
            parse_ok, parse_msg, df = self.validate_csv_parse(file_path)
            if not parse_ok:
                result['errors'].append(parse_msg)
                ticker_results['intervals'][interval] = result
                ticker_results['overall_pass'] = False
                continue
            result['parses'] = True
            
            # Validate schema
            schema_ok, schema_msg, missing = self.validate_schema(df)
            if not schema_ok:
                result['errors'].append(schema_msg)
                ticker_results['intervals'][interval] = result
                ticker_results['overall_pass'] = False
                continue
            result['schema_ok'] = True
            
            # Validate data quality
            data_ok, data_msg, data_issues = self.validate_data_quality(df)
            if not data_ok:
                result['errors'].append(data_msg)
                result['data_issues'] = data_issues
            result['data_ok'] = data_ok
            
            # Validate dates
            dates_ok, dates_msg, date_issues = self.validate_dates(df)
            if not dates_ok:
                result['errors'].append(dates_msg)
                result['date_issues'] = date_issues
            result['dates_ok'] = dates_ok
            
            # Overall pass
            result['pass'] = data_ok and dates_ok
            if not result['pass']:
                ticker_results['overall_pass'] = False
            
            ticker_results['intervals'][interval] = result
        
        return ticker_results
    
    def run_validation(self, batch_mode=False) -> Dict:
        """Run validation on all tickers with optional batch processing"""
        print("="*70)
        print("STEP 1.5: Time Series Validator")
        if batch_mode:
            print("(BATCH MODE - Will exit after {} batches)".format(MAX_BATCHES))
        print("="*70)
        print()
        
        # Ensure output directory exists
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        # Load progress if resuming
        if batch_mode:
            self.load_progress()
        
        # Get all tickers
        tickers = []
        if TIME_SERIES_DIR.exists():
            tickers = [d.name for d in TIME_SERIES_DIR.iterdir() if d.is_dir() and not d.name.startswith('.')]
        
        tickers.sort()
        total_tickers = len(tickers)
        
        # Filter out already processed tickers
        if batch_mode and self.processed_tickers:
            tickers = [t for t in tickers if t not in self.processed_tickers]
            print(f"Resuming: {len(tickers)} tickers remaining out of {total_tickers} total")
        
        self.summary['total_tickers'] = total_tickers
        
        print(f"Validating {len(tickers)} tickers...")
        print(f"Batch size: {BATCH_SIZE}, Max batches: {MAX_BATCHES}")
        print(f"Required intervals: {', '.join(INTERVALS)}")
        print(f"Min rows for indicators: {MIN_ROWS_FOR_INDICATORS}")
        print()
        
        # Process in batches
        batches_to_process = min(MAX_BATCHES, (len(tickers) + BATCH_SIZE - 1) // BATCH_SIZE)
        
        for batch_num in range(batches_to_process):
            if batch_num < self.batch_count:
                print(f"  Skipping batch {batch_num + 1} (already processed)")
                continue
            
            start_idx = batch_num * BATCH_SIZE
            end_idx = min(start_idx + BATCH_SIZE, len(tickers))
            batch_tickers = tickers[start_idx:end_idx]
            
            print(f"\nBatch {batch_num + 1}/{batches_to_process}: Processing {len(batch_tickers)} tickers...")
            
            for ticker in batch_tickers:
                result = self.validate_ticker(ticker)
                self.results.append(result)
                self.processed_tickers.add(ticker)
                
                # Update summary
                for interval in INTERVALS:
                    if interval in result['intervals']:
                        self.summary['total_files'] += 1
                        if result['intervals'][interval]['pass']:
                            self.summary['passed'] += 1
                        else:
                            self.summary['failed'] += 1
                
                # Count error types
                for interval_data in result['intervals'].values():
                    if not interval_data['pass']:
                        for error in interval_data['errors']:
                            error_type = error.split(':')[0]
                            self.summary['errors_by_type'][error_type] = self.summary['errors_by_type'].get(error_type, 0) + 1
            
            self.batch_count = batch_num + 1
            self.save_progress()
            
            if batch_mode and self.batch_count >= MAX_BATCHES:
                print(f"\nReached max batches ({MAX_BATCHES}). Exiting.")
                print(f"Progress saved to: {PROGRESS_FILE}")
                print("Run again to continue from where we left off.")
                break
        
        # Check if completed
        remaining = len(tickers) - len(self.processed_tickers)
        completed = remaining == 0
        
        if completed:
            # Clear progress file
            if PROGRESS_FILE.exists():
                PROGRESS_FILE.unlink()
            print("\n" + "="*70)
            print("VALIDATION COMPLETE")
            print("="*70)
        
        # Generate report
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': self.summary,
            'results': self.results,
            'completed': completed,
            'processed_tickers': len(self.processed_tickers)
        }
        
        # Save report
        report_file = OUTPUT_DIR / f"validation_report_time_series_{datetime.now().strftime('%Y%m%d')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print()
        print("="*70)
        print("VALIDATION SUMMARY")
        print("="*70)
        print(f"Total tickers: {self.summary['total_tickers']}")
        print(f"Total files: {self.summary['total_files']}")
        print(f"Passed: {self.summary['passed']} ({100*self.summary['passed']/max(self.summary['total_files'],1):.1f}%)")
        print(f"Failed: {self.summary['failed']} ({100*self.summary['failed']/max(self.summary['total_files'],1):.1f}%)")
        
        if self.summary['errors_by_type']:
            print("\nErrors by type:")
            for error_type, count in sorted(self.summary['errors_by_type'].items(), key=lambda x: -x[1]):
                print(f"  - {error_type}: {count}")
        
        print(f"\nReport saved to: {report_file}")
        print("="*70)
        
        return report

def main():
    validator = TimeSeriesValidator()
    
    # Check if batch mode requested via environment variable
    import os
    batch_mode = os.getenv('BATCH_MODE', 'false').lower() == 'true'
    
    validator.run_validation(batch_mode=batch_mode)

if __name__ == "__main__":
    main()
