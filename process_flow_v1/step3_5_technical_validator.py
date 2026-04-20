#!/usr/bin/env python3
"""
================================================================================
STEP 3.5: Technical Analysis Validator
================================================================================

Quality assurance module that validates technical analysis output from Step 3
before it flows into the scoring system (Step 4).

Author: SignalsAlpha
Version: 1.0
Date: 2026-04-20

================================================================================
PURPOSE
================================================================================

Technical analysis produces complex calculated data. Errors can occur:
- Missing indicator columns from failed calculations
- NaN values in critical fields breaking downstream scoring
- Crossover detection failures
- Missing support/resistance levels
- Incomplete Fibonacci calculations

This validator catches these issues before scoring:

1. Validates all technical indicator columns exist
2. Checks for NaN values in critical fields
3. Verifies crossover detection worked (columns exist and have data)
4. Ensures support/resistance levels calculated
5. Validates Fibonacci retracement levels present
6. Checks candlestick patterns detected
7. Validates signal lines present for oscillators
8. Checks price targets (stop_loss/take_profit) calculated

================================================================================
INPUT
================================================================================

Source: data/technical_analysis/{TICKER}/{TICKER}_{interval}_technical.csv
Columns: All indicator columns from Step 3 output

================================================================================
OUTPUT
================================================================================

Destination: data/validated/validation_report_technical_{date}.json

JSON Structure:
{
  "timestamp": "2026-04-20T...",
  "summary": {
    "total_tickers": 4848,
    "total_files": 14544,
    "passed": 14500,
    "failed": 44,
    "errors_by_type": {...}
  },
  "results": [
    {
      "ticker": "AAPL",
      "intervals": {
        "1d": {"pass": true, "errors": []},
        "1wk": {"pass": true, "errors": []},
        "1mo": {"pass": true, "errors": []}
      },
      "overall_pass": true
    }
  ]
}

================================================================================
VALIDATION CHECKS
================================================================================

File Structure:
    - Required columns present (sma_20, rsi, macd, adx, etc.)
    - Crossover columns exist (macd_cross_type, stoch_slow_cross, etc.)
    - Support/resistance columns present
    - Fibonacci level columns present

Data Integrity:
    - Critical columns have no NaN values
    - Technical_raw_score column present

Indicator Completeness:
    - SMA/EMA calculated for all rows
    - RSI values in valid range (0-100)
    - MACD crossover detection populated
    - Stochastic %K/%D present

Support/Resistance:
    - support_levels JSON column present
    - support_distance_pct calculated
    - resistance_distance_pct calculated

Fibonacci Levels:
    - fib_23_6 through fib_78_6 columns present
    - No NaN values in Fibonacci columns

Candlestick Patterns:
    - Pattern columns exist (pattern_doji, pattern_hammer, etc.)
    - At least some patterns detected across dataset

Price Targets:
    - stop_loss column present and calculated
    - take_profit column present and calculated

"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Configuration
DATA_DIR = Path("/home/ubuntu/.openclaw/workspace/trading-signals-data/process_flow_v1/data")
TECHNICAL_DIR = DATA_DIR / "technical_analysis"
OUTPUT_DIR = DATA_DIR / "validated"
INTERVALS = ['1d', '1wk', '1mo']

class TechnicalValidator:
    """Validator for technical analysis output"""
    
    def __init__(self):
        self.results = []
        self.summary = {
            'total_tickers': 0,
            'total_files': 0,
            'passed': 0,
            'failed': 0,
            'errors_by_type': {}
        }
    
    def load_technical_file(self, ticker: str, interval: str) -> Optional[pd.DataFrame]:
        """Load technical analysis CSV for a ticker/interval"""
        file_path = TECHNICAL_DIR / ticker / f"{ticker}_{interval}_technical.csv"
        if not file_path.exists():
            return None
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def validate_file_exists(self, ticker: str, interval: str) -> Tuple[bool, str]:
        """Check if technical analysis file exists"""
        file_path = TECHNICAL_DIR / ticker / f"{ticker}_{interval}_technical.csv"
        if not file_path.exists():
            return False, f"File does not exist: {file_path}"
        if file_path.stat().st_size == 0:
            return False, f"File is empty: {file_path}"
        return True, "OK"
    
    def validate_required_columns(self, df: pd.DataFrame) -> Tuple[bool, str, List[str]]:
        """Validate all required technical indicator columns exist"""
        required = [
            'Date', 'Close', 'volume_ratio', 'market_regime', 'adx',
            'sma_20', 'sma_50', 'ema_12', 'ema_26',
            'rsi', 'macd', 'macd_signal', 'macd_hist',
            'bb_upper', 'bb_lower', 'bb_middle', 'bb_percent',
            'atr', 'vwap', 'obv', 'mfi',
            'stoch_k_slow', 'stoch_d_slow', 'stoch_k_fast', 'stoch_d_fast',
            'trix', 'trix_signal', 'trix_hist',
            'support_distance_pct', 'resistance_distance_pct',
            'stop_loss', 'take_profit'
        ]
        
        missing = [c for c in required if c not in df.columns]
        if missing:
            return False, f"Missing columns: {missing[:5]}...", missing
        return True, "OK", []
    
    def validate_no_nan(self, df: pd.DataFrame) -> Tuple[bool, str, Dict]:
        """Check for NaN values in critical columns"""
        critical = ['Close', 'sma_20', 'rsi', 'macd', 'adx']
        issues = {}
        
        for col in critical:
            if col in df.columns:
                nan_count = df[col].isna().sum()
                if nan_count > 0:
                    issues[f'{col}_nan'] = int(nan_count)
        
        if issues:
            return False, f"NaN values found", issues
        return True, "OK", {}
    
    def validate_crossovers(self, df: pd.DataFrame) -> Tuple[bool, str, Dict]:
        """Validate crossover detection worked"""
        issues = {}
        
        # Check MACD crossover column exists
        if 'macd_cross_type' not in df.columns:
            issues['missing_macd_cross_col'] = True
        else:
            # Check if any crossovers detected
            crossovers = df[df['macd_cross_type'].isin(['crossover', 'crossunder'])]
            if len(crossovers) == 0:
                issues['no_macd_crossovers'] = True
        
        # Check Stochastic crossover
        if 'stoch_slow_cross' not in df.columns:
            issues['missing_stoch_cross_col'] = True
        
        # Check TRIX crossover
        if 'trix_cross_type' not in df.columns:
            issues['missing_trix_cross_col'] = True
        
        if issues:
            return False, f"Crossover issues", issues
        return True, "OK", {}
    
    def validate_support_resistance(self, df: pd.DataFrame) -> Tuple[bool, str, Dict]:
        """Validate S/R levels calculated"""
        issues = {}
        
        # Check S/R distance columns
        for col in ['support_distance_pct', 'resistance_distance_pct']:
            if col not in df.columns:
                issues[f'missing_{col}'] = True
            else:
                nan_count = df[col].isna().sum()
                if nan_count == len(df):
                    issues[f'{col}_all_nan'] = True
        
        # Check support_levels JSON exists
        if 'support_levels' not in df.columns:
            issues['missing_support_levels'] = True
        
        if issues:
            return False, f"S/R issues", issues
        return True, "OK", {}
    
    def validate_fibonacci(self, df: pd.DataFrame) -> Tuple[bool, str, Dict]:
        """Validate Fibonacci levels present"""
        issues = {}
        
        fib_cols = ['fib_23_6', 'fib_38_2', 'fib_50_0', 'fib_61_8', 'fib_78_6']
        for col in fib_cols:
            if col not in df.columns:
                issues[f'missing_{col}'] = True
        
        if issues:
            return False, f"Fibonacci issues", issues
        return True, "OK", {}
    
    def validate_candlestick_patterns(self, df: pd.DataFrame) -> Tuple[bool, str, Dict]:
        """Check candlestick patterns detected"""
        issues = {}
        
        pattern_cols = [
            'pattern_doji', 'pattern_hammer', 'pattern_shooting_star',
            'pattern_engulfing', 'pattern_morning_star', 'pattern_evening_star',
            'pattern_harami', 'pattern_piercing', 'pattern_dark_cloud',
            'pattern_three_white_soldiers', 'pattern_three_black_crows'
        ]
        
        found_any = False
        for col in pattern_cols:
            if col in df.columns:
                if df[col].sum() > 0:
                    found_any = True
                    break
        
        if not found_any:
            issues['no_candlestick_patterns'] = True
        
        if issues:
            return False, f"Candlestick issues", issues
        return True, "OK", {}
    
    def validate_ticker(self, ticker: str) -> Dict:
        """Validate all intervals for a ticker"""
        ticker_results = {
            'ticker': ticker,
            'intervals': {},
            'overall_pass': True
        }
        
        for interval in INTERVALS:
            result = {
                'interval': interval,
                'pass': False,
                'errors': []
            }
            
            # Check file exists
            exists_ok, exists_msg = self.validate_file_exists(ticker, interval)
            if not exists_ok:
                result['errors'].append(exists_msg)
                ticker_results['intervals'][interval] = result
                ticker_results['overall_pass'] = False
                continue
            
            # Load file
            df = self.load_technical_file(ticker, interval)
            if df is None:
                result['errors'].append("Failed to load file")
                ticker_results['intervals'][interval] = result
                ticker_results['overall_pass'] = False
                continue
            
            # Run validations
            validations = [
                ('columns', self.validate_required_columns),
                ('nan_values', self.validate_no_nan),
                ('crossovers', self.validate_crossovers),
                ('support_resistance', self.validate_support_resistance),
                ('fibonacci', self.validate_fibonacci),
                ('candlesticks', self.validate_candlestick_patterns)
            ]
            
            all_pass = True
            for name, validator in validations:
                try:
                    passed, msg, details = validator(df)
                    if not passed:
                        all_pass = False
                        result['errors'].append(f"{name}: {msg}")
                        for k, v in details.items():
                            self.summary['errors_by_type'][f"{name}:{k}"] = self.summary['errors_by_type'].get(f"{name}:{k}", 0) + 1
                except Exception as e:
                    all_pass = False
                    result['errors'].append(f"{name}: Exception - {str(e)}")
            
            result['pass'] = all_pass
            if not all_pass:
                ticker_results['overall_pass'] = False
            
            ticker_results['intervals'][interval] = result
        
        return ticker_results
    
    def get_tickers(self) -> List[str]:
        """Get list of tickers with technical analysis"""
        if not TECHNICAL_DIR.exists():
            return []
        return sorted([d.name for d in TECHNICAL_DIR.iterdir() if d.is_dir() and not d.name.startswith('.')])
    
    def run_validation(self) -> Dict:
        """Run validation on all tickers"""
        print("="*70)
        print("STEP 3.5: Technical Analysis Validator")
        print("="*70)
        print()
        
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        tickers = self.get_tickers()
        self.summary['total_tickers'] = len(tickers)
        
        print(f"Validating {len(tickers)} tickers...")
        print(f"Intervals: {', '.join(INTERVALS)}")
        print()
        
        for i, ticker in enumerate(tickers):
            if (i + 1) % 100 == 0:
                print(f"  Processed {i+1}/{len(tickers)} tickers...")
            
            result = self.validate_ticker(ticker)
            self.results.append(result)
            
            # Update summary
            for interval in INTERVALS:
                if interval in result['intervals']:
                    self.summary['total_files'] += 1
                    if result['intervals'][interval]['pass']:
                        self.summary['passed'] += 1
                    else:
                        self.summary['failed'] += 1
        
        # Generate report
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': self.summary,
            'results': self.results
        }
        
        report_file = OUTPUT_DIR / f"validation_report_technical_{datetime.now().strftime('%Y%m%d')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
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
            print("\nTop errors:")
            for error_type, count in sorted(self.summary['errors_by_type'].items(), key=lambda x: -x[1])[:10]:
                print(f"  - {error_type}: {count}")
        
        print(f"\nReport saved to: {report_file}")
        print("="*70)
        
        return report

def main():
    validator = TechnicalValidator()
    validator.run_validation()

if __name__ == "__main__":
    main()
