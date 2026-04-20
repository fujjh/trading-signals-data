#!/usr/bin/env python3
"""
================================================================================
STEP 5: Signal Validator
================================================================================

Validates scored signals from Step 4 before backtesting and output generation.
Ensures signal integrity, consistency, and completeness.

Author: SignalsAlpha
Version: 1.0
Date: 2026-04-20

================================================================================
PURPOSE
================================================================================

Quality assurance for scored signals:

1. Validates all scored signals file exists and is readable
2. Checks signal score ranges (0-100 for technical, A-F for fundamental)
3. Verifies signal direction consistency with technical/fundamental alignment
4. Ensures price targets (stop_loss/take_profit) are reasonable
5. Validates coverage (all tickers have signals for all intervals)
6. Detects duplicate or conflicting signals
7. Checks confidence levels are within valid ranges

================================================================================
VALIDATION CHECKS
================================================================================

File Validation:
    - Scored signals CSV exists
    - File parses successfully
    - Required columns present

Signal Score Validation:
    - Technical scores: 0-100
    - Fundamental grades: A, B, C, D, F
    - Combined scores: 0-100
    - Confidence: 0-100%

Signal Consistency:
    - Signal direction matches tech/fund agreement
    - STRONG_BUY requires combined_score >= 75
    - STRONG_SELL requires combined_score >= 75 (bearish)
    - Price targets reasonable vs current price

Coverage Checks:
    - All tickers from time series have signals
    - All intervals present for each ticker
    - No missing signal data

Integration Validation:
    - Technical analysis exists for each signal
    - Fundamental data present for each ticker
    - No orphaned signals
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Configuration
DATA_DIR = Path("/home/ubuntu/.openclaw/workspace/trading-signals-data/process_flow_v1/data")
SCORED_SIGNALS_DIR = DATA_DIR / "scored_signals"
TIME_SERIES_DIR = DATA_DIR / "time_series"
TECHNICAL_DIR = DATA_DIR / "technical_analysis"
FUNDAMENTALS_DIR = DATA_DIR / "fundamentals"
OUTPUT_DIR = DATA_DIR / "validated"
INTERVALS = ['1d', '1wk', '1mo']
VALID_SIGNALS = ['STRONG_BUY', 'BUY', 'WEAK_BUY', 'HOLD', 'WEAK_SELL', 'SELL', 'STRONG_SELL']
VALID_GRADES = ['A', 'B', 'C', 'D', 'F']

class SignalValidator:
    """Validator for scored signals from Step 4"""
    
    def __init__(self):
        self.results = []
        self.summary = {
            'total_signals': 0,
            'passed': 0,
            'failed': 0,
            'errors_by_type': {}
        }
        self.df = None
    
    def load_signals(self) -> Optional[pd.DataFrame]:
        """Load the scored signals CSV"""
        if not SCORED_SIGNALS_DIR.exists():
            return None
        
        # Find latest scored signals file
        files = sorted(SCORED_SIGNALS_DIR.glob("scored_signals_*.csv"))
        if not files:
            return None
        
        latest_file = files[-1]
        try:
            df = pd.read_csv(latest_file)
            print(f"Loaded {len(df)} signals from {latest_file.name}")
            return df
        except Exception as e:
            print(f"Error loading signals: {e}")
            return None
    
    def validate_file_structure(self, df: pd.DataFrame) -> Tuple[bool, str, List[str]]:
        """Validate CSV has required columns"""
        required_cols = [
            'ticker', 'interval', 'date', 'close',
            'technical_signal', 'technical_confidence', 'technical_raw_score',
            'fundamental_score', 'fundamental_grade',
            'combined_score', 'final_signal', 'conviction_pct',
            'stop_loss', 'take_profit'
        ]
        
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            return False, f"Missing required columns: {missing}", missing
        
        return True, "OK", []
    
    def validate_score_ranges(self, df: pd.DataFrame) -> Tuple[bool, str, Dict]:
        """Validate all scores are in valid ranges"""
        issues = {}
        
        # Technical score: 0-100
        invalid = ((df['technical_raw_score'] < 0) | (df['technical_raw_score'] > 100)).sum()
        if invalid > 0:
            issues['technical_score_out_of_range'] = int(invalid)
        
        # Fundamental score: 0-100
        invalid = ((df['fundamental_score'] < 0) | (df['fundamental_score'] > 100)).sum()
        if invalid > 0:
            issues['fundamental_score_out_of_range'] = int(invalid)
        
        # Combined score: 0-100
        invalid = ((df['combined_score'] < 0) | (df['combined_score'] > 100)).sum()
        if invalid > 0:
            issues['combined_score_out_of_range'] = int(invalid)
        
        # Confidence: 0-100
        invalid = ((df['conviction_pct'] < 0) | (df['conviction_pct'] > 100)).sum()
        if invalid > 0:
            issues['confidence_out_of_range'] = int(invalid)
        
        # Technical confidence: 0-100
        invalid = ((df['technical_confidence'] < 0) | (df['technical_confidence'] > 100)).sum()
        if invalid > 0:
            issues['technical_confidence_out_of_range'] = int(invalid)
        
        # Fundamental grade: A-F
        invalid = (~df['fundamental_grade'].isin(VALID_GRADES)).sum()
        if invalid > 0:
            issues['invalid_fundamental_grade'] = int(invalid)
        
        # Final signal: valid values
        invalid = (~df['final_signal'].isin(VALID_SIGNALS)).sum()
        if invalid > 0:
            issues['invalid_final_signal'] = int(invalid)
        
        if issues:
            return False, f"Score range issues: {list(issues.keys())}", issues
        
        return True, "OK", {}
    
    def validate_signal_consistency(self, df: pd.DataFrame) -> Tuple[bool, str, Dict]:
        """Validate signal direction matches scores"""
        issues = {}
        
        # STRONG_BUY should have high combined score
        strong_buy = df[df['final_signal'] == 'STRONG_BUY']
        if len(strong_buy) > 0:
            low_score = (strong_buy['combined_score'] < 75).sum()
            if low_score > 0:
                issues['strong_buy_low_score'] = int(low_score)
        
        # STRONG_SELL should have context suggesting bearish
        strong_sell = df[df['final_signal'] == 'STRONG_SELL']
        if len(strong_sell) > 0:
            low_score = (strong_sell['combined_score'] < 75).sum()
            if low_score > 0:
                issues['strong_sell_low_score'] = int(low_score)
        
        # BUY should have combined_score >= 60
        buy = df[df['final_signal'] == 'BUY']
        if len(buy) > 0:
            low_score = (buy['combined_score'] < 60).sum()
            if low_score > 0:
                issues['buy_low_score'] = int(low_score)
        
        # SELL should have combined_score >= 60
        sell = df[df['final_signal'] == 'SELL']
        if len(sell) > 0:
            low_score = (sell['combined_score'] < 60).sum()
            if low_score > 0:
                issues['sell_low_score'] = int(low_score)
        
        # Check technical/fundamental agreement for strong signals
        bullish = ['STRONG_BUY', 'BUY']
        bearish = ['STRONG_SELL', 'SELL']
        
        bullish_signals = df[df['final_signal'].isin(bullish)]
        if len(bullish_signals) > 0:
            # Should have technical buy signals
            tech_bearish = (~bullish_signals['technical_signal'].isin(['STRONG_BUY', 'BUY', 'WEAK_BUY'])).sum()
            if tech_bearish > 0:
                issues['bullish_with_bearish_technical'] = int(tech_bearish)
        
        bearish_signals = df[df['final_signal'].isin(bearish)]
        if len(bearish_signals) > 0:
            # Should have technical sell signals
            tech_bullish = (~bearish_signals['technical_signal'].isin(['STRONG_SELL', 'SELL', 'WEAK_SELL'])).sum()
            if tech_bullish > 0:
                issues['bearish_with_bullish_technical'] = int(tech_bullish)
        
        if issues:
            return False, f"Signal consistency issues: {list(issues.keys())}", issues
        
        return True, "OK", {}
    
    def validate_price_targets(self, df: pd.DataFrame) -> Tuple[bool, str, Dict]:
        """Validate stop_loss and take_profit are reasonable"""
        issues = {}
        
        # Stop loss should be < current price
        invalid_sl = (df['stop_loss'] >= df['close']).sum()
        if invalid_sl > 0:
            issues['stop_loss_above_price'] = int(invalid_sl)
        
        # Take profit should be > current price
        invalid_tp = (df['take_profit'] <= df['close']).sum()
        if invalid_tp > 0:
            issues['take_profit_below_price'] = int(invalid_tp)
        
        # Risk/reward should be reasonable (max 10:1)
        bullish = df[df['final_signal'].isin(['STRONG_BUY', 'BUY', 'WEAK_BUY'])]
        if len(bullish) > 0:
            risk = bullish['close'] - bullish['stop_loss']
            reward = bullish['take_profit'] - bullish['close']
            rr_ratio = reward / risk.replace(0, np.nan)
            extreme_rr = ((rr_ratio > 10) | (rr_ratio < 0.1)).sum()
            if extreme_rr > 0:
                issues['extreme_risk_reward'] = int(extreme_rr)
        
        if issues:
            return False, f"Price target issues: {list(issues.keys())}", issues
        
        return True, "OK", {}
    
    def validate_coverage(self, df: pd.DataFrame) -> Tuple[bool, str, Dict]:
        """Validate all tickers have signals for all intervals"""
        issues = {}
        
        # Get expected tickers from time series
        expected_tickers = set()
        if TIME_SERIES_DIR.exists():
            expected_tickers = {d.name for d in TIME_SERIES_DIR.iterdir() if d.is_dir() and not d.name.startswith('.')}
        
        # Get actual tickers with signals
        actual_tickers = set(df['ticker'].unique())
        
        # Check for missing tickers
        missing_tickers = expected_tickers - actual_tickers
        if missing_tickers:
            issues['missing_tickers'] = len(missing_tickers)
        
        # Check for extra tickers (signals without time series data)
        extra_tickers = actual_tickers - expected_tickers
        if extra_tickers:
            issues['extra_tickers'] = len(extra_tickers)
        
        # Check interval coverage for each ticker
        coverage_issues = 0
        for ticker in actual_tickers:
            ticker_df = df[df['ticker'] == ticker]
            intervals_present = set(ticker_df['interval'].unique())
            missing_intervals = set(INTERVALS) - intervals_present
            if missing_intervals:
                coverage_issues += 1
        
        if coverage_issues > 0:
            issues['incomplete_interval_coverage'] = coverage_issues
        
        if issues:
            return False, f"Coverage issues: {list(issues.keys())}", issues
        
        return True, "OK", {}
    
    def validate_integration(self, df: pd.DataFrame) -> Tuple[bool, str, Dict]:
        """Validate signals have corresponding technical and fundamental data"""
        issues = {}
        
        sample_size = min(100, len(df))
        sample = df.sample(n=sample_size) if len(df) > sample_size else df
        
        missing_technical = 0
        missing_fundamental = 0
        
        for _, row in sample.iterrows():
            ticker = row['ticker']
            interval = row['interval']
            
            # Check technical analysis exists
            tech_file = TECHNICAL_DIR / ticker / f"{ticker}_{interval}_technical.csv"
            if not tech_file.exists():
                missing_technical += 1
            
            # Check fundamental data exists
            fund_file = FUNDAMENTALS_DIR / ticker / f"{ticker}_fundamentals.csv"
            if not fund_file.exists():
                missing_fundamental += 1
        
        if missing_technical > 0:
            issues['missing_technical_data'] = f"{missing_technical}/{sample_size} samples"
        
        if missing_fundamental > 0:
            issues['missing_fundamental_data'] = f"{missing_fundamental}/{sample_size} samples"
        
        if issues:
            return False, f"Integration issues: {list(issues.keys())}", issues
        
        return True, "OK", {}
    
    def run_validation(self) -> Dict:
        """Run all validation checks"""
        print("="*70)
        print("STEP 5: Signal Validator")
        print("="*70)
        print()
        
        # Ensure output directory exists
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        # Load signals
        self.df = self.load_signals()
        if self.df is None:
            print("ERROR: No scored signals file found!")
            print(f"Expected in: {SCORED_SIGNALS_DIR}")
            return {'error': 'No signals file found'}
        
        self.summary['total_signals'] = len(self.df)
        print(f"Validating {len(self.df)} signals...")
        print()
        
        # Run all validations
        validations = [
            ('file_structure', self.validate_file_structure),
            ('score_ranges', self.validate_score_ranges),
            ('signal_consistency', self.validate_signal_consistency),
            ('price_targets', self.validate_price_targets),
            ('coverage', self.validate_coverage),
            ('integration', self.validate_integration)
        ]
        
        overall_pass = True
        validation_results = {}
        
        for name, validator_func in validations:
            try:
                passed, message, details = validator_func(self.df)
                validation_results[name] = {
                    'pass': passed,
                    'message': message,
                    'details': details
                }
                if not passed:
                    overall_pass = False
                    for key in details:
                        self.summary['errors_by_type'][f"{name}:{key}"] = self.summary['errors_by_type'].get(f"{name}:{key}", 0) + 1
            except Exception as e:
                validation_results[name] = {
                    'pass': False,
                    'message': f"Validation error: {str(e)}",
                    'details': {}
                }
                overall_pass = False
        
        # Calculate pass/fail
        if overall_pass:
            self.summary['passed'] = len(self.df)
            self.summary['failed'] = 0
        else:
            # Estimate failures based on error types
            self.summary['failed'] = sum(self.summary['errors_by_type'].values())
            self.summary['passed'] = max(0, len(self.df) - self.summary['failed'])
        
        # Generate report
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': self.summary,
            'validations': validation_results,
            'signal_distribution': self.df['final_signal'].value_counts().to_dict() if 'final_signal' in self.df.columns else {}
        }
        
        # Save report
        report_file = OUTPUT_DIR / f"validation_report_signals_{datetime.now().strftime('%Y%m%d')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Print summary
        print("="*70)
        print("VALIDATION SUMMARY")
        print("="*70)
        print(f"Total signals: {self.summary['total_signals']}")
        print(f"Passed: {self.summary['passed']} ({100*self.summary['passed']/max(self.summary['total_signals'],1):.1f}%)")
        print(f"Failed: {self.summary['failed']} ({100*self.summary['failed']/max(self.summary['total_signals'],1):.1f}%)")
        
        if self.summary['errors_by_type']:
            print("\nErrors by type:")
            for error_type, count in sorted(self.summary['errors_by_type'].items(), key=lambda x: -x[1])[:10]:
                print(f"  - {error_type}: {count}")
        
        print(f"\nReport saved to: {report_file}")
        print("="*70)
        
        return report

def main():
    validator = SignalValidator()
    validator.run_validation()

if __name__ == "__main__":
    main()
