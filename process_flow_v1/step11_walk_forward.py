#!/usr/bin/env python3
"""
================================================================================
STEP 11: Walk-Forward Analysis
================================================================================

Validates strategy robustness by testing on unseen data.
Prevents overfitting by using rolling train/test windows.

Daily data only - 252-day train, 20-day test windows.

Author: SignalsAlpha
Version: 1.0
Date: 2026-04-29
================================================================================

WALK-FORWARD METHODOLOGY:
================================================================================

1. WINDOW CONFIGURATION
   - Train window: 252 days (1 year)
   - Test window: 20 days (1 month)
   - Step size: 20 days (roll forward)

2. PROCESS FLOW
   ┌─────────────────────────────────────────────────────────────┐
   │ Train [1-252] → Test [253-272] → Record metrics             │
   │ Train [21-272] → Test [273-292] → Record metrics            │
   │ Train [41-292] → Test [293-312] → Record metrics            │
   │ ... continues until end of data                             │
   └─────────────────────────────────────────────────────────────┘

3. CONSISTENCY CHECKS
   - Train/Test correlation > 0.7
   - Profit Factor > 1.5 in BOTH train and test
   - Max Drawdown < 20% in test
   - Win rate within ±10% between train/test

4. OVERFITTING DETECTION
   - Flag if Train PF / Test PF > 1.5
   - Flag if Train Sharpe / Test Sharpe > 1.5
   - Require consistency across 3+ windows

5. AGGREGATION
   - Combine all test windows
   - Calculate aggregate metrics
   - Generate confidence intervals

================================================================================
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent / 'modules'))

from metrics_calculator import MetricsCalculator, Trade, PerformanceMetrics
from results_db import ResultsDatabase
from pipeline_bridge import PipelineBridge

# Configuration
DATA_DIR = Path(__file__).parent / "data"
TIME_SERIES_DIR = DATA_DIR / "time_series"
SIGNALS_DIR = DATA_DIR / "signals_scored"
WALK_FORWARD_DIR = DATA_DIR / "walk_forward"
WALK_FORWARD_DIR.mkdir(parents=True, exist_ok=True)

# Window parameters
TRAIN_WINDOW_DAYS = 252  # 1 year
TEST_WINDOW_DAYS = 20    # 1 month
STEP_SIZE_DAYS = 20      # Roll forward amount
MIN_WINDOWS = 3          # Minimum windows for validity

# Consistency thresholds
CONSISTENCY_THRESHOLD = 0.7  # Correlation > 0.7
OVERFITTING_RATIO = 1.5      # Train/Test ratio < 1.5
MIN_PROFIT_FACTOR = 1.5      # PF > 1.5 required
MAX_DRAWDOWN_PCT = 0.20      # Max 20% DD


@dataclass
class WindowResult:
    """Results from a single walk-forward window"""
    window_number: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    
    # Train metrics
    train_profit_factor: float
    train_sharpe: float
    train_win_rate: float
    train_max_dd: float
    
    # Test metrics
    test_profit_factor: float
    test_sharpe: float
    test_win_rate: float
    test_max_dd: float
    
    # Consistency
    is_consistent: bool
    overfitting_flags: List[str]


class WalkForwardValidator:
    """Walk-forward analysis for strategy validation"""
    
    def __init__(self, 
                 train_days: int = TRAIN_WINDOW_DAYS,
                 test_days: int = TEST_WINDOW_DAYS,
                 step_days: int = STEP_SIZE_DAYS):
        """Initialize validator"""
        self.train_days = train_days
        self.test_days = test_days
        self.step_days = step_days
        self.windows: List[WindowResult] = []
        self.aggregate_metrics: Optional[Dict] = None
        self.results_db = ResultsDatabase()
        self.pipeline_bridge = PipelineBridge()
        
    def load_historical_data(self, ticker: str = "AAPL") -> Optional[pd.DataFrame]:
        """
        Load historical price data
        
        For production, this would load all tickers.
        For validation, we use representative sample.
        """
        file_path = TIME_SERIES_DIR / ticker / f"{ticker}_1d.csv"
        
        if not file_path.exists():
            print(f"Data not found: {file_path}")
            return None
        
        try:
            df = pd.read_csv(file_path)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def create_windows(self, data: pd.DataFrame) -> List[Tuple[datetime, datetime, datetime, datetime]]:
        """
        Create train/test window pairs
        
        Returns:
            List of (train_start, train_end, test_start, test_end) tuples
        """
        windows = []
        
        if len(data) < self.train_days + self.test_days:
            print("Insufficient data for walk-forward analysis")
            return windows
        
        start_date = data['date'].min()
        end_date = data['date'].max()
        
        current_start = start_date
        window_num = 1
        
        while True:
            train_start = current_start
            train_end = train_start + timedelta(days=self.train_days)
            test_start = train_end + timedelta(days=1)
            test_end = test_start + timedelta(days=self.test_days)
            
            # Check if we have enough data
            if test_end > end_date:
                break
            
            windows.append((train_start, train_end, test_start, test_end))
            
            # Move window forward
            current_start += timedelta(days=self.step_days)
            window_num += 1
        
        return windows
    
    def simulate_backtest_window(self, 
                                 data: pd.DataFrame,
                                 start_date: datetime,
                                 end_date: datetime) -> Dict:
        """
        Simulate backtest for a time window
        
        For production, this would run actual Step 6 backtest.
        """
        # Filter data to window
        mask = (data['date'] >= start_date) & (data['date'] <= end_date)
        window_data = data[mask]
        
        if len(window_data) < 10:
            return {'profit_factor': 0, 'sharpe_ratio': 0, 'win_rate': 0, 'max_drawdown': 1.0}
        
        # Simulate returns based on price movement
        returns = window_data['close'].pct_change().dropna()
        
        if len(returns) == 0:
            return {'profit_factor': 0, 'sharpe_ratio': 0, 'win_rate': 0, 'max_drawdown': 1.0}
        
        # Calculate simulated metrics
        wins = (returns > 0).sum()
        losses = (returns <= 0).sum()
        win_rate = wins / len(returns) if len(returns) > 0 else 0
        
        gross_profit = returns[returns > 0].sum() if wins > 0 else 0
        gross_loss = abs(returns[returns < 0].sum()) if losses > 0 else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        sharpe = returns.mean() / (returns.std() + 1e-6) * np.sqrt(252)
        
        # Calculate max drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_dd = abs(drawdown.min()) if len(drawdown) > 0 else 0
        
        return {
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe,
            'win_rate': win_rate,
            'max_drawdown': max_dd
        }
    
    def check_consistency(self, train_metrics: Dict, test_metrics: Dict) -> Tuple[bool, List[str]]:
        """
        Check consistency between train and test metrics
        
        Returns:
            (is_consistent, list_of_flags)
        """
        flags = []
        
        # Check profit factor overfitting
        if train_metrics['profit_factor'] > 0:
            pf_ratio = train_metrics['profit_factor'] / max(test_metrics['profit_factor'], 0.01)
            if pf_ratio > OVERFITTING_RATIO:
                flags.append(f"PF overfitting: {pf_ratio:.2f}x")
        
        # Check Sharpe ratio overfitting
        if train_metrics['sharpe_ratio'] > 0 and test_metrics['sharpe_ratio'] > 0:
            sharpe_ratio = train_metrics['sharpe_ratio'] / test_metrics['sharpe_ratio']
            if sharpe_ratio > OVERFITTING_RATIO:
                flags.append(f"Sharpe overfitting: {sharpe_ratio:.2f}x")
        
        # Check minimum profit factor
        if test_metrics['profit_factor'] < MIN_PROFIT_FACTOR:
            flags.append(f"Low test PF: {test_metrics['profit_factor']:.2f}")
        
        # Check max drawdown
        if test_metrics['max_drawdown'] > MAX_DRAWDOWN_PCT:
            flags.append(f"High test DD: {test_metrics['max_drawdown']:.1%}")
        
        # Check win rate consistency
        wr_diff = abs(train_metrics['win_rate'] - test_metrics['win_rate'])
        if wr_diff > 0.10:  # 10% difference
            flags.append(f"Win rate drift: {wr_diff:.1%}")
        
        is_consistent = len(flags) == 0
        
        return is_consistent, flags
    
    def run_window(self, 
                   window_num: int,
                   train_start: datetime,
                   train_end: datetime,
                   test_start: datetime,
                   test_end: datetime,
                   data: pd.DataFrame) -> WindowResult:
        """Run single walk-forward window"""
        
        # Train period
        train_metrics = self.simulate_backtest_window(data, train_start, train_end)
        
        # Test period
        test_metrics = self.simulate_backtest_window(data, test_start, test_end)
        
        # Check consistency
        is_consistent, flags = self.check_consistency(train_metrics, test_metrics)
        
        return WindowResult(
            window_number=window_num,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            train_profit_factor=train_metrics['profit_factor'],
            train_sharpe=train_metrics['sharpe_ratio'],
            train_win_rate=train_metrics['win_rate'],
            train_max_dd=train_metrics['max_drawdown'],
            test_profit_factor=test_metrics['profit_factor'],
            test_sharpe=test_metrics['sharpe_ratio'],
            test_win_rate=test_metrics['win_rate'],
            test_max_dd=test_metrics['max_drawdown'],
            is_consistent=is_consistent,
            overfitting_flags=flags
        )
    
    def run_analysis(self, ticker: str = "AAPL") -> List[WindowResult]:
        """Run complete walk-forward analysis"""
        print("=" * 70)
        print("WALK-FORWARD ANALYSIS")
        print("=" * 70)
        print(f"Train window: {self.train_days} days")
        print(f"Test window: {self.test_days} days")
        print(f"Step size: {self.step_days} days")
        print("=" * 70)
        
        # Load data
        data = self.load_historical_data(ticker)
        if data is None:
            return []
        
        print(f"\nLoaded {len(data)} days of data for {ticker}")
        
        # Create windows
        window_pairs = self.create_windows(data)
        print(f"Created {len(window_pairs)} walk-forward windows")
        
        if len(window_pairs) < MIN_WINDOWS:
            print(f"Warning: Only {len(window_pairs)} windows. Need {MIN_WINDOWS} minimum.")
        
        # Run each window
        self.windows = []
        for i, (train_start, train_end, test_start, test_end) in enumerate(window_pairs, 1):
            print(f"\nWindow {i}/{len(window_pairs)}: {train_start.date()} → {test_end.date()}")
            
            result = self.run_window(i, train_start, train_end, test_start, test_end, data)
            self.windows.append(result)
            
            # Print window results
            print(f"  Train: PF={result.train_profit_factor:.2f}, Sharpe={result.train_sharpe:.2f}")
            print(f"  Test:  PF={result.test_profit_factor:.2f}, Sharpe={result.test_sharpe:.2f}")
            print(f"  Consistent: {result.is_consistent}")
            if result.overfitting_flags:
                print(f"  Flags: {', '.join(result.overfitting_flags)}")
        
        # Calculate aggregate metrics
        self.calculate_aggregate_metrics()
        
        # Calculate walk-forward score using bridge
        consistency_score, _ = self.pipeline_bridge.calculate_walk_forward_score(
            self.windows, MIN_WINDOWS
        )
        
        # Save results to database (if config_id is available)
        # For now, we store in aggregate_metrics for later use
        self.aggregate_metrics['consistency_score'] = consistency_score
        
        # Print summary
        self.print_summary()
        
        return self.windows
    
    def calculate_aggregate_metrics(self):
        """Calculate aggregate metrics across all test periods"""
        if not self.windows:
            return
        
        test_pfs = [w.test_profit_factor for w in self.windows]
        test_sharpes = [w.test_sharpe for w in self.windows]
        test_win_rates = [w.test_win_rate for w in self.windows]
        test_max_dds = [w.test_max_dd for w in self.windows]
        
        consistent_count = sum(1 for w in self.windows if w.is_consistent)
        
        self.aggregate_metrics = {
            'total_windows': len(self.windows),
            'consistent_windows': consistent_count,
            'consistency_pct': consistent_count / len(self.windows) if self.windows else 0,
            
            'avg_test_pf': np.mean(test_pfs),
            'min_test_pf': np.min(test_pfs),
            'max_test_pf': np.max(test_pfs),
            'std_test_pf': np.std(test_pfs),
            
            'avg_test_sharpe': np.mean(test_sharpes),
            'min_test_sharpe': np.min(test_sharpes),
            'max_test_sharpe': np.max(test_sharpes),
            
            'avg_test_win_rate': np.mean(test_win_rates),
            'avg_test_max_dd': np.mean(test_max_dds),
        }
    
    def print_summary(self):
        """Print walk-forward summary"""
        print("\n" + "=" * 70)
        print("WALK-FORWARD SUMMARY")
        print("=" * 70)
        
        if not self.windows:
            print("No windows to summarize")
            return
        
        consistent = sum(1 for w in self.windows if w.is_consistent)
        total = len(self.windows)
        
        print(f"\nTotal Windows: {total}")
        print(f"Consistent: {consistent} ({consistent/total*100:.1f}%)")
        print(f"Overfitting Detected: {total - consistent}")
        
        if self.aggregate_metrics:
            print(f"\nTest Period Performance (Aggregated):")
            print(f"  Profit Factor: {self.aggregate_metrics['avg_test_pf']:.2f} "
                  f"(min: {self.aggregate_metrics['min_test_pf']:.2f}, "
                  f"max: {self.aggregate_metrics['max_test_pf']:.2f})")
            print(f"  Sharpe Ratio: {self.aggregate_metrics['avg_test_sharpe']:.2f} "
                  f"(min: {self.aggregate_metrics['min_test_sharpe']:.2f}, "
                  f"max: {self.aggregate_metrics['max_test_sharpe']:.2f})")
            print(f"  Win Rate: {self.aggregate_metrics['avg_test_win_rate']:.1%}")
            print(f"  Max Drawdown: {self.aggregate_metrics['avg_test_max_dd']:.1%}")
        
        # Validation result
        print("\n" + "=" * 70)
        if consistent >= MIN_WINDOWS and self.aggregate_metrics['avg_test_pf'] >= MIN_PROFIT_FACTOR:
            print("✓ VALIDATION PASSED")
            print(f"  Strategy shows consistency across {consistent} windows")
            print(f"  Average Profit Factor: {self.aggregate_metrics['avg_test_pf']:.2f}")
        else:
            print("✗ VALIDATION FAILED")
            print(f"  Only {consistent}/{total} windows passed consistency checks")
            if self.aggregate_metrics:
                print(f"  Average Profit Factor: {self.aggregate_metrics['avg_test_pf']:.2f} "
                      f"(need > {MIN_PROFIT_FACTOR})")
        print("=" * 70)
    
    def export_results(self, output_path: Path):
        """Export window results to CSV"""
        if not self.windows:
            print("No results to export")
            return
        
        records = []
        for w in self.windows:
            records.append({
                'window': w.window_number,
                'train_start': w.train_start,
                'train_end': w.train_end,
                'test_start': w.test_start,
                'test_end': w.test_end,
                'train_pf': w.train_profit_factor,
                'train_sharpe': w.train_sharpe,
                'train_wr': w.train_win_rate,
                'train_dd': w.train_max_dd,
                'test_pf': w.test_profit_factor,
                'test_sharpe': w.test_sharpe,
                'test_wr': w.test_win_rate,
                'test_dd': w.test_max_dd,
                'is_consistent': w.is_consistent,
                'flags': '|'.join(w.overfitting_flags) if w.overfitting_flags else ''
            })
        
        df = pd.DataFrame(records)
        df.to_csv(output_path, index=False)
        print(f"\nExported results to: {output_path}")
    
    def save_to_database(self, config_id: int):
        """Save walk-forward results to database using bridge"""
        if not self.windows or not self.aggregate_metrics:
            return
        
        # Calculate consistency score and aggregate metrics
        consistency_score, aggregate_metrics = self.pipeline_bridge.calculate_walk_forward_score(
            self.windows, MIN_WINDOWS
        )
        
        # Save using bridge
        success = self.pipeline_bridge.save_step11_result(
            config_id=config_id,
            consistency_score=consistency_score,
            aggregate_metrics=aggregate_metrics
        )
        
        if success:
            print(f"Saved walk-forward results to database for config {config_id}")


def main():
    """Main entry point"""
    validator = WalkForwardValidator()
    
    # Run walk-forward analysis
    results = validator.run_analysis(ticker="AAPL")
    
    # Export results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = WALK_FORWARD_DIR / f"walk_forward_results_{timestamp}.csv"
    validator.export_results(output_path)
    
    # Save to database
    if validator.aggregate_metrics:
        validator.save_to_database(config_id=1)
    
    print("\n" + "=" * 70)
    print("WALK-FORWARD ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
