#!/usr/bin/env python3
"""
================================================================================
STEP 6 ENHANCED - ENHANCED BACKTEST METRICS
================================================================================

Bridge module to integrate metrics_calculator with existing Step 6 backtester.
Converts Step 6 trade data to enhanced metrics without modifying core backtester.

Author: SignalsAlpha
Version: 1.0
Date: 2026-04-28
================================================================================
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Import our modules
from metrics_calculator import MetricsCalculator, Trade, PerformanceMetrics
from results_db import ResultsDatabase


class Step6EnhancedMetrics:
    """Enhanced metrics wrapper for Step 6 backtest results"""
    
    def __init__(self, backtest_json_path: Optional[Path] = None):
        """
        Initialize with path to Step 6 backtest JSON
        
        Args:
            backtest_json_path: Path to backtest_report_*.json file
        """
        self.backtest_path = backtest_json_path
        self.calculator = MetricsCalculator()
        self.trades: List[Trade] = []
        self.metrics: Optional[PerformanceMetrics] = None
        
    def load_from_step6_json(self, json_path: Path) -> bool:
        """
        Load and parse Step 6 backtest JSON
        
        Returns:
            True if successful
        """
        try:
            with open(json_path) as f:
                data = json.load(f)
            
            # Extract trades from ticker_results
            all_trades = []
            ticker_results = data.get('ticker_results', {})
            
            for ticker, result in ticker_results.items():
                trades = result.get('trades', [])
                for trade_data in trades:
                    # Convert Step 6 trade format to Trade dataclass
                    trade = Trade(
                        entry_date=datetime.strptime(trade_data['entry_date'], '%Y-%m-%d'),
                        exit_date=datetime.strptime(trade_data['exit_date'], '%Y-%m-%d'),
                        entry_price=trade_data['entry_price'],
                        exit_price=trade_data['exit_price'],
                        position_size=0,  # Not tracked in Step 6
                        pnl=trade_data['pnl_pct'] * 100,  # Convert to dollars for simulation
                        return_pct=trade_data['pnl_pct'] / 100,
                        signal_type=trade_data.get('entry_signal', 'UNKNOWN'),
                        ticker=ticker
                    )
                    all_trades.append(trade)
            
            self.trades = all_trades
            print(f"Loaded {len(all_trades)} trades from {json_path.name}")
            return True
            
        except Exception as e:
            print(f"Error loading backtest JSON: {e}")
            return False
    
    def load_from_step6_data(self, ticker_results: Dict) -> bool:
        """
        Load directly from Step 6 ticker_results dictionary
        
        Args:
            ticker_results: Dict from Step 6 backtest
            
        Returns:
            True if successful
        """
        try:
            all_trades = []
            
            for ticker, result in ticker_results.items():
                trades = result.get('trades', [])
                for trade_data in trades:
                    trade = Trade(
                        entry_date=datetime.strptime(trade_data['entry_date'], '%Y-%m-%d'),
                        exit_date=datetime.strptime(trade_data['exit_date'], '%Y-%m-%d'),
                        entry_price=trade_data['entry_price'],
                        exit_price=trade_data['exit_price'],
                        position_size=0,
                        pnl=trade_data['pnl_pct'] * 100,
                        return_pct=trade_data['pnl_pct'] / 100,
                        signal_type=trade_data.get('entry_signal', 'UNKNOWN'),
                        ticker=ticker
                    )
                    all_trades.append(trade)
            
            self.trades = all_trades
            return True
            
        except Exception as e:
            print(f"Error loading trades: {e}")
            return False
    
    def calculate_enhanced_metrics(self) -> Optional[PerformanceMetrics]:
        """
        Calculate enhanced metrics from loaded trades
        
        Returns:
            PerformanceMetrics object or None
        """
        if not self.trades:
            print("No trades loaded. Call load_from_step6_* first.")
            return None
        
        self.metrics = self.calculator.calculate_from_trades(self.trades)
        return self.metrics
    
    def print_enhanced_report(self):
        """Print comprehensive performance report"""
        if not self.metrics:
            print("No metrics calculated. Run calculate_enhanced_metrics() first.")
            return
        
        report = self.calculator.print_report(self.metrics)
        print(report)
    
    def save_to_database(self, 
                        config_name: str,
                        config: Dict,
                        notes: str = "") -> Optional[int]:
        """
        Save enhanced metrics to results database
        
        Returns:
            Database ID or None
        """
        if not self.metrics:
            print("No metrics calculated.")
            return None
        
        with ResultsDatabase() as db:
            # Convert trades to serializable format
            trades_serializable = []
            for trade in self.trades:
                trades_serializable.append({
                    'ticker': trade.ticker,
                    'entry_date': trade.entry_date.strftime('%Y-%m-%d'),
                    'exit_date': trade.exit_date.strftime('%Y-%m-%d'),
                    'entry_price': trade.entry_price,
                    'exit_price': trade.exit_price,
                    'pnl_pct': trade.return_pct * 100,
                    'signal_type': trade.signal_type
                })
            
            result_id = db.save_backtest_result(
                config_name=config_name,
                config=config,
                metrics=self.metrics,
                trades=trades_serializable,
                notes=notes
            )
            
            print(f"Saved to database with ID: {result_id}")
            return result_id
    
    def compare_with_previous(self, config_name: str) -> Optional[pd.DataFrame]:
        """
        Compare current results with previous runs of same config
        
        Returns:
            Comparison DataFrame or None
        """
        if not self.metrics:
            print("No metrics calculated.")
            return None
        
        with ResultsDatabase() as db:
            # Get historical results
            historical = db.get_all_results(config_name)
            
            if len(historical) == 0:
                print(f"No previous results for config: {config_name}")
                return None
            
            # Create comparison
            comparison = pd.DataFrame({
                'Metric': ['Profit Factor', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate', 'Net Profit'],
                'Current': [
                    self.metrics.profit_factor,
                    self.metrics.sharpe_ratio,
                    self.metrics.max_drawdown * 100,
                    self.metrics.win_rate * 100,
                    self.metrics.net_profit
                ],
                'Previous Avg': [
                    historical['profit_factor'].mean(),
                    historical['sharpe_ratio'].mean(),
                    historical['max_drawdown'].mean() * 100,
                    historical['win_rate'].mean() * 100,
                    historical['net_profit'].mean()
                ],
                'Best': [
                    historical['profit_factor'].max(),
                    historical['sharpe_ratio'].max(),
                    historical['max_drawdown'].min() * 100,  # Lower is better
                    historical['win_rate'].max() * 100,
                    historical['net_profit'].max()
                ]
            })
            
            return comparison
    
    def export_trade_log(self, output_path: Path):
        """Export detailed trade log to CSV"""
        if not self.trades:
            print("No trades loaded.")
            return
        
        trade_data = []
        for trade in self.trades:
            trade_data.append({
                'ticker': trade.ticker,
                'entry_date': trade.entry_date,
                'exit_date': trade.exit_date,
                'days_held': (trade.exit_date - trade.entry_date).days,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'pnl_pct': trade.return_pct * 100,
                'signal_type': trade.signal_type
            })
        
        df = pd.DataFrame(trade_data)
        df.to_csv(output_path, index=False)
        print(f"Exported {len(df)} trades to {output_path}")
    
    def get_trade_statistics(self) -> Dict:
        """Get detailed trade statistics"""
        if not self.trades:
            return {}
        
        df = pd.DataFrame([
            {
                'ticker': t.ticker,
                'pnl': t.return_pct * 100,
                'days': (t.exit_date - t.entry_date).days
            }
            for t in self.trades
        ])
        
        return {
            'total_trades': len(df),
            'unique_tickers': df['ticker'].nunique(),
            'avg_hold_days': df['days'].mean(),
            'median_hold_days': df['days'].median(),
            'best_trade': df['pnl'].max(),
            'worst_trade': df['pnl'].min(),
            'avg_pnl': df['pnl'].mean(),
            'pnl_std': df['pnl'].std(),
            'skewness': df['pnl'].skew(),
            'kurtosis': df['pnl'].kurtosis()
        }


def enhance_existing_backtest(backtest_json_path: Path, 
                             config_name: str = "enhanced_v1"):
    """
    One-step function to enhance an existing Step 6 backtest
    
    Args:
        backtest_json_path: Path to Step 6 JSON output
        config_name: Name for this configuration
    """
    print("=" * 70)
    print("STEP 6 ENHANCED METRICS")
    print("=" * 70)
    
    # Load and enhance
    enhancer = Step6EnhancedMetrics()
    
    if not enhancer.load_from_step6_json(backtest_json_path):
        print("Failed to load backtest")
        return
    
    # Calculate enhanced metrics
    metrics = enhancer.calculate_enhanced_metrics()
    if not metrics:
        print("Failed to calculate metrics")
        return
    
    # Print report
    print("\n")
    enhancer.print_enhanced_report()
    
    # Save to database
    config = {
        'source_file': str(backtest_json_path.name),
        'enhanced': True
    }
    
    result_id = enhancer.save_to_database(
        config_name=config_name,
        config=config,
        notes=f"Enhanced metrics for {backtest_json_path.name}"
    )
    
    # Export trade log
    output_csv = backtest_json_path.parent / f"{backtest_json_path.stem}_trades_enhanced.csv"
    enhancer.export_trade_log(output_csv)
    
    # Trade statistics
    print("\n" + "=" * 70)
    print("TRADE STATISTICS")
    print("=" * 70)
    stats = enhancer.get_trade_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key:20s}: {value:.2f}")
        else:
            print(f"{key:20s}: {value}")
    
    print("\n" + "=" * 70)
    print(f"Enhanced metrics saved to database (ID: {result_id})")
    print(f"Trade log exported to: {output_csv}")
    print("=" * 70)


if __name__ == "__main__":
    # Example: Enhance the latest backtest
    import sys
    
    # Find latest backtest
    backtest_dir = Path(__file__).parent.parent / "data" / "backtests"
    backtest_files = list(backtest_dir.glob("backtest_report_*.json"))
    
    if backtest_files:
        latest = max(backtest_files, key=lambda p: p.stat().st_mtime)
        enhance_existing_backtest(latest, config_name="production_run")
    else:
        print("No backtest files found")
