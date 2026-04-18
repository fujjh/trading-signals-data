#!/usr/bin/env python3
"""
================================================================================
STEP 7: Walk-Forward Analyzer
================================================================================
Simulates day-by-day trading with current algorithm rules
- Tests on out-of-sample data
- Rolling window analysis
- Provides realistic performance expectations
================================================================================
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List

# Configuration
DATA_DIR = Path("/home/ubuntu/.openclaw/workspace/data")
TIME_SERIES_DIR = DATA_DIR / "time_series"
WALKFORWARD_DIR = DATA_DIR / "walkforward"

# Trading parameters
INITIAL_CAPITAL = 100000
MAX_POSITIONS = 10  # Max concurrent positions
RISK_PER_TRADE = 0.02  # 2% risk per trade


def ensure_dirs():
    """Create walkforward directory"""
    WALKFORWARD_DIR.mkdir(parents=True, exist_ok=True)


def generate_signals_for_day(df_window: pd.DataFrame) -> str:
    """
    Generate signal for a single day using current algorithm
    
    Returns: 'BUY', 'SELL', or 'HOLD'
    """
    if len(df_window) < 50:
        return 'HOLD'
    
    close = df_window['Close'].iloc[-1]
    
    # SMA
    sma_20 = df_window['Close'].rolling(20).mean().iloc[-1]
    sma_50 = df_window['Close'].rolling(50).mean().iloc[-1]
    
    # EMA
    ema_12 = df_window['Close'].ewm(span=12).mean().iloc[-1]
    ema_26 = df_window['Close'].ewm(span=26).mean().iloc[-1]
    
    # RSI
    delta = df_window['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean().iloc[-1]
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean().iloc[-1]
    rs = gain / loss if loss > 0 else 0
    rsi = 100 - (100 / (1 + rs))
    
    # MACD
    macd_line = ema_12 - ema_26
    signal_line = df_window['Close'].ewm(span=9).mean().iloc[-1]
    histogram = macd_line - signal_line
    
    # Bollinger
    bb_middle = df_window['Close'].rolling(20).mean().iloc[-1]
    bb_std = df_window['Close'].rolling(20).std().iloc[-1]
    bb_upper = bb_middle + (bb_std * 2)
    bb_lower = bb_middle - (bb_std * 2)
    
    # Scoring
    buy_score = 0
    sell_score = 0
    
    if close > sma_20 > sma_50:
        buy_score += 2
    elif close < sma_20 < sma_50:
        sell_score += 2
    
    if ema_12 > ema_26:
        buy_score += 1
    else:
        sell_score += 1
    
    if rsi < 30:
        buy_score += 2
    elif rsi > 70:
        sell_score += 2
    elif rsi > 50:
        buy_score += 1
    else:
        sell_score += 1
    
    if macd_line > signal_line and histogram > 0:
        buy_score += 2
    elif macd_line < signal_line and histogram < 0:
        sell_score += 2
    
    if close < bb_lower:
        buy_score += 1
    elif close > bb_upper:
        sell_score += 1
    
    # Return signal
    if buy_score >= 4:
        return 'BUY'
    elif sell_score >= 4:
        return 'SELL'
    
    return 'HOLD'


class WalkForwardSimulator:
    """
    Simulates trading day-by-day with the current algorithm
    """
    
    def __init__(self, initial_capital: float = INITIAL_CAPITAL):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = {}  # ticker -> position info
        self.trades = []
        self.daily_pnl = []
    
    def simulate_day(self, date: datetime, universe_data: Dict[str, pd.DataFrame]):
        """
        Simulate one trading day
        
        Args:
            date: Current date
            universe_data: Dict of ticker -> DataFrame (historical data up to date)
        """
        day_pnl = 0
        
        # Close existing positions if signal changed
        for ticker, position in list(self.positions.items()):
            if ticker not in universe_data:
                continue
            
            df = universe_data[ticker]
            current_price = df['Close'].iloc[-1]
            
            # Check if we should exit
            if position['days_held'] >= 5:  # Max hold
                self.close_position(ticker, current_price, date, 'time_exit')
            else:
                # Update position
                position['days_held'] += 1
                position['current_price'] = current_price
        
        # Look for new entries
        if len(self.positions) < MAX_POSITIONS:
            signals = []
            
            for ticker, df in universe_data.items():
                if ticker in self.positions:
                    continue
                
                signal = generate_signals_for_day(df)
                if signal in ['BUY', 'SELL']:
                    current_price = df['Close'].iloc[-1]
                    signals.append({
                        'ticker': ticker,
                        'signal': signal,
                        'price': current_price,
                        'confidence': 75  # Placeholder - could calculate actual
                    })
            
            # Take top signals by confidence
            if signals:
                signals.sort(key=lambda x: x['confidence'], reverse=True)
                
                for sig in signals[:MAX_POSITIONS - len(self.positions)]:
                    self.open_position(
                        sig['ticker'],
                        sig['price'],
                        date,
                        sig['signal']
                    )
        
        # Calculate daily P&L
        portfolio_value = self.calculate_portfolio_value(universe_data)
        self.daily_pnl.append({
            'date': date.strftime('%Y-%m-%d'),
            'portfolio_value': portfolio_value,
            'cash': self.capital,
            'positions': len(self.positions)
        })
    
    def open_position(self, ticker: str, price: float, date: datetime, direction: str):
        """Open a new position"""
        position_size = self.capital * RISK_PER_TRADE
        shares = position_size / price
        
        self.positions[ticker] = {
            'entry_price': price,
            'shares': shares,
            'direction': direction,
            'entry_date': date,
            'days_held': 0,
            'current_price': price
        }
        
        # Reduce available capital
        self.capital -= position_size
    
    def close_position(self, ticker: str, price: float, date: datetime, reason: str):
        """Close an existing position"""
        if ticker not in self.positions:
            return
        
        position = self.positions[ticker]
        
        # Calculate P&L
        if position['direction'] == 'BUY':
            pnl = (price - position['entry_price']) * position['shares']
        else:  # SELL (short)
            pnl = (position['entry_price'] - price) * position['shares']
        
        # Return capital + P&L
        position_value = position['shares'] * position['entry_price'] + pnl
        self.capital += position_value
        
        # Record trade
        self.trades.append({
            'ticker': ticker,
            'direction': position['direction'],
            'entry_price': position['entry_price'],
            'exit_price': price,
            'entry_date': position['entry_date'].strftime('%Y-%m-%d'),
            'exit_date': date.strftime('%Y-%m-%d'),
            'days_held': position['days_held'],
            'pnl': pnl,
            'pnl_pct': (pnl / (position['shares'] * position['entry_price'])) * 100,
            'exit_reason': reason
        })
        
        del self.positions[ticker]
    
    def calculate_portfolio_value(self, universe_data: Dict[str, pd.DataFrame]) -> float:
        """Calculate total portfolio value"""
        total = self.capital
        
        for ticker, position in self.positions.items():
            if ticker in universe_data:
                current_price = universe_data[ticker]['Close'].iloc[-1]
                position_value = position['shares'] * current_price
                total += position_value
        
        return total
    
    def get_results(self) -> Dict:
        """Get simulation results"""
        total_pnl = sum(t['pnl'] for t in self.trades)
        wins = sum(1 for t in self.trades if t['pnl'] > 0)
        
        return {
            'initial_capital': self.initial_capital,
            'final_capital': self.capital + sum(
                p['shares'] * p['current_price'] for p in self.positions.values()
            ),
            'total_trades': len(self.trades),
            'winning_trades': wins,
            'losing_trades': len(self.trades) - wins,
            'win_rate': (wins / len(self.trades) * 100) if self.trades else 0,
            'total_pnl': total_pnl,
            'total_return_pct': (total_pnl / self.initial_capital) * 100,
            'avg_trade_return': np.mean([t['pnl_pct'] for t in self.trades]) if self.trades else 0,
            'trades': self.trades,
            'daily_pnl': self.daily_pnl
        }


def run_walk_forward(days: int = 30, sample_tickers: int = 50):
    """Run walk-forward simulation"""
    print("="*60)
    print(f"Walk-Forward Analysis ({days} days)")
    print("="*60)
    print()
    
    ensure_dirs()
    
    # Load ticker data
    if not TIME_SERIES_DIR.exists():
        print(f"Error: Time series directory not found")
        return None
    
    tickers = [d.name for d in TIME_SERIES_DIR.iterdir() if d.is_dir()]
    tickers.sort()
    
    import random
    random.seed(42)
    selected_tickers = random.sample(tickers, min(sample_tickers, len(tickers)))
    
    print(f"Running simulation on {len(selected_tickers)} tickers...")
    print(f"Simulation period: {days} trading days")
    print()
    
    # Load all data
    universe_data = {}
    for ticker in selected_tickers:
        file_path = TIME_SERIES_DIR / ticker / f"{ticker}_1d.csv"
        if file_path.exists():
            try:
                df = pd.read_csv(file_path)
                df['date'] = pd.to_datetime(df['date'] if 'date' in df.columns else df.iloc[:, 0])
                df = df.sort_values('date')
                
                # Use last N days
                if len(df) > days + 50:
                    df = df.tail(days + 50)
                
                universe_data[ticker] = df
            except Exception as e:
                print(f"  Warning: Could not load {ticker}: {e}")
    
    if len(universe_data) < 10:
        print("Insufficient data for walk-forward analysis")
        return None
    
    # Get common date range
    min_date = max(df['date'].min() for df in universe_data.values())
    max_date = min(df['date'].max() for df in universe_data.values())
    
    trading_days = pd.date_range(start=min_date, end=max_date, freq='B')  # Business days
    
    print(f"Date range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
    print(f"Trading days: {len(trading_days)}")
    print()
    
    # Run simulation
    simulator = WalkForwardSimulator()
    
    for i, date in enumerate(trading_days):
        # Prepare data for each ticker up to current date
        day_data = {}
        for ticker, df in universe_data.items():
            mask = df['date'] <= date
            if mask.sum() >= 50:
                day_data[ticker] = df[mask]
        
        if day_data:
            simulator.simulate_day(date, day_data)
        
        if (i + 1) % 5 == 0:
            print(f"  Progress: Day {i+1}/{len(trading_days)}")
    
    # Get results
    results = simulator.get_results()
    
    # Save results
    output_file = WALKFORWARD_DIR / f"walkforward_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    return results, output_file


def main():
    """Main execution"""
    print("="*60)
    print("SignalsAlpha Walk-Forward Analyzer")
    print("="*60)
    print()
    
    results, output_file = run_walk_forward(days=30, sample_tickers=50)
    
    if results:
        print("\n" + "="*60)
        print("WALK-FORWARD RESULTS")
        print("="*60)
        print(f"Initial Capital: ${results['initial_capital']:,.2f}")
        print(f"Final Capital: ${results['final_capital']:,.2f}")
        print(f"Total Return: {results['total_return_pct']:.2f}%")
        print(f"Total Trades: {results['total_trades']}")
        print(f"Win Rate: {results['win_rate']:.1f}%")
        print(f"Avg Trade Return: {results['avg_trade_return']:.2f}%")
        print()
        print(f"Results saved to: {output_file}")
        print("="*60)
    else:
        print("\n✗ Walk-forward analysis failed")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
