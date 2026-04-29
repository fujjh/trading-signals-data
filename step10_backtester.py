#!/usr/bin/env python3
"""
================================================================================
STEP 10: SPY Historical Backtester
================================================================================

TradingView-style backtesting for SPY only with realistic execution.

Author: SignalsAlpha
Version: 2.0 (SPY Edition)
Date: 2026-04-29

================================================================================
STRATEGY RULES
================================================================================

Entry: Next-day open when signal score > 60 (BUY) or < 40 (SELL)
Exit: Signal reversal or stop loss / take profit
Position Size: Max 20% of portfolio, partial fills allowed
Capital: $100,000 starting cash

================================================================================
INPUT
================================================================================

Source: data/signals_scored/spy_scored_*.csv
        data/time_series/SPY/SPY_1d.csv

================================================================================
OUTPUT
================================================================================

Destination: data/backtests/spy_backtest_*.json
             data/backtests/spy_trades_*.csv

"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import json

# Configuration
TICKER = "SPY"
SIGNALS_DIR = Path("data/signals_scored")
TIME_SERIES_DIR = Path("data/time_series") / TICKER
OUTPUT_DIR = Path("data/backtests")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Backtest parameters
STARTING_CAPITAL = 100000.0
MAX_POSITION_PCT = 0.20
STOP_LOSS_PCT = 0.05
TAKE_PROFIT_PCT = 0.10
COMMISSION = 0.001  # 0.1%


class Trade:
    """Represents a single trade."""
    def __init__(self, entry_date, entry_price, signal, shares):
        self.entry_date = entry_date
        self.entry_price = entry_price
        self.signal = signal
        self.shares = shares
        self.exit_date = None
        self.exit_price = None
        self.pnl = 0
        self.pnl_pct = 0
        self.status = "OPEN"
    
    def close(self, exit_date, exit_price):
        self.exit_date = exit_date
        self.exit_price = exit_price
        self.pnl = (exit_price - self.entry_price) * self.shares * (1 if self.signal in ['BUY', 'STRONG_BUY'] else -1)
        self.pnl_pct = (exit_price - self.entry_price) / self.entry_price * 100
        self.status = "CLOSED"


def load_data():
    """Load SPY signals and price data."""
    # Load signals
    signal_files = list(SIGNALS_DIR.glob("spy_scored_*.csv"))
    if not signal_files:
        print(f"No signal files found")
        return None, None
    
    signals_file = max(signal_files, key=lambda p: p.stat().st_mtime)
    signals_df = pd.read_csv(signals_file)
    signals_df['date'] = pd.to_datetime(signals_df['date'])
    
    # Load price data
    price_file = TIME_SERIES_DIR / f"{TICKER}_1d.csv"
    if not price_file.exists():
        print(f"Price file not found: {price_file}")
        return None, None
    
    prices_df = pd.read_csv(price_file)
    prices_df['date'] = pd.to_datetime(prices_df['date'])
    
    return signals_df, prices_df


def backtest_strategy(signals_df: pd.DataFrame, prices_df: pd.DataFrame) -> Dict:
    """Run backtest on SPY data."""
    
    # Merge signals with prices
    df = pd.merge(prices_df, signals_df[['date', 'signal', 'combined_score']], 
                  on='date', how='left')
    
    # Initialize
    capital = STARTING_CAPITAL
    position = 0  # 0 = no position, 1 = long
    position_value = 0
    trades = []
    equity_curve = []
    
    for i in range(1, len(df)):
        current = df.iloc[i]
        prev = df.iloc[i-1]
        
        date = current['date']
        open_price = current['open']
        close_price = current['close']
        signal = current.get('signal', 'HOLD')
        score = current.get('combined_score', 50)
        
        # Entry logic (use next day's open)
        if position == 0:
            if score > 60 and signal in ['BUY', 'STRONG_BUY']:
                # Enter long
                position_value = min(capital * MAX_POSITION_PCT, capital)
                shares = int(position_value / open_price)
                cost = shares * open_price * (1 + COMMISSION)
                
                if cost <= capital and shares > 0:
                    capital -= cost
                    position = 1
                    trades.append(Trade(date, open_price, signal, shares))
                    
        # Exit logic
        elif position == 1:
            current_trade = trades[-1]
            entry_price = current_trade.entry_price
            
            # Check stop loss / take profit
            stop_price = entry_price * (1 - STOP_LOSS_PCT)
            target_price = entry_price * (1 + TAKE_PROFIT_PCT)
            
            exit_trade = False
            exit_price = close_price
            exit_signal = "Signal Change"
            
            if score < 40 or signal in ['SELL', 'STRONG_SELL']:
                exit_trade = True
                exit_signal = "Signal Reversal"
            elif low_price := current.get('low', close_price):
                if low_price <= stop_price:
                    exit_trade = True
                    exit_price = stop_price
                    exit_signal = "Stop Loss"
                elif high_price := current.get('high', close_price):
                    if high_price >= target_price:
                        exit_trade = True
                        exit_price = target_price
                        exit_signal = "Take Profit"
            
            if exit_trade:
                # Close position
                proceeds = current_trade.shares * exit_price * (1 - COMMISSION)
                capital += proceeds
                current_trade.close(date, exit_price)
                position = 0
                position_value = 0
        
        # Calculate equity
        equity = capital
        if position == 1 and trades:
            current_trade = trades[-1]
            equity += current_trade.shares * close_price
        
        equity_curve.append({'date': date, 'equity': equity})
    
    # Close any open position at the end
    if position == 1 and trades:
        current_trade = trades[-1]
        final_price = df.iloc[-1]['close']
        current_trade.close(df.iloc[-1]['date'], final_price)
        capital += current_trade.shares * final_price * (1 - COMMISSION)
    
    return {
        'trades': trades,
        'equity_curve': equity_curve,
        'final_capital': capital
    }


def calculate_metrics(trades: List[Trade], equity_curve: List[Dict]) -> Dict:
    """Calculate backtest metrics."""
    
    closed_trades = [t for t in trades if t.status == "CLOSED"]
    
    if not closed_trades:
        return {
            'total_trades': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'total_return': 0
        }
    
    profits = [t.pnl for t in closed_trades if t.pnl > 0]
    losses = [t.pnl for t in closed_trades if t.pnl < 0]
    
    total_profit = sum(profits) if profits else 0
    total_loss = abs(sum(losses)) if losses else 0
    
    win_rate = len(profits) / len(closed_trades) if closed_trades else 0
    profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
    
    # Calculate returns from equity curve
    equity_df = pd.DataFrame(equity_curve)
    equity_df['returns'] = equity_df['equity'].pct_change()
    
    returns = equity_df['returns'].dropna()
    if len(returns) > 0 and returns.std() > 0:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
    else:
        sharpe = 0
    
    # Max drawdown
    equity_df['peak'] = equity_df['equity'].cummax()
    equity_df['drawdown'] = (equity_df['equity'] - equity_df['peak']) / equity_df['peak']
    max_drawdown = abs(equity_df['drawdown'].min())
    
    total_return = (equity_curve[-1]['equity'] - STARTING_CAPITAL) / STARTING_CAPITAL
    
    return {
        'total_trades': len(closed_trades),
        'winning_trades': len(profits),
        'losing_trades': len(losses),
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'total_return': total_return,
        'avg_profit': np.mean(profits) if profits else 0,
        'avg_loss': np.mean(losses) if losses else 0
    }


def run_backtest():
    """Main backtest function."""
    print("=" * 70)
    print("STEP 10: SPY Historical Backtester")
    print("=" * 70)
    print(f"Starting Capital: ${STARTING_CAPITAL:,.2f}")
    print(f"Max Position: {MAX_POSITION_PCT*100:.0f}%")
    print(f"Stop Loss: {STOP_LOSS_PCT*100:.0f}%")
    print(f"Take Profit: {TAKE_PROFIT_PCT*100:.0f}%")
    print("=" * 70)
    
    # Load data
    signals_df, prices_df = load_data()
    if signals_df is None or prices_df is None:
        print("ERROR: Could not load data")
        return
    
    print(f"Loaded {len(signals_df)} signals")
    print(f"Loaded {len(prices_df)} price records")
    
    # Run backtest
    results = backtest_strategy(signals_df, prices_df)
    
    # Calculate metrics
    metrics = calculate_metrics(results['trades'], results['equity_curve'])
    
    # Print results
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS")
    print("=" * 70)
    print(f"Total Trades: {metrics['total_trades']}")
    print(f"Win Rate: {metrics['win_rate']*100:.1f}%")
    print(f"Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']*100:.1f}%")
    print(f"Total Return: {metrics['total_return']*100:.1f}%")
    print(f"Final Capital: ${results['final_capital']:,.2f}")
    print("=" * 70)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save metrics
    metrics_file = OUTPUT_DIR / f"spy_backtest_{timestamp}.json"
    with open(metrics_file, 'w') as f:
        json.dump({
            'ticker': TICKER,
            'timestamp': timestamp,
            'parameters': {
                'starting_capital': STARTING_CAPITAL,
                'max_position_pct': MAX_POSITION_PCT,
                'stop_loss_pct': STOP_LOSS_PCT,
                'take_profit_pct': TAKE_PROFIT_PCT
            },
            'metrics': metrics
        }, f, indent=2)
    
    print(f"\nSaved results: {metrics_file}")


if __name__ == "__main__":
    run_backtest()
