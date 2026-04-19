#!/usr/bin/env python3
"""
================================================================================
STEP 4: Portfolio-Based Backtester v2.0
================================================================================

Realistic portfolio simulation with cash management, position tracking,
and signal-priority entry system.

Author: SignalsAlpha
Version: 2.0
Date: 2026-04-19

================================================================================
KEY FEATURES v2.0
================================================================================

1. REALISTIC CASH MANAGEMENT
   - $100,000 starting cash
   - Max 20% position size ($20,000 target)
   - Partial fills allowed (buy with available cash)
   - Cannot exceed cash balance

2. PORTFOLIO TRACKING
   - Track all open positions
   - Calculate portfolio value daily (cash + positions)
   - Real-time P&L on open positions

3. SIGNAL PRIORITY SYSTEM
   - Rank signals by confidence score (highest first)
   - Enter strongest signals when cash available
   - Skip if insufficient cash for minimum position ($1000)

4. EXIT LOGIC
   - Exit on signal reversal only
   - No take profit % - let winners run
   - No stop loss in v2.0 (focus on signal quality)

5. BANKRUPTCY CHECK
   - Terminate if portfolio value = $0
   - Track drawdown from peak

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
import pickle
import math

# Configuration
DATA_DIR = Path("/home/ubuntu/.openclaw/workspace/trading-signals-data/process_flow_v1/data")
TIME_SERIES_DIR = DATA_DIR / "time_series"
SIGNALS_DIR = DATA_DIR / "signals_timeframe"
BACKTEST_DIR = DATA_DIR / "backtests"
PROGRESS_FILE = BACKTEST_DIR / "backtest_progress_v2.pkl"
INDIVIDUAL_DIR = BACKTEST_DIR / "individual_stocks"

# Backtest parameters
INITIAL_CAPITAL = 100000
MAX_POSITION_PCT = 0.20  # Max 20% per position ($20,000)
MIN_POSITION_SIZE = 1000  # Minimum $1000 to open position
RISK_FREE_RATE = 0.02
BATCH_SIZE = 50  # Process more tickers per batch for v2

# Portfolio state
cash_balance = INITIAL_CAPITAL
positions = {}  # {ticker: {'shares': int, 'entry_price': float, 'entry_date': str, 'cost_basis': float}}
trades = []  # List of completed trades
daily_equity = []  # Daily portfolio value tracking

# Progress tracking
processed_tickers = set()
current_batch = 0


def ensure_dirs():
    """Create necessary directories"""
    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
    INDIVIDUAL_DIR.mkdir(parents=True, exist_ok=True)


def save_progress():
    """Save progress for resume capability"""
    progress = {
        'processed_tickers': processed_tickers,
        'current_batch': current_batch,
        'cash_balance': cash_balance,
        'positions': positions,
        'trades': trades,
        'daily_equity': daily_equity,
        'timestamp': datetime.now().isoformat()
    }
    with open(PROGRESS_FILE, 'wb') as f:
        pickle.dump(progress, f)


def load_progress():
    """Load progress if exists"""
    global processed_tickers, current_batch, cash_balance, positions, trades, daily_equity
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, 'rb') as f:
            progress = pickle.load(f)
        processed_tickers = progress['processed_tickers']
        current_batch = progress['current_batch']
        cash_balance = progress.get('cash_balance', INITIAL_CAPITAL)
        positions = progress.get('positions', {})
        trades = progress.get('trades', [])
        daily_equity = progress.get('daily_equity', [])
        print(f"  Resuming from batch {current_batch}, {len(processed_tickers)} tickers already processed")
        print(f"  Current cash: ${cash_balance:,.2f}, Open positions: {len(positions)}")
        return True
    return False


def clear_progress():
    """Clear progress file"""
    if PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()


def get_signals_for_date(ticker: str, date_str: str) -> Optional[Dict]:
    """Get signal data for a specific ticker and date"""
    signal_file = SIGNALS_DIR / ticker / f"{ticker}_1d.csv"
    if not signal_file.exists():
        return None
    
    try:
        df = pd.read_csv(signal_file)
        if 'Date' not in df.columns:
            return None
        df['Date'] = pd.to_datetime(df['Date'])
        row = df[df['Date'] == date_str]
        if len(row) == 0:
            return None
        return row.iloc[0].to_dict()
    except Exception as e:
        return None


def get_price_for_date(ticker: str, date_str: str, price_type: str = 'Open') -> Optional[float]:
    """Get price data for a specific ticker and date"""
    price_file = TIME_SERIES_DIR / ticker / f"{ticker}_1d.csv"
    if not price_file.exists():
        return None
    
    try:
        df = pd.read_csv(price_file)
        # Normalize column names
        df.columns = [col.capitalize() if col.lower() in ['open', 'high', 'low', 'close', 'volume'] else col for col in df.columns]
        
        if 'Date' not in df.columns:
            return None
        df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_localize(None)
        row = df[df['Date'] == date_str]
        if len(row) == 0:
            return None
        
        price = row[price_type.capitalize()].iloc[0]
        if pd.isna(price) or price <= 0:
            return None
        return float(price)
    except Exception as e:
        return None


def get_portfolio_value(current_date: str) -> float:
    """Calculate total portfolio value (cash + positions)"""
    global cash_balance, positions
    
    total = cash_balance
    for ticker, pos in positions.items():
        current_price = get_price_for_date(ticker, current_date, 'Close')
        if current_price:
            total += pos['shares'] * current_price
    return total


def rank_signals_by_strength(signals: List[Dict]) -> List[Dict]:
    """Rank signals by confidence score (highest first)"""
    # Sort by confidence descending
    return sorted(signals, key=lambda x: x.get('confidence', 0), reverse=True)


def execute_buy(ticker: str, signal_date: str, entry_price: float, cash_available: float) -> Optional[Dict]:
    """
    Execute buy order with available cash (partial fills allowed)
    Returns trade dict if executed, None if skipped
    """
    global cash_balance
    
    # Calculate target position size (up to 20% or available cash)
    target_size = min(cash_available * 0.20, cash_available)
    
    # Skip if below minimum position size
    if target_size < MIN_POSITION_SIZE:
        return None
    
    # Calculate shares (integer only)
    shares = int(target_size / entry_price)
    if shares < 1:
        return None
    
    actual_cost = shares * entry_price
    
    # Execute trade
    cash_balance -= actual_cost
    
    trade = {
        'ticker': ticker,
        'entry_date': signal_date,
        'entry_price': entry_price,
        'shares': shares,
        'cost_basis': actual_cost,
        'exit_date': None,
        'exit_price': None,
        'pnl': None,
        'pnl_pct': None,
        'exit_reason': None
    }
    
    positions[ticker] = {
        'shares': shares,
        'entry_price': entry_price,
        'entry_date': signal_date,
        'cost_basis': actual_cost
    }
    
    return trade


def execute_sell(ticker: str, exit_date: str, exit_price: float, exit_reason: str) -> Optional[Dict]:
    """Execute sell order for held position"""
    global cash_balance, positions
    
    if ticker not in positions:
        return None
    
    pos = positions[ticker]
    proceeds = pos['shares'] * exit_price
    pnl = proceeds - pos['cost_basis']
    pnl_pct = (pnl / pos['cost_basis']) * 100
    
    trade = {
        'ticker': ticker,
        'entry_date': pos['entry_date'],
        'entry_price': pos['entry_price'],
        'shares': pos['shares'],
        'cost_basis': pos['cost_basis'],
        'exit_date': exit_date,
        'exit_price': exit_price,
        'pnl': pnl,
        'pnl_pct': pnl_pct,
        'exit_reason': exit_reason
    }
    
    cash_balance += proceeds
    del positions[ticker]
    trades.append(trade)
    
    return trade


def backtest_ticker(ticker: str) -> List[Dict]:
    """
    Backtest a single ticker - return list of potential signals
    Signals are collected but executed at portfolio level
    """
    signal_file = SIGNALS_DIR / ticker / f"{ticker}_1d.csv"
    if not signal_file.exists():
        return []
    
    try:
        df = pd.read_csv(signal_file)
        if df.empty or 'Date' not in df.columns or 'signal' not in df.columns:
            return []
        
        df['Date'] = pd.to_datetime(df['Date'])
        
        signals = []
        for _, row in df.iterrows():
            signal = row['signal']
            if signal not in ['BUY', 'STRONG_BUY', 'SELL', 'STRONG_SELL']:
                continue
            
            date_str = row['Date'].strftime('%Y-%m-%d')
            
            # Get next day's open for entry
            entry_price = get_price_for_date(ticker, date_str, 'Open')
            if not entry_price:
                continue
            
            # Get confidence score
            confidence = row.get('confidence', 50)
            if pd.isna(confidence):
                confidence = 50
            
            signals.append({
                'ticker': ticker,
                'date': date_str,
                'signal': signal,
                'confidence': float(confidence),
                'entry_price': entry_price
            })
        
        return signals
    except Exception as e:
        return []


def process_day(date_str: str, all_signals: List[Dict]) -> Dict:
    """
    Process a single trading day
    1. Check for exits (signal reversals)
    2. Check for entries (ranked by confidence)
    """
    global cash_balance, positions, trades
    
    day_trades = []
    
    # Step 1: Check for exits (signal reversals)
    tickers_to_exit = []
    for ticker in list(positions.keys()):
        # Find signal for this ticker on this date
        ticker_signals = [s for s in all_signals if s['ticker'] == ticker and s['date'] == date_str]
        if ticker_signals:
            signal = ticker_signals[0]['signal']
            # Exit long positions on SELL signals
            if signal in ['SELL', 'STRONG_SELL']:
                exit_price = get_price_for_date(ticker, date_str, 'Open')
                if exit_price:
                    trade = execute_sell(ticker, date_str, exit_price, 'SIGNAL_REVERSAL')
                    if trade:
                        day_trades.append(trade)
    
    # Step 2: Check for entries (rank by confidence)
    if cash_balance >= MIN_POSITION_SIZE:
        day_signals = [s for s in all_signals if s['date'] == date_str and s['ticker'] not in positions]
        day_signals = rank_signals_by_strength(day_signals)
        
        for signal in day_signals:
            if cash_balance < MIN_POSITION_SIZE:
                break
            
            ticker = signal['ticker']
            if signal['signal'] in ['BUY', 'STRONG_BUY']:
                trade = execute_buy(ticker, date_str, signal['entry_price'], cash_balance)
                if trade:
                    day_trades.append(trade)
    
    # Record daily equity
    portfolio_value = get_portfolio_value(date_str)
    daily_equity.append({
        'date': date_str,
        'cash': cash_balance,
        'positions': len(positions),
        'portfolio_value': portfolio_value
    })
    
    return {
        'date': date_str,
        'trades': day_trades,
        'portfolio_value': portfolio_value,
        'open_positions': len(positions)
    }


def calculate_metrics() -> Dict:
    """Calculate comprehensive performance metrics"""
    global trades, daily_equity
    
    if not trades:
        return {}
    
    closed_trades = [t for t in trades if t['exit_date'] is not None]
    if not closed_trades:
        return {}
    
    winning_trades = [t for t in closed_trades if t['pnl'] > 0]
    losing_trades = [t for t in closed_trades if t['pnl'] <= 0]
    
    total_trades = len(closed_trades)
    winning_count = len(winning_trades)
    losing_count = len(losing_trades)
    win_rate = (winning_count / total_trades * 100) if total_trades > 0 else 0
    
    gross_profits = sum(t['pnl'] for t in winning_trades) if winning_trades else 0
    gross_losses = abs(sum(t['pnl'] for t in losing_trades)) if losing_trades else 0
    profit_factor = gross_profits / gross_losses if gross_losses > 0 else 0
    
    avg_win = np.mean([t['pnl_pct'] for t in winning_trades]) if winning_trades else 0
    avg_loss = np.mean([t['pnl_pct'] for t in losing_trades]) if losing_trades else 0
    
    # Expectancy
    expectancy = (win_rate/100 * avg_win) - ((100-win_rate)/100 * abs(avg_loss))
    
    # Final portfolio value
    final_value = daily_equity[-1]['portfolio_value'] if daily_equity else INITIAL_CAPITAL
    total_return = ((final_value - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
    
    # Max drawdown
    peak = INITIAL_CAPITAL
    max_dd = 0
    for day in daily_equity:
        if day['portfolio_value'] > peak:
            peak = day['portfolio_value']
        dd = (peak - day['portfolio_value']) / peak * 100
        if dd > max_dd:
            max_dd = dd
    
    return {
        'total_trades': total_trades,
        'winning_trades': winning_count,
        'losing_trades': losing_count,
        'win_rate': win_rate,
        'avg_win_pct': avg_win,
        'avg_loss_pct': avg_loss,
        'profit_factor': profit_factor,
        'expectancy': expectancy,
        'total_return_pct': total_return,
        'max_drawdown_pct': max_dd,
        'final_portfolio_value': final_value,
        'cash_remaining': cash_balance,
        'open_positions': len(positions)
    }


def save_individual_stock_results(ticker: str, ticker_trades: List[Dict]):
    """Save individual stock CSV with its trades"""
    if not ticker_trades:
        return
    
    output_file = INDIVIDUAL_DIR / f"{ticker}_trades.csv"
    
    df = pd.DataFrame(ticker_trades)
    df.to_csv(output_file, index=False)


def run_backtest():
    """Run full portfolio backtest"""
    global cash_balance, positions, trades, daily_equity, current_batch, processed_tickers
    
    ensure_dirs()
    
    # Load progress if exists
    if load_progress():
        pass
    else:
        # Fresh start
        cash_balance = INITIAL_CAPITAL
        positions = {}
        trades = []
        daily_equity = []
        processed_tickers = set()
        current_batch = 0
    
    # Get all tickers with signals
    all_tickers = [d.name for d in SIGNALS_DIR.iterdir() if d.is_dir()]
    all_tickers = [t for t in all_tickers if t not in processed_tickers]
    
    if not all_tickers:
        print("No tickers to process!")
        return
    
    total_tickers = len(all_tickers) + len(processed_tickers)
    print(f"\n{'='*70}")
    print("PORTFOLIO BACKTEST v2.0")
    print(f"{'='*70}")
    print(f"Total tickers: {total_tickers}")
    print(f"Already processed: {len(processed_tickers)}")
    print(f"Remaining: {len(all_tickers)}")
    print(f"Starting cash: ${INITIAL_CAPITAL:,.2f}")
    print(f"{'='*70}\n")
    
    # Collect all signals first
    print("Collecting signals from all tickers...")
    all_signals = []
    for i, ticker in enumerate(all_tickers[:BATCH_SIZE]):
        signals = backtest_ticker(ticker)
        all_signals.extend(signals)
        processed_tickers.add(ticker)
    
    if not all_signals:
        print("No signals found!")
        return
    
    # Get unique dates
    dates = sorted(set(s['date'] for s in all_signals))
    print(f"Processing {len(dates)} trading days...\n")
    
    # Process each day
    for date_str in dates[:100]:  # Limit to first 100 days for this batch
        result = process_day(date_str, all_signals)
        
        # Check for bankruptcy
        if result['portfolio_value'] <= 0:
            print(f"\n{'='*70}")
            print("BANKRUPTCY - Portfolio value reached $0")
            print(f"{'='*70}\n")
            break
        
        if result['trades']:
            print(f"{date_str}: ${result['portfolio_value']:,.2f} | "
                  f"{result['open_positions']} positions | "
                  f"{len(result['trades'])} trades")
    
    # Calculate metrics
    metrics = calculate_metrics()
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save report
    report_file = BACKTEST_DIR / f"backtest_v2_report_{timestamp}.json"
    with open(report_file, 'w') as f:
        json.dump({
            'parameters': {
                'initial_capital': INITIAL_CAPITAL,
                'max_position_pct': MAX_POSITION_PCT,
                'min_position_size': MIN_POSITION_SIZE
            },
            'metrics': metrics,
            'tickers_tested': list(processed_tickers),
            'timestamp': datetime.now().isoformat()
        }, f, indent=2, default=str)
    
    # Save trade log
    trade_file = BACKTEST_DIR / f"trade_log_v2_{timestamp}.csv"
    closed_trades = [t for t in trades if t['exit_date'] is not None]
    if closed_trades:
        df = pd.DataFrame(closed_trades)
        df.to_csv(trade_file, index=False)
    
    # Save equity curve
    equity_file = BACKTEST_DIR / f"equity_curve_v2_{timestamp}.csv"
    if daily_equity:
        df = pd.DataFrame(daily_equity)
        df.to_csv(equity_file, index=False)
    
    # Save individual stock results
    print("\nSaving individual stock results...")
    ticker_trade_map = {}
    for trade in trades:
        ticker = trade['ticker']
        if ticker not in ticker_trade_map:
            ticker_trade_map[ticker] = []
        ticker_trade_map[ticker].append(trade)
    
    for ticker, ticker_trades in ticker_trade_map.items():
        save_individual_stock_results(ticker, ticker_trades)
    
    # Print summary
    print(f"\n{'='*70}")
    print("BACKTEST COMPLETE")
    print(f"{'='*70}")
    print(f"Total Trades: {metrics.get('total_trades', 0)}")
    print(f"Win Rate: {metrics.get('win_rate', 0):.1f}%")
    print(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")
    print(f"Total Return: {metrics.get('total_return_pct', 0):.2f}%")
    print(f"Max Drawdown: {metrics.get('max_drawdown_pct', 0):.1f}%")
    print(f"Final Portfolio: ${metrics.get('final_portfolio_value', 0):,.2f}")
    print(f"Open Positions: {metrics.get('open_positions', 0)}")
    print(f"{'='*70}\n")
    
    # Save individual stock results
    print(f"Individual stock files saved to: {INDIVIDUAL_DIR}")
    
    print(f"\nFiles saved:")
    print(f"  Report: {report_file}")
    print(f"  Trades: {trade_file}")
    print(f"  Equity: {equity_file}")
    
    # Clear progress on success
    clear_progress()


if __name__ == "__main__":
    run_backtest()
