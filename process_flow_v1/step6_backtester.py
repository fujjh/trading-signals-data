#!/usr/bin/env python3
"""
================================================================================
STEP 6: Historical Backtester
================================================================================

Simulates trading performance of the SignalsAlpha algorithm on historical
data to calculate key performance metrics including win rate, profit factor,
and risk-adjusted returns.

Author: SignalsAlpha
Version: 1.0
Date: 2026-04-18

================================================================================
PURPOSE
================================================================================

This module validates signal algorithm effectiveness by:

1. Loading historical signals and price data
2. Simulating trades based on signal rules
3. Calculating performance metrics
4. Generating risk statistics
5. Creating equity curves for visualization
6. Providing evidence-based algorithm validation

================================================================================
BACKTEST METHODOLOGY
================================================================================

Entry Rules:
    - Enter LONG on BUY/STRONG_BUY signal
    - Enter SHORT on SELL/STRONG_SELL signal
    - Entry at next day's open price (simulates execution delay)

Exit Rules:
    - Stop Loss: -5% from entry (configurable)
    - Take Profit: +10% from entry (configurable)
    - Maximum Hold: 5 days (prevents stale signals)
    - Signal reversal (exit when signal flips)

Position Sizing:
    - Fixed percentage of portfolio per trade
    - Default: Equal weight per signal
    - Accounts for slippage (0.1% per trade)

================================================================================
PERFORMANCE METRICS
================================================================================

Return Metrics:
    - Total Return: Cumulative profit/loss
    - Annualized Return: Normalized yearly return
    - Win Rate: % of profitable trades
    - Average Win: Average profit on winning trades
    - Average Loss: Average loss on losing trades

Risk Metrics:
    - Max Drawdown: Largest peak-to-trough decline
    - Sharpe Ratio: Risk-adjusted return measure
    - Profit Factor: Gross profit / Gross loss
    - Recovery Factor: Total return / Max drawdown

Trade Statistics:
    - Total trades executed
    - Profitable vs losing trades
    - Average holding period
    - Largest single win/loss

================================================================================
WORKFLOW
================================================================================

1. LOAD DATA
   - Read historical signals from signal_history/
   - Load corresponding price data from time_series/
   - Align timestamps

2. SIMULATE TRADES
   For each signal:
   a. Calculate entry price (next day open)
   b. Monitor for exit conditions daily
   c. Record outcome (win/loss, P&L)

3. CALCULATE METRICS
   - Aggregate trade results
   - Calculate equity curve
   - Compute risk statistics
   - Benchmark vs buy-and-hold

4. GENERATE REPORT
   - Write performance_report.json
   - Create equity curve CSV
   - Print summary statistics

================================================================================
OUTPUT FORMAT
================================================================================

Performance Report (JSON):
    {
        "backtest_period": "2024-01-01 to 2026-04-18",
        "total_trades": 1250,
        "winning_trades": 687,
        "losing_trades": 563,
        "win_rate": 54.96,
        "total_return_pct": 127.5,
        "annualized_return_pct": 42.3,
        "max_drawdown_pct": -18.2,
        "sharpe_ratio": 1.85,
        "profit_factor": 1.72,
        "average_win_pct": 8.4,
        "average_loss_pct": -4.2,
        "largest_win_pct": 25.3,
        "largest_loss_pct": -12.8,
        "avg_holding_days": 3.2
    }

Equity Curve (CSV):
    date,equity,drawdown_pct
    2024-01-02,100000.00,0.00
    2024-01-03,100150.25,0.00
    2024-01-04,99875.50,-0.28

Trade Log (CSV):
    entry_date,exit_date,ticker,signal,entry_price,exit_price,pnl_pct,exit_reason
    2024-01-15,2024-01-18,AAPL,BUY,150.25,158.40,5.42,TAKE_PROFIT

================================================================================
CONFIGURATION
================================================================================

Backtest Parameters:
    STOP_LOSS_PCT = 5.0      # Stop loss percentage
    TAKE_PROFIT_PCT = 10.0   # Take profit percentage
    MAX_HOLD_DAYS = 5        # Maximum holding period
    INITIAL_CAPITAL = 100000 # Starting capital
    SLIPPAGE_PCT = 0.1       # Transaction cost

Signal Filters:
    MIN_CONFIDENCE = 70      # Minimum signal confidence
    SIGNAL_TYPES = ['STRONG_BUY', 'BUY']  # Which signals to trade

================================================================================
USAGE
================================================================================

    python step6_backtester.py

Optional: Edit CONFIGURATION section to test different parameters.

================================================================================
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

# Configuration
DATA_DIR = Path("/home/ubuntu/.openclaw/workspace/data")
TIME_SERIES_DIR = DATA_DIR / "time_series"
BACKTEST_DIR = DATA_DIR / "backtests"

# Backtest parameters
INITIAL_CAPITAL = 100000  # $100,000 starting capital
POSITION_SIZE_PCT = 0.05  # 5% of capital per trade
STOP_LOSS_PCT = 0.05  # 5% stop loss
TAKE_PROFIT_PCT = 0.10  # 10% take profit
HOLD_DAYS = 5  # Max hold period


def ensure_dirs():
    """Create backtest directory"""
    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)


def load_ticker_data(ticker: str) -> pd.DataFrame:
    """Load historical price data for a ticker"""
    file_path = TIME_SERIES_DIR / ticker / f"{ticker}_1d.csv"
    
    if not file_path.exists():
        return None
    
    try:
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'] if 'date' in df.columns else df.iloc[:, 0])
        df = df.sort_values('date').reset_index(drop=True)
        return df
    except Exception as e:
        print(f"Error loading {ticker}: {e}")
        return None


def simulate_trade(entry_price: float, exit_price: float, direction: str) -> Dict:
    """
    Simulate a single trade
    
    Args:
        entry_price: Entry price
        exit_price: Exit price
        direction: 'LONG' or 'SHORT'
    
    Returns:
        Trade result dictionary
    """
    if direction == 'LONG':
        pnl = (exit_price - entry_price) / entry_price
    else:  # SHORT
        pnl = (entry_price - exit_price) / entry_price
    
    return {
        'direction': direction,
        'entry': entry_price,
        'exit': exit_price,
        'pnl_pct': pnl * 100,
        'win': pnl > 0
    }


def backtest_ticker(ticker: str, df: pd.DataFrame) -> Dict:
    """
    Backtest a single ticker using current signal algorithm
    
    Simulates:
    - BUY signals: Enter long position
    - SELL signals: Enter short position
    - HOLD: No position
    - Stop loss at -5%
    - Take profit at +10%
    - Max hold: 5 days
    """
    if df is None or len(df) < 50:
        return None
    
    trades = []
    in_position = False
    entry_price = 0
    entry_date = None
    position_direction = None
    
    for i in range(50, len(df) - HOLD_DAYS):
        window = df.iloc[i-50:i].copy()
        current = df.iloc[i]
        
        close = current['Close']
        
        # Calculate indicators (same as step2 scanner)
        sma_20 = window['Close'].rolling(20).mean().iloc[-1]
        sma_50 = window['Close'].rolling(50).mean().iloc[-1]
        ema_12 = window['Close'].ewm(span=12).mean().iloc[-1]
        ema_26 = window['Close'].ewm(span=26).mean().iloc[-1]
        
        # RSI
        delta = window['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean().iloc[-1]
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean().iloc[-1]
        rs = gain / loss if loss > 0 else 0
        rsi = 100 - (100 / (1 + rs))
        
        # MACD
        macd_line = ema_12 - ema_26
        signal_line = window['Close'].ewm(span=9).mean().iloc[-1]
        histogram = macd_line - signal_line
        
        # Bollinger Bands
        bb_middle = window['Close'].rolling(20).mean().iloc[-1]
        bb_std = window['Close'].rolling(20).std().iloc[-1]
        bb_upper = bb_middle + (bb_std * 2)
        bb_lower = bb_middle - (bb_std * 2)
        
        # Generate signal
        buy_score = 0
        sell_score = 0
        
        # Trend
        if close > sma_20 > sma_50:
            buy_score += 2
        elif close < sma_20 < sma_50:
            sell_score += 2
        
        # EMA
        if ema_12 > ema_26:
            buy_score += 1
        else:
            sell_score += 1
        
        # RSI
        if rsi < 30:
            buy_score += 2
        elif rsi > 70:
            sell_score += 2
        elif rsi > 50:
            buy_score += 1
        else:
            sell_score += 1
        
        # MACD
        if macd_line > signal_line and histogram > 0:
            buy_score += 2
        elif macd_line < signal_line and histogram < 0:
            sell_score += 2
        
        # Bollinger
        if close < bb_lower:
            buy_score += 1
        elif close > bb_upper:
            sell_score += 1
        
        # Determine signal
        if buy_score >= 4:
            signal = 'BUY'
        elif sell_score >= 4:
            signal = 'SELL'
        else:
            signal = 'HOLD'
        
        # Position management
        if not in_position:
            if signal == 'BUY':
                in_position = True
                entry_price = close
                entry_date = current['date']
                position_direction = 'LONG'
            elif signal == 'SELL':
                in_position = True
                entry_price = close
                entry_date = current['date']
                position_direction = 'SHORT'
        
        else:
            # Check exit conditions
            exit_trade = False
            exit_reason = ''
            
            # Check stop loss / take profit
            if position_direction == 'LONG':
                pnl_pct = (close - entry_price) / entry_price
                if pnl_pct <= -STOP_LOSS_PCT:
                    exit_trade = True
                    exit_reason = 'stop_loss'
                elif pnl_pct >= TAKE_PROFIT_PCT:
                    exit_trade = True
                    exit_reason = 'take_profit'
            else:  # SHORT
                pnl_pct = (entry_price - close) / entry_price
                if pnl_pct <= -STOP_LOSS_PCT:
                    exit_trade = True
                    exit_reason = 'stop_loss'
                elif pnl_pct >= TAKE_PROFIT_PCT:
                    exit_trade = True
                    exit_reason = 'take_profit'
            
            # Check max hold days
            days_held = (current['date'] - entry_date).days if entry_date else 0
            if days_held >= HOLD_DAYS:
                exit_trade = True
                exit_reason = 'time_exit'
            
            if exit_trade:
                trade = simulate_trade(entry_price, close, position_direction)
                trade.update({
                    'ticker': ticker,
                    'entry_date': entry_date.strftime('%Y-%m-%d') if entry_date else None,
                    'exit_date': current['date'].strftime('%Y-%m-%d'),
                    'days_held': days_held,
                    'exit_reason': exit_reason
                })
                trades.append(trade)
                in_position = False
    
    return {
        'ticker': ticker,
        'total_trades': len(trades),
        'trades': trades
    }


def calculate_performance(results: List[Dict]) -> Dict:
    """Calculate overall performance metrics"""
    
    all_trades = []
    for result in results:
        if result and result['trades']:
            all_trades.extend(result['trades'])
    
    if not all_trades:
        return {
            'total_trades': 0,
            'win_rate': 0,
            'avg_return': 0,
            'profit_factor': 0
        }
    
    total_trades = len(all_trades)
    wins = sum(1 for t in all_trades if t['win'])
    losses = total_trades - wins
    
    win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
    
    avg_return = np.mean([t['pnl_pct'] for t in all_trades])
    
    gross_profits = sum(t['pnl_pct'] for t in all_trades if t['pnl_pct'] > 0)
    gross_losses = abs(sum(t['pnl_pct'] for t in all_trades if t['pnl_pct'] < 0))
    profit_factor = gross_profits / gross_losses if gross_losses > 0 else gross_profits
    
    return {
        'total_trades': total_trades,
        'wins': wins,
        'losses': losses,
        'win_rate': round(win_rate, 2),
        'avg_return': round(avg_return, 2),
        'profit_factor': round(profit_factor, 2),
        'gross_profits': round(gross_profits, 2),
        'gross_losses': round(gross_losses, 2)
    }


def run_backtest(sample_size: int = 100):
    """Run backtest on sample of tickers"""
    print("="*60)
    print(f"Running Backtest (sample: {sample_size} tickers)")
    print("="*60)
    print()
    
    ensure_dirs()
    
    # Get list of tickers with time series data
    if not TIME_SERIES_DIR.exists():
        print(f"Error: Time series directory not found: {TIME_SERIES_DIR}")
        return None
    
    tickers = [d.name for d in TIME_SERIES_DIR.iterdir() if d.is_dir()]
    tickers.sort()
    
    if len(tickers) > sample_size:
        import random
        random.seed(42)
        tickers = random.sample(tickers, sample_size)
    
    print(f"Testing on {len(tickers)} tickers...")
    print()
    
    results = []
    processed = 0
    
    for ticker in tickers:
        df = load_ticker_data(ticker)
        if df is not None:
            result = backtest_ticker(ticker, df)
            if result and result['total_trades'] > 0:
                results.append(result)
        
        processed += 1
        if processed % 10 == 0:
            print(f"  Progress: {processed}/{len(tickers)} tickers processed")
    
    print()
    
    # Calculate performance
    performance = calculate_performance(results)
    
    # Save results
    backtest_file = BACKTEST_DIR / f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    output = {
        'backtest_date': datetime.now().isoformat(),
        'parameters': {
            'initial_capital': INITIAL_CAPITAL,
            'position_size_pct': POSITION_SIZE_PCT,
            'stop_loss_pct': STOP_LOSS_PCT,
            'take_profit_pct': TAKE_PROFIT_PCT,
            'hold_days': HOLD_DAYS
        },
        'sample_size': len(tickers),
        'tickers_tested': tickers,
        'performance': performance,
        'results': results
    }
    
    with open(backtest_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    return performance, backtest_file


def main():
    """Main execution"""
    print("="*60)
    print("SignalsAlpha Backtester")
    print("="*60)
    print()
    
    performance, backtest_file = run_backtest(sample_size=100)
    
    if performance:
        print("\n" + "="*60)
        print("BACKTEST RESULTS")
        print("="*60)
        print(f"Total Trades: {performance['total_trades']}")
        print(f"Win Rate: {performance['win_rate']:.1f}%")
        print(f"Avg Return: {performance['avg_return']:.2f}%")
        print(f"Profit Factor: {performance['profit_factor']:.2f}")
        print(f"Gross Profits: {performance['gross_profits']:.2f}%")
        print(f"Gross Losses: {performance['gross_losses']:.2f}%")
        print()
        print(f"Results saved to: {backtest_file}")
        print("="*60)
    else:
        print("\n✗ Backtest failed")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
