#!/usr/bin/env python3
"""
================================================================================
STEP 10: Enhanced Historical Backtester v2.0
================================================================================

TradingView-style backtester with realistic execution, comprehensive metrics,
and batch processing for OCI compatibility.

Author: SignalsAlpha
Version: 2.0
Date: 2026-04-19

================================================================================
KEY IMPROVEMENTS v2.0
================================================================================

1. LOOKAHEAD BIAS FIX
   - Entry at next day's open (not current close)
   - Realistic execution delay simulation

2. SIGNAL REVERSAL EXITS
   - Exit when signal flips direction
   - Long position exits on SELL signal
   - Short position exits on BUY signal

3. COMPREHENSIVE METRICS
   - Expectancy: (Win% × Avg Win) - (Loss% × Avg Loss)
   - Sharpe Ratio: Risk-adjusted returns
   - Sortino Ratio: Downside risk only
   - Profit Factor: Gross Profit / Gross Loss
   - Calmar Ratio: Return / Max Drawdown
   - R-Multiple: Reward-to-risk per trade

4. EQUITY CURVE & DRAWDOWN
   - Daily equity tracking
   - Running drawdown calculation
   - Underwater plot data

5. REPORTING FEATURES
   - Trade log CSV with detailed entries
   - Monthly performance breakdown
   - Exit reason analysis
   - Consecutive win/loss tracking

6. BATCH PROCESSING
   - 20 tickers per batch
   - Auto-restart with progress save
   - Resume capability for OCI timeouts

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

# Configuration
DATA_DIR = Path("/home/ubuntu/.openclaw/workspace/trading-signals-data/process_flow_v1/data")
TIME_SERIES_DIR = DATA_DIR / "time_series"
BACKTEST_DIR = DATA_DIR / "backtests"
PROGRESS_FILE = BACKTEST_DIR / "backtest_progress.pkl"

# Backtest parameters
INITIAL_CAPITAL = 100000
POSITION_SIZE_PCT = 0.05
STOP_LOSS_PCT = 0.05
TAKE_PROFIT_PCT = 0.10
HOLD_DAYS = 5
BATCH_SIZE = 20  # Process 20 tickers per batch
RISK_FREE_RATE = 0.02  # 2% annual for Sharpe calculation

# Progress tracking
processed_tickers = set()
current_batch = 0
all_results = []


def ensure_dirs():
    """Create backtest directory"""
    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)


def save_progress():
    """Save progress for resume capability"""
    progress = {
        'processed_tickers': processed_tickers,
        'current_batch': current_batch,
        'all_results': all_results,
        'timestamp': datetime.now().isoformat()
    }
    with open(PROGRESS_FILE, 'wb') as f:
        pickle.dump(progress, f)


def load_progress() -> bool:
    """Load progress if exists"""
    global processed_tickers, current_batch, all_results
    if PROGRESS_FILE.exists():
        try:
            with open(PROGRESS_FILE, 'rb') as f:
                progress = pickle.load(f)
            processed_tickers = progress['processed_tickers']
            current_batch = progress['current_batch']
            all_results = progress['all_results']
            print(f"  Resuming from batch {current_batch}, {len(processed_tickers)} tickers already processed")
            return True
        except Exception as e:
            print(f"  Warning: Could not load progress: {e}")
    return False


def clear_progress():
    """Clear progress file after successful completion"""
    if PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()


def load_ticker_data(ticker: str) -> Optional[pd.DataFrame]:
    """Load historical price data for a ticker"""
    file_path = TIME_SERIES_DIR / ticker / f"{ticker}_1d.csv"
    
    if not file_path.exists():
        return None
    
    try:
        df = pd.read_csv(file_path)
        
        # Normalize column names (handle both uppercase and lowercase)
        df.columns = [c.lower() for c in df.columns]
        
        # Parse date
        df['date'] = pd.to_datetime(df['date'], utc=True).dt.tz_localize(None)
        
        # Rename to capitalized for consistency
        df = df.rename(columns={
            'open': 'Open',
            'high': 'High', 
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })
        
        df = df.sort_values('date').reset_index(drop=True)
        
        # Ensure required columns exist
        required = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required:
            if col not in df.columns:
                # Try lowercase
                lower_col = col.lower()
                if lower_col in df.columns:
                    df[col] = df[lower_col]
        
        return df
    except Exception as e:
        print(f"  Error loading {ticker}: {e}")
        return None


def calculate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate trading signals using same logic as Step 2 scanner
    Returns dataframe with signal column added
    """
    df = df.copy()
    
    # Calculate indicators
    df['sma_20'] = df['Close'].rolling(window=20).mean()
    df['sma_50'] = df['Close'].rolling(window=50).mean()
    df['ema_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    df['macd_line'] = df['ema_12'] - df['ema_26']
    df['signal_line'] = df['macd_line'].ewm(span=9, adjust=False).mean()
    df['macd_histogram'] = df['macd_line'] - df['signal_line']
    
    # Bollinger Bands
    df['bb_middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    
    # Calculate signals
    signals = []
    
    for i in range(len(df)):
        if i < 50:  # Need enough data for indicators
            signals.append('HOLD')
            continue
        
        close = df['Close'].iloc[i]
        sma_20 = df['sma_20'].iloc[i]
        sma_50 = df['sma_50'].iloc[i]
        ema_12 = df['ema_12'].iloc[i]
        ema_26 = df['ema_26'].iloc[i]
        rsi = df['rsi'].iloc[i]
        macd_line = df['macd_line'].iloc[i]
        signal_line_val = df['signal_line'].iloc[i]
        histogram = df['macd_histogram'].iloc[i]
        bb_upper = df['bb_upper'].iloc[i]
        bb_lower = df['bb_lower'].iloc[i]
        
        # Skip if NaN values
        if pd.isna(sma_20) or pd.isna(rsi):
            signals.append('HOLD')
            continue
        
        buy_score = 0
        sell_score = 0
        
        # Trend (SMA alignment)
        if close > sma_20 > sma_50:
            buy_score += 2
        elif close < sma_20 < sma_50:
            sell_score += 2
        
        # EMA crossover
        if ema_12 > ema_26:
            buy_score += 1
        else:
            sell_score += 1
        
        # RSI levels
        if rsi < 30:
            buy_score += 2
        elif rsi > 70:
            sell_score += 2
        elif rsi > 50:
            buy_score += 1
        elif rsi < 50:
            sell_score += 1
        
        # MACD
        if macd_line > signal_line_val and histogram > 0:
            buy_score += 2
        elif macd_line < signal_line_val and histogram < 0:
            sell_score += 2
        
        # Bollinger Bands
        if close < bb_lower:
            buy_score += 1
        elif close > bb_upper:
            sell_score += 1
        
        # Determine signal
        if buy_score >= 4:
            signals.append('BUY')
        elif sell_score >= 4:
            signals.append('SELL')
        else:
            signals.append('HOLD')
    
    df['signal'] = signals
    return df


def backtest_ticker(ticker: str, df: pd.DataFrame) -> Dict:
    """
    Backtest a single ticker with realistic execution
    
    IMPROVEMENTS v2.0:
    - Entry at NEXT day's open (fixes lookahead bias)
    - Signal reversal exits
    - Comprehensive trade tracking
    """
    if df is None or len(df) < 60:
        return {'ticker': ticker, 'total_trades': 0, 'trades': []}
    
    # Calculate signals
    df = calculate_signals(df)
    
    trades = []
    in_position = False
    entry_price = 0
    entry_date = None
    position_direction = None
    entry_signal = None
    
    # Iterate through data (leave room for next day's open)
    for i in range(50, len(df) - 1):
        current = df.iloc[i]
        next_day = df.iloc[i + 1]  # For entry execution
        
        signal = current['signal']
        
        # Position management
        if not in_position:
            # ENTRY LOGIC - Execute at NEXT day's open
            if signal == 'BUY':
                # Validate price is valid
                if next_day['Open'] <= 0 or pd.isna(next_day['Open']):
                    continue
                in_position = True
                entry_price = float(next_day['Open'])  # LOOKAHEAD BIAS FIX
                entry_date = next_day['date']
                position_direction = 'LONG'
                entry_signal = 'BUY'
                
            elif signal == 'SELL':
                # Validate price is valid
                if next_day['Open'] <= 0 or pd.isna(next_day['Open']):
                    continue
                in_position = True
                entry_price = float(next_day['Open'])  # LOOKAHEAD BIAS FIX
                entry_date = next_day['date']
                position_direction = 'SHORT'
                entry_signal = 'SELL'
        
        else:
            # EXIT LOGIC - Skip if no valid entry
            if entry_price <= 0:
                continue
            
            current_price = current['Close']
            
            # Skip if current price is invalid
            if current_price <= 0 or pd.isna(current_price):
                continue
                
            exit_trade = False
            exit_reason = ''
            
            # Calculate current P&L
            if position_direction == 'LONG':
                pnl_pct = (current_price - entry_price) / entry_price
            else:  # SHORT
                pnl_pct = (entry_price - current_price) / entry_price
            
            # Check stop loss
            if pnl_pct <= -STOP_LOSS_PCT:
                exit_trade = True
                exit_reason = 'STOP_LOSS'
            
            # Check take profit
            elif pnl_pct >= TAKE_PROFIT_PCT:
                exit_trade = True
                exit_reason = 'TAKE_PROFIT'
            
            # Check signal reversal (NEW v2.0)
            elif position_direction == 'LONG' and signal == 'SELL':
                exit_trade = True
                exit_reason = 'SIGNAL_REVERSAL'
            elif position_direction == 'SHORT' and signal == 'BUY':
                exit_trade = True
                exit_reason = 'SIGNAL_REVERSAL'
            
            # Check max hold days
            days_held = (current['date'] - entry_date).days
            if days_held >= HOLD_DAYS:
                exit_trade = True
                exit_reason = 'TIME_EXIT'
            
            if exit_trade:
                # Record trade with native Python types (not numpy types)
                trade = {
                    'ticker': ticker,
                    'entry_date': entry_date.strftime('%Y-%m-%d'),
                    'exit_date': current['date'].strftime('%Y-%m-%d'),
                    'direction': position_direction,
                    'entry_price': float(round(entry_price, 2)),
                    'exit_price': float(round(current_price, 2)),
                    'pnl_pct': float(round(pnl_pct * 100, 2)),
                    'win': bool(pnl_pct > 0),
                    'days_held': int(days_held),
                    'exit_reason': exit_reason,
                    'entry_signal': entry_signal
                }
                trades.append(trade)
                in_position = False
                entry_price = 0
                entry_date = None
                position_direction = None
                entry_signal = None
    
    return {
        'ticker': ticker,
        'total_trades': len(trades),
        'trades': trades
    }


def calculate_comprehensive_metrics(results: List[Dict], all_trades: List[Dict]) -> Tuple[Dict, pd.DataFrame]:
    """
    Calculate comprehensive trading metrics
    
    Returns:
        metrics: Dictionary of performance statistics
        equity_curve: DataFrame with daily equity values
    """
    if not all_trades:
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'loss_rate': 0.0,
            'avg_return': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0,
            'profit_factor': 0.0,
            'gross_profits': 0.0,
            'gross_losses': 0.0,
            'expectancy': 0.0,
            'r_multiple_avg': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'calmar_ratio': 0.0,
            'max_drawdown': 0.0,
            'avg_holding_days': 0.0,
            'consecutive_wins_max': 0,
            'consecutive_losses_max': 0
        }, pd.DataFrame()
    
    # Basic counts
    total_trades = len(all_trades)
    winning_trades = [t for t in all_trades if t['win']]
    losing_trades = [t for t in all_trades if not t['win']]
    
    wins = len(winning_trades)
    losses = len(losing_trades)
    
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    loss_rate = (losses / total_trades * 100) if total_trades > 0 else 0
    
    # P&L statistics - filter out inf and nan values
    import math
    pnl_values = [t['pnl_pct'] for t in all_trades 
                  if not math.isinf(t['pnl_pct']) and not math.isnan(t['pnl_pct'])]
    
    if not pnl_values:
        pnl_values = [0.0]
    
    avg_return = np.mean(pnl_values)
    
    win_pnl = [t['pnl_pct'] for t in winning_trades 
               if not math.isinf(t['pnl_pct']) and not math.isnan(t['pnl_pct'])]
    loss_pnl = [t['pnl_pct'] for t in losing_trades 
                if not math.isinf(t['pnl_pct']) and not math.isnan(t['pnl_pct'])]
    
    avg_win = np.mean(win_pnl) if win_pnl else 0
    avg_loss = np.mean(loss_pnl) if loss_pnl else 0
    
    largest_win = max(win_pnl) if win_pnl else 0
    largest_loss = min(loss_pnl) if loss_pnl else 0
    
    gross_profits = sum(win_pnl) if win_pnl else 0
    gross_losses = abs(sum(loss_pnl)) if loss_pnl else 0
    profit_factor = gross_profits / gross_losses if gross_losses > 0 else gross_profits
    
    # EXPECTANCY (Win% × Avg Win) - (Loss% × |Avg Loss|)
    expectancy = (win_rate/100 * avg_win) - (loss_rate/100 * abs(avg_loss))
    
    # R-MULTIPLE (Avg Win / |Avg Loss|)
    r_multiple_avg = avg_win / abs(avg_loss) if avg_loss != 0 else 0
    
    # Holding period
    avg_holding_days = np.mean([t['days_held'] for t in all_trades])
    
    # Consecutive wins/losses
    consecutive_wins_max = 0
    consecutive_losses_max = 0
    current_streak = 0
    current_type = None
    
    for trade in all_trades:
        if trade['win']:
            if current_type == 'win':
                current_streak += 1
            else:
                current_streak = 1
                current_type = 'win'
            consecutive_wins_max = max(consecutive_wins_max, current_streak)
        else:
            if current_type == 'loss':
                current_streak += 1
            else:
                current_streak = 1
                current_type = 'loss'
            consecutive_losses_max = max(consecutive_losses_max, current_streak)
    
    # Build equity curve for Sharpe/Sortino/Drawdown
    equity_curve = build_equity_curve(all_trades)
    
    # Calculate Sharpe and Sortino ratios
    sharpe_ratio = 0.0
    sortino_ratio = 0.0
    calmar_ratio = 0.0
    max_drawdown = 0.0
    
    if len(equity_curve) > 1:
        returns = equity_curve['daily_return'].dropna()
        
        if len(returns) > 0 and returns.std() > 0:
            # Annualize returns (252 trading days)
            avg_return_daily = returns.mean()
            std_return_daily = returns.std()
            
            # Sharpe Ratio
            sharpe_ratio = ((avg_return_daily * 252) - RISK_FREE_RATE) / (std_return_daily * np.sqrt(252))
            
            # Sortino Ratio (downside deviation only)
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0:
                downside_std = downside_returns.std() * np.sqrt(252)
                if downside_std > 0:
                    sortino_ratio = ((avg_return_daily * 252) - RISK_FREE_RATE) / downside_std
        
        # Max Drawdown
        equity_values = equity_curve['equity'].values
        peak = np.maximum.accumulate(equity_values)
        drawdown = (equity_values - peak) / peak * 100
        max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0
        
        # Calmar Ratio (Annual Return / Max Drawdown)
        if max_drawdown > 0:
            total_return = (equity_values[-1] - equity_values[0]) / equity_values[0] * 100
            years = len(equity_curve) / 252
            annual_return = total_return / years if years > 0 else 0
            calmar_ratio = annual_return / max_drawdown
    
    metrics = {
        'total_trades': total_trades,
        'winning_trades': wins,
        'losing_trades': losses,
        'win_rate': round(win_rate, 2),
        'loss_rate': round(loss_rate, 2),
        'avg_return': round(avg_return, 2),
        'avg_win': round(avg_win, 2),
        'avg_loss': round(avg_loss, 2),
        'largest_win': round(largest_win, 2),
        'largest_loss': round(largest_loss, 2),
        'profit_factor': round(profit_factor, 2),
        'gross_profits': round(gross_profits, 2),
        'gross_losses': round(gross_losses, 2),
        'expectancy': round(expectancy, 2),
        'r_multiple_avg': round(r_multiple_avg, 2),
        'sharpe_ratio': round(sharpe_ratio, 2),
        'sortino_ratio': round(sortino_ratio, 2),
        'calmar_ratio': round(calmar_ratio, 2),
        'max_drawdown': round(max_drawdown, 2),
        'avg_holding_days': round(avg_holding_days, 1),
        'consecutive_wins_max': consecutive_wins_max,
        'consecutive_losses_max': consecutive_losses_max
    }
    
    return metrics, equity_curve


def build_equity_curve(trades: List[Dict]) -> pd.DataFrame:
    """
    Build daily equity curve from trades (Optimized O(n) version)
    
    Returns DataFrame with columns:
    - date: Trading date
    - equity: Portfolio value
    - daily_return: Daily percentage return
    - drawdown_pct: Current drawdown from peak
    """
    if not trades:
        return pd.DataFrame()
    
    # Create a dictionary of P&L by exit date for O(1) lookup
    pnl_by_date = {}
    for trade in trades:
        exit_date = trade['exit_date']
        if exit_date not in pnl_by_date:
            pnl_by_date[exit_date] = []
        pnl_by_date[exit_date].append(trade['pnl_pct'])
    
    # Get date range
    all_dates = set()
    for trade in trades:
        all_dates.add(trade['entry_date'])
        all_dates.add(trade['exit_date'])
    
    start_date = min([datetime.strptime(d, '%Y-%m-%d') for d in all_dates])
    end_date = max([datetime.strptime(d, '%Y-%m-%d') for d in all_dates])
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Build equity curve
    equity_values = []
    current_equity = INITIAL_CAPITAL
    peak_equity = INITIAL_CAPITAL
    
    for date in date_range:
        date_str = date.strftime('%Y-%m-%d')
        
        # O(1) lookup instead of O(n) loop
        if date_str in pnl_by_date:
            day_pnl = 0
            for pnl_pct in pnl_by_date[date_str]:
                position_value = current_equity * POSITION_SIZE_PCT
                trade_pnl = position_value * (pnl_pct / 100)
                day_pnl += trade_pnl
            current_equity += day_pnl
            peak_equity = max(peak_equity, current_equity)
        
        drawdown = (current_equity - peak_equity) / peak_equity * 100
        
        equity_values.append({
            'date': date,
            'equity': current_equity,
            'drawdown_pct': drawdown
        })
    
    equity_df = pd.DataFrame(equity_values)
    
    # Calculate daily returns
    if len(equity_df) > 1:
        equity_df['daily_return'] = equity_df['equity'].pct_change()
    else:
        equity_df['daily_return'] = 0.0
    
    return equity_df


def save_trade_log(all_trades: List[Dict], backtest_dir: Path):
    """Save detailed trade log to CSV"""
    if not all_trades:
        return None
    
    trade_df = pd.DataFrame(all_trades)
    trade_file = backtest_dir / f"trade_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    trade_df.to_csv(trade_file, index=False)
    return trade_file


def save_equity_curve(equity_curve: pd.DataFrame, backtest_dir: Path):
    """Save equity curve to CSV"""
    if equity_curve.empty:
        return None
    
    equity_file = backtest_dir / f"equity_curve_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    equity_curve.to_csv(equity_file, index=False)
    return equity_file


def generate_exit_analysis(all_trades: List[Dict]) -> Dict:
    """Analyze exit reasons"""
    if not all_trades:
        return {}
    
    exit_reasons = {}
    for trade in all_trades:
        reason = trade['exit_reason']
        if reason not in exit_reasons:
            exit_reasons[reason] = {'count': 0, 'avg_pnl': 0, 'wins': 0}
        exit_reasons[reason]['count'] += 1
        exit_reasons[reason]['avg_pnl'] += trade['pnl_pct']
        if trade['win']:
            exit_reasons[reason]['wins'] += 1
    
    # Calculate averages
    for reason in exit_reasons:
        count = exit_reasons[reason]['count']
        exit_reasons[reason]['avg_pnl'] = round(exit_reasons[reason]['avg_pnl'] / count, 2)
        exit_reasons[reason]['win_rate'] = round(exit_reasons[reason]['wins'] / count * 100, 2)
    
    return exit_reasons


def print_detailed_report(metrics: Dict, exit_analysis: Dict):
    """Print comprehensive backtest report"""
    print("\n" + "="*70)
    print("BACKTEST RESULTS - Detailed Report")
    print("="*70)
    
    print("\n📊 TRADE STATISTICS")
    print("-"*70)
    print(f"  Total Trades:           {metrics['total_trades']}")
    print(f"  Winning Trades:           {metrics['winning_trades']} ({metrics['win_rate']:.1f}%)")
    print(f"  Losing Trades:            {metrics['losing_trades']} ({metrics['loss_rate']:.1f}%)")
    print(f"  Max Consecutive Wins:     {metrics['consecutive_wins_max']}")
    print(f"  Max Consecutive Losses:   {metrics['consecutive_losses_max']}")
    
    print("\n💰 P&L METRICS")
    print("-"*70)
    print(f"  Average Return:           {metrics['avg_return']:.2f}%")
    print(f"  Average Win:              {metrics['avg_win']:.2f}%")
    print(f"  Average Loss:             {metrics['avg_loss']:.2f}%")
    print(f"  Largest Win:              {metrics['largest_win']:.2f}%")
    print(f"  Largest Loss:             {metrics['largest_loss']:.2f}%")
    print(f"  Gross Profits:            {metrics['gross_profits']:.2f}%")
    print(f"  Gross Losses:             -{metrics['gross_losses']:.2f}%")
    
    print("\n📈 ADVANCED METRICS")
    print("-"*70)
    print(f"  Profit Factor:            {metrics['profit_factor']:.2f}")
    print(f"  Expectancy:               {metrics['expectancy']:.2f}%")
    print(f"  R-Multiple (Avg):         {metrics['r_multiple_avg']:.2f}")
    print(f"  Sharpe Ratio:             {metrics['sharpe_ratio']:.2f}")
    print(f"  Sortino Ratio:            {metrics['sortino_ratio']:.2f}")
    print(f"  Calmar Ratio:             {metrics['calmar_ratio']:.2f}")
    print(f"  Max Drawdown:             -{metrics['max_drawdown']:.2f}%")
    print(f"  Avg Holding Days:         {metrics['avg_holding_days']:.1f}")
    
    if exit_analysis:
        print("\n🚪 EXIT ANALYSIS")
        print("-"*70)
        for reason, stats in exit_analysis.items():
            print(f"  {reason:20s}  Count: {stats['count']:3d}  "
                  f"Avg PnL: {stats['avg_pnl']:6.2f}%  Win Rate: {stats['win_rate']:5.1f}%")
    
    print("\n" + "="*70)


def run_backtest_batch(sample_size: int = 100) -> int:
    """
    Run backtest with batch processing
    
    Returns:
        0: Success
        99: More batches to process (restart needed)
    """
    global processed_tickers, current_batch, all_results
    
    ensure_dirs()
    
    # Load progress if exists
    resuming = load_progress()
    
    # Get list of tickers
    if not TIME_SERIES_DIR.exists():
        print(f"Error: Time series directory not found: {TIME_SERIES_DIR}")
        return 1
    
    all_tickers = [d.name for d in TIME_SERIES_DIR.iterdir() if d.is_dir()]
    all_tickers.sort()
    
    # Filter already processed
    remaining_tickers = [t for t in all_tickers if t not in processed_tickers]
    
    if not remaining_tickers:
        print("All tickers already processed!")
        # Calculate final results
        return finalize_backtest()
    
    # Take sample if specified
    if sample_size and len(remaining_tickers) > sample_size:
        import random
        random.seed(42)
        remaining_tickers = random.sample(remaining_tickers, sample_size)
    
    print(f"\n{'='*70}")
    print(f"BACKTEST BATCH PROCESSING")
    print(f"{'='*70}")
    print(f"Total tickers: {len(all_tickers)}")
    print(f"Already processed: {len(processed_tickers)}")
    print(f"Remaining: {len(remaining_tickers)}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Current batch: {current_batch}")
    print(f"{'='*70}\n")
    
    # Process one batch
    batch_tickers = remaining_tickers[:BATCH_SIZE]
    batch_results = []
    
    for i, ticker in enumerate(batch_tickers):
        print(f"  [{i+1}/{len(batch_tickers)}] Processing {ticker}...", end=' ')
        
        df = load_ticker_data(ticker)
        if df is not None:
            result = backtest_ticker(ticker, df)
            if result and result['total_trades'] > 0:
                batch_results.append(result)
                print(f"{result['total_trades']} trades")
            else:
                print("no trades")
        else:
            print("no data")
        
        processed_tickers.add(ticker)
    
    all_results.extend(batch_results)
    current_batch += 1
    
    # Save progress
    save_progress()
    
    # Check if more batches needed
    remaining_after = len([t for t in all_tickers if t not in processed_tickers])
    
    if remaining_after > 0:
        print(f"\n  Batch {current_batch} complete. {remaining_after} tickers remaining.")
        print(f"  Restarting for next batch...")
        return 99  # Signal to restart
    else:
        print(f"\n  All batches complete!")
        return finalize_backtest()


def finalize_backtest() -> int:
    """
    Finalize backtest and generate reports
    
    Returns:
        0: Success
        1: Failure
    """
    global all_results
    
    print(f"\n{'='*70}")
    print("FINALIZING BACKTEST")
    print(f"{'='*70}\n")
    
    # Collect all trades
    all_trades = []
    for result in all_results:
        if result and result['trades']:
            all_trades.extend(result['trades'])
    
    if not all_trades:
        print("✗ No trades generated")
        clear_progress()
        return 1
    
    # Calculate comprehensive metrics
    print("Calculating performance metrics...")
    metrics, equity_curve = calculate_comprehensive_metrics(all_results, all_trades)
    
    # Generate exit analysis
    print("Analyzing exit reasons...")
    exit_analysis = generate_exit_analysis(all_trades)
    
    # Save reports
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save JSON report
    report_file = BACKTEST_DIR / f"backtest_report_{timestamp}.json"
    full_report = {
        'backtest_date': datetime.now().isoformat(),
        'parameters': {
            'initial_capital': INITIAL_CAPITAL,
            'position_size_pct': POSITION_SIZE_PCT,
            'stop_loss_pct': STOP_LOSS_PCT,
            'take_profit_pct': TAKE_PROFIT_PCT,
            'hold_days': HOLD_DAYS,
            'risk_free_rate': RISK_FREE_RATE
        },
        'metrics': metrics,
        'exit_analysis': exit_analysis,
        'tickers_tested': list(processed_tickers),
        'total_tickers': len(processed_tickers)
    }
    
    with open(report_file, 'w') as f:
        json.dump(full_report, f, indent=2, default=str)
    
    # Save trade log CSV
    trade_file = save_trade_log(all_trades, BACKTEST_DIR)
    
    # Save equity curve CSV
    equity_file = save_equity_curve(equity_curve, BACKTEST_DIR)
    
    # Print detailed report
    print_detailed_report(metrics, exit_analysis)
    
    # Print file locations
    print("\n📁 OUTPUT FILES")
    print("-"*70)
    print(f"  JSON Report:     {report_file}")
    if trade_file:
        print(f"  Trade Log:       {trade_file}")
    if equity_file:
        print(f"  Equity Curve:    {equity_file}")
    print(f"{'='*70}\n")
    
    # Clear progress
    clear_progress()
    
    return 0


def finalize_backtest_from_loaded(all_trades: List[Dict], tickers: set) -> int:
    """
    Finalize backtest from already loaded data (for resuming)
    
    Returns:
        0: Success
        1: Failure
    """
    global processed_tickers, all_results
    processed_tickers = tickers
    
    print(f"\n{'='*70}")
    print("FINALIZING BACKTEST (from loaded data)")
    print(f"{'='*70}\n")
    
    if not all_trades:
        print("✗ No trades generated")
        return 1
    
    # Calculate comprehensive metrics
    print(f"Processing {len(all_trades)} trades from {len(tickers)} tickers...")
    print("Calculating performance metrics...")
    
    # Create mock results for metrics calculation
    all_results = [{'trades': all_trades}]
    
    metrics, equity_curve = calculate_comprehensive_metrics(all_results, all_trades)
    
    # Generate exit analysis
    print("Analyzing exit reasons...")
    exit_analysis = generate_exit_analysis(all_trades)
    
    # Save reports
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save JSON report
    report_file = BACKTEST_DIR / f"backtest_report_{timestamp}.json"
    full_report = {
        'backtest_date': datetime.now().isoformat(),
        'parameters': {
            'initial_capital': INITIAL_CAPITAL,
            'position_size_pct': POSITION_SIZE_PCT,
            'stop_loss_pct': STOP_LOSS_PCT,
            'take_profit_pct': TAKE_PROFIT_PCT,
            'hold_days': HOLD_DAYS,
            'risk_free_rate': RISK_FREE_RATE
        },
        'metrics': metrics,
        'exit_analysis': exit_analysis,
        'tickers_tested': list(tickers),
        'total_tickers': len(tickers)
    }
    
    with open(report_file, 'w') as f:
        json.dump(full_report, f, indent=2, default=str)
    
    # Save trade log CSV
    trade_file = save_trade_log(all_trades, BACKTEST_DIR)
    
    # Save equity curve CSV
    equity_file = save_equity_curve(equity_curve, BACKTEST_DIR)
    
    # Print detailed report
    print_detailed_report(metrics, exit_analysis)
    
    # Print file locations
    print("\n📁 OUTPUT FILES")
    print("-"*70)
    print(f"  JSON Report:     {report_file}")
    if trade_file:
        print(f"  Trade Log:       {trade_file}")
    if equity_file:
        print(f"  Equity Curve:    {equity_file}")
    print(f"{'='*70}\n")
    
    # Clear progress
    clear_progress()
    
    return 0


def main():
    """Main execution with batch support"""
    print("="*70)
    print("SignalsAlpha Enhanced Backtester v2.0")
    print("="*70)
    print()
    
    # Run backtest with batching
    result = run_backtest_batch(sample_size=100)
    
    return result


if __name__ == "__main__":
    sys.exit(main())

