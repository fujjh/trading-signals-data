#!/usr/bin/env python3
"""
================================================================================
METRICS CALCULATOR MODULE
================================================================================

Comprehensive performance metrics for trading strategy evaluation.
Calculates risk-adjusted returns, drawdown analysis, and trade statistics.

Author: SignalsAlpha
Version: 1.0
Date: 2026-04-28
================================================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Trade:
    """Individual trade record"""
    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    position_size: float
    pnl: float
    return_pct: float
    signal_type: str
    ticker: str


@dataclass  
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    # Profitability
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    gross_profit: float
    gross_loss: float
    net_profit: float
    profit_factor: float
    expectancy: float
    payoff_ratio: float
    avg_trade: float
    avg_win: float
    avg_loss: float
    
    # Risk-Adjusted Returns
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    
    # Drawdown
    max_drawdown: float
    max_drawdown_duration: int
    avg_drawdown: float
    ulcer_index: float
    
    # Returns
    total_return: float
    annualized_return: float
    volatility: float
    downside_volatility: float
    
    # Consistency
    consecutive_wins: int
    consecutive_losses: int
    max_consecutive_wins: int
    max_consecutive_losses: int


class MetricsCalculator:
    """Calculate comprehensive trading performance metrics"""
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize calculator
        
        Args:
            risk_free_rate: Annual risk-free rate for Sharpe calculation
        """
        self.risk_free_rate = risk_free_rate
    
    def calculate_from_trades(self, trades: List[Trade]) -> PerformanceMetrics:
        """
        Calculate all metrics from trade history
        
        Args:
            trades: List of completed trades
            
        Returns:
            PerformanceMetrics object with all calculations
        """
        if not trades:
            return self._empty_metrics()
        
        # Basic counts
        total_trades = len(trades)
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl <= 0]
        
        n_win = len(winning_trades)
        n_loss = len(losing_trades)
        win_rate = n_win / total_trades if total_trades > 0 else 0
        
        # Profit/Loss calculations
        gross_profit = sum(t.pnl for t in winning_trades)
        gross_loss = abs(sum(t.pnl for t in losing_trades))
        net_profit = gross_profit - gross_loss
        
        # Profit Factor
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Averages
        avg_trade = net_profit / total_trades if total_trades > 0 else 0
        avg_win = gross_profit / n_win if n_win > 0 else 0
        avg_loss = gross_loss / n_loss if n_loss > 0 else 0
        
        # Expectancy = (Win% × Avg Win) - (Loss% × Avg Loss)
        loss_rate = 1 - win_rate
        expectancy = (win_rate * avg_win) - (loss_rate * avg_loss) if total_trades > 0 else 0
        
        # Payoff Ratio = Avg Win / Avg Loss
        payoff_ratio = avg_win / avg_loss if avg_loss > 0 else 0
        
        # Returns series for volatility calculations
        returns = [t.return_pct for t in trades]
        
        # Sharpe Ratio (annualized)
        sharpe = self._calculate_sharpe(returns)
        
        # Sortino Ratio (downside risk only)
        sortino = self._calculate_sortino(returns)
        
        # Drawdown analysis
        max_dd, max_dd_duration, avg_dd, ulcer = self._calculate_drawdowns(trades)
        
        # Calmar Ratio
        total_return = sum(returns)
        calmar = total_return / max_dd if max_dd > 0 else 0
        
        # Volatility
        volatility = np.std(returns) * np.sqrt(252) if returns else 0
        downside_returns = [r for r in returns if r < 0]
        downside_vol = np.std(downside_returns) * np.sqrt(252) if downside_returns else 0
        
        # Consecutive wins/losses
        max_consec_wins, max_consec_losses = self._calculate_consecutive(trades)
        
        return PerformanceMetrics(
            total_trades=total_trades,
            winning_trades=n_win,
            losing_trades=n_loss,
            win_rate=win_rate,
            gross_profit=gross_profit,
            gross_loss=gross_loss,
            net_profit=net_profit,
            profit_factor=profit_factor,
            expectancy=expectancy,
            payoff_ratio=payoff_ratio,
            avg_trade=avg_trade,
            avg_win=avg_win,
            avg_loss=avg_loss,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown=max_dd,
            max_drawdown_duration=max_dd_duration,
            avg_drawdown=avg_dd,
            ulcer_index=ulcer,
            total_return=total_return,
            annualized_return=self._annualize_return(total_return, trades),
            volatility=volatility,
            downside_volatility=downside_vol,
            consecutive_wins=0,  # Current streak
            consecutive_losses=0,
            max_consecutive_wins=max_consec_wins,
            max_consecutive_losses=max_consec_losses
        )
    
    def _calculate_sharpe(self, returns: List[float]) -> float:
        """Calculate annualized Sharpe Ratio"""
        if not returns or len(returns) < 2:
            return 0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0
        
        # Annualized (assuming daily returns)
        sharpe = (mean_return - self.risk_free_rate/252) / std_return * np.sqrt(252)
        return sharpe
    
    def _calculate_sortino(self, returns: List[float]) -> float:
        """Calculate annualized Sortino Ratio (downside risk only)"""
        if not returns or len(returns) < 2:
            return 0
        
        mean_return = np.mean(returns)
        downside_returns = [r for r in returns if r < 0]
        
        if not downside_returns:
            return float('inf')  # No downside risk
        
        downside_std = np.std(downside_returns)
        
        if downside_std == 0:
            return 0
        
        sortino = (mean_return - self.risk_free_rate/252) / downside_std * np.sqrt(252)
        return sortino
    
    def _calculate_drawdowns(self, trades: List[Trade]) -> Tuple[float, int, float, float]:
        """
        Calculate drawdown metrics
        
        Returns:
            (max_drawdown, max_duration, avg_drawdown, ulcer_index)
        """
        if not trades:
            return 0, 0, 0, 0
        
        # Build equity curve
        equity = [1.0]  # Start with $1
        for trade in trades:
            equity.append(equity[-1] * (1 + trade.return_pct))
        
        # Calculate drawdowns
        peak = equity[0]
        max_drawdown = 0
        max_duration = 0
        current_duration = 0
        drawdowns = []
        
        for value in equity:
            if value > peak:
                peak = value
                if current_duration > max_duration:
                    max_duration = current_duration
                current_duration = 0
            else:
                drawdown = (peak - value) / peak
                drawdowns.append(drawdown)
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
                current_duration += 1
        
        avg_drawdown = np.mean(drawdowns) if drawdowns else 0
        
        # Ulcer Index = sqrt(mean of squared drawdowns)
        ulcer = np.sqrt(np.mean([d**2 for d in drawdowns])) if drawdowns else 0
        
        return max_drawdown, max_duration, avg_drawdown, ulcer
    
    def _calculate_consecutive(self, trades: List[Trade]) -> Tuple[int, int]:
        """Calculate max consecutive wins and losses"""
        if not trades:
            return 0, 0
        
        max_wins = 0
        max_losses = 0
        current_wins = 0
        current_losses = 0
        
        for trade in trades:
            if trade.pnl > 0:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)
        
        return max_wins, max_losses
    
    def _annualize_return(self, total_return: float, trades: List[Trade]) -> float:
        """Annualize total return based on trade period"""
        if not trades or len(trades) < 2:
            return 0
        
        # Get date range
        dates = [t.entry_date for t in trades] + [t.exit_date for t in trades]
        start = min(dates)
        end = max(dates)
        
        days = (end - start).days
        if days <= 0:
            return 0
        
        # Annualize: (1 + total)^(365/days) - 1
        annualized = (1 + total_return) ** (365 / days) - 1
        return annualized
    
    def _empty_metrics(self) -> PerformanceMetrics:
        """Return empty metrics for no trades"""
        return PerformanceMetrics(
            total_trades=0, winning_trades=0, losing_trades=0,
            win_rate=0, gross_profit=0, gross_loss=0, net_profit=0,
            profit_factor=0, expectancy=0, payoff_ratio=0,
            avg_trade=0, avg_win=0, avg_loss=0,
            sharpe_ratio=0, sortino_ratio=0, calmar_ratio=0,
            max_drawdown=0, max_drawdown_duration=0, avg_drawdown=0, ulcer_index=0,
            total_return=0, annualized_return=0, volatility=0, downside_volatility=0,
            consecutive_wins=0, consecutive_losses=0,
            max_consecutive_wins=0, max_consecutive_losses=0
        )
    
    def print_report(self, metrics: PerformanceMetrics) -> str:
        """Generate formatted performance report"""
        report = []
        report.append("=" * 70)
        report.append("PERFORMANCE METRICS REPORT")
        report.append("=" * 70)
        report.append("")
        report.append("PROFITABILITY")
        report.append("-" * 70)
        report.append(f"Total Trades:        {metrics.total_trades:>20,}")
        report.append(f"Winning Trades:      {metrics.winning_trades:>20,} ({metrics.win_rate*100:.1f}%)")
        report.append(f"Losing Trades:       {metrics.losing_trades:>20,}")
        report.append(f"Net Profit:          ${metrics.net_profit:>19,.2f}")
        report.append(f"Profit Factor:       {metrics.profit_factor:>20.2f}")
        report.append(f"Expectancy:          ${metrics.expectancy:>19,.2f}")
        report.append(f"Payoff Ratio:        {metrics.payoff_ratio:>20.2f}")
        report.append(f"Avg Trade:           ${metrics.avg_trade:>19,.2f}")
        report.append("")
        report.append("RISK-ADJUSTED RETURNS")
        report.append("-" * 70)
        report.append(f"Sharpe Ratio:        {metrics.sharpe_ratio:>20.2f}")
        report.append(f"Sortino Ratio:       {metrics.sortino_ratio:>20.2f}")
        report.append(f"Calmar Ratio:        {metrics.calmar_ratio:>20.2f}")
        report.append(f"Volatility:          {metrics.volatility*100:>19.2f}%")
        report.append("")
        report.append("DRAWDOWN ANALYSIS")
        report.append("-" * 70)
        report.append(f"Max Drawdown:        {metrics.max_drawdown*100:>19.2f}%")
        report.append(f"Max DD Duration:     {metrics.max_drawdown_duration:>20} days")
        report.append(f"Avg Drawdown:        {metrics.avg_drawdown*100:>19.2f}%")
        report.append(f"Ulcer Index:         {metrics.ulcer_index*100:>19.2f}%")
        report.append("")
        report.append("CONSISTENCY")
        report.append("-" * 70)
        report.append(f"Max Consecutive Wins:    {metrics.max_consecutive_wins:>16}")
        report.append(f"Max Consecutive Losses:  {metrics.max_consecutive_losses:>16}")
        report.append("=" * 70)
        
        return "\n".join(report)


if __name__ == "__main__":
    # Test with sample data
    calc = MetricsCalculator()
    
    # Create sample trades
    sample_trades = [
        Trade(datetime(2026, 1, 1), datetime(2026, 1, 5), 100, 110, 1000, 10000, 0.10, "BUY", "AAPL"),
        Trade(datetime(2026, 1, 6), datetime(2026, 1, 10), 110, 105, 1000, -5000, -0.045, "SELL", "AAPL"),
        Trade(datetime(2026, 1, 11), datetime(2026, 1, 15), 200, 220, 500, 10000, 0.10, "BUY", "MSFT"),
    ]
    
    metrics = calc.calculate_from_trades(sample_trades)
    print(calc.print_report(metrics))
