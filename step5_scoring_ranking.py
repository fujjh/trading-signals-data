#!/usr/bin/env python3
"""
================================================================================
STEP 5: SPY Scoring & Ranking - Step 5
================================================================================

Generates BUY/SELL signals for SPY based on technical analysis only.

Author: SignalsAlpha
Version: 1.1 (SPY Edition - Technical Only)
Date: 2026-04-29

================================================================================
SCORING METHODOLOGY
================================================================================

Technical Score (100%):
- Trend alignment (SMA, EMA): 0-25 points
- Momentum (RSI, MACD): 0-30 points
- Volatility (BB, ATR): 0-20 points
- HMA signals: 0-25 points
- Elder Impulse: 0-20 points

Signal: STRONG_BUY/BUY/HOLD/SELL/STRONG_SELL
================================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

TICKER = "SPY"
INTERVAL = "1d"
TECHNICAL_FILE = Path("data/technical_analysis") / TICKER / f"{TICKER}_{INTERVAL}_technical.csv"
OUTPUT_FILE = Path("data/signals_scored") / f"spy_scored_{datetime.now().strftime('%Y%m%d')}.csv"


def load_technical():
    """Load SPY technical analysis data."""
    if not TECHNICAL_FILE.exists():
        print(f"ERROR: Technical file not found: {TECHNICAL_FILE}")
        return None

    df = pd.read_csv(TECHNICAL_FILE)
    df['date'] = pd.to_datetime(df['date'])
    return df


def calculate_technical_score(row):
    """Calculate technical score for a single row."""
    score = 50  # Neutral base

    # Trend alignment
    if row['close'] > row.get('sma_20', 0) and row['sma_20'] > row.get('sma_50', 0):
        score += 10  # Bullish trend
    elif row['close'] < row.get('sma_20', 999) and row['sma_20'] < row.get('sma_50', 0):
        score -= 10  # Bearish trend

    # RSI
    rsi = row.get('rsi', 50)
    if rsi < 30:
        score += 15  # Oversold
    elif rsi > 70:
        score -= 15  # Overbought
    elif rsi < 45:
        score += 5
    elif rsi > 55:
        score -= 5

    # MACD
    macd = row.get('macd', 0)
    macd_signal = row.get('macd_signal', 0)
    if macd > macd_signal:
        score += 10
    else:
        score -= 10

    # HMA
    hma = row.get('hma_13', row['close'])
    if row['close'] > hma:
        score += 8
    else:
        score -= 8

    # Elder Impulse
    impulse = row.get('elder_impulse', 'blue')
    if impulse == 'green':
        score += 10
    elif impulse == 'red':
        score -= 10

    # Clamp to 0-100
    return max(0, min(100, score))


def determine_signal(combined_score):
    """Determine signal from combined score."""
    if combined_score >= 75:
        return "STRONG_BUY"
    elif combined_score >= 60:
        return "BUY"
    elif combined_score >= 40:
        return "HOLD"
    elif combined_score >= 25:
        return "SELL"
    else:
        return "STRONG_SELL"


def generate_scores():
    """Generate scores for SPY."""
    print("=" * 70)
    print("STEP 5: SPY Scoring & Ranking")
    print("=" * 70)

    # Load data
    tech_df = load_technical()
    if tech_df is None:
        return

    print(f"\nTechnical data: {len(tech_df)} rows")

    # Calculate technical scores for each date
    tech_scores = []
    for _, row in tech_df.iterrows():
        tech_score = calculate_technical_score(row)
        tech_scores.append(tech_score)

    tech_df['technical_score'] = tech_scores
    
    # Combined score = technical score only (100% technical)
    tech_df['combined_score'] = tech_df['technical_score']

    # Determine signals
    tech_df['signal'] = tech_df['combined_score'].apply(determine_signal)

    # Add timestamp
    tech_df['scored_date'] = datetime.now().strftime('%Y-%m-%d')

    # Save output
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Select output columns
    output_cols = ['date', 'close', 'technical_score',
                   'combined_score', 'signal', 'rsi', 'macd', 'hma_13', 'elder_impulse']

    # Only include columns that exist
    output_cols = [c for c in output_cols if c in tech_df.columns]

    output_df = tech_df[output_cols].copy()
    output_df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSaved to: {OUTPUT_FILE}")
    print(f"Total signals: {len(output_df)}")

    # Print latest signal
    latest = output_df.iloc[-1]
    print("\n" + "=" * 70)
    print("Latest Signal")
    print("=" * 70)
    print(f"Date: {latest['date'].strftime('%Y-%m-%d') if hasattr(latest['date'], 'strftime') else latest['date']}")
    print(f"Price: ${latest['close']:.2f}")
    print(f"Technical Score: {latest['technical_score']:.1f}")
    print(f"Combined Score: {latest['combined_score']:.1f}")
    print(f"Signal: {latest['signal']}")

    # Signal distribution
    print("\n" + "=" * 70)
    print("Signal Distribution (Last 100 Days)")
    print("=" * 70)
    recent = output_df.tail(100)['signal'].value_counts()
    for signal, count in recent.items():
        print(f"  {signal}: {count} ({count/len(recent)*100:.1f}%)")
    print("=" * 70)


if __name__ == "__main__":
    generate_scores()
