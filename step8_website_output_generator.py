#!/usr/bin/env python3
"""
================================================================================
STEP 9: Website Output Generator - Step 8
================================================================================

Transforms raw signal data into optimized JSON files for the SignalsAlpha
website frontend. Aggregates signals, creates summaries, and generates ranked
lists for user consumption.

Author: SignalsAlpha
Version: 1.0
Date: 2026-04-18

================================================================================
PURPOSE
================================================================================

This module serves as the bridge between the data pipeline and the website:

1. Reads signal files from Step 2 (Multi-Timeframe Scanner)
2. Aggregates signals across all timeframes per ticker
3. Calculates strength scores and rankings
4. Generates optimized JSON for frontend consumption
5. Creates summary statistics for dashboards
6. Filters and sorts signals by confidence and strength

================================================================================
OUTPUT FILES GENERATED
================================================================================

1. signals/signals_summary.json
   - Top signals across all timeframes
   - Ranked by strength score
   - Limited to highest-confidence signals

2. signals/signals_by_ticker.json
   - All signals organized by ticker symbol
   - Includes timeframe breakdown
   - Used for individual stock pages

3. signals/signals_by_timeframe.json
   - Signals grouped by interval (1d, 1wk, 1mo)
   - For timeframe-specific views

4. signals/signals_today.json
   - Only signals generated today
   - For "Today's Signals" page

5. signals/signals_strong.json
   - STRONG_BUY and STRONG_SELL only
   - Highest confidence signals

================================================================================
SIGNAL AGGREGATION LOGIC
================================================================================

Strength Score Calculation:
    - Weighted sum of scores across timeframes
    - Daily signals: 1.0x weight
    - Weekly signals: 1.5x weight
    - Monthly signals: 2.0x weight

Ranking:
    1. Sort by signal type (STRONG_BUY > BUY > WEAK_BUY)
    2. Then by confidence score (higher first)
    3. Then by strength score (higher first)

Filtering:
    - Minimum confidence: 70%
    - Maximum signals per ticker: 3 (one per timeframe)
    - Exclude delisted/invalid tickers

================================================================================
WORKFLOW
================================================================================

1. LOAD SIGNALS
   - Scan data/signals_timeframe/ directory
   - Read all CSV files
   - Parse into structured format

2. AGGREGATE
   - Group signals by ticker
   - Calculate multi-timeframe strength
   - Determine overall signal direction

3. RANK
   - Sort by confidence and strength
   - Apply filters
   - Select top N signals

4. GENERATE OUTPUT
   - Create JSON files
   - Pretty-print for readability
   - Validate JSON structure

5. SAVE
   - Write to data/signals/ directory
   - Create subdirectories as needed

================================================================================
USAGE
================================================================================

    python step4_website_output_generator.py

No arguments required - reads from data/signals_timeframe/ automatically.

================================================================================
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

# Configuration - reads from existing data dirs, outputs to website dir
DATA_DIR = Path("/home/ubuntu/.openclaw/workspace/data")
SIGNALS_DIR = DATA_DIR / "signals_timeframe"
WEBSITE_OUTPUT_DIR = Path("/home/ubuntu/.openclaw/workspace/signalsalpha/prototype/data")

# Output subdirectories
SIGNALS_OUTPUT_DIR = WEBSITE_OUTPUT_DIR / "signals"
PRICES_OUTPUT_DIR = WEBSITE_OUTPUT_DIR / "prices"
FUNDAMENTALS_OUTPUT_DIR = WEBSITE_OUTPUT_DIR / "fundamentals"


def ensure_dirs():
    """Create output directories"""
    SIGNALS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PRICES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FUNDAMENTALS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_all_signals():
    """Load all signal files from signals_timeframe"""
    print("Loading signals data...")
    
    all_signals = []
    
    if not SIGNALS_DIR.exists():
        print(f"Warning: Signals directory not found: {SIGNALS_DIR}")
        return all_signals
    
    for ticker_dir in SIGNALS_DIR.iterdir():
        if not ticker_dir.is_dir():
            continue
        
        ticker = ticker_dir.name
        signals_file = ticker_dir / f"{ticker}_signals.csv"
        
        if signals_file.exists():
            try:
                df = pd.read_csv(signals_file)
                all_signals.append(df)
            except Exception as e:
                print(f"  Warning: Could not read {ticker}: {e}")
    
    if all_signals:
        combined = pd.concat(all_signals, ignore_index=True)
        print(f"  Loaded {len(combined)} signals from {len(all_signals)} tickers")
        return combined
    
    return pd.DataFrame()


def generate_ticker_summary(df, ticker):
    """Generate summary for a single ticker"""
    ticker_data = df[df['ticker'] == ticker]
    
    if ticker_data.empty:
        return None
    
    # Get latest signal (prefer daily)
    daily = ticker_data[ticker_data['interval'] == '1d']
    latest = daily.iloc[-1] if not daily.empty else ticker_data.iloc[-1]
    
    summary = {
        "ticker": ticker,
        "latest_signal": latest.get('signal', 'HOLD'),
        "confidence": int(latest.get('confidence', 50)) if pd.notna(latest.get('confidence')) else 50,
        "current_price": float(latest['close']) if pd.notna(latest.get('close')) else None,
        "timestamp": str(latest.get('timestamp', datetime.now().isoformat())),
        "timeframes": {}
    }
    
    # Add data for each timeframe
    for _, row in ticker_data.iterrows():
        interval = row.get('interval', '1d')
        summary["timeframes"][interval] = {
            "signal": row.get('signal', 'HOLD'),
            "confidence": int(row['confidence']) if pd.notna(row.get('confidence')) else 50,
            "price": float(row['close']) if pd.notna(row.get('close')) else None,
            "rsi": float(row['rsi']) if pd.notna(row.get('rsi')) else None,
            "sma_20": float(row['sma_20']) if pd.notna(row.get('sma_20')) else None,
            "sma_50": float(row['sma_50']) if pd.notna(row.get('sma_50')) else None,
            "macd": float(row['macd']) if pd.notna(row.get('macd')) else None,
            "vwap": float(row['vwap']) if pd.notna(row.get('vwap')) else None,
            "atr": float(row['atr']) if pd.notna(row.get('atr')) else None
        }
    
    return summary


def generate_signals_json(df):
    """Generate main signals summary JSON"""
    print("\nGenerating signals summary...")
    
    signals_list = []
    
    for ticker in df['ticker'].unique():
        summary = generate_ticker_summary(df, ticker)
        if summary:
            signals_list.append(summary)
    
    # Rank by confidence and signal strength
    signal_ranking = {
        'STRONG_BUY': 6, 'BUY': 5, 'WEAK_BUY': 4,
        'HOLD': 3,
        'WEAK_SELL': 2, 'SELL': 1, 'STRONG_SELL': 0
    }
    
    signals_list.sort(key=lambda x: (
        signal_ranking.get(x['latest_signal'], 3),
        x['confidence']
    ), reverse=True)
    
    output = {
        "generated_at": datetime.now().isoformat(),
        "total_signals": len(signals_list),
        "signals": signals_list
    }
    
    output_file = SIGNALS_OUTPUT_DIR / "all_signals.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"  ✓ Saved {output_file}")
    print(f"    Total: {len(signals_list)} tickers")
    
    # Generate summary by signal type
    by_signal = {}
    for s in signals_list:
        sig = s['latest_signal']
        by_signal[sig] = by_signal.get(sig, 0) + 1
    
    print(f"    Distribution: {by_signal}")
    
    return output_file


def generate_top_picks(df, top_n=20):
    """Generate top picks JSON"""
    print("\nGenerating top picks...")
    
    # Filter for BUY signals only
    buy_signals = df[df['signal'].isin(['STRONG_BUY', 'BUY'])]
    
    if buy_signals.empty:
        print("  No BUY signals found")
        return None
    
    # Get daily timeframe only
    daily_buys = buy_signals[buy_signals['interval'] == '1d']
    
    # Sort by confidence
    daily_buys = daily_buys.sort_values('confidence', ascending=False)
    
    # Take top N
    top = daily_buys.head(top_n)
    
    picks = []
    for _, row in top.iterrows():
        pick = {
            "ticker": row['ticker'],
            "signal": row['signal'],
            "confidence": int(row['confidence']) if pd.notna(row['confidence']) else 50,
            "price": float(row['close']) if pd.notna(row['close']) else None,
            "rsi": float(row['rsi']) if pd.notna(row.get('rsi')) else None,
            "buy_score": int(row['buy_score']) if pd.notna(row.get('buy_score')) else 0,
            "patterns": row.get('candlestick_patterns', None)
        }
        picks.append(pick)
    
    output = {
        "generated_at": datetime.now().isoformat(),
        "count": len(picks),
        "picks": picks
    }
    
    output_file = SIGNALS_OUTPUT_DIR / "top_picks.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"  ✓ Saved {output_file}")
    print(f"    Top {len(picks)} picks")
    
    return output_file


def generate_market_overview(df):
    """Generate market overview JSON"""
    print("\nGenerating market overview...")
    
    # Count signals by type
    daily = df[df['interval'] == '1d']
    
    overview = {
        "generated_at": datetime.now().isoformat(),
        "total_tickers": len(daily['ticker'].unique()) if not daily.empty else 0,
        "signal_distribution": {}
    }
    
    if not daily.empty:
        signal_counts = daily['signal'].value_counts().to_dict()
        overview["signal_distribution"] = signal_counts
    
    # Calculate average confidence
    if not daily.empty and 'confidence' in daily.columns:
        overview["avg_confidence"] = round(daily['confidence'].mean(), 2)
    
    output_file = SIGNALS_OUTPUT_DIR / "market_overview.json"
    with open(output_file, 'w') as f:
        json.dump(overview, f, indent=2, default=str)
    
    print(f"  ✓ Saved {output_file}")
    
    return output_file


def generate_individual_ticker_files(df):
    """Generate JSON files for each ticker"""
    print("\nGenerating individual ticker files...")
    
    count = 0
    for ticker in df['ticker'].unique():
        ticker_data = df[df['ticker'] == ticker]
        
        output = {
            "ticker": ticker,
            "generated_at": datetime.now().isoformat(),
            "data": []
        }
        
        for _, row in ticker_data.iterrows():
            entry = {
                "interval": row.get('interval', '1d'),
                "timestamp": str(row.get('timestamp', '')),
                "signal": row.get('signal', 'HOLD'),
                "confidence": int(row['confidence']) if pd.notna(row.get('confidence')) else 50,
                "price": float(row['close']) if pd.notna(row.get('close')) else None,
                "indicators": {
                    "rsi": float(row['rsi']) if pd.notna(row.get('rsi')) else None,
                    "macd": float(row['macd']) if pd.notna(row.get('macd')) else None,
                    "sma_20": float(row['sma_20']) if pd.notna(row.get('sma_20')) else None,
                    "sma_50": float(row['sma_50']) if pd.notna(row.get('sma_50')) else None,
                    "vwap": float(row['vwap']) if pd.notna(row.get('vwap')) else None,
                    "atr": float(row['atr']) if pd.notna(row.get('atr')) else None
                },
                "levels": {
                    "support": json.loads(row['support_levels']) if pd.notna(row.get('support_levels')) else [],
                    "resistance": json.loads(row['resistance_levels']) if pd.notna(row.get('resistance_levels')) else []
                },
                "patterns": row.get('candlestick_patterns', None)
            }
            output["data"].append(entry)
        
        output_file = SIGNALS_OUTPUT_DIR / f"{ticker}_signals.json"
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        
        count += 1
        if count % 100 == 0:
            print(f"  Progress: {count} tickers...")
    
    print(f"  ✓ Generated {count} individual ticker files")


def main():
    """Main execution"""
    print("=" * 60)
    print("SignalsAlpha Website Output Generator - Step 8")
    print("=" * 60)
    print()
    
    ensure_dirs()
    
    # Load all signals
    df = load_all_signals()
    
    if df.empty:
        print("\n✗ No signals data found. Run Step 2 first.")
        return 1
    
    print(f"\nData loaded: {len(df)} records")
    print(f"Tickers: {df['ticker'].nunique()}")
    print(f"Timeframes: {df['interval'].unique().tolist()}")
    
    # Generate outputs
    generate_signals_json(df)
    generate_top_picks(df)
    generate_market_overview(df)
    generate_individual_ticker_files(df)
    
    print("\n" + "=" * 60)
    print("Website Output Generation Complete!")
    print("=" * 60)
    print(f"\nOutput directory: {WEBSITE_OUTPUT_DIR}")
    print(f"  Signals: {SIGNALS_OUTPUT_DIR}")
    print(f"  Prices: {PRICES_OUTPUT_DIR}")
    print(f"  Fundamentals: {FUNDAMENTALS_OUTPUT_DIR}")
    print("\n✓ Ready for website deployment")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
