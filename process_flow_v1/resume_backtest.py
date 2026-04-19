#!/usr/bin/env python3
"""
Resume backtest from saved progress
"""
import sys
sys.path.insert(0, '/home/ubuntu/.openclaw/workspace/trading-signals-data/process_flow_v1')

import step4_backtester as bt

# Load the saved progress
import pickle
progress_file = bt.BACKTEST_DIR / 'backtest_progress.pkl'

with open(progress_file, 'rb') as f:
    progress = pickle.load(f)

processed_tickers = progress['processed_tickers']
all_results = progress['all_results']

print(f"Loaded progress:")
print(f"  Tickers processed: {len(processed_tickers)}")
print(f"  Results collected: {len(all_results)}")
print()

# Collect all trades
all_trades = []
for result in all_results:
    if result and result['trades']:
        all_trades.extend(result['trades'])

print(f"Total trades: {len(all_trades)}")
print()

# Run finalization
print("Running finalization...")
sys.exit(bt.finalize_backtest_from_loaded(all_trades, processed_tickers))
