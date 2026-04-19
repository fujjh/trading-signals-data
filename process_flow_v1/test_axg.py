#!/usr/bin/env python3
"""
Test AXG data loading
"""
import sys
sys.path.insert(0, '/home/ubuntu/.openclaw/workspace/trading-signals-data/process_flow_v1')

import step4_backtester as bt

df = bt.load_ticker_data('AXG')
if df is not None:
    print("AXG loaded successfully")
    print(f"Columns: {list(df.columns)}")
    print(f"Shape: {df.shape}")
    print("\nSample data:")
    print(df[['date', 'Open', 'Close']].head())
    print("\n...")
    print(df[['date', 'Open', 'Close']].tail())
else:
    print("Failed to load AXG")
