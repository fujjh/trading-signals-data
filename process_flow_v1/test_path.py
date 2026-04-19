#!/usr/bin/env python3
"""
Test that we're using the correct data path
"""
import sys
sys.path.insert(0, '/home/ubuntu/.openclaw/workspace/trading-signals-data/process_flow_v1')

import step4_backtester as bt

print(f"TIME_SERIES_DIR: {bt.TIME_SERIES_DIR}")
print(f"Exists: {bt.TIME_SERIES_DIR.exists()}")
print()

# Count tickers
if bt.TIME_SERIES_DIR.exists():
    tickers = [d.name for d in bt.TIME_SERIES_DIR.iterdir() if d.is_dir()]
    print(f"Total tickers available: {len(tickers)}")
    print(f"Sample tickers: {tickers[:10]}")
else:
    print("ERROR: Directory not found!")
