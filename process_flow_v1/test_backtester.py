#!/usr/bin/env python3
"""
Quick test of the backtester with 5 tickers
"""

import sys
sys.path.insert(0, '/home/ubuntu/.openclaw/workspace/trading-signals-data/process_flow_v1')

# Override batch size and sample size for testing
import step4_backtester as bt
bt.BATCH_SIZE = 5

# Temporarily modify run_backtest_batch to only process 5 tickers
def test_backtest():
    print("="*60)
    print("TESTING BACKTESTER WITH 5 TICKERS")
    print("="*60)
    print()
    
    bt.ensure_dirs()
    
    # Get list of tickers
    if not bt.TIME_SERIES_DIR.exists():
        print(f"Error: Time series directory not found: {bt.TIME_SERIES_DIR}")
        return 1
    
    all_tickers = [d.name for d in bt.TIME_SERIES_DIR.iterdir() if d.is_dir()]
    all_tickers.sort()
    
    # Take only 5 for test
    test_tickers = all_tickers[:5]
    
    print(f"Testing with {len(test_tickers)} tickers: {test_tickers}")
    print()
    
    all_trades = []
    
    for ticker in test_tickers:
        print(f"Processing {ticker}...", end=' ')
        df = bt.load_ticker_data(ticker)
        if df is not None:
            result = bt.backtest_ticker(ticker, df)
            if result and result['total_trades'] > 0:
                all_trades.extend(result['trades'])
                print(f"{result['total_trades']} trades")
            else:
                print("no trades")
        else:
            print("no data")
    
    print()
    print("="*60)
    print("TEST COMPLETE")
    print("="*60)
    print(f"Total trades: {len(all_trades)}")
    
    if all_trades:
        wins = sum(1 for t in all_trades if t['win'])
        print(f"Winning trades: {wins}")
        print(f"Sample trade: {all_trades[0]}")
    
    return 0

if __name__ == "__main__":
    sys.exit(test_backtest())
