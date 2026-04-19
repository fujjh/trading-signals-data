#!/usr/bin/env python3
"""
Test the numerical fixes
"""
import sys
sys.path.insert(0, '/home/ubuntu/.openclaw/workspace/trading-signals-data/process_flow_v1')

import step4_backtester as bt

# Test with 2 tickers including RCAT which had the issue
bt.BATCH_SIZE = 2

def test_fix():
    print("Testing numerical fixes...")
    print()
    
    bt.ensure_dirs()
    
    # Get RCAT and one other ticker
    test_tickers = ['RCAT', 'AAPL']
    
    all_trades = []
    
    for ticker in test_tickers:
        print(f"Processing {ticker}...", end=' ')
        df = bt.load_ticker_data(ticker)
        if df is not None:
            result = bt.backtest_ticker(ticker, df)
            if result and result['total_trades'] > 0:
                all_trades.extend(result['trades'])
                print(f"{result['total_trades']} trades")
                # Check for inf values
                for t in result['trades'][:5]:
                    print(f"  Sample: entry={t['entry_price']}, pnl={t['pnl_pct']}")
            else:
                print("no trades")
        else:
            print("no data")
    
    print()
    print(f"Total trades: {len(all_trades)}")
    
    if all_trades:
        # Check for inf values
        import math
        inf_count = sum(1 for t in all_trades if math.isinf(t['pnl_pct']))
        print(f"Inf values: {inf_count}")
        
        # Test metrics calculation
        print("\nTesting metrics calculation...")
        metrics, _ = bt.calculate_comprehensive_metrics([{'trades': all_trades}], all_trades)
        print(f"Avg return: {metrics['avg_return']}")
        print(f"Win rate: {metrics['win_rate']}")
        print(f"All metrics valid: {not math.isinf(metrics['avg_return']) and not math.isnan(metrics['avg_return'])}")
    
    return 0

if __name__ == "__main__":
    sys.exit(test_fix())
