#!/bin/bash
# Time Series Collector - Batch restart script

cd /home/ubuntu/.openclaw/workspace
source venv/bin/activate

# Process 10 stocks then restart to prevent memory issues
python3 << 'EOF'
import os
import sys
import time
import yfinance as yf
import pandas as pd
from datetime import datetime

OUTPUT_BASE_DIR = "data/time_series"
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

INTERVALS = {
    '1m': '7d', '2m': '60d', '5m': '60d', '15m': '60d', '30m': '60d',
    '60m': '730d', '1h': '730d', '1d': 'max', '1wk': 'max', '1mo': 'max'
}

# Load symbols
symbols_df = pd.read_csv('data/signals_v3/all_signals_v3_final.csv')
all_symbols = symbols_df['ticker'].tolist()

# Check already processed
processed = []
for symbol in all_symbols:
    ticker_dir = os.path.join(OUTPUT_BASE_DIR, symbol)
    if os.path.exists(ticker_dir):
        files = [f for f in os.listdir(ticker_dir) if f.endswith('.csv')]
        if len(files) >= 10:  # All intervals
            processed.append(symbol)

symbols = [s for s in all_symbols if s not in processed]
print(f"Total: {len(all_symbols)}, Done: {len(processed)}, Remaining: {len(symbols)}")

# Process batch of 10
batch = symbols[:10]

for ticker in batch:
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] {ticker}")
    ticker_dir = os.path.join(OUTPUT_BASE_DIR, ticker)
    os.makedirs(ticker_dir, exist_ok=True)
    
    for interval, period in INTERVALS.items():
        file_path = os.path.join(ticker_dir, f"{ticker}_{interval}.csv")
        if os.path.exists(file_path):
            continue  # Skip if exists
        
        try:
            time.sleep(1)  # Rate limit
            stock = yf.Ticker(ticker)
            df = stock.history(period=period, interval=interval, prepost=False)
            
            if not df.empty:
                df = df.reset_index()
                df.columns = [col.replace(' ', '_').lower() for col in df.columns]
                df['ticker'] = ticker
                df['interval'] = interval
                df.to_csv(file_path, index=False)
                print(f"  {interval}: ✓ {len(df):,}")
            else:
                print(f"  {interval}: ✗ No data")
        except Exception as e:
            print(f"  {interval}: ✗ {str(e)[:30]}")

print(f"\nBatch complete. Processed {len(batch)} stocks.")
EOF

# Check remaining
REMAINING=$(python3 -c "
import os, pandas as pd
symbols = pd.read_csv('data/signals_v3/all_signals_v3_final.csv')['ticker'].tolist()
processed = []
for s in symbols:
    d = f'data/time_series/{s}'
    if os.path.exists(d) and len([f for f in os.listdir(d) if f.endswith('.csv')]) >= 10:
        processed.append(s)
print(len(symbols) - len(processed))
")

echo ""
echo "Remaining: $REMAINING"

if [ "$REMAINING" -gt "0" ]; then
    echo "Restarting..."
    exec bash "$0"
else
    echo "All complete!"
fi
