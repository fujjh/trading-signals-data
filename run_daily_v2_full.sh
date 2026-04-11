#!/bin/bash
# Daily Data Collector v2.0 - Full Collection with Self-Restart

cd /home/ubuntu/.openclaw/workspace
source venv/bin/activate

python3 << 'EOF'
import os
import sys
import time
import yfinance as yf
import pandas as pd
from datetime import datetime

OUTPUT_DIR = "data/daily_data_v2"
LOG_FILE = os.path.join(OUTPUT_DIR, "collection_log.txt")

INTERVALS = {
    '1d': {'period': 'max'},
    '1wk': {'period': 'max'},
    '1mo': {'period': 'max'}
}

MIN_DELAY = 1.0
last_call = 0

def rate_limit():
    global last_call
    elapsed = time.time() - last_call
    if elapsed < MIN_DELAY:
        time.sleep(MIN_DELAY - elapsed)
    last_call = time.time()

def log_message(msg):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S ET')
    log_line = f"[{timestamp}] {msg}"
    print(log_line)
    with open(LOG_FILE, 'a') as f:
        f.write(log_line + '\n')

def process_ticker(ticker):
    ticker_dir = os.path.join(OUTPUT_DIR, ticker)
    os.makedirs(ticker_dir, exist_ok=True)
    
    for interval, config in INTERVALS.items():
        file_path = os.path.join(ticker_dir, f"{ticker}_{interval}.csv")
        if os.path.exists(file_path):
            continue
        
        try:
            rate_limit()
            stock = yf.Ticker(ticker)
            df = stock.history(period=config['period'], interval=interval, prepost=False)
            
            if df.empty:
                continue
            
            df = df.reset_index()
            df.columns = [col.replace(' ', '_').lower() for col in df.columns]
            df['ticker'] = ticker
            df['interval'] = interval
            df['split_adjusted'] = True
            df['collected_at'] = datetime.now().isoformat()
            df.to_csv(file_path, index=False)
            
        except Exception as e:
            log_message(f"  {ticker} {interval}: {str(e)[:40]}")

# Load symbols
symbols_df = pd.read_csv('data/signals_v3/all_signals_v3_final.csv')
all_symbols = symbols_df['ticker'].tolist()

# Check already processed
processed = []
for symbol in all_symbols:
    sym_dir = os.path.join(OUTPUT_DIR, symbol)
    if os.path.exists(sym_dir):
        files = [f for f in os.listdir(sym_dir) if f.endswith('.csv')]
        if len(files) >= 3:
            processed.append(symbol)

symbols = [s for s in all_symbols if s not in processed]

log_message(f"Daily Data Collector v2.0 - Batch Start")
log_message(f"Total: {len(all_symbols)}, Done: {len(processed)}, Remaining: {len(symbols)}")

# Process batch of 20
batch = symbols[:20]
for ticker in batch:
    process_ticker(ticker)

log_message(f"Batch complete: {len(batch)} stocks")
EOF

# Check remaining
REMAINING=$(python3 -c "
import os, pandas as pd
symbols = pd.read_csv('data/signals_v3/all_signals_v3_final.csv')['ticker'].tolist()
processed = []
for s in symbols:
    d = f'data/daily_data_v2/{s}'
    if os.path.exists(d) and len([f for f in os.listdir(d) if f.endswith('.csv')]) >= 3:
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
