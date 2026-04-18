#!/bin/bash
# Process stocks in batches with full restart between batches
# This script exits completely after each batch to free all memory

cd /home/ubuntu/.openclaw/workspace
source venv/bin/activate

# Get remaining count
REMAINING=$(python3 -c "
import glob, os, pandas as pd
symbols = [os.path.basename(f).replace('.json', '') for f in glob.glob('data/raw_tickers/*.json')]
if os.path.exists('data/signals_v3/all_signals_v3.csv'):
    done = set(pd.read_csv('data/signals_v3/all_signals_v3.csv')['ticker'].tolist())
    symbols = [s for s in symbols if s not in done]
print(len(symbols))
")

echo "Remaining: $REMAINING stocks"

if [ "$REMAINING" -eq "0" ]; then
    echo "All stocks complete!"
    exit 0
fi

# Process just 20 stocks then exit
python3 << 'EOF'
import os
import sys
import json
import gc
import glob
import subprocess
import pandas as pd
from datetime import datetime

RAW_TICKERS_DIR = 'data/raw_tickers'
HISTORICAL_DATA_DIR = 'data/historical_stock_data'
OUTPUT_DIR = 'data/signals_v3'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Get remaining
json_files = glob.glob(f'{RAW_TICKERS_DIR}/*.json')
symbols = [os.path.basename(f).replace('.json', '') for f in json_files]

if os.path.exists(f'{OUTPUT_DIR}/all_signals_v3.csv'):
    done_df = pd.read_csv(f'{OUTPUT_DIR}/all_signals_v3.csv')
    done = set(done_df['ticker'].tolist())
    symbols = [s for s in symbols if s not in done]

print(f"Processing batch of 20 from {len(symbols)} remaining...")

# Take first 20
batch = symbols[:20]

for symbol in batch:
    try:
        json_path = f'{RAW_TICKERS_DIR}/{symbol}.json'
        csv_path = f'{HISTORICAL_DATA_DIR}/{symbol}.csv'
        
        if not os.path.exists(json_path) or not os.path.exists(csv_path):
            continue
        
        with open(json_path, 'r') as f:
            raw = json.load(f)
        
        df = pd.read_csv(csv_path)
        if len(df) < 20:
            continue
        
        # Use 90 days
        df = df.tail(90).copy()
        current_price = df['Close'].iloc[-1]
        
        # S/R
        window = 3
        res_prices = []
        sup_prices = []
        highs = df['High'].values
        lows = df['Low'].values
        
        for i in range(window, len(df) - window):
            if highs[i] >= max(highs[i-window:i]) and highs[i] >= max(highs[i+1:i+window+1]):
                if highs[i] > current_price:
                    res_prices.append(highs[i])
            if lows[i] <= min(lows[i-window:i]) and lows[i] <= min(lows[i+1:i+window+1]):
                if lows[i] < current_price:
                    sup_prices.append(lows[i])
        
        # Cluster 2%
        tol = 0.02
        def cluster(prices):
            if not prices:
                return []
            prices = sorted(set([round(p, 2) for p in prices]))
            result, current = [], [prices[0]]
            for p in prices[1:]:
                if abs(p - current[0]) / current[0] <= tol:
                    current.append(p)
                else:
                    result.append(round(sum(current)/len(current), 2))
                    current = [p]
            result.append(round(sum(current)/len(current), 2))
            return result[:10]
        
        res_levels = cluster(res_prices)
        sup_levels = cluster(sup_prices)
        
        res_json = json.dumps([{'price': p, 'type': 'pivot', 'touches': 1, 'strength': 50,
                               'distance_pct': round((p - current_price) / current_price * 100, 2)} for p in res_levels])
        sup_json = json.dumps([{'price': p, 'type': 'pivot', 'touches': 1, 'strength': 50,
                               'distance_pct': round((current_price - p) / current_price * 100, 2)} for p in sup_levels])
        
        peaks = [{'price': round(highs[i], 2)} for i in range(window, len(df) - window) 
                 if highs[i] >= max(highs[i-window:i+window+1])][-5:]
        troughs = [{'price': round(lows[i], 2)} for i in range(window, len(df) - window)
                   if lows[i] <= min(lows[i-window:i+window+1])][-5:]
        
        close = df['Close']
        sma_20 = close.tail(20).mean()
        sma_50 = close.tail(50).mean() if len(close) >= 50 else sma_20
        
        price_1d = (close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100 if len(close) > 1 else 0
        price_7d = (close.iloc[-1] - close.iloc[-8]) / close.iloc[-8] * 100 if len(close) > 7 else 0
        price_30d = (close.iloc[-1] - close.iloc[-31]) / close.iloc[-31] * 100 if len(close) > 30 else 0
        
        buy_score, sell_score, factors = 0, 0, []
        
        if current_price > sma_20 > sma_50:
            buy_score = 2
            factors.append('Golden Stack')
        elif current_price < sma_20 < sma_50:
            sell_score = 2
            factors.append('Death Stack')
        
        if sup_levels and sup_levels[0] > current_price * 0.98:
            buy_score += 1
            factors.append(f'Near support ${sup_levels[0]}')
        if res_levels and res_levels[0] < current_price * 1.02:
            sell_score += 1
            factors.append(f'Near resistance ${res_levels[0]}')
        
        if buy_score >= 3:
            signal, conf = 'BUY', 75
        elif buy_score >= 1:
            signal, conf = 'WEAK_BUY', 60
        elif sell_score >= 3:
            signal, conf = 'SELL', 75
        elif sell_score >= 1:
            signal, conf = 'WEAK_SELL', 60
        else:
            signal, conf = 'HOLD', 55
        
        info = raw.get('info', {})
        
        result = {
            'ticker': symbol, 'signal': signal, 'confidence': conf,
            'buy_score': buy_score, 'sell_score': sell_score,
            'current_price': round(current_price, 2),
            'price_change_1d': round(price_1d, 2),
            'price_change_7d': round(price_7d, 2),
            'price_change_30d': round(price_30d, 2),
            'sma_20': round(sma_20, 4), 'sma_50': round(sma_50, 4),
            'trailing_pe': info.get('trailingPE'), 'beta': info.get('beta'),
            'resistance_levels': res_json, 'support_levels': sup_json,
            'peaks': json.dumps(peaks), 'troughs': json.dumps(troughs),
            'factors': '; '.join(factors) if factors else 'None',
            'analyzed_at': datetime.now().isoformat()
        }
        
        output_path = f'{OUTPUT_DIR}/all_signals_v3.csv'
        row_df = pd.DataFrame([result])
        if os.path.exists(output_path):
            row_df.to_csv(output_path, mode='a', header=False, index=False)
        else:
            row_df.to_csv(output_path, index=False)
        
        print(f'{symbol}: {signal}')
        
    except Exception as e:
        print(f'{symbol}: Error - {e}')

print(f'Batch complete. {len(batch)} stocks processed.')
EOF

# Check if more remain
REMAINING=$(python3 -c "
import glob, os, pandas as pd
symbols = [os.path.basename(f).replace('.json', '') for f in glob.glob('data/raw_tickers/*.json')]
if os.path.exists('data/signals_v3/all_signals_v3.csv'):
    done = set(pd.read_csv('data/signals_v3/all_signals_v3.csv')['ticker'].tolist())
    symbols = [s for s in symbols if s not in done]
print(len(symbols))")

echo ""
echo "Remaining after batch: $REMAINING"

if [ "$REMAINING" -gt "0" ]; then
    echo "Restarting script for next batch..."
    exec bash "$0"  # Restart self
else
    echo "All complete!"
    # Final summary
    python3 -c "
import pandas as pd
df = pd.read_csv('data/signals_v3/all_signals_v3.csv')
print(f'\nTotal: {len(df)} stocks')
print('Distribution:')
print(df['signal'].value_counts())
"
fi
