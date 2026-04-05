#!/usr/bin/env python3
"""
Batched Signal Scanner - Processes stocks in batches with delays
Avoids Yahoo Finance rate limiting
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from typing import Dict, List, Optional
import time
import warnings
warnings.filterwarnings('ignore')

# Rate limiting configuration
BATCH_SIZE = 500         # 500 stocks per batch
DELAY_BETWEEN_BATCHES = 60   # 1 minute between batches
DELAY_BETWEEN_CALLS = 0.8    # 0.8 seconds between API calls
CHECKPOINT_FILE = "data/signals/scan_checkpoint.json"

class BatchedDataFetcher:
    """Fetch data with rate limiting"""
    
    def __init__(self):
        self.cache = {}
        self.api_calls = 0
    
    def fetch_with_retry(self, symbol: str, max_retries: int = 3) -> Optional[pd.DataFrame]:
        """Fetch with rate limiting and retry logic"""
        for attempt in range(max_retries):
            try:
                # Rate limit between calls
                time.sleep(DELAY_BETWEEN_CALLS)
                
                ticker = yf.Ticker(symbol)
                df = ticker.history(period="60d", interval="1d")
                self.api_calls += 1
                
                if len(df) > 0:
                    return df
                return None
                
            except Exception as e:
                if "Rate limited" in str(e) or "Too Many Requests" in str(e):
                    wait_time = 60 * (attempt + 1)  # 1min, 2min, 3min
                    print(f"    Rate limited! Waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    return None
        return None
    
    def fetch_fundamentals_with_retry(self, symbol: str, max_retries: int = 3) -> Dict:
        """Fetch fundamentals with rate limiting"""
        data = {
            'symbol': symbol,
            'trailing_pe': None, 'forward_pe': None, 'peg_ratio': None,
            'price_to_book': None, 'profit_margins': None, 'revenue_growth': None,
            'return_on_equity': None, 'debt_to_equity': None, 'beta': None,
            'held_percent_institutions': None, 'short_percent_of_float': None,
            'target_mean_price': None, 'recommendation_key': None,
            'year_high': None, 'year_low': None, 'average_volume': None,
            'fifty_day_average': None, 'two_hundred_day_average': None,
        }
        
        for attempt in range(max_retries):
            try:
                time.sleep(DELAY_BETWEEN_CALLS)
                
                ticker = yf.Ticker(symbol)
                self.api_calls += 1
                
                # Get fast_info
                fast = ticker.fast_info
                data['year_high'] = getattr(fast, 'year_high', None)
                data['year_low'] = getattr(fast, 'year_low', None)
                data['average_volume'] = getattr(fast, 'three_month_average_volume', None)
                data['fifty_day_average'] = getattr(fast, 'fifty_day_average', None)
                data['two_hundred_day_average'] = getattr(fast, 'two_hundred_day_average', None)
                
                # Get full info
                time.sleep(0.5)  # Small delay between info calls
                info = ticker.info
                self.api_calls += 1
                
                data['trailing_pe'] = info.get('trailingPE')
                data['forward_pe'] = info.get('forwardPE')
                data['peg_ratio'] = info.get('pegRatio')
                data['price_to_book'] = info.get('priceToBook')
                data['profit_margins'] = info.get('profitMargins')
                data['revenue_growth'] = info.get('revenueGrowth')
                data['return_on_equity'] = info.get('returnOnEquity')
                data['debt_to_equity'] = info.get('debtToEquity')
                data['beta'] = info.get('beta')
                data['held_percent_institutions'] = info.get('heldPercentInstitutions')
                data['short_percent_of_float'] = info.get('shortPercentOfFloat')
                data['target_mean_price'] = info.get('targetMeanPrice')
                data['recommendation_key'] = info.get('recommendationKey')
                
                return data
                
            except Exception as e:
                if "Rate limited" in str(e) or "Too Many Requests" in str(e):
                    wait_time = 60 * (attempt + 1)
                    print(f"    Rate limited on fundamentals! Waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    return data
        return data


class SimpleSignalGenerator:
    """Simplified signal generation"""
    
    def sma(self, prices: pd.Series, period: int) -> pd.Series:
        return prices.rolling(window=period).mean()
    
    def rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        return macd_line, signal_line, None
    
    def generate_signal(self, df: pd.DataFrame, fundamentals: Dict) -> Optional[Dict]:
        """Generate simple signal"""
        if df is None or len(df) < 50:
            return None
        
        latest = df.iloc[-1]
        latest_close = latest['Close']
        latest_volume = latest['Volume']
        
        # Technicals
        df['SMA_20'] = self.sma(df['Close'], 20)
        df['SMA_50'] = self.sma(df['Close'], 50)
        df['RSI'] = self.rsi(df['Close'], 14)
        df['MACD'], df['MACD_Signal'], _ = self.macd(df['Close'])
        
        sma_20 = df['SMA_20'].iloc[-1]
        sma_50 = df['SMA_50'].iloc[-1]
        rsi = df['RSI'].iloc[-1]
        
        # Price changes
        prev_close = df['Close'].iloc[-2] if len(df) > 1 else latest_close
        week_ago = df['Close'].iloc[-7] if len(df) > 7 else df['Close'].iloc[0]
        price_change_1d = ((latest_close - prev_close) / prev_close * 100)
        price_change_7d = ((latest_close - week_ago) / week_ago * 100)
        
        # Relative volume
        avg_volume = fundamentals.get('average_volume') or df['Volume'].mean()
        relative_volume = (latest_volume / avg_volume) if avg_volume and avg_volume > 0 else 1.0
        
        # 52-week position
        year_high = fundamentals.get('year_high')
        year_low = fundamentals.get('year_low')
        week_52_position = None
        if year_high and year_low and year_high > year_low:
            week_52_position = ((latest_close - year_low) / (year_high - year_low)) * 100
        
        # Simple signal logic
        buy_score = 0
        sell_score = 0
        factors = []
        
        if not pd.isna(sma_20) and not pd.isna(sma_50):
            if latest_close > sma_20 > sma_50:
                buy_score += 1
                factors.append("Uptrend")
        
        if not pd.isna(rsi):
            if rsi < 30:
                buy_score += 2
                factors.append(f"RSI oversold ({rsi:.1f})")
            elif rsi > 70:
                sell_score += 2
                factors.append(f"RSI overbought ({rsi:.1f})")
        
        # P/E check
        pe = fundamentals.get('trailing_pe')
        if pe:
            try:
                pe = float(pe)
                if 0 < pe < 15:
                    buy_score += 1
                    factors.append(f"Low P/E ({pe:.1f})")
                elif pe > 50:
                    sell_score += 2
                    factors.append(f"High P/E ({pe:.1f})")
            except:
                pass
        
        # MACD bearish
        macd = df['MACD'].iloc[-1]
        macd_signal = df['MACD_Signal'].iloc[-1]
        if not pd.isna(macd) and not pd.isna(macd_signal):
            if macd < macd_signal and macd_signal > 0:
                sell_score += 2
                factors.append("MACD bearish crossover")
        
        # Downtrend
        if not pd.isna(sma_20) and not pd.isna(sma_50):
            if latest_close < sma_20 < sma_50:
                sell_score += 2
                factors.append("Price below SMAs (downtrend)")
        
        if buy_score > sell_score:
            if buy_score >= 5:
                signal = "STRONG_BUY"
            elif buy_score >= 3:
                signal = "BUY"
            else:
                signal = "WEAK_BUY"
            confidence = 50 + buy_score * 9
        elif sell_score > buy_score:
            if sell_score >= 5:
                signal = "STRONG_SELL"
            elif sell_score >= 3:
                signal = "SELL"
            else:
                signal = "WEAK_SELL"
            confidence = 50 + sell_score * 9
        else:
            signal = "HOLD"
            confidence = 50
        
        return {
            'ticker': fundamentals['symbol'],
            'signal': signal,
            'confidence': min(confidence, 95),
            'current_price': round(latest_close, 4),
            'price_change_1d': round(price_change_1d, 2),
            'price_change_7d': round(price_change_7d, 2),
            'rsi': round(rsi, 2) if not pd.isna(rsi) else None,
            'sma_20': round(sma_20, 4) if not pd.isna(sma_20) else None,
            'sma_50': round(sma_50, 4) if not pd.isna(sma_50) else None,
            'relative_volume': round(relative_volume, 2),
            'week_52_position_pct': round(week_52_position, 2) if week_52_position else None,
            'trailing_pe': fundamentals.get('trailing_pe'),
            'beta': fundamentals.get('beta'),
            'recommendation_key': fundamentals.get('recommendation_key'),
            'factors': ', '.join(factors),
            'analyzed_at': datetime.now().isoformat()
        }


def save_checkpoint(processed_tickers: List[str], signals_count: int):
    """Save progress checkpoint"""
    checkpoint = {
        'processed_tickers': processed_tickers[-100:],  # Only last 100 to save memory
        'signals_count': signals_count,
        'last_update': datetime.now().isoformat()
    }
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f)


def load_checkpoint() -> tuple:
    """Load checkpoint if exists"""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            checkpoint = json.load(f)
        return checkpoint.get('processed_tickers', []), checkpoint.get('signals_count', 0)
    return [], 0


def main():
    print("=" * 80)
    print("BATCHED SIGNAL SCANNER")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Delay between calls: {DELAY_BETWEEN_CALLS}s")
    print(f"Delay between batches: {DELAY_BETWEEN_BATCHES}s")
    
    # Load tickers
    df_base = pd.read_csv("data/tickers/stock_ticker_base.csv")
    all_tickers = df_base.to_dict('records')
    print(f"\nTotal tickers to process: {len(all_tickers)}")
    
    # Load checkpoint
    processed_tickers, existing_signals_count = load_checkpoint()
    print(f"Previously processed: {len(processed_tickers)}")
    
    # Filter out already processed
    remaining_tickers = [t for t in all_tickers if t['symbol'] not in processed_tickers]
    print(f"Remaining: {len(remaining_tickers)}")
    
    # Initialize
    fetcher = BatchedDataFetcher()
    generator = SimpleSignalGenerator()
    signals = []
    batch_count = 0
    
    # Process in batches
    for i in range(0, len(remaining_tickers), BATCH_SIZE):
        batch = remaining_tickers[i:i+BATCH_SIZE]
        batch_count += 1
        
        print(f"\n{'='*80}")
        print(f"BATCH {batch_count}: Processing {len(batch)} tickers ({i+1}-{min(i+BATCH_SIZE, len(remaining_tickers))})")
        print(f"{'='*80}")
        
        batch_signals = []
        
        for j, ticker_data in enumerate(batch):
            symbol = ticker_data.get('symbol')
            if not symbol:
                continue
            
            print(f"  [{j+1}/{len(batch)}] {symbol}...", end=' ', flush=True)
            
            # Fetch data
            df = fetcher.fetch_with_retry(symbol)
            if df is None:
                print("No price data")
                processed_tickers.append(symbol)
                continue
            
            fundamentals = fetcher.fetch_fundamentals_with_retry(symbol)
            fundamentals['symbol'] = symbol
            
            # Generate signal
            signal = generator.generate_signal(df, fundamentals)
            
            if signal:
                batch_signals.append(signal)
                print(f"Signal: {signal['signal']} ({signal['confidence']}%)")
            else:
                print("No signal generated")
            
            processed_tickers.append(symbol)
        
        signals.extend(batch_signals)
        
        # Memory management: Save signals to disk and clear from memory every 500 signals
        if len(signals) >= 500:
            print(f"\n  Memory management: Saving {len(signals)} signals to disk...")
            output_dir = "data/signals"
            os.makedirs(output_dir, exist_ok=True)
            
            csv_path = f"{output_dir}/batched_signals.csv"
            df_batch = pd.DataFrame(signals)
            
            if os.path.exists(csv_path):
                df_batch.to_csv(csv_path, mode='a', header=False, index=False)
            else:
                df_batch.to_csv(csv_path, index=False)
            
            # Clear from memory
            signals.clear()
            print(f"  Cleared from memory, continuing...")
        
        # Save checkpoint (just count, not full signals)
        save_checkpoint(processed_tickers, len(signals) + (batch_count - 1) * 500)
        
        print(f"\n  Batch {batch_count} complete: {len(batch_signals)} signals")
        print(f"  Running total: ~{len(signals) + (batch_count - 1) * 500} signals")
        
        # Wait between batches
        if i + BATCH_SIZE < len(remaining_tickers):
            print(f"\n  Waiting {DELAY_BETWEEN_BATCHES}s before next batch...")
            time.sleep(DELAY_BETWEEN_BATCHES)
            fetcher.api_calls = 0

    # Final save of remaining signals
    if signals:
        output_dir = "data/signals"
        csv_path = f"{output_dir}/batched_signals.csv"
        df_final = pd.DataFrame(signals)
        
        if os.path.exists(csv_path):
            df_final.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            df_final.to_csv(csv_path, index=False)
    
    # Final summary
    print("\n" + "=" * 80)
    print("SCAN COMPLETE")
    print("=" * 80)
    print(f"Total processed: {len(processed_tickers)}")
    
    # Count total signals from CSV
    try:
        df_all = pd.read_csv(f"{output_dir}/batched_signals.csv")
        print(f"Total signals saved: {len(df_all)}")
        
        # Signal breakdown
        print(f"\nSignal distribution:")
        print(df_all['signal'].value_counts())
        
        # Top buys
        buys = df_all[df_all['signal'].str.contains('BUY', na=False)]
        if len(buys) > 0:
            print(f"\n--- Top BUY Signals ---")
            top_buys = buys.nlargest(5, 'confidence')
            for _, row in top_buys.iterrows():
                print(f"  {row['ticker']}: {row['signal']} ({row['confidence']}%)")
                
    except Exception as e:
        print(f"Error reading final results: {e}")
    
    print(f"\nResults saved to:")
    print(f"  - data/signals/batched_signals.csv")
    print(f"  - {CHECKPOINT_FILE}")


if __name__ == "__main__":
    main()
