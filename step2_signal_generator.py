#!/usr/bin/env python3
"""
Step 2: Signal Generation Pipeline
Generates trading signals from raw yfinance data.
Includes technical indicators, pattern recognition, and strength scoring.
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Dict, List
from glob import glob

# Input/Output
RAW_DATA_DIR = "data/raw_tickers"
OUTPUT_DIR = "data/signals_v2"

def load_raw_data(symbol: str) -> Optional[Dict]:
    """Load raw yfinance data for a ticker"""
    file_path = f"{RAW_DATA_DIR}/{symbol}.json"
    if not os.path.exists(file_path):
        return None
    
    with open(file_path, 'r') as f:
        return json.load(f)

def reconstruct_dataframe(historical_dict: Dict) -> pd.DataFrame:
    """Reconstruct DataFrame from JSON dict"""
    if not historical_dict:
        return pd.DataFrame()
    
    df = pd.DataFrame.from_dict(historical_dict)
    # Convert string timestamps to datetime and normalize to UTC
    df.index = pd.to_datetime(df.index, utc=True)
    return df

class TechnicalIndicators:
    """Calculate technical indicators from price data"""
    
    @staticmethod
    def sma(prices: pd.Series, period: int) -> pd.Series:
        return prices.rolling(window=period).mean()
    
    @staticmethod
    def ema(prices: pd.Series, period: int) -> pd.Series:
        return prices.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(prices: pd.Series, period: int = 20, std_dev: int = 2):
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, sma, lower_band
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

class SignalGenerator:
    """Generate trading signals with strength scoring"""
    
    def __init__(self):
        self.indicators = TechnicalIndicators()
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators"""
        df['SMA_20'] = self.indicators.sma(df['Close'], 20)
        df['SMA_50'] = self.indicators.sma(df['Close'], 50)
        df['EMA_12'] = self.indicators.ema(df['Close'], 12)
        df['RSI'] = self.indicators.rsi(df['Close'], 14)
        df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = self.indicators.macd(df['Close'])
        df['BB_Upper'], df['BB_Mid'], df['BB_Lower'] = self.indicators.bollinger_bands(df['Close'])
        df['ATR'] = self.indicators.atr(df['High'], df['Low'], df['Close'])
        return df
    
    def generate_signal(self, symbol: str, raw_data: Dict) -> Optional[Dict]:
        """Generate comprehensive signal with strength scoring"""
        
        # Reconstruct price data
        hist_dict = raw_data.get('historical_prices')
        if not hist_dict:
            return None
        
        df = reconstruct_dataframe(hist_dict)
        if len(df) < 50:
            return None
        
        # Calculate indicators
        df = self.calculate_all_indicators(df)
        
        # Get latest values
        latest = df.iloc[-1]
        latest_close = latest['Close']
        latest_volume = latest['Volume']
        
        # Technical values
        sma_20 = latest['SMA_20']
        sma_50 = latest['SMA_50']
        rsi = latest['RSI']
        macd = latest['MACD']
        macd_signal = latest['MACD_Signal']
        macd_hist = latest['MACD_Hist']
        bb_upper = latest['BB_Upper']
        bb_lower = latest['BB_Lower']
        atr = latest['ATR']
        
        # Get fundamentals from info
        info = raw_data.get('info') or {}
        trailing_pe = info.get('trailingPE')
        forward_pe = info.get('forwardPE')
        peg_ratio = info.get('pegRatio')
        beta = info.get('beta')
        target_high = info.get('targetHighPrice')
        target_low = info.get('targetLowPrice')
        target_mean = info.get('targetMeanPrice')
        recommendation = info.get('recommendationKey', 'none')
        
        # Calculate price changes
        prev_close = df['Close'].iloc[-2] if len(df) > 1 else latest_close
        week_ago = df['Close'].iloc[-7] if len(df) > 7 else df['Close'].iloc[0]
        month_ago = df['Close'].iloc[-30] if len(df) > 30 else df['Close'].iloc[0]
        
        price_change_1d = ((latest_close - prev_close) / prev_close * 100)
        price_change_7d = ((latest_close - week_ago) / week_ago * 100)
        price_change_30d = ((latest_close - month_ago) / month_ago * 100)
        
        # Volume analysis
        avg_volume = df['Volume'].mean()
        relative_volume = (latest_volume / avg_volume) if avg_volume > 0 else 1.0
        
        # 52-week position
        year_high = info.get('fiftyTwoWeekHigh')
        year_low = info.get('fiftyTwoWeekLow')
        week_52_position = None
        if year_high and year_low and year_high > year_low:
            week_52_position = ((latest_close - year_low) / (year_high - year_low)) * 100
        
        # ============================================
        # SIGNAL SCORING LOGIC
        # ============================================
        
        buy_score = 0
        sell_score = 0
        factors = []
        
        # TREND ANALYSIS
        if not pd.isna(sma_20) and not pd.isna(sma_50):
            if latest_close > sma_20 > sma_50:
                buy_score += 1
                factors.append("Uptrend (price > SMA20 > SMA50)")
            elif latest_close < sma_20 < sma_50:
                sell_score += 1
                factors.append("Downtrend (price < SMA20 < SMA50)")
            elif latest_close < sma_20 and latest_close < sma_50:
                sell_score += 2
                factors.append("Strong Downtrend (below both SMAs)")
        
        # RSI ANALYSIS
        if not pd.isna(rsi):
            if rsi < 30:
                buy_score += 2
                factors.append(f"RSI Oversold ({rsi:.1f})")
            elif rsi < 40:
                buy_score += 1
                factors.append(f"RSI Near Oversold ({rsi:.1f})")
            elif rsi > 70:
                sell_score += 2
                factors.append(f"RSI Overbought ({rsi:.1f})")
            elif rsi > 60:
                sell_score += 1
                factors.append(f"RSI Elevated ({rsi:.1f})")
        
        # MACD ANALYSIS
        if not pd.isna(macd) and not pd.isna(macd_signal):
            if macd > macd_signal and macd_signal < 0:
                buy_score += 2
                factors.append("MACD Bullish Crossover")
            elif macd > macd_signal:
                buy_score += 1
                factors.append("MACD Above Signal")
            elif macd < macd_signal and macd_signal > 0:
                sell_score += 2
                factors.append("MACD Bearish Crossover")
            elif macd < macd_signal:
                sell_score += 1
                factors.append("MACD Below Signal")
        
        # BOLLINGER BANDS
        if not pd.isna(bb_upper) and not pd.isna(bb_lower):
            if latest_close < bb_lower:
                buy_score += 2
                factors.append("Price Below Lower BB")
            elif latest_close > bb_upper:
                sell_score += 2
                factors.append("Price Above Upper BB")
        
        # FUNDAMENTAL ANALYSIS
        if trailing_pe:
            try:
                pe = float(trailing_pe)
                if 0 < pe < 15:
                    buy_score += 2
                    factors.append(f"Low P/E ({pe:.1f})")
                elif 0 < pe < 25:
                    buy_score += 1
                    factors.append(f"Reasonable P/E ({pe:.1f})")
                elif pe > 50:
                    sell_score += 2
                    factors.append(f"High P/E ({pe:.1f})")
                elif pe < 0:
                    sell_score += 1
                    factors.append("Negative P/E (unprofitable)")
            except:
                pass
        
        if peg_ratio:
            try:
                peg = float(peg_ratio)
                if 0 < peg < 1:
                    buy_score += 2
                    factors.append(f"Low PEG ({peg:.2f})")
                elif peg > 2:
                    sell_score += 1
                    factors.append(f"High PEG ({peg:.2f})")
            except:
                pass
        
        # VOLUME ANALYSIS
        if relative_volume > 2:
            if price_change_1d > 0:
                buy_score += 1
                factors.append("High Volume + Price Up")
            else:
                sell_score += 1
                factors.append("High Volume + Price Down")
        
        # ANALYST RECOMMENDATIONS
        if recommendation:
            if recommendation in ['strong_buy', 'buy']:
                buy_score += 1
                factors.append(f"Analyst {recommendation}")
            elif recommendation in ['strong_sell', 'sell']:
                sell_score += 1
                factors.append(f"Analyst {recommendation}")
        
        # PRICE TARGET ANALYSIS
        if target_mean and target_mean > 0:
            upside = ((target_mean - latest_close) / latest_close * 100)
            if upside > 20:
                buy_score += 1
                factors.append(f"High Upside Potential ({upside:.1f}%)")
            elif upside < -20:
                sell_score += 1
                factors.append(f"Downside Risk ({upside:.1f}%)")
        
        # ============================================
        # SIGNAL CLASSIFICATION WITH STRENGTH
        # ============================================
        
        signal = None
        confidence = 50
        
        if buy_score > sell_score:
            # BUY signals
            if buy_score >= 7:
                signal = "STRONG_BUY"
                confidence = 95
            elif buy_score >= 4:
                signal = "BUY"
                confidence = 80
            else:
                signal = "WEAK_BUY"
                confidence = 65
        elif sell_score > buy_score:
            # SELL signals
            if sell_score >= 7:
                signal = "STRONG_SELL"
                confidence = 95
            elif sell_score >= 4:
                signal = "SELL"
                confidence = 80
            else:
                signal = "WEAK_SELL"
                confidence = 65
        else:
            signal = "HOLD"
            confidence = 50
        
        return {
            'ticker': symbol,
            'signal': signal,
            'confidence': confidence,
            'buy_score': buy_score,
            'sell_score': sell_score,
            'current_price': round(latest_close, 4),
            'price_change_1d': round(price_change_1d, 2),
            'price_change_7d': round(price_change_7d, 2),
            'price_change_30d': round(price_change_30d, 2),
            'rsi': round(rsi, 2) if not pd.isna(rsi) else None,
            'sma_20': round(sma_20, 4) if not pd.isna(sma_20) else None,
            'sma_50': round(sma_50, 4) if not pd.isna(sma_50) else None,
            'macd': round(macd, 4) if not pd.isna(macd) else None,
            'macd_signal': round(macd_signal, 4) if not pd.isna(macd_signal) else None,
            'relative_volume': round(relative_volume, 2),
            'week_52_position_pct': round(week_52_position, 2) if week_52_position else None,
            'trailing_pe': trailing_pe,
            'forward_pe': forward_pe,
            'peg_ratio': peg_ratio,
            'beta': beta,
            'target_mean': target_mean,
            'upside_pct': round(((target_mean - latest_close) / latest_close * 100), 2) if target_mean else None,
            'analyst_recommendation': recommendation,
            'factors': '; '.join(factors) if factors else None,
            'analyzed_at': datetime.now().isoformat()
        }

def main():
    print("=" * 80)
    print("STEP 2: SIGNAL GENERATION PIPELINE")
    print("=" * 80)
    
    # Load all raw data files
    if not os.path.exists(RAW_DATA_DIR):
        print(f"Error: {RAW_DATA_DIR} does not exist. Run Step 1 first.")
        return
    
    raw_files = glob(f"{RAW_DATA_DIR}/*.json")
    print(f"Found {len(raw_files)} raw data files")
    
    # Initialize generator
    generator = SignalGenerator()
    
    # Generate signals for all tickers
    signals = []
    processed = 0
    errors = 0
    
    for i, file_path in enumerate(raw_files):
        symbol = os.path.basename(file_path).replace('.json', '')
        
        if i % 100 == 0:
            print(f"\nProcessing {i+1}/{len(raw_files)}: {symbol}")
        
        try:
            raw_data = load_raw_data(symbol)
            if not raw_data:
                continue
            
            signal = generator.generate_signal(symbol, raw_data)
            if signal:
                signals.append(signal)
                processed += 1
                
                # Print strong signals
                if signal['signal'] in ['STRONG_BUY', 'STRONG_SELL']:
                    print(f"  ⭐ {signal['signal']}: {symbol} ({signal['confidence']}%) - Buy:{signal['buy_score']}/Sell:{signal['sell_score']}")
        except Exception as e:
            errors += 1
            if i % 100 == 0:
                print(f"  Error: {e}")
    
    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save to CSV
    df_signals = pd.DataFrame(signals)
    csv_path = f"{OUTPUT_DIR}/all_signals_v2.csv"
    df_signals.to_csv(csv_path, index=False)
    
    # Save to JSON
    json_path = f"{OUTPUT_DIR}/all_signals_v2.json"
    with open(json_path, 'w') as f:
        json.dump(signals, f, indent=2, default=str)
    
    # Print summary
    print("\n" + "=" * 80)
    print("SIGNAL GENERATION COMPLETE")
    print("=" * 80)
    print(f"Total processed: {processed}")
    print(f"Errors: {errors}")
    print(f"\nSignal distribution:")
    print(df_signals['signal'].value_counts())
    
    print(f"\nTop STRONG_BUY signals:")
    strong_buys = df_signals[df_signals['signal'] == 'STRONG_BUY'].nlargest(10, 'confidence')
    for _, row in strong_buys.iterrows():
        print(f"  {row['ticker']}: {row['confidence']}% - {row['factors'][:80]}")
    
    print(f"\nTop STRONG_SELL signals:")
    strong_sells = df_signals[df_signals['signal'] == 'STRONG_SELL'].nlargest(10, 'confidence')
    for _, row in strong_sells.iterrows():
        print(f"  {row['ticker']}: {row['confidence']}% - {row['factors'][:80]}")
    
    print(f"\nFiles saved:")
    print(f"  - {csv_path}")
    print(f"  - {json_path}")

if __name__ == "__main__":
    main()
