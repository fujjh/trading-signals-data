#!/usr/bin/env python3
"""
Signal Generator v3.0 - Enhanced with Support/Resistance Levels
Integrates 3 S/R calculation methods + existing technical/fundamental analysis
"""

import os
import sys
import json
import glob
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
from support_resistance_calculator import SupportResistanceCalculator, SupportResistanceLevel

# Configuration
RAW_TICKERS_DIR = "data/raw_tickers"
HISTORICAL_DATA_DIR = "data/historical_stock_data"
OUTPUT_DIR = "data/signals_v3"

class TechnicalIndicators:
    """Calculate technical indicators from price data"""
    
    @staticmethod
    def sma(data: pd.Series, period: int) -> pd.Series:
        return data.rolling(window=period).mean()
    
    @staticmethod
    def ema(data: pd.Series, period: int) -> pd.Series:
        return data.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def rsi(data: pd.Series, period: int = 14) -> pd.Series:
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
        ema_fast = data.ewm(span=fast, adjust=False).mean()
        ema_slow = data.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(data: pd.Series, period: int = 20, std_dev: int = 2) -> tuple:
        sma = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()
    
    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        obv = pd.Series(index=close.index)
        obv.iloc[0] = volume.iloc[0]
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        return obv
    
    @staticmethod
    def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        typical_price = (high + low + close) / 3
        return (typical_price * volume).cumsum() / volume.cumsum()


class EnhancedSignalGenerator:
    """Generate trading signals with support/resistance levels"""
    
    def __init__(self):
        self.sr_calculator = SupportResistanceCalculator(
            lookback_days=252,  # 1 year
            min_touches=2,
            cluster_tolerance=0.02  # 2%
        )
    
    def load_ticker_data(self, symbol: str) -> Optional[Dict]:
        """Load raw JSON data and historical CSV for a ticker"""
        
        # Load raw JSON
        json_path = f"{RAW_TICKERS_DIR}/{symbol}.json"
        if not os.path.exists(json_path):
            return None
        
        with open(json_path, 'r') as f:
            raw_data = json.load(f)
        
        # Load historical CSV
        csv_path = f"{HISTORICAL_DATA_DIR}/{symbol}.csv"
        if not os.path.exists(csv_path):
            return None
        
        df = pd.read_csv(csv_path)
        if len(df) < 20:
            return None
        
        return {
            'symbol': symbol,
            'raw': raw_data,
            'df': df
        }
    
    def calculate_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate all technical indicators"""
        
        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']
        
        indicators = {
            'sma_20': TechnicalIndicators.sma(close, 20).iloc[-1] if len(close) >= 20 else None,
            'sma_50': TechnicalIndicators.sma(close, 50).iloc[-1] if len(close) >= 50 else None,
            'sma_200': TechnicalIndicators.sma(close, 200).iloc[-1] if len(close) >= 200 else None,
            'ema_12': TechnicalIndicators.ema(close, 12).iloc[-1],
            'ema_26': TechnicalIndicators.ema(close, 26).iloc[-1],
            'rsi': TechnicalIndicators.rsi(close).iloc[-1],
            'macd': None,
            'macd_signal': None,
            'bb_upper': None,
            'bb_middle': None,
            'bb_lower': None,
            'atr': TechnicalIndicators.atr(high, low, close).iloc[-1],
            'obv': TechnicalIndicators.obv(close, volume).iloc[-1],
            'vwap': TechnicalIndicators.vwap(high, low, close, volume).iloc[-1],
        }
        
        # MACD
        if len(close) >= 26:
            macd_line, signal_line, _ = TechnicalIndicators.macd(close)
            indicators['macd'] = macd_line.iloc[-1]
            indicators['macd_signal'] = signal_line.iloc[-1]
        
        # Bollinger Bands
        if len(close) >= 20:
            bb_upper, bb_middle, bb_lower = TechnicalIndicators.bollinger_bands(close)
            indicators['bb_upper'] = bb_upper.iloc[-1]
            indicators['bb_middle'] = bb_middle.iloc[-1]
            indicators['bb_lower'] = bb_lower.iloc[-1]
        
        return indicators
    
    def calculate_support_resistance(self, df: pd.DataFrame, current_price: float) -> Dict:
        """Calculate support and resistance levels"""
        
        resistance, support, metadata = self.sr_calculator.calculate_all_levels(
            df, current_price
        )
        
        return {
            'resistance_levels': resistance,
            'support_levels': support,
            'peaks': metadata['peaks'],
            'troughs': metadata['troughs'],
            'metadata': metadata
        }
    
    def generate_signal(self, ticker_data: Dict) -> Optional[Dict]:
        """Generate trading signal with full analysis including S/R levels"""
        
        symbol = ticker_data['symbol']
        raw = ticker_data['raw']
        df = ticker_data['df']
        
        if len(df) < 20:
            return None
        
        # Current price
        current_price = df['Close'].iloc[-1]
        
        # Calculate indicators
        indicators = self.calculate_indicators(df)
        
        # Calculate support/resistance
        sr_data = self.calculate_support_resistance(df, current_price)
        
        # Calculate price changes
        price_changes = {
            '1d': (df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2] * 100 if len(df) > 1 else 0,
            '7d': (df['Close'].iloc[-1] - df['Close'].iloc[-8]) / df['Close'].iloc[-8] * 100 if len(df) > 7 else 0,
            '30d': (df['Close'].iloc[-1] - df['Close'].iloc[-31]) / df['Close'].iloc[-31] * 100 if len(df) > 30 else 0,
        }
        
        # Volume analysis
        avg_volume = df['Volume'].tail(20).mean()
        current_volume = df['Volume'].iloc[-1]
        relative_volume = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Get fundamentals from raw data
        info = raw.get('info', {})
        trailing_pe = info.get('trailingPE')
        forward_pe = info.get('forwardPE')
        peg_ratio = info.get('pegRatio')
        beta = info.get('beta')
        target_mean = info.get('targetMeanPrice')
        analyst_rec = info.get('recommendationKey', 'none')
        
        # Calculate 52-week position
        year_high = info.get('fiftyTwoWeekHigh', df['High'].tail(252).max())
        year_low = info.get('fiftyTwoWeekLow', df['Low'].tail(252).min())
        week_52_position = (current_price - year_low) / (year_high - year_low) * 100 if year_high != year_low else 50
        
        # Scoring
        buy_score = 0
        sell_score = 0
        factors = []
        
        # RSI scoring
        if indicators['rsi'] and not pd.isna(indicators['rsi']):
            if indicators['rsi'] < 30:
                buy_score += 2
                factors.append("Oversold (RSI < 30)")
            elif indicators['rsi'] < 40:
                buy_score += 1
                factors.append("Near oversold")
            elif indicators['rsi'] > 70:
                sell_score += 2
                factors.append("Overbought (RSI > 70)")
            elif indicators['rsi'] > 60:
                sell_score += 1
                factors.append("Near overbought")
        
        # Support/Resistance proximity scoring
        if sr_data['support_levels']:
            closest_support = sr_data['support_levels'][0]
            if closest_support.distance_pct < 2:
                buy_score += 2
                factors.append(f"Near support ${closest_support.price:.2f}")
        
        if sr_data['resistance_levels']:
            closest_resistance = sr_data['resistance_levels'][0]
            if closest_resistance.distance_pct < 2:
                sell_score += 2
                factors.append(f"Near resistance ${closest_resistance.price:.2f}")
        
        # MACD scoring
        if indicators['macd'] and indicators['macd_signal']:
            if indicators['macd'] > indicators['macd_signal']:
                buy_score += 1
                if indicators['macd'] > 0:
                    buy_score += 1
                    factors.append("MACD Bullish Crossover")
                else:
                    factors.append("MACD Turning Bullish")
            else:
                sell_score += 1
                if indicators['macd'] < 0:
                    sell_score += 1
                    factors.append("MACD Bearish Crossover")
        
        # Moving average scoring
        if indicators['sma_20'] and indicators['sma_50']:
            if current_price > indicators['sma_20'] > indicators['sma_50']:
                buy_score += 1
                factors.append("Golden Stack (Price > SMA20 > SMA50)")
            elif current_price < indicators['sma_20'] < indicators['sma_50']:
                sell_score += 1
                factors.append("Death Stack (Price < SMA20 < SMA50)")
        
        # Volume scoring
        if relative_volume > 1.5:
            if price_changes['1d'] > 0:
                buy_score += 1
                factors.append(f"High Volume Breakout ({relative_volume:.1f}x)")
            elif price_changes['1d'] < 0:
                sell_score += 1
                factors.append(f"High Volume Sell-off ({relative_volume:.1f}x)")
        
        # Fundamental scoring
        if trailing_pe and trailing_pe < 20:
            buy_score += 1
            factors.append(f"Attractive P/E ({trailing_pe:.1f})")
        elif trailing_pe and trailing_pe > 40:
            sell_score += 1
            factors.append(f"High P/E ({trailing_pe:.1f})")
        
        if peg_ratio and peg_ratio < 1.5:
            buy_score += 1
            factors.append(f"Good PEG ({peg_ratio:.2f})")
        elif peg_ratio and peg_ratio > 2.5:
            sell_score += 1
            factors.append(f"Expensive PEG ({peg_ratio:.2f})")
        
        # Analyst targets
        if target_mean and current_price > 0:
            upside = (target_mean - current_price) / current_price * 100
            if upside > 20:
                buy_score += 1
                factors.append(f"Strong Upside Potential ({upside:.1f}%)")
            elif upside < -20:
                sell_score += 1
                factors.append(f"Downside Risk ({upside:.1f}%)")
        
        # Determine signal
        if buy_score >= 7:
            signal = "STRONG_BUY"
            confidence = 95
        elif buy_score >= 4:
            signal = "BUY"
            confidence = 70 + (buy_score - 4) * 8
        elif buy_score >= 1:
            signal = "WEAK_BUY"
            confidence = 55 + buy_score * 3
        elif sell_score >= 7:
            signal = "STRONG_SELL"
            confidence = 95
        elif sell_score >= 4:
            signal = "SELL"
            confidence = 70 + (sell_score - 4) * 8
        elif sell_score >= 1:
            signal = "WEAK_SELL"
            confidence = 55 + sell_score * 3
        else:
            signal = "HOLD"
            confidence = 60
        
        confidence = min(95, max(50, confidence))
        
        # Format S/R levels for output
        resistance_list = [
            {
                'price': r.price,
                'type': r.level_type,
                'touches': r.touches,
                'strength': round(r.strength_score, 1),
                'distance_pct': round(r.distance_pct, 2)
            }
            for r in sr_data['resistance_levels']
        ]
        
        support_list = [
            {
                'price': s.price,
                'type': s.level_type,
                'touches': s.touches,
                'strength': round(s.strength_score, 1),
                'distance_pct': round(s.distance_pct, 2)
            }
            for s in sr_data['support_levels']
        ]
        
        # Peaks and troughs
        recent_peaks = sr_data['peaks'][-5:] if sr_data['peaks'] else []
        recent_troughs = sr_data['troughs'][-5:] if sr_data['troughs'] else []
        
        return {
            'ticker': symbol,
            'signal': signal,
            'confidence': confidence,
            'buy_score': buy_score,
            'sell_score': sell_score,
            'current_price': round(current_price, 2),
            'price_change_1d': round(price_changes['1d'], 2),
            'price_change_7d': round(price_changes['7d'], 2),
            'price_change_30d': round(price_changes['30d'], 2),
            'rsi': round(indicators['rsi'], 2) if indicators['rsi'] else None,
            'sma_20': round(indicators['sma_20'], 4) if indicators['sma_20'] else None,
            'sma_50': round(indicators['sma_50'], 4) if indicators['sma_50'] else None,
            'macd': round(indicators['macd'], 4) if indicators['macd'] else None,
            'macd_signal': round(indicators['macd_signal'], 4) if indicators['macd_signal'] else None,
            'bb_upper': round(indicators['bb_upper'], 2) if indicators['bb_upper'] else None,
            'bb_lower': round(indicators['bb_lower'], 2) if indicators['bb_lower'] else None,
            'atr': round(indicators['atr'], 2) if indicators['atr'] else None,
            'relative_volume': round(relative_volume, 2),
            'week_52_position_pct': round(week_52_position, 2),
            'trailing_pe': trailing_pe,
            'forward_pe': forward_pe,
            'peg_ratio': peg_ratio,
            'beta': beta,
            'target_mean': target_mean,
            'upside_pct': round((target_mean - current_price) / current_price * 100, 2) if target_mean and current_price else None,
            'analyst_recommendation': analyst_rec,
            'resistance_levels': resistance_list,
            'support_levels': support_list,
            'peaks': recent_peaks,
            'troughs': recent_troughs,
            'factors': '; '.join(factors) if factors else 'None',
            'analyzed_at': datetime.now().isoformat()
        }


def main():
    print("=" * 80)
    print("ENHANCED SIGNAL GENERATOR v3.0")
    print("With Support/Resistance Analysis")
    print("=" * 80)
    print("Features:")
    print("  - 3 S/R Methods: Pivot + Volume Profile + Fibonacci")
    print("  - 1-Year Lookback with Recency Weighting")
    print("  - Minimum 2 Touches Required")
    print("  - 2% Cluster Tolerance")
    print("  - Peaks & Troughs Identification")
    print("=" * 80)
    
    generator = EnhancedSignalGenerator()
    
    # Find all tickers with data
    json_files = glob.glob(f"{RAW_TICKERS_DIR}/*.json")
    symbols = [os.path.basename(f).replace('.json', '') for f in json_files]
    
    print(f"\nFound {len(symbols)} tickers to analyze")
    
    # Check for existing output to resume
    output_path = f"{OUTPUT_DIR}/all_signals_v3.csv"
    existing_symbols = set()
    if os.path.exists(output_path):
        existing_df = pd.read_csv(output_path)
        existing_symbols = set(existing_df['ticker'].tolist())
        print(f"Resuming: {len(existing_symbols)} already processed")
    
    # Filter to unprocessed
    symbols = [s for s in symbols if s not in existing_symbols]
    print(f"Remaining: {len(symbols)} tickers")
    
    signals = []
    processed = len(existing_symbols)
    errors = 0
    
    # Process in smaller batches with memory management
    BATCH_FLUSH_SIZE = 50
    
    for i, symbol in enumerate(symbols):
        if i % 100 == 0:
            print(f"  [{processed+i+1}/{len(symbols)+len(existing_symbols)}] Processing... ({len(signals)} in batch)")
        
        try:
            ticker_data = generator.load_ticker_data(symbol)
            if ticker_data is None:
                continue
            
            signal = generator.generate_signal(ticker_data)
            if signal:
                signals.append(signal)
                processed += 1
                
                if processed <= 5:
                    print(f"\n  Sample: {symbol}")
                    print(f"    Signal: {signal['signal']} ({signal['confidence']}%)")
                    print(f"    Resistance: {len(signal['resistance_levels'])} levels")
                    print(f"    Support: {len(signal['support_levels'])} levels")
                    print(f"    Peaks: {len(signal['peaks'])}, Troughs: {len(signal['troughs'])}")
                
                # Flush to disk every BATCH_FLUSH_SIZE signals
                if len(signals) >= BATCH_FLUSH_SIZE:
                    _flush_signals(signals, output_path, i == len(symbols) - 1)
                    signals = []  # Clear memory
                    
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"  Error on {symbol}: {e}")
    
    # Final flush
    if signals:
        _flush_signals(signals, output_path, final=True)
    
    # Summary
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    
    if os.path.exists(output_path):
        df = pd.read_csv(output_path)
        print(f"Total signals: {len(df)}")
        print(f"Errors: {errors}")
        print(f"Output: {output_path}")
        
        # Signal distribution
        print("\nSignal Distribution:")
        signal_counts = df['signal'].value_counts()
        for sig, count in signal_counts.items():
            print(f"  {sig}: {count}")


def _flush_signals(signals: List[Dict], output_path: str, final: bool = False):
    """Flush signals to disk"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Flatten for CSV export
    flat_signals = []
    for s in signals:
        flat = s.copy()
        # Convert lists to JSON strings for CSV
        flat['resistance_levels'] = json.dumps(s['resistance_levels'])
        flat['support_levels'] = json.dumps(s['support_levels'])
        flat['peaks'] = json.dumps(s['peaks'])
        flat['troughs'] = json.dumps(s['troughs'])
        flat_signals.append(flat)
    
    batch_df = pd.DataFrame(flat_signals)
    
    # Append or create
    if os.path.exists(output_path) and not final:
        batch_df.to_csv(output_path, mode='a', header=False, index=False)
    else:
        # For final write, read existing and combine
        if os.path.exists(output_path):
            existing_df = pd.read_csv(output_path)
            combined_df = pd.concat([existing_df, batch_df], ignore_index=True)
            combined_df.to_csv(output_path, index=False)
        else:
            batch_df.to_csv(output_path, index=False)
    
    if not final:
        print(f"    Flushed {len(signals)} signals to disk")


if __name__ == "__main__":
    main()
