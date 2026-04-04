#!/usr/bin/env python3
"""
Signal Scanner - Technical Analysis Engine
Generates trading signals for stocks, crypto, and forex
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class TechnicalIndicators:
    """Calculate technical indicators for price data"""
    
    @staticmethod
    def sma(prices: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average"""
        return prices.rolling(window=period).mean()
    
    @staticmethod
    def ema(prices: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average"""
        return prices.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD indicator"""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(prices: pd.Series, period: int = 20, std: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands"""
        middle = prices.rolling(window=period).mean()
        band_std = prices.rolling(window=period).std()
        upper = middle + (band_std * std)
        lower = middle - (band_std * std)
        return upper, middle, lower
    
    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> Tuple[pd.Series, pd.Series]:
        """Stochastic Oscillator"""
        lowest_low = low.rolling(window=period).min()
        highest_high = high.rolling(window=period).max()
        k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d = k.rolling(window=3).mean()
        return k, d
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average True Range"""
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        return true_range.rolling(window=period).mean()


class SignalGenerator:
    """Generate trading signals based on technical analysis"""
    
    def __init__(self):
        self.indicators = TechnicalIndicators()
    
    def analyze_ticker(self, ticker: str, df: pd.DataFrame) -> Dict:
        """Analyze a single ticker and generate signal"""
        if df is None or len(df) < 50:
            return None
        
        # Get latest price
        latest_close = df['Close'].iloc[-1]
        latest_high = df['High'].iloc[-1]
        latest_low = df['Low'].iloc[-1]
        
        # Calculate indicators
        df['SMA_20'] = self.indicators.sma(df['Close'], 20)
        df['SMA_50'] = self.indicators.sma(df['Close'], 50)
        df['EMA_12'] = self.indicators.ema(df['Close'], 12)
        df['EMA_26'] = self.indicators.ema(df['Close'], 26)
        df['RSI'] = self.indicators.rsi(df['Close'], 14)
        df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = self.indicators.macd(df['Close'])
        df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = self.indicators.bollinger_bands(df['Close'])
        df['ATR'] = self.indicators.atr(df['High'], df['Low'], df['Close'], 14)
        
        # Get latest values
        sma_20 = df['SMA_20'].iloc[-1]
        sma_50 = df['SMA_50'].iloc[-1]
        ema_12 = df['EMA_12'].iloc[-1]
        ema_26 = df['EMA_26'].iloc[-1]
        rsi = df['RSI'].iloc[-1]
        macd = df['MACD'].iloc[-1]
        macd_signal = df['MACD_Signal'].iloc[-1]
        bb_upper = df['BB_Upper'].iloc[-1]
        bb_lower = df['BB_Lower'].iloc[-1]
        atr = df['ATR'].iloc[-1]
        
        # Generate signal
        signal = self._generate_signal(
            latest_close, sma_20, sma_50, rsi, macd, macd_signal,
            bb_upper, bb_lower
        )
        
        # Calculate price change
        price_change_1d = ((latest_close - df['Close'].iloc[-2]) / df['Close'].iloc[-2] * 100) if len(df) > 1 else 0
        price_change_7d = ((latest_close - df['Close'].iloc[-7]) / df['Close'].iloc[-7] * 100) if len(df) > 7 else 0
        
        return {
            'ticker': ticker,
            'signal': signal['signal'],
            'confidence': signal['confidence'],
            'strength': signal['strength'],
            'current_price': round(latest_close, 4),
            'sma_20': round(sma_20, 4) if not pd.isna(sma_20) else None,
            'sma_50': round(sma_50, 4) if not pd.isna(sma_50) else None,
            'rsi': round(rsi, 2) if not pd.isna(rsi) else None,
            'macd': round(macd, 4) if not pd.isna(macd) else None,
            'atr': round(atr, 4) if not pd.isna(atr) else None,
            'bb_upper': round(bb_upper, 4) if not pd.isna(bb_upper) else None,
            'bb_lower': round(bb_lower, 4) if not pd.isna(bb_lower) else None,
            'price_change_1d': round(price_change_1d, 2),
            'price_change_7d': round(price_change_7d, 2),
            'factors': signal['factors']
        }
    
    def _generate_signal(self, price: float, sma_20: float, sma_50: float, 
                         rsi: float, macd: float, macd_signal: float,
                         bb_upper: float, bb_lower: float) -> Dict:
        """Generate signal based on multiple indicators"""
        
        buy_score = 0
        sell_score = 0
        confidence = 0
        factors = []
        
        # Trend Analysis (SMA)
        if not pd.isna(sma_20) and not pd.isna(sma_50):
            if price > sma_20 > sma_50:
                buy_score += 2
                factors.append("Price above 20 & 50 SMA (uptrend)")
            elif price < sma_20 < sma_50:
                sell_score += 2
                factors.append("Price below 20 & 50 SMA (downtrend)")
        
        # RSI Analysis
        if not pd.isna(rsi):
            if rsi < 30:
                buy_score += 2
                factors.append(f"RSI oversold ({rsi:.1f})")
            elif rsi > 70:
                sell_score += 2
                factors.append(f"RSI overbought ({rsi:.1f})")
            elif 40 < rsi < 60:
                confidence += 1  # Neutral zone
        
        # MACD Analysis
        if not pd.isna(macd) and not pd.isna(macd_signal):
            if macd > macd_signal and macd_signal < 0:
                buy_score += 2
                factors.append("MACD bullish crossover")
            elif macd < macd_signal and macd_signal > 0:
                sell_score += 2
                factors.append("MACD bearish crossover")
        
        # Bollinger Bands Analysis
        if not pd.isna(bb_upper) and not pd.isna(bb_lower):
            if price < bb_lower:
                buy_score += 1
                factors.append("Price below lower Bollinger Band")
            elif price > bb_upper:
                sell_score += 1
                factors.append("Price above upper Bollinger Band")
        
        # Determine signal
        total_score = buy_score + sell_score
        
        if buy_score > sell_score + 1:
            if buy_score >= 4:
                signal = "STRONG_BUY"
                confidence = min(90, 50 + buy_score * 10)
                strength = "Strong"
            else:
                signal = "BUY"
                confidence = min(75, 40 + buy_score * 8)
                strength = "Moderate"
        elif sell_score > buy_score + 1:
            if sell_score >= 4:
                signal = "STRONG_SELL"
                confidence = min(90, 50 + sell_score * 10)
                strength = "Strong"
            else:
                signal = "SELL"
                confidence = min(75, 40 + sell_score * 8)
                strength = "Moderate"
        else:
            signal = "HOLD"
            confidence = 50
            strength = "Weak"
            factors.append("Mixed signals - insufficient conviction")
        
        return {
            'signal': signal,
            'confidence': confidence,
            'strength': strength,
            'factors': factors
        }


class DataFetcher:
    """Fetch market data from various sources"""
    
    def __init__(self):
        self.cache = {}
    
    def fetch_stock_data(self, symbol: str, period: str = "60d", interval: str = "1d") -> Optional[pd.DataFrame]:
        """Fetch stock data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            if len(df) > 0:
                return df
        except Exception as e:
            pass
        return None
    
    def fetch_crypto_data(self, symbol: str, period: str = "60d") -> Optional[pd.DataFrame]:
        """Fetch crypto data from Yahoo Finance"""
        # Yahoo crypto format: BTC-USD, ETH-USD
        return self.fetch_stock_data(symbol, period)
    
    def fetch_forex_data(self, symbol: str, period: str = "60d") -> Optional[pd.DataFrame]:
        """Fetch forex data from Yahoo Finance"""
        # Yahoo forex format: EURUSD=X, GBPUSD=X
        return self.fetch_stock_data(symbol, period)


class SignalScanner:
    """Main scanner class - orchestrates signal generation"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.fetcher = DataFetcher()
        self.generator = SignalGenerator()
        self.signals = []
    
    def load_tickers(self, asset_type: str = "stocks", limit: Optional[int] = None) -> List[Dict]:
        """Load tickers from saved data"""
        # Handle singular/plural naming
        ticker_filename = f"all_{asset_type}_tickers.json" if asset_type == 'crypto' or asset_type == 'forex' else f"all_{asset_type}_tickers.json"
        if asset_type == 'stocks':
            ticker_filename = "all_stock_tickers.json"
        ticker_file = os.path.join(self.data_dir, "tickers", ticker_filename)
        
        if not os.path.exists(ticker_file):
            print(f"Ticker file not found: {ticker_file}")
            return []
        
        with open(ticker_file, 'r') as f:
            tickers = json.load(f)
        
        if limit:
            tickers = tickers[:limit]
        
        return tickers
    
    def scan_stocks(self, limit: int = 100) -> List[Dict]:
        """Scan stock tickers and generate signals"""
        print(f"\nScanning {limit} stocks...")
        
        tickers = self.load_tickers("stocks", limit)
        signals = []
        
        for ticker_data in tickers:
            # Use yahoo_symbol if available (for TSX/TSXV), otherwise use symbol
            symbol = ticker_data.get('yahoo_symbol') or ticker_data.get('symbol')
            display_symbol = ticker_data.get('symbol', symbol)
            if not symbol:
                continue
            
            print(f"  Analyzing {display_symbol}...", end='\r')
            
            df = self.fetcher.fetch_stock_data(symbol)
            if df is not None:
                signal = self.generator.analyze_ticker(symbol, df)
                if signal:
                    signal['asset_type'] = 'stock'
                    signal['name'] = ticker_data.get('name', '')
                    signals.append(signal)
        
        print(f"\nGenerated {len(signals)} stock signals")
        return signals
    
    def scan_crypto(self, limit: int = 100) -> List[Dict]:
        """Scan crypto tickers and generate signals"""
        print(f"\nScanning {limit} cryptocurrencies...")
        
        tickers = self.load_tickers("crypto", limit)
        signals = []
        
        for ticker_data in tickers:
            symbol = ticker_data.get('symbol')
            if not symbol:
                continue
            
            # Convert to Yahoo Finance format if needed
            if '-' not in symbol and '/' in symbol:
                symbol = symbol.replace('/', '-')
            
            print(f"  Analyzing {symbol}...", end='\r')
            
            df = self.fetcher.fetch_crypto_data(symbol)
            if df is not None:
                signal = self.generator.analyze_ticker(symbol, df)
                if signal:
                    signal['asset_type'] = 'crypto'
                    signal['exchange'] = ticker_data.get('exchange', '')
                    signals.append(signal)
        
        print(f"\nGenerated {len(signals)} crypto signals")
        return signals
    
    def scan_forex(self) -> List[Dict]:
        """Scan forex pairs and generate signals"""
        print("\nScanning forex pairs...")
        
        tickers = self.load_tickers("forex")
        signals = []
        
        for ticker_data in tickers:
            symbol = ticker_data.get('yahoo_symbol')
            if not symbol:
                continue
            
            pair = ticker_data.get('symbol', '')
            print(f"  Analyzing {pair}...", end='\r')
            
            df = self.fetcher.fetch_forex_data(symbol)
            if df is not None:
                signal = self.generator.analyze_ticker(pair, df)
                if signal:
                    signal['asset_type'] = 'forex'
                    signal['pair'] = pair
                    signals.append(signal)
        
        print(f"\nGenerated {len(signals)} forex signals")
        return signals
    
    def save_signals(self, signals: List[Dict], filename: str):
        """Save signals to JSON and CSV"""
        output_dir = os.path.join(self.data_dir, "signals")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as JSON
        json_path = os.path.join(output_dir, f"{filename}.json")
        with open(json_path, 'w') as f:
            json.dump(signals, f, indent=2)
        
        # Save as CSV
        csv_path = os.path.join(output_dir, f"{filename}.csv")
        if signals:
            df = pd.DataFrame(signals)
            df.to_csv(csv_path, index=False)
        
        print(f"Saved {len(signals)} signals to {json_path} and {csv_path}")
    
    def run_full_scan(self, stock_limit: int = 50, crypto_limit: int = 50):
        """Run full scan across all asset types"""
        print("=" * 60)
        print("SIGNAL SCANNER - Technical Analysis Engine")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        all_signals = []
        
        # Scan stocks
        stock_signals = self.scan_stocks(stock_limit)
        self.save_signals(stock_signals, "stock_signals")
        all_signals.extend(stock_signals)
        
        # Scan crypto
        crypto_signals = self.scan_crypto(crypto_limit)
        self.save_signals(crypto_signals, "crypto_signals")
        all_signals.extend(crypto_signals)
        
        # Scan forex
        forex_signals = self.scan_forex()
        self.save_signals(forex_signals, "forex_signals")
        all_signals.extend(forex_signals)
        
        # Save combined
        self.save_signals(all_signals, "all_signals")
        
        # Print summary
        print("\n" + "=" * 60)
        print("SCAN COMPLETE")
        print("=" * 60)
        print(f"Stock Signals: {len(stock_signals)}")
        print(f"Crypto Signals: {len(crypto_signals)}")
        print(f"Forex Signals: {len(forex_signals)}")
        print(f"Total Signals: {len(all_signals)}")
        
        # Print top signals
        buy_signals = [s for s in all_signals if 'BUY' in s['signal']]
        sell_signals = [s for s in all_signals if 'SELL' in s['signal']]
        
        print(f"\nBUY Signals: {len(buy_signals)}")
        print(f"SELL Signals: {len(sell_signals)}")
        print(f"HOLD Signals: {len(all_signals) - len(buy_signals) - len(sell_signals)}")
        
        if buy_signals:
            print("\n--- Top BUY Signals ---")
            for sig in sorted(buy_signals, key=lambda x: x['confidence'], reverse=True)[:5]:
                print(f"  {sig['ticker']}: {sig['signal']} ({sig['confidence']}% confidence)")
        
        if sell_signals:
            print("\n--- Top SELL Signals ---")
            for sig in sorted(sell_signals, key=lambda x: x['confidence'], reverse=True)[:5]:
                print(f"  {sig['ticker']}: {sig['signal']} ({sig['confidence']}% confidence)")
        
        print("=" * 60)


if __name__ == "__main__":
    # Run the scanner
    scanner = SignalScanner()
    scanner.run_full_scan(stock_limit=50, crypto_limit=50)