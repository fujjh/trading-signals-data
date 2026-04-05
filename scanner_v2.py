#!/usr/bin/env python3
"""
Signal Scanner v2.0 - Enhanced Technical Analysis Engine
Generates trading signals with comprehensive yfinance data integration

NEW DATA POINTS ADDED:
- Phase 1: Fundamentals (P/E, PEG, P/B, ROE, Debt/Eq, etc.)
- Phase 2: Volume Analysis, Sentiment (Short ratio, Institutional %)
- Phase 3: Analyst data (Price targets, recommendations)
- Plus: 52-week highs/lows, Beta, Dividend yield
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
    def macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(prices: pd.Series, period: int = 20, std: int = 2):
        middle = prices.rolling(window=period).mean()
        band_std = prices.rolling(window=period).std()
        upper = middle + (band_std * std)
        lower = middle - (band_std * std)
        return upper, middle, lower
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        return true_range.rolling(window=period).mean()
    
    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """On-Balance Volume"""
        obv = (np.sign(close.diff()) * volume).cumsum()
        return obv
    
    @staticmethod
    def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Volume Weighted Average Price"""
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).cumsum() / volume.cumsum()
        return vwap


class DataFetcher:
    """Fetch market data AND fundamental data from Yahoo Finance"""
    
    def __init__(self):
        self.cache = {}
    
    def fetch_stock_data(self, symbol: str, period: str = "60d", interval: str = "1d") -> Optional[pd.DataFrame]:
        """Fetch OHLCV price data"""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            if len(df) > 0:
                return df
        except:
            pass
        return None
    
    def fetch_fundamentals(self, symbol: str) -> Dict:
        """
        Fetch comprehensive fundamental data from yfinance.
        Returns dictionary with all available metrics.
        """
        data = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            # Fundamental fields (default None)
            'sector': None, 'industry': None, 'country': None,
            'market_cap': None, 'enterprise_value': None,
            'trailing_pe': None, 'forward_pe': None, 'peg_ratio': None,
            'price_to_book': None, 'price_to_sales': None,
            'profit_margins': None, 'revenue_growth': None, 'earnings_growth': None,
            'current_ratio': None, 'quick_ratio': None, 'debt_to_equity': None,
            'return_on_equity': None, 'return_on_assets': None,
            'gross_margins': None, 'operating_margins': None, 'ebitda_margins': None,
            'beta': None, 'fifty_two_week_change': None,
            'dividend_rate': None, 'dividend_yield': None, 'payout_ratio': None,
            'total_debt': None, 'total_cash': None, 'total_revenue': None,
            'revenue_per_share': None, 'book_value': None,
            'held_percent_insiders': None, 'held_percent_institutions': None,
            'short_ratio': None, 'short_percent_of_float': None,
            'target_high_price': None, 'target_low_price': None,
            'target_mean_price': None, 'target_median_price': None,
            'recommendation_mean': None, 'recommendation_key': None,
            'number_of_analyst_opinions': None,
            # Price/Volume levels
            'fifty_day_average': None, 'two_hundred_day_average': None,
            'year_high': None, 'year_low': None,
            'average_volume': None, 'average_volume_10days': None,
        }
        
        try:
            ticker = yf.Ticker(symbol)
            
            # Get fast_info for price data
            fast = ticker.fast_info
            data['market_cap'] = getattr(fast, 'market_cap', None)
            data['fifty_day_average'] = getattr(fast, 'fifty_day_average', None)
            data['two_hundred_day_average'] = getattr(fast, 'two_hundred_day_average', None)
            data['year_high'] = getattr(fast, 'year_high', None)
            data['year_low'] = getattr(fast, 'year_low', None)
            data['average_volume'] = getattr(fast, 'three_month_average_volume', None)
            data['average_volume_10days'] = getattr(fast, 'ten_day_average_volume', None)
            data['beta'] = getattr(fast, 'beta', None)
            
            # Get full info for fundamentals
            info = ticker.info
            
            # Profile data
            data['sector'] = info.get('sector')
            data['industry'] = info.get('industry')
            data['country'] = info.get('country')
            
            # Valuation metrics
            data['trailing_pe'] = info.get('trailingPE')
            data['forward_pe'] = info.get('forwardPE')
            data['peg_ratio'] = info.get('pegRatio')
            data['price_to_book'] = info.get('priceToBook')
            data['price_to_sales'] = info.get('priceToSalesTrailing12Months')
            data['enterprise_value'] = info.get('enterpriseValue')
            
            # Profitability
            data['profit_margins'] = info.get('profitMargins')
            data['revenue_growth'] = info.get('revenueGrowth')
            data['earnings_growth'] = info.get('earningsGrowth')
            data['return_on_equity'] = info.get('returnOnEquity')
            data['return_on_assets'] = info.get('returnOnAssets')
            data['gross_margins'] = info.get('grossMargins')
            data['operating_margins'] = info.get('operatingMargins')
            data['ebitda_margins'] = info.get('ebitdaMargins')
            
            # Financial health
            data['current_ratio'] = info.get('currentRatio')
            data['quick_ratio'] = info.get('quickRatio')
            data['debt_to_equity'] = info.get('debtToEquity')
            data['total_debt'] = info.get('totalDebt')
            data['total_cash'] = info.get('totalCash')
            data['total_revenue'] = info.get('totalRevenue')
            data['revenue_per_share'] = info.get('revenuePerShare')
            data['book_value'] = info.get('bookValue')
            
            # Dividend
            data['dividend_rate'] = info.get('dividendRate')
            data['dividend_yield'] = info.get('dividendYield')
            data['payout_ratio'] = info.get('payoutRatio')
            
            # Sentiment
            data['held_percent_insiders'] = info.get('heldPercentInsiders')
            data['held_percent_institutions'] = info.get('heldPercentInstitutions')
            data['short_ratio'] = info.get('shortRatio')
            data['short_percent_of_float'] = info.get('shortPercentOfFloat')
            
            # Analyst data
            data['target_high_price'] = info.get('targetHighPrice')
            data['target_low_price'] = info.get('targetLowPrice')
            data['target_mean_price'] = info.get('targetMeanPrice')
            data['target_median_price'] = info.get('targetMedianPrice')
            data['recommendation_mean'] = info.get('recommendationMean')
            data['recommendation_key'] = info.get('recommendationKey')
            data['number_of_analyst_opinions'] = info.get('numberOfAnalystOpinions')
            
        except Exception as e:
            # Return what we have (mostly Nones)
            pass
        
        return data


class EnhancedSignalGenerator:
    """Generate signals with comprehensive data analysis"""
    
    def __init__(self):
        self.indicators = TechnicalIndicators()
    
    def analyze_ticker(self, symbol: str, df: pd.DataFrame, fundamentals: Dict) -> Optional[Dict]:
        """
        Comprehensive ticker analysis combining technical and fundamental data.
        """
        if df is None or len(df) < 50:
            return None
        
        # Get price data
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        week_ago = df.iloc[-7] if len(df) > 7 else df.iloc[0]
        
        latest_close = latest['Close']
        latest_high = latest['High']
        latest_low = latest['Low']
        latest_volume = latest['Volume']
        
        # Calculate technical indicators
        df['SMA_20'] = self.indicators.sma(df['Close'], 20)
        df['SMA_50'] = self.indicators.sma(df['Close'], 50)
        df['EMA_12'] = self.indicators.ema(df['Close'], 12)
        df['EMA_26'] = self.indicators.ema(df['Close'], 26)
        df['RSI'] = self.indicators.rsi(df['Close'], 14)
        df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = self.indicators.macd(df['Close'])
        df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = self.indicators.bollinger_bands(df['Close'])
        df['ATR'] = self.indicators.atr(df['High'], df['Low'], df['Close'], 14)
        df['OBV'] = self.indicators.obv(df['Close'], df['Volume'])
        
        # Get latest indicator values
        sma_20 = df['SMA_20'].iloc[-1]
        sma_50 = df['SMA_50'].iloc[-1]
        rsi = df['RSI'].iloc[-1]
        macd = df['MACD'].iloc[-1]
        macd_signal = df['MACD_Signal'].iloc[-1]
        bb_upper = df['BB_Upper'].iloc[-1]
        bb_lower = df['BB_Lower'].iloc[-1]
        atr = df['ATR'].iloc[-1]
        
        # Calculate price changes
        price_change_1d = ((latest_close - prev['Close']) / prev['Close'] * 100) if len(df) > 1 else 0
        price_change_7d = ((latest_close - week_ago['Close']) / week_ago['Close'] * 100) if len(df) > 7 else 0
        
        # Calculate relative volume
        avg_volume = fundamentals.get('average_volume') or df['Volume'].mean()
        relative_volume = (latest_volume / avg_volume) if avg_volume and avg_volume > 0 else 1.0
        
        # Calculate 52-week position
        year_high = fundamentals.get('year_high')
        year_low = fundamentals.get('year_low')
        week_52_position = None
        if year_high and year_low and year_high > year_low:
            week_52_position = ((latest_close - year_low) / (year_high - year_low)) * 100
        
        # Generate enhanced signal
        signal = self._generate_enhanced_signal(
            latest_close=latest_close,
            sma_20=sma_20,
            sma_50=sma_50,
            rsi=rsi,
            macd=macd,
            macd_signal=macd_signal,
            bb_upper=bb_upper,
            bb_lower=bb_lower,
            fundamentals=fundamentals,
            relative_volume=relative_volume,
            week_52_position=week_52_position
        )
        
        # Build comprehensive result
        result = {
            # Core identifiers
            'ticker': symbol,
            'signal': signal['signal'],
            'confidence': signal['confidence'],
            'strength': signal['strength'],
            'factors': signal['factors'],
            
            # Price data
            'current_price': round(latest_close, 4),
            'price_change_1d': round(price_change_1d, 2),
            'price_change_7d': round(price_change_7d, 2),
            
            # Technical indicators
            'sma_20': round(sma_20, 4) if not pd.isna(sma_20) else None,
            'sma_50': round(sma_50, 4) if not pd.isna(sma_50) else None,
            'rsi': round(rsi, 2) if not pd.isna(rsi) else None,
            'macd': round(macd, 4) if not pd.isna(macd) else None,
            'macd_signal': round(macd_signal, 4) if not pd.isna(macd_signal) else None,
            'atr': round(atr, 4) if not pd.isna(atr) else None,
            'bb_upper': round(bb_upper, 4) if not pd.isna(bb_upper) else None,
            'bb_lower': round(bb_lower, 4) if not pd.isna(bb_lower) else None,
            'bb_middle': round((bb_upper + bb_lower) / 2, 4) if not pd.isna(bb_upper) else None,
            
            # Volume analysis
            'volume': int(latest_volume),
            'average_volume': int(avg_volume) if avg_volume else None,
            'relative_volume': round(relative_volume, 2),
            
            # Price levels
            'week_52_high': year_high,
            'week_52_low': year_low,
            'week_52_position_pct': round(week_52_position, 2) if week_52_position else None,
            'fifty_day_average': fundamentals.get('fifty_day_average'),
            'two_hundred_day_average': fundamentals.get('two_hundred_day_average'),
            
            # Fundamentals - Phase 1
            'sector': fundamentals.get('sector'),
            'industry': fundamentals.get('industry'),
            'market_cap': fundamentals.get('market_cap'),
            'enterprise_value': fundamentals.get('enterprise_value'),
            'trailing_pe': fundamentals.get('trailing_pe'),
            'forward_pe': fundamentals.get('forward_pe'),
            'peg_ratio': fundamentals.get('peg_ratio'),
            'price_to_book': fundamentals.get('price_to_book'),
            'price_to_sales': fundamentals.get('price_to_sales'),
            
            # Profitability
            'profit_margins': fundamentals.get('profit_margins'),
            'revenue_growth': fundamentals.get('revenue_growth'),
            'earnings_growth': fundamentals.get('earnings_growth'),
            'return_on_equity': fundamentals.get('return_on_equity'),
            'return_on_assets': fundamentals.get('return_on_assets'),
            'gross_margins': fundamentals.get('gross_margins'),
            'operating_margins': fundamentals.get('operating_margins'),
            
            # Financial health
            'debt_to_equity': fundamentals.get('debt_to_equity'),
            'current_ratio': fundamentals.get('current_ratio'),
            'quick_ratio': fundamentals.get('quick_ratio'),
            'total_cash': fundamentals.get('total_cash'),
            'total_debt': fundamentals.get('total_debt'),
            
            # Risk/Volatility
            'beta': fundamentals.get('beta'),
            
            # Dividend
            'dividend_yield': fundamentals.get('dividend_yield'),
            'dividend_rate': fundamentals.get('dividend_rate'),
            'payout_ratio': fundamentals.get('payout_ratio'),
            
            # Sentiment - Phase 2
            'held_percent_institutions': fundamentals.get('held_percent_institutions'),
            'held_percent_insiders': fundamentals.get('held_percent_insiders'),
            'short_ratio': fundamentals.get('short_ratio'),
            'short_percent_of_float': fundamentals.get('short_percent_of_float'),
            
            # Analyst data - Phase 3
            'target_high': fundamentals.get('target_high_price'),
            'target_low': fundamentals.get('target_low_price'),
            'target_mean': fundamentals.get('target_mean_price'),
            'target_median': fundamentals.get('target_median_price'),
            'recommendation_key': fundamentals.get('recommendation_key'),
            'recommendation_mean': fundamentals.get('recommendation_mean'),
            'num_analysts': fundamentals.get('number_of_analyst_opinions'),
            
            # Metadata
            'analyzed_at': datetime.now().isoformat()
        }
        
        return result
    
    def _generate_enhanced_signal(self, latest_close, sma_20, sma_50, rsi, macd, macd_signal,
                                   bb_upper, bb_lower, fundamentals, relative_volume, week_52_position) -> Dict:
        """
        Enhanced signal generation combining technical and fundamental factors.
        """
        buy_score = 0
        sell_score = 0
        factors = []
        
        # ========== TECHNICAL ANALYSIS ==========
        
        # Trend Analysis (SMA)
        if not pd.isna(sma_20) and not pd.isna(sma_50):
            if latest_close > sma_20 > sma_50:
                buy_score += 2
                factors.append("✓ Price above 20 & 50 SMA (uptrend)")
            elif latest_close < sma_20 < sma_50:
                sell_score += 2
                factors.append("✗ Price below 20 & 50 SMA (downtrend)")
        
        # RSI Analysis
        if not pd.isna(rsi):
            if rsi < 30:
                buy_score += 2
                factors.append(f"✓ RSI oversold ({rsi:.1f})")
            elif rsi > 70:
                sell_score += 2
                factors.append(f"✗ RSI overbought ({rsi:.1f})")
        
        # MACD Analysis
        if not pd.isna(macd) and not pd.isna(macd_signal):
            if macd > macd_signal and macd_signal < 0:
                buy_score += 2
                factors.append("✓ MACD bullish crossover")
            elif macd < macd_signal and macd_signal > 0:
                sell_score += 2
                factors.append("✗ MACD bearish crossover")
        
        # Bollinger Bands
        if not pd.isna(bb_upper) and not pd.isna(bb_lower):
            if latest_close < bb_lower:
                buy_score += 1
                factors.append("✓ Price below lower BB")
            elif latest_close > bb_upper:
                sell_score += 1
                factors.append("✗ Price above upper BB")
        
        # Volume confirmation
        if relative_volume > 1.5:
            buy_score += 1
            factors.append(f"✓ High volume ({relative_volume:.1f}x avg)")
        
        # ========== FUNDAMENTAL ANALYSIS (Phase 1) ==========
        
        # P/E Valuation
        pe = fundamentals.get('trailing_pe')
        try:
            pe = float(pe) if pe else None
        except:
            pe = None
        if pe and pe > 0:
            if pe < 15:
                buy_score += 2
                factors.append(f"✓ Low P/E ({pe:.1f})")
            elif pe > 50:
                sell_score += 1
                factors.append(f"✗ High P/E ({pe:.1f})")
        
        # PEG Ratio
        peg = fundamentals.get('peg_ratio')
        try:
            peg = float(peg) if peg else None
        except:
            peg = None
        if peg and peg > 0:
            if peg < 1:
                buy_score += 2
                factors.append(f"✓ PEG < 1 ({peg:.2f})")
            elif peg > 3:
                sell_score += 1
                factors.append(f"✗ High PEG ({peg:.2f})")
        
        # Debt/Equity
        de = fundamentals.get('debt_to_equity')
        try:
            de = float(de) if de else None
        except:
            de = None
        if de:
            if de < 50:
                buy_score += 1
                factors.append(f"✓ Low D/E ({de:.1f}%)")
            elif de > 300:
                sell_score += 1
                factors.append(f"✗ High D/E ({de:.1f}%)")
        
        # ROE
        roe = fundamentals.get('return_on_equity')
        try:
            roe = float(roe) if roe else None
        except:
            roe = None
        if roe:
            if roe > 0.15:
                buy_score += 1
                factors.append(f"✓ High ROE ({roe*100:.1f}%)")
        
        # Profit Margin
        margin = fundamentals.get('profit_margins')
        try:
            margin = float(margin) if margin else None
        except:
            margin = None
        if margin and margin > 0.20:
            buy_score += 1
            factors.append(f"✓ Strong margins ({margin*100:.1f}%)")
        
        # ========== SENTIMENT ANALYSIS (Phase 2) ==========
        
        # Short interest (contrarian indicator)
        short_pct = fundamentals.get('short_percent_of_float')
        try:
            short_pct = float(short_pct) if short_pct else None
        except:
            short_pct = None
        if short_pct and short_pct > 0.10:
            buy_score += 1  # High short interest = potential squeeze
            factors.append(f"✓ High short interest ({short_pct*100:.1f}%)")
        
        # Institutional ownership
        inst_pct = fundamentals.get('held_percent_institutions')
        try:
            inst_pct = float(inst_pct) if inst_pct else None
        except:
            inst_pct = None
        if inst_pct and inst_pct > 0.70:
            buy_score += 1
            factors.append(f"✓ Strong institutional ({inst_pct*100:.0f}%)")
        
        # ========== ANALYST DATA (Phase 3) ==========
        
        # Price target upside
        target_mean = fundamentals.get('target_mean_price')
        try:
            target_mean = float(target_mean) if target_mean else None
        except:
            target_mean = None
        if target_mean and target_mean > latest_close:
            upside = ((target_mean - latest_close) / latest_close) * 100
            if upside > 20:
                buy_score += 2
                factors.append(f"✓ Analyst upside ({upside:.0f}%)")
        
        # Recommendation
        rec_key = fundamentals.get('recommendation_key')
        if rec_key == 'buy' or rec_key == 'strong_buy':
            buy_score += 1
            factors.append(f"✓ Analyst rating: {rec_key}")
        elif rec_key == 'sell' or rec_key == 'strong_sell':
            sell_score += 1
            factors.append(f"✗ Analyst rating: {rec_key}")
        
        # ========== DETERMINE SIGNAL ==========
        
        total_score = buy_score + sell_score
        
        if buy_score > sell_score + 2:
            if buy_score >= 8:
                signal = "STRONG_BUY"
                confidence = min(95, 60 + buy_score * 4)
                strength = "Very Strong"
            elif buy_score >= 5:
                signal = "BUY"
                confidence = min(85, 50 + buy_score * 5)
                strength = "Strong"
            else:
                signal = "WEAK_BUY"
                confidence = min(70, 40 + buy_score * 6)
                strength = "Moderate"
        elif sell_score > buy_score + 2:
            if sell_score >= 8:
                signal = "STRONG_SELL"
                confidence = min(95, 60 + sell_score * 4)
                strength = "Very Strong"
            elif sell_score >= 5:
                signal = "SELL"
                confidence = min(85, 50 + sell_score * 5)
                strength = "Strong"
            else:
                signal = "WEAK_SELL"
                confidence = min(70, 40 + sell_score * 6)
                strength = "Moderate"
        else:
            signal = "HOLD"
            confidence = 50
            strength = "Weak"
            factors.append("Mixed signals")
        
        return {
            'signal': signal,
            'confidence': confidence,
            'strength': strength,
            'factors': factors
        }


class EnhancedSignalScanner:
    """Enhanced scanner with comprehensive data integration"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.fetcher = DataFetcher()
        self.generator = EnhancedSignalGenerator()
        self.signals = []
    
    def load_tickers_from_csv(self, csv_file: str, limit: Optional[int] = None) -> List[Dict]:
        """Load tickers from CSV file (like stock_ticker_base.csv)"""
        if not os.path.exists(csv_file):
            print(f"File not found: {csv_file}")
            return []
        
        df = pd.read_csv(csv_file)
        tickers = df.to_dict('records')
        
        if limit:
            tickers = tickers[:limit]
        
        print(f"Loaded {len(tickers)} tickers from {csv_file}")
        return tickers
    
    def scan_stocks(self, tickers: List[Dict]) -> List[Dict]:
        """Scan stocks with comprehensive analysis"""
        print(f"\nScanning {len(tickers)} stocks with enhanced analysis...")
        
        signals = []
        
        for i, ticker_data in enumerate(tickers):
            symbol = ticker_data.get('yahoo_symbol') or ticker_data.get('symbol')
            display = ticker_data.get('symbol', symbol)
            
            if not symbol:
                continue
            
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{len(tickers)} | Signals: {len(signals)}", end='\r')
            
            # Fetch data
            df = self.fetcher.fetch_stock_data(symbol)
            fundamentals = self.fetcher.fetch_fundamentals(symbol)
            
            if df is not None:
                signal = self.generator.analyze_ticker(symbol, df, fundamentals)
                if signal:
                    signal['name'] = ticker_data.get('name', '')
                    signal['asset_type'] = 'stock'
                    signals.append(signal)
        
        print(f"\nGenerated {len(signals)} enhanced signals")
        return signals
    
    def save_signals(self, signals: List[Dict], filename: str):
        """Save signals to JSON and CSV"""
        output_dir = os.path.join(self.data_dir, "signals")
        os.makedirs(output_dir, exist_ok=True)
        
        # JSON
        json_path = os.path.join(output_dir, f"{filename}.json")
        with open(json_path, 'w') as f:
            json.dump(signals, f, indent=2, default=str)
        
        # CSV
        csv_path = os.path.join(output_dir, f"{filename}.csv")
        if signals:
            df = pd.DataFrame(signals)
            df.to_csv(csv_path, index=False)
        
        print(f"Saved {len(signals)} signals to:")
        print(f"  JSON: {json_path}")
        print(f"  CSV: {csv_path}")
    
    def run_scan(self, csv_file: str = "data/tickers/stock_ticker_base.csv", limit: int = 350):
        """Run full enhanced scan"""
        print("=" * 80)
        print("ENHANCED SIGNAL SCANNER v2.0")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        print("\nFeatures:")
        print("  ✓ Technical indicators (SMA, RSI, MACD, BB, ATR)")
        print("  ✓ Volume analysis (Relative volume)")
        print("  ✓ Fundamentals (P/E, PEG, ROE, D/E, Margins)")
        print("  ✓ Sentiment (Short %, Institutional %)")
        print("  ✓ Analyst data (Price targets, ratings)")
        print("  ✓ Risk metrics (Beta, 52-week position)")
        
        # Load tickers
        tickers = self.load_tickers_from_csv(csv_file, limit)
        
        if not tickers:
            print("No tickers loaded!")
            return
        
        # Scan
        signals = self.scan_stocks(tickers)
        
        # Save
        self.save_signals(signals, "enhanced_signals")
        
        # Summary
        buy_signals = [s for s in signals if 'BUY' in s['signal']]
        sell_signals = [s for s in signals if 'SELL' in s['signal']]
        hold_signals = [s for s in signals if s['signal'] == 'HOLD']
        
        print("\n" + "=" * 80)
        print("SCAN SUMMARY")
        print("=" * 80)
        print(f"Total Signals: {len(signals)}")
        print(f"  BUY: {len(buy_signals)}")
        print(f"  SELL: {len(sell_signals)}")
        print(f"  HOLD: {len(hold_signals)}")
        
        if buy_signals:
            print("\n--- Top BUY Signals ---")
            for sig in sorted(buy_signals, key=lambda x: x['confidence'], reverse=True)[:5]:
                print(f"  {sig['ticker']}: {sig['signal']} ({sig['confidence']}% confidence)")
                print(f"    Price: ${sig['current_price']:.2f} | P/E: {sig.get('trailing_pe', 'N/A')}")
        
        if sell_signals:
            print("\n--- Top SELL Signals ---")
            for sig in sorted(sell_signals, key=lambda x: x['confidence'], reverse=True)[:5]:
                print(f"  {sig['ticker']}: {sig['signal']} ({sig['confidence']}% confidence)")
        
        print("=" * 80)
        
        return signals


if __name__ == "__main__":
    scanner = EnhancedSignalScanner()
    scanner.run_scan(limit=350)
