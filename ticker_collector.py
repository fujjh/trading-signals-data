#!/usr/bin/env python3
"""
Ticker Collector for Trading Signals
Fetches comprehensive ticker lists for Stocks, Crypto, and FX
"""

import os
import json
import csv
import requests
from datetime import datetime
import yfinance as yf

# Try to import ccxt for crypto
try:
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False
    print("Warning: ccxt not available for crypto data")

class StockTickerCollector:
    """Collects stock tickers from major exchanges"""
    
    def __init__(self, data_dir="data/tickers"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
    def fetch_nasdaq_tickers(self):
        """Fetch NASDAQ tickers from official source"""
        print("Fetching NASDAQ tickers...")
        url = "https://www.nasdaq.com/api/screener/stocks?tableonly=true&exchange=NASDAQ&download=true"
        
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                data = response.json()
                tickers = []
                for row in data.get('data', {}).get('table', {}).get('rows', []):
                    tickers.append({
                        'symbol': row.get('symbol'),
                        'name': row.get('name'),
                        'exchange': 'NASDAQ',
                        'market_cap': row.get('marketCap'),
                        'sector': row.get('sector'),
                        'industry': row.get('industry')
                    })
                return tickers
        except Exception as e:
            print(f"Error fetching NASDAQ: {e}")
            
        # Fallback: try FTP method
        return self._fetch_nasdaq_via_ftp()
    
    def _fetch_nasdaq_via_ftp(self):
        """Fallback: Fetch from NASDAQ FTP"""
        print("Trying NASDAQ FTP fallback...")
        url = "ftp://ftp.nasdaqtrader.com/symboldirectory/nasdaqlisted.txt"
        try:
            response = requests.get(url, timeout=30)
            tickers = []
            lines = response.text.strip().split('\n')
            for line in lines[1:]:  # Skip header
                parts = line.split('|')
                if len(parts) >= 3:
                    tickers.append({
                        'symbol': parts[0],
                        'name': parts[1],
                        'exchange': 'NASDAQ',
                        'market_cap': None,
                        'sector': None,
                        'industry': None
                    })
            return tickers
        except Exception as e:
            print(f"FTP fallback failed: {e}")
            return []
    
    def fetch_nyse_tickers(self):
        """Fetch NYSE tickers"""
        print("Fetching NYSE tickers...")
        url = "https://www.nyse.com/api/quotes/filter"
        
        try:
            # NYSE API requires POST with filter
            payload = {
                "instrumentType": "EQUITY",
                "pageNumber": 1,
                "sortColumn": "TOTAL_VOLUME",
                "sortOrder": "DESC",
                "maxResultsPerPage": 10000
            }
            response = requests.post(url, json=payload, timeout=30)
            if response.status_code == 200:
                data = response.json()
                tickers = []
                for item in data:
                    tickers.append({
                        'symbol': item.get('symbolTicker'),
                        'name': item.get('instrumentName'),
                        'exchange': 'NYSE',
                        'market_cap': item.get('marketCap'),
                        'sector': item.get('sector'),
                        'industry': item.get('industry')
                    })
                return tickers
        except Exception as e:
            print(f"Error fetching NYSE: {e}")
            return []
    
    def fetch_tsx_tickers(self):
        """Fetch TSX and TSX.V tickers"""
        print("Fetching TSX/TSX.V tickers...")
        # TMX API endpoint
        url = "https://api.tmxmoney.com/en/TSX/quote.json"
        
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                data = response.json()
                tickers = []
                for item in data.get('results', []):
                    exchange = 'TSX' if item.get('market') == 'XTSE' else 'TSXV'
                    tickers.append({
                        'symbol': item.get('symbol'),
                        'name': item.get('name'),
                        'exchange': exchange,
                        'market_cap': item.get('marketCap'),
                        'sector': item.get('sector'),
                        'industry': item.get('industry')
                    })
                return tickers
        except Exception as e:
            print(f"Error fetching TSX: {e}")
            return []
    
    def fetch_yahoo_tickers(self, exchange):
        """Fetch tickers using Yahoo Finance screener"""
        print(f"Fetching {exchange} tickers via Yahoo Finance...")
        
        exchange_map = {
            'NASDAQ': 'NMS',
            'NYSE': 'NYQ',
            'TSX': 'TOR',
            'TSXV': 'VAN'
        }
        
        yahoo_exchange = exchange_map.get(exchange, exchange)
        
        try:
            # Yahoo Finance screener URL
            url = f"https://finance.yahoo.com/screener/predefined/ms_{yahoo_exchange}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=30)
            
            # Parse tickers from the page
            tickers = []
            # This is a simplified approach - in production, we'd use proper parsing
            return tickers
        except Exception as e:
            print(f"Error fetching via Yahoo: {e}")
            return []
    
    def save_tickers(self, tickers, filename):
        """Save tickers to JSON and CSV"""
        # Save as JSON
        json_path = os.path.join(self.data_dir, f"{filename}.json")
        with open(json_path, 'w') as f:
            json.dump(tickers, f, indent=2)
        
        # Save as CSV
        csv_path = os.path.join(self.data_dir, f"{filename}.csv")
        if tickers:
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=tickers[0].keys())
                writer.writeheader()
                writer.writerows(tickers)
        
        print(f"Saved {len(tickers)} tickers to {json_path} and {csv_path}")
    
    def collect_all_stocks(self):
        """Collect all stock tickers"""
        print("\n=== Collecting Stock Tickers ===\n")
        
        all_tickers = []
        
        # NASDAQ
        nasdaq = self.fetch_nasdaq_tickers()
        self.save_tickers(nasdaq, 'nasdaq_tickers')
        all_tickers.extend(nasdaq)
        
        # NYSE
        nyse = self.fetch_nyse_tickers()
        self.save_tickers(nyse, 'nyse_tickers')
        all_tickers.extend(nyse)
        
        # TSX/TSXV
        tsx = self.fetch_tsx_tickers()
        self.save_tickers(tsx, 'tsx_tickers')
        all_tickers.extend(tsx)
        
        # Save combined
        self.save_tickers(all_tickers, 'all_stock_tickers')
        
        print(f"\n=== Total Stock Tickers: {len(all_tickers)} ===\n")
        return all_tickers


class CryptoTickerCollector:
    """Collects cryptocurrency tickers from major exchanges"""
    
    def __init__(self, data_dir="data/tickers"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.exchanges = ['binance', 'coinbase', 'kraken', 'kucoin', 'bybit']
    
    def fetch_exchange_tickers(self, exchange_id='binance'):
        """Fetch USDC and USDT trading pairs from a specific exchange"""
        print(f"Fetching {exchange_id} USDC/USDT tickers...")
        
        if not CCXT_AVAILABLE:
            print("ccxt not available, skipping crypto collection")
            return []
        
        try:
            exchange = getattr(ccxt, exchange_id)()
            exchange.load_markets()
            
            tickers = []
            allowed_quotes = ['USDC', 'USDT']
            
            for symbol, market in exchange.markets.items():
                quote = market.get('quote', '')
                base = market.get('base', '')
                
                # Only include USDC and USDT pairs
                if quote in allowed_quotes:
                    tickers.append({
                        'symbol': symbol,
                        'base': base,
                        'quote': quote,
                        'exchange': exchange_id,
                        'type': market.get('type', 'spot'),
                        'active': market.get('active', True)
                    })
            
            print(f"Found {len(tickers)} USDC/USDT pairs on {exchange_id}")
            return tickers
        except Exception as e:
            print(f"Error fetching {exchange_id}: {e}")
            return []
    
    def get_top_cryptos_via_yfinance(self):
        """Get top cryptocurrencies with USD pairs via Yahoo Finance"""
        print("Fetching top cryptos via Yahoo Finance (USD pairs only)...")
        
        # Major crypto tickers on Yahoo Finance (USD pairs only)
        crypto_symbols = [
            'BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD',
            'SOL-USD', 'DOT-USD', 'AVAX-USD', 'MATIC-USD', 'LINK-USD',
            'UNI-USD', 'LTC-USD', 'BCH-USD', 'ALGO-USD', 'ATOM-USD',
            'ETC-USD', 'VET-USD', 'FIL-USD', 'TRX-USD', 'NEAR-USD',
            'ICP-USD', 'XLM-USD', 'MANA-USD', 'SAND-USD', 'AXS-USD',
            'FTM-USD', 'XTZ-USD', 'EGLD-USD', 'THETA-USD', 'GALA-USD'
        ]
        
        tickers = []
        for symbol in crypto_symbols:
            tickers.append({
                'symbol': symbol,
                'base': symbol.split('-')[0],
                'quote': 'USD',
                'exchange': 'YAHOO',
                'type': 'spot',
                'active': True
            })
        
        return tickers
    
    def save_tickers(self, tickers, filename):
        """Save tickers to JSON and CSV"""
        json_path = os.path.join(self.data_dir, f"{filename}.json")
        with open(json_path, 'w') as f:
            json.dump(tickers, f, indent=2)
        
        csv_path = os.path.join(self.data_dir, f"{filename}.csv")
        if tickers:
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=tickers[0].keys())
                writer.writeheader()
                writer.writerows(tickers)
        
        print(f"Saved {len(tickers)} crypto tickers to {json_path} and {csv_path}")
    
    def collect_all_crypto(self):
        """Collect all cryptocurrency tickers"""
        print("\n=== Collecting Crypto Tickers ===\n")
        
        all_tickers = []
        
        # Fetch from major exchanges via CCXT
        if CCXT_AVAILABLE:
            for exchange in self.exchanges:
                tickers = self.fetch_exchange_tickers(exchange)
                self.save_tickers(tickers, f'{exchange}_tickers')
                all_tickers.extend(tickers)
        
        # Also get top cryptos via Yahoo Finance
        yahoo_crypto = self.get_top_cryptos_via_yfinance()
        self.save_tickers(yahoo_crypto, 'yahoo_crypto_tickers')
        all_tickers.extend(yahoo_crypto)
        
        # Save combined
        self.save_tickers(all_tickers, 'all_crypto_tickers')
        
        print(f"\n=== Total Crypto Tickers: {len(all_tickers)} ===\n")
        return all_tickers


class ForexTickerCollector:
    """Collects forex currency pairs"""
    
    def __init__(self, data_dir="data/tickers"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Major forex pairs
        self.major_pairs = [
            'EURUSD', 'USDJPY', 'GBPUSD', 'USDCHF', 'AUDUSD',
            'USDCAD', 'NZDUSD', 'EURGBP', 'EURJPY', 'GBPJPY'
        ]
        
        self.minor_pairs = [
            'EURCHF', 'EURCAD', 'GBPCHF', 'GBPAUD', 'GBPCAD',
            'CHFJPY', 'AUDJPY', 'CADJPY', 'NZDJPY', 'AUDNZD'
            
        ]
        
        self.exotic_pairs = [
            'USDZAR', 'USDTRY', 'USDMXN', 'USDSGD', 'USDHKD',
            'USDNOK', 'USDSEK', 'USDDKK', 'USDPLN', 'USDHUF'
        ]
    
    def fetch_forex_tickers(self):
        """Fetch forex pairs"""
        print("Fetching forex pairs...")
        
        tickers = []
        
        # Major pairs
        for pair in self.major_pairs:
            tickers.append({
                'symbol': pair,
                'base': pair[:3],
                'quote': pair[3:],
                'type': 'major',
                'yahoo_symbol': f'{pair[:3]}{pair[3:]}=X'
            })
        
        # Minor pairs
        for pair in self.minor_pairs:
            tickers.append({
                'symbol': pair,
                'base': pair[:3],
                'quote': pair[3:],
                'type': 'minor',
                'yahoo_symbol': f'{pair[:3]}{pair[3:]}=X'
            })
        
        # Exotic pairs
        for pair in self.exotic_pairs:
            tickers.append({
                'symbol': pair,
                'base': pair[:3],
                'quote': pair[3:],
                'type': 'exotic',
                'yahoo_symbol': f'{pair[:3]}{pair[3:]}=X'
            })
        
        return tickers
    
    def save_tickers(self, tickers, filename):
        """Save tickers to JSON and CSV"""
        json_path = os.path.join(self.data_dir, f"{filename}.json")
        with open(json_path, 'w') as f:
            json.dump(tickers, f, indent=2)
        
        csv_path = os.path.join(self.data_dir, f"{filename}.csv")
        if tickers:
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=tickers[0].keys())
                writer.writeheader()
                writer.writerows(tickers)
        
        print(f"Saved {len(tickers)} forex pairs to {json_path} and {csv_path}")
    
    def collect_all_forex(self):
        """Collect all forex pairs"""
        print("\n=== Collecting Forex Pairs ===\n")
        
        tickers = self.fetch_forex_tickers()
        self.save_tickers(tickers, 'all_forex_tickers')
        
        print(f"\n=== Total Forex Pairs: {len(tickers)} ===\n")
        return tickers


def main():
    """Main execution"""
    print("=" * 60)
    print("Ticker Collector for Trading Signals")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    data_dir = "data/tickers"
    os.makedirs(data_dir, exist_ok=True)
    
    # Collect Stocks
    stock_collector = StockTickerCollector(data_dir)
    stocks = stock_collector.collect_all_stocks()
    
    # Collect Crypto
    crypto_collector = CryptoTickerCollector(data_dir)
    crypto = crypto_collector.collect_all_crypto()
    
    # Collect Forex
    forex_collector = ForexTickerCollector(data_dir)
    forex = forex_collector.collect_all_forex()
    
    # Summary
    print("\n" + "=" * 60)
    print("COLLECTION COMPLETE")
    print("=" * 60)
    print(f"Stocks: {len(stocks)}")
    print(f"Crypto: {len(crypto)}")
    print(f"Forex: {len(forex)}")
    print(f"Total: {len(stocks) + len(crypto) + len(forex)}")
    print(f"\nData saved to: {data_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()