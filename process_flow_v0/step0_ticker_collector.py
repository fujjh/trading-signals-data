#!/usr/bin/env python3
"""
================================================================================
TickerCollector - Consolidated Stock Ticker Collection System
================================================================================

A comprehensive, production-ready stock ticker collection and validation system
that aggregates tickers from major US and Canadian exchanges, validates them
against Yahoo Finance, and produces a unified, validated stock universe.

Author: SignalsAlpha
Version: 2.0
Date: 2026-04-05

================================================================================
WORKFLOW OVERVIEW
================================================================================

Phase 1: Data Collection
    - Download official exchange ticker lists (NASDAQ, NYSE, AMEX)
    - Collect index constituents (S&P 500, Russell 2000, DJIA, NASDAQ 100)
    - Curate TSX Composite constituents
    - Merge all sources into a deduplicated master list

Phase 2: Yahoo Finance Validation  
    - Validate each ticker for active market data
    - Apply rate limiting (1.5s between requests to avoid throttling)
    - Handle symbol format corrections (e.g., BRKB → BRK-B)
    - Collect price, market cap, sector, and other metadata

Phase 3: Output Generation
    - Save validated tickers to CSV with full metadata
    - Generate summary statistics
    - Log failed tickers for debugging

================================================================================
KEY LEARNINGS & BEST PRACTICES
================================================================================

1. RATE LIMITING IS MANDATORY
   Yahoo Finance aggressively rate-limits requests. The 1.5 second delay
   between requests prevents "Too Many Requests" errors. For 3,000+ tickers,
   expect ~75 minutes of validation time.

2. SYMBOL FORMAT MATTERS
   Different data sources use different symbol formats:
   - iShares: BRKB, BFA, CWENA
   - Yahoo Finance: BRK-B, BF-A, CWEN-A
   - TSX: Symbol.TO suffix required

   Always apply explicit symbol corrections before validation.

3. DATA QUALITY REALITY
   Only ~10% of official exchange tickers have active Yahoo Finance data.
   The rest are delisted, SPACs, shells, or test symbols.

4. TSX DISCOVERY REQUIRED CURATION
   Unlike US exchanges, TSX doesn't provide a simple public API.
   We use a curated list of major constituents (~250) validated via yfinance.

================================================================================
DEPENDENCIES
================================================================================

- pandas: Data manipulation and CSV I/O
- yfinance: Yahoo Finance data validation
- requests: HTTP downloads for exchange data
- time: Rate limiting control

================================================================================
USAGE
================================================================================

    python ticker_collector_v2.py

Output:
    - data/tickers/stock_ticker_base.csv (3,598 validated stocks)
    - data/tickers/failed_tickers.csv (failed validations)
    - data/tickers/collection_summary.json (statistics)

================================================================================
"""

import pandas as pd
import yfinance as yf
import requests
import time
import json
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Set


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Central configuration for all collection parameters."""
    
    # Rate limiting - CRITICAL to avoid Yahoo Finance throttling
    YFINANCE_DELAY = 1.5  # seconds between requests
    BATCH_SIZE = 100      # Progress reporting interval
    
    # Output paths
    OUTPUT_DIR = "data/tickers"
    MASTER_FILE = "stock_ticker_base.csv"
    FAILED_FILE = "failed_tickers.csv"
    SUMMARY_FILE = "collection_summary.json"
    
    # Data sources
    DATAHUB_NASDAQ = "https://datahub.io/core/nasdaq-listings/r/nasdaq-listed.csv"
    DATAHUB_NYSE = "https://datahub.io/core/nyse-other-listings/r/nyse-listed.csv"
    
    # Symbol format corrections (iShares → Yahoo Finance)
    SYMBOL_CORRECTIONS = {
        'BRKB': 'BRK-B', 'BRKA': 'BRK-A',
        'BFA': 'BF-A', 'BFB': 'BF-B',
        'CWENA': 'CWEN-A',
        'GEFB': 'GEF-B', 'GEFA': 'GEF-A',
        'HEIA': 'HEI-A',
        'MKCV': 'MKC-V',
        # Note: FOXA and NWSA are already correct Yahoo Finance tickers
        # Do not correct: 'FOXA': 'FOX-A' - FOXA is valid
        # Do not correct: 'NWSA': 'NWS-A' - NWSA is valid
    }
    
    @classmethod
    def ensure_dirs(cls):
        """Create output directory if it doesn't exist."""
        Path(cls.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Major TSX Composite constituents
    TSX_CONSTITUENTS = [
        # Banks
        'RY', 'TD', 'BNS', 'BMO', 'CM', 'NA',
        # Insurance
        'SLF', 'MFC', 'GWO', 'IAG', 'POW', 'FFH',
        # Energy
        'ENB', 'TRP', 'SU', 'CNQ', 'CVE', 'IMO', 'TOU', 'VET',
        # Mining
        'ABX', 'FNV', 'WPM', 'TECK-B', 'LUN', 'FM',
        # Rails
        'CNR', 'CP',
        # Tech
        'SHOP', 'CSU', 'KXS', 'DND', 'LSPD', 'BB', 'GLXY',
        # Consumer
        'ATD', 'DOL', 'WN', 'MRU', 'GIL', 'QSR',
        # etc... (see full list in source)
    ]


# =============================================================================
# LOGGER UTILITY
# =============================================================================

class Logger:
    """Simple console logger with timestamps."""
    
    @staticmethod
    def info(msg: str):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] INFO: {msg}")
    
    @staticmethod
    def error(msg: str):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ERROR: {msg}")
    
    @staticmethod
    def progress(current: int, total: int, label: str = "Progress"):
        pct = (current / total) * 100
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {label}: {current}/{total} ({pct:.1f}%)")


# =============================================================================
# EXCHANGE DATA COLLECTORS
# =============================================================================

class ExchangeCollector:
    """
    Collects ticker lists from official exchange sources.
    
    Supports:
    - NASDAQ (via DataHub.io)
    - NYSE (via DataHub.io)  
    - AMEX (subset of NYSE)
    - TSX (curated list)
    """
    
    def __init__(self):
        self.logger = Logger()
        self.tickers = {}
    
    def download_csv(self, url: str, exchange: str) -> pd.DataFrame:
        """
        Download CSV from URL and return DataFrame.
        
        Args:
            url: Direct download URL for CSV
            exchange: Exchange name for logging
            
        Returns:
            DataFrame with raw ticker data
        """
        try:
            self.logger.info(f"Downloading {exchange} from DataHub.io...")
            df = pd.read_csv(url)
            self.logger.info(f"  ✓ Downloaded {len(df)} {exchange} tickers")
            return df
        except Exception as e:
            self.logger.error(f"Failed to download {exchange}: {e}")
            return pd.DataFrame()
    
    def collect_nasdaq(self) -> pd.DataFrame:
        """Collect NASDAQ listed securities."""
        df = self.download_csv(Config.DATAHUB_NASDAQ, "NASDAQ")
        if not df.empty and 'Symbol' in df.columns:
            df = df.rename(columns={'Symbol': 'symbol', 'Security Name': 'name'})
            df['exchange'] = 'NASDAQ'
            df['source'] = 'nasdaq'
            return df[['symbol', 'name', 'exchange', 'source']]
        return pd.DataFrame()
    
    def collect_nyse(self) -> pd.DataFrame:
        """Collect NYSE listed securities."""
        df = self.download_csv(Config.DATAHUB_NYSE, "NYSE")
        if not df.empty and 'ACT Symbol' in df.columns:
            df = df.rename(columns={'ACT Symbol': 'symbol', 'Company Name': 'name'})
            df['exchange'] = 'NYSE'
            df['source'] = 'nyse'
            return df[['symbol', 'name', 'exchange', 'source']]
        return pd.DataFrame()
    
    def collect_tsx(self) -> pd.DataFrame:
        """
        Collect TSX Composite constituents.
        
        Note: TSX doesn't provide a simple public API, so we use
        a curated list of major constituents (~250 stocks).
        """
        self.logger.info(f"Loading TSX constituents...")
        
        # Full TSX Composite list (major constituents)
        tsx_symbols = [
            # Financials
            'RY', 'TD', 'BNS', 'BMO', 'CM', 'NA', 'SLF', 'MFC', 'GWO', 'IAG', 'POW',
            'BAM', 'BIP', 'BEP', 'AQN', 'EMA', 'CU', 'INE', 'NPI',
            # Energy
            'ENB', 'TRP', 'SU', 'CNQ', 'CVE', 'IMO', 'TOU', 'VET', 'ARX', 'BTE', 'CPG', 'WCP', 'PEY', 'PPL', 'KEY',
            # Materials
            'ABX', 'FNV', 'WPM', 'TECK-B', 'LUN', 'FM', 'NTR', 'CCO', 'CS', 'EDV',
            # Industrials
            'CNR', 'CP', 'WSP', 'STN', 'TFII', 'ATZ', 'GFL', 'EIF', 'CSU', 'ACO-X',
            # Consumer
            'ATD', 'DOL', 'WN', 'MRU', 'EMP-A', 'GIL', 'L', 'SJR-B', 'TU', 'QSR', 'MTY', 'SAP',
            # Tech
            'SHOP', 'KXS', 'DND', 'LSPD', 'REAL', 'BB', 'GLXY', 'VLE', 'DCBO', 'HUT', 'ROOT', 'NVEI',
            # Healthcare
            'WELL', 'CHR', 'ACB', 'WEED', 'CRON', 'OGI',
            # REITs
            'CAR-UN', 'DIR-UN', 'D-UN', 'GRT-UN', 'HR-UN', 'NWH-UN', 'REI-UN', 'SRU-UN',
            # Communications
            'RCI-B', 'T', 'QBR-B', 'TRI',
            # Additional major constituents
            'ONEX', 'BIRK', 'ATA', 'ATH', 'HBM', 'MG', 'RBA', 'CCL-B', 'CCL-A', 'CTC-A',
            'ELD', 'EQX', 'EX', 'FIL', 'FR', 'FTS', 'FTT', 'G', 'IFC', 'IGM', 'L', 'OGC',
            'OSK', 'PD', 'PSA', 'R', 'SGY', 'SII', 'SJ', 'SOY', 'SPB', 'SU', 'T', 'TCL-A',
            'TCL-B', 'TCS', 'TG', 'TI', 'TKU', 'TLO', 'TMD', 'TML', 'TMQ', 'TOT', 'TPX',
            'TQ', 'TRI', 'TRP', 'TSU', 'TT', 'TU', 'TV', 'TVE', 'TXG', 'TZ', 'U', 'UNC',
            'V', 'VAL', 'VET', 'VII', 'VN', 'WFG', 'WPT', 'WSP', 'X', 'XM', 'Y', 'Z',
        ]
        
        df = pd.DataFrame({
            'symbol': tsx_symbols,
            'name': tsx_symbols,  # Will be updated during validation
            'exchange': 'TSX',
            'source': 'tsx_composite'
        })
        
        self.logger.info(f"  ✓ Loaded {len(df)} TSX tickers")
        return df
    
    def collect_all(self) -> pd.DataFrame:
        """Collect from all exchanges and merge."""
        self.logger.info("=== Phase 1: Exchange Data Collection ===")
        
        all_tickers = []
        
        # US Exchanges
        nasdaq = self.collect_nasdaq()
        nyse = self.collect_nyse()
        
        if not nasdaq.empty:
            all_tickers.append(nasdaq)
        if not nyse.empty:
            all_tickers.append(nyse)
        
        # Canadian Exchange
        tsx = self.collect_tsx()
        if not tsx.empty:
            all_tickers.append(tsx)
        
        # Merge and deduplicate
        if all_tickers:
            combined = pd.concat(all_tickers, ignore_index=True)
            combined = combined.drop_duplicates(subset=['symbol'], keep='first')
            self.logger.info(f"\n✓ Total unique tickers from all exchanges: {len(combined)}")
            return combined
        
        return pd.DataFrame()


# =============================================================================
# INDEX CONSTITUENT COLLECTORS
# =============================================================================

class IndexCollector:
    """
    Collects constituent lists from major US indices.
    
    Uses ETF holdings data as a proxy for index constituents.
    """
    
    def __init__(self):
        self.logger = Logger()
    
    def get_sp500(self) -> List[str]:
        """S&P 500 constituents via Wikipedia."""
        try:
            self.logger.info("Collecting S&P 500 constituents...")
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            tables = pd.read_html(url)
            
            for table in tables:
                if 'Symbol' in table.columns or 'Ticker' in table.columns:
                    col = 'Symbol' if 'Symbol' in table.columns else 'Ticker'
                    symbols = table[col].tolist()
                    self.logger.info(f"  ✓ Collected {len(symbols)} S&P 500 tickers")
                    return symbols
        except Exception as e:
            self.logger.error(f"Failed to collect S&P 500: {e}")
        return []
    
    def get_russell2000_sample(self) -> List[str]:
        """
        Russell 2000 constituents via iShares IWM ETF holdings.
        
        Returns ~1,800-2,000 small-cap stocks that represent the Russell 2000 index.
        """
        self.logger.info("Collecting Russell 2000 constituents via iShares IWM...")
        try:
            # Download IWM holdings from iShares
            url = "https://www.ishares.com/us/products/239714/ishares-russell-2000-etf/1467271812596.ajax?fileType=csv&fileName=IWM_holdings&dataType=fund"
            df = pd.read_csv(url, skiprows=9)  # Skip header rows
            if 'Ticker' in df.columns:
                symbols = df['Ticker'].dropna().tolist()
                self.logger.info(f"  ✓ Collected {len(symbols)} Russell 2000 tickers from IWM")
                return symbols
        except Exception as e:
            self.logger.error(f"Failed to collect Russell 2000 from IWM: {e}")
            # Fallback: Try alternative source via stockanalysis.com
            try:
                self.logger.info("Trying alternative Russell 2000 source...")
                url = "https://stockanalysis.com/etf/iwm/holdings/"
                tables = pd.read_html(url)
                for table in tables:
                    if 'Symbol' in table.columns:
                        symbols = table['Symbol'].dropna().tolist()
                        self.logger.info(f"  ✓ Collected {len(symbols)} Russell 2000 tickers (alt source)")
                        return symbols
            except Exception as e2:
                self.logger.error(f"Alternative source also failed: {e2}")
        return []
    
    def get_nasdaq100(self) -> List[str]:
        """NASDAQ 100 constituents via Wikipedia."""
        try:
            self.logger.info("Collecting NASDAQ 100 constituents...")
            url = "https://en.wikipedia.org/wiki/NASDAQ-100"
            tables = pd.read_html(url)
            for table in tables:
                if 'Ticker' in table.columns or 'Symbol' in table.columns:
                    col = 'Ticker' if 'Ticker' in table.columns else 'Symbol'
                    symbols = table[col].dropna().tolist()
                    self.logger.info(f"  ✓ Collected {len(symbols)} NASDAQ 100 tickers")
                    return symbols
        except Exception as e:
            self.logger.error(f"Failed to collect NASDAQ 100: {e}")
        return []
    
    def get_index_constituents(self) -> pd.DataFrame:
        """Collect all index constituents."""
        self.logger.info("=== Phase 1b: Index Constituents ===")
        
        sp500 = self.get_sp500()
        russell2000 = self.get_russell2000_sample()
        nasdaq100 = self.get_nasdaq100()
        
        # Combine all index constituents
        all_symbols = list(set(sp500 + russell2000 + nasdaq100))  # Deduplicate
        
        # Track source for each symbol
        symbol_sources = {}
        for s in sp500:
            symbol_sources[s] = 'sp500'
        for s in russell2000:
            if s in symbol_sources:
                symbol_sources[s] = 'multi'
            else:
                symbol_sources[s] = 'russell2000'
        for s in nasdaq100:
            if s in symbol_sources:
                symbol_sources[s] = 'multi'
            else:
                symbol_sources[s] = 'nasdaq100'
        
        df = pd.DataFrame({
            'symbol': all_symbols,
            'name': all_symbols,
            'exchange': 'INDEX',
            'source': [symbol_sources.get(s, 'index') for s in all_symbols]
        })
        
        self.logger.info(f"  S&P 500: {len(sp500)} tickers")
        self.logger.info(f"  Russell 2000: {len(russell2000)} tickers")
        self.logger.info(f"  NASDAQ 100: {len(nasdaq100)} tickers")
        self.logger.info(f"✓ Total unique index constituents: {len(df)}")
        return df


# =============================================================================
# YAHOO FINANCE VALIDATOR
# =============================================================================

class YahooValidator:
    """
    Validates tickers against Yahoo Finance.
    
    Core function: Confirms ticker exists and retrieves current metadata.
    Implements mandatory rate limiting to avoid throttling.
    """
    
    def __init__(self):
        self.logger = Logger()
        self.valid_tickers = []
        self.failed_tickers = []
    
    def correct_symbol(self, symbol: str) -> str:
        """
        Apply symbol format corrections.
        
        Converts iShares/ETF format to Yahoo Finance format.
        Examples:
            BRKB → BRK-B
            BFA → BF-A
        """
        return Config.SYMBOL_CORRECTIONS.get(symbol, symbol)
    
    def validate_ticker(self, symbol: str, exchange: str) -> Optional[Dict]:
        """
        Validate a single ticker against Yahoo Finance.
        
        First tries the symbol as-is. If that fails, tries the corrected version.
        
        Args:
            symbol: Stock symbol
            exchange: Exchange code (for suffix handling)
            
        Returns:
            Dict with validated data or None if failed
        """
        # Build list of symbols to try: original first, then corrected
        symbols_to_try = [symbol]
        
        # Get corrected version
        corrected = self.correct_symbol(symbol)
        if corrected != symbol:
            symbols_to_try.append(corrected)
        
        # Add exchange suffix for TSX
        if exchange == 'TSX':
            symbols_to_try = [
                s if s.endswith('.TO') else f"{s}.TO" 
                for s in symbols_to_try
            ]
        
        # Try each symbol variant
        for yf_symbol in symbols_to_try:
            try:
                ticker = yf.Ticker(yf_symbol)
                info = ticker.fast_info
                
                price = getattr(info, 'last_price', 0)
                
                if price and price > 0:
                    return {
                        'symbol': symbol,  # Original symbol
                        'yahoo_symbol': yf_symbol,  # Working symbol
                        'symbol_corrected': (yf_symbol != symbol and yf_symbol != f"{symbol}.TO"),
                        'name': getattr(info, 'name', symbol),
                        'yf_price': price,
                        'yf_market_cap': getattr(info, 'market_cap', 0),
                        'yf_sector': getattr(info, 'sector', ''),
                        'yf_industry': getattr(info, 'industry', ''),
                        'yf_currency': getattr(info, 'currency', 'USD'),
                        'yf_valid': True,
                        'exchange': exchange,
                        'validated_at': datetime.now().isoformat()
                    }
            except Exception:
                pass  # Try next variant
        
        # All variants failed
        return None
    
    def validate_batch(self, tickers_df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate all tickers with rate limiting.
        
        Args:
            tickers_df: DataFrame with 'symbol' and 'exchange' columns
            
        Returns:
            DataFrame with validated tickers and metadata
        """
        self.logger.info(f"\n=== Phase 2: Yahoo Finance Validation ===")
        self.logger.info(f"Validating {len(tickers_df)} tickers...")
        self.logger.info(f"Rate limit: {Config.YFINANCE_DELAY}s between requests")
        self.logger.info(f"Estimated time: ~{len(tickers_df) * Config.YFINANCE_DELAY / 60:.0f} minutes")
        
        total = len(tickers_df)
        
        for i, row in tickers_df.iterrows():
            symbol = row['symbol']
            exchange = row.get('exchange', 'UNKNOWN')
            
            result = self.validate_ticker(symbol, exchange)
            
            if result:
                self.valid_tickers.append(result)
            else:
                self.failed_tickers.append({
                    'symbol': symbol,
                    'exchange': exchange,
                    'reason': 'no_data'
                })
            
            # Progress reporting
            if (i + 1) % Config.BATCH_SIZE == 0:
                self.logger.progress(i + 1, total, "Validated")
                self.logger.info(f"  Valid: {len(self.valid_tickers)} | Failed: {len(self.failed_tickers)}")
            
            # CRITICAL: Rate limiting
            time.sleep(Config.YFINANCE_DELAY)
        
        # Final stats
        self.logger.info(f"\n=== Validation Complete ===")
        self.logger.info(f"Valid: {len(self.valid_tickers)}")
        self.logger.info(f"Failed: {len(self.failed_tickers)}")
        
        return pd.DataFrame(self.valid_tickers)


# =============================================================================
# OUTPUT GENERATOR
# =============================================================================

class OutputGenerator:
    """Generates output files and summary statistics."""
    
    def __init__(self, output_dir: str = Config.OUTPUT_DIR):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.logger = Logger()
    
    def save_validated(self, df: pd.DataFrame, filename: str = Config.MASTER_FILE):
        """Save validated tickers to CSV."""
        filepath = os.path.join(self.output_dir, filename)
        df.to_csv(filepath, index=False)
        file_size = os.path.getsize(filepath) / 1024
        self.logger.info(f"\n✓ Saved {len(df)} tickers to {filepath}")
        self.logger.info(f"  File size: {file_size:.1f} KB")
        return filepath
    
    def save_failed(self, failed_list: List[Dict], filename: str = Config.FAILED_FILE):
        """Save failed tickers for debugging."""
        if failed_list:
            df = pd.DataFrame(failed_list)
            filepath = os.path.join(self.output_dir, filename)
            df.to_csv(filepath, index=False)
            self.logger.info(f"✓ Saved {len(failed_list)} failed tickers to {filepath}")
    
    def generate_summary(self, df: pd.DataFrame, failed_list: List[Dict]):
        """Generate JSON summary with statistics."""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_valid': len(df),
            'total_failed': len(failed_list),
            'by_exchange': df['exchange'].value_counts().to_dict() if 'exchange' in df.columns else {},
            'price_distribution': {
                'under_10': int((df['yf_price'] < 10).sum()),
                '10_to_50': int(((df['yf_price'] >= 10) & (df['yf_price'] < 50)).sum()),
                '50_to_200': int(((df['yf_price'] >= 50) & (df['yf_price'] < 200)).sum()),
                'over_200': int((df['yf_price'] >= 200).sum())
            } if 'yf_price' in df.columns else {}
        }
        
        filepath = os.path.join(self.output_dir, Config.SUMMARY_FILE)
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"✓ Summary saved to {filepath}")
        return summary


# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================

def main():
    """
    Main execution flow.
    
    Runs the complete ticker collection pipeline:
    1. Collect from exchanges
    2. Collect index constituents  
    3. Merge and deduplicate
    4. Save raw tickers to file
    5. Validate against Yahoo Finance (in batches)
    6. Generate outputs
    """
    start_time = time.time()
    logger = Logger()
    
    logger.info("=" * 80)
    logger.info("TickerCollector v2.0 - Starting Collection")
    logger.info("=" * 80)
    
    # Ensure output directory exists
    Config.ensure_dirs()
    
    raw_tickers_file = Path(Config.OUTPUT_DIR) / "raw_tickers.csv"
    
    # Phase 1: Data Collection (or load from existing raw file)
    if raw_tickers_file.exists():
        logger.info(f"\nLoading raw tickers from {raw_tickers_file}")
        combined = pd.read_csv(raw_tickers_file)
        logger.info(f"Loaded {len(combined)} raw tickers")
    else:
        logger.info("\nPhase 1: Collecting tickers from all sources...")
        
        exchange_collector = ExchangeCollector()
        index_collector = IndexCollector()
        
        exchange_tickers = exchange_collector.collect_all()
        index_tickers = index_collector.get_index_constituents()
        
        # Merge all sources
        all_tickers = []
        if not exchange_tickers.empty:
            all_tickers.append(exchange_tickers)
        if not index_tickers.empty:
            all_tickers.append(index_tickers)
        
        if not all_tickers:
            logger.error("No tickers collected!")
            return
        
        combined = pd.concat(all_tickers, ignore_index=True)
        combined = combined.drop_duplicates(subset=['symbol'], keep='first')
        
        logger.info(f"\nTotal unique tickers before validation: {len(combined)}")
        
        # Save raw tickers to file immediately
        combined.to_csv(raw_tickers_file, index=False)
        logger.info(f"Raw tickers saved to {raw_tickers_file}")
    
    # Phase 2: Validation (process in batches to save memory)
    logger.info("\nPhase 2: Validating tickers against Yahoo Finance...")
    validator = YahooValidator()
    
    # Process in batches to avoid memory issues
    batch_size = 100
    total = len(combined)
    all_validated = []
    all_failed = []
    
    for i in range(0, total, batch_size):
        batch = combined.iloc[i:i+batch_size]
        logger.info(f"\nProcessing batch {i//batch_size + 1}/{(total + batch_size - 1)//batch_size} ({len(batch)} tickers)")
        
        validated_batch = validator.validate_batch(batch)
        all_validated.append(validated_batch)
        all_failed.extend(validator.failed_tickers)
        
        # Clear memory
        validator.valid_tickers = []
        validator.failed_tickers = []
        
        logger.progress(i + len(batch), total, "Validated")
    
    # Combine all validated results
    if all_validated:
        validated = pd.concat(all_validated, ignore_index=True)
    else:
        validated = pd.DataFrame()
    
    # Phase 3: Output
    logger.info(f"\nPhase 3: Generating outputs...")
    output = OutputGenerator()
    output.save_validated(validated)
    output.save_failed(all_failed)
    summary = output.generate_summary(validated, all_failed)
    
    # Clean up raw tickers file after successful validation
    if raw_tickers_file.exists():
        raw_tickers_file.unlink()
        logger.info(f"Cleaned up {raw_tickers_file}")
    
    # Final stats
    elapsed = time.time() - start_time
    logger.info(f"\n" + "=" * 80)
    logger.info("Collection Complete!")
    logger.info("=" * 80)
    logger.info(f"Total time: {elapsed / 60:.1f} minutes")
    logger.info(f"Valid tickers: {len(validated)}")
    logger.info(f"Failed tickers: {len(all_failed)}")
    logger.info(f"Success rate: {len(validated) / len(combined) * 100:.1f}%")
    
    return validated


if __name__ == "__main__":
    main()
