#!/usr/bin/env python3
"""
================================================================================
STEP 2: Fundamental Data Collector
================================================================================

A comprehensive fundamental data retrieval system that fetches key financial
metrics, analyst ratings, and price targets from Yahoo Finance for all tickers
that have time series data. Includes date tracking for data freshness.

Author: SignalsAlpha
Version: 1.0
Date: 2026-04-20

================================================================================
PURPOSE
================================================================================

This module enriches the trading signal pipeline with fundamental data:

1. Retrieves valuation metrics (P/E, PEG, Price/Book, Price/Sales)
2. Collects profitability data (margins, ROE, ROA)
3. Gathers growth metrics (revenue growth, earnings growth)
4. Captures financial health indicators (debt/equity, current ratio)
5. Fetches analyst price targets and recommendations
6. Tracks institutional ownership and short interest
7. Stores dividend and beta information

================================================================================
DATA COLLECTED
================================================================================

Valuation Metrics:
    - trailingPE, forwardPE, pegRatio
    - priceToBook, priceToSalesTrailing12Months
    - enterpriseToEbitda, enterpriseToRevenue

Profitability:
    - profitMargins, grossMargins, ebitdaMargins
    - returnOnEquity, returnOnAssets, returnOnInvestment

Growth:
    - revenueGrowth, earningsGrowth
    - earningsQuarterlyGrowth

Financial Health:
    - totalDebt, totalCash
    - debtToEquity, currentRatio, quickRatio

Analyst Data:
    - recommendationMean, numberOfAnalystOpinions
    - targetHighPrice, targetLowPrice, targetMeanPrice
    - recommendationKey (strong buy/buy/hold/sell/strong sell)

Market Data:
    - marketCap, beta, dividendYield
    - fiftyTwoWeekHigh/Low
    - shortRatio, shortPercentOfFloat

Institutional:
    - heldPercentInstitutions, heldPercentInsiders

================================================================================
OUTPUT FORMAT
================================================================================

CSV files saved per ticker with columns:
    - ticker, collection_date (ISO format)
    - All fundamental fields (50+ metrics)
    - analyst fields with _analyst suffix

================================================================================
"""

import os
import sys
import json
import time
import pandas as pd
import yfinance as yf
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"
FUNDAMENTALS_DIR = DATA_DIR / "fundamentals"
TIME_SERIES_DIR = DATA_DIR / "time_series"

# Rate limiting
RATE_LIMIT_DELAY = 0.5  # seconds between API calls
BATCH_SIZE = 50  # Process in batches for memory management

# ============================================================================
# FUNDAMENTAL DATA FIELDS
# ============================================================================

VALUATION_FIELDS = [
    'trailingPE', 'forwardPE', 'pegRatio',
    'priceToBook', 'priceToSalesTrailing12Months',
    'enterpriseToEbitda', 'enterpriseToRevenue',
    'trailingPegRatio'
]

PROFITABILITY_FIELDS = [
    'profitMargins', 'grossMargins', 'ebitdaMargins',
    'operatingMargins', 'returnOnEquity', 'returnOnAssets',
    'returnOnInvestment'
]

GROWTH_FIELDS = [
    'revenueGrowth', 'earningsGrowth',
    'earningsQuarterlyGrowth', 'revenueQuarterlyGrowth'
]

FINANCIAL_HEALTH_FIELDS = [
    'totalDebt', 'totalCash', 'totalRevenue',
    'debtToEquity', 'currentRatio', 'quickRatio',
    'totalCashPerShare', 'totalDebtPerShare'
]

ANALYST_FIELDS = [
    'recommendationMean', 'numberOfAnalystOpinions',
    'targetHighPrice', 'targetLowPrice', 'targetMeanPrice',
    'targetMedianPrice', 'recommendationKey',
    'numberOfInstitutionalHolders'
]

MARKET_DATA_FIELDS = [
    'marketCap', 'beta', 'dividendYield', 'dividendRate',
    'exDividendDate', 'payoutRatio',
    'fiftyTwoWeekHigh', 'fiftyTwoWeekLow',
    'fiftyTwoWeekChange', 'fiftyDayAverage', 'twoHundredDayAverage',
    'shortRatio', 'shortPercentOfFloat', 'sharesShort',
    'sharesShortPriorMonth', 'sharesFloat', 'sharesOutstanding'
]

INSTITUTIONAL_FIELDS = [
    'heldPercentInstitutions', 'heldPercentInsiders',
    'institutionsCount', 'insidersPercentHeld'
]

INCOME_STATEMENT_FIELDS = [
    'totalRevenue', 'grossProfit', 'operatingIncome',
    'netIncome', 'ebitda', 'ebit'
]

SECTOR_INDUSTRY_FIELDS = [
    'sector', 'industry', 'country', 'state', 'city',
    'fullTimeEmployees', 'longName', 'shortName', 'longBusinessSummary'
]

EPS_FIELDS = [
    'epsTrailingTwelveMonths', 'epsForward', 'epsCurrentYear',
    'forwardEps', 'trailingEps', 'epsEstimateCurrentYear',
    'epsEstimateNextQuarter', 'epsEstimateNextYear'
]

RISK_FIELDS = [
    'auditRisk', 'boardRisk', 'compensationRisk', 'shareHolderRightsRisk',
    'overallRisk'
]

ADDITIONAL_ANALYST_FIELDS = [
    'recommendationLong', 'recommendationShort'
]

ADDITIONAL_MARKET_FIELDS = [
    'currency', 'financialCurrency', 'exchange', 'quoteType',
    'currentPrice', 'regularMarketVolume', 'averageVolume', 'averageVolume10days'
]

EARNINGS_CALENDAR_FIELDS = [
    'earningsDate', 'earningsDateStart', 'earningsDateEnd'
]

ALL_FIELDS = (VALUATION_FIELDS + PROFITABILITY_FIELDS + GROWTH_FIELDS +
              FINANCIAL_HEALTH_FIELDS + ANALYST_FIELDS + MARKET_DATA_FIELDS +
              INSTITUTIONAL_FIELDS + SECTOR_INDUSTRY_FIELDS + EPS_FIELDS +
              RISK_FIELDS + ADDITIONAL_ANALYST_FIELDS + ADDITIONAL_MARKET_FIELDS +
              EARNINGS_CALENDAR_FIELDS)

# ============================================================================
# DATA COLLECTION FUNCTIONS
# ============================================================================

def ensure_directories():
    """Create necessary directories"""
    FUNDAMENTALS_DIR.mkdir(parents=True, exist_ok=True)


def get_tickers_with_time_series() -> List[str]:
    """Get list of tickers that have time series data"""
    tickers = []
    
    if not TIME_SERIES_DIR.exists():
        print(f"Time series directory not found: {TIME_SERIES_DIR}")
        return tickers
    
    for ticker_dir in TIME_SERIES_DIR.iterdir():
        if ticker_dir.is_dir() and not ticker_dir.name.startswith('.'):
            # Check if ticker has data files
            data_files = list(ticker_dir.glob("*.csv"))
            if data_files:
                tickers.append(ticker_dir.name)
    
    return tickers


def fetch_fundamental_data(ticker: str) -> Dict[str, Any]:
    """
    Fetch all available fundamental data for a ticker from yfinance
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Dictionary with all fundamental data fields
    """
    data = {
        'ticker': ticker,
        'collection_date': datetime.now().isoformat(),
        'collection_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'data_available': False
    }
    
    try:
        # Create Ticker object
        yticker = yf.Ticker(ticker)
        
        # Get info (contains most fundamental data)
        info = yticker.info
        
        if not info or len(info) < 10:
            print(f"  Limited data for {ticker}")
            return data
        
        data['data_available'] = True
        
        # Extract all available fields
        for field in ALL_FIELDS:
            if field in info and info[field] is not None:
                value = info[field]
                # Convert to appropriate type
                if isinstance(value, (int, float)):
                    data[field] = value
                elif isinstance(value, str):
                    data[field] = value
                else:
                    data[field] = str(value)
            else:
                data[field] = None
        
        # Get additional financial data from financials
        try:
            financials = yticker.financials
            if financials is not None and not financials.empty:
                # Get most recent values
                latest_col = financials.columns[0]
                data['financials_revenue'] = financials.loc.get('Total Revenue', [None])[0] if 'Total Revenue' in financials.index else None
                data['financials_net_income'] = financials.loc.get('Net Income', [None])[0] if 'Net Income' in financials.index else None
                data['financials_gross_profit'] = financials.loc.get('Gross Profit', [None])[0] if 'Gross Profit' in financials.index else None
        except Exception as e:
            data['financials_error'] = str(e)
        
        # Get balance sheet data
        try:
            balance = yticker.balance_sheet
            if balance is not None and not balance.empty:
                latest_col = balance.columns[0]
                data['balance_total_assets'] = balance.loc.get('Total Assets', [None])[0] if 'Total Assets' in balance.index else None
                data['balance_total_liabilities'] = balance.loc.get('Total Liabilities Net Minority Interest', [None])[0] if 'Total Liabilities Net Minority Interest' in balance.index else None
                data['balance_stockholders_equity'] = balance.loc.get('Stockholders Equity', [None])[0] if 'Stockholders Equity' in balance.index else None
        except Exception as e:
            data['balance_error'] = str(e)
        
        # Get cash flow data
        try:
            cashflow = yticker.cashflow
            if cashflow is not None and not cashflow.empty:
                latest_col = cashflow.columns[0]
                data['cashflow_operating'] = cashflow.loc.get('Operating Cash Flow', [None])[0] if 'Operating Cash Flow' in cashflow.index else None
                data['cashflow_free'] = cashflow.loc.get('Free Cash Flow', [None])[0] if 'Free Cash Flow' in cashflow.index else None
                data['cashflow_capital_expenditure'] = cashflow.loc.get('Capital Expenditure', [None])[0] if 'Capital Expenditure' in cashflow.index else None
        except Exception as e:
            data['cashflow_error'] = str(e)
        
        # Get quarterly data for more recent metrics
        try:
            quarterly = yticker.quarterly_financials
            if quarterly is not None and not quarterly.empty:
                latest_col = quarterly.columns[0]
                data['quarterly_revenue'] = quarterly.loc.get('Total Revenue', [None])[0] if 'Total Revenue' in quarterly.index else None
                data['quarterly_net_income'] = quarterly.loc.get('Net Income', [None])[0] if 'Net Income' in quarterly.index else None
        except Exception as e:
            data['quarterly_error'] = str(e)
        
        # Calculate derived metrics
        if data.get('marketCap') and data.get('totalRevenue'):
            try:
                data['priceToSales_calculated'] = data['marketCap'] / data['totalRevenue'] if data['totalRevenue'] > 0 else None
            except:
                pass
        
        if data.get('returnOnEquity') is not None:
            data['roe_percent'] = data['returnOnEquity'] * 100
        
        if data.get('profitMargins') is not None:
            data['profit_margin_percent'] = data['profitMargins'] * 100
        
        # Analyst data summary
        if data.get('recommendationMean') is not None:
            mean_rec = data['recommendationMean']
            if mean_rec <= 1.5:
                data['analyst_sentiment'] = 'STRONG_BUY'
            elif mean_rec <= 2.5:
                data['analyst_sentiment'] = 'BUY'
            elif mean_rec <= 3.5:
                data['analyst_sentiment'] = 'HOLD'
            elif mean_rec <= 4.5:
                data['analyst_sentiment'] = 'SELL'
            else:
                data['analyst_sentiment'] = 'STRONG_SELL'
        
        # Price target upside/downside
        if data.get('targetMeanPrice') and data.get('regularMarketPrice'):
            current = data.get('regularMarketPrice')
            target = data['targetMeanPrice']
            if current > 0:
                data['analyst_upside_percent'] = ((target - current) / current) * 100
        
    except Exception as e:
        data['error'] = str(e)
        data['data_available'] = False
    
    return data


def save_fundamental_data(ticker: str, data: Dict[str, Any]) -> bool:
    """
    Save fundamental data to CSV file
    
    Args:
        ticker: Stock ticker symbol
        data: Dictionary with fundamental data
        
    Returns:
        True if saved successfully
    """
    try:
        # Create ticker directory
        ticker_dir = FUNDAMENTALS_DIR / ticker
        ticker_dir.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV
        output_file = ticker_dir / f"{ticker}_fundamentals.csv"
        
        # Convert to DataFrame (single row)
        df = pd.DataFrame([data])
        
        # Sort columns for consistency
        cols = sorted(df.columns.tolist())
        # Move key columns to front
        priority_cols = ['ticker', 'collection_date', 'collection_timestamp', 'data_available']
        other_cols = [c for c in cols if c not in priority_cols]
        final_cols = priority_cols + other_cols
        
        df = df[final_cols]
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        
        return True
        
    except Exception as e:
        print(f"  Error saving data for {ticker}: {e}")
        return False


def process_ticker_batch(tickers: List[str], batch_num: int, total_batches: int) -> Dict[str, int]:
    """
    Process a batch of tickers
    
    Args:
        tickers: List of ticker symbols
        batch_num: Current batch number
        total_batches: Total number of batches
        
    Returns:
        Dictionary with success/failure counts
    """
    results = {'success': 0, 'failed': 0, 'no_data': 0}
    
    print(f"\nProcessing batch {batch_num}/{total_batches} ({len(tickers)} tickers)...")
    
    for i, ticker in enumerate(tickers, 1):
        print(f"  [{i}/{len(tickers)}] Fetching {ticker}...", end=' ')
        
        try:
            data = fetch_fundamental_data(ticker)
            
            if data['data_available']:
                if save_fundamental_data(ticker, data):
                    results['success'] += 1
                    print("✓")
                else:
                    results['failed'] += 1
                    print("✗ Save failed")
            else:
                results['no_data'] += 1
                print("⚠ No data")
            
            # Rate limiting
            time.sleep(RATE_LIMIT_DELAY)
            
        except Exception as e:
            results['failed'] += 1
            print(f"✗ Error: {e}")
    
    return results


def main():
    """Main entry point"""
    print("=" * 80)
    print("STEP 2: Fundamental Data Collector")
    print("=" * 80)
    print(f"Data directory: {FUNDAMENTALS_DIR}")
    print(f"Rate limit: {RATE_LIMIT_DELAY}s between requests")
    print(f"Batch size: {BATCH_SIZE}")
    print("=" * 80)
    
    # Ensure directories exist
    ensure_directories()
    
    # Get tickers with time series data
    print("\nScanning for tickers with time series data...")
    tickers = get_tickers_with_time_series()
    
    if not tickers:
        print("No tickers found with time series data!")
        print("Please run Step 1 (Time Series Collector) first.")
        sys.exit(1)
    
    print(f"Found {len(tickers)} tickers with time series data")
    
    # Calculate batches
    total_batches = (len(tickers) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Processing in {total_batches} batches of {BATCH_SIZE}")
    
    # Process all batches
    total_results = {'success': 0, 'failed': 0, 'no_data': 0}
    
    for batch_num in range(1, total_batches + 1):
        start_idx = (batch_num - 1) * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, len(tickers))
        batch_tickers = tickers[start_idx:end_idx]
        
        batch_results = process_ticker_batch(batch_tickers, batch_num, total_batches)
        
        total_results['success'] += batch_results['success']
        total_results['failed'] += batch_results['failed']
        total_results['no_data'] += batch_results['no_data']
        
        # Progress update
        progress_pct = (batch_num / total_batches) * 100
        print(f"\n  Batch {batch_num} complete: {batch_results['success']} success, "
              f"{batch_results['failed']} failed, {batch_results['no_data']} no data")
        print(f"  Overall progress: {progress_pct:.1f}%")
    
    # Final summary
    print("\n" + "=" * 80)
    print("COLLECTION COMPLETE")
    print("=" * 80)
    print(f"Total tickers processed: {len(tickers)}")
    print(f"Successfully collected: {total_results['success']}")
    print(f"Failed: {total_results['failed']}")
    print(f"No data available: {total_results['no_data']}")
    print(f"Data saved to: {FUNDAMENTALS_DIR}")
    print("=" * 80)
    
    # Save summary
    summary = {
        'collection_timestamp': datetime.now().isoformat(),
        'total_tickers': len(tickers),
        'successful': total_results['success'],
        'failed': total_results['failed'],
        'no_data': total_results['no_data'],
        'fundamentals_directory': str(FUNDAMENTALS_DIR)
    }
    
    summary_file = FUNDAMENTALS_DIR / "collection_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to: {summary_file}")


if __name__ == "__main__":
    main()
