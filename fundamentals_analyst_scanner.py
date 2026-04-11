#!/usr/bin/env python3
"""
Fundamentals + Analyst Scanner
Extracts fundamental data and analyst targets from Yahoo Finance raw data
"""

import os
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuration
BASE_DIR = Path("/home/ubuntu/.openclaw/workspace/data")
RAW_TICKERS_DIR = BASE_DIR / "raw_tickers"
OUTPUT_DIR = BASE_DIR / "signals_fundamentals"

def ensure_dir(path):
    """Create directory if it doesn't exist"""
    path.mkdir(parents=True, exist_ok=True)

def load_raw_ticker_data(ticker):
    """Load raw Yahoo Finance data for a ticker"""
    file_path = RAW_TICKERS_DIR / f"{ticker}.json"
    if not file_path.exists():
        return None
    
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {ticker}: {e}")
        return None

def extract_fundamentals(data):
    """Extract fundamental metrics"""
    info = data.get('info', {})
    
    fundamentals = {
        # Valuation
        'pe_trailing': info.get('trailingPE'),
        'pe_forward': info.get('forwardPE'),
        'peg_ratio': info.get('pegRatio'),
        'price_to_book': info.get('priceToBook'),
        'price_to_sales': info.get('priceToSalesTrailing12Months'),
        'enterprise_value': info.get('enterpriseValue'),
        'ev_to_ebitda': info.get('enterpriseToEbitda'),
        'ev_to_revenue': info.get('enterpriseToRevenue'),
        
        # Profitability
        'profit_margin': info.get('profitMargins'),
        'operating_margin': info.get('operatingMargins'),
        'gross_margin': info.get('grossMargins'),
        'ebitda_margin': info.get('ebitdaMargins'),
        'return_on_assets': info.get('returnOnAssets'),
        'return_on_equity': info.get('returnOnEquity'),
        
        # Financial Health
        'debt_to_equity': info.get('debtToEquity'),
        'current_ratio': info.get('currentRatio'),
        'quick_ratio': info.get('quickRatio'),
        'total_cash': info.get('totalCash'),
        'total_debt': info.get('totalDebt'),
        'total_revenue': info.get('totalRevenue'),
        'revenue_growth': info.get('revenueGrowth'),
        'earnings_growth': info.get('earningsGrowth'),
        
        # Per Share
        'eps_trailing': info.get('trailingEps'),
        'eps_forward': info.get('forwardEps'),
        'book_value': info.get('bookValue'),
        'free_cashflow': info.get('freeCashflow'),
        
        # Company Info
        'sector': info.get('sector'),
        'industry': info.get('industry'),
        'employees': info.get('fullTimeEmployees'),
        'market_cap': info.get('marketCap'),
        'beta': info.get('beta'),
        'dividend_yield': info.get('dividendYield'),
        'ex_dividend_date': info.get('exDividendDate'),
    }
    
    return fundamentals

def extract_analyst_data(data):
    """Extract analyst recommendations and price targets"""
    info = data.get('info', {})
    
    analyst = {
        # Recommendation consensus
        'recommendation_mean': info.get('recommendationMean'),  # 1-5 scale
        'recommendation_key': info.get('recommendationKey'),  # strong_buy, buy, hold, etc.
        'num_analysts': info.get('numberOfAnalystOpinions'),
        
        # Price targets
        'target_low': info.get('targetLowPrice'),
        'target_mean': info.get('targetMeanPrice'),
        'target_median': info.get('targetMedianPrice'),
        'target_high': info.get('targetHighPrice'),
        
        # Current price for context
        'current_price': info.get('currentPrice'),
        'previous_close': info.get('previousClose'),
    }
    
    # Calculate upside potential
    if analyst['current_price'] and analyst['target_mean']:
        analyst['upside_potential'] = round(
            ((analyst['target_mean'] - analyst['current_price']) / analyst['current_price']) * 100, 2
        )
    else:
        analyst['upside_potential'] = None
    
    # Calculate consensus score (-2 to +2 scale)
    if analyst['recommendation_mean']:
        # Convert 1-5 scale to -2 to +2
        # 1 = Strong Buy (+2), 5 = Strong Sell (-2)
        raw_score = 3 - analyst['recommendation_mean']  # 2 to -2
        analyst['consensus_score'] = round(raw_score, 2)
    else:
        analyst['consensus_score'] = None
    
    return analyst

def generate_fundamental_signal(fundamentals, analyst):
    """Generate signal based on fundamentals and analyst data"""
    score = 0
    reasons = []
    
    # PE Ratio analysis
    pe = fundamentals.get('pe_trailing')
    if pe:
        if pe < 15:
            score += 2
            reasons.append(f"Low P/E ({pe:.1f})")
        elif pe < 25:
            score += 1
            reasons.append(f"Reasonable P/E ({pe:.1f})")
        elif pe > 40:
            score -= 1
            reasons.append(f"High P/E ({pe:.1f})")
    
    # PEG Ratio
    peg = fundamentals.get('peg_ratio')
    if peg:
        if peg < 1:
            score += 2
            reasons.append(f"Excellent PEG ({peg:.2f})")
        elif peg < 2:
            score += 1
            reasons.append(f"Good PEG ({peg:.2f})")
        elif peg > 3:
            score -= 1
            reasons.append(f"High PEG ({peg:.2f})")
    
    # Profit Margin
    margin = fundamentals.get('profit_margin')
    if margin:
        if margin > 0.2:
            score += 2
            reasons.append(f"Strong margins ({margin:.1%})")
        elif margin > 0.1:
            score += 1
            reasons.append(f"Good margins ({margin:.1%})")
        elif margin < 0:
            score -= 2
            reasons.append(f"Negative margins")
    
    # ROE
    roe = fundamentals.get('return_on_equity')
    if roe:
        if roe > 0.2:
            score += 2
            reasons.append(f"High ROE ({roe:.1%})")
        elif roe > 0.15:
            score += 1
            reasons.append(f"Good ROE ({roe:.1%})")
        elif roe < 0.05:
            score -= 1
            reasons.append(f"Low ROE ({roe:.1%})")
    
    # Debt/Equity
    de = fundamentals.get('debt_to_equity')
    if de:
        if de < 0.5:
            score += 1
            reasons.append(f"Low debt/equity ({de:.2f})")
        elif de > 2:
            score -= 1
            reasons.append(f"High debt/equity ({de:.2f})")
    
    # Revenue Growth
    rev_growth = fundamentals.get('revenue_growth')
    if rev_growth:
        if rev_growth > 0.2:
            score += 2
            reasons.append(f"Strong revenue growth ({rev_growth:.1%})")
        elif rev_growth > 0.1:
            score += 1
            reasons.append(f"Good revenue growth ({rev_growth:.1%})")
        elif rev_growth < 0:
            score -= 1
            reasons.append(f"Declining revenue")
    
    # Analyst consensus
    consensus = analyst.get('consensus_score')
    if consensus:
        if consensus >= 1.5:
            score += 3
            reasons.append(f"Strong analyst consensus ({consensus:.1f})")
        elif consensus >= 0.5:
            score += 2
            reasons.append(f"Positive analyst consensus ({consensus:.1f})")
        elif consensus <= -0.5:
            score -= 2
            reasons.append(f"Negative analyst consensus ({consensus:.1f})")
    
    # Upside potential
    upside = analyst.get('upside_potential')
    if upside:
        if upside > 30:
            score += 2
            reasons.append(f"High upside potential ({upside:.1f}%)")
        elif upside > 15:
            score += 1
            reasons.append(f"Good upside potential ({upside:.1f}%)")
        elif upside < -10:
            score -= 1
            reasons.append(f"Downside risk ({upside:.1f}%)")
    
    # Determine signal
    if score >= 8:
        signal = "STRONG_BUY"
    elif score >= 5:
        signal = "BUY"
    elif score >= 2:
        signal = "WEAK_BUY"
    elif score <= -5:
        signal = "STRONG_SELL"
    elif score <= -3:
        signal = "SELL"
    elif score <= -1:
        signal = "WEAK_SELL"
    else:
        signal = "HOLD"
    
    confidence = min(95, max(50, 50 + abs(score) * 5))
    
    return {
        'signal': signal,
        'score': score,
        'confidence': confidence,
        'reasons': "; ".join(reasons[:5])  # Top 5 reasons
    }

def scan_ticker(ticker):
    """Scan a single ticker for fundamentals and analyst data"""
    data = load_raw_ticker_data(ticker)
    if not data:
        return None
    
    fundamentals = extract_fundamentals(data)
    analyst = extract_analyst_data(data)
    
    # Skip if no analyst coverage
    if not analyst.get('num_analysts') or analyst['num_analysts'] == 0:
        return None
    
    signal_data = generate_fundamental_signal(fundamentals, analyst)
    
    return {
        'ticker': ticker,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        **fundamentals,
        **analyst,
        **signal_data
    }

def save_results(results, ticker):
    """Save results to CSV"""
    if not results:
        return
    
    ticker_dir = OUTPUT_DIR / ticker
    ensure_dir(ticker_dir)
    
    output_file = ticker_dir / f"{ticker}_fundamentals.csv"
    
    # Flatten results for CSV
    flat_results = {k: v for k, v in results.items()}
    df = pd.DataFrame([flat_results])
    df.to_csv(output_file, index=False)
    print(f"Saved {ticker} fundamentals to {output_file}")

def main():
    """Main execution"""
    ensure_dir(OUTPUT_DIR)
    
    # Get list of tickers from raw_tickers
    json_files = list(RAW_TICKERS_DIR.glob("*.json"))
    tickers = [f.stem for f in json_files]
    tickers.sort()
    
    print(f"Found {len(tickers)} tickers with raw data")
    print(f"Output directory: {OUTPUT_DIR}")
    print("-" * 60)
    
    processed = 0
    with_analyst = 0
    
    for ticker in tickers:
        result = scan_ticker(ticker)
        if result:
            save_results(result, ticker)
            with_analyst += 1
        processed += 1
        
        if processed % 100 == 0:
            print(f"Progress: {processed}/{len(tickers)} processed, {with_analyst} with analyst coverage")
    
    print("-" * 60)
    print(f"Complete! Processed {processed} tickers, {with_analyst} with analyst data")

if __name__ == "__main__":
    main()