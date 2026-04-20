#!/usr/bin/env python3
"""
================================================================================
STEP 4: Scoring & Ranking
================================================================================

Signal generation engine that combines technical and fundamental analysis
into actionable trade recommendations with confidence scores.

This is the DECISION MAKING layer of the pipeline.

Author: SignalsAlpha
Version: 1.0
Date: 2026-04-20

================================================================================
PURPOSE
================================================================================

This module generates final trade signals by combining two analysis streams:

1. Technical Analysis (60% weight): From Step 3 - raw indicator values
   - Indicator alignment (RSI, MACD, Stochastic, etc.)
   - Crossover detection strength and location
   - Support/Resistance proximity
   - Fibonacci level positioning
   - Candlestick pattern confirmation
   - Volume confirmation

2. Fundamental Analysis (40% weight): From Step 2 - valuation metrics
   - Valuation Score (25 pts): P/E, PEG, Price/Book ratios
   - Profitability Score (25 pts): Margins, ROE, ROA
   - Growth Score (20 pts): Revenue growth, earnings growth
   - Financial Health (15 pts): Debt/Equity, current ratio
   - Analyst Sentiment (15 pts): Recommendations, price targets

Final Combined Score (0-100):
    - Technical Score × 0.60 + Fundamental Score × 0.40
    - Bonus: +15% when technical and fundamental agree
    - Penalty: -15% when they disagree

Signal Generation Thresholds:
    - STRONG_BUY: Combined >= 75, bullish consensus
    - BUY: Combined >= 60, technical or fundamental bullish
    - WEAK_BUY: Combined >= 40, technical bullish
    - HOLD: Everything else
    - WEAK_SELL: Combined >= 40, technical bearish
    - SELL: Combined >= 60, technical or fundamental bearish
    - STRONG_SELL: Combined >= 75, bearish consensus

================================================================================
INPUTS
================================================================================

Source Technical: data/technical_analysis/{TICKER}/{TICKER}_{interval}_technical.csv
    Columns: sma_20, rsi, macd, macd_cross_type, support_distance_pct, etc.

Source Fundamental: data/fundamentals/{TICKER}/{TICKER}_fundamentals.csv
    Columns: trailingPE, profitMargins, revenueGrowth, recommendationMean, etc.

================================================================================
OUTPUT
================================================================================

Destination: data/scored_signals/scored_signals_{date}.csv

Columns Added:
    - ticker: Stock symbol
    - interval: Timeframe (1d, 1wk, 1mo)
    - date: Analysis date
    - close: Current price

    Technical Scores:
        - technical_signal: STRONG_BUY/BUY/WEAK_BUY/HOLD/WEAK_SELL/SELL/STRONG_SELL
        - technical_confidence: 0-100%
        - technical_buy_score: Raw bullish points
        - technical_sell_score: Raw bearish points
        - technical_raw_score: Normalized 0-100

    Fundamental Scores:
        - fundamental_score: 0-100
        - fundamental_grade: A/B/C/D/F
        - fundamental_valuation: 0-25 pts
        - fundamental_profitability: 0-25 pts
        - fundamental_growth: 0-20 pts
        - fundamental_health: 0-15 pts
        - fundamental_analyst: 0-15 pts

    Combined Signal:
        - combined_score: 0-100 (weighted technical + fundamental)
        - final_signal: STRONG_BUY/BUY/WEAK_BUY/HOLD/WEAK_SELL/SELL/STRONG_SELL
        - conviction_pct: 0-100% confidence level

    Context:
        - market_regime: TRENDING_UP/TRENDING_DOWN/RANGING/NEUTRAL
        - adx: Current ADX value
        - volume_ratio: Current vs average volume
        - stop_loss: Calculated ATR-based stop
        - take_profit: Risk-reward based target

================================================================================
SCORING METHODOLOGY
================================================================================

Market Regime Classification:
    Uses ADX to determine trending vs ranging markets
    - ADX >= 25: Trending (higher weights on trend indicators)
    - ADX < 20: Ranging (higher weights on mean-reversion)

Technical Scoring Algorithm:
    Each indicator contributes +points to buy_score or sell_score:
    - RSI < 30: +weight(rsi) points
    - MACD crossover: +weight(macd) + cross_strength/3
    - Price > VWAP: +1 point
    - Stochastic < 20: +weight(stoch) points
    - Bollinger touch: +weight(bb) points
    - Volume > 1.5x avg: +1 point
    - S/R breakout: +2/+3 points
    
    Weights adjusted by regime (trending vs ranging)

Fundamental Scoring Algorithm:
    P/E Ratio:
        - PE < 15: +10 pts (value)
        - PE 15-25: +5 pts (fair value)
        - PE > 50: -5 pts (overvalued)
    
    PEG Ratio:
        - PEG < 1: +8 pts (growth at reasonable price)
        - PEG 1-2: +4 pts
        - PEG > 3: -3 pts
    
    Profit Margins:
        - Margin > 20%: +10 pts
        - Margin 10-20%: +6 pts
        - Margin < 5%: -5 pts
    
    ROE:
        - ROE > 20%: +10 pts (excellent)
        - ROE 15-20%: +6 pts (good)
        - ROE < 5%: -3 pts (poor)

Agreement Bonus:
    If both technical and fundamental agree on direction:
    - combined_score × 1.15 (15% boost)
    If they disagree:
    - combined_score × 0.85 (15% penalty)

"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Optional

# Configuration
DATA_DIR = Path("/home/ubuntu/.openclaw/workspace/trading-signals-data/process_flow_v1/data")
TECHNICAL_DIR = DATA_DIR / "technical_analysis"
FUNDAMENTALS_DIR = DATA_DIR / "fundamentals"
OUTPUT_DIR = DATA_DIR / "signals_scored"
INTERVALS = ['1d', '1wk', '1mo']

# Scoring weights
TECHNICAL_WEIGHT = 0.6
FUNDAMENTAL_WEIGHT = 0.4

print("Step 4 file created")

# ============================================================================
# TECHNICAL SCORING
# ============================================================================

def get_technical_weights(regime: str) -> Dict:
    """Get indicator weights based on market regime"""
    if regime in ['TRENDING_UP', 'TRENDING_DOWN']:
        return {'sma': 1, 'ema': 2, 'macd': 3, 'trix': 3, 'rsi': 1, 'stoch': 1, 'mfi': 1, 'bb': 1}
    elif regime == 'RANGING':
        return {'sma': 2, 'ema': 1, 'macd': 1, 'trix': 1, 'rsi': 3, 'stoch': 3, 'mfi': 3, 'bb': 3}
    else:
        return {'sma': 1, 'ema': 1, 'macd': 2, 'trix': 2, 'rsi': 2, 'stoch': 2, 'mfi': 2, 'bb': 2}

def score_support_resistance(close: float, prev_close: float, sr_levels: Dict) -> Tuple[int, int, List, float, float]:
    """Score support/resistance signals"""
    buy_points = 0
    sell_points = 0
    sr_context = []
    support_dist = 0
    resistance_dist = 0
    
    # Calculate distances
    if sr_levels.get('support'):
        nearest_support = sr_levels['support'][0]['price']
        support_dist = round((close - nearest_support) / close * 100, 2)
    
    if sr_levels.get('resistance'):
        nearest_resistance = sr_levels['resistance'][0]['price']
        resistance_dist = round((nearest_resistance - close) / close * 100, 2)
    
    # Support held/breakout
    if sr_levels.get('support'):
        for level in sr_levels['support']:
            if prev_close < level['price'] and close >= level['price']:
                buy_points += 3
                sr_context.append('SUPPORT_BREAKOUT')
                break
            elif prev_close > level['price'] * 1.01 and close >= level['price'] and close < prev_close * 1.005:
                buy_points += 2
                sr_context.append('SUPPORT_HELD')
                break
    
    # Resistance breakout/rejected
    if sr_levels.get('resistance'):
        for level in sr_levels['resistance']:
            if prev_close < level['price'] and close >= level['price']:
                buy_points += 3
                sr_context.append('RESISTANCE_BREAKOUT')
                break
            elif prev_close > level['price'] * 0.99 and close <= level['price'] and close > prev_close * 0.995:
                sell_points += 2
                sr_context.append('RESISTANCE_REJECTED')
                break
    
    return buy_points, sell_points, sr_context, support_dist, resistance_dist

def score_fibonacci_levels(close: float, prev_close: float, fib_levels: Dict, regime: str) -> Tuple[int, int, List]:
    """Score Fibonacci retracement signals"""
    buy_points = 0
    sell_points = 0
    fib_context = []
    
    if regime == 'RANGING':
        base_weight = 3
    elif regime == 'NEUTRAL':
        base_weight = 2
    else:
        base_weight = 1
    
    levels = {
        '23.6%': fib_levels.get('23.6%'),
        '38.2%': fib_levels.get('38.2%'),
        '50.0%': fib_levels.get('50.0%'),
        '61.8%': fib_levels.get('61.8%'),
        '78.6%': fib_levels.get('78.6%'),
        '100.0%': fib_levels.get('100.0%')
    }
    
    # Golden Zone (61.8% - 78.6%)
    if levels['61.8%'] and levels['78.6%']:
        golden_low = min(levels['61.8%'], levels['78.6%'])
        golden_high = max(levels['61.8%'], levels['78.6%'])
        
        if golden_low <= close <= golden_high:
            if prev_close > golden_high and close <= golden_high:
                buy_points += base_weight + 1
                fib_context.append('FIB_GOLDEN_ZONE_PULLBACK')
            elif prev_close < golden_low and close >= golden_low:
                buy_points += base_weight
                fib_context.append('FIB_GOLDEN_ZONE_BOUNCE')
        elif prev_close <= golden_high and close > golden_high:
            buy_points += base_weight
            fib_context.append('FIB_GOLDEN_ZONE_BREAKOUT')
        elif prev_close >= golden_low and close < golden_low:
            sell_points += base_weight
            fib_context.append('FIB_GOLDEN_ZONE_BREAKDOWN')
    
    return buy_points, sell_points, fib_context

def calculate_technical_score(ta_row: pd.Series) -> Tuple[str, int, int, int]:
    """
    Calculate technical score from Step 3 output
    Returns: (signal, confidence, buy_score, sell_score)
    """
    close = ta_row['Close']
    prev_close = close * 0.99  # Approximate
    
    # Get weights based on regime
    weights = get_technical_weights(ta_row['market_regime'])
    
    buy_score = 0
    sell_score = 0
    
    # VWAP
    if close > ta_row['vwap']:
        buy_score += 1
    else:
        sell_score += 1
    
    # SMA
    if close > ta_row['sma_20'] > ta_row['sma_50']:
        buy_score += weights['sma']
    elif close < ta_row['sma_20'] < ta_row['sma_50']:
        sell_score += weights['sma']
    
    # EMA
    if ta_row['ema_12'] > ta_row['ema_26']:
        buy_score += weights['ema']
    else:
        sell_score += weights['ema']
    
    # RSI
    if ta_row['rsi'] < 30:
        buy_score += weights['rsi']
    elif ta_row['rsi'] > 70:
        sell_score += weights['rsi']
    elif ta_row['rsi'] > 50:
        buy_score += weights['rsi'] // 2
    else:
        sell_score += weights['rsi'] // 2
    
    # MACD crossover
    if ta_row['macd_cross_type'] == 'crossover':
        buy_score += weights['macd'] + (ta_row['macd_cross_strength'] // 3)
    elif ta_row['macd_cross_type'] == 'crossunder':
        sell_score += weights['macd'] + (ta_row['macd_cross_strength'] // 3)
    elif ta_row['macd'] > ta_row['macd_signal']:
        buy_score += weights['macd'] // 2
    else:
        sell_score += weights['macd'] // 2
    
    # Bollinger Bands
    if close < ta_row['bb_lower']:
        buy_score += weights['bb']
    elif close > ta_row['bb_upper']:
        sell_score += weights['bb']
    
    # Stochastic Slow
    if ta_row['stoch_slow_cross'] == 'crossover':
        buy_score += weights['stoch'] + (ta_row['stoch_slow_strength'] // 3)
    elif ta_row['stoch_slow_cross'] == 'crossunder':
        sell_score += weights['stoch'] + (ta_row['stoch_slow_strength'] // 3)
    elif ta_row['stoch_k_slow'] < 20:
        buy_score += weights['stoch'] // 2
    elif ta_row['stoch_k_slow'] > 80:
        sell_score += weights['stoch'] // 2
    
    # Stochastic Fast
    if ta_row['stoch_fast_cross'] == 'crossover':
        buy_score += 1 + (ta_row['stoch_fast_strength'] // 5)
    elif ta_row['stoch_fast_cross'] == 'crossunder':
        sell_score += 1 + (ta_row['stoch_fast_strength'] // 5)
    
    # MFI
    if ta_row['mfi'] < 20:
        buy_score += weights['mfi']
    elif ta_row['mfi'] > 80:
        sell_score += weights['mfi']
    elif ta_row['mfi'] < 30:
        buy_score += weights['mfi'] // 2
    elif ta_row['mfi'] > 70:
        sell_score += weights['mfi'] // 2
    
    # TRIX
    if ta_row['trix_cross_type'] == 'crossover':
        buy_score += weights['trix'] + (ta_row['trix_cross_strength'] // 3)
    elif ta_row['trix_cross_type'] == 'crossunder':
        sell_score += weights['trix'] + (ta_row['trix_cross_strength'] // 3)
    elif ta_row['trix'] > ta_row['trix_signal']:
        buy_score += weights['trix'] // 2
    else:
        sell_score += weights['trix'] // 2
    
    # Volume ratio
    if ta_row['volume_ratio'] > 1.5:
        if buy_score > sell_score:
            buy_score += 1
        elif sell_score > buy_score:
            sell_score += 1
    
    # S/R scoring from pre-calculated distances
    if ta_row['support_distance_pct'] < 1 and ta_row['support_distance_pct'] > 0:
        buy_score += 2
    if ta_row['resistance_distance_pct'] < 1 and ta_row['resistance_distance_pct'] > 0:
        sell_score += 2
    
    # Determine signal
    signal = "HOLD"
    confidence = 50
    
    if buy_score >= 10:
        signal = "STRONG_BUY"
        confidence = min(95, 50 + (buy_score * 5))
    elif buy_score >= 6:
        signal = "BUY"
        confidence = min(95, 50 + (buy_score * 6))
    elif sell_score >= 10:
        signal = "STRONG_SELL"
        confidence = min(95, 50 + (sell_score * 5))
    elif sell_score >= 6:
        signal = "SELL"
        confidence = min(95, 50 + (sell_score * 6))
    elif buy_score >= 3:
        signal = "WEAK_BUY"
        confidence = min(95, 50 + (buy_score * 8))
    elif sell_score >= 3:
        signal = "WEAK_SELL"
        confidence = min(95, 50 + (sell_score * 8))
    
    return signal, confidence, buy_score, sell_score

print("Technical scoring functions added")

# ============================================================================
# FUNDAMENTAL SCORING
# ============================================================================

def calculate_fundamental_score(fund_row: pd.Series) -> Tuple[int, str, Dict]:
    """
    Calculate fundamental score from Step 2 output
    Returns: (score, quality_grade, breakdown)
    """
    score = 0
    breakdown = {}
    
    # Valuation Score (0-25)
    valuation_score = 0
    
    # P/E ratio (lower is better for value)
    pe = fund_row.get('trailingPE')
    if pd.notna(pe):
        if pe < 15:
            valuation_score += 10
        elif pe < 25:
            valuation_score += 5
        elif pe > 50:
            valuation_score -= 5
    breakdown['pe_score'] = valuation_score
    
    # PEG ratio (lower is better, <1 is good)
    peg = fund_row.get('pegRatio')
    if pd.notna(peg) and peg > 0:
        if peg < 1:
            valuation_score += 8
        elif peg < 2:
            valuation_score += 4
        elif peg > 3:
            valuation_score -= 3
    breakdown['peg_score'] = valuation_score - breakdown['pe_score']
    
    # Price to Book
    pb = fund_row.get('priceToBook')
    if pd.notna(pb):
        if pb < 3:
            valuation_score += 5
        elif pb > 10:
            valuation_score -= 3
    breakdown['pb_score'] = valuation_score - sum([breakdown.get('pe_score', 0), breakdown.get('peg_score', 0)])
    
    valuation_score = min(25, max(0, valuation_score))
    breakdown['valuation_total'] = valuation_score
    score += valuation_score
    
    # Profitability Score (0-25)
    profitability_score = 0
    
    # Profit margins
    profit_margin = fund_row.get('profitMargins')
    if pd.notna(profit_margin):
        if profit_margin > 0.20:
            profitability_score += 10
        elif profit_margin > 0.10:
            profitability_score += 6
        elif profit_margin > 0.05:
            profitability_score += 3
        elif profit_margin < 0:
            profitability_score -= 5
    
    # ROE
    roe = fund_row.get('returnOnEquity')
    if pd.notna(roe):
        if roe > 0.20:
            profitability_score += 10
        elif roe > 0.15:
            profitability_score += 6
        elif roe > 0.10:
            profitability_score += 3
        elif roe < 0.05:
            profitability_score -= 3
    
    # Operating margin
    op_margin = fund_row.get('operatingMargins')
    if pd.notna(op_margin):
        if op_margin > 0.15:
            profitability_score += 5
        elif op_margin < 0.05:
            profitability_score -= 2
    
    profitability_score = min(25, max(0, profitability_score))
    breakdown['profitability_total'] = profitability_score
    score += profitability_score
    
    # Growth Score (0-20)
    growth_score = 0
    
    # Revenue growth
    rev_growth = fund_row.get('revenueGrowth')
    if pd.notna(rev_growth):
        if rev_growth > 0.20:
            growth_score += 10
        elif rev_growth > 0.10:
            growth_score += 6
        elif rev_growth > 0.05:
            growth_score += 3
        elif rev_growth < 0:
            growth_score -= 3
    
    # Earnings growth
    earn_growth = fund_row.get('earningsGrowth')
    if pd.notna(earn_growth):
        if earn_growth > 0.25:
            growth_score += 10
        elif earn_growth > 0.15:
            growth_score += 6
        elif earn_growth > 0.05:
            growth_score += 3
        elif earn_growth < 0:
            growth_score -= 3
    
    growth_score = min(20, max(0, growth_score))
    breakdown['growth_total'] = growth_score
    score += growth_score
    
    # Financial Health (0-15)
    health_score = 0
    
    # Debt to Equity
    de = fund_row.get('debtToEquity')
    if pd.notna(de):
        if de < 50:
            health_score += 8
        elif de < 100:
            health_score += 4
        elif de > 200:
            health_score -= 3
    
    # Current ratio
    cr = fund_row.get('currentRatio')
    if pd.notna(cr):
        if cr > 2:
            health_score += 4
        elif cr > 1:
            health_score += 2
        elif cr < 1:
            health_score -= 2
    
    # Free cash flow
    fcf = fund_row.get('freeCashflow')
    if pd.notna(fcf) and fcf > 0:
        health_score += 3
    elif pd.notna(fcf) and fcf < 0:
        health_score -= 2
    
    health_score = min(15, max(0, health_score))
    breakdown['health_total'] = health_score
    score += health_score
    
    # Analyst Sentiment (0-15)
    analyst_score = 0
    
    # Recommendation mean (1-5 scale, lower is better)
    rec_mean = fund_row.get('recommendationMean')
    if pd.notna(rec_mean):
        if rec_mean <= 1.5:
            analyst_score += 10  # Strong Buy
        elif rec_mean <= 2.5:
            analyst_score += 6   # Buy
        elif rec_mean <= 3.5:
            analyst_score += 2   # Hold
        elif rec_mean > 4:
            analyst_score -= 5   # Sell
    
    # Price target upside
    target_mean = fund_row.get('targetMeanPrice')
    current_price = fund_row.get('regularMarketPrice')
    if pd.notna(target_mean) and pd.notna(current_price) and current_price > 0:
        upside = (target_mean - current_price) / current_price * 100
        if upside > 30:
            analyst_score += 5
        elif upside > 15:
            analyst_score += 3
        elif upside > 5:
            analyst_score += 1
        elif upside < -10:
            analyst_score -= 3
    
    # Number of analysts
    num_analysts = fund_row.get('numberOfAnalystOpinions')
    if pd.notna(num_analysts):
        if num_analysts >= 20:
            analyst_score += 2  # Strong coverage
        elif num_analysts >= 10:
            analyst_score += 1
        elif num_analysts < 3:
            analyst_score -= 2  # Weak coverage
    
    analyst_score = min(15, max(0, analyst_score))
    breakdown['analyst_total'] = analyst_score
    score += analyst_score
    
    # Quality grade
    if score >= 80:
        grade = 'A'
    elif score >= 65:
        grade = 'B'
    elif score >= 50:
        grade = 'C'
    elif score >= 35:
        grade = 'D'
    else:
        grade = 'F'
    
    return score, grade, breakdown

print("Fundamental scoring functions added")

# ============================================================================
# COMBINED SCORING
# ============================================================================

def calculate_combined_score(technical_score: int, fundamental_score: int, 
                            tech_signal: str, fund_grade: str) -> Tuple[int, str, float]:
    """
    Calculate combined score from technical and fundamental scores
    Returns: (combined_score, final_signal, conviction)
    """
    # Normalize technical score to 0-100
    # Technical raw scores: buy_score and sell_score typically 0-20
    # Convert to 0-100 scale
    tech_normalized = min(100, technical_score * 5)
    
    # Fundamental is already 0-100
    fund_normalized = fundamental_score
    
    # Weighted average
    combined = (tech_normalized * TECHNICAL_WEIGHT) + (fund_normalized * FUNDAMENTAL_WEIGHT)
    
    # Adjust based on agreement
    # If both bullish or both bearish, boost score
    # If disagreement, reduce score
    multiplier = 1.0
    
    tech_bullish = tech_signal in ['STRONG_BUY', 'BUY', 'WEAK_BUY']
    tech_bearish = tech_signal in ['STRONG_SELL', 'SELL', 'WEAK_SELL']
    fund_bullish = fund_grade in ['A', 'B']
    fund_bearish = fund_grade in ['D', 'F']
    
    if (tech_bullish and fund_bullish) or (tech_bearish and fund_bearish):
        multiplier = 1.15  # 15% boost when aligned
    elif (tech_bullish and fund_bearish) or (tech_bearish and fund_bullish):
        multiplier = 0.85  # 15% penalty when opposed
    
    combined = min(100, combined * multiplier)
    
    # Determine final signal
    if combined >= 75:
        if tech_bullish:
            final_signal = 'STRONG_BUY'
        elif tech_bearish:
            final_signal = 'STRONG_SELL'
        else:
            final_signal = 'BUY' if fund_bullish else 'SELL'
    elif combined >= 60:
        if tech_bullish:
            final_signal = 'BUY'
        elif tech_bearish:
            final_signal = 'SELL'
        else:
            final_signal = 'HOLD'
    elif combined >= 40:
        final_signal = 'WEAK_BUY' if tech_bullish else 'WEAK_SELL' if tech_bearish else 'HOLD'
    else:
        final_signal = 'HOLD'
    
    # Conviction level (0-100%)
    conviction = combined
    
    return int(combined), final_signal, conviction

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def load_technical_data(ticker: str, interval: str) -> Optional[pd.DataFrame]:
    """Load technical analysis data from Step 3"""
    file_path = TECHNICAL_DIR / ticker / f"{ticker}_{interval}_technical.csv"
    if file_path.exists():
        return pd.read_csv(file_path)
    return None

def load_fundamental_data(ticker: str) -> Optional[pd.DataFrame]:
    """Load fundamental data from Step 2"""
    file_path = FUNDAMENTALS_DIR / ticker / f"{ticker}_fundamentals.csv"
    if file_path.exists():
        return pd.read_csv(file_path)
    return None

def process_ticker(ticker: str, interval: str) -> Optional[Dict]:
    """Process single ticker for scoring"""
    # Load technical data
    ta_df = load_technical_data(ticker, interval)
    if ta_df is None or ta_df.empty:
        return None
    
    ta_row = ta_df.iloc[0]
    
    # Calculate technical score
    tech_signal, tech_confidence, tech_buy, tech_sell = calculate_technical_score(ta_row)
    tech_raw_score = max(tech_buy, tech_sell)
    
    # Load fundamental data
    fund_df = load_fundamental_data(ticker)
    fund_score = 50
    fund_grade = 'C'
    fund_breakdown = {}
    
    if fund_df is not None and not fund_df.empty:
        fund_row = fund_df.iloc[0]
        fund_score, fund_grade, fund_breakdown = calculate_fundamental_score(fund_row)
    
    # Calculate combined score
    combined_score, final_signal, conviction = calculate_combined_score(
        tech_raw_score, fund_score, tech_signal, fund_grade
    )
    
    return {
        'ticker': ticker,
        'interval': interval,
        'date': ta_row['Date'],
        'close': ta_row['Close'],
        
        # Technical scores
        'technical_signal': tech_signal,
        'technical_confidence': tech_confidence,
        'technical_buy_score': tech_buy,
        'technical_sell_score': tech_sell,
        'technical_raw_score': tech_raw_score,
        
        # Fundamental scores
        'fundamental_score': fund_score,
        'fundamental_grade': fund_grade,
        'fundamental_valuation': fund_breakdown.get('valuation_total', 0),
        'fundamental_profitability': fund_breakdown.get('profitability_total', 0),
        'fundamental_growth': fund_breakdown.get('growth_total', 0),
        'fundamental_health': fund_breakdown.get('health_total', 0),
        'fundamental_analyst': fund_breakdown.get('analyst_total', 0),
        
        # Combined scores
        'combined_score': combined_score,
        'final_signal': final_signal,
        'conviction_pct': round(conviction, 1),
        
        # Market context
        'market_regime': ta_row['market_regime'],
        'adx': ta_row['adx'],
        'volume_ratio': ta_row['volume_ratio'],
        
        # Price targets
        'stop_loss': ta_row['stop_loss'],
        'take_profit': ta_row['take_profit'],
    }

def ensure_dirs():
    """Create output directories"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def get_tickers_with_technical_data() -> List[str]:
    """Get list of tickers that have technical analysis data"""
    tickers = []
    if not TECHNICAL_DIR.exists():
        return tickers
    
    for ticker_dir in TECHNICAL_DIR.iterdir():
        if ticker_dir.is_dir() and not ticker_dir.name.startswith('.'):
            tickers.append(ticker_dir.name)
    
    return tickers

def main():
    """Main processing loop"""
    print("="*70)
    print("STEP 4: Scoring & Ranking")
    print("Technical Weight: {:.0%}, Fundamental Weight: {:.0%}".format(
        TECHNICAL_WEIGHT, FUNDAMENTAL_WEIGHT))
    print("="*70)
    print()
    
    ensure_dirs()
    
    tickers = get_tickers_with_technical_data()
    tickers.sort()
    
    if not tickers:
        print("No tickers found with technical analysis data!")
        print(f"Run Step 3 first to generate technical data in: {TECHNICAL_DIR}")
        return
    
    print(f"Processing {len(tickers)} tickers...")
    print()
    
    all_results = []
    
    for ticker in tickers:
        for interval in INTERVALS:
            try:
                result = process_ticker(ticker, interval)
                if result:
                    all_results.append(result)
            except Exception as e:
                print(f"Error processing {ticker} {interval}: {e}")
    
    if all_results:
        # Save all results
        df = pd.DataFrame(all_results)
        output_file = OUTPUT_DIR / f"scored_signals_{pd.Timestamp.now().strftime('%Y%m%d')}.csv"
        df.to_csv(output_file, index=False)
        
        # Summary statistics
        print("\n" + "="*70)
        print("SCORING SUMMARY")
        print("="*70)
        print(f"Total signals generated: {len(df)}")
        print(f"\nFinal Signal Distribution:")
        print(df['final_signal'].value_counts())
        print(f"\nTechnical Signal Distribution:")
        print(df['technical_signal'].value_counts())
        print(f"\nFundamental Grade Distribution:")
        print(df['fundamental_grade'].value_counts())
        print(f"\nAverage Scores:")
        print(f"  Technical Raw Score: {df['technical_raw_score'].mean():.1f}")
        print(f"  Fundamental Score: {df['fundamental_score'].mean():.1f}")
        print(f"  Combined Score: {df['combined_score'].mean():.1f}")
        print(f"\nOutput saved to: {output_file}")
        print("="*70)
    else:
        print("No signals generated!")

if __name__ == "__main__":
    main()
