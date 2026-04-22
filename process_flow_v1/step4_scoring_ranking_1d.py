#!/usr/bin/env python3
"""
================================================================================
STEP 4: Scoring & Ranking v3 - 1D INTERVAL ONLY
================================================================================

Signal generation engine focused exclusively on 1-day interval signals
for active trading decisions.

Technical Weight: 75%, Fundamental Weight: 25%

Includes detailed score breakdowns and human-readable key drivers.

Author: SignalsAlpha
Version: 3.0
Date: 2026-04-21
================================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = Path("/home/ubuntu/.openclaw/workspace/trading-signals-data/process_flow_v1/data")
TECHNICAL_DIR = DATA_DIR / "technical_analysis"
FUNDAMENTALS_DIR = DATA_DIR / "fundamentals"
OUTPUT_DIR = DATA_DIR / "signals_scored"
INTERVALS = ['1d']  # Only 1-day interval
VALID_SIGNALS = ['STRONG_BUY', 'BUY', 'WEAK_BUY', 'HOLD', 'WEAK_SELL', 'SELL', 'STRONG_SELL']

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Scoring weights (75% technical, 25% fundamental)
TECHNICAL_WEIGHT = 0.75
FUNDAMENTAL_WEIGHT = 0.25


def get_technical_weights(regime: str) -> Dict:
    """Get indicator weights based on market regime"""
    if regime in ['TRENDING_UP', 'TRENDING_DOWN']:
        return {'sma': 1, 'ema': 2, 'macd': 3, 'trix': 3, 'rsi': 1, 'stoch': 1, 'mfi': 1, 'bb': 1}
    elif regime == 'RANGING':
        return {'sma': 2, 'ema': 1, 'macd': 1, 'trix': 1, 'rsi': 3, 'stoch': 3, 'mfi': 3, 'bb': 3}
    else:
        return {'sma': 1, 'ema': 1, 'macd': 2, 'trix': 2, 'rsi': 2, 'stoch': 2, 'mfi': 2, 'bb': 2}


def calculate_technical_score_detailed(ta_row: pd.Series) -> Tuple[int, int, List[str], List[str]]:
    """Calculate technical score with detailed breakdown"""
    close = ta_row['Close']
    prev_close = close * 0.99
    
    weights = get_technical_weights(ta_row['market_regime'])
    
    buy_score = 0
    sell_score = 0
    drivers = []
    detailed_scores = []
    
    # VWAP
    vwap = ta_row.get('vwap', 0)
    if close > vwap and vwap > 0:
        buy_score += 1
        drivers.append(f"Price ${close:.2f} above VWAP ${vwap:.2f}")
        detailed_scores.append("VWAP:+1")
    elif close < vwap and vwap > 0:
        sell_score += 1
        detailed_scores.append("VWAP:-1")
    
    # SMA
    sma_20 = ta_row.get('sma_20', 0)
    sma_50 = ta_row.get('sma_50', 0)
    if close > sma_20 > sma_50 and sma_50 > 0:
        buy_score += weights['sma']
        drivers.append(f"Above SMA20(${sma_20:.2f}) > SMA50(${sma_50:.2f})")
        detailed_scores.append(f"SMA:+{weights['sma']}")
    elif close < sma_20 < sma_50 and sma_50 > 0:
        sell_score += weights['sma']
        detailed_scores.append(f"SMA:-{weights['sma']}")
    
    # EMA
    ema_12 = ta_row.get('ema_12', 0)
    ema_26 = ta_row.get('ema_26', 0)
    if ema_12 > ema_26 and ema_26 > 0:
        buy_score += weights['ema']
        drivers.append(f"EMA12 > EMA26 bullish")
        detailed_scores.append(f"EMA:+{weights['ema']}")
    elif ema_12 < ema_26 and ema_26 > 0:
        sell_score += weights['ema']
        detailed_scores.append(f"EMA:-{weights['ema']}")
    
    # RSI
    rsi = ta_row.get('rsi', 50)
    if rsi < 30:
        buy_score += weights['rsi']
        drivers.append(f"RSI oversold {rsi:.1f}")
        detailed_scores.append(f"RSI:+{weights['rsi']}")
    elif rsi > 70:
        sell_score += weights['rsi']
        detailed_scores.append(f"RSI:-{weights['rsi']}")
    else:
        detailed_scores.append(f"RSI:0({rsi:.1f})")
    
    # MACD
    macd_cross = ta_row.get('macd_cross_type', 'none')
    macd_strength = ta_row.get('macd_cross_strength', 0)
    if macd_cross == 'crossover':
        buy_score += weights['macd'] + int(macd_strength / 3)
        drivers.append(f"MACD crossover (strength: {macd_strength})")
        detailed_scores.append(f"MACD:+{weights['macd'] + int(macd_strength / 3)}")
    elif macd_cross == 'crossunder':
        sell_score += weights['macd'] + int(macd_strength / 3)
        detailed_scores.append(f"MACD:-{weights['macd'] + int(macd_strength / 3)}")
    
    # Bollinger
    bb_upper = ta_row.get('bb_upper', 0)
    bb_lower = ta_row.get('bb_lower', 0)
    if close >= bb_upper and bb_upper > 0:
        sell_score += weights['bb']
        detailed_scores.append(f"BB:-{weights['bb']}")
    elif close <= bb_lower and bb_lower > 0:
        buy_score += weights['bb']
        drivers.append(f"Price at lower BB")
        detailed_scores.append(f"BB:+{weights['bb']}")
    
    # Stochastic
    stoch_cross = ta_row.get('stoch_slow_cross', 'none')
    if stoch_cross == 'crossover':
        buy_score += weights['stoch']
        drivers.append("Stoch crossover")
        detailed_scores.append(f"Stoch:+{weights['stoch']}")
    elif stoch_cross == 'crossunder':
        sell_score += weights['stoch']
        detailed_scores.append(f"Stoch:-{weights['stoch']}")
    
    # Volume
    vol_ratio = ta_row.get('volume_ratio', 1.0)
    if vol_ratio > 1.5:
        buy_score += 1
        drivers.append(f"Volume {vol_ratio:.1f}x avg")
        detailed_scores.append("Volume:+1")
    elif vol_ratio < 0.5:
        sell_score += 1
        detailed_scores.append("Volume:-1")
    
    # Support/Resistance
    support_dist = ta_row.get('support_distance_pct', 0)
    resistance_dist = ta_row.get('resistance_distance_pct', 0)
    if support_dist < 2 and support_dist > 0:
        buy_score += 2
        drivers.append(f"Near support ({support_dist:.1f}%)")
        detailed_scores.append("S/R:+2")
    if resistance_dist < 2 and resistance_dist > 0:
        sell_score += 2
        detailed_scores.append("S/R:-2")
    
    return buy_score, sell_score, drivers, detailed_scores


def calculate_fundamental_score_detailed(fund_row: pd.Series) -> Tuple[int, List[str], List[str]]:
    """Calculate fundamental score with detailed breakdown"""
    score = 0
    drivers = []
    detailed_scores = []
    
    # P/E
    pe = fund_row.get('trailingPE', None)
    if pd.notna(pe) and pe > 0:
        if pe < 15:
            score += 10
            drivers.append(f"P/E {pe:.1f} deep value")
            detailed_scores.append(f"P/E:+10({pe:.1f})")
        elif pe < 25:
            score += 5
            detailed_scores.append(f"P/E:+5({pe:.1f})")
        elif pe > 50:
            score -= 5
            detailed_scores.append(f"P/E:-5({pe:.1f})")
    
    # PEG
    peg = fund_row.get('pegRatio', None)
    if pd.notna(peg) and peg > 0:
        if peg < 1:
            score += 8
            drivers.append(f"PEG {peg:.2f} < 1")
            detailed_scores.append(f"PEG:+8({peg:.2f})")
        elif peg < 2:
            score += 4
            detailed_scores.append(f"PEG:+4({peg:.2f})")
        elif peg > 3:
            score -= 3
            detailed_scores.append(f"PEG:-3({peg:.2f})")
    
    # Margin
    margin = fund_row.get('profitMargins', 0)
    if pd.notna(margin):
        margin_pct = margin * 100
        if margin_pct > 20:
            score += 10
            drivers.append(f"Margin {margin_pct:.1f}% excellent")
            detailed_scores.append(f"Margin:+10({margin_pct:.1f}%)")
        elif margin_pct > 10:
            score += 6
            detailed_scores.append(f"Margin:+6({margin_pct:.1f}%)")
        elif margin_pct < 5:
            score -= 5
            detailed_scores.append(f"Margin:-5({margin_pct:.1f}%)")
    
    # ROE
    roe = fund_row.get('returnOnEquity', 0)
    if pd.notna(roe):
        roe_pct = roe * 100
        if roe_pct > 20:
            score += 10
            drivers.append(f"ROE {roe_pct:.1f}% excellent")
            detailed_scores.append(f"ROE:+10({roe_pct:.1f}%)")
        elif roe_pct > 15:
            score += 6
            detailed_scores.append(f"ROE:+6({roe_pct:.1f}%)")
        elif roe_pct < 5:
            score -= 3
            detailed_scores.append(f"ROE:-3({roe_pct:.1f}%)")
    
    # Debt/Equity
    debt_eq = fund_row.get('debtToEquity', None)
    if pd.notna(debt_eq):
        if debt_eq < 0.5:
            score += 5
            detailed_scores.append(f"D/E:+5({debt_eq:.2f})")
        elif debt_eq > 2:
            score -= 5
            detailed_scores.append(f"D/E:-5({debt_eq:.2f})")
    
    # Growth
    rev_growth = fund_row.get('revenueGrowth', None)
    if pd.notna(rev_growth):
        growth_pct = rev_growth * 100
        if growth_pct > 20:
            score += 8
            drivers.append(f"Growth {growth_pct:.1f}% strong")
            detailed_scores.append(f"Growth:+8({growth_pct:.1f}%)")
        elif growth_pct > 10:
            score += 4
            detailed_scores.append(f"Growth:+4({growth_pct:.1f}%)")
        elif growth_pct < 0:
            score -= 3
            detailed_scores.append(f"Growth:-3({growth_pct:.1f}%)")
    
    # Analyst
    rec_mean = fund_row.get('recommendationMean', None)
    if pd.notna(rec_mean):
        if rec_mean <= 2:
            score += 5
            drivers.append(f"Analyst {rec_mean:.1f} bullish")
            detailed_scores.append(f"Analyst:+5({rec_mean:.1f})")
        elif rec_mean >= 4:
            score -= 3
            detailed_scores.append(f"Analyst:-3({rec_mean:.1f})")
    
    return score, drivers, detailed_scores


def get_grade(score: int) -> str:
    """Convert score to letter grade"""
    if score >= 80:
        return 'A'
    elif score >= 65:
        return 'B'
    elif score >= 50:
        return 'C'
    elif score >= 35:
        return 'D'
    else:
        return 'F'


def determine_signal(tech_buy: int, tech_sell: int, fund_score: int) -> Tuple[str, int, float]:
    """Determine final signal and combined score"""
    tech_raw = tech_buy - tech_sell
    tech_score = max(0, min(100, (tech_raw + 20) * 2.5))
    fund_score_norm = max(0, min(100, fund_score + 50))
    
    combined = (tech_score * TECHNICAL_WEIGHT) + (fund_score_norm * FUNDAMENTAL_WEIGHT)
    
    tech_bullish = tech_buy > tech_sell
    fund_bullish = fund_score > 0
    
    if tech_bullish and fund_bullish:
        combined *= 1.15
    elif not tech_bullish and not fund_bullish:
        combined *= 1.15
    elif tech_bullish != fund_bullish:
        combined *= 0.85
    
    combined = min(100, combined)
    
    if combined >= 75 and tech_buy > tech_sell and fund_score > 0:
        signal = 'STRONG_BUY'
    elif combined >= 60 and (tech_buy > tech_sell or fund_score > 10):
        signal = 'BUY'
    elif combined >= 40 and tech_buy > tech_sell:
        signal = 'WEAK_BUY'
    elif combined >= 75 and tech_sell > tech_buy and fund_score < 0:
        signal = 'STRONG_SELL'
    elif combined >= 60 and (tech_sell > tech_buy or fund_score < -10):
        signal = 'SELL'
    elif combined >= 40 and tech_sell > tech_buy:
        signal = 'WEAK_SELL'
    else:
        signal = 'HOLD'
    
    conviction = min(100, combined)
    
    return signal, int(combined), conviction


def process_ticker(ticker: str, interval: str = '1d') -> Optional[Dict]:
    """Process a single ticker for 1d interval"""
    
    tech_file = TECHNICAL_DIR / ticker / f"{ticker}_{interval}_technical.csv"
    if not tech_file.exists():
        return None
    
    try:
        ta_df = pd.read_csv(tech_file)
        if len(ta_df) == 0:
            return None
        ta_row = ta_df.iloc[0]
    except Exception:
        return None
    
    fund_file = FUNDAMENTALS_DIR / ticker / f"{ticker}_fundamentals.csv"
    fund_score = 0
    fund_grade = 'F'
    fund_drivers = []
    fund_detailed = []
    
    if fund_file.exists():
        try:
            fund_df = pd.read_csv(fund_file)
            if len(fund_df) > 0:
                fund_row = fund_df.iloc[0]
                fund_score, fund_drivers, fund_detailed = calculate_fundamental_score_detailed(fund_row)
                fund_grade = get_grade(fund_score + 50)
        except Exception:
            pass
    
    tech_buy, tech_sell, tech_drivers, tech_detailed = calculate_technical_score_detailed(ta_row)
    
    final_signal, combined_score, conviction = determine_signal(tech_buy, tech_sell, fund_score)
    
    tech_raw = tech_buy - tech_sell
    tech_confidence = min(100, max(0, (tech_raw + 20) * 2.5))
    
    key_drivers = []
    if tech_drivers:
        key_drivers.extend(tech_drivers[:2])
    if fund_drivers:
        key_drivers.extend(fund_drivers[:2])
    
    key_drivers_str = "; ".join(key_drivers) if key_drivers else "No strong drivers"
    
    return {
        'ticker': ticker,
        'interval': interval,
        'date': ta_row.get('Date', datetime.now().strftime('%Y-%m-%d')),
        'close': ta_row.get('Close', 0),
        
        'technical_signal': 'BUY' if tech_buy > tech_sell else 'SELL' if tech_sell > tech_buy else 'HOLD',
        'technical_confidence': round(tech_confidence, 1),
        'technical_buy_score': tech_buy,
        'technical_sell_score': tech_sell,
        'technical_raw_score': tech_raw,
        'technical_drivers': '|'.join(tech_drivers) if tech_drivers else 'None',
        'technical_detailed_scores': '|'.join(tech_detailed),
        
        'fundamental_score': fund_score + 50,
        'fundamental_grade': fund_grade,
        'fundamental_drivers': '|'.join(fund_drivers) if fund_drivers else 'None',
        'fundamental_detailed_scores': '|'.join(fund_detailed),
        
        'combined_score': combined_score,
        'final_signal': final_signal,
        'conviction_pct': round(conviction, 1),
        'key_drivers': key_drivers_str,
        
        'market_regime': ta_row.get('market_regime', 'NEUTRAL'),
        'adx': ta_row.get('adx', 0),
        'volume_ratio': ta_row.get('volume_ratio', 1.0),
        'stop_loss': ta_row.get('stop_loss', 0),
        'take_profit': ta_row.get('take_profit', 0),
    }


def main():
    print("="*70)
    print("STEP 4: Scoring & Ranking v3 - 1D INTERVAL ONLY")
    print("Technical Weight: 75%, Fundamental Weight: 25%")
    print("="*70)
    print()
    
    tickers = []
    if TECHNICAL_DIR.exists():
        for ticker_dir in TECHNICAL_DIR.iterdir():
            if ticker_dir.is_dir() and not ticker_dir.name.startswith('.'):
                tickers.append(ticker_dir.name)
    
    tickers.sort()
    print(f"Processing {len(tickers)} tickers (1d interval only)...")
    print()
    
    results = []
    processed = 0
    errors = 0
    
    for ticker in tickers:
        try:
            result = process_ticker(ticker, '1d')
            if result:
                results.append(result)
                processed += 1
        except Exception as e:
            errors += 1
            print(f"Error: {ticker}: {e}")
    
    if results:
        df = pd.DataFrame(results)
        output_file = OUTPUT_DIR / f"scored_signals_1d_{datetime.now().strftime('%Y%m%d')}.csv"
        df.to_csv(output_file, index=False)
        
        print(f"\n{'='*70}")
        print("1D SIGNAL RANKING SUMMARY")
        print(f"{'='*70}")
        print(f"Total 1d signals: {len(results)}")
        print(f"Processed: {processed}, Errors: {errors}")
        print(f"\nSignal Distribution:")
        print(df['final_signal'].value_counts())
        print(f"\nGrade Distribution:")
        print(df['fundamental_grade'].value_counts().sort_index())
        print(f"\nTop 10 STRONG BUY:")
        top_buy = df[df['final_signal'] == 'STRONG_BUY'].nlargest(10, 'combined_score')
        for idx, row in top_buy.iterrows():
            print(f"  {row['ticker']:6s} | Score: {row['combined_score']:2d} | Grade: {row['fundamental_grade']} | Tech: {row['technical_buy_score']}/{row['technical_sell_score']}")
        print(f"\nOutput saved to: {output_file}")
        print(f"{'='*70}")
    else:
        print("No results generated!")


if __name__ == "__main__":
    main()
