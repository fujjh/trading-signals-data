#!/usr/bin/env python3
"""
================================================================================
STEP 4: Scoring & Ranking v2 - WITH DETAILED DRIVERS
================================================================================

Signal generation engine that combines technical and fundamental analysis
into actionable trade recommendations with confidence scores.

Technical Weight: 75%, Fundamental Weight: 25%

Includes detailed score breakdowns and human-readable key drivers.

Author: SignalsAlpha
Version: 2.1
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
INTERVALS = ['1d', '1wk', '1mo']
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
    """
    Calculate technical score with detailed breakdown.
    Returns: (buy_score, sell_score, technical_drivers, detailed_scores)
    """
    close = ta_row['Close']
    prev_close = close * 0.99  # Approximate for change calc
    
    weights = get_technical_weights(ta_row['market_regime'])
    
    buy_score = 0
    sell_score = 0
    drivers = []
    detailed_scores = []
    
    # VWAP - Price vs Volume Weighted Average Price
    vwap = ta_row.get('vwap', 0)
    if close > vwap and vwap > 0:
        buy_score += 1
        drivers.append(f"Price ${close:.2f} above VWAP ${vwap:.2f}")
        detailed_scores.append("VWAP:+1")
    elif close < vwap and vwap > 0:
        sell_score += 1
        drivers.append(f"Price ${close:.2f} below VWAP ${vwap:.2f}")
        detailed_scores.append("VWAP:-1")
    
    # SMA trend
    sma_20 = ta_row.get('sma_20', 0)
    sma_50 = ta_row.get('sma_50', 0)
    if close > sma_20 > sma_50 and sma_50 > 0:
        buy_score += weights['sma']
        drivers.append(f"Price above SMA20(${sma_20:.2f}) > SMA50(${sma_50:.2f}) bullish trend")
        detailed_scores.append(f"SMA:+{weights['sma']}")
    elif close < sma_20 < sma_50 and sma_50 > 0:
        sell_score += weights['sma']
        drivers.append(f"Price below SMA20(${sma_20:.2f}) < SMA50(${sma_50:.2f}) bearish trend")
        detailed_scores.append(f"SMA:-{weights['sma']}")
    
    # EMA crossover
    ema_12 = ta_row.get('ema_12', 0)
    ema_26 = ta_row.get('ema_26', 0)
    if ema_12 > ema_26 and ema_26 > 0:
        buy_score += weights['ema']
        drivers.append(f"EMA12(${ema_12:.2f}) above EMA26(${ema_26:.2f}) bullish")
        detailed_scores.append(f"EMA:+{weights['ema']}")
    elif ema_12 < ema_26 and ema_26 > 0:
        sell_score += weights['ema']
        drivers.append(f"EMA12(${ema_12:.2f}) below EMA26(${ema_26:.2f}) bearish")
        detailed_scores.append(f"EMA:-{weights['ema']}")
    
    # RSI - Momentum
    rsi = ta_row.get('rsi', 50)
    if rsi < 30:
        buy_score += weights['rsi']
        drivers.append(f"RSI oversold at {rsi:.1f} (<30)")
        detailed_scores.append(f"RSI:+{weights['rsi']}")
    elif rsi > 70:
        sell_score += weights['rsi']
        drivers.append(f"RSI overbought at {rsi:.1f} (>70)")
        detailed_scores.append(f"RSI:-{weights['rsi']}")
    else:
        detailed_scores.append(f"RSI:0({rsi:.1f})")
    
    # MACD - INCREASED CROSSOVER BONUS
    macd = ta_row.get('macd', 0)
    macd_signal = ta_row.get('macd_signal', 0)
    macd_cross = ta_row.get('macd_cross_type', 'none')
    macd_strength = ta_row.get('macd_cross_strength', 0)
    
    if macd_cross == 'crossover':
        # Increased bonus: base weight + strength/2 (was /3) + additional crossover bonus
        macd_bonus = weights['macd'] + int(macd_strength / 2) + 2
        buy_score += macd_bonus
        drivers.append(f"MACD bullish crossover (strength: {macd_strength}) [+{macd_bonus}]")
        detailed_scores.append(f"MACD:+{macd_bonus}(crossover)")
    elif macd_cross == 'crossunder':
        macd_bonus = weights['macd'] + int(macd_strength / 2) + 2
        sell_score += macd_bonus
        drivers.append(f"MACD bearish crossunder (strength: {macd_strength}) [-{macd_bonus}]")
        detailed_scores.append(f"MACD:-{macd_bonus}(crossunder)")
    elif macd > macd_signal:
        buy_score += 1
        detailed_scores.append("MACD:+1(above_signal)")
    elif macd < macd_signal:
        sell_score += 1
        detailed_scores.append("MACD:-1(below_signal)")
    
    # Bollinger Bands
    bb_upper = ta_row.get('bb_upper', 0)
    bb_lower = ta_row.get('bb_lower', 0)
    if close >= bb_upper and bb_upper > 0:
        sell_score += weights['bb']
        drivers.append(f"Price at upper Bollinger Band (${bb_upper:.2f})")
        detailed_scores.append(f"BB:-{weights['bb']}")
    elif close <= bb_lower and bb_lower > 0:
        buy_score += weights['bb']
        drivers.append(f"Price at lower Bollinger Band (${bb_lower:.2f})")
        detailed_scores.append(f"BB:+{weights['bb']}")
    
    # Stochastic
    stoch_k = ta_row.get('stoch_k_slow', 50)
    stoch_cross = ta_row.get('stoch_slow_cross', 'none')
    
    if stoch_cross == 'crossover':
        # Increased bonus: base weight + additional crossover bonus
        stoch_bonus = weights['stoch'] + 2
        buy_score += stoch_bonus
        drivers.append(f"Stochastic bullish crossover (%K: {stoch_k:.1f}) [+{stoch_bonus}]")
        detailed_scores.append(f"Stoch:+{stoch_bonus}(crossover)")
    elif stoch_cross == 'crossunder':
        stoch_bonus = weights['stoch'] + 2
        sell_score += stoch_bonus
        drivers.append(f"Stochastic bearish crossunder (%K: {stoch_k:.1f}) [-{stoch_bonus}]")
        detailed_scores.append(f"Stoch:-{stoch_bonus}(crossunder)")
    elif stoch_k < 20:
        buy_score += 1
        detailed_scores.append("Stoch:+1(oversold)")
    elif stoch_k > 80:
        sell_score += 1
        detailed_scores.append("Stoch:-1(overbought)")
    
    # Volume confirmation
    vol_ratio = ta_row.get('volume_ratio', 1.0)
    if vol_ratio > 1.5:
        buy_score += 1
        drivers.append(f"Volume {vol_ratio:.1f}x average (strong interest)")
        detailed_scores.append("Volume:+1")
    elif vol_ratio < 0.5:
        sell_score += 1
        detailed_scores.append("Volume:-1(low)")
    
    # HMA (Hull Moving Average) - New Indicator
    hma = ta_row.get('hma_13', 0)
    hma_signal = ta_row.get('hma_signal', 'none')
    hma_slope = ta_row.get('hma_slope_2bar', 0)
    hma_turn = ta_row.get('hma_turn', 'none')
    hma_peak_valley = ta_row.get('hma_peak_valley', 'none')
    
    if hma > 0:
        # Price vs HMA position - INCREASED CROSSOVER BONUS
        if hma_signal == 'cross_above':
            buy_score += 5  # Increased from 3
            drivers.append(f"HMA bullish crossover (price crossed above HMA13) [+5]")
            detailed_scores.append("HMA:+5(cross_above)")
        elif hma_signal == 'cross_below':
            sell_score += 5  # Increased from 3
            drivers.append(f"HMA bearish crossunder (price crossed below HMA13) [-5]")
            detailed_scores.append("HMA:-5(cross_below)")
        elif close > hma:
            buy_score += 1
            detailed_scores.append("HMA:+1(above)")
        elif close < hma:
            sell_score += 1
            detailed_scores.append("HMA:-1(below)")
        
        # HMA slope (2-bar) - momentum
        if hma_slope > 0.5:
            buy_score += 2
            drivers.append(f"HMA slope rising {hma_slope:.2f}% (strong momentum)")
            detailed_scores.append(f"HMA_Slope:+2({hma_slope:.2f}%)")
        elif hma_slope > 0.1:
            buy_score += 1
            detailed_scores.append(f"HMA_Slope:+1({hma_slope:.2f}%)")
        elif hma_slope < -0.5:
            sell_score += 2
            drivers.append(f"HMA slope falling {hma_slope:.2f}% (strong decline)")
            detailed_scores.append(f"HMA_Slope:-2({hma_slope:.2f}%)")
        elif hma_slope < -0.1:
            sell_score += 1
            detailed_scores.append(f"HMA_Slope:-1({hma_slope:.2f}%)")
        
        # HMA turn detection
        if hma_turn == 'up':
            buy_score += 2
            drivers.append("HMA turning up (reversal signal)")
            detailed_scores.append("HMA_Turn:+2")
        elif hma_turn == 'down':
            sell_score += 2
            drivers.append("HMA turning down (reversal signal)")
            detailed_scores.append("HMA_Turn:-2")
        
        # HMA peak/valley
        if hma_peak_valley == 'valley':
            buy_score += 1
            detailed_scores.append("HMA_Valley:+1")
        elif hma_peak_valley == 'peak':
            sell_score += 1
            detailed_scores.append("HMA_Peak:-1")
    
    # Elder Impulse System - New Indicator
    elder_impulse = ta_row.get('elder_impulse', 'blue')
    elder_strength = ta_row.get('elder_trend_strength', 5)
    
    if elder_impulse == 'green':
        # Green: EMA rising AND histogram rising - bullish
        buy_score += 3 + min(2, elder_strength // 4)  # Up to +5 for strong green
        drivers.append(f"Elder Impulse GREEN (strength: {elder_strength}/10) - bullish momentum")
        detailed_scores.append(f"Elder:+{3 + min(2, elder_strength // 4)}(green)")
    elif elder_impulse == 'red':
        # Red: EMA falling AND histogram falling - bearish
        sell_score += 3 + min(2, elder_strength // 4)
        drivers.append(f"Elder Impulse RED (strength: {elder_strength}/10) - bearish momentum")
        detailed_scores.append(f"Elder:-{3 + min(2, elder_strength // 4)}(red)")
    else:
        # Blue: mixed signals
        detailed_scores.append(f"Elder:0(blue,{elder_strength})")
    
    # Support/Resistance distance
    support_dist = ta_row.get('support_distance_pct', 0)
    resistance_dist = ta_row.get('resistance_distance_pct', 0)
    
    if support_dist < 2 and support_dist > 0:
        buy_score += 2
        drivers.append(f"Near support level ({support_dist:.1f}% away)")
        detailed_scores.append("S/R:+2(near support)")
    if resistance_dist < 2 and resistance_dist > 0:
        sell_score += 2
        drivers.append(f"Near resistance level ({resistance_dist:.1f}% away)")
        detailed_scores.append("S/R:-2(near resistance)")
    
    return buy_score, sell_score, drivers, detailed_scores


def calculate_fundamental_score_detailed(fund_row: pd.Series) -> Tuple[int, List[str], List[str]]:
    """
    Calculate fundamental score with detailed breakdown.
    Returns: (score, drivers, detailed_scores)
    """
    score = 0
    drivers = []
    detailed_scores = []
    
    # P/E Ratio
    pe = fund_row.get('trailingPE', None)
    if pd.notna(pe) and pe > 0:
        if pe < 15:
            score += 10
            drivers.append(f"P/E {pe:.1f} < 15 (deep value)")
            detailed_scores.append(f"P/E:+10({pe:.1f})")
        elif pe < 25:
            score += 5
            drivers.append(f"P/E {pe:.1f} fair value")
            detailed_scores.append(f"P/E:+5({pe:.1f})")
        elif pe > 50:
            score -= 5
            drivers.append(f"P/E {pe:.1f} > 50 (overvalued)")
            detailed_scores.append(f"P/E:-5({pe:.1f})")
        else:
            detailed_scores.append(f"P/E:0({pe:.1f})")
    
    # PEG Ratio
    peg = fund_row.get('pegRatio', None)
    if pd.notna(peg) and peg > 0:
        if peg < 1:
            score += 8
            drivers.append(f"PEG {peg:.2f} < 1 (growth at reasonable price)")
            detailed_scores.append(f"PEG:+8({peg:.2f})")
        elif peg < 2:
            score += 4
            detailed_scores.append(f"PEG:+4({peg:.2f})")
        elif peg > 3:
            score -= 3
            detailed_scores.append(f"PEG:-3({peg:.2f})")
    
    # Profit Margin
    margin = fund_row.get('profitMargins', 0)
    if pd.notna(margin):
        margin_pct = margin * 100
        if margin_pct > 20:
            score += 10
            drivers.append(f"Profit margin {margin_pct:.1f}% excellent")
            detailed_scores.append(f"Margin:+10({margin_pct:.1f}%)")
        elif margin_pct > 10:
            score += 6
            drivers.append(f"Profit margin {margin_pct:.1f}% good")
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
            drivers.append(f"ROE {roe_pct:.1f}% good")
            detailed_scores.append(f"ROE:+6({roe_pct:.1f}%)")
        elif roe_pct < 5:
            score -= 3
            detailed_scores.append(f"ROE:-3({roe_pct:.1f}%)")
    
    # Debt/Equity
    debt_eq = fund_row.get('debtToEquity', None)
    if pd.notna(debt_eq):
        if debt_eq < 0.5:
            score += 5
            drivers.append(f"Low debt/equity {debt_eq:.2f}")
            detailed_scores.append(f"D/E:+5({debt_eq:.2f})")
        elif debt_eq > 2:
            score -= 5
            detailed_scores.append(f"D/E:-5({debt_eq:.2f})")
    
    # Revenue Growth
    rev_growth = fund_row.get('revenueGrowth', None)
    if pd.notna(rev_growth):
        growth_pct = rev_growth * 100
        if growth_pct > 20:
            score += 8
            drivers.append(f"Revenue growing {growth_pct:.1f}% (strong)")
            detailed_scores.append(f"Growth:+8({growth_pct:.1f}%)")
        elif growth_pct > 10:
            score += 4
            detailed_scores.append(f"Growth:+4({growth_pct:.1f}%)")
        elif growth_pct < 0:
            score -= 3
            detailed_scores.append(f"Growth:-3({growth_pct:.1f}%)")
    
    # Analyst recommendation
    rec_mean = fund_row.get('recommendationMean', None)
    if pd.notna(rec_mean):
        if rec_mean <= 2:  # Strong Buy/Buy
            score += 5
            drivers.append(f"Analyst rating {rec_mean:.1f} (bullish)")
            detailed_scores.append(f"Analyst:+5({rec_mean:.1f})")
        elif rec_mean >= 4:  # Hold/Sell
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
    
    # Normalize scores to 0-100
    tech_raw = tech_buy - tech_sell
    tech_score = max(0, min(100, (tech_raw + 20) * 2.5))  # Scale to 0-100
    
    fund_score_norm = max(0, min(100, fund_score + 50))  # Shift to 0-100
    
    # Combined score (75% technical, 25% fundamental) - technical weighted higher
    combined = (tech_score * TECHNICAL_WEIGHT) + (fund_score_norm * FUNDAMENTAL_WEIGHT)
    
    # Agreement bonus/penalty
    tech_bullish = tech_buy > tech_sell
    fund_bullish = fund_score > 0
    
    if tech_bullish and fund_bullish:
        combined *= 1.15  # +15% bonus
    elif not tech_bullish and not fund_bullish:
        combined *= 1.15
    elif tech_bullish != fund_bullish:
        combined *= 0.85  # -15% penalty
    
    combined = min(100, combined)
    
    # Determine signal
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


def process_ticker(ticker: str, interval: str) -> Optional[Dict]:
    """Process a single ticker/interval and return detailed scoring data"""
    
    # Load technical data
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
    
    # Load fundamental data
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
    
    # Calculate technical score
    tech_buy, tech_sell, tech_drivers, tech_detailed = calculate_technical_score_detailed(ta_row)
    
    # Determine signal
    final_signal, combined_score, conviction = determine_signal(tech_buy, tech_sell, fund_score)
    
    # Technical raw score
    tech_raw = tech_buy - tech_sell
    tech_confidence = min(100, max(0, (tech_raw + 20) * 2.5))
    
    # Build key drivers summary (top 3 from each)
    key_drivers = []
    if tech_drivers:
        key_drivers.extend(tech_drivers[:2])
    if fund_drivers:
        key_drivers.extend(fund_drivers[:2])
    
    key_drivers_str = "; ".join(key_drivers) if key_drivers else "No strong drivers identified"
    
    return {
        'ticker': ticker,
        'interval': interval,
        'date': ta_row.get('Date', datetime.now().strftime('%Y-%m-%d')),
        'close': ta_row.get('Close', 0),
        
        # Technical scores
        'technical_signal': 'BUY' if tech_buy > tech_sell else 'SELL' if tech_sell > tech_buy else 'HOLD',
        'technical_confidence': round(tech_confidence, 1),
        'technical_buy_score': tech_buy,
        'technical_sell_score': tech_sell,
        'technical_raw_score': tech_raw,
        'technical_drivers': '|'.join(tech_drivers) if tech_drivers else 'None',
        'technical_detailed_scores': '|'.join(tech_detailed),
        
        # Fundamental scores
        'fundamental_score': fund_score + 50,  # Normalized to 0-100
        'fundamental_grade': fund_grade,
        'fundamental_drivers': '|'.join(fund_drivers) if fund_drivers else 'None',
        'fundamental_detailed_scores': '|'.join(fund_detailed),
        
        # Combined
        'combined_score': combined_score,
        'final_signal': final_signal,
        'conviction_pct': round(conviction, 1),
        'key_drivers': key_drivers_str,
        
        # Context
        'market_regime': ta_row.get('market_regime', 'NEUTRAL'),
        'adx': ta_row.get('adx', 0),
        'volume_ratio': ta_row.get('volume_ratio', 1.0),
        'stop_loss': ta_row.get('stop_loss', 0),
        'take_profit': ta_row.get('take_profit', 0),
    }


def main():
    print("="*70)
    print("STEP 4: Scoring & Ranking v2 - WITH DETAILED DRIVERS")
    print("Technical Weight: 60%, Fundamental Weight: 40%")
    print("="*70)
    print()
    
    # Get all tickers with technical analysis
    tickers = []
    if TECHNICAL_DIR.exists():
        for ticker_dir in TECHNICAL_DIR.iterdir():
            if ticker_dir.is_dir() and not ticker_dir.name.startswith('.'):
                tickers.append(ticker_dir.name)
    
    tickers.sort()
    print(f"Processing {len(tickers)} tickers...")
    print()
    
    results = []
    processed = 0
    errors = 0
    
    for ticker in tickers:
        for interval in INTERVALS:
            try:
                result = process_ticker(ticker, interval)
                if result:
                    results.append(result)
                    processed += 1
            except Exception as e:
                errors += 1
                print(f"Error: {ticker} {interval}: {e}")
    
    # Save results
    if results:
        df = pd.DataFrame(results)
        output_file = OUTPUT_DIR / f"scored_signals_{datetime.now().strftime('%Y%m%d')}.csv"
        df.to_csv(output_file, index=False)
        
        print(f"\n{'='*70}")
        print("SCORING SUMMARY")
        print(f"{'='*70}")
        print(f"Total signals generated: {len(results)}")
        print(f"Processed: {processed}, Errors: {errors}")
        print(f"\nFinal Signal Distribution:")
        print(df['final_signal'].value_counts())
        print(f"\nOutput saved to: {output_file}")
        print(f"{'='*70}")
    else:
        print("No results generated!")


if __name__ == "__main__":
    main()
