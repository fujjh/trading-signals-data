#!/usr/bin/env python3
"""
Signal Generator v3.0 with Support/Resistance Levels
Incorporates 3 approaches:
  1. Pivot Analysis (local highs/lows)
  2. Volume Profile (high-volume price levels)
  3. Fibonacci Retracements

Features:
- 1-year lookback with recency weighting
- Minimum 2 touches required
- 2% cluster tolerance
- Peaks and troughs identification
- Up to 10 resistance levels above, 10 support below
"""

import os
import json
import glob
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class SupportResistanceLevel:
    price: float
    level_type: str  # 'pivot', 'volume', 'fibonacci'
    touches: int
    strength_score: float  # 0-100
    is_resistance: bool  # True if above current price
    distance_pct: float  # Distance from current price

class SupportResistanceCalculator:
    """Calculate support and resistance levels using multiple methods"""
    
    def __init__(self, lookback_days: int = 252, min_touches: int = 2, 
                 cluster_tolerance: float = 0.02):
        self.lookback_days = lookback_days  # ~1 year
        self.min_touches = min_touches
        self.cluster_tolerance = cluster_tolerance  # 2%
    
    def calculate_all_levels(self, df: pd.DataFrame, current_price: float) -> Tuple[
        List[SupportResistanceLevel], List[SupportResistanceLevel], Dict]:
        """
        Calculate support and resistance using all 3 methods
        Returns: (resistance_levels, support_levels, metadata)
        """
        
        # Limit to lookback period
        if len(df) > self.lookback_days:
            df = df.tail(self.lookback_days).copy()
        
        # Method 1: Pivot Analysis
        pivot_levels = self._calculate_pivot_levels(df, current_price)
        
        # Method 2: Volume Profile
        volume_levels = self._calculate_volume_profile(df, current_price)
        
        # Method 3: Fibonacci Retracements
        fibonacci_levels = self._calculate_fibonacci_levels(df, current_price)
        
        # Combine and cluster all levels
        all_levels = pivot_levels + volume_levels + fibonacci_levels
        clustered = self._cluster_levels(all_levels, current_price)
        
        # Split into resistance and support
        resistance = [l for l in clustered if l.is_resistance]
        support = [l for l in clustered if not l.is_resistance]
        
        # Sort by strength and distance
        resistance.sort(key=lambda x: (x.strength_score, -x.distance_pct), reverse=True)
        support.sort(key=lambda x: (x.strength_score, x.distance_pct), reverse=True)
        
        # Take top 10 each
        resistance = resistance[:10]
        support = support[:10]
        
        # Peaks and troughs from pivot analysis
        peaks, troughs = self._identify_peaks_troughs(df)
        
        metadata = {
            'pivot_levels_found': len(pivot_levels),
            'volume_levels_found': len(volume_levels),
            'fibonacci_levels_found': len(fibonacci_levels),
            'total_levels_before_clustering': len(all_levels),
            'total_levels_after_clustering': len(clustered),
            'peaks': peaks,
            'troughs': troughs,
            'lookback_days': len(df),
            'current_price': current_price
        }
        
        return resistance, support, metadata
    
    def _identify_peaks_troughs(self, df: pd.DataFrame, window: int = 5) -> Tuple[List, List]:
        """Identify peaks (local maxima) and troughs (local minima)"""
        
        peaks = []
        troughs = []
        
        highs = df['High'].values
        lows = df['Low'].values
        dates = df['Date'].values if 'Date' in df.columns else df.index.strftime('%Y-%m-%d').values
        
        for i in range(window, len(df) - window):
            # Check if local maximum
            if all(highs[i] >= highs[i-j] for j in range(1, window+1)) and \
               all(highs[i] >= highs[i+j] for j in range(1, window+1)):
                peaks.append({
                    'date': dates[i],
                    'price': round(highs[i], 2),
                    'index': i
                })
            
            # Check if local minimum
            if all(lows[i] <= lows[i-j] for j in range(1, window+1)) and \
               all(lows[i] <= lows[i+j] for j in range(1, window+1)):
                troughs.append({
                    'date': dates[i],
                    'price': round(lows[i], 2),
                    'index': i
                })
        
        return peaks, troughs
    
    def _calculate_pivot_levels(self, df: pd.DataFrame, current_price: float) -> List[SupportResistanceLevel]:
        """Calculate support/resistance from pivot highs and lows"""
        
        levels = []
        
        # Find pivot highs (resistance candidates)
        pivot_highs = []
        for i in range(5, len(df) - 5):
            if df['High'].iloc[i] == df['High'].iloc[i-5:i+6].max():
                pivot_highs.append({
                    'price': df['High'].iloc[i],
                    'date': df.index[i] if hasattr(df.index, '__getitem__') else i,
                    'days_ago': len(df) - i
                })
        
        # Find pivot lows (support candidates)
        pivot_lows = []
        for i in range(5, len(df) - 5):
            if df['Low'].iloc[i] == df['Low'].iloc[i-5:i+6].min():
                pivot_lows.append({
                    'price': df['Low'].iloc[i],
                    'date': df.index[i] if hasattr(df.index, '__getitem__') else i,
                    'days_ago': len(df) - i
                })
        
        # Count touches for each pivot level
        for pivot in pivot_highs:
            touches = self._count_touches(df, pivot['price'], is_high=True)
            if touches >= self.min_touches:
                strength = self._calculate_strength(touches, pivot['days_ago'], 'pivot')
                is_resistance = pivot['price'] > current_price
                distance_pct = abs(pivot['price'] - current_price) / current_price * 100
                
                levels.append(SupportResistanceLevel(
                    price=round(pivot['price'], 2),
                    level_type='pivot',
                    touches=touches,
                    strength_score=strength,
                    is_resistance=is_resistance,
                    distance_pct=distance_pct
                ))
        
        for pivot in pivot_lows:
            touches = self._count_touches(df, pivot['price'], is_high=False)
            if touches >= self.min_touches:
                strength = self._calculate_strength(touches, pivot['days_ago'], 'pivot')
                is_resistance = pivot['price'] > current_price
                distance_pct = abs(pivot['price'] - current_price) / current_price * 100
                
                levels.append(SupportResistanceLevel(
                    price=round(pivot['price'], 2),
                    level_type='pivot',
                    touches=touches,
                    strength_score=strength,
                    is_resistance=is_resistance,
                    distance_pct=distance_pct
                ))
        
        return levels
    
    def _calculate_volume_profile(self, df: pd.DataFrame, current_price: float) -> List[SupportResistanceLevel]:
        """Calculate support/resistance from volume profile (price levels with high volume)"""
        
        levels = []
        
        # Create price bins (use closing prices)
        price_min = df['Low'].min()
        price_max = df['High'].max()
        num_bins = 50
        bin_size = (price_max - price_min) / num_bins
        
        # Calculate volume at each price level
        volume_profile = defaultdict(float)
        for idx, row in df.iterrows():
            # Distribute volume across the day's range (weighted toward VWAP)
            typical_price = (row['High'] + row['Low'] + row['Close']) / 3
            bin_idx = int((typical_price - price_min) / bin_size)
            bin_idx = max(0, min(bin_idx, num_bins - 1))
            volume_profile[bin_idx] += row['Volume']
        
        # Find peaks in volume profile
        sorted_bins = sorted(volume_profile.items(), key=lambda x: x[1], reverse=True)
        top_bins = sorted_bins[:10]  # Top 10 volume levels
        
        for bin_idx, volume in top_bins:
            price_level = price_min + (bin_idx + 0.5) * bin_size
            
            # Count touches (days where price traded near this level)
            touches = 0
            for idx, row in df.iterrows():
                if price_level * 0.98 <= row['Close'] <= price_level * 1.02:
                    touches += 1
            
            if touches >= self.min_touches:
                avg_volume = df['Volume'].mean()
                if avg_volume > 0:
                    strength = min(100, (volume / avg_volume) * 10)
                else:
                    strength = 50  # Default if no volume data
                is_resistance = price_level > current_price
                distance_pct = abs(price_level - current_price) / current_price * 100
                
                levels.append(SupportResistanceLevel(
                    price=round(price_level, 2),
                    level_type='volume',
                    touches=touches,
                    strength_score=strength,
                    is_resistance=is_resistance,
                    distance_pct=distance_pct
                ))
        
        return levels
    
    def _calculate_fibonacci_levels(self, df: pd.DataFrame, current_price: float) -> List[SupportResistanceLevel]:
        """Calculate Fibonacci retracement levels from swing high/low"""
        
        levels = []
        
        # Find swing high and low in the lookback period
        swing_high = df['High'].max()
        swing_low = df['Low'].min()
        
        if swing_high == swing_low:
            return levels
        
        # Fibonacci ratios
        fib_ratios = [0.236, 0.382, 0.5, 0.618, 0.786]
        
        for ratio in fib_ratios:
            fib_level = swing_high - (swing_high - swing_low) * ratio
            
            # Count touches
            touches = 0
            for idx, row in df.iterrows():
                if abs(row['Close'] - fib_level) / fib_level < self.cluster_tolerance:
                    touches += 1
            
            # Fibonacci levels are psychological - give them base strength
            strength = 60 if touches >= self.min_touches else 40
            is_resistance = fib_level > current_price
            distance_pct = abs(fib_level - current_price) / current_price * 100
            
            levels.append(SupportResistanceLevel(
                price=round(fib_level, 2),
                level_type='fibonacci',
                touches=max(touches, 1),  # At least 1 for fib levels
                strength_score=strength,
                is_resistance=is_resistance,
                distance_pct=distance_pct
            ))
        
        return levels
    
    def _count_touches(self, df: pd.DataFrame, price_level: float, is_high: bool) -> int:
        """Count how many times price touched this level"""
        touches = 0
        tolerance = price_level * self.cluster_tolerance
        
        for idx, row in df.iterrows():
            if is_high:
                # For resistance - count when high approached level
                if abs(row['High'] - price_level) <= tolerance:
                    touches += 1
            else:
                # For support - count when low approached level
                if abs(row['Low'] - price_level) <= tolerance:
                    touches += 1
        
        return touches
    
    def _calculate_strength(self, touches: int, days_ago: int, level_type: str) -> float:
        """Calculate strength score (0-100) based on touches and recency"""
        
        # Base score from touches (max 50 points)
        touch_score = min(50, touches * 15)
        
        # Recency score (max 50 points) - more recent = higher
        # Exponential decay - recent touches matter more
        recency_score = 50 * np.exp(-days_ago / 60)  # 60-day half-life
        
        # Type bonus
        type_bonus = {'pivot': 10, 'volume': 5, 'fibonacci': 0}.get(level_type, 0)
        
        total = touch_score + recency_score + type_bonus
        return min(100, max(0, total))
    
    def _cluster_levels(self, levels: List[SupportResistanceLevel], current_price: float) -> List[SupportResistanceLevel]:
        """Cluster nearby levels and merge them"""
        
        if not levels:
            return []
        
        # Sort by price
        sorted_levels = sorted(levels, key=lambda x: x.price)
        
        clustered = []
        current_cluster = [sorted_levels[0]]
        
        for level in sorted_levels[1:]:
            # Check if this level is within tolerance of cluster average
            cluster_avg = sum(l.price for l in current_cluster) / len(current_cluster)
            
            if abs(level.price - cluster_avg) / cluster_avg <= self.cluster_tolerance:
                current_cluster.append(level)
            else:
                # Merge current cluster
                merged = self._merge_cluster(current_cluster)
                clustered.append(merged)
                current_cluster = [level]
        
        # Don't forget last cluster
        if current_cluster:
            merged = self._merge_cluster(current_cluster)
            clustered.append(merged)
        
        return clustered
    
    def _merge_cluster(self, cluster: List[SupportResistanceLevel]) -> SupportResistanceLevel:
        """Merge a cluster of levels into a single level"""
        
        # Weighted average by strength
        total_weight = sum(l.strength_score for l in cluster)
        avg_price = sum(l.price * l.strength_score for l in cluster) / total_weight
        
        # Sum touches
        total_touches = sum(l.touches for l in cluster)
        
        # Max strength
        max_strength = max(l.strength_score for l in cluster)
        
        # Primary type (the strongest)
        primary_type = max(cluster, key=lambda x: x.strength_score).level_type
        
        return SupportResistanceLevel(
            price=round(avg_price, 2),
            level_type=f"{primary_type}_cluster",
            touches=total_touches,
            strength_score=max_strength,
            is_resistance=cluster[0].is_resistance,
            distance_pct=cluster[0].distance_pct
        )


# Example usage for testing
if __name__ == "__main__":
    print("Support/Resistance Calculator v3.0")
    print("Testing with sample data...")
    
    # Test data (AAPL-like)
    dates = pd.date_range(end=datetime.now(), periods=252, freq='D')
    np.random.seed(42)
    prices = 150 + np.cumsum(np.random.randn(252) * 2)
    
    df = pd.DataFrame({
        'Date': dates,
        'Open': prices - 1,
        'High': prices + 2,
        'Low': prices - 2,
        'Close': prices,
        'Volume': np.random.randint(1000000, 50000000, 252)
    })
    
    calculator = SupportResistanceCalculator(lookback_days=252, min_touches=2)
    resistance, support, metadata = calculator.calculate_all_levels(df, current_price=prices[-1])
    
    print(f"\nResistance Levels (above ${prices[-1]:.2f}):")
    for r in resistance[:5]:
        print(f"  ${r.price:.2f} | {r.level_type} | {r.touches} touches | {r.strength_score:.1f} strength")
    
    print(f"\nSupport Levels (below ${prices[-2]:.2f}):")
    for s in support[:5]:
        print(f"  ${s.price:.2f} | {s.level_type} | {s.touches} touches | {s.strength_score:.1f} strength")
    
    print(f"\nMetadata:")
    print(f"  Peaks found: {len(metadata['peaks'])}")
    print(f"  Troughs found: {len(metadata['troughs'])}")
    print(f"  Total levels: {metadata['total_levels_after_clustering']}")
