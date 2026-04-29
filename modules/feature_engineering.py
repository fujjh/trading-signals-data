#!/usr/bin/env python3
"""
================================================================================
ML FEATURE ENGINEERING MODULE
================================================================================

Advanced feature engineering for machine learning-based signal enhancement.
Generates price action, momentum, volatility, and regime detection features.

Daily data only - optimized for swing trading timeframes.

Author: SignalsAlpha
Version: 1.0
Date: 2026-04-28
================================================================================
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


@dataclass
class MLFeatures:
    """Container for ML-engineered features"""
    # Price Action
    returns_5d: float
    returns_20d: float
    returns_60d: float
    price_vs_high_20d: float
    price_vs_low_20d: float
    
    # Volatility
    volatility_20d: float
    volatility_60d: float
    atr_ratio: float
    bb_squeeze: bool
    
    # Momentum
    rsi_slope_5d: float
    macd_histogram_slope: float
    momentum_10d: float
    momentum_30d: float
    
    # Trend
    adx_trend_strength: float
    trend_direction: int  # 1=up, -1=down, 0=neutral
    higher_highs_count: int
    lower_lows_count: int
    
    # Volume
    volume_ratio_20d: float
    volume_trend: float
    obv_slope: float
    money_flow_trend: float
    
    # Regime
    is_trending: bool
    is_ranging: bool
    volatility_regime: str  # 'high', 'normal', 'low'
    
    # Pattern
    support_touch_count: int
    resistance_touch_count: int
    breakout_detected: bool
    breakdown_detected: bool
    
    # Composite
    composite_momentum: float
    composite_strength: float


class FeatureEngineer:
    """Generate ML features from OHLCV and technical data"""
    
    def __init__(self):
        """Initialize feature engineer"""
        self.feature_columns = []
    
    def calculate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all ML features from price data
        
        Args:
            df: DataFrame with OHLCV columns (open, high, low, close, volume)
            
        Returns:
            DataFrame with added ML feature columns
        """
        df = df.copy()
        
        # Ensure required columns
        required = ['open', 'high', 'low', 'close', 'volume']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Price Action Features
        df = self._add_price_action_features(df)
        
        # Volatility Features
        df = self._add_volatility_features(df)
        
        # Momentum Features
        df = self._add_momentum_features(df)
        
        # Trend Features
        df = self._add_trend_features(df)
        
        # Volume Features
        df = self._add_volume_features(df)
        
        # Regime Features
        df = self._add_regime_features(df)
        
        # Pattern Features
        df = self._add_pattern_features(df)
        
        # Composite Features
        df = self._add_composite_features(df)
        
        return df
    
    def _add_price_action_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price action based features"""
        close = df['close']
        
        # Returns over different periods
        df['returns_5d'] = close.pct_change(5)
        df['returns_20d'] = close.pct_change(20)
        df['returns_60d'] = close.pct_change(60)
        
        # Price position within recent range
        high_20d = df['high'].rolling(20).max()
        low_20d = df['low'].rolling(20).min()
        df['price_vs_high_20d'] = (close - high_20d) / high_20d
        df['price_vs_low_20d'] = (close - low_20d) / low_20d
        
        # Price distance from moving averages
        df['dist_from_sma20'] = (close - close.rolling(20).mean()) / close
        df['dist_from_sma50'] = (close - close.rolling(50).mean()) / close
        
        return df
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility based features"""
        returns = df['close'].pct_change()
        
        # Rolling volatility
        df['volatility_20d'] = returns.rolling(20).std() * np.sqrt(252)
        df['volatility_60d'] = returns.rolling(60).std() * np.sqrt(252)
        
        # Volatility trend
        df['volatility_trend'] = df['volatility_20d'] / df['volatility_60d']
        
        # ATR ratio
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift(1))
        tr3 = abs(df['low'] - df['close'].shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_14 = true_range.rolling(14).mean()
        df['atr_ratio'] = atr_14 / df['close']
        
        # Bollinger Band squeeze detection
        bb_width = (df['close'].rolling(20).std() * 2) / df['close'].rolling(20).mean()
        df['bb_squeeze'] = bb_width < bb_width.rolling(120).quantile(0.2)
        
        return df
    
    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum based features"""
        close = df['close']
        
        # RSI slope (if RSI exists, otherwise calculate)
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        df['rsi_slope_5d'] = rsi.diff(5)
        
        # MACD histogram slope (if available)
        ema_12 = close.ewm(span=12).mean()
        ema_26 = close.ewm(span=26).mean()
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9).mean()
        histogram = macd - signal
        df['macd_histogram_slope'] = histogram.diff(3)
        
        # Momentum over periods
        df['momentum_10d'] = close.pct_change(10)
        df['momentum_30d'] = close.pct_change(30)
        
        # Rate of change
        df['roc_10d'] = (close - close.shift(10)) / close.shift(10)
        df['roc_30d'] = (close - close.shift(30)) / close.shift(30)
        
        return df
    
    def _add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend based features"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Higher highs / lower lows detection
        df['higher_high'] = (high > high.shift(1)) & (high.shift(1) > high.shift(2))
        df['lower_low'] = (low < low.shift(1)) & (low.shift(1) < low.shift(2))
        
        df['higher_highs_count'] = df['higher_high'].rolling(20).sum()
        df['lower_lows_count'] = df['lower_low'].rolling(20).sum()
        
        # Trend direction (SMA alignment)
        sma_20 = close.rolling(20).mean()
        sma_50 = close.rolling(50).mean()
        sma_200 = close.rolling(200).mean()
        
        df['trend_direction'] = np.where(
            (close > sma_20) & (sma_20 > sma_50) & (sma_50 > sma_200), 1,
            np.where((close < sma_20) & (sma_20 < sma_50) & (sma_50 < sma_200), -1, 0)
        )
        
        # ADX approximation for trend strength
        tr = pd.concat([
            high - low,
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        ], axis=1).max(axis=1)
        
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        atr_14 = tr.rolling(14).mean()
        plus_di = 100 * plus_dm.rolling(14).mean() / atr_14
        minus_di = 100 * minus_dm.rolling(14).mean() / atr_14
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        df['adx_trend_strength'] = dx.rolling(14).mean()
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume based features"""
        volume = df['volume']
        close = df['close']
        
        # Volume ratios
        df['volume_ratio_20d'] = volume / volume.rolling(20).mean()
        df['volume_ratio_5d'] = volume / volume.rolling(5).mean()
        
        # Volume trend
        df['volume_trend'] = volume.rolling(10).apply(
            lambda x: stats.linregress(range(len(x)), x)[0] if len(x) > 1 else 0,
            raw=True
        )
        
        # OBV slope
        obv = (np.sign(close.diff()) * volume).cumsum()
        df['obv_slope'] = obv.diff(5)
        
        # Money flow
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * volume
        df['money_flow_trend'] = money_flow.diff(10)
        
        # Volume-price divergence
        price_change = close.pct_change(10)
        volume_change = volume.pct_change(10)
        df['volume_price_divergence'] = np.where(
            (price_change > 0) & (volume_change < 0), -1,  # Rising price, falling volume (bearish)
            np.where((price_change < 0) & (volume_change > 0), 1, 0)  # Falling price, rising volume (bullish)
        )
        
        return df
    
    def _add_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market regime detection features"""
        volatility = df['volatility_20d']
        adx = df['adx_trend_strength']
        
        # Trending vs ranging
        df['is_trending'] = adx > 25
        df['is_ranging'] = (adx < 20) & (volatility < volatility.quantile(0.5))
        
        # Volatility regime
        vol_percentile = volatility.rank(pct=True)
        df['volatility_regime'] = pd.cut(
            vol_percentile,
            bins=[0, 0.33, 0.67, 1.0],
            labels=['low', 'normal', 'high']
        )
        
        return df
    
    def _add_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add pattern detection features"""
        close = df['close']
        high = df['high']
        low = df['low']
        
        # Support/Resistance touch count (simplified)
        # Uses recent highs/lows as S/R levels
        recent_highs = high.rolling(20).max()
        recent_lows = low.rolling(20).min()
        
        # Count touches within 1% of level
        df['support_touch_count'] = ((close - recent_lows) / close < 0.01).rolling(20).sum()
        df['resistance_touch_count'] = ((recent_highs - close) / close < 0.01).rolling(20).sum()
        
        # Breakout detection
        df['breakout_detected'] = close > recent_highs.shift(1) * 1.02
        df['breakdown_detected'] = close < recent_lows.shift(1) * 0.98
        
        return df
    
    def _add_composite_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add composite multi-factor features"""
        # Composite momentum (combines price, volume, and RSI momentum)
        df['composite_momentum'] = (
            df['momentum_10d'].fillna(0) * 0.4 +
            df['volume_ratio_20d'].fillna(0).apply(lambda x: (x - 1) * 0.1) +
            df['rsi_slope_5d'].fillna(0).apply(lambda x: x / 100 * 0.3) +
            df['macd_histogram_slope'].fillna(0).apply(lambda x: x * 0.2)
        )
        
        # Composite strength (trend + momentum alignment)
        trend_score = df['trend_direction']
        momentum_score = np.where(df['momentum_10d'] > 0, 1, -1)
        volume_score = np.where(df['volume_ratio_20d'] > 1.2, 1, 0)
        
        df['composite_strength'] = (
            trend_score * 0.4 +
            momentum_score * 0.4 +
            volume_score * 0.2
        )
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """Return list of all engineered feature names"""
        return [
            'returns_5d', 'returns_20d', 'returns_60d',
            'price_vs_high_20d', 'price_vs_low_20d',
            'dist_from_sma20', 'dist_from_sma50',
            'volatility_20d', 'volatility_60d', 'volatility_trend', 'atr_ratio', 'bb_squeeze',
            'rsi_slope_5d', 'macd_histogram_slope', 'momentum_10d', 'momentum_30d',
            'roc_10d', 'roc_30d',
            'adx_trend_strength', 'trend_direction', 'higher_highs_count', 'lower_lows_count',
            'volume_ratio_20d', 'volume_ratio_5d', 'volume_trend', 'obv_slope', 'money_flow_trend',
            'volume_price_divergence',
            'is_trending', 'is_ranging', 'volatility_regime',
            'support_touch_count', 'resistance_touch_count', 'breakout_detected', 'breakdown_detected',
            'composite_momentum', 'composite_strength'
        ]
    
    def validate_features(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Validate feature quality and coverage
        
        Returns:
            Dictionary with validation results
        """
        feature_names = self.get_feature_names()
        results = {
            'total_features': len(feature_names),
            'present': [],
            'missing': [],
            'null_counts': {},
            'coverage': {}
        }
        
        for feature in feature_names:
            if feature in df.columns:
                results['present'].append(feature)
                null_count = df[feature].isnull().sum()
                results['null_counts'][feature] = null_count
                results['coverage'][feature] = (len(df) - null_count) / len(df)
            else:
                results['missing'].append(feature)
        
        return results


def add_ml_features_to_step3(step3_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function to add ML features to Step 3 output
    
    Args:
        step3_df: DataFrame from Step 3 technical analysis
        
    Returns:
        DataFrame with additional ML features
    """
    engineer = FeatureEngineer()
    return engineer.calculate_all_features(step3_df)


if __name__ == "__main__":
    # Test with sample data
    import numpy as np
    
    np.random.seed(42)
    n_days = 100
    
    # Create sample OHLCV data
    base_price = 100
    prices = base_price + np.cumsum(np.random.randn(n_days) * 2)
    
    sample_df = pd.DataFrame({
        'date': pd.date_range('2026-01-01', periods=n_days),
        'open': prices + np.random.randn(n_days) * 0.5,
        'high': prices + abs(np.random.randn(n_days)) * 2,
        'low': prices - abs(np.random.randn(n_days)) * 2,
        'close': prices + np.random.randn(n_days) * 0.5,
        'volume': np.random.randint(1000000, 10000000, n_days)
    })
    
    # Calculate features
    engineer = FeatureEngineer()
    result_df = engineer.calculate_all_features(sample_df)
    
    print("ML Features Generated:")
    print("=" * 70)
    
    # Print feature statistics
    feature_names = engineer.get_feature_names()
    for feature in feature_names[:10]:  # Show first 10
        if feature in result_df.columns:
            stats = result_df[feature].describe()
            print(f"\n{feature}:")
            print(f"  Mean: {stats['mean']:.4f}")
            print(f"  Std:  {stats['std']:.4f}")
            print(f"  Null: {result_df[feature].isnull().sum()}")
    
    print(f"\n\nTotal features: {len(feature_names)}")
    print(f"Final DataFrame shape: {result_df.shape}")
