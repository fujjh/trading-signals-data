#!/usr/bin/env python3
"""
================================================================================
STEP 9: SPY XGBoost Signal Predictor
================================================================================

XGBoost classifier to predict buy/sell signals based on technical indicators.
Uses gradient boosting to capture non-linear relationships in market data.

Author: SignalsAlpha
Version: 1.0 (XGBoost Edition)
Date: 2026-04-29

================================================================================
APPROACH:
================================================================================

1. Generate labels from forward-looking returns (buy if next N days positive)
2. Feature engineering: technical indicators + lagged features
3. Train XGBoost classifier with hyperparameter tuning
4. Compare against GA-optimized strategy
================================================================================
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple

try:
    import xgboost as xgb
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.preprocessing import StandardScaler
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not installed. Install with: pip install xgboost scikit-learn")

# Configuration
TICKER = "SPY"
TECHNICAL_DIR = Path("data/technical_analysis") / TICKER
OUTPUT_DIR = Path("data/optimizer")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Label generation parameters
FORWARD_LOOK_DAYS = 5  # Predict if price will be higher in N days
RETURN_THRESHOLD = 0.02  # 2% minimum expected return


def load_spy_data():
    """Load SPY technical data."""
    tech_file = TECHNICAL_DIR / f"{TICKER}_1d_technical.csv"
    if not tech_file.exists():
        print(f"ERROR: Technical file not found: {tech_file}")
        return None
    
    df = pd.read_csv(tech_file)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create feature set for XGBoost."""
    features = pd.DataFrame(index=df.index)
    
    # Price-based features
    features['close'] = df['close']
    features['returns_1d'] = df['close'].pct_change()
    features['returns_5d'] = df['close'].pct_change(5)
    features['returns_20d'] = df['close'].pct_change(20)
    
    # Technical indicators
    if 'rsi' in df.columns:
        features['rsi'] = df['rsi']
        features['rsi_lag1'] = df['rsi'].shift(1)
        features['rsi_lag5'] = df['rsi'].shift(5)
    
    if 'macd' in df.columns:
        features['macd'] = df['macd']
        features['macd_signal'] = df['macd_signal']
        features['macd_hist'] = df['macd_hist']
    
    if 'sma_20' in df.columns:
        features['price_vs_sma20'] = df['close'] / df['sma_20'] - 1
    if 'sma_50' in df.columns:
        features['price_vs_sma50'] = df['close'] / df['sma_50'] - 1
    if 'sma_200' in df.columns:
        features['price_vs_sma200'] = df['close'] / df['sma_200'] - 1
    
    if 'hma_13' in df.columns:
        features['price_vs_hma'] = df['close'] / df['hma_13'] - 1
    
    if 'bb_upper' in df.columns:
        features['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    if 'atr' in df.columns:
        features['atr_ratio'] = df['atr'] / df['close']
    
    # 52-week features
    if 'pct_from_52w_high' in df.columns:
        features['pct_from_52w_high'] = df['pct_from_52w_high']
    if 'pct_from_52w_low' in df.columns:
        features['pct_from_52w_low'] = df['pct_from_52w_low']
    
    # Volume features
    if 'volume' in df.columns:
        features['volume_ma20'] = df['volume'].rolling(20).mean()
        features['volume_ratio'] = df['volume'] / features['volume_ma20']
    
    # Volatility
    features['volatility_20d'] = features['returns_1d'].rolling(20).std() * np.sqrt(252)
    
    # Trend features
    features['higher_highs'] = (df['high'] > df['high'].shift(1).rolling(5).max()).astype(int)
    features['lower_lows'] = (df['low'] < df['low'].shift(1).rolling(5).min()).astype(int)
    
    return features


def create_labels(df: pd.DataFrame, forward_days: int = 5, threshold: float = 0.02) -> pd.Series:
    """
    Create labels for classification.
    1 = BUY (forward return > threshold)
    0 = HOLD/SELL (forward return < threshold)
    """
    future_returns = df['close'].shift(-forward_days) / df['close'] - 1
    labels = (future_returns > threshold).astype(int)
    return labels


def run_xgboost_training():
    """Main XGBoost training and evaluation."""
    print("=" * 70)
    print("STEP 9: SPY XGBoost Signal Predictor")
    print("=" * 70)
    
    if not XGBOOST_AVAILABLE:
        print("ERROR: XGBoost not installed. Install with: pip install xgboost scikit-learn")
        return
    
    df = load_spy_data()
    if df is None:
        return
    
    print(f"Loaded {len(df)} rows")
    print(f"Forward look: {FORWARD_LOOK_DAYS} days")
    print(f"Return threshold: {RETURN_THRESHOLD*100:.1f}%")
    print("=" * 70)
    
    # Create features and labels
    print("\nCreating features...")
    X = create_features(df)
    y = create_labels(df, FORWARD_LOOK_DAYS, RETURN_THRESHOLD)
    
    # Remove rows with NaN
    valid_idx = X.notna().all(axis=1) & y.notna()
    X = X[valid_idx]
    y = y[valid_idx]
    
    print(f"Feature set: {X.shape[1]} features, {len(X)} samples")
    print(f"Label distribution: {y.value_counts().to_dict()}")
    
    # Time-based split (no shuffle - respect temporal order)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"\nTrain: {len(X_train)} samples")
    print(f"Test: {len(X_test)} samples")
    print(f"Train period: {df.iloc[valid_idx].iloc[:split_idx]['date'].iloc[0].strftime('%Y-%m-%d')} to {df.iloc[valid_idx].iloc[:split_idx]['date'].iloc[-1].strftime('%Y-%m-%d')}")
    print(f"Test period: {df.iloc[valid_idx].iloc[split_idx:]['date'].iloc[0].strftime('%Y-%m-%d')} to {df.iloc[valid_idx].iloc[split_idx:]['date'].iloc[-1].strftime('%Y-%m-%d')}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train XGBoost
    print("\n" + "=" * 70)
    print("Training XGBoost...")
    print("=" * 70)
    
    # Base model
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )
    
    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_test_scaled, y_test)],
        verbose=False
    )
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"\nTest Set Performance:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    
    # Feature importance
    print(f"\nTop 10 Feature Importances:")
    importance = model.feature_importances_
    feature_names = X.columns
    indices = np.argsort(importance)[::-1][:10]
    for i in indices:
        print(f"  {feature_names[i]}: {importance[i]:.4f}")
    
    # Strategy simulation
    print("\n" + "=" * 70)
    print("Strategy Backtest on Test Set")
    print("=" * 70)
    
    test_df = df.iloc[valid_idx].iloc[split_idx:].copy()
    test_df['signal'] = y_pred
    test_df['signal_proba'] = y_pred_proba
    test_df['returns'] = test_df['close'].pct_change()
    
    # Calculate strategy returns (enter when signal=1, exit when signal=0)
    position = np.zeros(len(test_df))
    for i in range(1, len(test_df)):
        if test_df['signal'].iloc[i-1] == 1:
            position[i] = 1.0
        else:
            position[i] = 0.0
    
    test_df['position'] = position
    test_df['strategy_returns'] = test_df['position'].shift(1) * test_df['returns']
    test_df['strategy_returns'] = test_df['strategy_returns'].fillna(0)
    
    # Metrics
    total_return = test_df['strategy_returns'].sum()
    volatility = test_df['strategy_returns'].std() * np.sqrt(252)
    sharpe = (test_df['strategy_returns'].mean() * 252) / volatility if volatility > 0 else 0
    
    cumulative = (1 + test_df['strategy_returns']).cumprod()
    peak = cumulative.expanding().max()
    drawdown = (peak - cumulative) / peak
    max_dd = drawdown.max()
    
    trades = (test_df['position'].diff().abs() > 0.01).sum()
    win_rate = (test_df['strategy_returns'] > 0).sum() / (test_df['strategy_returns'] != 0).sum()
    
    print(f"Total Return: {total_return*100:.2f}%")
    print(f"Sharpe: {sharpe:.2f}")
    print(f"Max DD: {max_dd*100:.1f}%")
    print(f"Win Rate: {win_rate*100:.1f}%")
    print(f"Trades: {int(trades)}")
    
    # Buy and hold comparison
    bh_return = (test_df['close'].iloc[-1] / test_df['close'].iloc[0] - 1)
    print(f"\nBuy & Hold Return: {bh_return*100:.2f}%")
    print(f"Outperformance: {(total_return - bh_return)*100:.2f}%")
    
    # Save model
    output_file = OUTPUT_DIR / f"spy_xgboost_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    model.save_model(str(output_file).replace('.json', '.xgb'))
    
    # Save config
    config_file = OUTPUT_DIR / f"spy_xgboost_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(config_file, 'w') as f:
        json.dump({
            'ticker': TICKER,
            'forward_look_days': FORWARD_LOOK_DAYS,
            'return_threshold': RETURN_THRESHOLD,
            'test_accuracy': float(accuracy),
            'test_precision': float(precision),
            'test_recall': float(recall),
            'test_f1': float(f1),
            'strategy_return': float(total_return),
            'strategy_sharpe': float(sharpe),
            'strategy_max_dd': float(max_dd),
            'strategy_win_rate': float(win_rate),
            'strategy_trades': int(trades),
            'buy_hold_return': float(bh_return),
            'feature_importance': {feature_names[i]: float(importance[i]) for i in indices},
            'generated_at': datetime.now().isoformat()
        }, f, indent=2)
    
    print(f"\nSaved to: {config_file}")
    print("=" * 70)


if __name__ == "__main__":
    run_xgboost_training()
