#!/usr/bin/env python3
"""
================================================================================
STEP 9: SPY LSTM Signal Predictor
================================================================================

LSTM (Long Short-Term Memory) neural network for time series prediction.
Captures temporal patterns and sequential dependencies in market data.

Author: SignalsAlpha
Version: 1.0 (LSTM Edition)
Date: 2026-04-29

================================================================================
APPROACH:
================================================================================

1. Create sequences of technical indicators (lookback window)
2. LSTM layers to capture temporal patterns
3. Dense layers for binary classification (buy/sell)
4. Compare against GA and XGBoost strategies
================================================================================
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Tuple, List

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not installed. Install with: pip install tensorflow")

# Configuration
TICKER = "SPY"
TECHNICAL_DIR = Path("data/technical_analysis") / TICKER
OUTPUT_DIR = Path("data/optimizer")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# LSTM parameters
LOOKBACK_WINDOW = 20  # Days to look back
FORWARD_LOOK_DAYS = 5
RETURN_THRESHOLD = 0.02
EPOCHS = 50
BATCH_SIZE = 32


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
    """Create feature set for LSTM."""
    features = pd.DataFrame(index=df.index)
    
    # Price features
    features['close_norm'] = df['close'] / df['close'].rolling(252).mean()
    features['returns_1d'] = df['close'].pct_change()
    features['returns_5d'] = df['close'].pct_change(5)
    
    # Technical indicators
    if 'rsi' in df.columns:
        features['rsi'] = df['rsi'] / 100.0  # Normalize
    
    if 'macd' in df.columns:
        features['macd'] = df['macd'] / df['close']  # Normalize by price
        features['macd_hist'] = df['macd_hist'] / df['close']
    
    if 'sma_20' in df.columns and 'sma_50' in df.columns:
        features['sma20_50_ratio'] = df['sma_20'] / df['sma_50']
    
    if 'sma_200' in df.columns:
        features['price_vs_sma200'] = df['close'] / df['sma_200'] - 1
    
    if 'hma_13' in df.columns:
        features['price_vs_hma'] = df['close'] / df['hma_13'] - 1
    
    if 'bb_upper' in df.columns:
        features['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        features['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    if 'atr' in df.columns:
        features['atr_ratio'] = df['atr'] / df['close']
    
    # 52-week features
    if 'pct_from_52w_high' in df.columns:
        features['pct_from_52w_high'] = df['pct_from_52w_high'] / 100.0
    if 'pct_from_52w_low' in df.columns:
        features['pct_from_52w_low'] = df['pct_from_52w_low'] / 100.0
    
    # Volume
    if 'volume' in df.columns:
        features['volume_norm'] = df['volume'] / df['volume'].rolling(20).mean()
    
    return features


def create_sequences(X: np.ndarray, y: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
    """Create sequences for LSTM input."""
    X_seq, y_seq = [], []
    for i in range(lookback, len(X)):
        X_seq.append(X[i-lookback:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)


def create_labels(df: pd.DataFrame) -> np.ndarray:
    """Create labels based on forward returns."""
    future_returns = df['close'].shift(-FORWARD_LOOK_DAYS) / df['close'] - 1
    labels = (future_returns > RETURN_THRESHOLD).astype(int).values
    return labels


def run_lstm_training():
    """Main LSTM training and evaluation."""
    print("=" * 70)
    print("STEP 9: SPY LSTM Signal Predictor")
    print("=" * 70)
    
    if not TF_AVAILABLE:
        print("ERROR: TensorFlow not installed. Install with: pip install tensorflow")
        return
    
    df = load_spy_data()
    if df is None:
        return
    
    print(f"Loaded {len(df)} rows")
    print(f"Lookback window: {LOOKBACK_WINDOW} days")
    print(f"Forward look: {FORWARD_LOOK_DAYS} days")
    print("=" * 70)
    
    # Create features
    print("\nCreating features...")
    X = create_features(df)
    y = create_labels(df)
    
    # Remove rows with NaN
    valid_idx = X.notna().all(axis=1)
    X = X[valid_idx].values
    y = y[valid_idx]
    
    print(f"Feature set: {X.shape[1]} features, {len(X)} samples")
    print(f"Label distribution: {np.bincount(y)}")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create sequences
    X_seq, y_seq = create_sequences(X_scaled, y, LOOKBACK_WINDOW)
    
    print(f"Sequence shape: {X_seq.shape}")
    print(f"Sequences: {len(X_seq)}")
    
    # Time-based split
    split_idx = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
    
    print(f"\nTrain: {len(X_train)} sequences")
    print(f"Test: {len(X_test)} sequences")
    
    # Build LSTM model
    print("\n" + "=" * 70)
    print("Building LSTM model...")
    print("=" * 70)
    
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(LOOKBACK_WINDOW, X_seq.shape[2])),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print(model.summary())
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    ]
    
    # Train
    print("\nTraining...")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    print("\n" + "=" * 70)
    print("Evaluation on Test Set")
    print("=" * 70)
    
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Strategy simulation
    print("\n" + "=" * 70)
    print("Strategy Backtest on Test Set")
    print("=" * 70)
    
    test_start_idx = np.where(valid_idx)[0][0] + LOOKBACK_WINDOW + split_idx
    test_df = df.iloc[test_start_idx:test_start_idx + len(y_test)].copy()
    test_df['signal'] = y_pred
    test_df['signal_proba'] = y_pred_proba.flatten()
    test_df['returns'] = test_df['close'].pct_change()
    
    # Calculate strategy returns
    position = np.zeros(len(test_df))
    for i in range(1, len(test_df)):
        if test_df['signal'].iloc[i-1] == 1:
            position[i] = 1.0
    
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
    model_file = OUTPUT_DIR / f"spy_lstm_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"
    model.save(str(model_file))
    
    # Save config
    config_file = OUTPUT_DIR / f"spy_lstm_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(config_file, 'w') as f:
        json.dump({
            'ticker': TICKER,
            'lookback_window': LOOKBACK_WINDOW,
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
            'epochs_trained': len(history.history['loss']),
            'generated_at': datetime.now().isoformat()
        }, f, indent=2)
    
    print(f"\nSaved to: {config_file}")
    print("=" * 70)


if __name__ == "__main__":
    run_lstm_training()
