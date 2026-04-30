# Machine Learning Techniques for SPY Trading

## Overview

This directory contains multiple machine learning approaches for generating SPY trading signals:

1. **Genetic Algorithm (GA)** - Evolutionary optimization (current best: fitness 0.7569)
2. **XGBoost** - Gradient boosted decision trees for classification
3. **LSTM** - Recurrent neural networks for time series prediction
4. **Deep Reinforcement Learning (PPO)** - Policy gradient learning

## Comparison Summary

| Method | Approach | Strengths | Weaknesses |
|--------|----------|-----------|------------|
| **GA** | Evolutionary optimization | Interpretable, finds global optimum | Slower, needs many evaluations |
| **XGBoost** | Gradient boosting | Fast, handles non-linearities, feature importance | Can overfit, requires careful tuning |
| **LSTM** | Neural sequence model | Captures temporal patterns | Needs lots of data, black box |
| **PPO** | Reinforcement learning | Learns optimal policy directly | Unstable training, complex reward design |

## Current Best Results

### Genetic Algorithm (100-seed + Grid Search)
- **Fitness**: 0.7569
- **Return**: 336.30%
- **Sharpe**: 0.86
- **Max DD**: 16.4%
- **Trades**: 71

### Configuration Files
- `step9_genetic_optimizer.py` - Single/multi-seed GA
- `step9_grid_search_focused.py` - Grid search refinement
- `step9_xgboost_trainer.py` - XGBoost implementation
- `step9_lstm_trainer.py` - LSTM neural network
- `step9_rl_trainer.py` - PPO reinforcement learning
- `step9_ml_comparison.py` - Comparison dashboard

## Running the Comparisons

```bash
# Run all ML techniques
python3 step9_ml_comparison.py

# Run individual techniques
python3 step9_xgboost_trainer.py
python3 step9_lstm_trainer.py
python3 step9_rl_trainer.py
```

## Expected Results

Based on literature and preliminary testing:

- **XGBoost**: Should achieve similar or slightly lower performance than GA, but faster
- **LSTM**: May capture temporal patterns but requires careful regularization
- **PPO**: Can learn complex policies but training is unstable

## Notes

- All techniques use the same features (technical indicators) for fair comparison
- Train/test split is time-based (80/20) to avoid lookahead bias
- Performance measured on held-out test set
- GA remains the benchmark due to interpretability and proven performance
