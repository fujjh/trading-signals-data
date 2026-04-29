# SignalsAlpha SPY Process Flow

## Overview

This repository contains the **SPY-only** data pipeline for SignalsAlpha - a focused S&P 500 ETF signal generation system. This is a streamlined version of the full multi-ticker pipeline, optimized for single-ticker analysis with daily data only.

**Key Features:**
- **SPY Only:** Single-ticker focus on S&P 500 ETF
- **Daily Data Only:** No weekly/monthly intervals for simplicity
- **Technical Analysis Only:** 100% technical scoring (no fundamentals)
- **Sequential Processing:** No batch processing needed for single ticker
- **Full Backtesting:** Complete recursive backtesting infrastructure retained

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SIGNALSALPHA SPY PIPELINE                                │
└─────────────────────────────────────────────────────────────────────────────┘

STEP 1: Time Series Collection
    │
    ▼
┌──────────────────────┐
│ TimeSeriesCollector  │──► Collects SPY daily OHLCV data (incremental updates)
│       v1.0           │──► Fetches from Yahoo Finance
└──────────────────────┘──► Outputs: data/time_series/SPY/SPY_1d.csv
    │
    ▼
STEP 2: Time Series Validation
    │
    ▼
┌─────────────────────────┐
│ TimeSeriesValidator     │──► Validates OHLCV data integrity
│         v1.0            │──► Checks for nulls, outliers, date ranges
└─────────────────────────┘──► Outputs: data/validated/validation_report_spy_{date}.json
    │
    ▼
STEP 3: Technical Analysis
    │
    ▼
┌─────────────────────────┐
│ TechnicalAnalyzer       │──► Calculates technical indicators
│         v1.0            │──► SMA/EMA, RSI, MACD, HMA, Elder Impulse
│                         │──► Bollinger Bands, ATR
└─────────────────────────┘──► Outputs: data/technical_analysis/SPY/SPY_1d_technical.csv
    │
    ▼
STEP 4: Technical Analysis Validation
    │
    ▼
┌─────────────────────────────┐
│ TechnicalValidator          │──► Validates indicator calculations
│           v1.0              │──► Checks required columns, NaN values
└─────────────────────────────┘──► Outputs: data/validated/validation_report_technical_spy_{date}.json
    │
    ▼
STEP 5: Scoring & Ranking
    │
    ▼
┌─────────────────────────┐
│ ScoringRanker           │──► 100% Technical scoring (no fundamentals)
│         v1.0            │──► Generates BUY/SELL/HOLD signals
│                         │──► Signal strength: STRONG_BUY to STRONG_SELL
└─────────────────────────┘──► Outputs: data/signals_scored/spy_scored_{date}.csv
    │
    ▼
STEP 6: Signal Validation
    │
    ▼
┌─────────────────────────────┐
│ SignalValidator             │──► Validates scored signals
│           v1.0              │──► Checks score ranges, signal values
└─────────────────────────────┘──► Outputs: data/validated/validation_report_signals_spy_{date}.json

═══════════════════════════════════════════════════════════════════════════
                            BACKTESTING & OPTIMIZATION
═══════════════════════════════════════════════════════════════════════════

STEP 7: Signal History Tracker
    │
    ▼
┌─────────────────────────────┐
│ SignalHistoryTracker        │──► Tracks all generated signals over time
│           v1.0              │──► Stores rationale for each signal
└─────────────────────────────┘──► Outputs: data/signals_history/signals_history.csv

STEP 8: Website Output Generator
    │
    ▼
┌─────────────────────────────┐
│ WebsiteOutputGenerator      │──► Formats signals for website display
│           v1.0              │──► JSON output for frontend integration
└─────────────────────────────┘──► Outputs: data/website_output/signals.json

STEP 9: Genetic Optimizer
    │
    ▼
┌─────────────────────────────┐
│ GeneticOptimizer            │──► Optimizes indicator weights via genetic algorithm
│           v1.0              │──► 14 genes (weights, thresholds, position sizing)
│                             │──► Population: 30, Generations: 50
└─────────────────────────────┘──► Outputs: optimized_configs/genome_best.json

STEP 10: Historical Backtester
    │
    ▼
┌─────────────────────────────┐
│ HistoricalBacktester        │──► TradingView-style backtesting
│           v2.0              │──► Realistic execution (next-day open)
│                             │──► $100K starting, max 20% position
└─────────────────────────────┘──► Outputs: data/backtests/backtest_report_{timestamp}.json

STEP 11: Walk-Forward Analysis
    │
    ▼
┌─────────────────────────────┐
│ WalkForwardAnalysis         │──► Out-of-sample validation
│           v1.0              │──► 252d train, 20d test windows
│                             │──► Rolling window approach
└─────────────────────────────┘──► Outputs: walk_forward_results/aggregate_results.json

STEP 12: Monte Carlo Stress Testing
    │
    ▼
┌─────────────────────────────┐
│ MonteCarloStressTester      │──► Statistical robustness testing
│           v1.0              │──► 10,000 simulations
│                             │──► Black swan injection, regime simulation
└─────────────────────────────┘──► Outputs: monte_carlo_results/stress_test_results.json

STEP 13: Model Selector
    │
    ▼
┌─────────────────────────────┐
│ ModelSelector               │──► Multi-criteria ranking for production
│           v1.0              │──► Composite scoring: PF, Sharpe, Walk-Forward, Monte Carlo
│                             │──► Production config generation
└─────────────────────────────┘──► Outputs: production_config/production_weights.json
```

## Pipeline Steps

### Core Pipeline (Steps 1-6)

| Step | File | Purpose | Input | Output |
|------|------|---------|-------|--------|
| 1 | step1_time_series_collector.py | Collect SPY daily data | Yahoo Finance API | data/time_series/SPY/SPY_1d.csv |
| 2 | step2_time_series_validator.py | Validate price data | SPY_1d.csv | validation_report_spy_{date}.json |
| 3 | step3_technical_analysis.py | Calculate indicators | SPY_1d.csv | SPY_1d_technical.csv |
| 4 | step4_technical_validator.py | Validate indicators | SPY_1d_technical.csv | validation_report_technical_spy_{date}.json |
| 5 | step5_scoring_ranking.py | Generate signals | SPY_1d_technical.csv | spy_scored_{date}.csv |
| 6 | step6_signal_validator.py | Validate signals | spy_scored_{date}.csv | validation_report_signals_spy_{date}.json |

### Backtesting & Optimization (Steps 7-13)

| Step | File | Purpose |
|------|------|---------|
| 7 | step7_signal_history_tracker.py | Track signal history over time |
| 8 | step8_website_output_generator.py | Format for website display |
| 9 | step9_genetic_optimizer.py | Optimize indicator weights |
| 10 | step10_backtester.py | Historical backtesting with realistic execution |
| 11 | step11_walk_forward.py | Out-of-sample validation |
| 12 | step12_monte_carlo.py | Statistical stress testing |
| 13 | step13_model_selector.py | Production model selection |

## Usage

### Run Full Pipeline

```bash
bash run_spy_pipeline.sh
```

This executes steps 1-6 in sequence.

### Run Individual Steps

```bash
# Step 1: Collect data
python3 step1_time_series_collector.py

# Step 2: Validate
python3 step2_time_series_validator.py

# Step 3: Technical analysis
python3 step3_technical_analysis.py

# Step 4: Validate technical
python3 step4_technical_validator.py

# Step 5: Generate signals
python3 step5_scoring_ranking.py

# Step 6: Validate signals
python3 step6_signal_validator.py
```

### Run Backtesting

```bash
# Generate signal history first
python3 step7_signal_history_tracker.py

# Run backtest
python3 step10_backtester.py

# Optimize weights
python3 step9_genetic_optimizer.py

# Walk-forward validation
python3 step11_walk_forward.py

# Monte Carlo stress test
python3 step12_monte_carlo.py

# Select production model
python3 step13_model_selector.py
```

## Directory Structure

```
process_flow_v1_SPY/
├── step1_time_series_collector.py      # Step 1
├── step2_time_series_validator.py        # Step 2
├── step3_technical_analysis.py          # Step 3
├── step4_technical_validator.py          # Step 4
├── step5_scoring_ranking.py             # Step 5
├── step6_signal_validator.py            # Step 6
├── step7_signal_history_tracker.py       # Step 7
├── step8_website_output_generator.py     # Step 8
├── step9_genetic_optimizer.py           # Step 9
├── step10_backtester.py                  # Step 10
├── step11_walk_forward.py                # Step 11
├── step12_monte_carlo.py                 # Step 12
├── step13_model_selector.py              # Step 13
├── run_spy_pipeline.sh                   # Pipeline runner
├── README.md                             # This file
└── modules/                              # Supporting modules
    ├── metrics_calculator.py
    ├── feature_engineering.py
    ├── results_db.py
    └── pipeline_bridge.py

data/
├── time_series/SPY/                      # Step 1 output
│   └── SPY_1d.csv
├── technical_analysis/SPY/               # Step 3 output
│   └── SPY_1d_technical.csv
├── signals_scored/                       # Step 5 output
│   └── spy_scored_{date}.csv
├── validated/                            # Validation reports
│   ├── validation_report_spy_{date}.json
│   ├── validation_report_technical_spy_{date}.json
│   └── validation_report_signals_spy_{date}.json
├── signals_history/                      # Step 7 output
├── website_output/                       # Step 8 output
├── backtests/                            # Step 10 output
├── walk_forward_results/                 # Step 11 output
├── monte_carlo_results/                  # Step 12 output
└── production_config/                    # Step 13 output
```

## Key Differences from Full Pipeline

| Feature | Full Pipeline (v1) | SPY Pipeline |
|---------|-------------------|--------------|
| Tickers | ~4,800 stocks | SPY only |
| Intervals | 1d, 1wk, 1mo | 1d only |
| Batch Processing | Yes (required for OCI) | No |
| Fundamentals | Yes (Step 2) | No |
| Scoring | 75% Tech / 25% Fund | 100% Technical |
| Auto-Restart Scripts | Multiple | None needed |

## Technical Indicators Calculated

- **Trend:** SMA 20/50, EMA 12/26, HMA 13
- **Momentum:** RSI, MACD (with signal line)
- **Volatility:** Bollinger Bands, ATR
- **System:** Elder Impulse (trend + momentum)

## Scoring Methodology (Step 5)

**Technical Score (100% of combined):**
- Base: 50 (neutral)
- Trend alignment: ±10
- RSI oversold/overbought: ±15
- MACD crossover: ±10
- HMA position: ±8
- Elder Impulse: ±10

**Signal Thresholds:**
- STRONG_BUY: ≥75
- BUY: 60-74
- HOLD: 40-59
- SELL: 25-39
- STRONG_SELL: <25

## Backtesting v2.0 Features

- **Realistic Execution:** Entry at next-day open (no lookahead bias)
- **Portfolio Simulation:** $100K starting capital
- **Position Sizing:** Max 20% per position, partial fills allowed
- **Risk Management:** Cash-constrained entries, signal prioritization
- **Comprehensive Metrics:** Sharpe, Sortino, Profit Factor, Expectancy, Max Drawdown

## Requirements

```bash
pip install pandas numpy yfinance pytz
```

## Notes

- All dates use UTC timezone for consistency
- Incremental collection only fetches missing data since last update
- Validation reports saved to `data/validated/`
- Final signals saved to `data/signals_scored/`

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-04-29 | Initial SPY-only release |

## Author

SignalsAlpha
