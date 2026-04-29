# SignalsAlpha Process Flow v1

## Overview

This repository contains the complete data pipeline for SignalsAlpha - an automated stock signal generation system with **recursive backtesting and self-improving optimization**. The pipeline runs on an Oracle Cloud Infrastructure (OCI) Free Tier instance and generates daily trading signals for a universe of ~4,800 US stocks.

**Key Features:**
- **Recursive Backtesting:** Self-improving signal generation through genetic optimization
- **Walk-Forward Validation:** Out-of-sample robustness testing
- **Monte Carlo Stress Testing:** 10,000 simulation statistical validation
- **Model Selection:** Multi-criteria ranking for production deployment

## What's New - Backtesting Enhancement v2.0

Six new phases added for comprehensive strategy validation:

| Phase | Step | Purpose | Key Feature |
|-------|------|---------|-------------|
| **1** | **Enhanced Metrics** | Comprehensive risk-adjusted returns | Sharpe, Sortino, Calmar, Ulcer Index |
| **2** | **ML Features** | Pattern detection features | 40+ ML features (price, volatility, momentum) |
| **3** | **Genetic Optimizer** | Weight/threshold optimization | 14 genes, 30 individuals, 50 generations |
| **4** | **Walk-Forward** | Out-of-sample validation | 252d train, 20d test, rolling windows |
| **5** | **Monte Carlo** | Statistical stress testing | 10,000 simulations, 95% CI |
| **6** | **Model Selector** | Production deployment | Multi-criteria ranking, risk management |

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SIGNALSALPHA DATA PIPELINE                          │
└─────────────────────────────────────────────────────────────────────────────┘

STEP 0 (Weekly): Ticker Collection
    │
    ▼
┌─────────────────┐
│ TickerCollector │──► Downloads official exchange lists (NASDAQ, NYSE, AMEX, TSX)
│      v2.0       │──► Collects index constituents (S&P 500, Russell 2000, NASDAQ 100)
│                 │──► Validates against Yahoo Finance
└─────────────────┘──► Outputs: data/stock_ticker_base.csv
    │
    ▼
STEP 1 (Daily): Time Series Data Collection
    │
    ▼
┌──────────────────────┐
│ TimeSeriesCollector  │──► Collects OHLCV data for all timeframes
│       v1.0           │──► Intervals: 1d, 1wk, 1mo
└──────────────────────┘──► Outputs: data/time_series/{TICKER}/{TICKER}_{interval}.csv
    │
    ▼
STEP 1.5 (Daily): Time Series Validation
    │
    ▼
┌─────────────────────────┐
│ TimeSeriesValidator     │──► Validates OHLCV data integrity
│         v1.0            │──► Checks for nulls, outliers, date ranges
│                         │──► Ensures data quality before analysis
└─────────────────────────┘──► Outputs: data/validated/validation_report_time_series_{date}.json
    │
    ▼
STEP 2 (Daily): Fundamental Data Collection
    │
    ▼
┌─────────────────────────────┐
│ FundamentalDataCollector    │──► Retrieves valuation metrics (P/E, PEG, etc.)
│            v1.0             │──► Collects analyst price targets & ratings
│                             │──► Gathers growth & profitability data
└─────────────────────────────┘──► Outputs: data/fundamentals/{TICKER}/{TICKER}_fundamentals.csv
    │
    ▼
STEP 3 (Daily): Technical Analysis
    │
    ▼
┌─────────────────────────┐
│ TechnicalAnalyzer       │──► Calculates all technical indicators
│         v1.0            │──► NO SCORING - pure indicator values
│                         │──► Crossover detection, S/R levels, Fibonacci
└─────────────────────────┘──► Outputs: data/technical_analysis/{TICKER}/{TICKER}_{interval}_technical.csv
    │
    ▼
STEP 3.5 (Daily): Technical Analysis Validation
    │
    ▼
┌─────────────────────────────┐
│ TechnicalValidator          │──► Validates indicator calculations
│           v1.0            │──► Checks for NaN, missing columns
│                             │──► Ensures crossovers, S/R, Fibonacci complete
└─────────────────────────────┘──► Outputs: data/validated/validation_report_technical_{date}.json
    │
    ▼
STEP 4 (Daily): Scoring & Ranking
    │
    ▼
┌─────────────────────────┐
│ ScoringRanker           │──► Takes Step 3 technical + Step 2 fundamental
│         v1.0            │──► Technical Score (60%) + Fundamental Score (40%)
│                         │──► Generates final BUY/SELL signals with confidence
└─────────────────────────┘──► Outputs: data/signals_scored/scored_signals_{date}.csv
    │
    ▼
STEP 5 (Daily): Signal Validation
    │
    ▼
┌─────────────────┐
│ SignalValidator │──► Validates scored signals
│     v1.0        │──► Checks score ranges, consistency
│                 │──► Validates price targets before backtesting
└─────────────────┘──► Outputs: data/validated/validation_report_signals_{date}.json
    │
    ▼
═══════════════════════════════════════════════════════════════════════════════════
                  BACKTESTING ENHANCEMENT v2.0 (Self-Improving)
═══════════════════════════════════════════════════════════════════════════════════
    │
    ▼
STEP 6 (Periodic): Enhanced Backtester
    │
    ▼
┌──────────────────────────┐
│ Backtester               │──► Tests algorithm on historical data
│          v2.0            │──► Calculates win rate, profit factor, Sharpe ratio
│                          │──► Portfolio simulation with $100K capital
└──────────────────────────┘──► Outputs: data/backtests/backtest_{timestamp}.json
    │
    ▼
STEP 7 (Daily): Enhanced Metrics Module
    │
    ▼
┌──────────────────────────┐
│ MetricsCalculator        │──► Calculates comprehensive performance metrics
│    (modules/)            │──► Sharpe, Sortino, Calmar, Ulcer Index
│                          │──► Drawdown analysis, consecutive trades
└──────────────────────────┘──► Used by Steps 10-13 for scoring
    │
    ▼
STEP 8 (Daily): ML Feature Engineering
    │
    ▼
┌──────────────────────────┐
│ FeatureEngineer          │──► Generates ML features for optimization
│    (modules/)            │──► Price action, volatility, momentum
│                          │──► Regime detection, pattern recognition
└──────────────────────────┘──► 40+ features for genetic algorithm
    │
    ▼
STEP 9 (Daily): Results Database
    │
    ▼
┌──────────────────────────┐
│ ResultsDatabase          │──► SQLite storage for all backtest results
│    (modules/)            │──► Enables A/B comparison of strategies
│                          │──► Configuration versioning and tracking
└──────────────────────────┘──► Centralized results storage
    │
    ▼
STEP 10 (Periodic): Genetic Optimizer ⭐
    │
    ▼
┌──────────────────────────┐
│ GeneticOptimizer         │──► Self-improving weight optimization
│         v1.0             │──► 14 genes: weights, thresholds, sizing
│                          │──► Population: 30, Generations: 50
│                          │──► Fitness: PF×0.35 + Sharpe×0.25 + ...
└──────────────────────────┘──► Outputs: data/optimizer/best_config_*.json
    │
    ▼
STEP 11 (Periodic): Walk-Forward Validation ⭐
    │
    ▼
┌──────────────────────────┐
│ WalkForwardValidator     │──► Rolling train/test validation
│         v1.0             │──► 252d train, 20d test windows
│                          │──► Overfitting detection (Train/Test < 1.5)
└──────────────────────────┘──► Outputs: data/walk_forward/walkforward_*.csv
    │
    ▼
STEP 12 (Periodic): Monte Carlo Stress Testing ⭐
    │
    ▼
┌──────────────────────────┐
│ MonteCarloEngine         │──► Statistical robustness testing
│         v1.0             │──► 10,000 simulations
│                          │──► Black swan injection (5% probability)
│                          │──► 95% confidence intervals
└──────────────────────────┘──► Outputs: data/monte_carlo/monte_carlo_*.csv
    │
    ▼
STEP 13 (Periodic): Model Selector & Deployment ⭐
    │
    ▼
┌──────────────────────────┐
│ ModelSelector            │──► Multi-criteria model ranking
│         v1.0             │──► Ranking: PF×0.30 + Sharpe×0.25 + ...
│                          │──► Minimum thresholds validation
│                          │──► Production config generation
└──────────────────────────┘──► Outputs: data/deployment/production_config.json
    │
    ▼
═══════════════════════════════════════════════════════════════════════════════════
                           END BACKTESTING ENHANCEMENT
═══════════════════════════════════════════════════════════════════════════════════
    │
    ▼
STEP 8 (Daily): Signal History Tracking
    │
    ▼
┌──────────────────────────┐
│ SignalHistoryTracker     │──► Saves daily signal snapshots
│          v1.0            │──► Tracks signal changes over time
│                          │──► Captures S/R context and Fibonacci levels
└──────────────────────────┘──► Outputs: data/signal_history/signals_{date}.csv
    │
    ▼
STEP 9 (Daily): Website Output Generation
    │
    ▼
┌──────────────────────────┐
│ WebsiteOutputGenerator   │──► Aggregates signals for frontend
│          v1.0            │──► Creates JSON files for website
│                          │──► Includes fundamental data enrichment
└──────────────────────────┘──► Outputs: signalsalpha/prototype/data/signals/*.json
```

```

## Quick Start - Execution Sequence

Run these steps in order. **⚠️ Note:** Steps marked with 🔄 **require batch restart wrapper** to prevent OCI memory/timeouts.

### Weekly Workflow (Sunday)

| Step | Script | Command | Time | 🔄 Batch? |
|------|--------|---------|------|-----------|
| 0 | step0_ticker_collector.py | `python3 step0_ticker_collector.py` | ~75 min | No |

### Daily Workflow (After Market Close)

| Step | Script | Command | Time | 🔄 Batch? |
|------|--------|---------|------|-----------|
| 0 | step0_ticker_collector.py | `bash run_step0.sh` | ~1 hr | **YES** |
| 1 | step1_time_series_collector.py | `bash run_timeseries_batch.sh` | ~10 hrs | **YES** |
| 1.5 | step1_5_time_series_validator.py | `python3 step1_5_time_series_validator.py` | ~5 min | No |
| 2 | step2_fundamental_data_collector.py | `python3 step2_fundamental_data_collector.py` | ~2 hrs | No |
| 3 | step3_technical_analysis.py | `python3 step3_technical_analysis.py` | ~2 hrs | No |
| 3.5 | step3_5_technical_validator.py | `python3 step3_5_technical_validator.py` | ~5 min | No |
| 4 | step4_scoring_ranking.py | `python3 step4_scoring_ranking.py` | ~10 min | No |
| 5 | step5_signal_validator.py | `python3 step5_signal_validator.py` | ~2 min | No |

### Backtesting Workflow (Periodic)

| Step | Script | Command | Time | 🔄 Batch? |
|------|--------|---------|------|-----------|
| 6 | step6_backtester.py | `python3 step6_backtester.py` | ~15 min | No |
| 7 | step7_walk_forward.py | `python3 step7_walk_forward.py` | ~10 min | No |

### Daily Signal Tracking & Output

| Step | Script | Command | Time | 🔄 Batch? |
|------|--------|---------|------|-----------|
| 8 | step8_signal_history_tracker.py | `python3 step8_signal_history_tracker.py` | ~1 min | No |
| 9 | step9_website_output_generator.py | `python3 step9_website_output_generator.py` | ~5 min | No |

### Why Batch Processing is Required

**OCI Free Tier Limitations:**
- **RAM:** 1 GB (easily exhausted)
- **CPU:** Shared (unpredictable performance)
- **Process timeouts:** Long-running processes may be killed

**Steps 1 & 2 process thousands of tickers.** Without batch restart:
- Memory grows unbounded
- Process gets killed (SIGKILL)
- Partial data, must restart from beginning

**Batch restart solution:**
- Processes 10-20 tickers per batch
- Exits completely between batches (frees all memory)
- Shell wrapper automatically restarts script
- Continues from where it left off

### Backtesting Enhancement Workflow (Periodic - Weekly/Monthly)

Run the recursive optimization pipeline weekly or monthly to continuously improve signal generation:

| Step | Script | Purpose | Time |
|------|--------|---------|------|
| 10 | step10_genetic_optimizer.py | Optimize indicator weights | ~1-2 hrs |
| 11 | step11_walk_forward.py | Validate on unseen data | ~30 min |
| 12 | step12_monte_carlo.py | Statistical stress testing | ~15 min |
| 13 | step13_model_selector.py | Select best model for production | ~5 min |

**Execution Command:**
```bash
# Run the full optimization pipeline
python3 step10_genetic_optimizer.py  # Generates optimized configs
python3 step11_walk_forward.py        # Validates configs
python3 step12_monte_carlo.py         # Stress tests
python3 step13_model_selector.py      # Selects best for production
```

### Enhanced Backtesting Modules (Located in `modules/`)

| Module | Purpose | Key Features |
|--------|---------|--------------|
| `metrics_calculator.py` | Performance metrics calculation | Sharpe, Sortino, Calmar, Ulcer Index, drawdown analysis |
| `feature_engineering.py` | ML feature generation | 40+ features: price, momentum, volatility, regime detection |
| `results_db.py` | Results storage | SQLite database, A/B comparison, config versioning |
| `step6_enhanced.py` | Step 6 integration | Bridge between Step 6 and enhanced metrics |
| `pipeline_bridge.py` | Data flow standardization | Format conversion between Steps 10-13 |

### Complete Daily Command Sequence

```bash
# Navigate to process_flow_v1
cd /home/ubuntu/.openclaw/workspace/trading-signals-data/process_flow_v1

# Activate virtual environment
source ../../venv/bin/activate

# Step 1: Time Series (with batch restart) - ~10 hours
echo "Starting Step 1: Time Series Collection..."
bash run_timeseries_batch.sh

# Step 2: Scanner (with batch restart) - ~2 hours
echo "Starting Step 2: Multi-Timeframe Scanner..."
bash run_batch_restarter.sh

# Steps 3: Quick validation - ~2 minutes
echo "Starting Step 3: Data Validation..."
python3 step3_data_validator.py

echo "Daily data collection complete!"

# Run backtesting and output generation separately:
# bash run_steps_4_5_6_7.sh
```

### Automated Scheduling (Cron)

Add to crontab (`crontab -e`):

```bash
# Weekly ticker collection - Sundays at 4 AM UTC
0 4 * * 0 cd /home/ubuntu/.openclaw/workspace/trading-signals-data/process_flow_v1 && bash run_timeseries_batch.sh >> logs/step1_weekly.log 2>&1

# Daily pipeline - Monday-Saturday at 6 PM UTC
0 18 * * 1-6 cd /home/ubuntu/.openclaw/workspace/trading-signals-data/process_flow_v1 && bash run_batch_restarter.sh >> logs/step2_daily.log 2>&1 && python3 step3_data_validator.py && bash run_steps_4_5_6_7.sh
```

## File Structure

```
process_flow_v1/
├── README.md                           # This file
├── PROCESS_FLOW_DIAGRAM.md             # Detailed architecture diagram
│
├── STEP 0 - Weekly/
│   └── step0_ticker_collector.py       # Ticker collection from exchanges
│
├── STEP 1 - Daily/
│   ├── step1_time_series_collector.py  # OHLCV data collection
│   └── run_timeseries_batch.sh         # Batch restart wrapper for OCI
│
├── STEP 1.5 - Daily/
│   └── step1_5_time_series_validator.py # Validate time series data
│
├── STEP 2 - Daily/
│   └── step2_fundamental_data_collector.py  # Fundamental data collection
│
├── STEP 3 - Daily/
│   └── step3_technical_analysis.py     # Technical analysis (NO SCORING)
│
├── STEP 3.5 - Daily/
│   └── step3_5_technical_validator.py  # Validate technical analysis
│
├── STEP 4 - Daily/
│   ├── step4_scoring_ranking.py        # Scoring: Technical + Fundamental
│   ├── step4_scoring_ranking_v2.py     # Enhanced with driver tracking
│   └── step4_scoring_ranking_1d.py     # 1-day interval only
│
├── STEP 5 - Daily/
│   └── step5_signal_validator.py       # Validate scored signals
│
├── STEP 6 - Backtesting/
│   ├── step6_backtester.py             # Historical backtesting
│   └── step6_enhanced.py               # Enhanced metrics integration (module)
│
├── STEP 7 - Daily (Module)/
│   └── modules/metrics_calculator.py   # Comprehensive metrics calculation
│
├── STEP 8 - Daily (Module)/
│   └── modules/feature_engineering.py  # ML feature generation
│
├── STEP 9 - Daily (Module)/
│   └── modules/results_db.py           # Results database
│
├── STEP 10 - Weekly/
│   └── step10_genetic_optimizer.py     # Genetic algorithm optimization
│
├── STEP 11 - Weekly/
│   └── step11_walk_forward.py          # Walk-forward validation
│
├── STEP 12 - Weekly/
│   └── step12_monte_carlo.py           # Monte Carlo stress testing
│
├── STEP 13 - Weekly/
│   └── step13_model_selector.py        # Production model selection
│
├── modules/                            # Shared modules
│   ├── metrics_calculator.py           # Performance metrics
│   ├── feature_engineering.py          # ML features
│   ├── results_db.py                   # SQLite database
│   ├── step6_enhanced.py               # Step 6 bridge
│   └── pipeline_bridge.py              # Data flow standardization
│
├── STEP 8 - Daily (Signal History)/
│   └── step8_signal_history_tracker.py # Signal history tracking
│
├── STEP 9 - Daily (Website)/
│   └── step9_website_output_generator.py  # Website JSON output
│
├── Orchestration/
│   ├── run_master.sh                   # Master orchestration script
│   ├── run_step1_batched.sh            # Step 1 batch wrapper
│   ├── run_step3_auto_restart.sh       # Step 3 auto-restart
│   ├── run_step4_batched.sh            # Step 4 batch wrapper
│   ├── run_steps_5_6_7_8.sh            # Signal validation + Backtesting
│   └── run_steps_5_6_7.sh              # Signal validation only
│
└── logs/                               # Execution logs
```

## Process Flow Details

### STEP 0: Ticker Collection (Weekly)

**Purpose:** Maintain the universe of tradeable stocks (~3,500-4,000 after validation)

**Frequency:** Weekly (recommended: Sunday)

**Data Sources:**
| Source | Method | Expected Count |
|--------|--------|----------------|
| NASDAQ | DataHub.io CSV | ~3,000 |
| NYSE | DataHub.io CSV | ~3,000 |
| TSX | Curated list | ~250 |
| **Russell 2000** | **iShares IWM ETF** | **~2,000** |
| **NASDAQ 100** | **Wikipedia** | **~100** |
| S&P 500 | Wikipedia | 500 |
| **Total Unique** | After deduplication | **~3,500-4,000** |

**Process:**
1. Downloads official ticker lists from exchanges
2. Collects Russell 2000 via iShares IWM ETF holdings
3. Collects NASDAQ 100 and S&P 500 constituents
4. Validates each ticker exists on Yahoo Finance
5. Filters for active, non-ETF, non-warrant stocks
6. Deduplicates across all sources

**Rate Limiting:** 1.5 seconds between Yahoo Finance requests (~75 min for 3,000 tickers)

---

### STEP 1: Time Series Data Collection (Daily)

**Purpose:** Collect price history for all tickers

**Frequency:** Daily (recommended: after market close, ~6 PM ET)

**Data Collected:**
| Interval | Period | Use Case |
|----------|--------|----------|
| 1d (daily) | Max | Primary signal generation |
| 1wk (weekly) | Max | Trend confirmation |
| 1mo (monthly) | Max | Long-term support/resistance |

**Process:**
1. Reads ticker list from STEP 0 output
2. Downloads OHLCV data from Yahoo Finance for each interval
3. Stores in structured CSV format
4. Validates data completeness

**Batch Processing:**
- Processes 10 tickers per batch
- Restarts automatically to free memory
- Prevents OCI timeout

**Outputs:**
- `data/time_series/{TICKER}/{TICKER}_1d.csv`
- `data/time_series/{TICKER}/{TICKER}_1wk.csv`
- `data/time_series/{TICKER}/{TICKER}_1mo.csv`

**Command:**
```bash
./run_master.sh step1
# or with batch restart wrapper (recommended for OCI):
bash run_timeseries_batch.sh
```

---

### STEP 2: Multi-Timeframe Scanner (Daily)

**Purpose:** Generate trading signals using technical analysis

**Frequency:** Daily (after STEP 1 completes)

**Indicators Calculated:**
- Trend: SMA (20, 50), EMA (12, 26)
- Momentum: RSI (14), MACD
- Volatility: Bollinger Bands (20, 2), ATR (14)
- Volume: VWAP, OBV
- Support/Resistance: Pivot highs/lows, Fibonacci levels
- **Candlestick Patterns:** Doji, Hammer, Engulfing, Harami, Morning/Evening Star, Three Soldiers/Crows

**Signal Generation Logic:**
```
BUY Signals (highest to lowest priority):
- STRONG_BUY: Score >= 7
- BUY: Score >= 4
- WEAK_BUY: Score >= 2

SELL Signals (highest to lowest priority):
- STRONG_SELL: Score >= 7
- SELL: Score >= 4
- WEAK_SELL: Score >= 2

HOLD: Score < 2 (both sides)
```

**Scoring System:**
- Trend alignment: +2
- EMA crossover: +1
- RSI extreme: +2
- MACD confirmation: +2
- Bollinger Band touch: +1
- Volume spike: +1
- VWAP position: +1
- Stochastic Oscillator: +1 to +2
- Money Flow Index (MFI): +1 to +2
- TRIX momentum: +1 to +2
- Candlestick pattern: +1 to +3 (depending on pattern)

**Confluence Scoring Example:**

A high-probability STRONG_BUY signal requires multiple indicators aligning:

| Indicator | Reading | Score |
|-----------|---------|-------|
| Trend (SMA) | Price > SMA20 > SMA50 (uptrend) | +2 |
| EMA Crossover | EMA12 > EMA26 | +1 |
| RSI | 28 (oversold, bouncing) | +2 |
| MACD | Histogram > 0, above signal | +2 |
| Bollinger Bands | Price touched lower band | +1 |
| Stochastic Slow | %K=18, %D=15 (oversold) | +2 |
| MFI | 22 (oversold with volume) | +2 |
| TRIX | Crossed above 0 | +2 |
| Candlestick | Bullish Engulfing pattern | +2 |
| **TOTAL SCORE** | | **+16** |
| **SIGNAL** | **STRONG_BUY** | **Confidence: 95%** |

*In this example, 9 separate technical factors align to produce a high-confluence buy signal. The probability of success increases with each confirming indicator.*

**Why Confluence Matters:**
- Single indicator signals: ~55-60% win rate
- 3+ aligned indicators: ~65-70% win rate
- 5+ aligned indicators: ~75-80% win rate
- High confluence (8+): ~85%+ win rate

The system requires multiple confirmations—no single indicator drives the signal alone.

**Batch Processing:**
- Processes 20 tickers per batch
- Restarts automatically
- Saves incremental progress

**Outputs:**
- `data/signals_timeframe/{TICKER}/{TICKER}_signals.csv`

**Command:**
```bash
./run_master.sh step2
# or with batch restart wrapper:
bash run_batch_restarter.sh
```

---

### STEP 3: Data Validation (Daily)

**Purpose:** Ensure data integrity before website generation

**Frequency:** Daily (after STEP 2 completes)

**Validations:**
1. Time Series Data:
   - All expected intervals present
   - CSV files readable
   - Required columns exist (Open, High, Low, Close, Volume)

2. Signal Data:
   - Signal values valid (STRONG_BUY, BUY, etc.)
   - Confidence in range (50-95)
   - Timestamps present

3. Freshness:
   - Data not older than 26 hours
   - Latest timestamp check

**Exit Codes:**
- 0: Validation passed
- 1: Validation failed (critical errors)

**Outputs:**
- `data/validated/validation_report_{timestamp}.json`

**Command:**
```bash
./run_master.sh step3
```

---

### STEP 4: Website Output Generation (Daily)

**Purpose:** Generate JSON files for website frontend

**Frequency:** Daily (after STEP 3 validation passes)

**Generated Files:**

| File | Description |
|------|-------------|
| `all_signals.json` | Complete signal list, ranked by confidence |
| `top_picks.json` | Top 20 BUY signals for homepage |
| `market_overview.json` | Signal distribution statistics |
| `{TICKER}_signals.json` | Individual ticker detailed data |

**Output Structure:**
```json
{
  "ticker": "AAPL",
  "latest_signal": "BUY",
  "confidence": 75,
  "current_price": 185.50,
  "timestamp": "2026-04-16 16:00:00",
  "timeframes": {
    "1d": { "signal": "BUY", "confidence": 75, ... },
    "1wk": { "signal": "HOLD", "confidence": 55, ... }
  }
}
```

**Destination:**
- `signalsalpha/prototype/data/signals/`

**Command:**
```bash
./run_master.sh step4
```

---

### STEP 5: Signal Validation (Daily)

**Purpose:** Ensure scored signal data integrity before backtesting

**Frequency:** Daily (after STEP 4 completes)

**Validations:**
| Check | Description |
|-------|-------------|
| Score Range | 0-100 for all scores |
| Signal Consistency | Direction matches technical/fundamental scores |
| Price Targets | stop_loss < price < take_profit |
| Coverage | All tickers have signals |
| Grade Distribution | Reasonable spread (A, B, C, D, F) |

**Exit Codes:**
- 0: Validation passed
- 1: Validation failed (critical errors)

**Outputs:**
- `data/validated/validation_report_signals_{date}.json`

**Command:**
```bash
python3 step5_signal_validator.py
```

---

### STEP 6: Backtester v2.0 (Periodic)

**Purpose:** Portfolio simulation with realistic constraints

**Frequency:** Weekly or after algorithm changes

**Portfolio Parameters:**
| Parameter | Value |
|-----------|-------|
| Initial Capital | $100,000 |
| Max Position Size | 20% ($20,000) |
| Partial Fills | Allowed |
| Cash-Constrained | Yes |
| Signal Priority | Strength-based entry |
| Short Selling | No |

**Process:**
1. Simulates real portfolio with capital constraints
2. Entry at next-day open (eliminates lookahead bias)
3. Signal reversal exits
4. Tracks daily P&L and equity curve

**Outputs:**
- `data/backtests/backtest_{timestamp}.json`

**Command:**
```bash
python3 step6_backtester.py
```

---

### STEP 7: Enhanced Metrics Module (Daily)

**Purpose:** Calculate comprehensive performance metrics from backtest results

**Location:** `modules/metrics_calculator.py`

**Metrics Calculated:**
| Category | Metrics |
|----------|---------|
| Profitability | Profit Factor, Expectancy, Payoff Ratio |
| Risk-Adjusted | Sharpe Ratio, Sortino Ratio, Calmar Ratio |
| Drawdown | Max Drawdown, Duration, Ulcer Index |
| Consistency | Consecutive Wins/Losses tracking |

**Usage:**
```python
from modules.metrics_calculator import MetricsCalculator
calc = MetricsCalculator()
metrics = calc.calculate_from_trades(trade_list)
```

---

### STEP 8: ML Feature Engineering (Daily)

**Purpose:** Generate machine learning features for optimization

**Location:** `modules/feature_engineering.py`

**Feature Categories (40+ total):**
| Category | Features |
|----------|----------|
| Price Action | Returns (5d, 20d, 60d), Price vs range |
| Volatility | Volatility (20d, 60d), ATR ratio, BB squeeze |
| Momentum | RSI slope, MACD hist slope, ROC |
| Trend | ADX strength, Higher highs/lows |
| Volume | Volume ratios, OBV slope, Money flow |
| Regime | Trending/ranging detection |
| Patterns | S/R touches, Breakout/breakdown |

**Usage:**
```python
from modules.feature_engineering import FeatureEngineer
engineer = FeatureEngineer()
features_df = engineer.calculate_all_features(price_df)
```

---

### STEP 9: Results Database (Daily)

**Purpose:** Centralized storage for all backtest results

**Location:** `modules/results_db.py`

**Tables:**
- `backtest_results` - Individual backtest runs
- `optimization_runs` - Genetic algorithm tracking
- `walk_forward_results` - Walk-forward analysis
- `config_versions` - Configuration lineage

**Features:**
- SQLite database
- A/B comparison queries
- Historical tracking
- Config versioning

**Usage:**
```python
from modules.results_db import ResultsDatabase
db = ResultsDatabase()
db.save_backtest_result(config_name, config, metrics, trades)
```

---

### STEP 10: Genetic Optimizer ⭐ (Periodic)

**Purpose:** Self-improving signal generation through genetic algorithm

**Frequency:** Weekly or monthly

**Genome Structure (14 genes):**
| Gene | Range | Description |
|------|-------|-------------|
| macd_weight | [0, 5] | MACD indicator weight |
| hma_weight | [0, 5] | HMA indicator weight |
| rsi_weight | [0, 5] | RSI indicator weight |
| stoch_weight | [0, 5] | Stochastic weight |
| sma_weight | [0, 5] | SMA weight |
| ema_weight | [0, 5] | EMA weight |
| mfi_weight | [0, 5] | MFI weight |
| bb_weight | [0, 5] | Bollinger Bands weight |
| oversold_threshold | [20, 35] | RSI oversold level |
| overbought_threshold | [65, 80] | RSI overbought level |
| adx_strong | [20, 35] | ADX strong trend threshold |
| volume_confirm | [1.0, 2.0] | Volume confirmation multiplier |
| max_position_pct | [0.10, 0.30] | Max position size |
| volatility_adj | [0, 1] | Volatility adjustment (binary) |

**Algorithm Parameters:**
| Parameter | Value |
|-----------|-------|
| Population | 30 individuals |
| Generations | 50 max |
| Elitism | 5 preserved |
| Mutation Rate | 15% |
| Early Stop | 10 generations no improvement |

**Fitness Function:**
```
Fitness = (PF × 0.35) + (Sharpe × 0.25) + (Expectancy × 0.20) + (1/DD × 0.15) + (WR × 0.05)
```

**Outputs:**
- `data/optimizer/best_config_{timestamp}.json`
- `data/optimizer/generation_log.csv`
- `data/optimizer/population_history.json`

**Command:**
```bash
python3 step10_genetic_optimizer.py
```

**Auto-Resume:**
The optimizer saves progress every 5 generations. If interrupted, restart with:
```bash
# Automatically resumes from checkpoint
python3 step10_genetic_optimizer.py
```

---

### STEP 11: Walk-Forward Validation ⭐ (Periodic)

**Purpose:** Test strategy on unseen data to prevent overfitting

**Frequency:** After genetic optimization

**Window Configuration:**
| Parameter | Value |
|-----------|-------|
| Train Window | 252 days (1 year) |
| Test Window | 20 days (1 month) |
| Step Size | 20 days |
| Min Windows | 3 |

**Consistency Checks:**
- Train/Test correlation > 0.7
- Profit Factor > 1.5 in both
- Max Drawdown < 20% in test
- Win rate within ±10%

**Overfitting Detection:**
- Flag if Train/Test PF > 1.5
- Flag if Train/Test Sharpe > 1.5

**Outputs:**
- `data/walk_forward/walkforward_{timestamp}.csv`
- `data/walk_forward/consistency_report.json`

**Command:**
```bash
python3 step11_walk_forward.py
```

---

### STEP 12: Monte Carlo Stress Testing ⭐ (Periodic)

**Purpose:** Statistical robustness through randomized simulation

**Frequency:** After walk-forward validation

**Methods:**
| Method | Description |
|--------|-------------|
| Trade Shuffling | Randomize trade sequence (10,000 iterations) |
| Parameter Perturbation | Vary weights ±10% |
| Black Swan Injection | 5% probability of extreme events |
| Market Regime Simulation | Volatile/trending/ranging markets |

**Confidence Analysis:**
| Metric | Output |
|--------|--------|
| PF Probability | Prob(PF > 1.5) at 95% CI |
| Sharpe Probability | Prob(Sharpe > 1.0) at 95% CI |
| DD Probability | Prob(DD < 20%) at 95% CI |

**Outputs:**
- `data/monte_carlo/monte_carlo_{timestamp}.csv`
- `data/monte_carlo/confidence_intervals.json`

**Command:**
```bash
python3 step12_monte_carlo.py
```

---

### STEP 13: Model Selector & Deployment ⭐ (Periodic)

**Purpose:** Select best configuration for production

**Frequency:** After all optimization steps complete

**Selection Criteria:**
| Criterion | Weight | Threshold |
|-----------|--------|-----------|
| Profit Factor | 30% | ≥ 1.5 |
| Sharpe Ratio | 25% | ≥ 1.0 |
| Walk-Forward Consistency | 20% | ≥ 3 windows |
| Monte Carlo Confidence | 15% | PF prob ≥ 0.95 |
| Max Drawdown | 10% | ≤ 20% |

**Composite Score:**
```
Score = (PF × 0.30) + (Sharpe × 0.25) + (WF × 0.20) + (MC × 0.15) + (1/DD × 0.10)
```

**Outputs:**
- `data/deployment/production_config.json` - Production-ready configuration
- `data/deployment/deployment_report.html` - Full comparison report
- `data/deployment/model_ranking.csv` - All candidates ranked

**Command:**
```bash
python3 step13_model_selector.py
```

---

### STEP 7: Walk-Forward Analyzer (Periodic)

**Purpose:** Simulate day-by-day trading with current algorithm rules for realistic expectations

**Frequency:** Monthly or before deploying algorithm changes

**Difference from Backtester:**
- Backtester: Tests on complete historical dataset
- Walk-forward: Simulates trading day-by-day (out-of-sample)

**Simulation Parameters:**
| Parameter | Value | Description |
|-----------|-------|-------------|
| Initial Capital | $100,000 | Starting portfolio value |
| Max Positions | 10 | Concurrent trades |
| Risk per Trade | 2% | Maximum loss per trade |
| Simulation Period | 30 days | Rolling window |

**Process:**
1. Simulates trading day-by-day
2. Makes decisions with only historical data available at that time
3. Manages portfolio with realistic constraints
4. Tracks daily P&L and positions

**Outputs:**
- `data/walkforward/walkforward_{timestamp}.json`

**Performance Metrics:**
- Total Return (%)
- Final Portfolio Value
- Win Rate
- Number of Trades
- Daily P&L tracking

**Command:**
```bash
python3 step7_walk_forward.py
```

## Step 13: Model Selector (Periodic)

**Purpose:** Select best configuration for production deployment

**Frequency:** After optimization completes (weekly/monthly)

**Selection Criteria:**
| Criterion | Weight | Threshold |
|-----------|--------|-----------|
| Profit Factor | 30% | ≥ 1.5 |
| Sharpe Ratio | 25% | ≥ 1.0 |
| Walk-Forward Consistency | 20% | ≥ 3 windows |
| Monte Carlo Confidence | 15% | PF prob ≥ 0.95 |
| Max Drawdown | 10% | ≤ 20% |

**Composite Score:**
```
Score = (PF × 0.30) + (Sharpe × 0.25) + (WF × 0.20) + (MC × 0.15) + (1/DD × 0.10)
```

**Outputs:**
- `data/deployment/production_config.json` - Production-ready configuration
- `data/deployment/deployment_report.html` - Full comparison report
- `data/deployment/model_ranking.csv` - All candidates ranked

**Command:**
```bash
python3 step13_model_selector.py
```

---

## Oracle Cloud Infrastructure (OCI) Considerations

### Memory Management

OCI Free Tier instances have limited RAM (1 GB). To prevent timeouts:

1. **Batch Processing:** All steps use batch processing with automatic restart
2. **Garbage Collection:** Explicit `gc.collect()` after each batch
3. **Process Exit:** Scripts exit completely between batches, freeing all memory
4. **No Memory Leaks:** Fresh Python interpreter for each batch

### Batch Restart Mechanism

The shell wrapper scripts (`run_timeseries_batch.sh`, `run_batch_restarter.sh`) handle this:

```bash
# Pseudocode
while remaining_tickers > 0:
    process_batch_of_10_or_20()
    if remaining_tickers > 0:
        exec $0  # Restart script completely
```

### Scheduling

**Recommended Crontab:**
```bash
# Weekly - Sunday 4 AM UTC (midnight ET)
0 4 * * 0 cd /path/to/process_flow_v0 && ./run_master.sh weekly >> logs/weekly_cron.log 2>&1

# Daily - 6 PM UTC (2 PM ET, after market close)
0 18 * * 1-6 cd /path/to/process_flow_v0 && ./run_master.sh daily >> logs/daily_cron.log 2>&1
```

---

## Quick Start

### Initial Setup

```bash
# 1. Clone/copy process_flow_v0 to your workspace
cp -r process_flow_v0 /home/ubuntu/.openclaw/workspace/trading-signals-data/
cd /home/ubuntu/.openclaw/workspace/trading-signals-data/process_flow_v0

# 2. Make scripts executable
chmod +x *.sh

# 3. Create necessary directories
mkdir -p logs data
```

### Running the Pipeline

```bash
# Option 1: Master script (recommended)
./run_master.sh weekly   # Full pipeline including ticker collection
./run_master.sh daily    # Daily pipeline (steps 1-4 only)

# Option 2: Individual steps
./run_master.sh step0    # Ticker collection only
./run_master.sh step1    # Time series only
./run_master.sh step2    # Scanner only
./run_master.sh step3    # Validation only
./run_master.sh step4    # Website output only

# Option 3: Direct batch wrappers (for OCI)
bash run_timeseries_batch.sh   # Step 1 with batch restart
bash run_batch_restarter.sh    # Step 2 with batch restart
```

---

## Troubleshooting

### Issue: OCI Timeout During Step 1
**Solution:** Use batch restart wrapper:
```bash
bash run_timeseries_batch.sh
```

### Issue: Memory Error During Step 2
**Solution:** Use batch restart wrapper:
```bash
bash run_batch_restarter.sh
```

### Issue: Validation Fails
**Check:**
```bash
# View validation report
cat data/validated/validation_report_*.json
```

### Issue: Missing Data Files
**Check:**
- STEP 0 completed successfully
- STEP 1 produced time series files
- STEP 2 produced signal files

---

## Dependencies

```bash
# Python packages required
pip install pandas numpy yfinance

# Optional (for development)
pip install matplotlib plotly
```

---

## Output for Website

The final website-ready outputs are in:
```
signalsalpha/prototype/data/
├── signals/
│   ├── all_signals.json          # Complete ranked list
│   ├── top_picks.json            # Top 20 for homepage
│   ├── market_overview.json      # Statistics
│   └── {TICKER}_signals.json     # Individual ticker data
├── prices/                       # Price history (if needed)
└── fundamentals/                 # Fundamentals (if implemented)
```

These JSON files are consumed by the SignalsAlpha website frontend.

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| **v2.0** | **2026-04-29** | **Backtesting Enhancement: 6-phase optimization pipeline** |
| | | - Step 10: Genetic Algorithm Optimizer (self-improving) |
| | | - Step 11: Walk-Forward Validation (overfitting prevention) |
| | | - Step 12: Monte Carlo Stress Testing (10,000 simulations) |
| | | - Step 13: Model Selector (multi-criteria ranking) |
| | | - Modules: metrics_calculator, feature_engineering, results_db |
| **v1.0** | **2026-04-18** | **Initial v1: Steps 0-9 with backtesting, validation, HMA/Elder** |
| | | - Step 1.5, 3.5, 5: Validators for data quality |
| | | - Step 2: Fundamental data collection |
| | | - Step 3/4: Technical/Scoring separation (75/25 weighting) |
| | | - Step 6: Enhanced backtester v2.0 |
| | | - Step 3: HMA 13 and Elder Impulse System added |
| v0.1 | 2026-04-16 | Initial release with 4-step pipeline |

---

## License

Proprietary - SignalsAlpha

## Contact

For issues or questions, refer to the main SignalsAlpha repository.
