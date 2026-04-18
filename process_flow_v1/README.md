# SignalsAlpha Process Flow v1

## Overview

This repository contains the complete data pipeline for SignalsAlpha - an automated stock signal generation system with **backtesting and performance analysis**. The pipeline runs on an Oracle Cloud Infrastructure (OCI) Free Tier instance and generates daily trading signals for a universe of US stocks.

**New in v1:**
- Signal History Tracking (Step 5)
- Backtester with Win Rate Analysis (Step 6)
- Walk-Forward Performance Simulator (Step 7)

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
STEP 2 (Daily): Multi-Timeframe Scanner
    │
    ▼
┌─────────────────────────┐
│ MultiTimeframeScanner   │──► Technical analysis on all intervals
│         v1.0            │──► Generates buy/sell signals with confidence
└─────────────────────────┘──► Outputs: data/signals_timeframe/{TICKER}/{TICKER}_signals.csv
    │
    ▼
STEP 3 (Daily): Data Validation
    │
    ▼
┌─────────────────┐
│ DataValidator   │──► Validates data integrity
│     v1.0        │──► Checks freshness and completeness
└─────────────────┘──► Outputs: data/validated/validation_report_{timestamp}.json
    │
    ▼
STEP 4 (Daily): Website Output Generation
    │
    ▼
┌──────────────────────────┐
│ WebsiteOutputGenerator   │──► Aggregates signals for frontend
│          v1.0            │──► Creates JSON files for website
└──────────────────────────┘──► Outputs: signalsalpha/prototype/data/signals/*.json
    │
    ▼
STEP 5 (Daily): Signal History Tracking
    │
    ▼
┌──────────────────────────┐
│ SignalHistoryTracker     │──► Saves daily signal snapshots
│          v1.0            │──► Tracks signal changes over time
└──────────────────────────┘──► Outputs: data/signal_history/signals_{date}.csv
    │
    ▼
STEP 6 (Periodic): Backtester
    │
    ▼
┌──────────────────────────┐
│ Backtester               │──► Tests algorithm on historical data
│          v1.0            │──► Calculates win rate, profit factor
└──────────────────────────┘──► Outputs: data/backtests/backtest_{timestamp}.json
    │
    ▼
STEP 7 (Periodic): Walk-Forward Analyzer
    │
    ▼
┌──────────────────────────┐
│ WalkForwardAnalyzer      │──► Day-by-day trading simulation
│          v1.0            │──► Realistic performance expectations
└──────────────────────────┘──► Outputs: data/walkforward/walkforward_{timestamp}.json
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
├── STEP 2 - Daily/
│   ├── step2_multi_timeframe_scanner.py # Signal generation with candlestick patterns
│   └── run_batch_restarter.sh          # Batch restart wrapper for OCI
│
├── STEP 3 - Daily/
│   └── step3_data_validator.py         # Data integrity validation
│
├── STEP 4 - Daily/
│   └── step4_website_output_generator.py # Website JSON output
│
├── STEP 5 - Daily/
│   └── step5_signal_history_tracker.py   # Signal history tracking
│
├── STEP 6 - Periodic/
│   └── step6_backtester.py               # Historical backtesting
│
├── STEP 7 - Periodic/
│   └── step7_walk_forward.py             # Walk-forward analysis
│
├── Orchestration/
│   └── run_master.sh                   # Master orchestration script
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
- Candlestick pattern: +1 to +3 (depending on pattern)

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

### STEP 5: Signal History Tracking (Daily)

**Purpose:** Track signal history over time to analyze performance and changes

**Frequency:** Daily (after STEP 4 completes)

**Process:**
1. Saves daily snapshot of all signals
2. Detects signal changes from previous day (BUY -> SELL, etc.)
3. Calculates holding periods for each signal type
4. Tracks signal distribution trends

**Outputs:**
- `data/signal_history/signals_{date}.csv` - Daily snapshots
- `data/signal_history/changes_{date}.csv` - Signal changes
- `data/signal_history/summary_{date}.json` - Historical statistics

**Metrics Tracked:**
- Signal persistence (avg holding period)
- Signal flip frequency
- Distribution changes over time

**Command:**
```bash
python3 step5_signal_history_tracker.py
```

---

### STEP 6: Backtester (Periodic)

**Purpose:** Test signal algorithm on historical data to calculate performance metrics

**Frequency:** Weekly or after algorithm changes

**Simulation Parameters:**
| Parameter | Value | Description |
|-----------|-------|-------------|
| Initial Capital | $100,000 | Starting portfolio value |
| Position Size | 5% | Capital allocated per trade |
| Stop Loss | -5% | Exit losing trades |
| Take Profit | +10% | Exit winning trades |
| Max Hold | 5 days | Time-based exit |

**Process:**
1. Runs algorithm on historical data (past 1-2 years)
2. Simulates trades based on generated signals
3. Calculates P&L for each trade
4. Aggregates performance statistics

**Outputs:**
- `data/backtests/backtest_{timestamp}.json`

**Performance Metrics:**
- Win Rate (%)
- Profit Factor (gross profits / gross losses)
- Average Return per Trade
- Total Trades
- Gross Profits/Losses

**Command:**
```bash
python3 step6_backtester.py
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
| **v1.0** | **2026-04-18** | **Added backtesting, signal history, walk-forward analysis** |
| v0.1 | 2026-04-16 | Initial release with 4-step pipeline |

---

## License

Proprietary - SignalsAlpha

## Contact

For issues or questions, refer to the main SignalsAlpha repository.
