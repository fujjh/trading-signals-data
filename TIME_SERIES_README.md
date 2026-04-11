# Time Series Data Collector

## Overview
Multi-interval OHLCV data collector using yfinance. Fetches stock price data at multiple time resolutions for candlestick charting and technical analysis.

## Intervals Collected

| Interval | Period | Rows (AAPL example) |
|----------|--------|---------------------|
| 1m | 7 days | 2,730 |
| 2m | 60 days | 6,825 |
| 5m | 60 days | 4,557 |
| 15m | 60 days | 1,521 |
| 30m | 60 days | 761 |
| 60m | 730 days | 5,069 |
| 1h | 730 days | 5,069 |
| 1d | max | 11,423 |
| 1wk | max | 2,366 |
| 1mo | max | 496 |

## Files

- `time_series_collector.py` - Main collector script
- `run_timeseries_batch.sh` - Self-restarting batch processor
- `time_series/{TICKER}/{TICKER}_{interval}.csv` - Data files

## CSV Format

```csv
datetime,open,high,low,close,volume,dividends,stock_splits,ticker,interval
2026-04-01 09:30:00-04:00,223.52,223.65,223.46,223.52,1197303,0.0,0.0,AAPL,1m
```

## Usage

### Test with single stock
```bash
python3 time_series_collector.py --test AAPL
```

### Run full collection
```bash
bash run_timeseries_batch.sh
```

## Progress

- **Completed**: 255 stocks (all 10 intervals each)
- **Remaining**: 3,228 stocks
- **Status**: Paused for optimization review

## Sample Data Available

- AAPL
- MSFT
- GOOGL
- TSLA
- NVDA

Each has 10 CSV files (one per interval) with complete OHLCV data.
