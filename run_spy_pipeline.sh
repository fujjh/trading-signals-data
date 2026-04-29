#!/bin/bash
#================================================================================
# Run SPY Pipeline (Simplified)
# Sequential execution for SPY technical analysis only
#================================================================================

echo "=========================================="
echo "SPY Trading Signal Pipeline"
echo "Technical Analysis Only"
echo "=========================================="
echo ""

# Ensure we're in the right directory
cd "$(dirname "$0")"

# Step 1: Collect daily data (incremental)
echo "Step 1: Collecting SPY daily data..."
python3 step1_time_series_collector.py
if [ $? -ne 0 ]; then
    echo "ERROR: Step 1 failed"
    exit 1
fi
echo ""

# Step 2: Validate time series
echo "Step 2: Validating time series data..."
python3 step2_time_series_validator.py
if [ $? -ne 0 ]; then
    echo "ERROR: Step 2 failed"
    exit 1
fi
echo ""

# Step 3: Technical analysis
echo "Step 3: Generating technical analysis..."
python3 step3_technical_analysis.py
if [ $? -ne 0 ]; then
    echo "ERROR: Step 3 failed"
    exit 1
fi
echo ""

# Step 4: Validate technical
echo "Step 4: Validating technical analysis..."
python3 step4_technical_validator.py
if [ $? -ne 0 ]; then
    echo "ERROR: Step 4 failed"
    exit 1
fi
echo ""

# Step 5: Score signals
echo "Step 5: Scoring and ranking..."
python3 step5_scoring_ranking.py
if [ $? -ne 0 ]; then
    echo "ERROR: Step 5 failed"
    exit 1
fi
echo ""

# Step 6: Validate signals
echo "Step 6: Validating signals..."
python3 step6_signal_validator.py
if [ $? -ne 0 ]; then
    echo "ERROR: Step 6 failed"
    exit 1
fi
echo ""

# Steps 7-13 are optional and run separately via run_spy_backtesting.sh
echo "=========================================="
echo "Core Pipeline Complete (Steps 1-6)"
echo "=========================================="
echo ""
echo "For backtesting and optimization (Steps 7-13), run:"
echo "  bash run_spy_backtesting.sh"
echo ""
echo "Output files:"
echo "  - data/signals_scored/spy_scored_*.csv"
echo "  - data/validated/validation_report_signals_spy_*.json"
