#!/bin/bash
# =============================================================================
# STEP 4 BACKTESTER - BATCHED EXECUTION WRAPPER
# =============================================================================
#
# Purpose: Run backtester in batches to avoid OCI timeouts
#
# The backtester processes 20 tickers at a time, saves progress,
# and exits with code 99 to signal "more work needed".
# This wrapper auto-restarts until all tickers are processed.
#
# Usage:
#   bash run_step4_batched.sh
#
# Output:
#   - data/backtests/backtest_report_*.json
#   - data/backtests/trade_log_*.csv
#   - data/backtests/equity_curve_*.csv
#
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="/home/ubuntu/.openclaw/workspace/trading-signals-data/process_flow_v1/data"
BACKTEST_DIR="$DATA_DIR/backtests"
PROGRESS_FILE="$BACKTEST_DIR/backtest_progress.pkl"

echo "============================================================================"
echo "STEP 4: ENHANCED BACKTESTER v2.0"
echo "============================================================================"
echo ""
echo "This script will run the backtester in batches of 20 tickers"
echo "It will auto-restart until all tickers are processed"
echo ""
echo "Data Directory: $DATA_DIR"
echo "Output Directory: $BACKTEST_DIR"
echo ""
echo "============================================================================"

# Create output directory if it doesn't exist
mkdir -p "$BACKTEST_DIR"

BATCH_COUNT=0

while true; do
    BATCH_COUNT=$((BATCH_COUNT + 1))
    echo ""
    echo "============================================================================"
    echo "Starting batch #$BATCH_COUNT"
    echo "============================================================================"
    
    # Run the backtester
    python3 "$SCRIPT_DIR/step5_backtester.py"
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo ""
        echo "============================================================================"
        echo "✓ BACKTEST COMPLETE"
        echo "============================================================================"
        echo ""
        echo "Results saved to: $BACKTEST_DIR"
        echo ""
        # Show latest files
        echo "Latest output files:"
        ls -lt "$BACKTEST_DIR" | head -5
        echo ""
        echo "To view the report:"
        echo "  cat $BACKTEST_DIR/backtest_report_*.json | less"
        echo ""
        break
    elif [ $EXIT_CODE -eq 99 ]; then
        echo ""
        echo "Batch complete. More tickers remaining..."
        echo "Auto-restarting in 2 seconds..."
        sleep 2
    else
        echo ""
        echo "============================================================================"
        echo "✗ BACKTEST FAILED with exit code $EXIT_CODE"
        echo "============================================================================"
        echo ""
        echo "You can resume by running this script again."
        echo "Progress is saved to: $PROGRESS_FILE"
        echo ""
        exit 1
    fi
done

echo "Backtester finished successfully!"
