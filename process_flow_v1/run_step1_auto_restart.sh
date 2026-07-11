#!/bin/bash
# Auto-restart Step 1 time series collector

cd /home/ubuntu/.openclaw/workspace/trading-signals-data/process_flow_v1

RESTART_DELAY=30
RESTART_COUNT=0

while true; do
    echo "=========================================="
    echo "Step 1 Run #$((RESTART_COUNT + 1))"
    echo "=========================================="
    
    # Check progress
    TICKERS_DONE=$(ls -d data/time_series/* 2>/dev/null | wc -l)
    echo "Tickers with data: $TICKERS_DONE"
    echo ""
    
    # Run Step 1
    /usr/bin/python3 step1_time_series_collector.py
    EXIT_CODE=$?
    
    echo ""
    echo "Batch finished with exit code: $EXIT_CODE"
    
    if [ $EXIT_CODE -eq 137 ] || [ $EXIT_CODE -eq 124 ] || [ $EXIT_CODE -eq 99 ] || [ $EXIT_CODE -eq 1 ]; then
        echo "Process needs restart (exit code $EXIT_CODE). Restarting in $RESTART_DELAY seconds..."
        sleep $RESTART_DELAY
        RESTART_COUNT=$((RESTART_COUNT + 1))
    else
        echo "Process completed normally."
        break
    fi
done

echo "Step 1 update complete."
