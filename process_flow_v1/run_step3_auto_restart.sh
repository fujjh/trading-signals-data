#!/bin/bash
# Auto-restart Step 3 batch processing

cd /home/ubuntu/.openclaw/workspace/trading-signals-data/process_flow_v1

RESTART_DELAY=30  # seconds to wait before restart
# No max restart limit - process until complete
RESTART_COUNT=0

while true; do
    echo "=========================================="
    echo "Step 3 Run #$((RESTART_COUNT + 1))/$MAX_RESTARTS"
    echo "=========================================="
    
    # Check current progress
    TICKERS_DONE=$(ls -d data/technical_analysis/* 2>/dev/null | wc -l)
    echo "Tickers currently processed: $TICKERS_DONE"
    echo ""
    
    # Run Step 3
    export BATCH_MODE=true
    timeout 300 python3 step3_technical_analysis.py
    EXIT_CODE=$?
    
    # Check if completed
    TICKERS_DONE=$(ls -d data/technical_analysis/* 2>/dev/null | wc -l)
    TOTAL_TICKERS=$(ls -d data/time_series/* 2>/dev/null | wc -l)
    
    echo ""
    echo "Batch finished. Processed: $TICKERS_DONE / $TOTAL_TICKERS"
    
    if [ $TICKERS_DONE -ge $TOTAL_TICKERS ]; then
        echo "=========================================="
        echo "Step 3 COMPLETE!"
        echo "=========================================="
        exit 0
    fi
    
    # Check if process was killed or timed out
    if [ $EXIT_CODE -eq 137 ] || [ $EXIT_CODE -eq 124 ]; then
        echo "Process killed/timed out. Restarting in $RESTART_DELAY seconds..."
        sleep $RESTART_DELAY
        RESTART_COUNT=$((RESTART_COUNT + 1))
    else
        echo "Process exited with code $EXIT_CODE"
        break
    fi
done

echo "Max restarts reached or process completed."
