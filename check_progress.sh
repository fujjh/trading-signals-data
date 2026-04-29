#!/bin/bash
# Check progress of 100-seed optimization

echo "=== 100-Seed Optimization Progress ==="
echo ""

# Check if process is running
PID=$(pgrep -f "step9_genetic_optimizer_multiseed.py" | head -1)
if [ -z "$PID" ]; then
    echo "Process not running - check multiseed_100.log for results"
    tail -50 multiseed_100.log 2>/dev/null || echo "No log file yet"
    exit 0
fi

echo "Process running (PID: $PID)"
echo ""

# Show runtime
RUNTIME=$(ps -o etime= -p $PID 2>/dev/null || echo "unknown")
echo "Runtime: $RUNTIME"
echo ""

# Check log if it exists
if [ -f multiseed_100.log ]; then
    echo "Log output (last 30 lines):"
    tail -30 multiseed_100.log
else
    echo "Log file not created yet - process still initializing"
fi

echo ""
echo "Estimated completion: ~30-45 minutes from start"
