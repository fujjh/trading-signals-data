#!/bin/bash
# Run Steps 5, 6, and 7 for backtesting analysis

cd /home/ubuntu/.openclaw/workspace/trading-signals-data/process_flow_v1
source ../../venv/bin/activate

echo "=========================================="
echo "Running Steps 5-7: Backtesting Analysis"
echo "=========================================="
echo ""

echo "Step 5: Signal History Tracker"
echo "------------------------------------------"
python3 step5_signal_history_tracker.py
echo ""

echo "Step 6: Backtester (this may take 10-15 minutes)"
echo "------------------------------------------"
python3 step6_backtester.py
echo ""

echo "Step 7: Walk-Forward Analyzer (this may take 5-10 minutes)"
echo "------------------------------------------"
python3 step7_walk_forward.py
echo ""

echo "=========================================="
echo "All steps complete!"
echo "=========================================="
