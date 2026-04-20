#!/bin/bash
# Run Steps 6, 7, 8, and 9 for backtesting analysis and website output

cd /home/ubuntu/.openclaw/workspace/trading-signals-data/process_flow_v1
source ../../venv/bin/activate

echo "=========================================="
echo "Running Steps 6-9: Backtesting + Output"
echo "=========================================="
echo ""

echo "Step 6: Backtester (this may take 10-15 minutes)"
echo "------------------------------------------"
python3 step5_backtester.py
echo ""

echo "Step 7: Walk-Forward Analyzer (this may take 5-10 minutes)"
echo "------------------------------------------"
python3 step6_walk_forward.py
echo ""

echo "Step 8: Signal History Tracker"
echo "------------------------------------------"
python3 step7_signal_history_tracker.py
echo ""

echo "Step 9: Website Output Generator"
echo "------------------------------------------"
python3 step8_website_output_generator.py
echo ""

echo "=========================================="
echo "All steps complete!"
echo "=========================================="
