#!/bin/bash
# Run Step 8: Signal History Tracker only

cd /home/ubuntu/.openclaw/workspace/trading-signals-data/process_flow_v1
source ../../venv/bin/activate

echo "=========================================="
echo "Running Step 8: Signal History Tracker"
echo "=========================================="
python3 step5_signal_history_tracker.py
echo "Done!"
