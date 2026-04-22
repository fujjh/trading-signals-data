#!/bin/bash
# Run Step 3 with batch processing to avoid OCI timeout

cd /home/ubuntu/.openclaw/workspace/trading-signals-data/process_flow_v1

echo "=========================================="
echo "Running Step 3: Technical Analysis"
echo "BATCH MODE - Will process in chunks"
echo "=========================================="
echo ""

# Run in batch mode
export BATCH_MODE=true
python3 step3_technical_analysis.py

# Check exit code
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Step 3 completed successfully!"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "Step 3 batch finished (may need restart)"
    echo "=========================================="
fi
