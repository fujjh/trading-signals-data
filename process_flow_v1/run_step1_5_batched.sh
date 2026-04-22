#!/bin/bash
# Run Step 1.5 with batch processing to avoid OCI timeout

cd /home/ubuntu/.openclaw/workspace/trading-signals-data/process_flow_v1

# Create progress directory if needed
mkdir -p data/validated

echo "=========================================="
echo "Running Step 1.5: Time Series Validator"
echo "BATCH MODE - Will process in chunks"
echo "=========================================="
echo ""

# Run in batch mode
export BATCH_MODE=true
python3 step1_5_time_series_validator.py

# Check exit code
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Step 1.5 completed successfully!"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "Step 1.5 batch finished (may need restart)"
    echo "=========================================="
fi
