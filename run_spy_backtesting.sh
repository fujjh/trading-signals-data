#!/bin/bash
#================================================================================
# Run SPY Backtesting & Optimization (Steps 7-13)
#================================================================================

echo "=========================================="
echo "SPY Backtesting & Optimization"
echo "Steps 7-13"
echo "=========================================="
echo ""

# Ensure we're in the right directory
cd "$(dirname "$0")"

# Check if core pipeline has been run
if [ ! -f "data/signals_scored/spy_scored_*.csv" ]; then
    echo "WARNING: No signals found. Run core pipeline first:"
    echo "  bash run_spy_pipeline.sh"
    echo ""
fi

# Step 7: Signal History Tracker
echo "Step 7: Signal History Tracker..."
python3 step7_signal_history_tracker.py
if [ $? -ne 0 ]; then
    echo "WARNING: Step 7 failed (continuing)"
fi
echo ""

# Step 8: Website Output Generator
echo "Step 8: Website Output Generator..."
python3 step8_website_output_generator.py
if [ $? -ne 0 ]; then
    echo "WARNING: Step 8 failed (continuing)"
fi
echo ""

# Step 9: Genetic Optimizer
echo "Step 9: Genetic Optimizer..."
echo "(This may take several minutes)"
python3 step9_genetic_optimizer.py
if [ $? -ne 0 ]; then
    echo "WARNING: Step 9 failed (continuing)"
fi
echo ""

# Step 10: Historical Backtester
echo "Step 10: Historical Backtester..."
python3 step10_backtester.py
if [ $? -ne 0 ]; then
    echo "WARNING: Step 10 failed (continuing)"
fi
echo ""

# Step 11: Walk-Forward Analysis
echo "Step 11: Walk-Forward Analysis..."
python3 step11_walk_forward.py
if [ $? -ne 0 ]; then
    echo "WARNING: Step 11 failed (continuing)"
fi
echo ""

# Step 12: Monte Carlo Stress Testing
echo "Step 12: Monte Carlo Stress Testing..."
echo "(This may take a few minutes)"
python3 step12_monte_carlo.py
if [ $? -ne 0 ]; then
    echo "WARNING: Step 12 failed (continuing)"
fi
echo ""

# Step 13: Model Selector
echo "Step 13: Model Selector..."
python3 step13_model_selector.py
if [ $? -ne 0 ]; then
    echo "WARNING: Step 13 failed (continuing)"
fi
echo ""

echo "=========================================="
echo "Backtesting & Optimization Complete!"
echo "=========================================="
echo ""
echo "Output files:"
echo "  - data/signal_history/spy_*.csv"
echo "  - data/website_output/spy_*.json"
echo "  - data/optimizer/spy_best_config_*.json"
echo "  - data/backtests/spy_backtest_*.json"
echo "  - data/walk_forward/spy_walk_forward_*.json"
echo "  - data/monte_carlo/spy_monte_carlo_*.json"
echo "  - data/production_config/spy_production_config.json"
