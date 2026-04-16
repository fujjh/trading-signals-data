#!/bin/bash
# =============================================================================
# SignalsAlpha Process Flow Master Script
# =============================================================================
# This script orchestrates the complete data pipeline from ticker collection
# to signal generation
# =============================================================================

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$SCRIPT_DIR/logs/process_flow_$(date +%Y%m%d_%H%M%S).log"
WORKSPACE_DIR="/home/ubuntu/.openclaw/workspace"
DATA_DIR="$WORKSPACE_DIR/data"

# Create log directory
mkdir -p "$SCRIPT_DIR/logs"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Check if virtual environment exists and activate
check_venv() {
    if [ -d "$WORKSPACE_DIR/venv" ]; then
        source "$WORKSPACE_DIR/venv/bin/activate"
        log "Virtual environment activated"
    else
        log "ERROR: Virtual environment not found at $WORKSPACE_DIR/venv"
        exit 1
    fi
}

# =============================================================================
# STEP 0: Weekly - Ticker Collection
# =============================================================================
run_step0_ticker_collection() {
    log "=========================================="
    log "STEP 0: Ticker Collection (Weekly)"
    log "=========================================="
    
    python3 "$SCRIPT_DIR/step0_ticker_collector.py" 2>&1 | tee -a "$LOG_FILE"
    
    if [ $? -eq 0 ]; then
        log "✓ Step 0 completed successfully"
    else
        log "✗ Step 0 failed"
        exit 1
    fi
}

# =============================================================================
# STEP 1: Daily - Time Series Data Collection
# =============================================================================
run_step1_time_series() {
    log "=========================================="
    log "STEP 1: Time Series Data Collection (Daily)"
    log "=========================================="
    
    # Run batch collector with restart mechanism
    log "Starting time series batch collection..."
    
    # Process in batches of 10 to prevent memory issues
    python3 "$SCRIPT_DIR/step1_time_series_collector.py" 2>&1 | tee -a "$LOG_FILE"
    
    if [ $? -eq 0 ]; then
        log "✓ Step 1 completed successfully"
    else
        log "✗ Step 1 failed"
        exit 1
    fi
}

# =============================================================================
# STEP 2: Daily - Multi-Timeframe Scanner
# =============================================================================
run_step2_scanner() {
    log "=========================================="
    log "STEP 2: Multi-Timeframe Scanner (Daily)"
    log "=========================================="
    
    python3 "$SCRIPT_DIR/step2_multi_timeframe_scanner.py" 2>&1 | tee -a "$LOG_FILE"
    
    if [ $? -eq 0 ]; then
        log "✓ Step 2 completed successfully"
    else
        log "✗ Step 2 failed"
        exit 1
    fi
}

# =============================================================================
# STEP 3: Data Validation
# =============================================================================
run_step3_validation() {
    log "=========================================="
    log "STEP 3: Data Validation"
    log "=========================================="
    
    python3 "$SCRIPT_DIR/step3_data_validator.py" 2>&1 | tee -a "$LOG_FILE"
    
    if [ $? -eq 0 ]; then
        log "✓ Step 3 completed successfully"
    else
        log "✗ Step 3 failed"
        exit 1
    fi
}

# =============================================================================
# STEP 4: Generate Website Output
# =============================================================================
run_step4_website_output() {
    log "=========================================="
    log "STEP 4: Generate Website Output"
    log "=========================================="
    
    python3 "$SCRIPT_DIR/step4_website_output_generator.py" 2>&1 | tee -a "$LOG_FILE"
    
    if [ $? -eq 0 ]; then
        log "✓ Step 4 completed successfully"
    else
        log "✗ Step 4 failed"
        exit 1
    fi
}

# =============================================================================
# BATCH PROCESSING FOR ORACLE INSTANCE
# =============================================================================
run_with_batch_restart() {
    local step=$1
    local batch_size=$2
    local script=$3
    
    log "Running $step with batch restart (batch size: $batch_size)"
    
    # Create progress tracker
    PROGRESS_FILE="$DATA_DIR/.${step}_progress"
    
    while true; do
        # Check if complete
        if [ -f "$PROGRESS_FILE.complete" ]; then
            log "$step already complete"
            rm -f "$PROGRESS_FILE.complete"
            break
        fi
        
        # Run batch
        python3 "$script" --batch-size "$batch_size" 2>&1 | tee -a "$LOG_FILE"
        
        # Check for completion marker
        if [ -f "$PROGRESS_FILE.complete" ]; then
            log "Batch processing complete"
            rm -f "$PROGRESS_FILE.complete"
            break
        fi
        
        # Memory cleanup
        log "Restarting process to free memory..."
        sleep 2
    done
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================
main() {
    log "Starting SignalsAlpha Process Flow"
    log "Log file: $LOG_FILE"
    
    check_venv
    
    # Parse command line arguments
    case "${1:-all}" in
        step0)
            run_step0_ticker_collection
            ;;
        step1)
            run_step1_time_series
            ;;
        step2)
            run_step2_scanner
            ;;
        step3)
            run_step3_validation
            ;;
        step4)
            run_step4_website_output
            ;;
        daily)
            log "Running DAILY workflow (Steps 1-4)"
            run_step1_time_series
            run_step2_scanner
            run_step3_validation
            run_step4_website_output
            ;;
        weekly)
            log "Running WEEKLY workflow (All Steps 0-4)"
            run_step0_ticker_collection
            run_step1_time_series
            run_step2_scanner
            run_step3_validation
            run_step4_website_output
            ;;
        all)
            log "Running COMPLETE workflow"
            run_step0_ticker_collection
            run_step1_time_series
            run_step2_scanner
            run_step3_validation
            run_step4_website_output
            ;;
        *)
            echo "Usage: $0 [step0|step1|step2|step3|step4|daily|weekly|all]"
            echo ""
            echo "  step0  - Weekly ticker collection"
            echo "  step1  - Daily time series collection"
            echo "  step2  - Multi-timeframe scanner"
            echo "  step3  - Data validation"
            echo "  step4  - Website output generation"
            echo "  daily  - Run Steps 1-4 (daily workflow)"
            echo "  weekly - Run Steps 0-4 (weekly workflow)"
            echo "  all    - Run complete workflow"
            exit 1
            ;;
    esac
    
    log "=========================================="
    log "Process Flow Complete!"
    log "Log saved to: $LOG_FILE"
    log "=========================================="
}

# Run main function
main "$@"
