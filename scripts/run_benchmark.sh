#!/bin/bash
# Wrapper script to run benchmarks with server lifecycle management

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

# Function to start server and wait for it to be ready
start_server() {
    local strategy=$1
    echo "Starting server with prime_strategy=${strategy}..."
    python examples/transec_udp_demo.py server \
        --slot_duration 0.050 \
        --drift_window 3 \
        --prime_strategy "$strategy" \
        > /tmp/transec_server_${strategy}.log 2>&1 &
    
    SERVER_PID=$!
    echo "Server PID: $SERVER_PID"
    
    # Wait for server to start
    sleep 2
    
    # Check if server is still running
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "Error: Server failed to start"
        cat /tmp/transec_server_${strategy}.log
        exit 1
    fi
}

# Function to stop server
stop_server() {
    if [ ! -z "$SERVER_PID" ]; then
        echo "Stopping server (PID: $SERVER_PID)..."
        kill $SERVER_PID 2>/dev/null || true
        wait $SERVER_PID 2>/dev/null || true
        sleep 1
    fi
}

# Trap to ensure server is stopped on exit
trap stop_server EXIT

# Test 1: Baseline (none strategy) with drift
echo "================================"
echo "Test 1: Baseline with drift"
echo "================================"
start_server "none"
python examples/transec_udp_demo.py benchmark \
    --count 200 \
    --prime_strategy none \
    --drift_window 3 \
    --slot_duration 0.050 \
    --skew_slots 3 \
    --out baseline_drift.log
stop_server

# Test 2: Prime strategy (nearest) with drift
echo ""
echo "================================"
echo "Test 2: Prime strategy with drift"
echo "================================"
start_server "nearest"
python examples/transec_udp_demo.py benchmark \
    --count 200 \
    --prime_strategy nearest \
    --drift_window 3 \
    --slot_duration 0.050 \
    --skew_slots 3 \
    --out prime_drift.log
stop_server

echo ""
echo "Benchmarks complete!"
echo "Logs: baseline_drift.log, prime_drift.log"
