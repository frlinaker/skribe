#!/usr/bin/env bash
# Monitor benchmark progress from the two log files.
# Usage: ./benchmarks/monitor.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

LOG_OPENAI="artifacts/benchmark_results/run_openai.log"
LOG_GEMINI="artifacts/benchmark_results/run_gemini.log"

summarise() {
    local log="$1" group="$2"
    [ -f "$log" ] || { echo "  $group: no log yet"; return; }
    local done failed current
    done=$(grep -c "^✓ " "$log" 2>/dev/null || true)
    failed=$(grep -c "^✗ " "$log" 2>/dev/null || true)
    current=$(grep "── \[$group\] Model" "$log" 2>/dev/null | tail -1 || true)
    echo "  $group: $done done  $failed failed  |  $current"
}

echo "Monitoring benchmark run. Ctrl-C to stop."
echo "Logs: $LOG_OPENAI  |  $LOG_GEMINI"
echo ""

while true; do
    echo "── $(date '+%H:%M:%S') ─────────────────────────────────────────────"
    summarise "$LOG_OPENAI" "openai"
    summarise "$LOG_GEMINI" "gemini"
    # Last accuracy line from each log
    for log in "$LOG_OPENAI" "$LOG_GEMINI"; do
        [ -f "$log" ] && grep "accuracy=" "$log" | tail -3 | sed 's/^/    /' || true
    done
    echo ""
    sleep 60
done
