#!/usr/bin/env bash
# Monitor benchmark progress from cache files and log files.
# Usage: ./benchmarks/monitor.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

PYTHON=".venv/bin/python"
CACHE_DIR="artifacts/benchmark_results/cache"
LOG_OPENAI="artifacts/benchmark_results/run_openai.log"
LOG_GOOGLE="artifacts/benchmark_results/run_google.log"

count_cache() {
    local pattern="$1"
    find "$CACHE_DIR" -name "$pattern" 2>/dev/null | wc -l | tr -d ' '
}

summarise_log() {
    local log="$1" group="$2"
    [ -f "$log" ] || { echo "  $group: no log yet"; return; }
    local done failed current age
    done=$(grep -c "accuracy=" "$log" 2>/dev/null || true)
    failed=$(grep -c "^  ✗" "$log" 2>/dev/null || true)
    current=$(grep "── \[$group\]" "$log" 2>/dev/null | tail -1 || true)
    age=$(( $(date +%s) - $(date -r "$log" +%s) ))
    if [ "$age" -gt 300 ]; then
        echo "  $group: [STALE log — ${age}s old, may be from a previous run]"
    else
        echo "  $group: $done done  $failed failed  |  $current"
    fi
}

echo "Monitoring benchmark run. Ctrl-C to stop."
echo "Cache: $CACHE_DIR"
echo ""

while true; do
    echo "── $(date '+%H:%M:%S') ─────────────────────────────────────────────"

    # Cache file counts per model type
    LOGREG=$(count_cache "*-logreg-*.json")
    XGBOOST=$(count_cache "*-xgboost-*.json")
    TABPFN=$(count_cache "*-tabpfn-*.json")
    SKRIBE=$(count_cache "*.json")
    SKRIBE=$((SKRIBE - LOGREG - XGBOOST - TABPFN))
    echo "  cache files — logreg: $LOGREG/16  xgboost: $XGBOOST/16  tabpfn: $TABPFN/16  skribe: $SKRIBE"

    # LLM log summaries
    summarise_log "$LOG_OPENAI" "openai"
    summarise_log "$LOG_GOOGLE" "google"

    # Last few accuracy results from each log
    for log in "$LOG_OPENAI" "$LOG_GOOGLE"; do
        [ -f "$log" ] && grep "accuracy=" "$log" | tail -3 | sed 's/^/    /' || true
    done

    echo ""
    sleep 30
done
