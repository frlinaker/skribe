#!/usr/bin/env bash
# Monitor benchmark progress from cache files and log files.
# Usage: ./benchmarks/monitor.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

PYTHON=".venv/bin/python"
CACHE_DIR="artifacts/benchmark_results/cache"
LOG_LLM="artifacts/benchmark_results/run_llm.log"

count_cache() {
    local pattern="$1"
    find "$CACHE_DIR" -name "$pattern" 2>/dev/null | wc -l | tr -d ' '
}

summarise_log() {
    local log="$1"
    [ -f "$log" ] || { echo "  llm: no log yet"; return; }
    local done failed current age
    done=$(grep -c "accuracy=" "$log" 2>/dev/null || true)
    failed=$(grep -c "FAILED after\|^  ✗" "$log" 2>/dev/null || true)
    current=$(grep -E "\[.+ × .+\]   →" "$log" 2>/dev/null | tail -1 || true)
    age=$(( $(date +%s) - $(date -r "$log" +%s) ))
    if [ "$age" -gt 300 ]; then
        echo "  llm: [STALE log — ${age}s old, may be from a previous run]"
    else
        echo "  llm: $done done  $failed failed  |  $current"
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

    # LLM log summary
    summarise_log "$LOG_LLM"

    # Last few accuracy results from the log
    [ -f "$LOG_LLM" ] && grep "accuracy=" "$LOG_LLM" | tail -3 | sed 's/^/    /' || true

    echo ""
    sleep 30
done
