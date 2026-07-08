#!/usr/bin/env bash
# Run all models with maximum parallelism:
#   - OpenAI and Gemini model groups run concurrently (different APIs/rate limits)
#   - Within each group, models run sequentially to avoid per-model rate limits
#   - Each model uses --workers 4 to parallelise across datasets
# Resumes automatically — already-cached results are skipped.
# Pass --no-cache to force re-run of everything.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

source .venv/bin/activate

EXTRA_ARGS="${@}"

OPENAI_MODELS=(
    "gpt-4o"
    "gpt-4o-mini"
    "gpt-4o-mini+web"
    "gpt-4.1"
    "gpt-4.1+web"
    "gpt-5.4-mini"
    "gpt-5.4-mini+web"
    "gpt-5.5"
    "gpt-5.5+web"
)

GEMINI_MODELS=(
    "vertex_ai/gemini-2.5-flash-lite"
    "vertex_ai/gemini-2.5-flash-lite+web"
    "vertex_ai/gemini-2.5-flash"
    "vertex_ai/gemini-2.5-flash+web"
    "vertex_ai/gemini-3.5-flash"
    "vertex_ai/gemini-3.5-flash+web"
    "vertex_ai/gemini-2.5-pro"
    "vertex_ai/gemini-2.5-pro+web"
)

run_group() {
    local GROUP_NAME="$1"
    shift
    local MODELS=("$@")
    local TOTAL=${#MODELS[@]}
    local IDX=0

    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "  Starting group: $GROUP_NAME  ($TOTAL models)"
    echo "════════════════════════════════════════════════════════════════"

    for MODEL in "${MODELS[@]}"; do
        IDX=$((IDX + 1))
        echo ""
        echo "  ── [$GROUP_NAME] Model $IDX/$TOTAL: $MODEL ──"
        python benchmarks/run_skribe.py --llm "$MODEL" --workers 4 $EXTRA_ARGS || {
            echo "  ✗ [$GROUP_NAME] $MODEL failed — continuing"
        }
    done

    echo ""
    echo "  [$GROUP_NAME] group done."
}

# Run OpenAI and Gemini groups in parallel; log to separate files.
LOG_DIR="artifacts/benchmark_results"
mkdir -p "$LOG_DIR"

run_group "openai" "${OPENAI_MODELS[@]}" 2>&1 | tee "$LOG_DIR/run_openai.log" &
PID_OPENAI=$!

run_group "gemini" "${GEMINI_MODELS[@]}" 2>&1 | tee "$LOG_DIR/run_gemini.log" &
PID_GEMINI=$!

echo "OpenAI group PID: $PID_OPENAI  (log: $LOG_DIR/run_openai.log)"
echo "Gemini group PID: $PID_GEMINI  (log: $LOG_DIR/run_gemini.log)"
echo "Waiting for both groups to finish…"

wait $PID_OPENAI && OPENAI_RC=0 || OPENAI_RC=$?
wait $PID_GEMINI && GEMINI_RC=0 || GEMINI_RC=$?

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  All models done."
echo "  OpenAI exit code: $OPENAI_RC"
echo "  Gemini exit code: $GEMINI_RC"
echo "════════════════════════════════════════════════════════════════"

[ $OPENAI_RC -eq 0 ] && [ $GEMINI_RC -eq 0 ]
