#!/usr/bin/env bash
# Run all models sequentially, one at a time, 1 worker each.
# Resumes automatically — already-cached results are skipped.
# Pass --no-cache to force re-run of everything.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

source .venv/bin/activate

EXTRA_ARGS="${@}"

MODELS=(
    "gpt-4o"
    "gpt-4o-mini"
    "gpt-4o-mini+web"
    "gpt-4.1"
    "gpt-4.1+web"
    "gpt-5.4-mini"
    "gpt-5.4-mini+web"
    "gpt-5.5"
    "gpt-5.5+web"
    "vertex_ai/gemini-2.5-flash"
    "vertex_ai/gemini-2.5-flash+web"
    "vertex_ai/gemini-2.5-pro"
    "vertex_ai/gemini-2.5-pro+web"
    "vertex_ai/gemini-2.5-flash-lite"
    "vertex_ai/gemini-2.5-flash-lite+web"
    "vertex_ai/gemini-3.5-flash"
    "vertex_ai/gemini-3.5-flash+web"
)

TOTAL=${#MODELS[@]}
IDX=0

for MODEL in "${MODELS[@]}"; do
    IDX=$((IDX + 1))
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "  Model $IDX/$TOTAL: $MODEL"
    echo "════════════════════════════════════════════════════════════════"
    python benchmarks/run_skribe.py --llm "$MODEL" --workers 1 $EXTRA_ARGS || {
        echo "  ✗ Model $MODEL failed — continuing to next model"
    }
done

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  All models done."
echo "════════════════════════════════════════════════════════════════"
