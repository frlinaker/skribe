#!/usr/bin/env bash
# Orchestrate the full OpenML benchmark:
#   1. Baselines (logreg, xgboost, tabpfn) — each model loops 16 datasets sequentially
#   2. Skribe LLM variants — OpenAI and Google groups run in parallel;
#      within each group, models run sequentially; datasets run in parallel via xargs
# Already-cached results are skipped automatically.
# Pass --no-cache to force re-run everything.
# Pass --no-collate to skip the final collate step.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

PYTHON=".venv/bin/python"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

NO_CACHE=""
NO_COLLATE=""
BASELINES_ONLY=""
DATASET_WORKERS=2   # parallel dataset invocations per LLM model (2 CPUs, 8 GB RAM)
EXTRA_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --no-cache)       NO_CACHE="--no-cache" ;;
        --no-collate)     NO_COLLATE="1" ;;
        --baselines-only) BASELINES_ONLY="1" ; NO_COLLATE="1" ;;
        --workers=*)      DATASET_WORKERS="${arg#--workers=}" ;;
        *)                EXTRA_ARGS+=("$arg") ;;
    esac
done

LOG_DIR="artifacts/benchmark_results"
mkdir -p "$LOG_DIR"

# All 16 dataset keys, derived from benchmark_utils
DATASETS=($($PYTHON - <<'EOF'
import sys
sys.path.insert(0, "benchmarks")
from benchmark_utils import DEFAULT_DATASETS
for d in DEFAULT_DATASETS:
    print(d)
EOF
))

# ---------------------------------------------------------------------------
# Baselines — sequential datasets (fast, no API calls)
# ---------------------------------------------------------------------------
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Running baselines  (${#DATASETS[@]} datasets each)"
echo "════════════════════════════════════════════════════════════════"

for MODEL in logreg xgboost tabpfn; do
    echo ""
    echo "  ── baseline: $MODEL ──"
    for DS in "${DATASETS[@]}"; do
        $PYTHON benchmarks/run_openml_fit.py \
            --model "$MODEL" \
            --dataset "$DS" \
            $NO_CACHE \
            "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" || {
            echo "  ✗ $MODEL/$DS failed — continuing"
        }
    done
done

echo ""
echo "  Baselines done."

[ -n "$BASELINES_ONLY" ] && exit 0

# ---------------------------------------------------------------------------
# LLM variants — model IDs grouped by provider
# ---------------------------------------------------------------------------

OPENAI_MODELS=($($PYTHON - <<'EOF'
import sys
sys.path.insert(0, "benchmarks")
from benchmark_utils import MODEL_PROGRESSION
for m in MODEL_PROGRESSION:
    if m.get("provider") == "openai":
        print(m["model_id"])
EOF
))

GOOGLE_MODELS=($($PYTHON - <<'EOF'
import sys
sys.path.insert(0, "benchmarks")
from benchmark_utils import MODEL_PROGRESSION
for m in MODEL_PROGRESSION:
    if m.get("provider") == "google":
        print(m["model_id"])
EOF
))

run_llm_group() {
    local GROUP_NAME="$1"
    shift
    local MODELS=("$@")
    local TOTAL=${#MODELS[@]}
    local IDX=0

    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "  Starting LLM group: $GROUP_NAME  ($TOTAL models × ${#DATASETS[@]} datasets)"
    echo "════════════════════════════════════════════════════════════════"

    for MODEL_ID in "${MODELS[@]}"; do
        IDX=$((IDX + 1))
        echo ""
        echo "  ── [$GROUP_NAME] $IDX/$TOTAL: $MODEL_ID ──"

        # Run datasets in parallel via xargs
        printf '%s\n' "${DATASETS[@]}" | xargs -P "$DATASET_WORKERS" -I{} \
            $PYTHON benchmarks/run_openml_fit.py \
                --model skribe \
                --llm "$MODEL_ID" \
                --dataset {} \
                $NO_CACHE \
                "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" || {
            echo "  ✗ [$GROUP_NAME] $MODEL_ID had failures — continuing"
        }
    done

    echo ""
    echo "  [$GROUP_NAME] group done."
}

# Clear stale LLM logs before starting fresh runs.
rm -f "$LOG_DIR/run_openai.log" "$LOG_DIR/run_google.log"

# Run OpenAI and Google groups in parallel; log to separate files.
run_llm_group "openai" "${OPENAI_MODELS[@]}" 2>&1 | tee "$LOG_DIR/run_openai.log" &
PID_OPENAI=$!

run_llm_group "google" "${GOOGLE_MODELS[@]}" 2>&1 | tee "$LOG_DIR/run_google.log" &
PID_GOOGLE=$!

echo ""
echo "OpenAI group PID: $PID_OPENAI  (log: $LOG_DIR/run_openai.log)"
echo "Google group PID: $PID_GOOGLE  (log: $LOG_DIR/run_google.log)"
echo "Waiting for both LLM groups to finish…"

wait $PID_OPENAI && OPENAI_RC=0 || OPENAI_RC=$?
wait $PID_GOOGLE && GOOGLE_RC=0 || GOOGLE_RC=$?

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  All models done."
echo "  OpenAI exit code: $OPENAI_RC"
echo "  Google exit code: $GOOGLE_RC"
echo "════════════════════════════════════════════════════════════════"

# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------
if [ -z "$NO_COLLATE" ]; then
    echo ""
    echo "  Running collate…"
    $PYTHON benchmarks/collate.py "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
fi

[ $OPENAI_RC -eq 0 ] && [ $GOOGLE_RC -eq 0 ]
