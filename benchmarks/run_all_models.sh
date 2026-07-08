#!/usr/bin/env bash
# Orchestrate the full OpenML benchmark:
#   1. Baselines (logreg, xgboost, tabpfn) — each model loops 16 datasets sequentially
#   2. Skribe LLM variants — every (model, dataset) pair across all providers is
#      flattened into one queue; --workers workers pull whatever pair is next,
#      so a slow straggler on one model/provider never idles other workers.
# Already-cached results are skipped automatically.
# Pass --no-cache to force re-run everything.
# Pass --no-collate to skip the final collate step.
# Pass --skip-baselines to skip straight to the LLM section.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

PYTHON=".venv/bin/python"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

NO_CACHE=""
NO_COLLATE=""
BASELINES_ONLY=""
SKIP_BASELINES=""
DATASET_WORKERS=2   # parallel dataset invocations per LLM model (2 CPUs, 8 GB RAM)
EXTRA_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --no-cache)       NO_CACHE="--no-cache" ;;
        --no-collate)     NO_COLLATE="1" ;;
        --baselines-only) BASELINES_ONLY="1" ; NO_COLLATE="1" ;;
        --skip-baselines) SKIP_BASELINES="1" ;;
        --workers=*)      DATASET_WORKERS="${arg#--workers=}" ;;
        *)                EXTRA_ARGS+=("$arg") ;;
    esac
done

LOG_DIR="artifacts/benchmark_results"
mkdir -p "$LOG_DIR"

# All 16 dataset keys, derived from benchmark_utils and sorted smallest-to-largest
# by row count (all datasets are already locally cached, so this is a fast local
# read — no network calls). Smallest-first means quick feedback and lets the LLM
# queue clear its fastest work first.
DATASETS=($($PYTHON - <<'EOF'
import sys
sys.path.insert(0, "benchmarks")
from benchmark_utils import DEFAULT_DATASETS, load_dataset
sizes = []
for name, spec in DEFAULT_DATASETS.items():
    openml_name, version = spec[0], spec[1]
    csv_path = spec[2] if len(spec) > 2 else None
    target_col = spec[3] if len(spec) > 3 else None
    description = spec[4] if len(spec) > 4 else None
    X, _, _, _ = load_dataset(
        openml_name, version, None,
        csv_path=csv_path, target_col=target_col, description=description,
        require_description=False,
    )
    sizes.append((len(X), name))
sizes.sort()
for _, name in sizes:
    print(name)
EOF
))

# ---------------------------------------------------------------------------
# Baselines — sequential datasets (fast, no API calls)
# ---------------------------------------------------------------------------
if [ -z "$SKIP_BASELINES" ]; then
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
else
    echo ""
    echo "  Skipping baselines (--skip-baselines)."
fi

[ -n "$BASELINES_ONLY" ] && exit 0

# ---------------------------------------------------------------------------
# LLM variants — all providers share one flat (model × dataset) work queue
# ---------------------------------------------------------------------------

LLM_MODELS=($($PYTHON - <<'EOF'
import sys
sys.path.insert(0, "benchmarks")
from benchmark_utils import MODEL_PROGRESSION
for m in MODEL_PROGRESSION:
    if not m["model_id"].endswith("+web"):
        print(m["model_id"])
EOF
))

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Starting LLM runs  (${#LLM_MODELS[@]} models × ${#DATASETS[@]} datasets, $DATASET_WORKERS workers)"
echo "════════════════════════════════════════════════════════════════"

# Flatten the full (dataset × model) cross product into one queue, mixing
# providers together. Ordered dataset-outer / model-inner over the
# already-size-sorted DATASETS list: all models for the smallest dataset
# first, then all models for the next-smallest, etc. A slow straggler on
# one model/provider can't stall workers that could be making progress
# elsewhere — every worker just pulls whatever pair is next.
PAIRS=()
for DS in "${DATASETS[@]}"; do
    for MODEL_ID in "${LLM_MODELS[@]}"; do
        PAIRS+=("$MODEL_ID" "$DS")
    done
done

LLM_LOG="$LOG_DIR/run_llm.log"
printf '%s\n' "${PAIRS[@]}" | xargs -P "$DATASET_WORKERS" -n 2 sh -c '
    "$0" benchmarks/run_openml_fit.py --model skribe --llm "$1" --dataset "$2" '"$NO_CACHE"' '"${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"'
' "$PYTHON" 2>&1 | tee "$LLM_LOG"
LLM_RC=${PIPESTATUS[0]}

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  All models done.  exit code: $LLM_RC  (log: $LLM_LOG)"
echo "════════════════════════════════════════════════════════════════"

# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------
if [ -z "$NO_COLLATE" ]; then
    echo ""
    echo "  Running collate…"
    $PYTHON benchmarks/collate.py "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
fi

[ "$LLM_RC" -eq 0 ]
