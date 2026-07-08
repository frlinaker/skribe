#!/usr/bin/env bash
# Benchmark AdaptiveSkribeEngineer (AFE): for every dataset in
# benchmarks/config.yaml, fit logreg and xgboost both with and without AFE
# applied first, via run_openml_fit.py --fe-model. Already-cached results are
# skipped automatically. Then run plot_afe_lift.py to produce the delta table
# and chart.
#
# Usage:
#   benchmarks/run_afe_benchmark.sh
#   benchmarks/run_afe_benchmark.sh --fe-model=gpt-5.5
#   benchmarks/run_afe_benchmark.sh --no-cache
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

PYTHON=".venv/bin/python"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

NO_CACHE=""
FE_MODEL=""
EXTRA_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --no-cache)    NO_CACHE="--no-cache" ;;
        --fe-model=*)  FE_MODEL="${arg#--fe-model=}" ;;
        *)             EXTRA_ARGS+=("$arg") ;;
    esac
done

# Default FE model: latest base (non-"+web") OpenAI model in config.yaml,
# same choice run_adaptive_fe_benchmark.py used to make automatically.
if [ -z "$FE_MODEL" ]; then
    FE_MODEL=$($PYTHON - <<'EOF'
import sys
sys.path.insert(0, "benchmarks")
from benchmark_utils import MODEL_PROGRESSION
openai_models = [m for m in MODEL_PROGRESSION if m["provider"] == "openai" and not m.get("web_search")]
print(max(openai_models, key=lambda m: m["release_date"])["model_id"])
EOF
)
fi

DATASETS=($($PYTHON - <<'EOF'
import sys
sys.path.insert(0, "benchmarks")
from benchmark_utils import DEFAULT_DATASETS
for name in DEFAULT_DATASETS:
    print(name)
EOF
))

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  AFE benchmark — fe-model=$FE_MODEL  (${#DATASETS[@]} datasets × {logreg, xgboost} × {with, without})"
echo "════════════════════════════════════════════════════════════════"

for DS in "${DATASETS[@]}"; do
    for MODEL in logreg xgboost; do
        # Without AFE (baseline)
        $PYTHON benchmarks/run_openml_fit.py \
            --model "$MODEL" --dataset "$DS" \
            $NO_CACHE "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" || {
            echo "  ✗ $MODEL/$DS (no FE) failed — continuing"
        }
        # With AFE
        $PYTHON benchmarks/run_openml_fit.py \
            --model "$MODEL" --dataset "$DS" --fe-model "$FE_MODEL" \
            $NO_CACHE "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" || {
            echo "  ✗ $MODEL/$DS (+FE) failed — continuing"
        }
    done
done

echo ""
echo "  AFE benchmark runs done. Plotting…"
$PYTHON benchmarks/plot_afe_lift.py --fe-model "$FE_MODEL"
