#!/usr/bin/env bash
# LoCoMo Benchmark Runner
# Usage:
#   ./run_benchmark.sh              # Full pipeline (clean → ingest → qa → judge → report)
#   ./run_benchmark.sh --no-clean   # Skip clean, reuse existing data
#   ./run_benchmark.sh --qa-only    # Only run QA + judge + report (skip ingest)
#   ./run_benchmark.sh --judge-only # Only run judge + report on existing QA results

set -euo pipefail

# ---- Config ----
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATA_DIR="$PROJECT_DIR/.ogmem_data"
EVAL_CMD="uv run python $SCRIPT_DIR/eval.py"
REPORT_CMD="uv run python $SCRIPT_DIR/generate_report.py"
DATA_FILE="data/locomo10.json"
RESULT_DIR="$SCRIPT_DIR/result"
PARALLEL="${PARALLEL:-2}"

# ---- Determine next run number ----
next_run_number() {
    local max=0
    for d in "$RESULT_DIR"/run*/; do
        [ -d "$d" ] || continue
        local n=$(basename "$d" | sed 's/run//')
        [ "$n" -gt "$max" ] 2>/dev/null && max=$n
    done
    echo $((max + 1))
}

RUN_NUM=$(next_run_number)
RUN_NAME="run${RUN_NUM}"
RUN_DIR="$RESULT_DIR/$RUN_NAME"

# ---- Parse args ----
NO_CLEAN=false
QA_ONLY=false
JUDGE_ONLY=false
COMPARE_RUNS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --no-clean)   NO_CLEAN=true; shift ;;
        --qa-only)    QA_ONLY=true; shift ;;
        --judge-only) JUDGE_ONLY=true; shift ;;
        --compare)    COMPARE_RUNS="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--no-clean|--qa-only|--judge-only] [--compare \"run1 run2\"]"
            exit 0 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "============================================"
echo "  LoCoMo Benchmark: $RUN_NAME"
echo "============================================"

# ---- Step 0: Clean ----
if [ "$JUDGE_ONLY" = false ] && [ "$NO_CLEAN" = false ]; then
    echo ""
    echo "[0/5] Cleaning old data..."

    # Stop services
    pkill -f "agfs-server" 2>/dev/null || true
    pkill -f "ogmem.*start" 2>/dev/null || true
    sleep 2

    # Clean data
    rm -rf "$DATA_DIR/agfs" "$DATA_DIR/chroma"
    mkdir -p "$DATA_DIR/agfs/accounts" "$DATA_DIR/chroma"
    rm -f "$SCRIPT_DIR/.ingest_record.json" "$PROJECT_DIR/.ingest_record.json"
    rm -f "$DATA_DIR/ce.pid" "$DATA_DIR/agfs.pid"

    echo "  Cleaned ChromaDB, AGFS, ingest records."

    # Restart services
    echo "  Starting services..."
    cd "$PROJECT_DIR"
    uv run ogmem start --config ogmem.yaml &
    sleep 8

    # Verify health
    HEALTH=$(curl -s http://127.0.0.1:8090/api/v1/health 2>/dev/null || echo "")
    if echo "$HEALTH" | grep -q '"status":"ok"'; then
        echo "  Services OK."
    else
        echo "  ERROR: Services not healthy. Response: $HEALTH"
        exit 1
    fi
fi

# ---- Step 1: Ingest ----
if [ "$QA_ONLY" = false ] && [ "$JUDGE_ONLY" = false ]; then
    echo ""
    echo "[1/5] Running ingest..."
    cd "$SCRIPT_DIR"
    $EVAL_CMD ingest $DATA_FILE
    echo "  Ingest complete."

    INGESTED=$(python3 -c "import json; print(len(json.load(open('$SCRIPT_DIR/.ingest_record.json'))))")
    echo "  Total sessions ingested: $INGESTED"
fi

# ---- Step 2: Create run directory ----
mkdir -p "$RUN_DIR"

# ---- Step 3: QA ----
if [ "$JUDGE_ONLY" = false ]; then
    echo ""
    echo "[2/5] Running QA (parallel=$PARALLEL)..."
    cd "$SCRIPT_DIR"
    $EVAL_CMD qa $DATA_FILE --output "$RUN_NAME/qa" --parallel $PARALLEL
    QA_COUNT=$(python3 -c "import csv; print(len(list(csv.DictReader(open('$RUN_DIR/qa.csv')))))")
    echo "  QA complete. Questions: $QA_COUNT"
fi

# ---- Step 4: Judge ----
echo ""
echo "[3/5] Running judge..."
cd "$SCRIPT_DIR"
$EVAL_CMD judge "$RUN_NAME/qa.csv" --output "$RUN_NAME/qa_judged"
echo "  Judge complete."

# ---- Step 5: Stats ----
echo ""
echo "[4/5] Stats..."
cd "$SCRIPT_DIR"
$EVAL_CMD stat "$RUN_NAME/qa_judged.csv"

# ---- Step 6: Report ----
echo ""
echo "[5/5] Generating report..."

# Determine runs to compare
if [ -n "$COMPARE_RUNS" ]; then
    COMPARE_ARGS="--runs $COMPARE_RUNS $RUN_NAME"
else
    # Find previous run for comparison
    PREV=$((RUN_NUM - 1))
    if [ -d "$RESULT_DIR/run${PREV}" ]; then
        COMPARE_ARGS="--runs run${PREV} $RUN_NAME"
    else
        COMPARE_ARGS="--runs $RUN_NAME"
    fi
fi

cd "$SCRIPT_DIR"
$REPORT_CMD $COMPARE_ARGS --output "$RUN_NAME/report"
echo "  Report: $RUN_DIR/report/report.md"

# ---- Done ----
echo ""
echo "============================================"
echo "  Done! Results in: $RUN_DIR"
echo "============================================"
echo ""
ls -la "$RUN_DIR/"
