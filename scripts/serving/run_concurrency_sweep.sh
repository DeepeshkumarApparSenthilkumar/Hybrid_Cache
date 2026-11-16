#!/usr/bin/env bash

set -euo pipefail

MODEL="${1:-}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
LIMIT="${LIMIT:-466}"
MAX_TOKENS="${MAX_TOKENS:-16}"

CONCURRENCIES=(1 2 4 8 16 32)

if [[ "$MODEL" != "qwen" && "$MODEL" != "falcon" ]]; then
    echo "Usage: $0 {qwen|falcon}"
    exit 1
fi

echo "============================================================"
echo "CONTROLLED CONCURRENCY SWEEP"
echo "============================================================"
echo "Model       : $MODEL"
echo "Base URL    : $BASE_URL"
echo "Requests    : $LIMIT"
echo "Max tokens  : $MAX_TOKENS"
echo "Concurrency : ${CONCURRENCIES[*]}"
echo
echo "IMPORTANT:"
echo "Each concurrency level must start from an equivalent"
echo "vLLM prefix-cache state."
echo "============================================================"

for C in "${CONCURRENCIES[@]}"; do
    printf "\n"
    echo "------------------------------------------------------------"
    echo "READY FOR CONCURRENCY = $C"
    echo "------------------------------------------------------------"
    echo
    echo "Checking vLLM server..."

    if ! curl -fsS "$BASE_URL/health" >/dev/null; then
        echo "ERROR: vLLM server is not healthy at $BASE_URL"
        exit 1
    fi

    echo "Resetting local vLLM prefix cache..."

    RESET_RESPONSE=$(curl -fsS -X POST "$BASE_URL/reset_prefix_cache")

    echo "Reset response: $RESET_RESPONSE"

    RESET_OK=$(python -c 'import json,sys; print(str(json.load(sys.stdin).get("success", False)).lower())' <<< "$RESET_RESPONSE")

    if [[ "$RESET_OK" != "true" ]]; then
        echo "ERROR: Prefix-cache reset failed."
        echo "The server may still have running requests or held cache blocks."
        exit 1
    fi

    echo "Prefix cache reset successfully."
    sleep 1

    echo
    echo "Running $MODEL at concurrency $C..."

    python scripts/serving/run_controlled_load.py \
        --model "$MODEL" \
        --concurrency "$C" \
        --limit "$LIMIT" \
        --max-tokens "$MAX_TOKENS" \
        --base-url "$BASE_URL"

    RUN_DIR=$(printf "results/runtime/controlled_load/%s/c%02d" "$MODEL" "$C")

    echo
    echo "Analyzing cache metrics..."

    python scripts/analysis/analyze_cache_metrics.py \
        "$RUN_DIR"

    echo
    echo "Completed concurrency $C."
done

echo
echo "============================================================"
echo "SWEEP COMPLETE"
echo "============================================================"
