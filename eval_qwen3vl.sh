#!/bin/bash
# eval_qwen3vl.sh
# Quick evaluation test runner for Qwen3-VL models

set -euo pipefail

log() {
    echo -e "\033[1;32m[eval_qwen3vl.sh]\033[0m $1"
}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── 1. CONFIGURATION OVERRIDES ─────────────────────────────────────────────
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-VL-2B-Instruct}"
DATASET_ROOT="${DATASET_ROOT:-$SCRIPT_DIR/evaluation_dataset}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/results}"
SPLIT="${SPLIT:-Test}"
DATA_LEVEL="${DATA_LEVEL:-both}"         # clip, full, or both
MAX_FRAMES="${MAX_FRAMES:-16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
TEMPERATURE="${TEMPERATURE:-0.1}"
MODE="${MODE:-inference}"                      # all, inference, or judge
USE_FLASH="${USE_FLASH:-false}"

# LLM Judge configuration (used when MODE=all or MODE=judge)
JUDGE_BASE_URL="${JUDGE_BASE_URL:-https://api.gapgpt.app/v1}"
JUDGE_MODEL="${JUDGE_MODEL:-deepseek-v4-flash}"
PROVIDER_API_KEY="${PROVIDER_API_KEY:-}"

export PROVIDER_API_KEY
export HF_HOME="$SCRIPT_DIR/hf_cache"

log "=================================================="
log "  Qwen3-VL Evaluation Test Runner"
log "=================================================="
log "Model ID:       $MODEL_ID"
log "Dataset Root:   $DATASET_ROOT"
log "Output Dir:     $OUTPUT_DIR"
log "Split:          $SPLIT"
log "Data Level:     $DATA_LEVEL"
log "Max Frames:     $MAX_FRAMES"
log "Execution Mode: $MODE"
log "Flash-Attn 2:   $USE_FLASH"
log "Judge Model:    $JUDGE_MODEL"
log "=================================================="

# ── 2. LOCATE PYTHON ENVIRONMENT ───────────────────────────────────────────
QWEN_VENV="$SCRIPT_DIR/.venv-qwen3vl"

get_venv_python() {
    local venv_path="$1"
    if [ -f "$venv_path/Scripts/python.exe" ]; then
        echo "$venv_path/Scripts/python.exe"
    elif [ -f "$venv_path/Scripts/python" ]; then
        echo "$venv_path/Scripts/python"
    elif [ -f "$venv_path/bin/python" ]; then
        echo "$venv_path/bin/python"
    elif command -v python3 &>/dev/null; then
        echo "python3"
    else
        echo "python"
    fi
}

QWEN_PYTHON=$(get_venv_python "$QWEN_VENV")
log "Using Python interpreter: $QWEN_PYTHON"

# ── 3. FLASH ATTENTION CHECK ────────────────────────────────────────────────
FLASH_ARG=""
if [ "$USE_FLASH" = "true" ]; then
    log "Checking / installing flash-attn for Qwen3-VL..."
    if command -v uv &>/dev/null; then
        uv pip install --python "$QWEN_PYTHON" --no-build-isolation "flash-attn>=2.6.0"
    else
        "$QWEN_PYTHON" -m pip install -q --no-build-isolation "flash-attn>=2.6.0"
    fi
    FLASH_ARG="--use-flash-attn"
fi

# ── 4. RUN DRY-RUN VERIFICATION (Optional Quick Check) ──────────────────────
if [[ "${1:-}" == "--dry-run" ]]; then
    log "Running dataset & pipeline verification (dry-run)..."
    "$QWEN_PYTHON" main.py \
        --dry-run \
        --dataset-root "$DATASET_ROOT" \
        --splits "$SPLIT" \
        --data-level "$DATA_LEVEL"
    log "Dry-run verification completed successfully!"
    exit 0
fi

# ── 5. EXECUTE EVALUATION ──────────────────────────────────────────────────
log "Starting Qwen3-VL evaluation run..."

"$QWEN_PYTHON" main.py \
    --mode "$MODE" \
    --model-family qwen3vl \
    --model-id "$MODEL_ID" \
    --dataset-root "$DATASET_ROOT" \
    --splits "$SPLIT" \
    --data-level "$DATA_LEVEL" \
    --output-dir "$OUTPUT_DIR" \
    --max-frames "$MAX_FRAMES" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE" \
    --judge-base-url "$JUDGE_BASE_URL" \
    --judge-model "$JUDGE_MODEL" \
    --judge-api-key-env "PROVIDER_API_KEY" \
    $FLASH_ARG

log "=================================================="
log "Qwen3-VL evaluation completed!"
log "Results saved in: $OUTPUT_DIR"
log "=================================================="
