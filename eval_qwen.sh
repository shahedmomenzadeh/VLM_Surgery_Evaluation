#!/bin/bash
# eval_qwen.sh
# Evaluation runner for Qwen3-VL-2B-Instruct offline LLM Judge grading

set -euo pipefail

log() {
    echo -e "\033[1;34m[eval_qwen.sh]\033[0m $1"
}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── 1. CONFIGURATION OVERRIDES ─────────────────────────────────────────────
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/results}"
JUDGE_BASE_URL="${JUDGE_BASE_URL:-http://0.0.0.0:8000/v1}"
JUDGE_MODEL="${JUDGE_MODEL:-auto}"
OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-123}"

# Export the key so Python's os.environ can pick it up
export OPENROUTER_API_KEY

log "Output Directory: $OUTPUT_DIR"
log "Judge Base URL: $JUDGE_BASE_URL"
log "Judge Model: $JUDGE_MODEL"

if [ -z "$OPENROUTER_API_KEY" ]; then
    log "\033[1;33mWARNING:\033[0m OPENROUTER_API_KEY is not set. API judge calls will fail or fall back."
fi

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
    else
        echo "$venv_path/bin/python"
    fi
}

QWEN_PYTHON=$(get_venv_python "$QWEN_VENV")

if [ ! -f "$QWEN_PYTHON" ]; then
    log "Error: Qwen virtual environment not found at $QWEN_VENV."
    log "Please run run_qwen.sh first to set up the environment."
    exit 1
fi

# ── 3. RUN EVALUATION ──────────────────────────────────────────────────────
log "Running LLM Judge Evaluation on Qwen3-VL-2B-Instruct responses..."
"$QWEN_PYTHON" main.py \
    --mode judge \
    --model-family qwen3vl \
    --model-id Qwen/Qwen3-VL-2B-Instruct \
    --data-level both \
    --output-dir "$OUTPUT_DIR" \
    --judge-base-url "$JUDGE_BASE_URL" \
    --judge-model "$JUDGE_MODEL" \
    --judge-api-key-env "OPENROUTER_API_KEY"
