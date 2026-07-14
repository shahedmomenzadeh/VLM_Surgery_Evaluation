#!/bin/bash
# eval_all.sh
# Automated evaluation scanner and runner for all VLM model responses

set -euo pipefail

log() {
    echo -e "\033[1;36m[eval_all.sh]\033[0m $1"
}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── 1. CONFIGURATION OVERRIDES (from eval_qwen.sh settings) ────────────────
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/results}"
JUDGE_BASE_URL="${JUDGE_BASE_URL:-http://0.0.0.0:8000/v1}"
JUDGE_MODEL="${JUDGE_MODEL:-auto}"
OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-123}"

export OPENROUTER_API_KEY

log "Output Directory: $OUTPUT_DIR"
log "Judge Base URL: $JUDGE_BASE_URL"
log "Judge Model: $JUDGE_MODEL"

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
    log "Error: Python environment not found at $QWEN_VENV."
    log "Please ensure the Qwen virtual environment has been created."
    exit 1
fi

# ── 3. SCAN FOR COMPLETED RESPONSES ────────────────────────────────────────
log "Scanning $OUTPUT_DIR for responses to grade..."

if [ ! -d "$OUTPUT_DIR" ]; then
    log "Error: Output directory $OUTPUT_DIR does not exist."
    exit 1
fi

# Find all unique tags from responses.jsonl files
# e.g., hulumed_hulu_med_7b_clip_responses.jsonl -> tag: hulumed_hulu_med_7b
declare -A tags

for file in "$OUTPUT_DIR"/*_responses.jsonl; do
    [ -e "$file" ] || continue
    filename=$(basename "$file")
    
    # Extract tag by removing suffix _clip_responses.jsonl or _full_responses.jsonl
    tag="${filename%_clip_responses.jsonl}"
    tag="${tag%_full_responses.jsonl}"
    
    tags["$tag"]=1
done

if [ ${#tags[@]} -eq 0 ]; then
    log "No response files (*_responses.jsonl) found in $OUTPUT_DIR."
    exit 0
fi

log "Found ${#tags[@]} unique model output tag(s) to evaluate."

# ── 4. EXECUTE EVALUATION ──────────────────────────────────────────────────
for tag in "${!tags[@]}"; do
    log "--------------------------------------------------"
    log "Processing model tag: $tag"
    
    # Determine model family and model ID from tag prefix
    if [[ "$tag" == hulumed_* ]]; then
        model_family="hulumed"
        # Extract everything after hulumed_
        clean_name="${tag#hulumed_}"
        model_id="hulumed/$clean_name"
    elif [[ "$tag" == qwen3vl_* ]]; then
        model_family="qwen3vl"
        # Extract everything after qwen3vl_
        clean_name="${tag#qwen3vl_}"
        model_id="Qwen/$clean_name"
    else
        log "Warning: Unknown model family prefix for tag '$tag'. Skipping."
        continue
    fi
    
    # Check data resolution availability for this model
    has_clip=false
    has_full=false
    if [ -f "$OUTPUT_DIR/${tag}_clip_responses.jsonl" ]; then
        has_clip=true
    fi
    if [ -f "$OUTPUT_DIR/${tag}_full_responses.jsonl" ]; then
        has_full=true
    fi
    
    data_level=""
    if $has_clip && $has_full; then
        data_level="both"
    elif $has_clip; then
        data_level="clip"
    elif $has_full; then
        data_level="full"
    else
        log "Warning: No response files found for tag '$tag'. Skipping."
        continue
    fi
    
    log "Model Family: $model_family"
    log "Model ID:     $model_id"
    log "Data Level:   $data_level"
    
    # Run evaluation
    "$QWEN_PYTHON" main.py \
        --mode judge \
        --model-family "$model_family" \
        --model-id "$model_id" \
        --data-level "$data_level" \
        --output-dir "$OUTPUT_DIR" \
        --judge-base-url "$JUDGE_BASE_URL" \
        --judge-model "$JUDGE_MODEL" \
        --judge-api-key-env "OPENROUTER_API_KEY"
done

log "--------------------------------------------------"
log "All evaluations completed successfully!"
