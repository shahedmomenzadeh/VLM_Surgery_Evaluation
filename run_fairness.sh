#!/bin/bash
# run_fairness.sh
# Automated LLM-Judge Fairness and Stability Experiment Runner
#
# Runs fairness_experiment.py k times on identical model responses to quantify
# judge scoring stability, score variance, and consistency across runs.
#
# Usage:
#   # Run on a single model tag (e.g. qwen3vl_qwen3_vl_2b_instruct) with k=3
#   bash run_fairness.sh --tag qwen3vl_qwen3_vl_2b_instruct --k 3
#
#   # Auto-scan and run fairness on all completed models in ./results
#   bash run_fairness.sh --k 3 --output-dir ./results
#
#   # Custom judge model and endpoint
#   bash run_fairness.sh --tag qwen3vl_qwen3_vl_2b_instruct --judge-base-url http://localhost:8000/v1 --judge-model qwen3.8-max

set -euo pipefail

log() {
    echo -e "\033[1;36m[run_fairness.sh]\033[0m $1"
}

warn() {
    echo -e "\033[1;33m[run_fairness.sh:WARN]\033[0m $1"
}

err() {
    echo -e "\033[1;31m[run_fairness.sh:ERROR]\033[0m $1"
}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── 1. CONFIGURATION DEFAULTS ──────────────────────────────────────────────
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/results}"
JUDGE_BASE_URL="${JUDGE_BASE_URL:-http://localhost:8000/v1}"
JUDGE_MODEL="${JUDGE_MODEL:-qwen3.8-max}"
PROVIDER_API_KEY="${PROVIDER_API_KEY:-}"
JUDGE_API_KEY_ENV="${JUDGE_API_KEY_ENV:-PROVIDER_API_KEY}"
K="${K:-3}"
DELAY="${DELAY:-1.0}"
JUDGE_RETRIES="${JUDGE_RETRIES:-3}"
TAG="${TAG:-}"
SKIP_CLIP=false
SKIP_NARRATION=false
NUM_WORKERS="${NUM_WORKERS:-1}"

# Load .env if present (strip Windows CR line endings)
if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a
    while IFS= read -r line; do
        line="${line%$'\r'}"
        [[ "$line" =~ ^[A-Za-z_]+= ]] && export "$line"
    done < "$SCRIPT_DIR/.env"
    set +a
fi

# Parse optional command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -t|--tag)
            TAG="$2"
            shift 2
            ;;
        -k|--k)
            K="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --judge-base-url)
            JUDGE_BASE_URL="$2"
            shift 2
            ;;
        --judge-model)
            JUDGE_MODEL="$2"
            shift 2
            ;;
        --judge-api-key-env)
            JUDGE_API_KEY_ENV="$2"
            shift 2
            ;;
        --delay)
            DELAY="$2"
            shift 2
            ;;
        --judge-retries)
            JUDGE_RETRIES="$2"
            shift 2
            ;;
        --skip-clip)
            SKIP_CLIP=true
            shift 1
            ;;
        --skip-narration)
            SKIP_NARRATION=true
            shift 1
            ;;
        -w|--num-workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: bash run_fairness.sh [options]"
            echo ""
            echo "Options:"
            echo "  -t, --tag <tag>            Model tag (e.g. qwen3vl_qwen3_vl_2b_instruct). If omitted, scans results dir."
            echo "  -k, --k <num>              Number of repeated judge runs per response (default: 3)"
            echo "  --output-dir <dir>         Directory containing model response JSONL files (default: ./results)"
            echo "  --judge-base-url <url>     LLM judge OpenAI-compatible base URL (default: http://localhost:8000/v1)"
            echo "  --judge-model <model>      LLM judge model identifier (default: qwen3.8-max)"
            echo "  --judge-api-key-env <var>  Environment variable name for judge API key (default: PROVIDER_API_KEY)"
            echo "  --delay <sec>              Delay in seconds between judge API calls (default: 1.0)"
            echo "  --judge-retries <num>      Number of API retry attempts on failure (default: 3)"
            echo "  --skip-clip                Skip clip MCQ scoring evaluation"
            echo "  --skip-narration           Skip full-video narration scoring evaluation"
            echo "  -w, --num-workers <N>      Number of concurrent model workers for multiple models (default: 1)"
            echo "  -h, --help                 Show this help message and exit"
            exit 0
            ;;
        *)
            err "Unknown argument: $1"
            exit 1
            ;;
    esac
done

export PROVIDER_API_KEY

log "=================================================="
log "  LLM-Judge Fairness Experiment Runner"
log "=================================================="
log "Target Tag:          ${TAG:-[Auto-scan all available]}"
log "Repetitions (k):     $K"
log "Output Directory:    $OUTPUT_DIR"
log "Judge Base URL:      $JUDGE_BASE_URL"
log "Judge Model:         $JUDGE_MODEL"
log "Judge API Key Env:   $JUDGE_API_KEY_ENV"
log "Call Delay:          ${DELAY}s"
log "Retries:             $JUDGE_RETRIES"
log "Skip Clip:           $SKIP_CLIP"
log "Skip Narration:      $SKIP_NARRATION"
log "Concurrent Workers:  $NUM_WORKERS"
log "=================================================="

# ── 2. LOCATE PYTHON ENVIRONMENT ───────────────────────────────────────────
get_venv_python() {
    local venv_path="$1"
    if [ -f "$venv_path/Scripts/python.exe" ]; then
        echo "$venv_path/Scripts/python.exe"
    elif [ -f "$venv_path/Scripts/python" ]; then
        echo "$venv_path/Scripts/python"
    elif [ -f "$venv_path/bin/python" ]; then
        echo "$venv_path/bin/python"
    else
        echo ""
    fi
}

PYTHON_BIN=""
for candidate in "$SCRIPT_DIR/.venv-qwen3vl" "$SCRIPT_DIR/.venv-hulumed" "$SCRIPT_DIR/.venv-magevl"; do
    if [ -d "$candidate" ]; then
        found=$(get_venv_python "$candidate")
        if [ -n "$found" ] && [ -f "$found" ]; then
            PYTHON_BIN="$found"
            break
        fi
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    if command -v python3 &>/dev/null; then
        PYTHON_BIN="python3"
    elif command -v python &>/dev/null; then
        PYTHON_BIN="python"
    else
        err "No Python interpreter found."
        exit 1
    fi
fi

log "Using Python interpreter: $PYTHON_BIN"

# ── 3. SCAN & VALIDATE TARGET MODEL(S) ─────────────────────────────────────
if [ ! -d "$OUTPUT_DIR" ]; then
    err "Output directory '$OUTPUT_DIR' does not exist."
    exit 1
fi

TARGET_TAGS=()

if [ -n "$TAG" ]; then
    clip_file="$OUTPUT_DIR/${TAG}_clip_responses.jsonl"
    full_file="$OUTPUT_DIR/${TAG}_full_responses.jsonl"
    if [ ! -f "$clip_file" ] && [ ! -f "$full_file" ]; then
        err "No response files found for tag '$TAG' in $OUTPUT_DIR"
        err "Expected '$clip_file' or '$full_file'"
        exit 1
    fi
    TARGET_TAGS+=("$TAG")
else
    declare -A unique_tags
    for file in "$OUTPUT_DIR"/*_responses.jsonl; do
        [ -e "$file" ] || continue
        filename=$(basename "$file")
        tag="${filename%_clip_responses.jsonl}"
        tag="${tag%_full_responses.jsonl}"
        unique_tags["$tag"]=1
    done

    if [ ${#unique_tags[@]} -eq 0 ]; then
        warn "No response files (*_responses.jsonl) found in $OUTPUT_DIR."
        exit 0
    fi

    for tag in "${!unique_tags[@]}"; do
        TARGET_TAGS+=("$tag")
    done
fi

log "Found ${#TARGET_TAGS[@]} model tag(s) to process for fairness evaluation:"
for t in "${TARGET_TAGS[@]}"; do
    clip_count=0
    full_count=0
    [ -f "$OUTPUT_DIR/${t}_clip_responses.jsonl" ] && clip_count=$(wc -l < "$OUTPUT_DIR/${t}_clip_responses.jsonl" 2>/dev/null || echo 0)
    [ -f "$OUTPUT_DIR/${t}_full_responses.jsonl" ] && full_count=$(wc -l < "$OUTPUT_DIR/${t}_full_responses.jsonl" 2>/dev/null || echo 0)
    log "  - $t (Clip: $clip_count records, Full: $full_count records)"
done

# ── 4. EXECUTE FAIRNESS EXPERIMENT ─────────────────────────────────────────
EXTRA_ARGS=()
if [ "$SKIP_CLIP" = "true" ]; then
    EXTRA_ARGS+=("--skip-clip")
fi
if [ "$SKIP_NARRATION" = "true" ]; then
    EXTRA_ARGS+=("--skip-narration")
fi

run_single_fairness() {
    local tag="$1"
    local log_file="$OUTPUT_DIR/fairness_${tag}.log"
    local is_background="${2:-false}"

    log "[Running Fairness] Model: $tag (k=$K, workers=$NUM_WORKERS)..."

    if [ "$is_background" = "true" ]; then
        # Background mode for concurrent multi-model runs
        if "$PYTHON_BIN" "$SCRIPT_DIR/fairness_experiment.py" \
            --tag "$tag" \
            --k "$K" \
            --num-workers "$NUM_WORKERS" \
            --output-dir "$OUTPUT_DIR" \
            --judge-base-url "$JUDGE_BASE_URL" \
            --judge-model "$JUDGE_MODEL" \
            --judge-api-key-env "$JUDGE_API_KEY_ENV" \
            --judge-retries "$JUDGE_RETRIES" \
            --delay "$DELAY" \
            ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} > "$log_file" 2>&1; then
            log "[Finished] Successfully ran fairness for: $tag (Log: $log_file)"
        else
            err "[Failed] Fairness run encountered errors for: $tag (Check log: $log_file)"
            return 1
        fi
    else
        # Foreground mode with live interactive progress
        if "$PYTHON_BIN" "$SCRIPT_DIR/fairness_experiment.py" \
            --tag "$tag" \
            --k "$K" \
            --num-workers "$NUM_WORKERS" \
            --output-dir "$OUTPUT_DIR" \
            --judge-base-url "$JUDGE_BASE_URL" \
            --judge-model "$JUDGE_MODEL" \
            --judge-api-key-env "$JUDGE_API_KEY_ENV" \
            --judge-retries "$JUDGE_RETRIES" \
            --delay "$DELAY" \
            ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}; then
            log "[Finished] Successfully completed fairness evaluation for: $tag"
        else
            err "[Failed] Fairness evaluation failed for: $tag"
            return 1
        fi
    fi
}

pids=()
failed=0

if [ "$NUM_WORKERS" -gt 1 ] && [ ${#TARGET_TAGS[@]} -gt 1 ]; then
    log "Starting concurrent fairness evaluations across ${#TARGET_TAGS[@]} models (Process concurrency: $NUM_WORKERS)..."
    for tag in "${TARGET_TAGS[@]}"; do
        while [ "$(jobs -rp | wc -l)" -ge "$NUM_WORKERS" ]; do
            sleep 2
        done
        run_single_fairness "$tag" "true" &
        pids+=($!)
    done

    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            failed=$((failed + 1))
        fi
    done
else
    for tag in "${TARGET_TAGS[@]}"; do
        if ! run_single_fairness "$tag"; then
            failed=$((failed + 1))
        fi
    done
fi

echo ""
log "=================================================="
if [ "$failed" -eq 0 ]; then
    log "All ${#TARGET_TAGS[@]} fairness evaluation(s) completed successfully!"
else
    warn "$failed model fairness evaluation(s) encountered issues. Check logs in $OUTPUT_DIR."
fi
log "Fairness summaries and scores saved in $OUTPUT_DIR:"
for tag in "${TARGET_TAGS[@]}"; do
    log "  - $OUTPUT_DIR/${tag}_fairness_summary.json"
done
log "=================================================="
