#!/bin/bash
# eval_all.sh
# Automated evaluation scanner and concurrent judge runner for completed VLM model responses
#
# Usage:
#   bash eval_all.sh [--num-workers 3] [--output-dir ./results] [--judge-model combo]

set -euo pipefail

log() {
    echo -e "\033[1;36m[eval_all.sh]\033[0m $1"
}

warn() {
    echo -e "\033[1;33m[eval_all.sh:WARN]\033[0m $1"
}

err() {
    echo -e "\033[1;31m[eval_all.sh:ERROR]\033[0m $1"
}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── SIGNAL HANDLING ─────────────────────────────────────────────────────────
# Ctrl+C / Ctrl+Z / terminate: kill every spawned worker so no orphan judge
# processes survive (a plain Ctrl+Z only *freezes* the process group).
cleanup_jobs() {
    err "Signal caught — stopping evaluation and terminating all workers..."
    local child
    for child in $(jobs -p); do
        kill -9 "$child" 2>/dev/null || true
    done
    pkill -9 -P $$ 2>/dev/null || true
    kill -9 -$$ 2>/dev/null || true   # fallback: kill the whole process group
    exit 130
}
trap cleanup_jobs INT TERM HUP TSTP

# ── 1. CONFIGURATION DEFAULTS ──────────────────────────────────────────────
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/results}"
JUDGE_BASE_URL="${JUDGE_BASE_URL:-https://opencode.ai/zen/go/v1/responses}"
JUDGE_MODEL="${JUDGE_MODEL:-muse-spark-1.3-contributor}"
PROVIDER_API_KEY="${PROVIDER_API_KEY:-}"
JUDGE_API_KEY_ENV="${JUDGE_API_KEY_ENV:-PROVIDER_API_KEY}"
NUM_WORKERS="${NUM_WORKERS:-1}"

# Load .env if present (strip Windows CR line endings AND surrounding quotes)
if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a
    while IFS= read -r line; do
        line="${line%$'\r'}"
        [[ "$line" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]] || continue
        key="${line%%=*}"
        val="${line#*=}"
        val="${val#\"}"; val="${val%\"}"
        val="${val#\'}"; val="${val%\'}"
        export "$key=$val"
    done < "$SCRIPT_DIR/.env"
    set +a
fi

# Expected test dataset thresholds for complete runs
# 989 clip-level records (293 visual_description + 486 mcq + 210 phase),
# each producing ONE response (explanation format, no CoT/direct variants).
EXPECTED_CLIP_COUNT=989
EXPECTED_FULL_COUNT=15

# Parse optional command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -w|--num-workers)
            NUM_WORKERS="$2"
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
        -h|--help)
            echo "Usage: bash eval_all.sh [options]"
            echo "Options:"
            echo "  -w, --num-workers <N>      Number of concurrent model evaluation workers (default: 3)"
            echo "  --output-dir <dir>         Directory containing model response JSONL files"
            echo "  --judge-base-url <url>     LLM judge OpenAI-compatible base URL"
            echo "  --judge-model <model>      LLM judge model identifier"
            echo "  --judge-api-key-env <var>  Environment variable name for judge API key"
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
log "  VLM Model Responses Evaluation Scanner"
log "=================================================="
log "Output Directory:    $OUTPUT_DIR"
log "Judge Base URL:      $JUDGE_BASE_URL"
log "Judge Model:         $JUDGE_MODEL"
log "Concurrent Workers:  $NUM_WORKERS"
log "Target Requirements: Clip >= $EXPECTED_CLIP_COUNT lines, Full-Video >= $EXPECTED_FULL_COUNT lines"
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
    else
        echo "python"
    fi
}

QWEN_PYTHON=$(get_venv_python "$QWEN_VENV")

# Fallback to system python if venv python doesn't exist
if [ ! -f "$QWEN_PYTHON" ] && ! command -v "$QWEN_PYTHON" &>/dev/null; then
    QWEN_PYTHON="python"
fi

log "Using Python: $QWEN_PYTHON"

# ── 3. SCAN & VALIDATE COMPLETED RESPONSES ─────────────────────────────────
if [ ! -d "$OUTPUT_DIR" ]; then
    err "Output directory '$OUTPUT_DIR' does not exist."
    exit 1
fi

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

log "Found ${#unique_tags[@]} unique model tag(s) in output directory. Inspecting completeness..."

# Arrays to hold eligible finished models
FINISHED_MODELS=()
INCOMPLETE_MODELS=()

for tag in "${!unique_tags[@]}"; do
    clip_file="$OUTPUT_DIR/${tag}_clip_responses.jsonl"
    full_file="$OUTPUT_DIR/${tag}_full_responses.jsonl"
    
    clip_count=0
    full_count=0
    
    if [ -f "$clip_file" ]; then
        clip_count=$(wc -l < "$clip_file" 2>/dev/null || echo 0)
    fi
    if [ -f "$full_file" ]; then
        full_count=$(wc -l < "$full_file" 2>/dev/null || echo 0)
    fi

    # Determine model family and model ID
    if [[ "$tag" == hulumed_* ]]; then
        model_family="hulumed"
        clean_name="${tag#hulumed_}"
        if [[ "$clean_name" == "hulu_med_7b" ]]; then
            model_id="ZJU-AI4H/Hulu-Med-7B"
        elif [[ "$clean_name" == "hulu_med_4b" ]]; then
            model_id="ZJU-AI4H/Hulu-Med-4B"
        else
            model_id="hulumed/$clean_name"
        fi
    elif [[ "$tag" == qwen3vl_* ]]; then
        model_family="qwen3vl"
        clean_name="${tag#qwen3vl_}"
        if [[ "$clean_name" == "qwen3_vl_2b_instruct" ]]; then
            model_id="Qwen/Qwen3-VL-2B-Instruct"
        elif [[ "$clean_name" == "qwen3_vl_2b_thinking" ]]; then
            model_id="Qwen/Qwen3-VL-2B-Thinking"
        elif [[ "$clean_name" == "qwen3_vl_4b_instruct" ]]; then
            model_id="Qwen/Qwen3-VL-4B-Instruct"
        elif [[ "$clean_name" == "qwen3_vl_4b_thinking" ]]; then
            model_id="Qwen/Qwen3-VL-4B-Thinking"
        elif [[ "$clean_name" == "qwen3_vl_8b_instruct" ]]; then
            model_id="Qwen/Qwen3-VL-8B-Instruct"
        elif [[ "$clean_name" == "qwen3_vl_8b_thinking" ]]; then
            model_id="Qwen/Qwen3-VL-8B-Thinking"
        else
            model_id="Qwen/$clean_name"
        fi
    elif [[ "$tag" == lingshu_* ]]; then
        model_family="lingshu"
        model_id="lingshu-medical-mllm/Lingshu-7B"
    elif [[ "$tag" == mage_vl_* ]]; then
        model_family="mage_vl"
        model_id="microsoft/Mage-VL"
    else
        warn "Unknown model family prefix for tag '$tag'. Skipping."
        continue
    fi

    # Check if both clip and full video levels are complete
    if [ "$clip_count" -ge "$EXPECTED_CLIP_COUNT" ] && [ "$full_count" -ge "$EXPECTED_FULL_COUNT" ]; then
        log "  [READY] $tag ($model_family | $model_id) -> Clip: $clip_count/$EXPECTED_CLIP_COUNT, Full: $full_count/$EXPECTED_FULL_COUNT"
        FINISHED_MODELS+=("$tag|$model_family|$model_id")
    else
        warn "  [SKIP INCOMPLETE] $tag -> Clip: $clip_count/$EXPECTED_CLIP_COUNT, Full: $full_count/$EXPECTED_FULL_COUNT (both levels required)"
        INCOMPLETE_MODELS+=("$tag (Clip: $clip_count/$EXPECTED_CLIP_COUNT, Full: $full_count/$EXPECTED_FULL_COUNT)")
    fi
done

echo ""
log "=================================================="
log "  Status Summary:"
log "  Finished & Ready for Judge Evaluation: ${#FINISHED_MODELS[@]}"
log "  Incomplete / Skipped:                  ${#INCOMPLETE_MODELS[@]}"
log "=================================================="

if [ ${#FINISHED_MODELS[@]} -eq 0 ]; then
    warn "No models have completed both clip and full-video inference responses."
    warn "Evaluation cannot proceed. Please wait for inference to finish."
    exit 0
fi

# ── 4. CONCURRENT EVALUATION EXECUTION ─────────────────────────────────────
log "Starting concurrent evaluation on ${#FINISHED_MODELS[@]} model(s) (Concurrency: $NUM_WORKERS worker(s))..."

run_model_evaluation() {
    local tag="$1"
    local family="$2"
    local id="$3"
    local log_file="$OUTPUT_DIR/judge_${tag}.log"

    # Ensure the output dir + log file are usable before launching.
    # On WSL drvfs/9p, deleting files concurrently with a running worker can
    # cause transient ENOENT on redirect — retry before giving up.
    mkdir -p "$OUTPUT_DIR" 2>/dev/null || true
    local attempt=0
    while ! : > "$log_file" 2>/dev/null; do
        attempt=$((attempt + 1))
        if [ "$attempt" -ge 5 ]; then
            err "Cannot create log file '$log_file' (is $OUTPUT_DIR writable?) — aborting worker for $tag"
            return 1
        fi
        sleep 1
    done

    log "[Worker Started] Evaluating: $tag ($id) on $JUDGE_MODEL @ $JUDGE_BASE_URL ..."
    
    if "$QWEN_PYTHON" main.py \
        --mode judge \
        --model-family "$family" \
        --model-id "$id" \
        --data-level both \
        --output-dir "$OUTPUT_DIR" \
        --judge-base-url "$JUDGE_BASE_URL" \
        --judge-model "$JUDGE_MODEL" \
        --judge-api-key-env "$JUDGE_API_KEY_ENV" \
        --num-workers "$NUM_WORKERS" > "$log_file" 2>&1; then
        log "[Worker Finished] Successfully graded: $tag (Log: $log_file)"
    else
        err "[Worker Failed] Evaluation encountered errors for: $tag (Check log: $log_file)"
        return 1
    fi
}

pids=()
failed=0
declare -A PID_TAG=()

for item in "${FINISHED_MODELS[@]}"; do
    IFS="|" read -r tag model_family model_id <<< "$item"
    
    # Throttle active background jobs to NUM_WORKERS
    if [ "$(jobs -rp | wc -l)" -ge "$NUM_WORKERS" ]; then
        log "Throttle: $NUM_WORKERS worker(s) active — waiting for a slot before launching $tag..."
    fi
    while [ "$(jobs -rp | wc -l)" -ge "$NUM_WORKERS" ]; do
        sleep 2
    done

    run_model_evaluation "$tag" "$model_family" "$model_id" &
    PID_TAG[$!]="$tag"
    pids+=($!)
    log "[Launched] $tag ($model_family | $model_id) -> log: judge_${tag}.log"
done

# ── 5. LIVE PROGRESS MONITOR ───────────────────────────────────────────────
# Shows a compact per-model status line (last progress from each judge log)
# while the workers run, including their tqdm bars (converted \r -> newline).
if [ ${#pids[@]} -gt 0 ]; then
    log "Monitoring ${#pids[@]} worker(s) — live progress from judge logs..."
    start_ts=$(date +%s)
    while :; do
        active=0
        status_line=""
        for pid in "${pids[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                active=1
                tag="${PID_TAG[$pid]}"
                last_line=$(tail -c 500 "$OUTPUT_DIR/judge_${tag}.log" 2>/dev/null | tr '\r' '\n' | tail -1)
                status_line+=" $tag=[${last_line:0:68}] "
            fi
        done
        if [ "$active" -eq 0 ]; then
            break
        fi
        elapsed=$(( $(date +%s) - start_ts ))
        printf "\r\033[2K[%02d:%02d] active:%s" "$((elapsed / 60))" "$((elapsed % 60))" "${status_line:0:200}"
        sleep 4
    done
    echo ""
fi

# Wait for all background workers to complete
for pid in "${pids[@]}"; do
    tag="${PID_TAG[$pid]}"
    if ! wait "$pid"; then
        failed=$((failed + 1))
        err "Worker failed: $tag (Check log: $OUTPUT_DIR/judge_${tag}.log)"
    else
        log "[Finished] $tag — graded successfully (Log: $OUTPUT_DIR/judge_${tag}.log)"
    fi
done

echo ""
log "=================================================="
if [ "$failed" -eq 0 ]; then
    log "All ${#FINISHED_MODELS[@]} model evaluations completed successfully!"
else
    warn "$failed model evaluation worker(s) encountered issues. Check logs in $OUTPUT_DIR."
fi
log "Score status per model (expected clip=989, full=15):"
for tag in "${!unique_tags[@]}"; do
    clip_scores=0; full_scores=0
    if [ -f "$OUTPUT_DIR/${tag}_clip_scores.jsonl" ]; then
        clip_scores=$(wc -l < "$OUTPUT_DIR/${tag}_clip_scores.jsonl" 2>/dev/null || echo 0)
    fi
    if [ -f "$OUTPUT_DIR/${tag}_full_scores.jsonl" ]; then
        full_scores=$(wc -l < "$OUTPUT_DIR/${tag}_full_scores.jsonl" 2>/dev/null || echo 0)
    fi
    if [ "$clip_scores" -ge "$EXPECTED_CLIP_COUNT" ] && [ "$full_scores" -ge "$EXPECTED_FULL_COUNT" ]; then
        log "  [DONE]   $tag: clip_scores=$clip_scores/$EXPECTED_CLIP_COUNT  full_scores=$full_scores/$EXPECTED_FULL_COUNT"
    else
        warn "  [PENDING] $tag: clip_scores=$clip_scores/$EXPECTED_CLIP_COUNT  full_scores=$full_scores/$EXPECTED_FULL_COUNT"
    fi
done
log "=================================================="
