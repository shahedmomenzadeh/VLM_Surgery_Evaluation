#!/bin/bash
# eval_full_timestamped.sh
# Full-Video Cataract Surgery Evaluation Runner with Visual Timestamp Overlays (MM:SS)
# Runs procedural narration inference and LLM-as-a-judge evaluation across VLM families.

set -euo pipefail

log() {
    echo -e "\033[1;32m[eval_full_timestamped.sh]\033[0m $1"
}
warn() {
    echo -e "\033[1;33m[eval_full_timestamped.sh WARNING]\033[0m $1"
}
err() {
    echo -e "\033[1;31m[eval_full_timestamped.sh ERROR]\033[0m $1" >&2
}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Clean shutdown handler for background/child processes
cleanup_jobs() {
    warn "Received interruption/stop signal! Terminating evaluation processes..."
    trap - INT TERM HUP TSTP
    pkill -P $$ 2>/dev/null || true
    kill -TERM 0 2>/dev/null || true
    wait 2>/dev/null || true
    exit 130
}
trap cleanup_jobs INT TERM HUP TSTP

# ── 1. CONFIGURATION OVERRIDES ─────────────────────────────────────────────
MODEL_FAMILY="${MODEL_FAMILY:-qwen3vl}"     # qwen3vl, hulumed, lingshu, qwen2_5_vl, mage_vl
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-VL-2B-Instruct}"
DATASET_ROOT="${DATASET_ROOT:-$SCRIPT_DIR/evaluation_dataset}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/results}"
TIMESTAMP_CACHE_DIR="${TIMESTAMP_CACHE_DIR:-$SCRIPT_DIR/timestamped_cache}"
TIMESTAMP_FORMAT="${TIMESTAMP_FORMAT:-mmss}" # mmss or hms
FONTSIZE="${FONTSIZE:-48}"
BOXCOLOR="${BOXCOLOR:-black@0.85}"
BOXBORDERW="${BOXBORDERW:-10}"
SPLIT="${SPLIT:-Test}"
MAX_FRAMES="${MAX_FRAMES:-32}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
TEMPERATURE="${TEMPERATURE:-0.1}"
MODE="${MODE:-inference}"                   # inference, judge, or all
USE_FLASH="${USE_FLASH:-false}"
FORCE_REENCODE="${FORCE_REENCODE:-false}"
PROMPT_HINT="${PROMPT_HINT:-true}"
TAG="${TAG:-}"

# Judge configuration
JUDGE_BASE_URL="${JUDGE_BASE_URL:-https://opencode.ai/zen/go/v1/responses}"
JUDGE_MODEL="${JUDGE_MODEL:-Gemini-3.8-flash}"
PROVIDER_API_KEY="${PROVIDER_API_KEY:-}"

# Parse command line overrides
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=true
            EXTRA_ARGS+=("--dry-run")
            shift
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --model-family)
            MODEL_FAMILY="$2"
            shift 2
            ;;
        --model-id)
            MODEL_ID="$2"
            shift 2
            ;;
        --max-frames)
            MAX_FRAMES="$2"
            shift 2
            ;;
        --tag)
            TAG="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# Load .env if present (strip Windows CR line endings and surrounding quotes)
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

export PROVIDER_API_KEY
export HF_HOME="${HF_HOME:-$SCRIPT_DIR/hf_cache}"

log "=================================================="
log "  Full-Video Timestamped Narration Runner"
log "=================================================="
log "Model Family:     $MODEL_FAMILY"
log "Model ID:         $MODEL_ID"
log "Dataset Root:     $DATASET_ROOT"
log "Output Dir:       $OUTPUT_DIR"
log "Timestamp Cache:  $TIMESTAMP_CACHE_DIR"
log "Timestamp Format: ${TIMESTAMP_FORMAT^^} (Size=${FONTSIZE}px, Box=${BOXCOLOR})"
log "Execution Mode:   $MODE"
log "Max Frames:       $MAX_FRAMES"
log "Temperature:      $TEMPERATURE"
log "Judge Model:      $JUDGE_MODEL"
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
        return 1
    fi
}

RESOLVED_PYTHON=""

# Try family-specific venv first
CANDIDATE_VENVS=(
    "$SCRIPT_DIR/.venv-${MODEL_FAMILY}"
    "$SCRIPT_DIR/.venv-qwen3vl"
    "$SCRIPT_DIR/.venv-hulumed"
    "$SCRIPT_DIR/.venv-magevl"
    "$SCRIPT_DIR/.venv"
)

for venv in "${CANDIDATE_VENVS[@]}"; do
    if PY=$(get_venv_python "$venv" 2>/dev/null); then
        RESOLVED_PYTHON="$PY"
        log "Found virtual environment Python: $RESOLVED_PYTHON"
        break
    fi
done

if [ -z "$RESOLVED_PYTHON" ]; then
    if command -v python3 &>/dev/null; then
        RESOLVED_PYTHON="python3"
    else
        RESOLVED_PYTHON="python"
    fi
    warn "No virtual environment found. Falling back to system interpreter: $RESOLVED_PYTHON"
fi

# ── 3. PRE-FLIGHT CHECKS ───────────────────────────────────────────────────
if ! command -v ffmpeg &>/dev/null; then
    err "ffmpeg binary is required for burning timestamps into videos, but was not found in PATH."
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
mkdir -p "$TIMESTAMP_CACHE_DIR"

# ── 4. FLASH ATTENTION FLAG ────────────────────────────────────────────────
FLASH_ARG=()
if [ "$USE_FLASH" = "true" ]; then
    FLASH_ARG=("--use-flash-attn")
fi

# ── 5. TAG ARGUMENT ────────────────────────────────────────────────────────
TAG_ARG=()
if [ -n "$TAG" ]; then
    TAG_ARG=("--tag" "$TAG")
fi

# ── 6. RE-ENCODE & HINT FLAGS ──────────────────────────────────────────────
EXTRA_FLAGS=()
if [ "$FORCE_REENCODE" = "true" ]; then
    EXTRA_FLAGS+=("--force-reencode")
fi
if [ "$PROMPT_HINT" = "false" ]; then
    EXTRA_FLAGS+=("--no-prompt-hint")
fi

# ── 7. EXECUTE RUNNER ──────────────────────────────────────────────────────
log "Invoking full_video_timestamped_inference.py..."

"$RESOLVED_PYTHON" "$SCRIPT_DIR/full_video_timestamped_inference.py" \
    --mode "$MODE" \
    --model-family "$MODEL_FAMILY" \
    --model-id "$MODEL_ID" \
    --dataset-root "$DATASET_ROOT" \
    --splits "$SPLIT" \
    --output-dir "$OUTPUT_DIR" \
    --timestamp-cache-dir "$TIMESTAMP_CACHE_DIR" \
    --timestamp-format "$TIMESTAMP_FORMAT" \
    --fontsize "$FONTSIZE" \
    --boxcolor "$BOXCOLOR" \
    --boxborderw "$BOXBORDERW" \
    --max-frames "$MAX_FRAMES" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE" \
    --judge-base-url "$JUDGE_BASE_URL" \
    --judge-model "$JUDGE_MODEL" \
    --judge-api-key-env "PROVIDER_API_KEY" \
    "${TAG_ARG[@]}" \
    "${FLASH_ARG[@]}" \
    "${EXTRA_FLAGS[@]}" \
    "${EXTRA_ARGS[@]}"

log "=================================================="
log "Full-video timestamped evaluation finished!"
log "Artifacts directory: $OUTPUT_DIR"
log "=================================================="
