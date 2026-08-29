#!/bin/bash
# qwen3vl_kaggle.sh
# Kaggle inference runner for Qwen3-VL-2B-Instruct and Qwen3-VL-2B-Thinking models
# Runs in --mode inference only; results are zipped for download.

set -euo pipefail

log() {
    echo -e "\033[1;36m[qwen3vl_kaggle.sh]\033[0m $1"
}

# ── 1. CONFIGURATION ──────────────────────────────────────────────────────
WORKING_DIR="/kaggle/working"
DATASET_DIR="$WORKING_DIR/dataset"
OUTPUT_DIR="$WORKING_DIR/results"
RESULTS_ZIP="$WORKING_DIR/results_qwen3vl_2b_thinking_instruct.zip"

MAX_FRAMES="${MAX_FRAMES:-32}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
TEMPERATURE="${TEMPERATURE:-0.1}"
SPLIT="Test"
USE_FLASH="${USE_FLASH:-false}"

# HuggingFace Dataset repo identifier
HF_DATASET_REPO="${HF_DATASET_REPO:-shahedm2001/cataract_surgery_vlm_eval}"

# HuggingFace token for gated model / dataset access (set as Kaggle secret or env var)
# HF_TOKEN="${HF_TOKEN:-}"
# export HF_TOKEN
export HF_HOME="$WORKING_DIR/hf_cache"

if [ -n "${HF_TOKEN:-}" ]; then
    export HF_TOKEN
fi

log "=================================================="
log "  Qwen3-VL 2B Kaggle Inference Runner"
log "=================================================="
log "Working Dir:  $WORKING_DIR"
log "HF Dataset:   $HF_DATASET_REPO"
log "Dataset Dir:  $DATASET_DIR"
log "Output Dir:   $OUTPUT_DIR"
log "Split:        $SPLIT"
log "Max Frames:   $MAX_FRAMES"
log "Flash-Attn 2: $USE_FLASH"
log "=================================================="

# ── 2. INSTALL DEPENDENCIES ────────────────────────────────────────────────
log "Installing Python dependencies via pip..."

pip install -q --upgrade pip

# Core inference and dataset dependencies
pip install -q \
    "git+https://github.com/huggingface/transformers.git" \
    "accelerate" \
    "bitsandbytes>=0.43.0" \
    "qwen-vl-utils[decord]" \
    "openai" \
    "tqdm" \
    "imageio" \
    "huggingface_hub" \
    "pyarrow"

# ── 3. DOWNLOAD & FLATTEN DATASET ─────────────────────────────────────────
if [ ! -d "$DATASET_DIR" ] || [ -z "$(ls -A "$DATASET_DIR" 2>/dev/null)" ]; then
    log "Downloading and flattening Hugging Face dataset '$HF_DATASET_REPO' -> '$DATASET_DIR'..."
    python flatten_dataset.py \
        --hf-repo "$HF_DATASET_REPO" \
        --output-dir "$DATASET_DIR"
    log "Dataset ready and flattened at $DATASET_DIR."
else
    log "Dataset already present at $DATASET_DIR. Skipping download."
fi

FLASH_ARG=""
if [ "$USE_FLASH" = "true" ]; then
    log "Installing flash-attn for accelerated inference..."
    pip install -q flash-attn --no-build-isolation
    FLASH_ARG="--use-flash-attn"
fi

log "All dependencies installed."

# ── 4. RUN INFERENCE ───────────────────────────────────────────────────────
mkdir -p "$OUTPUT_DIR"

MODELS=(
    "Qwen/Qwen3-VL-2B-Instruct"
    "Qwen/Qwen3-VL-2B-Thinking"
)

for MODEL_ID in "${MODELS[@]}"; do
    log "--------------------------------------------------"
    log "Running inference: $MODEL_ID"
    log "--------------------------------------------------"

    python main.py \
        --mode inference \
        --model-family qwen3vl \
        --model-id "$MODEL_ID" \
        --dataset-root "$DATASET_DIR" \
        --splits "$SPLIT" \
        --data-level both \
        --output-dir "$OUTPUT_DIR" \
        --max-frames "$MAX_FRAMES" \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        --temperature "$TEMPERATURE" \
        --no-4bit \
        $FLASH_ARG

    log "Inference completed for $MODEL_ID."
done

# ── 5. ZIP RESULTS ─────────────────────────────────────────────────────────
log "Zipping results from $OUTPUT_DIR -> $RESULTS_ZIP..."
cd "$WORKING_DIR"
zip -r "results_qwen3vl_2b_thinking_instruct.zip" results/

log "=================================================="
log "All done! Results archived at:"
log "  $RESULTS_ZIP"
log "=================================================="
