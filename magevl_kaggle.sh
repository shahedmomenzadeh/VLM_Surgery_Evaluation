#!/bin/bash
# magevl_kaggle.sh
# Kaggle inference runner for microsoft/Mage-VL (codec-native video MLLM)
# Runs in --mode inference only; results are zipped for download.

set -euo pipefail

log() {
    echo -e "\033[1;36m[magevl_kaggle.sh]\033[0m $1"
}

# ── 1. CONFIGURATION ──────────────────────────────────────────────────────
WORKING_DIR="/kaggle/working"
DATASET_DIR="$WORKING_DIR/dataset"
OUTPUT_DIR="$WORKING_DIR/results"
RESULTS_ZIP="$WORKING_DIR/results_magevl.zip"

MODEL_ID="${MODEL_ID:-microsoft/Mage-VL}"
MAX_FRAMES="${MAX_FRAMES:-16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
VIDEO_BACKEND="${VIDEO_BACKEND:-frames}"     # frames or codec
SPLIT="Test"

# Google Drive file ID for Test.zip
DRIVE_FILE_ID="${DRIVE_FILE_ID:-1wDf0F5r5YlI6IgJHBADbn8nbbm8VkIBN}"

# HuggingFace token for gated model access (set as Kaggle secret or env var)
# HF_TOKEN="${HF_TOKEN:-}"
# export HF_TOKEN
export HF_HOME="$WORKING_DIR/hf_cache"

if [ -n "${HF_TOKEN:-}" ]; then
    export HF_TOKEN
fi

log "=================================================="
log "  Mage-VL Kaggle Inference Runner"
log "=================================================="
log "Working Dir:    $WORKING_DIR"
log "Dataset Dir:    $DATASET_DIR"
log "Output Dir:     $OUTPUT_DIR"
log "Model ID:       $MODEL_ID"
log "Split:          $SPLIT"
log "Max Frames:     $MAX_FRAMES"
log "Video Backend:  $VIDEO_BACKEND"
log "=================================================="

# ── 2. DOWNLOAD DATASET ────────────────────────────────────────────────────
if [ ! -d "$DATASET_DIR/Test" ]; then
    log "Installing gdown..."
    pip install -q gdown

    log "Downloading Test.zip from Google Drive (ID: $DRIVE_FILE_ID)..."
    gdown "$DRIVE_FILE_ID" --output "$WORKING_DIR/Test.zip"

    log "Unzipping Test.zip into $DATASET_DIR..."
    mkdir -p "$DATASET_DIR"
    unzip -q -o "$WORKING_DIR/Test.zip" -d "$DATASET_DIR"
    rm -f "$WORKING_DIR/Test.zip"
    log "Dataset ready at $DATASET_DIR."
else
    log "Test dataset already present at $DATASET_DIR/Test. Skipping download."
fi

# ── 3. INSTALL DEPENDENCIES ────────────────────────────────────────────────
log "Installing Python dependencies via pip..."

pip install -q --upgrade pip

log "[1/5] Installing numpy & build tools..."
pip install -q packaging ninja wheel setuptools "numpy>=2.0,<2.4"

log "[2/5] Installing causal-conv1d (--no-build-isolation)..."
pip install -q --no-build-isolation "causal-conv1d>=1.4.0"

log "[3/5] Installing mamba-ssm (--no-build-isolation)..."
pip install -q --no-build-isolation mamba-ssm

log "[4/5] Installing codec-video-prep & OpenCV..."
pip install -q "codec-video-prep" "opencv-python" "pillow"

log "[5/5] Installing remaining Mage-VL dependencies..."
pip install -q \
    "transformers>=5.7" \
    "accelerate" \
    "bitsandbytes>=0.43.0" \
    "tqdm" \
    "openai" \
    "imageio"

log "All dependencies installed successfully."

# ── 4. RUN INFERENCE ───────────────────────────────────────────────────────
mkdir -p "$OUTPUT_DIR"

log "--------------------------------------------------"
log "Running inference: $MODEL_ID (video_backend=$VIDEO_BACKEND)"
log "--------------------------------------------------"

python main.py \
    --mode inference \
    --model-family mage_vl \
    --model-id "$MODEL_ID" \
    --dataset-root "$DATASET_DIR" \
    --splits "$SPLIT" \
    --data-level both \
    --output-dir "$OUTPUT_DIR" \
    --max-frames "$MAX_FRAMES" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --mage-video-backend "$VIDEO_BACKEND" \
    --no-4bit

log "Inference completed for $MODEL_ID."

# ── 5. ZIP RESULTS ─────────────────────────────────────────────────────────
log "Zipping results from $OUTPUT_DIR -> $RESULTS_ZIP..."
cd "$WORKING_DIR"
zip -r "results_magevl.zip" results/

log "=================================================="
log "All done! Results archived at:"
log "  $RESULTS_ZIP"
log "=================================================="
