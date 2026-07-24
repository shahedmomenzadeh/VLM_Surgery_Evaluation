#!/bin/bash
# test_lingshu.sh
# Test runner for Lingshu-7B (Qwen2.5-VL) VLM evaluation

set -euo pipefail

log() {
    echo -e "\033[1;32m[test_lingshu.sh]\033[0m $1"
}

# ── 1. CHECK AND LOCATE UV ────────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
    log "uv not found in PATH. Checking home directories..."
    if [ -f "$HOME/.local/bin/uv" ]; then
        export PATH="$HOME/.local/bin:$PATH"
    elif [ -f "$USERPROFILE/.local/bin/uv" ]; then
        export PATH="$USERPROFILE/.local/bin:$PATH"
    elif [ -f "$HOME/.cargo/bin/uv" ]; then
        export PATH="$HOME/.cargo/bin:$PATH"
    fi
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── 2. CONFIGURATION OVERRIDES ─────────────────────────────────────────────
export HF_HOME="${HF_HOME:-$SCRIPT_DIR/hf_cache}"

# ── DOWNLOAD TEST DATASET ──────────────────────────────────────────────────
if ! command -v gdown &>/dev/null; then
    log "Installing gdown for dataset download..."
    pip install gdown || true
fi

if [ ! -d "$SCRIPT_DIR/dataset/Test" ]; then
    log "Downloading Test split from Google Drive..."
    gdown 1ziUmbavxCsnjfHu59BTMWLxJAgQrdJaw --output "$SCRIPT_DIR/Test.zip"
    log "Extracting Test.zip into dataset directory..."
    unzip -o "$SCRIPT_DIR/Test.zip" -d "$SCRIPT_DIR/dataset"
    rm -f "$SCRIPT_DIR/Test.zip"
    log "Test dataset ready."
else
    log "Test dataset already exists, skipping download."
fi

if [ -d "$SCRIPT_DIR/dataset" ]; then
    DATASET_ROOT="${DATASET_ROOT:-$SCRIPT_DIR/dataset}"
else
    DATASET_ROOT="${DATASET_ROOT:-D:/programming/MSc_project/Video pipeline/dataset}"
fi
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/results}"
MAX_FRAMES="${MAX_FRAMES:-32}"

log "Dataset Root:     $DATASET_ROOT"
log "Output Directory: $OUTPUT_DIR"
log "Max Frames:       $MAX_FRAMES"

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

# ── 3. LOCATE OR SETUP VIRTUAL ENVIRONMENT ────────────────────────────────
QWEN_VENV="$SCRIPT_DIR/.venv-qwen3vl"
if [ ! -d "$QWEN_VENV" ]; then
    log "Creating virtual environment for Qwen2.5-VL / Lingshu-7B..."
    uv venv "$QWEN_VENV" --python 3.12
    QWEN_PYTHON=$(get_venv_python "$QWEN_VENV")
    log "Installing PyTorch with CUDA support..."
    uv pip install --python "$QWEN_PYTHON" torch torchvision --index-url https://download.pytorch.org/whl/cu126
    log "Installing transformers, qwen-vl-utils, and dependencies..."
    uv pip install --python "$QWEN_PYTHON" \
        "git+https://github.com/huggingface/transformers.git" \
        "accelerate" \
        "bitsandbytes>=0.43.0" \
        "qwen-vl-utils[decord]" \
        "openai" \
        "tqdm" \
        "imageio"
else
    QWEN_PYTHON=$(get_venv_python "$QWEN_VENV")
fi

log "Using Python interpreter: $QWEN_PYTHON"

# ── 4. RUN LINGSHU-7B INFERENCE TEST ──────────────────────────────────────
log "Running Lingshu-7B inference test (model-id: lingshu-medical-mllm/Lingshu-7B)..."
"$QWEN_PYTHON" main.py \
    --mode inference \
    --model-family lingshu \
    --model-id "lingshu-medical-mllm/Lingshu-7B" \
    --dataset-root "$DATASET_ROOT" \
    --data-level both \
    --output-dir "$OUTPUT_DIR" \
    --max-frames "$MAX_FRAMES" \
    --max-new-tokens 4096

log "Lingshu-7B test run completed successfully!"
