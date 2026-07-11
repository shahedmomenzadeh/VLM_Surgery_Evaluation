#!/bin/bash
# run.sh
# Environment preparation and CLI test runner for cataract surgery VLM evaluation

set -euo pipefail

# ── 1. CHECK AND INSTALL UV ───────────────────────────────────────────────
log() {
    echo -e "\033[1;32m[run.sh]\033[0m $1"
}

# Determine if uv is available
if ! command -v uv &>/dev/null; then
    log "uv not found in PATH. Checking home directories..."
    # Check standard Windows/Unix locations
    if [ -f "$HOME/.local/bin/uv" ]; then
        export PATH="$HOME/.local/bin:$PATH"
    elif [ -f "$USERPROFILE/.local/bin/uv" ]; then
        export PATH="$USERPROFILE/.local/bin:$PATH"
    elif [ -f "$HOME/.cargo/bin/uv" ]; then
        export PATH="$HOME/.cargo/bin:$PATH"
    else
        log "Installing uv automatically..."
        if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
            # Windows PowerShell installation
            powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
            # Add typical Windows path
            export PATH="$LOCALAPPDATA/programs/uv:$PATH"
            export PATH="$HOME/.local/bin:$PATH"
        else
            # Unix-like installation
            curl -LsSf https://astral.sh/uv/install.sh | sh
            export PATH="$HOME/.local/bin:$PATH"
        fi
    fi
fi

# Final verify
if ! command -v uv &>/dev/null; then
    echo "Error: Failed to locate or install uv. Please install it manually." >&2
    exit 1
fi
log "uv version: $(uv --version)"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── 2. CONFIGURATION OVERRIDES ─────────────────────────────────────────────
# Set Hugging Face Access Token if downloading gated models (e.g. Qwen3-VL)
# Replace 'hf_abcd123' with your actual Hugging Face token
HF_TOKEN="${HF_TOKEN:-}"
if [ -n "$HF_TOKEN" ]; then
    export HF_TOKEN
fi

# Save models locally inside the working directory under 'hf_cache'
export HF_HOME="$SCRIPT_DIR/hf_cache"

# Default dataset path. Overridable via env var: DATASET_ROOT="path" ./run.sh
# Automatically detects if there is a local 'dataset' folder in the script directory, otherwise falls back to D:/
if [ -d "$SCRIPT_DIR/dataset" ]; then
    DATASET_ROOT="${DATASET_ROOT:-$SCRIPT_DIR/dataset}"
else
    DATASET_ROOT="${DATASET_ROOT:-D:/programming/MSc_project/Video pipeline/dataset}"
fi
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/results}"
MAX_FRAMES="${MAX_FRAMES:-32}"

log "Dataset Root: $DATASET_ROOT"
log "Output Directory: $OUTPUT_DIR"
log "Max Frames: $MAX_FRAMES"

# Helper to find python inside venv cross-platform (Windows Scripts/ vs Linux bin/)
get_venv_python() {
    local venv_path="$1"
    if [ -f "$venv_path/Scripts/python.exe" ]; then
        echo "$venv_path/Scripts/python.exe"
    elif [ -f "$venv_path/Scripts/python" ]; then
        echo "$venv_path/Scripts/python"
    elif [ -f "$venv_path/bin/python" ]; then
        echo "$venv_path/bin/python"
    else
        echo "$venv_path/bin/python" # fallback default
    fi
}

# ── 3. HULUMED ENVIRONMENT SETTINGS ───────────────────────────────────────
HULUMED_VENV="$SCRIPT_DIR/.venv-hulumed"
if [ ! -d "$HULUMED_VENV" ]; then
    log "Creating HuluMed virtual environment..."
    uv venv "$HULUMED_VENV" --python 3.12
fi

HULUMED_PYTHON=$(get_venv_python "$HULUMED_VENV")

log "Installing PyTorch with CUDA 12.6 for HuluMed..."
uv pip install --python "$HULUMED_PYTHON" \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cu126

log "Checking and installing HuluMed remaining requirements..."
uv pip install --python "$HULUMED_PYTHON" \
    "transformers==4.51.2" \
    "accelerate==1.7.0" \
    "bitsandbytes>=0.43.0" \
    "ffmpeg-python" \
    "decord" \
    "opencv-python" \
    "Pillow" \
    "openai" \
    "tqdm" \
    "imageio"

# ── 4. QWEN3-VL ENVIRONMENT SETTINGS ───────────────────────────────────────
QWEN_VENV="$SCRIPT_DIR/.venv-qwen3vl"
if [ ! -d "$QWEN_VENV" ]; then
    log "Creating Qwen3-VL virtual environment..."
    uv venv "$QWEN_VENV" --python 3.12
fi

QWEN_PYTHON=$(get_venv_python "$QWEN_VENV")

log "Installing PyTorch with CUDA 12.6 for Qwen3-VL..."
uv pip install --python "$QWEN_PYTHON" \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cu126

log "Checking and installing Qwen3-VL remaining requirements..."
uv pip install --python "$QWEN_PYTHON" \
    "git+https://github.com/huggingface/transformers.git" \
    "accelerate" \
    "bitsandbytes>=0.43.0" \
    "qwen-vl-utils[decord]" \
    "openai" \
    "tqdm" \
    "imageio"

# ── 5. RUN EVALUATION INFERENCE ───────────────────────────────────────────
# log "Running VLM inference on ZJU-AI4H/Hulu-Med-4B (both levels)..."
# "$HULUMED_PYTHON" main.py \
#     --mode inference \
#     --model-family hulumed \
#     --model-id ZJU-AI4H/Hulu-Med-4B \
#     --dataset-root "$DATASET_ROOT" \
#     --data-level both \
#     --output-dir "$OUTPUT_DIR" \
#     --max-frames "$MAX_FRAMES" \
#     --max-new-tokens 4096

log "Running VLM inference on Qwen/Qwen3-VL-2B-Instruct (both levels)..."
"$QWEN_PYTHON" main.py \
    --mode inference \
    --model-family qwen3vl \
    --model-id Qwen/Qwen3-VL-2B-Instruct \
    --dataset-root "$DATASET_ROOT" \
    --data-level both \
    --output-dir "$OUTPUT_DIR" \
    --max-frames "$MAX_FRAMES" \
    --max-new-tokens 4096