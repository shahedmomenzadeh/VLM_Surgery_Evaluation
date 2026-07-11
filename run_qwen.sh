#!/bin/bash
# run_qwen.sh
# Environment preparation and inference runner for Qwen3-VL-2B-Instruct evaluation

set -euo pipefail

# ── 1. CHECK AND INSTALL UV ───────────────────────────────────────────────
log() {
    echo -e "\033[1;32m[run_qwen.sh]\033[0m $1"
}

if ! command -v uv &>/dev/null; then
    log "uv not found in PATH. Checking home directories..."
    if [ -f "$HOME/.local/bin/uv" ]; then
        export PATH="$HOME/.local/bin:$PATH"
    elif [ -f "$USERPROFILE/.local/bin/uv" ]; then
        export PATH="$USERPROFILE/.local/bin:$PATH"
    elif [ -f "$HOME/.cargo/bin/uv" ]; then
        export PATH="$HOME/.cargo/bin:$PATH"
    else
        log "Installing uv automatically..."
        if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
            powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
            export PATH="$LOCALAPPDATA/programs/uv:$PATH"
            export PATH="$HOME/.local/bin:$PATH"
        else
            curl -LsSf https://astral.sh/uv/install.sh | sh
            export PATH="$HOME/.local/bin:$PATH"
        fi
    fi
fi

if ! command -v uv &>/dev/null; then
    echo "Error: Failed to locate or install uv. Please install it manually." >&2
    exit 1
fi
log "uv version: $(uv --version)"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── 2. CONFIGURATION OVERRIDES ─────────────────────────────────────────────
HF_TOKEN="${HF_TOKEN:-}"
if [ -n "$HF_TOKEN" ]; then
    export HF_TOKEN
fi

export HF_HOME="$SCRIPT_DIR/hf_cache"

if [ -d "$SCRIPT_DIR/dataset" ]; then
    DATASET_ROOT="${DATASET_ROOT:-$SCRIPT_DIR/dataset}"
else
    DATASET_ROOT="${DATASET_ROOT:-D:/programming/MSc_project/Video pipeline/dataset}"
fi
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/results}"
MAX_FRAMES="${MAX_FRAMES:-48}"

log "Dataset Root: $DATASET_ROOT"
log "Output Directory: $OUTPUT_DIR"
log "Max Frames: $MAX_FRAMES"

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

# ── 3. QWEN3-VL ENVIRONMENT SETTINGS ───────────────────────────────────────
QWEN_VENV="$SCRIPT_DIR/.venv-qwen3vl"
if [ ! -d "$QWEN_VENV" ]; then
    log "Creating Qwen3-VL virtual environment..."
    uv venv "$QWEN_VENV" --python 3.12
fi

QWEN_PYTHON=$(get_venv_python "$QWEN_VENV")

log "Installing PyTorch with CUDA 12.6 for Qwen3-VL..."
uv pip install --python "$QWEN_PYTHON" \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cu130

log "Checking and installing Qwen3-VL remaining requirements..."
uv pip install --python "$QWEN_PYTHON" \
    "git+https://github.com/huggingface/transformers.git" \
    "accelerate" \
    "bitsandbytes>=0.43.0" \
    "qwen-vl-utils[decord]" \
    "openai" \
    "tqdm" \
    "imageio"

# ── 4. RUN QWEN3-VL INFERENCE ──────────────────────────────────────────────
log "Running Qwen3-VL-2B-Instruct inference (clip + full-video, max_frames=$MAX_FRAMES)..."
"$QWEN_PYTHON" main.py \
    --mode inference \
    --model-family qwen3vl \
    --model-id Qwen/Qwen3-VL-2B-Instruct \
    --dataset-root "$DATASET_ROOT" \
    --data-level both \
    --output-dir "$OUTPUT_DIR" \
    --max-frames "$MAX_FRAMES" \
    --max-new-tokens 4096
