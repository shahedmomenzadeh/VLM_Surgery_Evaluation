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

# ── EVALUATION DATASET ─────────────────────────────────────────────────────
# The flat evaluation dataset (evaluation_dataset/) lives next to this script:
# one-line JSONL records + prefixed .mp4 files, no subfolders.
# Overridable via env var: DATASET_ROOT="path" ./run.sh
DATASET_ROOT="${DATASET_ROOT:-$SCRIPT_DIR/evaluation_dataset}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/results}"
MAX_FRAMES="${MAX_FRAMES:-32}"
USE_FLASH="${USE_FLASH:-false}"

log "Dataset Root: $DATASET_ROOT"
log "Output Directory: $OUTPUT_DIR"
log "Max Frames: $MAX_FRAMES"
log "Use FlashAttention-2: $USE_FLASH"

FLASH_ARG=""
if [ "$USE_FLASH" = "true" ]; then
    FLASH_ARG="--use-flash-attn"
fi

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

log "Installing PyTorch with CUDA 13.0 for HuluMed..."
uv pip install --python "$HULUMED_PYTHON" \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cu130

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

if [ "$USE_FLASH" = "true" ]; then
    log "Installing flash-attn for HuluMed..."
    uv pip install --python "$HULUMED_PYTHON" --no-build-isolation "flash-attn>=2.6.0"
fi

# ── 4. QWEN3-VL ENVIRONMENT SETTINGS ───────────────────────────────────────
QWEN_VENV="$SCRIPT_DIR/.venv-qwen3vl"
if [ ! -d "$QWEN_VENV" ]; then
    log "Creating Qwen3-VL virtual environment..."
    uv venv "$QWEN_VENV" --python 3.12
fi

QWEN_PYTHON=$(get_venv_python "$QWEN_VENV")

log "Installing PyTorch with CUDA 13.0 for Qwen3-VL..."
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

if [ "$USE_FLASH" = "true" ]; then
    log "Installing flash-attn for Qwen3-VL / Lingshu..."
    uv pip install --python "$QWEN_PYTHON" --no-build-isolation "flash-attn>=2.6.0"
fi

# ── 5. MAGE-VL ENVIRONMENT SETTINGS ────────────────────────────────────────
MAGEVL_VENV="$SCRIPT_DIR/.venv-magevl"
if [ ! -d "$MAGEVL_VENV" ]; then
    log "Creating Mage-VL virtual environment (Python 3.12)..."
    uv venv "$MAGEVL_VENV" --python 3.12
fi

MAGEVL_PYTHON=$(get_venv_python "$MAGEVL_VENV")

log "[Step/1] Installing matched PyTorch/torchvision/torchaudio trio for CUDA 12.6..."
uv pip install --python "$MAGEVL_PYTHON" \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu126

log "[Step/2] Installing numpy>=2.0,<2.4..."
uv pip install --python "$MAGEVL_PYTHON" "numpy>=2.0,<2.4"

log "[Step/3] Installing build tools needed by mamba-ssm/causal-conv1d..."
uv pip install --python "$MAGEVL_PYTHON" packaging ninja wheel setuptools

log "[Step/4] Installing causal-conv1d (--no-build-isolation)..."
uv pip install --python "$MAGEVL_PYTHON" --no-build-isolation "causal-conv1d>=1.4.0"

log "[Step/5] Installing mamba-ssm (--no-build-isolation)..."
uv pip install --python "$MAGEVL_PYTHON" --no-build-isolation mamba-ssm

log "[Step/6] Installing remaining Mage-VL dependencies..."
uv pip install --python "$MAGEVL_PYTHON" \
    "transformers>=5.7" \
    "accelerate" \
    "bitsandbytes>=0.43.0" \
    "pillow" \
    "opencv-python" \
    "codec-video-prep" \
    "tqdm" \
    "openai" \
    "imageio"

# ── 6. RUN EVALUATION INFERENCE ───────────────────────────────────────────
# ── HuluMed Evaluation (Largest First) ──────────────────────────────────
log "Running HuluMed inference on ZJU-AI4H/Hulu-Med-7B (both levels, temp=0.6)..."
"$HULUMED_PYTHON" main.py \
    --mode inference \
    --model-family hulumed \
    --model-id "ZJU-AI4H/Hulu-Med-7B" \
    --dataset-root "$DATASET_ROOT" \
    --data-level both \
    --output-dir "$OUTPUT_DIR" \
    --max-frames "$MAX_FRAMES" \
    --max-new-tokens 4096 \
    --temperature 0.6 \
    --frame-size 224 \
    $FLASH_ARG

# ── Qwen3-VL Evaluation (Largest First) ──────────────────────────────────
QWEN_LARGEST_MODELS=(
    "Qwen/Qwen3-VL-8B-Thinking:8192"
    "Qwen/Qwen3-VL-8B-Instruct:4096"
)
for item in "${QWEN_LARGEST_MODELS[@]}"; do
    model="${item%%:*}"
    tokens="${item##*:}"
    log "Running Qwen3-VL inference on $model (both levels, max-tokens=$tokens)..."
    "$QWEN_PYTHON" main.py \
        --mode inference \
        --model-family qwen3vl \
        --model-id "$model" \
        --dataset-root "$DATASET_ROOT" \
        --data-level both \
        --output-dir "$OUTPUT_DIR" \
        --max-frames "$MAX_FRAMES" \
        --max-new-tokens "$tokens" \
        --max-pixels 66976 \
        $FLASH_ARG
done

# ── HuluMed Evaluation (Remaining) ──────────────────────────────────────
log "Running HuluMed inference on ZJU-AI4H/Hulu-Med-4B (both levels, temp=0.6)..."
"$HULUMED_PYTHON" main.py \
    --mode inference \
    --model-family hulumed \
    --model-id "ZJU-AI4H/Hulu-Med-4B" \
    --dataset-root "$DATASET_ROOT" \
    --data-level both \
    --output-dir "$OUTPUT_DIR" \
    --max-frames "$MAX_FRAMES" \
    --max-new-tokens 4096 \
    --temperature 0.6 \
    $FLASH_ARG

# ── Qwen3-VL Evaluation (Remaining) ─────────────────────────────────────
QWEN_REMAINING_MODELS=(
    "Qwen/Qwen3-VL-2B-Thinking:8192"
    "Qwen/Qwen3-VL-2B-Instruct:4096"
    "Qwen/Qwen3-VL-4B-Thinking:8192"
    "Qwen/Qwen3-VL-4B-Instruct:4096"
)
for item in "${QWEN_REMAINING_MODELS[@]}"; do
    model="${item%%:*}"
    tokens="${item##*:}"
    log "Running Qwen3-VL inference on $model (both levels, max-tokens=$tokens)..."
    "$QWEN_PYTHON" main.py \
        --mode inference \
        --model-family qwen3vl \
        --model-id "$model" \
        --dataset-root "$DATASET_ROOT" \
        --data-level both \
        --output-dir "$OUTPUT_DIR" \
        --max-frames "$MAX_FRAMES" \
        --max-new-tokens "$tokens" \
        $FLASH_ARG
done

# ── Lingshu-7B Evaluation (Qwen2.5-VL) ──────────────────────────────────
log "Running Lingshu-7B inference on lingshu-medical-mllm/Lingshu-7B (both levels)..."
"$QWEN_PYTHON" main.py \
    --mode inference \
    --model-family lingshu \
    --model-id "lingshu-medical-mllm/Lingshu-7B" \
    --dataset-root "$DATASET_ROOT" \
    --data-level both \
    --output-dir "$OUTPUT_DIR" \
    --max-frames "$MAX_FRAMES" \
    --max-new-tokens 4096 \
    $FLASH_ARG

# ── Mage-VL Evaluation (microsoft/Mage-VL, codec-native 4B) ─────────────
log "Running Mage-VL inference on microsoft/Mage-VL (both levels, frame-sampling backend)..."
"$MAGEVL_PYTHON" main.py \
    --mode inference \
    --model-family mage_vl \
    --model-id "microsoft/Mage-VL" \
    --dataset-root "$DATASET_ROOT" \
    --data-level both \
    --output-dir "$OUTPUT_DIR" \
    --max-frames "$MAX_FRAMES" \
    --max-new-tokens 4096 \
    --mage-video-backend frames