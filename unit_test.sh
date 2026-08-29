#!/bin/bash
# unit_test.sh
# Comprehensive WSL test suite for Hugging Face dataset download,
# flattening, record reconstruction fidelity, and dataset loader integration.

set -euo pipefail

log_step() {
    echo -e "\n\033[1;34m[TEST STEP $1]\033[0m \033[1;37m$2\033[0m"
}

log_pass() {
    echo -e "  \033[1;32m✓ PASS:\033[0m $1"
}

log_fail() {
    echo -e "  \033[1;31m✗ FAIL:\033[0m $1" >&2
    exit 1
}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── 1. SELECT PYTHON INTERPRETER ───────────────────────────────────────────
log_step "1" "Locating Python environment in WSL..."

get_python() {
    if [ -f "$SCRIPT_DIR/.hf_venv/bin/python" ]; then
        echo "$SCRIPT_DIR/.hf_venv/bin/python"
    elif [ -f "$SCRIPT_DIR/.venv-qwen3vl/bin/python" ]; then
        echo "$SCRIPT_DIR/.venv-qwen3vl/bin/python"
    elif [ -f "$SCRIPT_DIR/.venv-magevl/bin/python" ]; then
        echo "$SCRIPT_DIR/.venv-magevl/bin/python"
    elif [ -f "$SCRIPT_DIR/.venv-hulumed/bin/python" ]; then
        echo "$SCRIPT_DIR/.venv-hulumed/bin/python"
    elif command -v python3 &>/dev/null; then
        echo "python3"
    else
        echo "python"
    fi
}

PYTHON="$(get_python)"
echo "Using Python: $PYTHON ($("$PYTHON" --version 2>&1))"

# Check required test libraries (pyarrow / datasets / huggingface_hub)
"$PYTHON" -c "
for mod in ['json', 'pathlib', 'shutil', 'pyarrow']:
    try:
        __import__(mod)
    except ImportError:
        print(f'Missing module: {mod}')
        exit(1)
" || {
    echo "Installing missing pyarrow dependency..."
    "$PYTHON" -m pip install -q pyarrow huggingface_hub
}

log_pass "Python environment and dependencies verified."

# Test directories
TEST_FLATTEN_DIR="$SCRIPT_DIR/test_scratch_flat"
TEST_AUTO_DIR="$SCRIPT_DIR/test_scratch_autoflat"
rm -rf "$TEST_FLATTEN_DIR" "$TEST_AUTO_DIR"

# ── 2. TEST DIRECT FLATTENING FROM HF FORMAT ──────────────────────────────
log_step "2" "Testing flatten_dataset.py CLI from local 'hf_upload/' package..."

if [ ! -d "$SCRIPT_DIR/hf_upload" ]; then
    log_fail "'hf_upload' directory not found in $SCRIPT_DIR."
fi

"$PYTHON" flatten_dataset.py \
    --input-dir "$SCRIPT_DIR/hf_upload" \
    --output-dir "$TEST_FLATTEN_DIR" \
    --copy-videos

log_pass "Dataset flattening completed without exceptions."

# ── 3. TEST RECORD COUNTS AND BREAKDOWNS ───────────────────────────────────
log_step "3" "Validating file counts and category breakdowns in flattened directory..."

JSONL_COUNT=$(find "$TEST_FLATTEN_DIR" -maxdepth 1 -name "*.jsonl" | wc -l)
MP4_COUNT=$(find "$TEST_FLATTEN_DIR" -maxdepth 1 -name "*.mp4" | wc -l)

echo "Found $JSONL_COUNT JSONL files, $MP4_COUNT MP4 files."

if [ "$JSONL_COUNT" -ne 1004 ]; then
    log_fail "Expected 1004 JSONL records, but got $JSONL_COUNT."
fi
log_pass "1004 JSONL records successfully generated."

if [ "$MP4_COUNT" -ne 518 ]; then
    log_fail "Expected 518 MP4 video files, but got $MP4_COUNT."
fi
log_pass "518 MP4 video files linked/copied to root."

# ── 4. TEST RECORD CONTENT FIDELITY ───────────────────────────────────────
log_step "4" "Validating content and schema fidelity against ground-truth evaluation_dataset..."

"$PYTHON" -c "
import json
from pathlib import Path

ref_root = Path('evaluation_dataset')
gen_root = Path('test_scratch_flat')

test_samples = [
    '0xUbMicNy-w_clip_01_visual_description.jsonl',
    '0xUbMicNy-w_clip_01_step_identification.jsonl',
    '0xUbMicNy-w_clip_01_instrument_identification.jsonl',
    '0xUbMicNy-w_clip_01_visual_observation.jsonl',
    'PH_0003_2933_S2_0121_boundary_detection.jsonl',
    'PH_0003_2933_S2_0404_temporal_localization.jsonl',
    'PH_0003_2933_S2_0381_timestamp_to_phase.jsonl',
    'PH_0003_2933_S2_0071_contextual_phase_recognition.jsonl',
    '0xUbMicNy-w_full_video_narration.jsonl',
]

for filename in test_samples:
    ref_file = ref_root / filename
    gen_file = gen_root / filename
    
    if not ref_file.exists():
        continue
    if not gen_file.exists():
        raise FileNotFoundError(f'Missing generated record: {gen_file}')
        
    ref_obj = json.loads(ref_file.read_text(encoding='utf-8').strip().splitlines()[0])
    gen_obj = json.loads(gen_file.read_text(encoding='utf-8').strip().splitlines()[0])
    
    # Core envelope checks
    for key in ['record_id', 'task_category', 'question_type', 'reward_type', 'split', 'track', 'video']:
        assert ref_obj[key] == gen_obj[key], f'{filename}: mismatch in {key} ({ref_obj[key]} vs {gen_obj[key]})'
        
    # Task specific answer / message checks
    if 'correct_answer' in ref_obj:
        assert ref_obj['correct_answer'] == gen_obj['correct_answer'], f'{filename}: correct_answer mismatch'
        
    # Check that video file exists
    vid_file = gen_root / gen_obj['video']
    assert vid_file.is_file(), f'{filename}: video {vid_file} does not exist'

print('All sample records verified with 100% schema and video reference accuracy!')
"

log_pass "Content fidelity verified across visual_description, mcq, phase, and narration tasks."

# ── 5. TEST DATASET LOADER INTEGRATION ────────────────────────────────────
log_step "5" "Testing dataset_loader.py record loading from flattened directory..."

"$PYTHON" -c "
import dataset_loader

clip_recs = dataset_loader.load_clip_records('test_scratch_flat', splits=['Test'], validate_videos=True)
full_recs = dataset_loader.load_full_video_records('test_scratch_flat', splits=['Test'], validate_videos=True)

print(f'Loaded clip records: {len(clip_recs)} (expected 989)')
print(f'Loaded full video records: {len(full_recs)} (expected 15)')

assert len(clip_recs) == 989, f'Expected 989 clip records, got {len(clip_recs)}'
assert len(full_recs) == 15, f'Expected 15 full video records, got {len(full_recs)}'
"

log_pass "dataset_loader loaded 989 clip records and 15 full-video records with zero missing videos."

# ── 6. TEST AUTOMATIC HF-DIRECTORY DETECTION & FLATTENING ─────────────────
log_step "6" "Testing automatic HF-dataset detection in dataset_loader..."

"$PYTHON" -c "
import dataset_loader

# Pass the HF format directory (hf_upload) directly to dataset_loader with a custom target
clip_recs = dataset_loader.load_clip_records(
    'hf_upload',
    splits=['Test'],
    validate_videos=True,
    flatten_dir='test_scratch_autoflat'
)

print(f'Auto-flattened and loaded {len(clip_recs)} clip records from hf_upload/')
assert len(clip_recs) == 989
"

log_pass "Auto-detection and automatic flattening on-the-fly successfully executed."

# ── 7. TEST MAIN.PY CLI DRY-RUN ───────────────────────────────────────────
log_step "7" "Testing main.py --dry-run with flattened and HF directories..."

"$PYTHON" main.py \
    --dry-run \
    --dataset-root "$TEST_FLATTEN_DIR" \
    --splits Test \
    --data-level both

"$PYTHON" main.py \
    --dry-run \
    --dataset-root "$SCRIPT_DIR/hf_upload" \
    --flatten-dir "$TEST_AUTO_DIR" \
    --splits Test \
    --data-level both

log_pass "main.py CLI dry-run succeeded for both pre-flattened and auto-flattened workflows."

# ── 8. CLEANUP ────────────────────────────────────────────────────────────
log_step "8" "Cleaning up test directories..."
rm -rf "$TEST_FLATTEN_DIR" "$TEST_AUTO_DIR"
log_pass "Scratch test directories cleaned up."

echo -e "\n\033[1;32m=================================================="
echo -e "  ALL UNIT TESTS PASSED SUCCESSFULLY!"
echo -e "==================================================\033[0m\n"
