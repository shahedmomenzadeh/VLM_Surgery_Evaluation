# VLM Surgery Evaluation

A framework for benchmarking Vision-Language Models (VLMs) on cataract surgery video understanding tasks. This pipeline generates model responses to structured surgical questions and evaluates them using a combination of deterministic scoring and LLM-as-a-judge evaluation.

## Overview

This project evaluates how well VLMs understand cataract surgery videos by testing them on two granularities:

- **Clip-level**: Short surgical video segments with multiple-choice questions about visual observations
- **Full-video level**: Complete surgery recordings requiring procedural narration and chronological step ordering

Currently supported model families:
- [HuluMed](https://huggingface.co/ZJU-AI4H/Hulu-Med-4B) (ZJU-AI4H)
- [Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) (Alibaba)
- [Lingshu-7B](https://huggingface.co/lingshu-medical-mllm/Lingshu-7B) (Qwen2.5-VL based medical MLLM)
- [Mage-VL](https://huggingface.co/microsoft/Mage-VL) (Microsoft, codec-native video MLLM)

## VLM Response Types

The framework generates three categories of responses from VLMs:

### 1. Clip-Level Multiple-Choice (MCQ)

Models are shown short surgical video clips and asked to answer multiple-choice questions about what they observe. Each question is evaluated under two prompting strategies:

| Strategy | Description |
|----------|-------------|
| **Chain-of-Thought (CoT)** | Model reasons step-by-step, then outputs `ANSWER: <letter>` |
| **Direct** | Model outputs only the answer letter without reasoning |

### 2. Full-Video Narration

Models watch a complete cataract surgery video and produce a freeform chronological narration describing the surgical procedure step-by-step. The narration is evaluated against an expert-written reference.

### 3. Full-Video Sequence Ordering

Models are shown a full surgery video and given a shuffled list of surgical step descriptions. They must reorder the steps into the correct chronological sequence. Two prompting variants are tested:

| Strategy | Description |
|----------|-------------|
| **Direct** | Model outputs only the ordered letter sequence (e.g., `B, D, A, C`) |
| **Visual CoT** | Model first describes observed steps, then outputs `SEQUENCE: <letters>` |

## Evaluation Metrics

### Clip-Level Metrics

| Metric | Scale | Method | Description |
|--------|-------|--------|-------------|
| **Normalised Accuracy** | 0-1 | Deterministic | Exact match of extracted answer letter vs. ground truth |
| **LLM Judge Score** | 0-3 | LLM-as-a-judge | Evaluates both answer correctness and reasoning quality |

The LLM Judge scoring rubric (0-3):
- **3**: Correct answer AND reasoning describes accurate visual details
- **2**: Correct answer but reasoning is vague, generic, or partially wrong
- **1**: Wrong answer but reasoning shows partial understanding
- **0**: Wrong answer and reasoning is incorrect or irrelevant

### Full-Video Narration Metrics

Each narration is scored by an LLM judge across five dimensions (all 0-5):

| Dimension | Description |
|-----------|-------------|
| **Step Coverage** | Whether every major surgical step in the reference is mentioned |
| **Chronological Accuracy** | Whether the narrated order matches the actual sequence of events |
| **Visual & Technical Accuracy** | Whether instruments, tissue interactions, and maneuvers are correctly described |
| **Narrative Flow** | Whether the narration reads as a coherent, fluid real-time account |
| **Overall Score** | Holistic judgment weighting factual/chronological correctness over style |

### Full-Video Sequence Ordering Metrics

| Metric | Description |
|--------|-------------|
| **Kendall's Tau** | Rank correlation between predicted and ground-truth step orderings (-1 to 1) |
| **Exact Match Rate** | Proportion of videos where the predicted sequence exactly matches ground truth |
| **Valid Extraction Rate** | Proportion of responses from which a complete permutation could be extracted |

Sequence extraction uses regex-based parsing first, with LLM-judge fallback for ambiguous responses.

### Judge Response Truncation

Hallucinatory or looping VLM outputs can exceed the judge's context window. Before submission, judge requests truncate model responses to ~2048 tokens (approx. 4 chars/token), appending a `[TRUNCATED]` marker and preserving any trailing `ANSWER:` / `SEQUENCE:` suffix if it was cut off.

## Project Structure

```
.
├── main.py                 # CLI entry point and orchestrator
├── dataset_loader.py       # Loads clip-level and full-video records
├── prompts.py              # All inference and judge prompt templates
├── llm_judge.py            # LLM judge scoring and metric computation
├── hulumed_inference.py    # HuluMed model inference pipeline
├── qwen3VL_inference.py    # Qwen3-VL model inference pipeline
├── lingshu_inference.py    # Lingshu-7B / Qwen2.5-VL inference pipeline
├── mage_vl_inference.py    # Mage-VL inference pipeline (frames & codec backends)
├── run.sh                  # Environment setup and execution script
├── eval_all.sh             # Automated offline judge runner over all response files
└── results/                # Generated responses, scores, and summaries
```

## Usage

### Quick Start

```bash
# Set your HuggingFace token for gated models
export HF_TOKEN="your_token_here"

# Run full evaluation (inference + judging)
python main.py \
    --mode all \
    --model-family qwen3vl \
    --model-id Qwen/Qwen3-VL-2B-Instruct \
    --dataset-root /path/to/dataset \
    --data-level both \
    --output-dir ./results \
    --max-frames 32
```

### Execution Modes

| Mode | Description |
|------|-------------|
| `all` | Run inference and immediately judge responses |
| `inference` | Generate responses only (for offline judging later) |
| `judge` | Grade pre-generated responses offline |

### Key Arguments

| Argument | Description |
|----------|-------------|
| `--model-family` | `hulumed`, `qwen3vl`, `lingshu`, `qwen2_5_vl`, or `mage_vl` |
| `--data-level` | `clip`, `full`, or `both` |
| `--max-frames` | Number of frames to sample per video |
| `--load-in-4bit` | Use 4-bit NF4 quantization (default: enabled) |
| `--mage-video-backend` | Mage-VL video input: `frames` (uniform sampling, default) or `codec` |
| `--mage-codec-engine` | Mage-VL codec engine when using `codec` backend: `traditional` (HEVC) or `neural` (DCVC) |
| `--judge-model` | LLM judge model identifier (default: `openai/gpt-oss-120b:free` via OpenRouter) |

### Offline Judging

```bash
python main.py \
    --mode judge \
    --model-family qwen3vl \
    --model-id Qwen/Qwen3-VL-2B-Instruct \
    --data-level both \
    --output-dir ./results
```

To grade every `*_responses.jsonl` file found in the output directory in one shot (mapping tag prefixes to model families automatically), run:

```bash
./eval_all.sh
```

Set `JUDGE_BASE_URL` and `JUDGE_MODEL` in `eval_all.sh` (or as environment variables) to point the judge at your preferred OpenAI-compatible endpoint.

## Output Format

The pipeline produces three output files per evaluation run:

- `*_responses.jsonl` - Raw model responses with reference data
- `*_scores.jsonl` - Per-question scores and judge justifications
- `*_summary.json` - Aggregated metrics across all questions

## Requirements

- Python 3.12+
- CUDA-capable GPU(s)
- [uv](https://github.com/astral-sh/uv) package manager (for `run.sh`)
- OpenRouter API key (for LLM judge, set via `OPENROUTER_API_KEY` env var)

Model-specific dependencies are managed via separate virtual environments (`.venv-hulumed`, `.venv-qwen3vl`, and `.venv-magevl`). Run `./run.sh` to automatically set up all environments, including the CUDA 12.6 PyTorch trio and `mamba-ssm`/`causal-conv1d` build dependencies required by Mage-VL.
