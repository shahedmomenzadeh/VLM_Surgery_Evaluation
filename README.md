# VLM Surgery Evaluation

A framework for benchmarking Vision-Language Models (VLMs) on cataract surgery video understanding tasks. This pipeline generates model responses to structured surgical questions and evaluates them using a combination of deterministic scoring and LLM-as-a-judge evaluation.

## Overview

This project evaluates how well VLMs understand cataract surgery videos across two granularities:

- **Clip-level**: Short surgical video segments evaluating open-ended visual descriptions, multi-choice surgical questions (step, instrument, visual cue), and phase recognition against a fixed 13-phase ontology.
- **Full-video level**: Complete surgery recordings requiring comprehensive procedural narration.

Currently supported model families:
- [HuluMed](https://huggingface.co/ZJU-AI4H/Hulu-Med-4B) (ZJU-AI4H)
- [Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) (Alibaba)
- [Lingshu-7B](https://huggingface.co/lingshu-medical-mllm/Lingshu-7B) (Qwen2.5-VL based medical MLLM)
- [Mage-VL](https://huggingface.co/microsoft/Mage-VL) (Microsoft, codec-native video MLLM)

## VLM Task Categories & Prompting Strategies

Every task is evaluated under two complementary prompting strategies:
- **Direct**: Model provides the final answer or direct description without reasoning preamble.
- **Chain-of-Thought (CoT)**: Model generates step-by-step clinical observations and reasoning before concluding with the final answer.

---

### 1. Clip-Level Visual Description (YouTube & Phase Tracks)
- **Source**: First line of `clip_*_sft.jsonl`.
- **Description**: Open-ended visual description of surgical actions, instruments, and anatomical structures.
- **Evaluation**: **LLM-as-a-judge** (0–5 scale rubric) comparing the model's description against ground-truth clinical descriptions.

### 2. Clip-Level Multiple-Choice Questions (YouTube Track)
- **Source**: `clip_*_grpo.jsonl` (3 MCQs per clip).
- **Questions**: Step Identification, Instrument Identification, and Visual Observation / Cue.
- **Evaluation**: **Deterministic scoring** (exact option match `A`–`D`).

### 3. Clip-Level Phase Recognition (Phase Track `PH_*`)
- **Source**: `clip_*_grpo.jsonl` (1 question per phase clip).
- **Questions**: Identifies surgical phase against the 13-phase cataract ontology (`P01`–`P13`).
- **Evaluation**: **Deterministic scoring** (exact phase match `P01`–`P13`).

### 4. Full-Video Procedural Narration
- **Source**: `full_video_sft.jsonl`.
- **Description**: Flowing, chronological narration of the complete uncut surgical procedure.
- **Evaluation**: **LLM-as-a-judge** across 4 clinical dimensions (0–5 each: Step Coverage, Chronological Accuracy, Visual/Technical Accuracy, Narrative Flow) plus an Overall Score.

---

## Evaluation Metrics Summary

| Task Category | Prompt Variants | Evaluation Method | Metric / Scale |
|---|---|---|---|
| **Visual Description** | Direct, CoT | LLM-as-a-judge | 0–5 Clinical Rubric (Normalized 0–1) |
| **YouTube MCQs** (Step, Instrument, Visual Cue) | Direct, CoT | Deterministic | Exact Match (0 or 1) |
| **Phase Recognition** (`P01`–`P13`) | Direct, CoT | Deterministic | Exact Match (0 or 1) |
| **Full-Video Narration** | Standard | LLM-as-a-judge | 5 Dimensions (0–5 scale each) |

---

## Project Structure

```
.
├── main.py                 # CLI entry point and orchestrator
├── dataset_loader.py       # Loads clip-level and full-video records
├── prompts.py              # All inference and judge prompt templates
├── llm_judge.py            # LLM judge scoring and deterministic metric computation
├── hulumed_inference.py    # HuluMed model inference pipeline
├── qwen3VL_inference.py    # Qwen3-VL model inference pipeline
├── lingshu_inference.py    # Lingshu-7B / Qwen2.5-VL inference pipeline
├── mage_vl_inference.py    # Mage-VL inference pipeline (frames & codec backends)
├── run.sh                  # Environment setup and execution script
├── eval_all.sh             # Automated offline judge runner over all response files
└── results/                # Generated responses, scores, and summaries
```

---

## Usage

### Quick Start

```bash
# Set your HuggingFace token for gated models
export HF_TOKEN="your_token_here"

# Set provider API key for LLM judge (e.g., OpenRouter)
export PROVIDER_API_KEY="your_api_key_here"

# Run full evaluation (inference + judging)
python main.py \
    --mode all \
    --model-family qwen3vl \
    --model-id Qwen/Qwen3-VL-2B-Instruct \
    --dataset-root ./dataset \
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

### Offline Judging

```bash
python main.py \
    --mode judge \
    --model-family qwen3vl \
    --model-id Qwen/Qwen3-VL-2B-Instruct \
    --data-level both \
    --output-dir ./results
```

To grade every `*_responses.jsonl` file found in the output directory in one shot:

```bash
./eval_all.sh
```

---

## Output Format

The pipeline produces three output files per evaluation run:
- `*_responses.jsonl` - Raw model responses with reference data and prompts.
- `*_scores.jsonl` - Per-task scores, normalized accuracies, and judge justifications.
- `*_summary.json` - Aggregated metrics across all task categories and prompting modes.
