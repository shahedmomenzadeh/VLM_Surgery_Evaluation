# Cataract Surgery VLM Benchmark: Experimental Results & Academic Reference

This directory contains the official evaluation outputs, raw model predictions, granular scores, and aggregated summaries for Vision-Language Models (VLMs) benchmarked on cataract surgery video understanding.

---

## 1. Benchmark Overview & Evaluation Protocol

The benchmark evaluates surgical video understanding across **two temporal granularities** and **five distinct task formulations**:

1. **Clip-Level Tasks (989 records, 0–30s video clips)**:
   - **Visual Description** ($n=293$): Open-ended surgical scene description evaluated against ground truth by an expert LLM judge.
   - **Multiple-Choice Questions (MCQ)** ($n=486$): Categorical surgical identification across 3 subtasks ($n=162$ each):
     - *Step Identification*
     - *Instrument Identification*
     - *Visual Observation / Anatomical Cue Detection*
   - **Phase Understanding** ($n=210$): Precise temporal and categorical phase recognition across 4 subtasks:
     - *Boundary Detection* ($n=49$)
     - *Temporal Localization* ($n=54$)
     - *Timestamp-to-Phase* ($n=47$)
     - *Contextual Phase Recognition* ($n=60$)

2. **Full-Video Procedural Narration (15 records, complete uncut surgeries)**:
   - Evaluated across 5 clinical dimensions against expert surgical narration.

All records instruct the model to produce a strict JSON output contract: `{"explanation": "...", "answer": ...}`. Evaluation is strictly zero-shot/direct inference (one prediction per record).

LLM-judge scoring (visual description + full-video narration) was performed with **Gemini-3.8-flash**; MCQ and phase tasks are scored deterministically.

---

## 2. Formal Task Definitions & Scoring Metrics

### 2.1. Multiple-Choice Questions (MCQ)
Models must select the correct single letter ($A$–$D$) within the JSON `answer` field.

```math
\mathcal{R}_{\text{task}} = \begin{cases} 1.0 & \text{if } \text{normalize}(\hat{y}) = y^* \\ 0.0 & \text{otherwise} \end{cases}
```

Scores are strictly deterministic.

---

### 2.2. Phase Understanding (13-Phase Clinical Ontology)
All phase tasks utilize the standardized 13-phase cataract surgery ontology:

```math
\text{P01 (Incision)} \dots \text{P13 (Idle)}
```

#### A. Boundary Detection ($n=49$)
The model predicts a clip-local boundary timestamp $\hat{t}$ in seconds. The reward applies an exponential decay parameterized by tolerance threshold $\tau = 1.5\text{ s}$:

```math
\mathcal{R}_{\text{task}} = \exp\left(-\frac{|\hat{t} - t^*|}{1.5}\right) \in (0, 1]
```

#### B. Temporal Localization ($n=54$)
The model predicts a time interval $[\hat{s}, \hat{e}]$ representing the active duration of a queried phase. Scored using Intersection-over-Union (IoU):

```math
\mathcal{R}_{\text{task}} = \text{IoU}([\hat{s}, \hat{e}], [s^*, e^*]) = \frac{\max(0, \min(\hat{e}, e^*) - \max(\hat{s}, s^*))}{(\hat{e} - \hat{s}) + (e^* - s^*) - \text{Intersection}}
```

#### C. Timestamp-to-Phase & Contextual Phase Recognition ($n=107$)
The model identifies the active surgical phase ID from visual frames or surrounding temporal context:

```math
\mathcal{R}_{\text{task}} = \begin{cases} 1.0 & \text{if } \hat{y}_{\text{phase}} = y^*_{\text{phase}} \\ 0.0 & \text{otherwise} \end{cases}
```

#### D. Strict JSON Format Bonus (Phase Tasks Only)
To incentivize structured clinical reporting, a decoupled format reward $\mathcal{R}_{\text{fmt}} \in \{0, 1\}$ awards 1.0 if the output strictly parses as a valid JSON object containing exactly `{"explanation", "answer"}` without markdown fences:

```math
\mathcal{R}_{\text{total}} = \mathcal{R}_{\text{task}} + 0.05 \cdot \mathcal{R}_{\text{fmt}} \quad (\text{Max Score} = 1.05)
```

The normalized task score is computed as $\mathcal{S}_{\text{norm}} = \frac{\mathcal{R}_{\text{total}}}{1.05}$.

---

### 2.3. Visual Description (LLM-as-a-Judge)
Open-ended descriptions are evaluated by an ophthalmic surgical LLM judge against ground-truth descriptions across four clinical criteria:
1. **Surgical Actions & Maneuvers**: Accurate identification of operative steps.
2. **Instruments & Tools**: Correct naming and interaction description.
3. **Anatomical Structures**: Accurate recognition of intraocular structures.
4. **Factuality & Hallucination Prevention**: Strict penalization of unsupported claims.

- **Scale**: Integer $0 \text{ to } 5$ ($5 = \text{Excellent}, 0 = \text{Irrelevant/Contradictory}$).
- **Normalized Score**: $\mathcal{S}_{\text{norm}} = \frac{\text{Score}}{5} \in [0.0, 1.0]$.

---

### 2.4. Full-Video Procedural Narration (LLM-as-a-Judge)
Complete surgeries (5–25 minutes) require generating a comprehensive chronological narration. The judge evaluates five dimensions ($0 \text{ to } 5$ scale):
1. **Step Coverage**: Percentage and completeness of major surgical phases captured.
2. **Chronological Accuracy**: Sequence alignment relative to actual surgical flow.
3. **Visual & Technical Accuracy**: Correctness of specific tissue interactions and tool manipulations.
4. **Narrative Flow**: Fluid, real-time surgical commentary vs. disjointed bullet points.
5. **Overall Score**: Holistic surgical competence judgment (weighted towards procedural veracity).

---

## 3. Benchmark Leaderboard & Comparative Results

### Table 1: Overall Model Performance Summary
*All clip-level scores are reported as normalized accuracy $[0.0, 1.0]$. Narration overall is reported on the native $0 \text{ to } 5$ clinical scale (with normalized value in brackets). LLM-judge tasks (Visual Desc, Narration) scored with Gemini-3.8-flash.*

| Model Architecture | Parameters | Modality / Focus | Overall Clip Acc ($n=989$) | MCQ Macro ($n=486$) | Phase Macro ($n=210$) | Visual Desc ($n=293$) | Full Narration Overall ($n=15$) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Qwen3-VL-2B-Cataract-SFT** | 2B | Domain SFT | **0.5376** | **0.8354** | 0.1925 | 0.2935 | 0.3333 [0.0667] |
| **Hulu-Med-4B** | 4B | Medical Specialist | 0.4376 | 0.6502 | 0.1953 | 0.2594 | 0.2000 [0.0400] |
| **Qwen3-VL-2B-Thinking** | 2B | Generalist (Reasoning) | 0.4338 | 0.6235 | 0.1754 | 0.3044 | 0.1333 [0.0267] |
| **Lingshu-7B** | 7B | Medical Specialist | 0.4208 | 0.6090 | 0.2157 | 0.2601 | **0.8000 [0.1600]** |
| **Qwen3-VL-4B-Instruct** | 4B | Generalist (Instruct) | 0.4059 | 0.5926 | 0.0825 | **0.3276** | 0.1333 [0.0267] |
| **Hulu-Med-7B** | 7B | Medical Specialist | 0.3978 | 0.5453 | **0.2171** | 0.2826 | 0.6000 [0.1200] |
| **Qwen3-VL-2B-Instruct** | 2B | Generalist (Instruct) | 0.3709 | 0.5309 | 0.1631 | 0.2539 | 0.1333 [0.0267] |

---

### Table 2: Granular Clip-Level Performance Breakdown
*Per-task normalized average accuracy across all 8 clip-level subtasks.*

| Model | Visual Desc ($n=293$) | Step ID ($n=162$) | Instrument ID ($n=162$) | Visual Obs ($n=162$) | Boundary Det. ($n=49$) | Temporal Loc. ($n=54$) | Timestamp $\to$ Phase ($n=47$) | Context Phase ($n=60$) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Qwen3-VL-2B-Cataract-SFT** | 0.2935 | **0.7778** | **0.8395** | **0.8889** | 0.2457 | 0.2237 | 0.1894 | 0.1111 |
| **Hulu-Med-4B** | 0.2594 | 0.4630 | 0.6358 | 0.8519 | 0.1061 | 0.3182 | **0.2300** | 0.1270 |
| **Qwen3-VL-2B-Thinking** | 0.3044 | 0.3519 | 0.6852 | 0.8333 | 0.2012 | 0.1928 | 0.1489 | 0.1587 |
| **Lingshu-7B** | 0.2601 | 0.4444 | 0.6481 | 0.7346 | **0.3170** | 0.2566 | 0.2097 | 0.0793 |
| **Qwen3-VL-4B-Instruct** | **0.3276** | 0.3148 | 0.6543 | 0.8086 | 0.0577 | 0.0530 | 0.1084 | 0.1111 |
| **Hulu-Med-7B** | 0.2826 | 0.3333 | 0.5494 | 0.7531 | 0.0859 | **0.4618** | 0.2097 | 0.1111 |
| **Qwen3-VL-2B-Instruct** | 0.2539 | 0.3704 | 0.5432 | 0.6790 | 0.2170 | 0.1161 | 0.1287 | **0.1905** |

---

### Table 3: Full-Video Procedural Narration Multi-Dimensional Breakdown
*Scores evaluated on complete surgical procedures across 5 clinical dimensions on a $0 \text{ to } 5$ scale.*

| Model | Step Coverage (/5) | Chronological Accuracy (/5) | Visual & Tech Accuracy (/5) | Narrative Flow (/5) | Overall Score (/5) | Normalized Score (/1.0) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Lingshu-7B** | **0.8000** | **0.8000** | 0.4000 | **2.4667** | **0.8000** | **0.1600** |
| **Hulu-Med-7B** | 0.6000 | 0.7333 | **0.6000** | 2.0000 | 0.6000 | 0.1200 |
| **Qwen3-VL-2B-Cataract-SFT** | 0.3333 | 0.4000 | 0.4000 | 1.3333 | 0.3333 | 0.0667 |
| **Hulu-Med-4B** | 0.2000 | 0.2667 | 0.1333 | 1.6667 | 0.2000 | 0.0400 |
| **Qwen3-VL-4B-Instruct** | 0.2000 | 0.1333 | 0.2667 | 1.0667 | 0.1333 | 0.0267 |
| **Qwen3-VL-2B-Thinking** | 0.0667 | 0.1333 | 0.2000 | 0.8000 | 0.1333 | 0.0267 |
| **Qwen3-VL-2B-Instruct** | 0.2667 | 0.2000 | 0.2000 | 0.6667 | 0.1333 | 0.0267 |

---

## 4. Key Academic Findings & Discussion

1. **Targeted Surgical SFT Delivers State-of-the-Art Clip Understanding**:
   - **Qwen3-VL-2B-Cataract-SFT** achieves **0.5376** overall clip accuracy, outperforming the previous top model (**Hulu-Med-4B** at **0.4376**) by an absolute **+10.00%** and the base **Qwen3-VL-2B-Instruct** (**0.3709**) by **+16.67%**.
   - This leap is driven by surgical multiple-choice precision (**0.8354** MCQ macro vs. 0.6502 for Hulu-Med-4B), with unprecedented gains on **Step Identification** (**0.7778** vs. 0.4630, a **+31.48%** absolute jump) and **Instrument Identification** (**0.8395** vs. 0.6852, **+15.43%**).
   - These results demonstrate that lightweight 2B foundation backbones fine-tuned directly on cataract-specific multimodal tuples can drastically outperform larger 4B–7B medical generalists on discrete domain recognition.

2. **Procedural Narration Across Granularities**:
   - **Lingshu-7B** (**0.8000 / 5**) and **Hulu-Med-7B** (**0.6000 / 5**) maintain their lead on uncut full-video procedural narration, benefiting from broader pretraining contexts that preserve long-range narrative continuity and chronological step flow.
   - However, **Qwen3-VL-2B-Cataract-SFT** achieves **0.3333 / 5** (normalized 0.0667), making it the highest-performing model under 7B on procedural narration—surpassing Hulu-Med-4B (0.2000) and achieving 2.5× the score of base Qwen3-VL 2B/4B instruct models (0.1333) with doubled chronological accuracy (0.4000 vs. 0.2000).

3. **Visual Description Quality & Reasoning Trade-offs**:
   - **Qwen3-VL-4B-Instruct** (**0.3276**), **Qwen3-VL-2B-Thinking** (**0.3044**), and **Qwen3-VL-2B-Cataract-SFT** (**0.2935**) form the top tier on short-clip visual descriptions, capturing fine intraocular anatomical detail and tool visibility.
   - Test-time reasoning in Qwen3-VL-2B-Thinking provides strong clip-level gains over the base instruct model (**0.4338** vs. **0.3709**), but targeted domain SFT achieves a substantially higher clip performance ceiling (**0.5376**) without test-time compute overhead.

4. **Temporal Grounding Remains the Critical Frontier**:
   - Despite SFT's massive recognition gains, continuous temporal tasks remain challenging across all models: **Temporal Localization** (cohort average: 0.232) and **Boundary Detection** (cohort average: 0.176) lag far behind static recognition like **Visual Observation** (cohort average: 0.793).
   - SFT yields moderate gains in boundary detection (0.2457 vs. 0.2170 base) and temporal localization (0.2237 vs. 0.1161 base), but domain regression on continuous timestamp tokens remains the primary target for next-generation surgical video foundation models.

---

## 5. Artifact Directory Structure

For every model evaluation run, the pipeline generates three synchronized files:

```
results/
├── <tag>_clip_responses.jsonl   # Raw model outputs, reference labels, and prompts
├── <tag>_clip_scores.jsonl      # Itemized scores, extracted answers, and judge justifications
├── <tag>_clip_summary.json     # Aggregated clip-level metrics
├── <tag>_full_responses.jsonl   # Raw full-video narration outputs
├── <tag>_full_scores.jsonl      # Multi-dimensional narration scores per surgery
├── <tag>_full_summary.json     # Aggregated narration metrics
├── judge_<tag>.log              # Detailed execution and judge API communication logs
└── archive/                     # Preserved past scoring runs
```

---

## 6. Citation & Reproducibility

If you use these benchmark results or evaluation protocols in academic work, please cite:

```bibtex
@article{cataract_vlm_benchmark2026,
  title={Benchmarking Vision-Language Models for Cataract Surgery: From Clip-Level Phase Recognition to Full-Video Procedural Narration},
  author={Evaluation Team},
  journal={arXiv preprint},
  year={2026}
}
```
