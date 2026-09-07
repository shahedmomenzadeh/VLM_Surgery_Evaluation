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
| **Hulu-Med-4B** | 4B | Medical Specialist | **0.4376** | **0.6502** | 0.1953 | 0.2594 | 0.2000 [0.0400] |
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
| **Hulu-Med-4B** | 0.2594 | **0.4630** | 0.6358 | **0.8519** | 0.1061 | 0.3182 | **0.2300** | 0.1270 |
| **Qwen3-VL-2B-Thinking** | 0.3044 | 0.3519 | **0.6852** | 0.8333 | 0.2012 | 0.1928 | 0.1489 | 0.1587 |
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
| **Hulu-Med-4B** | 0.2000 | 0.2667 | 0.1333 | 1.6667 | 0.2000 | 0.0400 |
| **Qwen3-VL-4B-Instruct** | 0.2000 | 0.1333 | 0.2667 | 1.0667 | 0.1333 | 0.0267 |
| **Qwen3-VL-2B-Thinking** | 0.0667 | 0.1333 | 0.2000 | 0.8000 | 0.1333 | 0.0267 |
| **Qwen3-VL-2B-Instruct** | 0.2667 | 0.2000 | 0.2000 | 0.6667 | 0.1333 | 0.0267 |

---

## 4. Key Academic Findings & Discussion

1. **Medical Specialization Dominates Long-Horizon Procedural Flow**:
   - **Lingshu-7B** and **Hulu-Med-7B** lead full-video procedural narration (**0.8000** and **0.6000 / 5** overall), demonstrating that domain pretraining on surgical video helps maintain phase progression and chronological coherence across uncut recordings.
   - Generalist models (Qwen3-VL 2B/4B) struggle with full-procedure context, often producing repetitive descriptions or hallucinating phase transitions, resulting in low narrative flow and coverage scores ($0.1333 / 5$).

2. **Visual Description Quality & Reasoning Trade-offs**:
   - **Qwen3-VL-4B-Instruct** (**0.3276**) and **Qwen3-VL-2B-Thinking** (**0.3044**) achieve top performance on short-clip visual descriptions, capturing fine intraocular anatomical detail and tool visibility.
   - Test-time thinking improves clip-level precision (**0.4338** for Qwen3-VL-2B-Thinking vs. **0.3709** for Instruct), particularly on Instrument Identification (**0.6852** vs. **0.5432**) and Visual Observation (**0.8333** vs. **0.6790**).

3. **Temporal Grounding Remains the Critical Frontier**:
   - All models exhibit significant gaps on continuous temporal tasks: **Temporal Localization** (average: 0.233) and **Boundary Detection** (average: 0.164) remain far lower than categorical **Visual Observation** (average: 0.777).
   - Video sampling density and continuous timestamp regression remain primary targets for fine-tuning future surgical foundation models.

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
