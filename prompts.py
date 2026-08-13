# prompts.py
# Centralized prompt configurations for cataract surgery VLM evaluation

# =============================================================================
# INFERENCE SUFFIXES
# =============================================================================

# --- 1. Clip-Level Multiple-Choice Questions (YouTube MCQs) ---

# Suffix for CoT reasoning (reasoning trace before final letter)
CLIP_MCQ_COT_SUFFIX = (
    "\n\nInstructions: Think through the question step-by-step based on what "
    "you observe in the video. Be concise and do not repeat yourself. "
    "Conclude with your final answer on a new line in the format: ANSWER: <letter>\n"
    "Stop generating immediately after providing the answer."
)
# Backward-compatible alias
CLIP_INFERENCE_SUFFIX = CLIP_MCQ_COT_SUFFIX

# Suffix for Direct answering (only the option letter)
CLIP_MCQ_DIRECT_SUFFIX = (
    "\n\nInstructions: Answer the question by outputting only the letter of the correct option "
    "(e.g., A, B, C, or D) on a new line in the format: ANSWER: <letter>\n"
    "Provide no reasoning or extra text, and stop generating immediately."
)
# Backward-compatible alias
CLIP_DIRECT_INFERENCE_SUFFIX = CLIP_MCQ_DIRECT_SUFFIX


# --- 2. Phase Recognition (Phase Track Ontology P01-P13) ---

# Suffix for CoT reasoning
PHASE_COT_SUFFIX = (
    "\n\nInstructions: Think through the video step-by-step based on visible evidence from the surgical ontology. "
    "Conclude with your final answer on a new line in the format: Final answer: PXX\n"
    "Stop generating immediately after providing the answer."
)

# Suffix for Direct answering
PHASE_DIRECT_SUFFIX = (
    "\n\nInstructions: Identify the surgical phase from the surgical ontology. "
    "Output only the final ontology ID on a new line in the format: Final answer: PXX\n"
    "Provide no reasoning or extra text, and stop generating immediately."
)


# --- 3. Visual Description (YouTube & Phase Track Line 1) ---

# Suffix for Direct Visual Description
DESCRIPTION_DIRECT_SUFFIX = (
    "\n\nInstructions: Provide a concise, detailed, and factual description of the visible surgical actions, "
    "instruments, and anatomical changes in this clip. Focus strictly on what is directly visible. "
    "Do not include step-by-step reasoning preamble or meta-commentary."
)

# Suffix for Chain-of-Thought Visual Description
DESCRIPTION_COT_SUFFIX = (
    "\n\nInstructions: Analyze this clip step-by-step. First, systematically observe and identify all visible "
    "instruments and anatomical structures. Next, explain each surgical maneuver observed chronologically. "
    "Finally, synthesize a comprehensive, cohesive summary description of the surgical activity."
)


# --- 4. Full-Video Narration ---

# Suffix to append to full-video narration questions
NARRATION_INFERENCE_SUFFIX = (
    "\n\nInstructions: Describe the procedure as a single, flowing chronological narration of "
    "what you observe happening, step by step. Be concise, precise, and avoid repeating the same actions or getting stuck in loops. "
    "Do not include raw timestamps. Only describe the events you clearly see, and stop generating once the video concludes."
)


# =============================================================================
# VISUAL DESCRIPTION JUDGE PROMPTS
# =============================================================================

DESCRIPTION_JUDGE_SYSTEM_PROMPT = """You are a senior ophthalmic surgeon and surgical educator evaluating a Vision-Language Model's visual description of a cataract surgery video clip against an expert ground-truth reference description.

Evaluate the model's response across the following clinical criteria:
1. SURGICAL ACTIONS & MANEUVERS: Are the active surgical steps correctly identified and described?
2. INSTRUMENTS & TOOLS: Are the instruments used (or shown) accurately named and described?
3. ANATOMICAL STRUCTURES: Are the relevant intraocular anatomical structures and tissue responses accurately depicted?
4. FACTUALITY & HALLUCINATIONS: Does the model avoid hallucinating actions, instruments, or structures not supported by the reference?

Score the description on an integer scale of 0–5:
  5 — Excellent: Captures all key surgical actions, instruments, and anatomical interactions with clinical accuracy; no hallucinations.
  4 — Good: Accurately captures the primary surgical maneuver and visible tools; minor non-critical omissions.
  3 — Moderate: Broadly correct regarding the general step, but lacks specific details, misnames an instrument, or contains minor inaccuracies.
  2 — Poor: Misses the primary surgical action, misidentifies critical anatomy/instruments, or contains notable hallucinations.
  1 — Very Poor: Highly inaccurate or generic boilerplate that barely relates to the actual surgical scene.
  0 — Irrelevant / Contradictory: Completely wrong, describes unrelated surgery, or contradicts the reference.

Respond ONLY with a JSON object — no extra text, no markdown fences:
{
  "score": <integer 0-5>,
  "max_score": 5,
  "justification": "<one to two sentences explaining the score based on actions, instruments, and anatomy>"
}"""

DESCRIPTION_JUDGE_USER_TEMPLATE = """REFERENCE DESCRIPTION (ground truth):
{reference_description}

MODEL RESPONSE:
{model_response}

Score the model's visual description against the reference."""


# =============================================================================
# FULL-VIDEO NARRATION JUDGE PROMPTS
# =============================================================================

NARRATION_JUDGE_SYSTEM_PROMPT = """You are a senior ophthalmic surgeon and surgical education expert evaluating a Vision-Language Model's narration of a complete cataract surgery video.

You are given:
  1. A REFERENCE NARRATION — a ground-truth, expert description of every step that occurs in the video, in the order it occurs.
  2. The MODEL'S NARRATION — the model's freeform description of the same video, produced without access to the reference.

Judge how well the model's narration captures the actual surgical flow, using the reference as ground truth for WHAT happened and WHEN. Score each dimension with an integer 0-5:

1. STEP COVERAGE (0-5)
   5: Every major step/phase in the reference is mentioned, nothing significant omitted.
   3: Most major steps mentioned; one or two notable omissions.
   1: Only a minority of steps mentioned, or narration is too generic to map to specific steps.
   0: Steps mentioned are unrelated to, or contradict, the reference.

2. CHRONOLOGICAL ACCURACY (0-5)
   5: The narrated order exactly matches the reference's sequence of events.
   3: Order is mostly correct with one or two adjacent steps swapped or merged.
   1: Order is substantially scrambled relative to the reference.
   0: No coherent chronological structure, or order is reversed/random.

3. VISUAL & TECHNICAL ACCURACY (0-5)
   5: Specific instruments, tissue interactions, and maneuvers described match the reference's descriptions in substance (not wording).
   3: Broadly plausible and consistent with the reference but lacks specificity, or has minor inaccuracies.
   1: Generic/templated ("the surgeon carefully proceeds to the next step") that could apply to almost any cataract surgery, OR contains inaccuracies not supported by the reference.
   0: Actively contradicts the reference or describes steps/instruments absent from it.

4. NARRATIVE FLOW (0-5)
   5: A single fluid, well-transitioned narration reading naturally as a real-time account.
   3: Reasonably coherent but choppy, list-like, or repetitive.
   1: Disjointed fragments with little connective narration.
   0: Incoherent or not narration-style at all.

Be strict:
- Do NOT reward vague, generic, or boilerplate descriptions that don't demonstrate the model observed the specific events in the reference.
- Penalize hallucinated steps, instruments, or complications not present in the reference.
- Penalize narrations that merely restate the question or list option letters instead of narrating.
- A fluent but factually disconnected narration should score low on dimensions 1-3 regardless of style.

After scoring all four dimensions, give an OVERALL_SCORE (0-5) — your holistic judgment of narration quality and surgical-flow accuracy (need not be a simple average; weight factual/chronological correctness over style).

Respond ONLY with a JSON object — no extra text, no markdown fences:
{
  "step_coverage": <int 0-5>,
  "chronological_accuracy": <int 0-5>,
  "visual_technical_accuracy": <int 0-5>,
  "narrative_flow": <int 0-5>,
  "overall_score": <int 0-5>,
  "justification": "<2-3 sentence justification covering coverage, order, and accuracy>"
}"""

NARRATION_JUDGE_USER_TEMPLATE = """REFERENCE NARRATION (ground truth, chronological order):
{reference_narration}

MODEL'S NARRATION:
{model_response}

Score the model's narration."""


# =============================================================================
# DETERMINISTIC FALLBACK EXTRACTORS
# =============================================================================

CLIP_EXTRACTOR_SYSTEM_PROMPT = """You are a strict text parser extracting the final answer letter from a model's response to a multiple-choice question.
The model was instructed to output 'ANSWER: <letter>'. If it used a slightly different format (e.g., 'ANSWER:A', 'Answer is A', 'A.', 'Answer - A'), extract the letter.

Rules:
- Fix structured formatting misalignments (e.g., missing spaces, unusual delimiters).
- CRITICAL: Do NOT attempt to deduce the answer by reading the reasoning trace. ONLY extract the letter if it is explicitly provided as a final answer statement.
- If the model did not provide a final answer statement, return 'NONE'.

Respond ONLY with a JSON object — no extra text, no markdown fences:
{
  "extracted_answer": "<letter A-D, or 'NONE'>"
}"""

CLIP_EXTRACTOR_USER_TEMPLATE = """Model response:
{model_response}

Extract the final answer letter."""


PHASE_EXTRACTOR_SYSTEM_PROMPT = """You are a strict text parser extracting the final phase identifier (P01 to P13) from a model's response to a cataract surgery phase recognition question.
The model was instructed to output 'Final answer: PXX' or 'ANSWER: PXX'. If it used a slightly different format (e.g., 'P09', 'Answer: P09', 'Phase is P09'), extract the phase ID.

Rules:
- Only extract a valid phase ID in the set P01, P02, P03, P04, P05, P06, P07, P08, P09, P10, P11, P12, P13.
- CRITICAL: Do NOT attempt to deduce the phase by reading the reasoning trace. ONLY extract the phase if it is explicitly stated as the final answer.
- If no valid phase statement is found, return 'NONE'.

Respond ONLY with a JSON object — no extra text, no markdown fences:
{
  "extracted_answer": "<phase ID P01-P13, or 'NONE'>"
}"""

PHASE_EXTRACTOR_USER_TEMPLATE = """Model response:
{model_response}

Extract the final phase identifier."""