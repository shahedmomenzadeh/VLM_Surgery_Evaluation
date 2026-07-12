# prompts.py
# Centralized prompt configurations for cataract surgery VLM evaluation

# =============================================================================
# INFERENCE SUFFIXES
# =============================================================================

# Suffix to append to clip-level multiple-choice questions
CLIP_INFERENCE_SUFFIX = (
    "\n\nInstructions: Think through the question step-by-step based on what "
    "you observe in the video. Be concise and do not repeat yourself. "
    "Conclude with your final answer on a new line in the format: ANSWER: <letter>\n"
    "Stop generating immediately after providing the answer."
)

# Suffix to append to clip-level multiple-choice questions for direct answering (no reasoning)
CLIP_DIRECT_INFERENCE_SUFFIX = (
    "\n\nInstructions: Answer the question by outputting only the letter of the correct option "
    "(e.g., A, B, C, or D) on a new line in the format: ANSWER: <letter>\n"
    "Provide no reasoning or extra text, and stop generating immediately."
)

# Suffix to append to full-video narration questions
NARRATION_INFERENCE_SUFFIX = (
    "\n\nInstructions: Describe the procedure as a single, flowing chronological narration of "
    "what you observe happening, step by step. Be concise, precise, and avoid repeating the same actions or getting stuck in loops. "
    "Do not include raw timestamps. Only describe the events you clearly see, and stop generating once the video concludes."
)

# Suffix to append for direct sequence ordering (no reasoning)
ORDERING_DIRECT_INFERENCE_SUFFIX = (
    "\n\nInstructions: Output only the correct sequence of letters separated by commas "
    "(e.g., B, D, A, C).\n"
    "Provide no reasoning or extra text, and stop generating immediately."
)

# Suffix to append for Visual Chain-of-Thought (CoT) sequence ordering (reasoning first)
ORDERING_COT_INFERENCE_SUFFIX = (
    "\n\nInstructions: Analyze the video and think through the chronological flow "
    "step-by-step. First, describe the sequence of surgical steps you observe in the "
    "video concisely without repeating yourself. Conclude with your final answer on a new line in the format: "
    "SEQUENCE: <letters separated by commas> (e.g., SEQUENCE: B, D, A, C).\n"
    "Stop generating immediately after providing the sequence."
)



# =============================================================================
# CLIP-LEVEL JUDGE PROMPTS (visual_observation)
# =============================================================================

CLIP_JUDGE_SYSTEM_PROMPT = """You are an expert surgical educator evaluating a Vision-Language Model's answer to a multiple-choice question about a cataract surgery video clip.

Your job is to score the model's response on a scale of 0–3:

  3 — Correct letter AND reasoning clearly describes accurate visual details
  2 — Correct letter but reasoning is vague, generic, or partially wrong
  1 — Wrong letter but reasoning shows partial understanding of the visual scene
  0 — Wrong letter and reasoning is incorrect or irrelevant

Be strict: do NOT give credit for lucky correct answers backed by bad reasoning.

Respond ONLY with a JSON object — no extra text, no markdown fences:
{
  "score": <integer 0-3>,
  "extracted_answer": "<letter the model gave, or 'NONE'>",
  "justification": "<one sentence>"
}"""

CLIP_JUDGE_USER_TEMPLATE = """Question asked to the model:
{question_text}

Correct answer: {correct_answer}
Reference reasoning: {reference_reasoning}

Model response:
{model_response}

Score the model response."""


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
# FULL-VIDEO SEQUENCE ORDERING PROMPTS
# =============================================================================

ORDERING_JUDGE_SYSTEM_PROMPT = """You are extracting a Vision-Language Model's predicted ordering of surgical steps from its raw response to a step-reordering question about a cataract surgery video.

The model was shown a list of lettered surgical step descriptions (in shuffled order) and asked to output the letters in the chronological order the steps actually occur in the video.

Your job: read the model's response and extract the FINAL sequence of letters it intended as its structured answer.

Rules:
- Output every distinct letter from the valid option set below exactly once, in the order the model intends.
- Fix any structured formatting misalignments (e.g., missing spaces, incorrect delimiters like A,B,C instead of A, B, C).
- CRITICAL: Do NOT attempt to deduce or extract the answer from the model's reasoning trace. ONLY extract the final committed answer sequence. If the model did not provide a final answer statement, return an empty list.
- Do not include letters outside the valid option set.
- Do not add letters the model never mentioned.

Valid option set: {valid_letters}

Respond ONLY with a JSON object — no extra text, no markdown fences:
{{
  "predicted_sequence": ["<letter>", "<letter>", ...],
  "complete": <true/false>
}}"""

ORDERING_JUDGE_USER_TEMPLATE = """Question shown to the model:
{question_text}

Model response:
{model_response}

Extract the predicted chronological ordering."""

# =============================================================================
# CLIP-LEVEL DETERMINISTIC FALLBACK EXTRACTOR
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