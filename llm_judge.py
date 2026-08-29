# llm_judge.py
# OpenAI/OpenRouter compatible LLM judge implementation for VLM evaluation scoring
#
# Scoring model for the flat evaluation dataset:
#   - visual_description  -> LLM judge rubric 0-5 (normalized /5)
#   - narration           -> LLM judge 5 dimensions 0-5 (normalized /5)
#   - mcq (letter A-D)    -> deterministic exact match, R_task in {0,1}
#   - phase tasks         -> deterministic (boundary exp decay / temporal IoU / phase-id match)
#                           + format bonus: R_total = R_task + 0.05 * R_fmt (max 1.05)

import re
import os
import json
import time
import math
import logging
from openai import OpenAI

from prompts import (
    DESCRIPTION_JUDGE_SYSTEM_PROMPT,
    DESCRIPTION_JUDGE_USER_TEMPLATE,
    NARRATION_JUDGE_SYSTEM_PROMPT,
    NARRATION_JUDGE_USER_TEMPLATE,
    CLIP_EXTRACTOR_SYSTEM_PROMPT,
    CLIP_EXTRACTOR_USER_TEMPLATE,
    PHASE_EXTRACTOR_SYSTEM_PROMPT,
    PHASE_EXTRACTOR_USER_TEMPLATE,
    BOUNDARY_EXTRACTOR_SYSTEM_PROMPT,
    BOUNDARY_EXTRACTOR_USER_TEMPLATE,
    INTERVAL_EXTRACTOR_SYSTEM_PROMPT,
    INTERVAL_EXTRACTOR_USER_TEMPLATE
)

log = logging.getLogger("llm_judge")

# Phase-understanding task types scored deterministically
PHASE_TASK_TYPES = {
    "boundary_detection",
    "temporal_localization",
    "timestamp_to_phase",
    "contextual_phase_recognition",
}


# =============================================================================
# TEXT / JSON PARSING HELPERS
# =============================================================================

def _to_float(value):
    """Converts a value to float, returns None on failure."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_strict_json(text: str):
    """
    Attempts to parse `text` as a strict JSON object.
    Returns (obj, is_strict) — is_strict is True only when the raw text is a
    valid JSON object with no surrounding prose/markdown fences.
    """
    text = (text or "").strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj, True
    except json.JSONDecodeError:
        pass
    return None, False


def extract_json_object(text: str):
    """
    Tolerant JSON object extraction: strips code fences, tries a full parse,
    then falls back to the first balanced {...} block in the text.
    Returns a dict or None.
    """
    if not text:
        return None
    cleaned = re.sub(r"```(?:json)?|```", "", text).strip()
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(cleaned)):
            if cleaned[i] == "{":
                depth += 1
            elif cleaned[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(cleaned[start:i + 1])
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        pass
                    break
    return None


def extract_answer_letter(text: str) -> str:
    """
    Extracts the answer letter (A-D) from a model response.
    Priority: explicit 'ANSWER: X' -> 'answer is X' -> last bare A/B/C/D.
    Returns '' if nothing found.
    """
    m = re.search(r"ANSWER\s*:\s*([A-D])", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    m = re.search(
        r"(?:answer\s+is\s+|^)([A-D])[).\s]",
        text, re.IGNORECASE | re.MULTILINE,
    )
    if m:
        return m.group(1).upper()

    # Last resort: an isolated UPPERCASE A-D token (avoids matching the word "a").
    letters = re.findall(r"\b([A-D])\b", text)
    return letters[-1] if letters else ""


def extract_answer_phase(text: str) -> str:
    """
    Extracts the surgical phase identifier (P01-P13) from a model response.
    Priority: 'Final answer: PXX' -> 'ANSWER: PXX' -> 'answer is PXX' -> last isolated PXX.
    Returns '' if nothing found.
    """
    m = re.search(r"(?:Final answer|ANSWER)\s*:\s*(P0[1-9]|P1[0-3])", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    m = re.search(
        r"(?:(?:final\s+)?answer\s+is\s+|phase\s+is\s+|phase\s*:?\s*)(P0[1-9]|P1[0-3])\b",
        text, re.IGNORECASE
    )
    if m:
        return m.group(1).upper()

    phases = re.findall(r"\b(P0[1-9]|P1[0-3])\b", text.upper())
    return phases[-1] if phases else ""


def build_score_record(resp_id: str, task_category: str, question_type: str,
                       reward_type: str, correct_answer, score_info: dict) -> dict:
    """Assembles a uniform scores.jsonl record from a judge score_info dict."""
    normalised = round(score_info["score"] / score_info["max_score"], 4) if score_info.get("max_score", 0) > 0 else 0.0
    record = {
        "record_id": resp_id,
        "task_category": task_category,
        "question_type": question_type,
        "reward_type": reward_type,
        "correct_answer": correct_answer,
        "extracted_answer": score_info["extracted_answer"],
        "score": score_info["score"],
        "max_score": score_info["max_score"],
        "normalised_score": normalised,
        "correct": score_info["correct"],
        "method": score_info["method"],
        "justification": score_info.get("justification", ""),
    }
    for extra in ("task_score", "format_valid", "format_bonus"):
        if extra in score_info:
            record[extra] = score_info[extra]
    return record


def truncate_model_response(model_response: str, max_tokens: int = 2048) -> str:
    """
    Truncates a model response if it exceeds ~max_tokens (approx 4 chars per token).
    Prevents prompt context overflow when evaluating hallucinatory/looping VLM outputs.
    """
    if not model_response:
        return ""
    max_chars = max_tokens * 4
    if len(model_response) > max_chars:
        truncated = model_response[:max_chars]
        answer_suffix_match = re.search(r"(?:ANSWER|Final answer)\s*:.*$", model_response, re.IGNORECASE)
        suffix = f"\n\n[TRUNCATED: Response exceeded {max_tokens} tokens]"
        if answer_suffix_match and answer_suffix_match.group(0) not in truncated:
            suffix += f"\n{answer_suffix_match.group(0)}"
        return truncated + suffix
    return model_response


class LLMJudge:
    """
    Unified LLM judge class that handles deterministic scoring and API-based LLM grading
    for both clip-level and full-video evaluations.
    """
    def __init__(self, base_url: str, api_key: str, model: str, retries: int = 3, num_workers: int = 3):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.retries = retries
        self.num_workers = num_workers

        api_key = (api_key or "").strip()
        # Local self-hosted endpoints (vLLM / llama.cpp / etc.) typically ignore
        # the Authorization header but the OpenAI client still requires a non-empty
        # key string — use a placeholder for localhost/loopback only.
        if not api_key and base_url and ("localhost" in base_url or "127.0.0.1" in base_url):
            api_key = "local-no-key"

        # Initialize OpenAI client if api_key is available
        if api_key:
            self.client = OpenAI(base_url=self.base_url, api_key=api_key)
        else:
            self.client = None
            log.warning("No API key provided for LLMJudge. LLM-based scoring will fall back to deterministic scoring.")

    # -------------------------------------------------------------------------
    # LLM extractor fallbacks (only invoked when regex/JSON parsing fails)
    # -------------------------------------------------------------------------

    def _extract_clip_letter_llm(self, model_response: str) -> str:
        """Fallback to LLM to extract MCQ answer letter for structured misalignments."""
        if not self.client:
            return ""

        model_response = truncate_model_response(model_response, max_tokens=2048)
        user_msg = CLIP_EXTRACTOR_USER_TEMPLATE.format(model_response=model_response)

        for attempt in range(1, self.retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=1024,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": CLIP_EXTRACTOR_SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg}
                    ]
                )
                raw = response.choices[0].message.content.strip()
                raw = re.sub(r"```(?:json)?|```", "", raw).strip()
                result = json.loads(raw)
                ans = result.get("extracted_answer", "").strip().upper()
                if ans in ["A", "B", "C", "D"]:
                    return ans
                return ""
            except Exception as e:
                log.warning(f"Clip MCQ extractor API attempt {attempt}/{self.retries} failed: {e}")
            if attempt < self.retries:
                time.sleep(1)
        return ""

    def _extract_phase_llm(self, model_response: str) -> str:
        """Fallback to LLM to extract phase identifier (P01-P13) for structured misalignments."""
        if not self.client:
            return ""

        model_response = truncate_model_response(model_response, max_tokens=2048)
        user_msg = PHASE_EXTRACTOR_USER_TEMPLATE.format(model_response=model_response)

        for attempt in range(1, self.retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=1024,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": PHASE_EXTRACTOR_SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg}
                    ]
                )
                raw = response.choices[0].message.content.strip()
                raw = re.sub(r"```(?:json)?|```", "", raw).strip()
                result = json.loads(raw)
                ans = result.get("extracted_answer", "").strip().upper()
                valid_phases = {f"P{i:02d}" for i in range(1, 14)}
                if ans in valid_phases:
                    return ans
                return ""
            except Exception as e:
                log.warning(f"Phase extractor API attempt {attempt}/{self.retries} failed: {e}")
            if attempt < self.retries:
                time.sleep(1)
        return ""

    def _extract_timestamp_llm(self, model_response: str) -> str:
        """Fallback to LLM to extract a boundary timestamp (seconds). Returns float-like str or ''."""
        if not self.client:
            return ""

        model_response = truncate_model_response(model_response, max_tokens=2048)
        user_msg = BOUNDARY_EXTRACTOR_USER_TEMPLATE.format(model_response=model_response)

        for attempt in range(1, self.retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=1024,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": BOUNDARY_EXTRACTOR_SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg}
                    ]
                )
                raw = response.choices[0].message.content.strip()
                raw = re.sub(r"```(?:json)?|```", "", raw).strip()
                result = json.loads(raw)
                ans = result.get("extracted_answer")
                if isinstance(ans, (int, float)):
                    return str(ans)
                s = str(ans or "").strip()
                if s.upper() != "NONE" and s:
                    return s
                return ""
            except Exception as e:
                log.warning(f"Timestamp extractor API attempt {attempt}/{self.retries} failed: {e}")
            if attempt < self.retries:
                time.sleep(1)
        return ""

    def _extract_interval_llm(self, model_response: str) -> dict:
        """Fallback to LLM to extract a start/end interval. Returns {"start": float, "end": float} or {}."""
        if not self.client:
            return {}

        model_response = truncate_model_response(model_response, max_tokens=2048)
        user_msg = INTERVAL_EXTRACTOR_USER_TEMPLATE.format(model_response=model_response)

        for attempt in range(1, self.retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=1024,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": INTERVAL_EXTRACTOR_SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg}
                    ]
                )
                raw = response.choices[0].message.content.strip()
                raw = re.sub(r"```(?:json)?|```", "", raw).strip()
                result = json.loads(raw)
                ans = result.get("extracted_answer")
                if isinstance(ans, dict):
                    start, end = _to_float(ans.get("start")), _to_float(ans.get("end"))
                    if start is not None and end is not None:
                        return {"start": start, "end": end}
                if isinstance(ans, str):
                    m = re.match(r"\s*([0-9]*\.?[0-9]+)\s*[,;\-\s]\s*([0-9]*\.?[0-9]+)\s*$", ans)
                    if m:
                        return {"start": float(m.group(1)), "end": float(m.group(2))}
                return {}
            except Exception as e:
                log.warning(f"Interval extractor API attempt {attempt}/{self.retries} failed: {e}")
            if attempt < self.retries:
                time.sleep(1)
        return {}

    # -------------------------------------------------------------------------
    # Deterministic scoring (MCQ + phase understanding)
    # -------------------------------------------------------------------------

    def score_mcq(self, model_response: str, correct_answer: str) -> dict:
        """
        Deterministic scoring for multiple-choice questions (letter A-D).
        The model is instructed to reply with {"explanation": ..., "answer": "<letter>"}.
        Parses the JSON 'answer' key; falls back to regex and LLM extractor.
        R_task = 1 if normalized pred == gold else 0  (max 1.0).
        """
        correct_clean = str(correct_answer).strip().upper()
        extracted = ""
        method = "deterministic"

        obj, _ = parse_strict_json(model_response)
        if obj is None:
            obj = extract_json_object(model_response)
        if isinstance(obj, dict) and obj.get("answer") is not None:
            extracted = str(obj["answer"]).strip().upper()

        if not extracted:
            extracted = extract_answer_letter(model_response)

        if not extracted and self.client:
            extracted = self._extract_clip_letter_llm(model_response)
            if extracted:
                method = "llm_extractor"

        is_correct = bool(extracted) and extracted == correct_clean
        return {
            "score": 1 if is_correct else 0,
            "max_score": 1,
            "normalised_score": 1.0 if is_correct else 0.0,
            "extracted_answer": extracted or "NONE",
            "correct": is_correct,
            "method": method,
            "justification": "",
        }

    def score_phase_task(self, question_type: str, model_response: str, correct_answer) -> dict:
        """
        Deterministic scoring for the four phase-understanding tasks.

          boundary_detection:            R_task = exp(-|t_pred - t_gt| / 1.5)
          temporal_localization:         R_task = IoU([start,end] pred vs gold)
          timestamp_to_phase / contextual_phase_recognition:
                                         R_task = 1 if phase_id exact match else 0

        Format bonus (all phase tasks):
          R_fmt  = 1 if output is strict JSON with exactly {explanation, answer} else 0
          R_total = R_task + 0.05 * R_fmt   (max 1.05)
        """
        correct = correct_answer if isinstance(correct_answer, dict) else {}

        obj, strict = parse_strict_json(model_response)
        if obj is None:
            obj = extract_json_object(model_response)
        format_valid = 1 if (strict and obj is not None and set(obj.keys()) == {"explanation", "answer"}) else 0

        answer = obj.get("answer") if isinstance(obj, dict) else None
        method = "deterministic"

        if question_type == "boundary_detection":
            pred = _to_float(answer.get("timestamp")) if isinstance(answer, dict) else None
            if pred is None:
                m = re.search(r"(?:timestamp|time|t)\s*[:=]\s*([0-9]*\.?[0-9]+)", model_response, re.IGNORECASE)
                if not m:
                    m = re.search(r"(?:at|is)\s+([0-9]*\.?[0-9]+)\s*(?:s|sec|seconds)\b", model_response, re.IGNORECASE)
                if m:
                    pred = float(m.group(1))
            if pred is None and self.client:
                pred = _to_float(self._extract_timestamp_llm(model_response))
                if pred is not None:
                    method = "llm_extractor"

            gt = _to_float(correct.get("timestamp"))
            if pred is None or gt is None:
                task_score, extracted = 0.0, "NONE"
            else:
                err = abs(pred - gt)
                task_score = math.exp(-err / 1.5)
                extracted = f"{pred:.3f}"
            correct_flag = task_score >= 0.5

        elif question_type == "temporal_localization":
            pred_start = pred_end = None
            if isinstance(answer, dict):
                pred_start = _to_float(answer.get("start"))
                pred_end = _to_float(answer.get("end"))
            if pred_start is None or pred_end is None:
                m1 = re.search(r"start\s*[:=]\s*([0-9]*\.?[0-9]+)", model_response, re.IGNORECASE)
                m2 = re.search(r"end\s*[:=]\s*([0-9]*\.?[0-9]+)", model_response, re.IGNORECASE)
                if m1:
                    pred_start = float(m1.group(1))
                if m2:
                    pred_end = float(m2.group(1))
            if (pred_start is None or pred_end is None) and self.client:
                interval = self._extract_interval_llm(model_response)
                if interval:
                    pred_start = interval.get("start")
                    pred_end = interval.get("end")
                    method = "llm_extractor"

            gt_start = _to_float(correct.get("start"))
            gt_end = _to_float(correct.get("end"))
            if pred_start is None or pred_end is None or gt_start is None or gt_end is None:
                task_score, extracted = 0.0, "NONE"
            else:
                inter = max(0.0, min(pred_end, gt_end) - max(pred_start, gt_start))
                union = (pred_end - pred_start) + (gt_end - gt_start) - inter
                task_score = inter / union if union > 0 else 0.0
                extracted = f"[{pred_start:.3f}, {pred_end:.3f}]"
            correct_flag = task_score >= 0.5

        else:  # timestamp_to_phase / contextual_phase_recognition
            pred_id = ""
            if isinstance(answer, dict):
                pred_id = str(answer.get("phase_id", "")).strip().upper()
            if not pred_id:
                pred_id = extract_answer_phase(model_response)
            if not pred_id and self.client:
                pred_id = self._extract_phase_llm(model_response)
                if pred_id:
                    method = "llm_extractor"

            gt_id = str(correct.get("phase_id", "")).strip().upper()
            task_score = 1.0 if (pred_id and pred_id == gt_id) else 0.0
            extracted = pred_id or "NONE"
            correct_flag = task_score == 1.0

        total = round(task_score + 0.05 * format_valid, 6)
        return {
            "score": total,
            "task_score": round(task_score, 6),
            "format_valid": format_valid,
            "format_bonus": round(0.05 * format_valid, 6),
            "max_score": 1.05,
            "normalised_score": round(total / 1.05, 4),
            "extracted_answer": extracted,
            "correct": correct_flag,
            "method": method,
            "justification": "",
        }

    def score_clip_deterministic(self, model_response: str, correct_answer, question_type: str = "", task_category: str = "") -> dict:
        """Deterministic scoring dispatcher for clip-level tasks (MCQ letters and phase understanding)."""
        if task_category == "phase" or question_type in PHASE_TASK_TYPES:
            return self.score_phase_task(
                question_type=question_type,
                model_response=model_response,
                correct_answer=correct_answer
            )
        return self.score_mcq(model_response=model_response, correct_answer=correct_answer)

    # -------------------------------------------------------------------------
    # LLM judge scoring (visual description + narration)
    # -------------------------------------------------------------------------

    def score_description(self, reference_description: str, model_response: str) -> dict:
        """
        Scores a visual description of a clip using the LLM judge (0-5 scale).
        Compares actions, instruments, and anatomy against the reference description.
        """
        if not self.client:
            log.warning("LLM client not initialized for visual description scoring. Scoring as 0.")
            return {
                "score": 0,
                "max_score": 5,
                "normalised_score": 0.0,
                "extracted_answer": "N/A",
                "correct": False,
                "method": "llm_judge_fallback",
                "justification": "LLM client not initialized"
            }

        model_response = truncate_model_response(model_response, max_tokens=2048)
        user_msg = DESCRIPTION_JUDGE_USER_TEMPLATE.format(
            reference_description=reference_description,
            model_response=model_response
        )

        for attempt in range(1, self.retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=2048,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": DESCRIPTION_JUDGE_SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg}
                    ]
                )
                raw = response.choices[0].message.content.strip()
                raw = re.sub(r"```(?:json)?|```", "", raw).strip()
                result = json.loads(raw)
                score = int(result.get("score", 0))
                score = max(0, min(5, score))
                normalised = round(score / 5.0, 4)
                return {
                    "score": score,
                    "max_score": 5,
                    "normalised_score": normalised,
                    "extracted_answer": "N/A",
                    "correct": score >= 3,
                    "method": "llm_judge",
                    "justification": result.get("justification", "")
                }
            except json.JSONDecodeError as e:
                log.warning(f"Description judge JSON parse attempt {attempt}/{self.retries} failed: {e}")
            except Exception as e:
                log.warning(f"Description judge API attempt {attempt}/{self.retries} failed: {e}")
            if attempt < self.retries:
                time.sleep(2 * attempt)

        log.error("Description judge failed all attempts. Scoring as 0.")
        return {
            "score": 0,
            "max_score": 5,
            "normalised_score": 0.0,
            "extracted_answer": "N/A",
            "correct": False,
            "method": "llm_judge_failed",
            "justification": "All LLM judge attempts failed"
        }

    def score_clip_llm_judge(self, question_text: str = "", correct_answer: str = "", reference_reasoning: str = "", reference_description: str = "", model_response: str = "") -> dict:
        """Alias for score_description for clip visual description judge scoring."""
        ref = reference_description or reference_reasoning
        return self.score_description(reference_description=ref, model_response=model_response)

    def score_narration(self, reference_narration: str, model_response: str) -> dict:
        """Scores a full-video narration using the LLM judge (0-5 per dimension)."""
        if not self.client:
            log.warning("LLM client not initialized for narration scoring. Scoring as 0.")
            return {
                "step_coverage": 0,
                "chronological_accuracy": 0,
                "visual_technical_accuracy": 0,
                "narrative_flow": 0,
                "overall_score": 0,
                "max_score": 5,
                "normalized_score": 0.0,
                "justification": "LLM client not initialized",
                "method": "llm_judge_fallback"
            }

        model_response = truncate_model_response(model_response, max_tokens=2048)
        user_msg = NARRATION_JUDGE_USER_TEMPLATE.format(
            reference_narration=reference_narration,
            model_response=model_response
        )

        for attempt in range(1, self.retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=8192,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": NARRATION_JUDGE_SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg}
                    ]
                )
                raw = response.choices[0].message.content.strip()
                raw = re.sub(r"```(?:json)?|```", "", raw).strip()
                result = json.loads(raw)
                overall = int(result.get("overall_score", 0))
                overall = max(0, min(5, overall))
                return {
                    "step_coverage": int(result.get("step_coverage", 0)),
                    "chronological_accuracy": int(result.get("chronological_accuracy", 0)),
                    "visual_technical_accuracy": int(result.get("visual_technical_accuracy", 0)),
                    "narrative_flow": int(result.get("narrative_flow", 0)),
                    "overall_score": overall,
                    "max_score": 5,
                    "normalized_score": round(overall / 5.0, 4),
                    "justification": result.get("justification", ""),
                    "method": "llm_judge"
                }
            except json.JSONDecodeError as e:
                log.warning(f"Narration judge JSON parse attempt {attempt}/{self.retries} failed: {e}")
            except Exception as e:
                log.warning(f"Narration judge API attempt {attempt}/{self.retries} failed: {e}")
            if attempt < self.retries:
                time.sleep(2 * attempt)

        log.error("Narration judge failed all attempts. Scoring as 0.")
        return {
            "step_coverage": 0, "chronological_accuracy": 0,
            "visual_technical_accuracy": 0, "narrative_flow": 0,
            "overall_score": 0, "max_score": 5, "normalized_score": 0.0,
            "justification": "All LLM judge attempts failed", "method": "llm_judge_failed"
        }

    # -------------------------------------------------------------------------
    # Offline grading of pre-generated response files
    # -------------------------------------------------------------------------

    def grade_responses_file(self, responses_path: str, scores_path: str, summary_path: str, level: str, model_id: str, tag: str) -> dict:
        """
        Reads a self-contained responses JSONL file, evaluates each record,
        writes scores to a scores file, and generates a summary JSON.
        Automatically detects and re-evaluates records that previously failed judge attempts,
        replacing failed entries with fresh evaluations.
        """
        if not os.path.exists(responses_path):
            raise FileNotFoundError(f"Responses file not found at: {responses_path}")

        def is_valid_score(row: dict) -> bool:
            method = str(row.get("method", ""))
            justification = str(row.get("justification", ""))
            if method in ("llm_judge_failed", "llm_judge_fallback"):
                return False
            if "All LLM judge attempts failed" in justification or "LLM client not initialized" in justification:
                return False
            return True

        def resp_key(record: dict) -> tuple:
            """Uniform (record_id, task_key) for resume/caching purposes."""
            task_key = record.get("question_type") or record.get("task_type")
            resp_id = record.get("record_id") or record.get("clip_id") or record.get("yt_id")
            return (resp_id, task_key)

        # Load existing valid scores into memory
        valid_scores_by_key = {}
        if os.path.exists(scores_path):
            with open(scores_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                        if is_valid_score(row):
                            key = (row.get("record_id") or row.get("clip_id") or row.get("yt_id"),
                                   row.get("question_type") or row.get("task_type"))
                            valid_scores_by_key[key] = row
                    except (json.JSONDecodeError, KeyError):
                        continue

        log.info(f"Offline grading of {responses_path} started. Valid scores already cached: {len(valid_scores_by_key)}")

        # Read all responses to preserve original order
        all_responses = []
        with open(responses_path, "r", encoding="utf-8") as resp_f:
            for line_idx, line in enumerate(resp_f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    all_responses.append(json.loads(line))
                except json.JSONDecodeError as e:
                    log.error(f"Failed to parse JSON response line {line_idx}: {e}")
                    continue

        # Helper to atomically flush all scores to disk
        def flush_scores_to_file():
            tmp_scores_path = f"{scores_path}.tmp"
            with open(tmp_scores_path, "w", encoding="utf-8") as score_f:
                for resp_rec in all_responses:
                    key = resp_key(resp_rec)
                    if key in valid_scores_by_key:
                        score_f.write(json.dumps(valid_scores_by_key[key]) + "\n")
            if os.path.exists(tmp_scores_path):
                os.replace(tmp_scores_path, scores_path)

        # Grade any un-evaluated or previously failed responses
        new_or_updated = 0
        for record in all_responses:
            key = resp_key(record)
            if key in valid_scores_by_key:
                continue

            resp_id, task_key = key

            if level == "clip":
                question_type = record.get("question_type", task_key or "")
                reward_type = record.get("reward_type")
                task_category = record.get("task_category", "")
                correct_answer = record.get("correct_answer")
                model_response = record.get("model_response")
                reference_description = record.get("reference_description") or record.get("reference_reasoning", "")

                log.info(f"Grading clip {resp_id} ({question_type})...")
                try:
                    if reward_type == "llm_judge" or task_category == "visual_description" or "visual_description" in question_type:
                        score_info = self.score_description(
                            reference_description=reference_description,
                            model_response=model_response
                        )
                    elif task_category == "phase" or question_type in PHASE_TASK_TYPES:
                        score_info = self.score_phase_task(
                            question_type=question_type,
                            model_response=model_response,
                            correct_answer=correct_answer
                        )
                    else:
                        score_info = self.score_mcq(
                            model_response=model_response,
                            correct_answer=correct_answer
                        )

                    score_record = build_score_record(
                        resp_id=resp_id,
                        task_category=task_category,
                        question_type=question_type,
                        reward_type=reward_type,
                        correct_answer=correct_answer,
                        score_info=score_info,
                    )
                    valid_scores_by_key[key] = score_record
                    new_or_updated += 1
                    if new_or_updated % 10 == 0:
                        flush_scores_to_file()
                except Exception as e:
                    log.error(f"Error grading clip {resp_id}: {e}")

            else:  # level == "full"
                task_type = record.get("task_type", task_key or "")
                model_response = record.get("model_response")

                log.info(f"Grading full video {resp_id} ({task_type})...")
                try:
                    if task_type == "narration":
                        reference_narration = record.get("reference_narration", "")
                        score_info = self.score_narration(
                            reference_narration=reference_narration,
                            model_response=model_response
                        )
                        score_record = {
                            "record_id": resp_id,
                            "task_type": task_type,
                            **score_info
                        }
                        valid_scores_by_key[key] = score_record
                        new_or_updated += 1
                        flush_scores_to_file()
                except Exception as e:
                    log.error(f"Error grading full video {resp_id} task {task_type}: {e}")

        # Final flush to ensure all scores are written in exact original order
        flush_scores_to_file()
        log.info(f"Grading completed for {responses_path}. Total scored: {len(valid_scores_by_key)} (New/Replaced: {new_or_updated})")

        return self._generate_summary(scores_path, summary_path, level, model_id, tag)

    def _generate_summary(self, scores_path: str, summary_path: str, level: str, model_id: str, tag: str) -> dict:
        """Helper to generate summary JSON from scores file."""
        if level == "clip":
            all_normalised = []
            per_type_agg = {}
            with open(scores_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                        qt = row["question_type"]
                        ns = float(row["normalised_score"])
                        per_type_agg.setdefault(qt, []).append(ns)
                        all_normalised.append(ns)
                    except Exception:
                        continue

            summary = {
                "model_id": model_id,
                "tag": tag,
                "total_scored": len(all_normalised),
                "overall_normalised_accuracy": round(sum(all_normalised) / len(all_normalised), 4) if all_normalised else 0.0,
                "per_type": {
                    qt: {
                        "n_samples": len(scores),
                        "avg_normalised_score": round(sum(scores) / len(scores), 4)
                    }
                    for qt, scores in per_type_agg.items()
                }
            }
        else:  # level == "full"
            narration_rows = []
            with open(scores_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                        tt = row["task_type"]
                        if tt == "narration":
                            narration_rows.append(row)
                    except Exception:
                        continue

            def avg(values):
                values = [v for v in values if v is not None]
                return round(sum(values) / len(values), 4) if values else None

            narration_summary = {
                "n_samples": len(narration_rows),
                "avg_overall_score": avg([r.get("overall_score") for r in narration_rows]),
                "avg_normalized_score": avg([r.get("normalized_score") for r in narration_rows]),
                "avg_step_coverage": avg([r.get("step_coverage") for r in narration_rows]),
                "avg_chronological_accuracy": avg([r.get("chronological_accuracy") for r in narration_rows]),
                "avg_visual_technical_accuracy": avg([r.get("visual_technical_accuracy") for r in narration_rows]),
                "avg_narrative_flow": avg([r.get("narrative_flow") for r in narration_rows])
            }

            summary = {
                "model_id": model_id,
                "tag": tag,
                "narration": narration_summary
            }

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)
        return summary