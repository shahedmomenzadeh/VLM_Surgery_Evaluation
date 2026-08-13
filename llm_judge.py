# llm_judge.py
# OpenAI/OpenRouter compatible LLM judge implementation for VLM evaluation scoring

import re
import os
import json
import time
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
    PHASE_EXTRACTOR_USER_TEMPLATE
)

log = logging.getLogger("llm_judge")


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

    letters = re.findall(r"\b([A-D])\b", text.upper())
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


def extract_clip_answer(text: str, correct_answer: str) -> str:
    """Dispatches extraction based on whether the expected answer is a phase code or MCQ letter."""
    if correct_answer.strip().upper().startswith("P"):
        return extract_answer_phase(text)
    return extract_answer_letter(text)


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
    def __init__(self, base_url: str, api_key: str, model: str, retries: int = 3):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.retries = retries
        
        # Initialize OpenAI client if api_key is available
        if self.api_key:
            self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        else:
            self.client = None
            log.warning("No API key provided for LLMJudge. LLM-based scoring will fall back to deterministic scoring.")

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

    def score_clip_deterministic(self, model_response: str, correct_answer: str) -> dict:
        """
        Deterministic scoring for MCQs and Phase identification (0 or 1).
        Supports A-D letters and P01-P13 phase codes with LLM extractor fallback.
        """
        correct_clean = correct_answer.strip().upper()
        is_phase = correct_clean.startswith("P")
        
        extracted = extract_clip_answer(model_response, correct_clean)
        method = "deterministic"
        
        if not extracted and self.client:
            if is_phase:
                extracted = self._extract_phase_llm(model_response)
            else:
                extracted = self._extract_clip_letter_llm(model_response)
            if extracted:
                method = "llm_extractor"
                
        is_correct = (extracted == correct_clean)
        return {
            "score": 1 if is_correct else 0,
            "max_score": 1,
            "extracted_answer": extracted or "NONE",
            "correct": is_correct,
            "method": method,
            "justification": ""
        }

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

    def grade_responses_file(self, responses_path: str, scores_path: str, summary_path: str, level: str, model_id: str, tag: str) -> dict:
        """
        Reads a self-contained responses JSONL file, evaluates each record,
        writes scores to a scores file, and generates a summary JSON.
        """
        if not os.path.exists(responses_path):
            raise FileNotFoundError(f"Responses file not found at: {responses_path}")

        processed_ids = set()
        if os.path.exists(scores_path):
            with open(scores_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                        if level == "clip":
                            processed_ids.add((row["clip_id"], row["question_type"]))
                        else:
                            processed_ids.add((row["yt_id"], row["task_type"]))
                    except (json.JSONDecodeError, KeyError):
                        continue

        log.info(f"Offline grading of {responses_path} started. Graded count to resume: {len(processed_ids)}")

        with open(responses_path, "r", encoding="utf-8") as resp_f, \
             open(scores_path, "a", encoding="utf-8") as score_f:
            
            for line_idx, line in enumerate(resp_f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    log.error(f"Failed to parse JSON response line {line_idx}: {e}")
                    continue

                if level == "clip":
                    clip_id = record.get("clip_id")
                    question_type = record.get("question_type")
                    reward_type = record.get("reward_type")
                    correct_answer = record.get("correct_answer")
                    model_response = record.get("model_response")
                    reference_description = record.get("reference_description") or record.get("reference_reasoning", "")

                    if (clip_id, question_type) in processed_ids:
                        continue

                    log.info(f"Grading clip {clip_id} ({question_type})...")
                    try:
                        if reward_type == "llm_judge" or "visual_description" in question_type:
                            score_info = self.score_description(
                                reference_description=reference_description,
                                model_response=model_response
                            )
                        else:
                            score_info = self.score_clip_deterministic(
                                model_response=model_response,
                                correct_answer=correct_answer
                            )
                        
                        normalised = round(score_info["score"] / score_info["max_score"], 4) if score_info.get("max_score", 0) > 0 else 0.0
                        score_record = {
                            "clip_id": clip_id,
                            "question_type": question_type,
                            "reward_type": reward_type,
                            "correct_answer": correct_answer,
                            "extracted_answer": score_info["extracted_answer"],
                            "score": score_info["score"],
                            "max_score": score_info["max_score"],
                            "normalised_score": normalised,
                            "correct": score_info["correct"],
                            "method": score_info["method"],
                            "justification": score_info.get("justification", "")
                        }
                        score_f.write(json.dumps(score_record) + "\n")
                        score_f.flush()
                    except Exception as e:
                        log.error(f"Error grading clip {clip_id}: {e}")

                else:  # level == "full"
                    yt_id = record.get("yt_id")
                    task_type = record.get("task_type")
                    model_response = record.get("model_response")

                    if (yt_id, task_type) in processed_ids:
                        continue

                    log.info(f"Grading full video {yt_id} ({task_type})...")
                    try:
                        if task_type == "narration":
                            reference_narration = record.get("reference_narration", "")
                            score_info = self.score_narration(
                                reference_narration=reference_narration,
                                model_response=model_response
                            )
                            score_record = {
                                "yt_id": yt_id,
                                "task_type": task_type,
                                **score_info
                            }
                            score_f.write(json.dumps(score_record) + "\n")
                            score_f.flush()
                    except Exception as e:
                        log.error(f"Error grading full video {yt_id} task {task_type}: {e}")

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