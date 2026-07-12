# llm_judge.py
# OpenAI/OpenRouter compatible LLM judge implementation for VLM evaluation scoring

import re
import os
import json
import time
import logging
from openai import OpenAI

from prompts import (
    CLIP_JUDGE_SYSTEM_PROMPT,
    CLIP_JUDGE_USER_TEMPLATE,
    NARRATION_JUDGE_SYSTEM_PROMPT,
    NARRATION_JUDGE_USER_TEMPLATE,
    ORDERING_JUDGE_SYSTEM_PROMPT,
    ORDERING_JUDGE_USER_TEMPLATE,
    CLIP_EXTRACTOR_SYSTEM_PROMPT,
    CLIP_EXTRACTOR_USER_TEMPLATE
)

log = logging.getLogger("llm_judge")


def extract_answer_letter(text: str) -> str:
    """
    Extracts the answer letter from a model response.
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


def parse_letter_sequence(text: str, valid_letters: set[str]) -> list[str]:
    """
    Programmatic extraction of a full ordering of `valid_letters` from free text.

    Scans for runs of single uppercase letters separated by commas, arrows,
    "then", or "and" (e.g. "B, C, A", "B -> C -> A", "B then C then A").
    A run only counts as a candidate if it is an exact permutation of
    `valid_letters` (same letters, same count, each used once).

    Returns the LAST such candidate found, or [] if none qualifies.
    """
    n = len(valid_letters)
    sep = r"(?:\s*,\s*|\s*->\s*|\s*→\s*|\s+THEN\s+|\s+AND\s+)"
    pattern = rf"\b[A-Z]\b(?:{sep}\b[A-Z]\b)*"

    candidates = []
    for chunk in re.findall(pattern, text.upper()):
        letters = re.findall(r"[A-Z]", chunk)
        if len(letters) == n and set(letters) == valid_letters:
            candidates.append(letters)

    return candidates[-1] if candidates else []


def kendalls_tau(reference_order: list[str], predicted_order: list[str]) -> float | None:
    """Kendall's tau-a between two permutations of the same label set."""
    if (
        not predicted_order
        or set(reference_order) != set(predicted_order)
        or len(reference_order) != len(predicted_order)
    ):
        return None

    rank_ref = {label: i for i, label in enumerate(reference_order)}
    rank_pred = {label: i for i, label in enumerate(predicted_order)}

    labels = list(rank_ref.keys())
    n = len(labels)
    total_pairs = n * (n - 1) / 2
    if total_pairs == 0:
        return 1.0

    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            a, b = labels[i], labels[j]
            ref_sign = rank_ref[a] - rank_ref[b]
            pred_sign = rank_pred[a] - rank_pred[b]
            if ref_sign * pred_sign > 0:
                concordant += 1
            elif ref_sign * pred_sign < 0:
                discordant += 1

    return (concordant - discordant) / total_pairs


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
        """Fallback to LLM to extract answer letter for structured misalignments."""
        if not self.client:
            return ""
            
        user_msg = CLIP_EXTRACTOR_USER_TEMPLATE.format(model_response=model_response)
        
        for attempt in range(1, self.retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=64,
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
                log.warning(f"Clip extractor API attempt {attempt}/{self.retries} failed: {e}")
            if attempt < self.retries:
                time.sleep(1)
        return ""

    def score_clip_deterministic(self, model_response: str, correct_answer: str) -> dict:
        """Deterministic exact letter match scoring (0 or 1). Falls back to LLM extractor."""
        extracted = extract_answer_letter(model_response)
        method = "deterministic"
        
        if not extracted and self.client:
            extracted = self._extract_clip_letter_llm(model_response)
            if extracted:
                method = "llm_extractor"
                
        is_correct = (extracted == correct_answer.upper())
        return {
            "score": 1 if is_correct else 0,
            "max_score": 1,
            "extracted_answer": extracted or "NONE",
            "correct": is_correct,
            "method": method,
            "justification": ""
        }

    def score_clip_llm_judge(
        self,
        question_text: str,
        correct_answer: str,
        reference_reasoning: str,
        model_response: str
    ) -> dict:
        """Scores a clip multiple-choice question using the LLM judge (0-3 scale)."""
        if not self.client:
            log.warning("LLM client not initialized. Falling back to deterministic scoring for LLM-judge task.")
            result = self.score_clip_deterministic(model_response, correct_answer)
            result["method"] = "llm_judge_fallback"
            result["max_score"] = 3
            result["score"] = result["score"] * 3  # Scale 0/1 to 0/3
            return result

        user_msg = CLIP_JUDGE_USER_TEMPLATE.format(
            question_text=question_text,
            correct_answer=correct_answer,
            reference_reasoning=reference_reasoning,
            model_response=model_response
        )

        for attempt in range(1, self.retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=256,
                    messages=[
                        {"role": "system", "content": CLIP_JUDGE_SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg}
                    ]
                )
                raw = response.choices[0].message.content.strip()
                raw = re.sub(r"```(?:json)?|```", "", raw).strip()
                result = json.loads(raw)
                return {
                    "score": int(result.get("score", 0)),
                    "max_score": 3,
                    "extracted_answer": result.get("extracted_answer", "NONE"),
                    "correct": result.get("extracted_answer", "").upper() == correct_answer.upper(),
                    "method": "llm_judge",
                    "justification": result.get("justification", "")
                }
            except json.JSONDecodeError as e:
                log.warning(f"Clip judge JSON parse attempt {attempt}/{self.retries} failed: {e}")
            except Exception as e:
                log.warning(f"Clip judge API attempt {attempt}/{self.retries} failed: {e}")
            if attempt < self.retries:
                time.sleep(2 * attempt)

        log.error("Clip judge failed all attempts. Falling back to deterministic scoring.")
        result = self.score_clip_deterministic(model_response, correct_answer)
        result["method"] = "llm_judge_failed"
        result["max_score"] = 3
        result["score"] = result["score"] * 3
        return result

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

        user_msg = NARRATION_JUDGE_USER_TEMPLATE.format(
            reference_narration=reference_narration,
            model_response=model_response
        )

        for attempt in range(1, self.retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=4096,
                    messages=[
                        {"role": "system", "content": NARRATION_JUDGE_SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg}
                    ]
                )
                raw = response.choices[0].message.content.strip()
                raw = re.sub(r"```(?:json)?|```", "", raw).strip()
                result = json.loads(raw)
                overall = int(result.get("overall_score", 0))
                return {
                    "step_coverage": int(result.get("step_coverage", 0)),
                    "chronological_accuracy": int(result.get("chronological_accuracy", 0)),
                    "visual_technical_accuracy": int(result.get("visual_technical_accuracy", 0)),
                    "narrative_flow": int(result.get("narrative_flow", 0)),
                    "overall_score": overall,
                    "max_score": 5,
                    "normalized_score": round(overall / 5, 4),
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

    def score_ordering(self, question_text: str, correct_answer: str, model_response: str) -> dict:
        """
        Scores a full-video sequence ordering question.
        Uses regex first, and falls back to LLM judge if regex fails.
        """
        correct_sequence = [c.strip().upper() for c in correct_answer.split(",") if c.strip()]
        valid_letters = set(correct_sequence)

        # 1. Try programmatic regex extraction
        predicted_sequence = parse_letter_sequence(model_response, valid_letters)
        method = "regex"

        if not predicted_sequence and self.client:
            # 2. LLM judge fallback
            method = "llm_judge"
            system_prompt = ORDERING_JUDGE_SYSTEM_PROMPT.format(
                valid_letters=", ".join(sorted(valid_letters))
            )
            user_msg = ORDERING_JUDGE_USER_TEMPLATE.format(
                question_text=question_text,
                model_response=model_response
            )

            llm_sequence = None
            for attempt in range(1, self.retries + 1):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        max_tokens=4096,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_msg}
                        ]
                    )
                    raw = response.choices[0].message.content.strip()
                    raw = re.sub(r"```(?:json)?|```", "", raw).strip()
                    result = json.loads(raw)
                    llm_sequence = [str(x).strip().upper() for x in result.get("predicted_sequence", [])]
                    break
                except json.JSONDecodeError as e:
                    log.warning(f"Ordering judge JSON parse attempt {attempt}/{self.retries} failed: {e}")
                except Exception as e:
                    log.warning(f"Ordering judge API attempt {attempt}/{self.retries} failed: {e}")
                if attempt < self.retries:
                    time.sleep(2 * attempt)

            if llm_sequence is None:
                log.error("Ordering judge failed all attempts. Empty predicted sequence.")
                predicted_sequence = []
                method = "llm_judge_failed"
            else:
                predicted_sequence = llm_sequence

        tau = kendalls_tau(correct_sequence, predicted_sequence)
        valid = tau is not None
        exact = valid and (predicted_sequence == correct_sequence)

        return {
            "correct_sequence": correct_sequence,
            "predicted_sequence": predicted_sequence,
            "valid_sequence": valid,
            "kendalls_tau": tau,
            "exact_match": exact,
            "method": method
        }

    def grade_responses_file(self, responses_path: str, scores_path: str, summary_path: str, level: str, model_id: str, tag: str) -> dict:
        """
        Reads a self-contained responses JSONL file, evaluates each record,
        writes scores to a scores file, and generates a summary JSON.
        """
        if not os.path.exists(responses_path):
            raise FileNotFoundError(f"Responses file not found at: {responses_path}")

        # Get already scored ids for resume support
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
                    question_text = record.get("question_text")
                    model_response = record.get("model_response")
                    reference_reasoning = record.get("reference_reasoning", "")

                    if (clip_id, question_type) in processed_ids:
                        continue

                    log.info(f"Grading clip {clip_id} ({question_type})...")
                    try:
                        if reward_type == "llm_judge":
                            score_info = self.score_clip_llm_judge(
                                question_text=question_text,
                                correct_answer=correct_answer,
                                reference_reasoning=reference_reasoning,
                                model_response=model_response
                            )
                        else:
                            score_info = self.score_clip_deterministic(
                                model_response=model_response,
                                correct_answer=correct_answer
                            )
                        
                        normalised = round(score_info["score"] / score_info["max_score"], 4)
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
                    question_text = record.get("question_text")
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
                        else:  # sequence_ordering
                            correct_answer = record.get("correct_answer", "")
                            score_info = self.score_ordering(
                                question_text=question_text,
                                correct_answer=correct_answer,
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

        # Generate summary
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
            ordering_direct_rows = []
            ordering_cot_rows = []
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
                        elif tt in ("sequence_ordering", "sequence_ordering_direct"):
                            ordering_direct_rows.append(row)
                        elif tt == "sequence_ordering_cot":
                            ordering_cot_rows.append(row)
                    except Exception:
                        continue

            def avg(values):
                values = [v for v in values if v is not None]
                return round(sum(values) / len(values), 4) if values else None

            def compile_ordering_summary(rows):
                if not rows:
                    return {}
                valid_ordering = [r for r in rows if r.get("valid_sequence")]
                return {
                    "n_samples": len(rows),
                    "n_valid": len(valid_ordering),
                    "valid_rate": round(len(valid_ordering) / len(rows), 4) if rows else None,
                    "avg_kendalls_tau": avg([r.get("kendalls_tau") for r in valid_ordering]),
                    "exact_match_rate": (
                        round(sum(1 for r in rows if r.get("exact_match")) / len(rows), 4)
                        if rows else None
                    ),
                    "extraction_methods": {
                        m: sum(1 for r in rows if r.get("method") == m)
                        for m in {r.get("method") for r in rows if r.get("method")}
                    }
                }

            narration_summary = {
                "n_samples": len(narration_rows),
                "avg_overall_score": avg([r.get("overall_score") for r in narration_rows]),
                "avg_normalized_score": avg([r.get("normalized_score") for r in narration_rows]),
                "avg_step_coverage": avg([r.get("step_coverage") for r in narration_rows]),
                "avg_chronological_accuracy": avg([r.get("chronological_accuracy") for r in narration_rows]),
                "avg_visual_technical_accuracy": avg([r.get("visual_technical_accuracy") for r in narration_rows]),
                "avg_narrative_flow": avg([r.get("narrative_flow") for r in narration_rows])
            }

            ordering_direct_summary = compile_ordering_summary(ordering_direct_rows)
            ordering_cot_summary = compile_ordering_summary(ordering_cot_rows)

            summary = {
                "model_id": model_id,
                "tag": tag,
                "narration": narration_summary,
                "sequence_ordering_direct": ordering_direct_summary,
                "sequence_ordering_cot": ordering_cot_summary
            }

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)
        return summary