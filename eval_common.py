# eval_common.py
# Unified evaluation infrastructure and shared utilities across all VLM models
#
# The evaluation dataset embeds the full task instruction (including the strict
# JSON {"explanation", "answer"} output contract) in every record, so questions
# are passed to the model as-is: one prompt per record, no CoT/direct variants.

import os
import gc
import json
import logging
import traceback
import torch
from tqdm import tqdm

from llm_judge import PHASE_TASK_TYPES, build_score_record

log = logging.getLogger("eval_common")


# =============================================================================
# 1. GPU & SYSTEM UTILITIES
# =============================================================================

def vram_stats(label: str = "") -> str:
    """Returns a string describing allocated and reserved VRAM across all CUDA devices."""
    if not torch.cuda.is_available():
        return "CUDA unavailable"
    lines = []
    for i in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(i) / (1024 ** 3)
        res = torch.cuda.memory_reserved(i) / (1024 ** 3)
        lines.append(f"GPU{i}: alloc={alloc:.1f}GB res={res:.1f}GB")
    tag = f" [{label}]" if label else ""
    return "  ".join(lines) + tag


def flush_memory(*objs) -> None:
    """Deletes objects, triggers garbage collection, and clears CUDA memory cache."""
    for o in objs:
        del o
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log.info(f"VRAM after flush — {vram_stats()}")


def first_device(model: torch.nn.Module) -> torch.device:
    """Returns the primary device of a model (supporting device_map='auto')."""
    hf_map = getattr(model, "hf_device_map", {})
    if hf_map:
        for key in ("model.embed_tokens", "transformer.wte", "lm_head", "visual"):
            if key in hf_map:
                dev = hf_map[key]
                if dev == "cpu":
                    return torch.device("cpu")
                return torch.device(f"cuda:{dev}" if isinstance(dev, int) else dev)
        first_val = next(iter(hf_map.values()))
        if first_val == "cpu":
            return torch.device("cpu")
        return torch.device(f"cuda:{first_val}" if isinstance(first_val, int) else first_val)
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def move_inputs_to_device(inputs: dict, device: torch.device) -> dict:
    """Moves all tensors in an inputs dictionary to the designated device and casts floats to float16."""
    moved = {}
    for k, v in inputs.items():
        if not isinstance(v, torch.Tensor):
            moved[k] = v
            continue
        v = v.to(device)
        if v.is_floating_point():
            v = v.to(torch.float16)
        moved[k] = v
    return moved


def probe_total_frames(video_path: str) -> int | None:
    """
    Returns total frame count of a video via fast metadata probing with decord.
    Returns None if decord is unavailable or probing fails.
    """
    try:
        import decord
        vr = decord.VideoReader(video_path, num_threads=1)
        return len(vr)
    except Exception as e:
        log.warning(f"probe_total_frames: could not read metadata for {video_path}: {e}")
        return None


# =============================================================================
# 2. RESUME TRACKING & JSONL I/O
# =============================================================================

def get_processed_ids(file_path: str, id_key: str, type_key: str) -> set[tuple]:
    """Extracts a set of (id, type) tuples from a JSONL results file for resume support."""
    processed = set()
    if not os.path.exists(file_path):
        return processed
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                method = str(row.get("method", ""))
                justification = str(row.get("justification", ""))
                if method in ("llm_judge_failed", "llm_judge_fallback") or "All LLM judge attempts failed" in justification or "LLM client not initialized" in justification:
                    continue
                processed.add((row[id_key], row[type_key]))
            except (json.JSONDecodeError, KeyError):
                continue
    return processed


def write_jsonl(file_handle, record: dict) -> None:
    """Appends a single JSON record to an open file and flushes buffer."""
    file_handle.write(json.dumps(record) + "\n")
    file_handle.flush()


# =============================================================================
# 3. GENERIC EVALUATION EXECUTION LOOPS
# =============================================================================

def run_clip_evaluation_loop(
    generate_fn,
    records: list[dict],
    judge,
    output_dir: str,
    tag: str,
    args,
    logger=None
) -> dict:
    """
    Generic execution loop for clip-level evaluation:
      - Iterates once over each clip task (visual description, MCQs, phase understanding)
        — one prompt per record, no CoT/direct variants.
      - Dispatches scoring: LLM judge for visual descriptions,
        deterministic for MCQs and phase tasks.
      - Produces responses.jsonl, scores.jsonl, and summary.json.

    Args:
        generate_fn: Callable(video_path: str, question_text: str, log_id: str) -> str | None
        records: List of clip records from dataset_loader.
        judge: LLMJudge instance.
        output_dir: Directory where results will be written.
        tag: Run tag identifier.
        args: Command-line arguments namespace.
        logger: Optional logger instance.
    """
    _log = logger or log
    os.makedirs(output_dir, exist_ok=True)
    responses_path = os.path.join(output_dir, f"{tag}_responses.jsonl")
    scores_path = os.path.join(output_dir, f"{tag}_scores.jsonl")

    if args.mode == "inference":
        processed_ids = get_processed_ids(responses_path, "record_id", "question_type")
        _log.info(f"Inference-only mode. Resuming from responses file. Pre-existing count: {len(processed_ids)}")
    else:
        processed_ids = get_processed_ids(scores_path, "record_id", "question_type")
        _log.info(f"Resuming clip-level evaluation: {len(processed_ids)} questions already scored.")

    n_ok = n_skip = n_error = 0

    resp_f = open(responses_path, "a", encoding="utf-8")
    score_f = open(scores_path, "a", encoding="utf-8") if args.mode != "inference" else None

    try:
        pbar = tqdm(records, desc=f"Clip Eval [{tag}]", leave=True, dynamic_ncols=True)
        for record in pbar:
            record_id = record["record_id"]
            video_path = record["video_path"]
            qtype = record["question_type"]
            task_category = record["task_category"]
            reward_type = record["reward_type"]

            if (record_id, qtype) in processed_ids:
                n_skip += 1
                continue

            question_text = record["question_text"]

            pbar.set_postfix_str(f"Gen {record_id}", refresh=True)
            model_response = generate_fn(
                video_path=video_path,
                question_text=question_text,
                log_id=f"clip/{record_id}"
            )

            if model_response is None:
                n_error += 1
                continue

            # Write raw model response
            write_jsonl(resp_f, {
                "record_id": record_id,
                "task_category": task_category,
                "question_type": qtype,
                "reward_type": reward_type,
                "correct_answer": record["correct_answer"],
                "question_text": question_text,
                "reference_reasoning": record.get("reference_reasoning", ""),
                "reference_description": record.get("reference_description", ""),
                "model_response": model_response
            })

            # Execute scoring if judge is active
            if args.mode != "inference" and score_f is not None:
                pbar.set_postfix_str(f"Judge {record_id}", refresh=True)
                try:
                    if reward_type == "llm_judge" or task_category == "visual_description":
                        score_info = judge.score_description(
                            reference_description=record.get("reference_description") or record.get("reference_reasoning", ""),
                            model_response=model_response
                        )
                    elif task_category == "phase" or qtype in PHASE_TASK_TYPES:
                        score_info = judge.score_phase_task(
                            question_type=qtype,
                            model_response=model_response,
                            correct_answer=record["correct_answer"]
                        )
                    else:
                        score_info = judge.score_mcq(
                            model_response=model_response,
                            correct_answer=record["correct_answer"]
                        )

                    score_record = build_score_record(
                        resp_id=record_id,
                        task_category=task_category,
                        question_type=qtype,
                        reward_type=reward_type,
                        correct_answer=record["correct_answer"],
                        score_info=score_info,
                    )
                    write_jsonl(score_f, score_record)
                    n_ok += 1
                except Exception as e:
                    _log.error(f"Error scoring clip {record_id} ({qtype}): {e}\n{traceback.format_exc()}")
                    n_error += 1
            else:
                n_ok += 1

            pbar.set_postfix(ok=n_ok, skip=n_skip, err=n_error)
    finally:
        resp_f.close()
        if score_f is not None:
            score_f.close()

    if args.mode == "inference":
        _log.info("Inference-only mode run completed. Output is recorded offline.")
        return {}

    # Aggregate and save summary
    all_normalised = []
    per_type_agg = {}
    if os.path.exists(scores_path):
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
        "model_id": args.model_id,
        "tag": tag,
        "total_scored": len(all_normalised),
        "overall_normalised_accuracy": round(sum(all_normalised) / len(all_normalised), 4) if all_normalised else 0.0,
        "run_stats": {"ok": n_ok, "skip": n_skip, "error": n_error},
        "per_type": {
            qt: {
                "n_samples": len(scores),
                "avg_normalised_score": round(sum(scores) / len(scores), 4)
            }
            for qt, scores in per_type_agg.items()
        }
    }

    summary_path = os.path.join(output_dir, f"{tag}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    return summary


def run_full_video_evaluation_loop(
    generate_fn,
    records: list[dict],
    judge,
    output_dir: str,
    tag: str,
    args,
    logger=None
) -> dict:
    """
    Generic execution loop for full-video level evaluation:
      - Iterates across uncut surgery recordings (narration task only).
      - Evaluates narration across 4 clinical dimensions + overall score.
      - Produces responses.jsonl, scores.jsonl, and summary.json.
    """
    _log = logger or log
    os.makedirs(output_dir, exist_ok=True)
    responses_path = os.path.join(output_dir, f"{tag}_responses.jsonl")
    scores_path = os.path.join(output_dir, f"{tag}_scores.jsonl")

    if args.mode == "inference":
        processed_ids = get_processed_ids(responses_path, "record_id", "task_type")
        _log.info(f"Inference-only mode. Resuming from responses file. Pre-existing count: {len(processed_ids)}")
    else:
        processed_ids = get_processed_ids(scores_path, "record_id", "task_type")
        _log.info(f"Resuming full-video evaluation: {len(processed_ids)} tasks already scored.")

    n_ok = n_skip = n_error = 0

    resp_f = open(responses_path, "a", encoding="utf-8")
    score_f = open(scores_path, "a", encoding="utf-8") if args.mode != "inference" else None

    try:
        pbar = tqdm(records, desc=f"Full Video Eval [{tag}]", leave=True, dynamic_ncols=True)
        for record in pbar:
            record_id = record["record_id"]
            video_path = record["video_path"]

            # Narration Task
            if (record_id, "narration") in processed_ids:
                n_skip += 1
            else:
                question_text = record["narration_question"]
                pbar.set_postfix_str(f"Narr {record_id}", refresh=True)

                model_response = generate_fn(
                    video_path=video_path,
                    question_text=question_text,
                    log_id=f"{record_id}/narration"
                )

                if model_response is None:
                    n_error += 1
                else:
                    write_jsonl(resp_f, {
                        "record_id": record_id,
                        "yt_id": record.get("yt_id", record_id),
                        "task_type": "narration",
                        "question_text": question_text,
                        "reference_narration": record["narration_reference"],
                        "model_response": model_response
                    })

                    if args.mode != "inference" and score_f is not None:
                        pbar.set_postfix_str(f"Judge Narr {record_id}", refresh=True)
                        try:
                            score_info = judge.score_narration(
                                reference_narration=record["narration_reference"],
                                model_response=model_response
                            )
                            write_jsonl(score_f, {
                                "record_id": record_id,
                                "task_type": "narration",
                                **score_info
                            })
                            n_ok += 1
                        except Exception as e:
                            _log.error(f"Error scoring narration for {record_id}: {e}\n{traceback.format_exc()}")
                            n_error += 1
                    else:
                        n_ok += 1

            pbar.set_postfix(ok=n_ok, skip=n_skip, err=n_error)
    finally:
        resp_f.close()
        if score_f is not None:
            score_f.close()

    if args.mode == "inference":
        _log.info("Inference-only mode run completed. Output is recorded offline.")
        return {}

    # Aggregate narration summary
    narration_rows = []
    if os.path.exists(scores_path):
        with open(scores_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    if row.get("task_type") == "narration":
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
        "model_id": args.model_id,
        "tag": tag,
        "run_stats": {"ok": n_ok, "skip": n_skip, "error": n_error},
        "narration": narration_summary
    }

    summary_path = os.path.join(output_dir, f"{tag}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    return summary