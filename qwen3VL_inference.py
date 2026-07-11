# qwen3VL_inference.py
# Inference and evaluation execution for Qwen3-VL model series

import os
import gc
import json
import logging
import time
import traceback
import torch
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

from prompts import (
    CLIP_INFERENCE_SUFFIX,
    CLIP_DIRECT_INFERENCE_SUFFIX,
    NARRATION_INFERENCE_SUFFIX,
    ORDERING_DIRECT_INFERENCE_SUFFIX,
    ORDERING_COT_INFERENCE_SUFFIX
)

log = logging.getLogger("qwen3vl_inference")


def vram_stats(label: str = "") -> str:
    if not torch.cuda.is_available():
        return "CUDA unavailable"
    lines = []
    for i in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(i) / 1024 ** 3
        res = torch.cuda.memory_reserved(i) / 1024 ** 3
        lines.append(f"GPU{i}: alloc={alloc:.1f}GB res={res:.1f}GB")
    tag = f" [{label}]" if label else ""
    return "  ".join(lines) + tag


def flush_memory(*objs) -> None:
    for o in objs:
        del o
    gc.collect()
    torch.cuda.empty_cache()
    log.info(f"VRAM after flush — {vram_stats()}")


def first_device(model: torch.nn.Module) -> torch.device:
    """Return the device of the embedding layer as placed by device_map='auto'."""
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
        return torch.device("cuda:0")


def move_inputs_to_device(inputs: dict, device: torch.device) -> dict:
    """Move all tensors in the inputs dict to the designated device and cast floats to float16."""
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


def get_processed_ids_clip(scores_path: str) -> set[tuple]:
    processed = set()
    if not os.path.exists(scores_path):
        return processed
    with open(scores_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                processed.add((row["clip_id"], row["question_type"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return processed


def get_processed_ids_full(scores_path: str) -> set[tuple]:
    processed = set()
    if not os.path.exists(scores_path):
        return processed
    with open(scores_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                processed.add((row["yt_id"], row["task_type"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return processed


def get_processed_ids_from_responses_clip(responses_path: str) -> set[tuple]:
    processed = set()
    if not os.path.exists(responses_path):
        return processed
    with open(responses_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                processed.add((row["clip_id"], row["question_type"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return processed


def get_processed_ids_from_responses_full(responses_path: str) -> set[tuple]:
    processed = set()
    if not os.path.exists(responses_path):
        return processed
    with open(responses_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                processed.add((row["yt_id"], row["task_type"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return processed


def write_jsonl(file_handle, record: dict) -> None:
    file_handle.write(json.dumps(record) + "\n")
    file_handle.flush()


def build_inputs(
    processor,
    video_path: str,
    question_text: str,
    max_frames: int,
    max_pixels: int,
    min_pixels: int,
    primary_device: torch.device
) -> dict:
    """Constructs tokenized multimodal inputs for Qwen3-VL using the recommended utils flow."""
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "min_pixels": min(min_pixels, max_pixels),
                    "max_pixels": max_pixels,
                    "nframes": max_frames
                },
                {
                    "type": "text",
                    "text": question_text
                }
            ]
        }
    ]
    
    # Apply text template
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Process visual data with decord wrapper
    image_inputs, video_inputs = process_vision_info(messages)
    
    # Processor tokenization and tensor mapping
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )
    
    # Move parameters to primary model device
    inputs = move_inputs_to_device(inputs, primary_device)
    return inputs


def run_qwen3vl_generation(
    model,
    processor,
    video_path: str,
    question_text: str,
    max_frames: int,
    max_pixels: int,
    min_pixels: int,
    max_new_tokens: int,
    temperature: float,
    primary_device: torch.device,
    log_id: str
) -> str | None:
    """Runs generation with progressive frame-count retry on OOM or short-video errors."""
    model_response = None
    
    # Build retry schedule: max_frames, max_frames//2, max_frames//4, ..., down to 2
    retry_frames = []
    f = max_frames
    while f >= 2:
        retry_frames.append(f)
        f = f // 2
    if not retry_frames:
        retry_frames = [2]
    
    for attempt_frames in retry_frames:
        if attempt_frames != max_frames:
            log.warning(f"{log_id} — Retry with {attempt_frames} frames (was {max_frames}).")
            
        try:
            inputs = build_inputs(
                processor=processor,
                video_path=video_path,
                question_text=question_text,
                max_frames=attempt_frames,
                max_pixels=max_pixels,
                min_pixels=min_pixels,
                primary_device=primary_device
            )
            input_len = inputs["input_ids"].shape[1]
            
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=(temperature > 0.0),
                    temperature=temperature if temperature > 0.0 else None,
                    use_cache=True,
                    pad_token_id=processor.tokenizer.eos_token_id
                )
                
            generated_ids = output_ids[:, input_len:]
            model_response = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0].strip()
            
            del inputs, output_ids, generated_ids
            break # Success
        except torch.cuda.OutOfMemoryError as e:
            log.error(f"{log_id} — CUDA OOM (frames={attempt_frames}): {e} | {vram_stats()}")
            gc.collect()
            torch.cuda.empty_cache()
            # Loop continues to smaller frames count
        except ValueError as e:
            if "nframes" in str(e):
                log.warning(f"{log_id} — Frame count too high for video (frames={attempt_frames}): {e}")
                gc.collect()
                torch.cuda.empty_cache()
                # Loop continues to smaller frames count
            else:
                log.error(f"{log_id} — Generation error: {e}\n{traceback.format_exc()}")
                break
        except Exception as e:
            log.error(f"{log_id} — Generation error: {e}\n{traceback.format_exc()}")
            break
            
    gc.collect()
    torch.cuda.empty_cache()
    return model_response


def run_clip_evaluation(model, processor, records: list[dict], judge, output_dir: str, tag: str, args, primary_device) -> dict:
    """Executes clip-level evaluation loop for Qwen3-VL."""
    os.makedirs(output_dir, exist_ok=True)
    responses_path = os.path.join(output_dir, f"{tag}_responses.jsonl")
    scores_path = os.path.join(output_dir, f"{tag}_scores.jsonl")
    
    # Resume checking setup
    if args.mode == "inference":
        processed_ids = get_processed_ids_from_responses_clip(responses_path)
        log.info(f"Inference-only mode. Resuming from responses file. Pre-existing count: {len(processed_ids)}")
    else:
        processed_ids = get_processed_ids_clip(scores_path)
        log.info(f"Resuming clip-level evaluation: {len(processed_ids)} questions already scored.")
        
    n_ok = n_skip = n_error = 0
    
    resp_f = open(responses_path, "a", encoding="utf-8")
    score_f = open(scores_path, "a", encoding="utf-8") if args.mode != "inference" else None
         
    try:
        pbar = tqdm(records, desc=f"Clip Eval [{tag}]", leave=True, dynamic_ncols=True)
        for record in pbar:
            clip_id = record["clip_id"]
            video_path = record["video_path"]
            base_qtype = record["question_type"]
            reward_type = record["reward_type"]
            
            # Evaluate both CoT (with reasoning) and Direct (without reasoning) tasks
            tasks = [
                {
                    "suffix": "_cot",
                    "prompt_suffix": CLIP_INFERENCE_SUFFIX,
                    "reward_type": reward_type,
                    "log_type": "cot"
                },
                {
                    "suffix": "_direct",
                    "prompt_suffix": CLIP_DIRECT_INFERENCE_SUFFIX,
                    "reward_type": "deterministic",
                    "log_type": "direct"
                }
            ]
            
            for task in tasks:
                qtype_with_suffix = f"{base_qtype}{task['suffix']}"
                
                if (clip_id, qtype_with_suffix) in processed_ids:
                    n_skip += 1
                    continue
                    
                question_text = record["question_text"] + task["prompt_suffix"]
                
                pbar.set_postfix_str(f"Gen {clip_id} ({task['log_type']})", refresh=True)
                model_response = run_qwen3vl_generation(
                    model=model,
                    processor=processor,
                    video_path=video_path,
                    question_text=question_text,
                    max_frames=args.max_frames,
                    max_pixels=args.max_pixels,
                    min_pixels=args.min_pixels,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    primary_device=primary_device,
                    log_id=f"clip/{clip_id}_{task['log_type']}"
                )
                
                if model_response is None:
                    n_error += 1
                    continue
                    
                # Write raw response with reference reasoning saved
                write_jsonl(resp_f, {
                    "clip_id": clip_id,
                    "question_type": qtype_with_suffix,
                    "reward_type": task["reward_type"],
                    "correct_answer": record["correct_answer"],
                    "question_text": record["question_text"],
                    "reference_reasoning": record.get("reference_reasoning", ""),
                    "model_response": model_response
                })
                
                # Score
                if args.mode != "inference" and score_f is not None:
                    pbar.set_postfix_str(f"Judge {clip_id} ({task['log_type']})", refresh=True)
                    try:
                        if task["reward_type"] == "llm_judge":
                            score_info = judge.score_clip_llm_judge(
                                question_text=record["question_text"],
                                correct_answer=record["correct_answer"],
                                reference_reasoning=record["reference_reasoning"],
                                model_response=model_response
                            )
                        else:
                            score_info = judge.score_clip_deterministic(
                                model_response=model_response,
                                correct_answer=record["correct_answer"]
                            )
                            
                        normalised = round(score_info["score"] / score_info["max_score"], 4)
                        write_jsonl(score_f, {
                            "clip_id": clip_id,
                            "question_type": qtype_with_suffix,
                            "reward_type": task["reward_type"],
                            "correct_answer": record["correct_answer"],
                            "extracted_answer": score_info["extracted_answer"],
                            "score": score_info["score"],
                            "max_score": score_info["max_score"],
                            "normalised_score": normalised,
                            "correct": score_info["correct"],
                            "method": score_info["method"],
                            "justification": score_info.get("justification", "")
                        })
                        n_ok += 1
                    except Exception as e:
                        log.error(f"Error scoring clip {clip_id} ({qtype_with_suffix}): {e}\n{traceback.format_exc()}")
                        n_error += 1
                else:
                    n_ok += 1
                    
            pbar.set_postfix(ok=n_ok, skip=n_skip, err=n_error)
    finally:
        resp_f.close()
        if score_f is not None:
            score_f.close()
            
    if args.mode == "inference":
        log.info("Inference-only mode run completed. Output is recorded offline. Skipping compilation of summaries.")
        return {}
            
    # Aggregate summary
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


def run_full_video_evaluation(model, processor, records: list[dict], judge, output_dir: str, tag: str, args, primary_device) -> dict:
    """Executes full-video narration and sequence ordering evaluation for Qwen3-VL."""
    os.makedirs(output_dir, exist_ok=True)
    responses_path = os.path.join(output_dir, f"{tag}_responses.jsonl")
    scores_path = os.path.join(output_dir, f"{tag}_scores.jsonl")
    
    # Resume checking setup
    if args.mode == "inference":
        processed_ids = get_processed_ids_from_responses_full(responses_path)
        log.info(f"Inference-only mode. Resuming from responses file. Pre-existing count: {len(processed_ids)}")
    else:
        processed_ids = get_processed_ids_full(scores_path)
        log.info(f"Resuming full-video evaluation: {len(processed_ids)} tasks already scored.")
        
    n_ok = n_skip = n_error = 0
    
    resp_f = open(responses_path, "a", encoding="utf-8")
    score_f = open(scores_path, "a", encoding="utf-8") if args.mode != "inference" else None
         
    try:
        pbar = tqdm(records, desc=f"Full Video Eval [{tag}]", leave=True, dynamic_ncols=True)
        for record in pbar:
            yt_id = record["yt_id"]
            video_path = record["video_path"]
            
            # --- Task 1: Narration ---
            if (yt_id, "narration") in processed_ids:
                n_skip += 1
            else:
                question_text = record["narration_question"] + NARRATION_INFERENCE_SUFFIX
                pbar.set_postfix_str(f"Narr {yt_id}", refresh=True)
                
                model_response = run_qwen3vl_generation(
                    model=model,
                    processor=processor,
                    video_path=video_path,
                    question_text=question_text,
                    max_frames=args.max_frames,
                    max_pixels=args.max_pixels,
                    min_pixels=args.min_pixels,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    primary_device=primary_device,
                    log_id=f"{yt_id}/narration"
                )
                
                if model_response is None:
                    n_error += 1
                else:
                    write_jsonl(resp_f, {
                        "yt_id": yt_id,
                        "task_type": "narration",
                        "question_text": question_text,
                        "reference_narration": record["narration_reference"],
                        "model_response": model_response
                    })
                    
                    if args.mode != "inference" and score_f is not None:
                        pbar.set_postfix_str(f"Judge Narr {yt_id}", refresh=True)
                        try:
                            score_info = judge.score_narration(
                                reference_narration=record["narration_reference"],
                                model_response=model_response
                            )
                            write_jsonl(score_f, {
                                "yt_id": yt_id,
                                "task_type": "narration",
                                **score_info
                            })
                            n_ok += 1
                        except Exception as e:
                            log.error(f"Error scoring narration for {yt_id}: {e}\n{traceback.format_exc()}")
                            n_error += 1
                    else:
                        n_ok += 1
            
            # --- Task 2: Sequence Ordering (Direct Prompting) ---
            if (yt_id, "sequence_ordering_direct") in processed_ids:
                n_skip += 1
            else:
                question_text = record["ordering_question"] + ORDERING_DIRECT_INFERENCE_SUFFIX
                pbar.set_postfix_str(f"Order Dir {yt_id}", refresh=True)
                
                model_response = run_qwen3vl_generation(
                    model=model,
                    processor=processor,
                    video_path=video_path,
                    question_text=question_text,
                    max_frames=args.max_frames,
                    max_pixels=args.max_pixels,
                    min_pixels=args.min_pixels,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    primary_device=primary_device,
                    log_id=f"{yt_id}/ordering_direct"
                )
                
                if model_response is None:
                    n_error += 1
                else:
                    write_jsonl(resp_f, {
                        "yt_id": yt_id,
                        "task_type": "sequence_ordering_direct",
                        "question_text": question_text,
                        "correct_answer": record["correct_answer"],
                        "model_response": model_response
                    })
                    
                    if args.mode != "inference" and score_f is not None:
                        pbar.set_postfix_str(f"Judge Order Dir {yt_id}", refresh=True)
                        try:
                            score_info = judge.score_ordering(
                                question_text=question_text,
                                correct_answer=record["correct_answer"],
                                model_response=model_response
                            )
                            write_jsonl(score_f, {
                                "yt_id": yt_id,
                                "task_type": "sequence_ordering_direct",
                                **score_info
                            })
                            n_ok += 1
                        except Exception as e:
                            log.error(f"Error scoring sequence ordering direct for {yt_id}: {e}\n{traceback.format_exc()}")
                            n_error += 1
                    else:
                        n_ok += 1
                        
            # --- Task 3: Sequence Ordering (Visual CoT) ---
            if (yt_id, "sequence_ordering_cot") in processed_ids:
                n_skip += 1
            else:
                original_q = record["ordering_question"]
                if "\n\nOutput only" in original_q:
                    base_q = original_q.split("\n\nOutput only")[0]
                else:
                    base_q = original_q
                question_text_cot = base_q.strip() + ORDERING_COT_INFERENCE_SUFFIX
                
                pbar.set_postfix_str(f"Order CoT {yt_id}", refresh=True)
                
                model_response = run_qwen3vl_generation(
                    model=model,
                    processor=processor,
                    video_path=video_path,
                    question_text=question_text_cot,
                    max_frames=args.max_frames,
                    max_pixels=args.max_pixels,
                    min_pixels=args.min_pixels,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    primary_device=primary_device,
                    log_id=f"{yt_id}/ordering_cot"
                )
                
                if model_response is None:
                    n_error += 1
                else:
                    write_jsonl(resp_f, {
                        "yt_id": yt_id,
                        "task_type": "sequence_ordering_cot",
                        "question_text": question_text_cot,
                        "correct_answer": record["correct_answer"],
                        "model_response": model_response
                    })
                    
                    if args.mode != "inference" and score_f is not None:
                        pbar.set_postfix_str(f"Judge Order CoT {yt_id}", refresh=True)
                        try:
                            score_info = judge.score_ordering(
                                question_text=question_text_cot,
                                correct_answer=record["correct_answer"],
                                model_response=model_response
                            )
                            write_jsonl(score_f, {
                                "yt_id": yt_id,
                                "task_type": "sequence_ordering_cot",
                                **score_info
                            })
                            n_ok += 1
                        except Exception as e:
                            log.error(f"Error scoring sequence ordering cot for {yt_id}: {e}\n{traceback.format_exc()}")
                            n_error += 1
                    else:
                        n_ok += 1
                        
            pbar.set_postfix(ok=n_ok, skip=n_skip, err=n_error)
    finally:
        resp_f.close()
        if score_f is not None:
            score_f.close()
            
    if args.mode == "inference":
        log.info("Inference-only mode run completed. Output is recorded offline. Skipping compilation of summaries.")
        return {}
            
    # Aggregate narration and ordering summary
    narration_rows = []
    ordering_direct_rows = []
    ordering_cot_rows = []
    
    if os.path.exists(scores_path):
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
        "model_id": args.model_id,
        "tag": tag,
        "run_stats": {"ok": n_ok, "skip": n_skip, "error": n_error},
        "narration": narration_summary,
        "sequence_ordering_direct": ordering_direct_summary,
        "sequence_ordering_cot": ordering_cot_summary
    }

    summary_path = os.path.join(output_dir, f"{tag}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
        
    return summary


def run(args, records: dict, judge) -> dict:
    """Main runner for Qwen3-VL inference called by main.py."""
    # Force max_new_tokens to 4096 to prevent response truncation
    args.max_new_tokens = 4096

    # 0. Skip loading model if judge-only mode is active
    if args.mode == "judge":
        log.info("Mode 'judge' is active. Skipping model initialization for Qwen3-VL.")
        return {}

    # 1. Setup quantization and memory budgets
    quant_config = None
    if args.load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
    elif args.load_in_8bit:
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True
        )
        
    n_gpus = torch.cuda.device_count()
    max_memory = None
    if args.gpu_memory_budget and n_gpus > 0:
        max_memory = {i: args.gpu_memory_budget for i in range(n_gpus)}
        log.info(f"Loading Qwen3-VL with max_memory={max_memory}")

    # 2. Load model & processor
    log.info(f"Loading Qwen3-VL processor: {args.model_id}")
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    
    log.info(f"Loading Qwen3-VL model: {args.model_id}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id,
        quantization_config=quant_config,
        device_map="auto",
        max_memory=max_memory,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    model.eval()
    
    primary_device = first_device(model)
    log.info(f"Primary model device determined: {primary_device}")
    
    hf_map = getattr(model, "hf_device_map", {})
    if hf_map:
        gpu_counts = {}
        for dev in hf_map.values():
            gpu_counts[str(dev)] = gpu_counts.get(str(dev), 0) + 1
        log.info(f"Layer distribution: {gpu_counts}")
        
    log.info(f"VRAM after model load: {vram_stats()}")
    
    summaries = {}
    
    # 3. Dispatch to data level evaluations
    if args.data_level in ("clip", "both"):
        log.info("Starting clip-level Qwen3-VL evaluation...")
        clip_records = records.get("clip", [])
        if clip_records:
            summaries["clip"] = run_clip_evaluation(
                model=model,
                processor=processor,
                records=clip_records,
                judge=judge,
                output_dir=args.output_dir,
                tag=f"{args.tag}_clip",
                args=args,
                primary_device=primary_device
            )
        else:
            log.warning("No clip-level records loaded.")
            
    if args.data_level in ("full", "both"):
        log.info("Starting full-video Qwen3-VL evaluation...")
        full_records = records.get("full", [])
        if full_records:
            summaries["full"] = run_full_video_evaluation(
                model=model,
                processor=processor,
                records=full_records,
                judge=judge,
                output_dir=args.output_dir,
                tag=f"{args.tag}_full",
                args=args,
                primary_device=primary_device
            )
        else:
            log.warning("No full-video records loaded.")
            
    flush_memory(model, processor)
    return summaries
