# qwen3VL_inference.py
# Inference and evaluation execution for Qwen3-VL model series

import os
import gc
import re
import logging
import traceback
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

from eval_common import (
    vram_stats,
    flush_memory,
    first_device,
    move_inputs_to_device,
    probe_total_frames,
    run_clip_evaluation_loop,
    run_full_video_evaluation_loop
)

log = logging.getLogger("qwen3vl_inference")


def build_inputs(
    processor,
    video_path: str,
    question_text: str,
    max_frames: int,
    max_pixels: int,
    min_pixels: int,
    primary_device: torch.device
) -> dict:
    """Constructs tokenized multimodal inputs for Qwen3-VL using process_vision_info."""
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
    
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )
    
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

    effective_max = max_frames
    probed_total = probe_total_frames(video_path)
    if probed_total and probed_total > 0 and max_frames > probed_total:
        log.info(
            f"{log_id} — Video has {probed_total} total frames, fewer than requested "
            f"{max_frames}. Capping nframes to {probed_total} for this clip only."
        )
        effective_max = probed_total

    retry_frames = []
    f = effective_max
    while f >= 2:
        retry_frames.append(f)
        f = f // 2
    if not retry_frames:
        retry_frames = [2]

    for attempt_frames in retry_frames:
        if attempt_frames != effective_max:
            log.warning(f"{log_id} — Retry with {attempt_frames} frames (was {effective_max}).")
            
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
                    top_p=0.8 if temperature > 0.0 else None,
                    top_k=20 if temperature > 0.0 else None,
                    min_p=0.0 if temperature > 0.0 else None,
                    repetition_penalty=1.05,
                    use_cache=True,
                    eos_token_id=[151645, 151643],
                    pad_token_id=processor.tokenizer.eos_token_id
                )
                
            generated_ids = output_ids[:, input_len:]
            model_response = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0].strip()
            
            # Clean thinking tags
            model_response = re.sub(r"<think>.*?(?:</think>|$)", "", model_response, flags=re.DOTALL)
            if "</think>" in model_response:
                model_response = model_response.split("</think>")[-1]
                
            model_response = model_response.strip()
            
            del inputs, output_ids, generated_ids
            break
        except torch.cuda.OutOfMemoryError as e:
            log.error(f"{log_id} — CUDA OOM (frames={attempt_frames}): {e} | {vram_stats()}")
            gc.collect()
            torch.cuda.empty_cache()
        except ValueError as e:
            if "nframes" in str(e):
                log.warning(f"{log_id} — Frame count too high for video (frames={attempt_frames}): {e}")
                gc.collect()
                torch.cuda.empty_cache()
            else:
                log.error(f"{log_id} — Generation error: {e}\n{traceback.format_exc()}")
                break
        except Exception as e:
            log.error(f"{log_id} — Generation error: {e}\n{traceback.format_exc()}")
            break
            
    gc.collect()
    torch.cuda.empty_cache()
    return model_response


def run(args, records: dict, judge) -> dict:
    """Main runner for Qwen3-VL inference called by main.py."""
    args.max_new_tokens = 4096

    if args.mode == "judge":
        log.info("Mode 'judge' is active. Skipping model initialization for Qwen3-VL.")
        return {}

    quant_config = None
    if args.load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
    elif args.load_in_8bit:
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
        
    n_gpus = torch.cuda.device_count()
    max_memory = None
    if args.gpu_memory_budget and n_gpus > 0:
        max_memory = {i: args.gpu_memory_budget for i in range(n_gpus)}
        log.info(f"Loading Qwen3-VL with max_memory={max_memory}")

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
    
    def generate_fn(video_path: str, question_text: str, log_id: str) -> str | None:
        return run_qwen3vl_generation(
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
            log_id=log_id
        )
    
    summaries = {}
    
    if args.data_level in ("clip", "both"):
        log.info("Starting clip-level Qwen3-VL evaluation...")
        clip_records = records.get("clip", [])
        if clip_records:
            summaries["clip"] = run_clip_evaluation_loop(
                generate_fn=generate_fn,
                records=clip_records,
                judge=judge,
                output_dir=args.output_dir,
                tag=f"{args.tag}_clip",
                args=args,
                logger=log
            )
        else:
            log.warning("No clip-level records loaded.")
            
    if args.data_level in ("full", "both"):
        log.info("Starting full-video Qwen3-VL evaluation...")
        full_records = records.get("full", [])
        if full_records:
            summaries["full"] = run_full_video_evaluation_loop(
                generate_fn=generate_fn,
                records=full_records,
                judge=judge,
                output_dir=args.output_dir,
                tag=f"{args.tag}_full",
                args=args,
                logger=log
            )
        else:
            log.warning("No full-video records loaded.")
            
    flush_memory(model, processor)
    return summaries
