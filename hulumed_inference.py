# hulumed_inference.py
# Inference and evaluation execution for HuluMed model series

import os
import gc
import logging
import traceback
import torch
from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig

from eval_common import (
    vram_stats,
    flush_memory,
    probe_total_frames,
    run_clip_evaluation_loop,
    run_full_video_evaluation_loop
)

log = logging.getLogger("hulumed_inference")


def run_hulumed_generation(
    model,
    processor,
    video_path: str,
    question_text: str,
    fps: float,
    max_frames: int,
    frame_size: int | list[int],
    max_new_tokens: int,
    temperature: float,
    log_id: str
) -> str | None:
    """Runs generation with progressive frame-count retry on OOM or short-video errors."""
    model_response = None

    effective_max = max_frames
    probed_total = probe_total_frames(video_path)
    if probed_total and probed_total > 0 and max_frames > probed_total:
        log.info(
            f"{log_id} — Video has {probed_total} total frames, fewer than requested "
            f"{max_frames}. Capping max_frames to {probed_total} for this clip only."
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
            
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": {
                            "video_path": video_path,
                            "fps": fps,
                            "max_frames": attempt_frames,
                            "size": min(frame_size) if isinstance(frame_size, list) else frame_size
                        }
                    },
                    {
                        "type": "text",
                        "text": question_text
                    }
                ]
            }
        ]
        
        try:
            inputs = processor(
                conversation=conversation,
                add_system_prompt=True,
                add_generation_prompt=True,
                return_tensors="pt"
            )
            
            inputs = {
                k: (v.cuda().to(torch.float16) if isinstance(v, torch.Tensor) and v.is_floating_point()
                    else v.cuda() if isinstance(v, torch.Tensor)
                    else v)
                for k, v in inputs.items()
            }
            
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=(temperature > 0.0),
                    temperature=temperature if temperature > 0.0 else None,
                    use_cache=True,
                    pad_token_id=processor.tokenizer.eos_token_id
                )
                
            model_response = processor.batch_decode(
                output_ids,
                skip_special_tokens=True,
                use_think=False
            )[0].strip()
            
            del inputs, output_ids
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


def _patch_hulumed_processor_compatibility():
    """Patches ProcessorMixin to support HulumedProcessor under transformers >= 4.49."""
    import inspect
    import transformers.processing_utils

    orig_from_pretrained = transformers.processing_utils.ProcessorMixin.from_pretrained

    @classmethod
    def patched_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        orig_get_args = getattr(cls, "_get_arguments_from_pretrained", None)
        if orig_get_args is not None and not getattr(orig_get_args, "_patched_compat", False):
            try:
                sig = inspect.signature(orig_get_args)
                pos_params = [
                    p for p in sig.parameters.values()
                    if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                ]
                if len(pos_params) <= 2:
                    @classmethod
                    def wrapped_get_args(subcls, path, *extra_args, **kw):
                        return orig_get_args(path, **kw)
                    wrapped_get_args._patched_compat = True
                    cls._get_arguments_from_pretrained = wrapped_get_args
            except Exception:
                pass
        return orig_from_pretrained.__func__(cls, pretrained_model_name_or_path, *args, **kwargs)

    transformers.processing_utils.ProcessorMixin.from_pretrained = patched_from_pretrained


def run(args, records: dict, judge) -> dict:
    """Main runner for HuluMed inference called by main.py."""
    args.max_new_tokens = 4096

    if args.mode == "judge":
        log.info("Mode 'judge' is active. Skipping model initialization for HuluMed.")
        return {}

    _patch_hulumed_processor_compatibility()

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
        
    log.info(f"Loading HuluMed processor: {args.model_id}")
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    
    log.info(f"Loading HuluMed model: {args.model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        device_map="auto",
        quantization_config=quant_config,
        attn_implementation=getattr(args, "attn_implementation", "sdpa")
    )
    log.info(f"Using attention implementation: {getattr(args, 'attn_implementation', 'sdpa')}")
    model.eval()
    log.info(f"VRAM after model load: {vram_stats()}")
    
    def generate_fn(video_path: str, question_text: str, log_id: str) -> str | None:
        return run_hulumed_generation(
            model=model,
            processor=processor,
            video_path=video_path,
            question_text=question_text,
            fps=args.fps,
            max_frames=args.max_frames,
            frame_size=args.frame_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            log_id=log_id
        )
    
    summaries = {}
    
    if args.data_level in ("clip", "both"):
        log.info("Starting clip-level HuluMed evaluation...")
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
        log.info("Starting full-video HuluMed evaluation...")
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
