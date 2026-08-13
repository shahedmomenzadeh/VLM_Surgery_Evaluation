# mage_vl_inference.py
# Inference and evaluation execution for microsoft/Mage-VL

import os
import gc
import logging
import traceback
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig

from eval_common import (
    vram_stats,
    flush_memory,
    first_device,
    run_clip_evaluation_loop,
    run_full_video_evaluation_loop
)

log = logging.getLogger("mage_vl_inference")


def sample_video(video_path: str, num_frames: int) -> list:
    """
    Uniformly samples up to `num_frames` RGB PIL frames from video.
    Official implementation from Mage-VL reference inference.py.
    """
    import cv2
    import numpy as np

    capture = cv2.VideoCapture(video_path)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        capture.release()
        raise ValueError(f"Could not read video: {video_path}")

    indices = np.linspace(0, frame_count - 1, min(num_frames, frame_count), dtype=int)
    frames = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        if not ok:
            capture.release()
            raise ValueError(f"Could not decode frame {index} from: {video_path}")
        frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    capture.release()
    return frames


def build_inputs(
    processor,
    model_device: torch.device,
    model_dtype: torch.dtype,
    video_path: str,
    question_text: str,
    num_frames: int,
    video_backend: str = "frames",
    model_path: str | None = None,
    max_pixels: int = 150000,
    codec_engine: str = "traditional"
) -> dict:
    """Builds tokenized inputs matching official Mage-VL reference inference.py."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video"},
                {"type": "text", "text": question_text}
            ]
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    if video_backend == "codec":
        codec_config = {
            "engine": "hevc" if codec_engine == "traditional" else "dcvc-rt",
            "target_canvas": num_frames,
            "patch": 16,
        }
        if codec_engine == "neural" and model_path:
            codec_config["dcvc"] = {
                "pkg_dir": os.path.join(model_path, "neural_codec"),
                "device": str(model_device),
            }
        inputs = processor(
            text=[text],
            videos=[video_path],
            video_backend="codec",
            max_pixels=max_pixels,
            codec_config=codec_config,
            return_tensors="pt",
            padding=True,
        )
    else:
        frames = sample_video(video_path, num_frames)
        inputs = processor(
            text=[text],
            videos=[frames],
            return_tensors="pt",
            padding=True,
        )

    inputs = {k: (v.to(model_device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(model_dtype)

    return inputs


def run_mage_vl_generation(
    model,
    processor,
    video_path: str,
    question_text: str,
    max_frames: int,
    max_new_tokens: int,
    primary_device: torch.device,
    log_id: str,
    video_backend: str = "frames",
    model_path: str | None = None,
    max_pixels: int = 150000,
    codec_engine: str = "traditional"
) -> str | None:
    """Runs generation with fallback to smaller frame counts on OOM."""
    model_response = None

    retry_frames = []
    f = max_frames
    while f >= 2:
        retry_frames.append(f)
        f = f // 2
    if not retry_frames:
        retry_frames = [2]

    model_dtype = getattr(model, "dtype", torch.float16)

    for attempt_frames in retry_frames:
        if attempt_frames != max_frames:
            log.warning(f"{log_id} — Retry with {attempt_frames} frames (was {max_frames}).")

        try:
            inputs = build_inputs(
                processor=processor,
                model_device=primary_device,
                model_dtype=model_dtype,
                video_path=video_path,
                question_text=question_text,
                num_frames=attempt_frames,
                video_backend=video_backend,
                model_path=model_path,
                max_pixels=max_pixels,
                codec_engine=codec_engine
            )

            input_len = inputs["input_ids"].shape[1]

            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )

            generated_ids = output_ids[0, input_len:]
            model_response = processor.tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            ).strip()

            del inputs, output_ids, generated_ids
            break
        except torch.cuda.OutOfMemoryError as e:
            log.error(f"{log_id} — CUDA OOM (frames={attempt_frames}): {e} | {vram_stats()}")
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            log.error(f"{log_id} — Generation error: {e}\n{traceback.format_exc()}")
            break

    gc.collect()
    torch.cuda.empty_cache()
    return model_response


def run(args, records: dict, judge) -> dict:
    """Main runner for Mage-VL inference called by main.py."""
    args.max_new_tokens = 4096

    if args.mode == "judge":
        log.info("Mode 'judge' is active. Skipping model initialization for Mage-VL.")
        return {}

    video_backend = getattr(args, "mage_video_backend", "frames")
    codec_engine = getattr(args, "mage_codec_engine", "traditional")
    max_pixels = getattr(args, "max_pixels", 150000)

    model_path = args.model_id
    if video_backend == "codec" and codec_engine == "neural":
        from huggingface_hub import snapshot_download
        log.info(f"Resolving local snapshot path for neural codec package: {args.model_id}")
        model_path = snapshot_download(args.model_id)

    log.info(f"Loading Mage-VL processor: {args.model_id}")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

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
        log.info(f"Loading Mage-VL with max_memory={max_memory}")

    log.info(f"Loading Mage-VL model: {args.model_id}")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quant_config,
            device_map="auto",
            max_memory=max_memory,
            torch_dtype="auto",
            trust_remote_code=True
        )
    except Exception as e:
        if quant_config is not None:
            log.error(
                f"Quantized load failed for Mage-VL ({e}). Retrying in native precision (torch_dtype='auto')..."
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                max_memory=max_memory,
                torch_dtype="auto",
                trust_remote_code=True
            )
        else:
            raise
    model.eval()

    primary_device = first_device(model)
    log.info(f"Primary model device determined: {primary_device}")
    log.info(f"VRAM after model load: {vram_stats()}")

    def generate_fn(video_path: str, question_text: str, log_id: str) -> str | None:
        return run_mage_vl_generation(
            model=model,
            processor=processor,
            video_path=video_path,
            question_text=question_text,
            max_frames=args.max_frames,
            max_new_tokens=args.max_new_tokens,
            primary_device=primary_device,
            log_id=log_id,
            video_backend=video_backend,
            model_path=model_path,
            max_pixels=max_pixels,
            codec_engine=codec_engine
        )

    summaries = {}

    if args.data_level in ("clip", "both"):
        log.info("Starting clip-level Mage-VL evaluation...")
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
        log.info("Starting full-video Mage-VL evaluation...")
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
