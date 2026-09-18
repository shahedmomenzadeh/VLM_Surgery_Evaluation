# full_video_timestamped_inference.py
# Full-Video Timestamped Procedural Narration Inference & Evaluation Runner
# Evaluates VLMs on complete cataract surgeries (n=15) with burned visual timestamps (MM:SS)

import os
import sys
import json
import argparse
import logging
import traceback
from pathlib import Path
from typing import Optional

# Setup standard logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger("full_video_ts")

import dataset_loader
from llm_judge import LLMJudge
from eval_common import run_full_video_evaluation_loop, flush_memory
from video_timestamping import get_or_create_timestamped_video, extract_preview_frame

DEFAULT_PROMPT_HINT = (
    "\n\nNote: An elapsed time code (MM:SS) is burned into the top-right corner of the video. "
    "Please use these visual timestamps to ground your chronological timestamps accurately."
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Full-Video Cataract Surgery Procedural Narration with Burned Timestamps"
    )

    # Execution mode
    parser.add_argument("--mode", type=str, default="inference", choices=["all", "inference", "judge"],
                        help="Execution mode: 'inference' (generate responses), 'judge' (grade offline), or 'all' (both).")

    # Model configuration
    parser.add_argument("--model-family", type=str, default="qwen3vl",
                        choices=["qwen3vl", "hulumed", "lingshu", "qwen2_5_vl", "mage_vl"],
                        help="Model architecture family.")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen3-VL-2B-Instruct",
                        help="Hugging Face model ID or local checkpoint path.")

    # Dataset configuration
    parser.add_argument("--dataset-root", type=str, default="./evaluation_dataset",
                        help="Path to flat evaluation dataset folder or HF dataset directory.")
    parser.add_argument("--flatten-dir", type=str, default=None,
                        help="Target directory if flattening an HF parquet dataset.")
    parser.add_argument("--splits", type=str, nargs="+", default=["Test"],
                        help="Split(s) to evaluate (default: Test).")
    parser.add_argument("--hf-token", type=str, default=None,
                        help="Hugging Face authentication token.")

    # Output configuration
    parser.add_argument("--output-dir", type=str, default="./results",
                        help="Directory to save responses, scores, and summaries.")
    parser.add_argument("--tag", type=str, default=None,
                        help="Tag identifier for output filenames. If not set, generated automatically.")

    # Timestamp burning configuration
    parser.add_argument("--timestamp-cache-dir", type=str, default="./timestamped_cache",
                        help="Directory to cache burned timestamped videos.")
    parser.add_argument("--timestamp-format", type=str, default="mmss", choices=["mmss", "hms"],
                        help="Timestamp format (default: mmss -> MM:SS).")
    parser.add_argument("--fontsize", type=int, default=48,
                        help="Font size in pixels for timestamp overlay (default: 48).")
    parser.add_argument("--boxcolor", type=str, default="black@0.85",
                        help="Background box color and opacity (default: black@0.85).")
    parser.add_argument("--boxborderw", type=int, default=10,
                        help="Background box padding width in pixels (default: 10).")
    parser.add_argument("--force-reencode", action="store_true", default=False,
                        help="Force re-encoding of timestamped videos even if cache exists.")
    parser.add_argument("--no-timestamp", action="store_true", default=False,
                        help="Skip timestamp burning and use original un-timestamped videos (for baseline comparisons).")

    # Prompt hint configuration
    parser.add_argument("--no-prompt-hint", action="store_true", default=False,
                        help="Do not append the visual timestamp guidance hint to the narration prompt.")
    parser.add_argument("--custom-prompt-hint", type=str, default=None,
                        help="Override default prompt hint string.")

    # Inference hyperparameters
    parser.add_argument("--max-frames", type=int, default=32,
                        help="Maximum frames sampled from full surgery video.")
    parser.add_argument("--max-new-tokens", type=int, default=4096,
                        help="Maximum generated tokens for narration output.")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Sampling temperature (0.0 for greedy).")
    parser.add_argument("--fps", type=float, default=1.0,
                        help="Video sampling frame rate for models that require it (HuluMed/Lingshu).")
    parser.add_argument("--frame-size", type=str, default="480,640",
                        help="Frame resizing parameter for HuluMed (height,width or int).")
    parser.add_argument("--max-pixels", type=int, default=307200,
                        help="Qwen3-VL/Lingshu max_pixels parameter (default: 640*480).")
    parser.add_argument("--min-pixels", type=int, default=100352,
                        help="Qwen3-VL/Lingshu min_pixels parameter (default: 128*28*28).")

    # Attention and quantization
    parser.add_argument("--use-flash-attn", action="store_true", default=False,
                        help="Enable FlashAttention-2 if installed.")
    parser.add_argument("--attn-implementation", type=str, default=None,
                        choices=["sdpa", "flash_attention_2", "eager"],
                        help="Attention backend implementation.")
    parser.add_argument("--load-in-4bit", action="store_true", default=False,
                        help="Load model in 4-bit NF4 format.")
    parser.add_argument("--load-in-8bit", action="store_true", default=False,
                        help="Load model in 8-bit format.")
    parser.add_argument("--gpu-memory-budget", type=str, default=None,
                        help="Per-GPU memory limit (e.g. '15GiB').")

    # Judge API settings
    parser.add_argument("--judge-base-url", type=str, default="https://opencode.ai/zen/go/v1/responses",
                        help="LLM judge endpoint.")
    parser.add_argument("--judge-model", type=str, default="Gemini-3.8-flash",
                        help="LLM judge model.")
    parser.add_argument("--judge-api-key-env", type=str, default="PROVIDER_API_KEY",
                        help="Env variable containing judge API key.")
    parser.add_argument("--judge-retries", type=int, default=3,
                        help="Retry count for judge API requests.")
    parser.add_argument("--num-workers", type=int, default=3,
                        help="Concurrent worker count for judge calls.")

    # Operational
    parser.add_argument("--dry-run", action="store_true", default=False,
                        help="Pre-process videos, print sample prompt, and exit without loading VLM.")

    args = parser.parse_args()

    # Attention resolution
    if args.attn_implementation is None:
        args.attn_implementation = "flash_attention_2" if args.use_flash_attn else "sdpa"

    # Frame size parsing
    if args.frame_size:
        size_str = args.frame_size.strip()
        if "," in size_str:
            args.frame_size = [int(x.strip()) for x in size_str.split(",")]
        elif "x" in size_str:
            args.frame_size = [int(x.strip()) for x in size_str.split("x")]
        else:
            args.frame_size = int(size_str)

    # Generate default tag if not specified
    if not args.tag:
        clean_model = args.model_id.strip("/").replace("/", "_").replace("-", "_").lower()
        suffix = "ts" if not args.no_timestamp else "nots"
        args.tag = f"{args.model_family}_{clean_model}_{suffix}"

    # Forced setting: full-video level only
    args.data_level = "full"

    return args


def prepare_full_video_records(raw_records: list[dict], args) -> list[dict]:
    """
    Applies visual timestamps to videos and appends procedural time guidance to the prompt.
    """
    hint_text = (
        args.custom_prompt_hint if args.custom_prompt_hint is not None
        else DEFAULT_PROMPT_HINT
    )

    processed_records = []
    log.info(f"Preparing {len(raw_records)} full-video records (Timestamping enabled: {not args.no_timestamp})...")

    for i, rec in enumerate(raw_records, 1):
        rec_copy = dict(rec)
        orig_video = rec["video_path"]

        if not args.no_timestamp:
            ts_video = get_or_create_timestamped_video(
                video_path=orig_video,
                cache_dir=args.timestamp_cache_dir,
                time_format=args.timestamp_format,
                fontsize=args.fontsize,
                boxcolor=args.boxcolor,
                boxborderw=args.boxborderw,
                force_reencode=args.force_reencode
            )
            rec_copy["video_path"] = ts_video
            rec_copy["original_video_path"] = orig_video
        else:
            rec_copy["original_video_path"] = orig_video

        # Append prompt hint if enabled
        if not args.no_prompt_hint and not args.no_timestamp:
            rec_copy["narration_question"] = rec["narration_question"] + hint_text

        processed_records.append(rec_copy)

    log.info("All full-video records prepared successfully.")
    return processed_records


def print_narration_summary(tag: str, summary: dict):
    """Prints a formatted evaluation table for full video procedural narration."""
    narration = summary.get("narration", {})
    if not narration or narration.get("n_samples", 0) == 0:
        log.warning(f"No narration results to report for tag: {tag}")
        return

    n_samples = narration.get("n_samples", 0)
    overall = narration.get("avg_overall_score", 0.0)
    norm = narration.get("avg_normalized_score", 0.0)
    cov = narration.get("avg_step_coverage", 0.0)
    chrono = narration.get("avg_chronological_accuracy", 0.0)
    tech = narration.get("avg_visual_technical_accuracy", 0.0)
    flow = narration.get("avg_narrative_flow", 0.0)

    print("\n" + "=" * 78)
    print(f"  FULL-VIDEO PROCEDURAL NARRATION RESULTS (TIMESTAMPED): {tag}")
    print("=" * 78)
    print(f"  Sample Count:                      {n_samples} surgeries")
    print(f"  Overall Score (/5):                {overall:.4f}")
    print(f"  Normalized Overall Score (/1.0):   {norm:.4f}")
    print("-" * 78)
    print(f"  Step Coverage (/5):                {cov:.4f}")
    print(f"  Chronological Accuracy (/5):       {chrono:.4f}")
    print(f"  Visual & Technical Accuracy (/5):  {tech:.4f}")
    print(f"  Narrative Flow (/5):               {flow:.4f}")
    print("=" * 78 + "\n")


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    full_tag = f"{args.tag}_full"
    responses_path = os.path.join(args.output_dir, f"{full_tag}_responses.jsonl")
    scores_path = os.path.join(args.output_dir, f"{full_tag}_scores.jsonl")
    summary_path = os.path.join(args.output_dir, f"{full_tag}_summary.json")

    log.info("==================================================")
    log.info("  Full-Video Timestamped Narration Runner")
    log.info("==================================================")
    log.info(f"Model Family:      {args.model_family}")
    log.info(f"Model ID:          {args.model_id}")
    log.info(f"Output Tag:        {full_tag}")
    log.info(f"Execution Mode:    {args.mode}")
    log.info(f"Timestamp Format:  {args.timestamp_format.upper()} (FontSize={args.fontsize}, Box={args.boxcolor})")
    log.info(f"Timestamp Cache:   {args.timestamp_cache_dir}")
    log.info(f"Max Frames:        {args.max_frames}")
    log.info(f"Max New Tokens:    {args.max_new_tokens}")
    log.info(f"Temperature:       {args.temperature}")
    log.info("==================================================")

    # 1. Resolve Judge API Key
    judge_api_key = os.environ.get(args.judge_api_key_env, "").strip("'\" \r\n")
    if not judge_api_key and args.mode in ("judge", "all"):
        if "localhost" in args.judge_base_url or "127.0.0.1" in args.judge_base_url:
            judge_api_key = "local-no-key"
        else:
            log.warning(f"Judge API key in ${args.judge_api_key_env} is not set. Online grading will fall back.")

    # 2. Instantiate LLM Judge
    judge = LLMJudge(
        base_url=args.judge_base_url,
        api_key=judge_api_key,
        model=args.judge_model,
        retries=args.judge_retries,
        num_workers=args.num_workers
    )

    # 3. Handle Offline Judge-Only Mode
    if args.mode == "judge":
        if not os.path.exists(responses_path):
            log.error(f"Responses file not found: {responses_path}. Cannot grade offline.")
            sys.exit(1)

        log.info(f"Grading full video responses from {responses_path}...")
        summary = judge.grade_responses_file(
            responses_path=responses_path,
            scores_path=scores_path,
            summary_path=summary_path,
            level="full",
            model_id=args.model_id,
            tag=full_tag
        )
        print_narration_summary(full_tag, summary)
        sys.exit(0)

    # 4. Load & Prepare Full-Video Dataset Records
    log.info(f"Loading full-video records from: {args.dataset_root}...")
    raw_records = dataset_loader.load_full_video_records(
        dataset_root=args.dataset_root,
        splits=args.splits,
        validate_videos=True,
        flatten_dir=args.flatten_dir,
        hf_token=args.hf_token or os.environ.get("HF_TOKEN")
    )

    if not raw_records:
        log.error(f"No full-video records found in {args.dataset_root} for split(s): {args.splits}")
        sys.exit(1)

    # In dry-run mode, only prepare 1 record for rapid verification
    if args.dry_run:
        log.info("Dry-run: Limiting preparation to first record for verification.")
        raw_records = raw_records[:1]

    records = prepare_full_video_records(raw_records, args)

    # 5. Handle Dry-Run Mode
    if args.dry_run:
        log.info("Dry-run mode active. Printing sample record and previewing timestamp extraction:")
        sample = records[0]
        log.info(f"Record ID:       {sample['record_id']}")
        log.info(f"Original Video:  {sample.get('original_video_path')}")
        log.info(f"Timestamp Video: {sample['video_path']}")
        log.info(f"Narration Query: {sample['narration_question']}")

        preview_out = os.path.join(args.timestamp_cache_dir, "sample_dryrun_preview.png")
        try:
            extract_preview_frame(
                video_path=sample["video_path"],
                timestamp_sec=75.0,
                output_image_path=preview_out,
                time_format=args.timestamp_format,
                fontsize=args.fontsize,
                boxcolor=args.boxcolor,
                boxborderw=args.boxborderw
            )
            log.info(f"Sample preview frame extracted to: {preview_out}")
        except Exception as e:
            log.warning(f"Could not extract preview frame: {e}")

        log.info("Dry-run verification completed successfully.")
        sys.exit(0)

    # 6. Execute Model Inference
    # Wrap records dictionary for model runners
    records_dict = {"full": records}
    summaries = {}

    log.info(f"Launching {args.model_family} inference on timestamped full videos...")
    if args.model_family == "qwen3vl":
        import qwen3VL_inference
        summaries = qwen3VL_inference.run(args, records_dict, judge)
    elif args.model_family == "hulumed":
        import hulumed_inference
        summaries = hulumed_inference.run(args, records_dict, judge)
    elif args.model_family in ("lingshu", "qwen2_5_vl"):
        import lingshu_inference
        summaries = lingshu_inference.run(args, records_dict, judge)
    elif args.model_family == "mage_vl":
        import mage_vl_inference
        summaries = mage_vl_inference.run(args, records_dict, judge)
    else:
        raise ValueError(f"Unsupported model family: {args.model_family}")

    # 7. Print summary metrics if evaluated
    full_summary = summaries.get("full", {})
    if full_summary:
        print_narration_summary(full_tag, full_summary)
    else:
        log.info(f"Full-video timestamped inference completed. Responses saved in: {responses_path}")


if __name__ == "__main__":
    main()
