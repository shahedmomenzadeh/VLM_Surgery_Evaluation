# main.py
# Command-line entry point to orchestrate VLM model family evaluations

import os
import sys
import argparse
import logging
from datetime import datetime

# Configure base logging before importing other modules
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger("main_orchestrator")

import dataset_loader
from llm_judge import LLMJudge


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmarking Cataract Surgery VLMs (HuluMed and Qwen3-VL)")
    
    # Execution mode
    parser.add_argument("--mode", type=str, default="all", choices=["all", "inference", "judge"],
                        help="Execution mode: 'inference' (only generate responses), 'judge' (only grade pre-generated responses offline), or 'all' (both sequentially).")
    
    # Model parameters
    parser.add_argument("--model-family", type=str, required=True, choices=["hulumed", "qwen3vl"],
                        help="Model family architecture type.")
    parser.add_argument("--model-id", type=str, required=True,
                        help="Hugging Face model identifier or path.")
    
    # Dataset configuration
    parser.add_argument("--dataset-root", type=str, default=None,
                        help="Absolute path to cataract surgery VLM dataset root (optional/not required in 'judge' mode).")
    parser.add_argument("--splits", type=str, nargs="+", default=["Test"], choices=["Train", "Validation", "Test"],
                        help="Split name or names to evaluate.")
    parser.add_argument("--data-level", type=str, default="clip", choices=["clip", "full", "both"],
                        help="Data resolution level: clip (short segment), full (narration/reordering), or both.")
    
    # Inference parameters
    parser.add_argument("--output-dir", type=str, default="./results",
                        help="Directory to write results JSONL files and summary reports.")
    parser.add_argument("--tag", type=str, default=None,
                        help="Tag identifier for output filenames. If not set, generated automatically.")
    
    parser.add_argument("--max-frames", type=int, default=8,
                        help="Maximum frames to sample from each video.")
    parser.add_argument("--frame-size", type=int, default=224,
                        help="HuluMed video frame height/width resizing parameter.")
    parser.add_argument("--fps", type=float, default=1.0,
                        help="HuluMed video sampling frame rate.")
    
    parser.add_argument("--max-pixels", type=int, default=307200,
                        help="Qwen3-VL max pixels parameter (default: 640*480).")
    parser.add_argument("--min-pixels", type=int, default=100352,
                        help="Qwen3-VL min pixels parameter (default: 128*28*28).")
    
    parser.add_argument("--max-new-tokens", type=int, default=4096,
                        help="Maximum new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Model sampling temperature. Use 0.0 for greedy decoding.")
    
    # Quantization and Memory parameters
    parser.add_argument("--load-in-4bit", action="store_true", default=True,
                        help="Load model in 4-bit NF4 format using bitsandbytes (default: True).")
    parser.add_argument("--no-4bit", dest="load_in_4bit", action="store_false",
                        help="Disable 4-bit quantization loading.")
    parser.add_argument("--load-in-8bit", action="store_true", default=False,
                        help="Load model in 8-bit format using bitsandbytes.")
    parser.add_argument("--gpu-memory-budget", type=str, default=None,
                        help="Per-GPU memory budget constraint (e.g., '12GiB') to balance load.")
    
    # Judge API settings
    parser.add_argument("--judge-base-url", type=str, default="https://openrouter.ai/api/v1",
                        help="Base URL for OpenAI-compatible LLM judge API endpoint.")
    parser.add_argument("--judge-model", type=str, default="openai/gpt-oss-120b:free",
                        help="LLM judge model identifier.")
    parser.add_argument("--judge-api-key-env", type=str, default="OPENROUTER_API_KEY",
                        help="Environment variable name that holds LLM judge API key.")
    parser.add_argument("--judge-retries", type=int, default=3,
                        help="Number of times to retry judge API calls on failure.")
    
    # Operational parameters
    parser.add_argument("--dry-run", action="store_true",
                        help="Only load the dataset records and print samples. Does not instantiate models.")
                        
    return parser.parse_args()


def print_summary_comparison(model_family: str, summaries: dict):
    """Formats and prints final summary tables based on evaluation run."""
    print("\n" + "=" * 60)
    print(f"  {model_family.upper()} EVALUATION SUMMARY")
    print("=" * 60)
    
    if "clip" in summaries and summaries["clip"]:
        clip_sum = summaries["clip"]
        print("\n  CLIP-LEVEL EVALUATION SUMMARY:")
        print(f"    Overall Normalised Accuracy : {clip_sum.get('overall_normalised_accuracy', 0.0):.4f}")
        if "run_stats" in clip_sum:
            stats = clip_sum["run_stats"]
            print(f"    Total Processed Questions   : {stats.get('ok', 0)}")
            print(f"    Total Skipped               : {stats.get('skip', 0)}")
            print(f"    Errors                      : {stats.get('error', 0)}")
        print("    Per Question Type Metrics:")
        for qtype, metrics in clip_sum.get("per_type", {}).items():
            print(f"      - {qtype:<30}: {metrics.get('avg_normalised_score'):.4f} (n={metrics.get('n_samples')})")
            
    if "full" in summaries and summaries["full"]:
        full_sum = summaries["full"]
        print("\n  FULL-VIDEO LEVEL EVALUATION SUMMARY:")
        if "run_stats" in full_sum:
            stats = full_sum["run_stats"]
            print(f"    Videos Successfully Evaluated: {stats.get('ok', 0) // 3}")
            print(f"    Videos Skipped                : {stats.get('skip', 0) // 3}")
            print(f"    Errors                        : {stats.get('error', 0)}")
        
        narration = full_sum.get("narration", {})
        print("\n    1. NARRATION METRICS:")
        print(f"      - Overall Narration Score : {narration.get('avg_overall_score')}/5")
        print(f"      - Step Coverage           : {narration.get('avg_step_coverage')}/5")
        print(f"      - Chronological Accuracy  : {narration.get('avg_chronological_accuracy')}/5")
        print(f"      - Visual/Tech Accuracy    : {narration.get('avg_visual_technical_accuracy')}/5")
        print(f"      - Narrative Flow          : {narration.get('avg_narrative_flow')}/5")
        
        ordering_dir = full_sum.get("sequence_ordering_direct", {})
        ordering_cot = full_sum.get("sequence_ordering_cot", {})
        
        if not ordering_dir and "sequence_ordering" in full_sum:
            ordering_dir = full_sum["sequence_ordering"]

        print("\n    2. SEQUENCE ORDERING METRICS (DIRECT PROMPTING):")
        if ordering_dir:
            print(f"      - Kendall's Tau Score     : {ordering_dir.get('avg_kendalls_tau')}")
            print(f"      - Exact Sequence Match    : {ordering_dir.get('exact_match_rate')}")
            print(f"      - Valid Extraction Rate   : {ordering_dir.get('valid_rate')} (Extraction Methods: {ordering_dir.get('extraction_methods')})")
        else:
            print("      - No direct sequence ordering results found.")

        print("\n    3. SEQUENCE ORDERING METRICS (VISUAL CoT):")
        if ordering_cot:
            print(f"      - Kendall's Tau Score     : {ordering_cot.get('avg_kendalls_tau')}")
            print(f"      - Exact Sequence Match    : {ordering_cot.get('exact_match_rate')}")
            print(f"      - Valid Extraction Rate   : {ordering_cot.get('valid_rate')} (Extraction Methods: {ordering_cot.get('extraction_methods')})")
        else:
            print("      - No Visual CoT sequence ordering results found.")
        
    print("=" * 60)


def main():
    args = parse_args()
    
    # Validate arguments for root dataset directory
    if args.mode != "judge" and not args.dataset_root:
        log.error("Error: --dataset-root is required in 'inference' or 'all' modes.")
        sys.exit(1)
    
    # 1. Determine tag
    if args.tag is None:
        model_name_clean = args.model_id.split("/")[-1].replace("-", "_").lower()
        args.tag = f"{args.model_family}_{model_name_clean}"
        
    log.info(f"Model ID: {args.model_id}")
    log.info(f"Tag Label: {args.tag}")
    log.info(f"Execution Mode: {args.mode}")
    
    # 2. Retrieve Judge API Key
    judge_api_key = os.environ.get(args.judge_api_key_env, "")
    if not judge_api_key and args.mode != "inference":
        log.warning(f"Judge API key env var '{args.judge_api_key_env}' is not set. API judge calls will fail or fall back.")
        
    # 3. Handle Offline Judge-Only Mode
    if args.mode == "judge":
        log.info("Mode 'judge' is active. Starting offline responses evaluation...")
        judge = LLMJudge(
            base_url=args.judge_base_url,
            api_key=judge_api_key,
            model=args.judge_model,
            retries=args.judge_retries
        )
        summaries = {}
        
        # Clip Evaluation
        if args.data_level in ("clip", "both"):
            suffix = "_clip"
            resp_path = os.path.join(args.output_dir, f"{args.tag}{suffix}_responses.jsonl")
            score_path = os.path.join(args.output_dir, f"{args.tag}{suffix}_scores.jsonl")
            sum_path = os.path.join(args.output_dir, f"{args.tag}{suffix}_summary.json")
            
            if os.path.exists(resp_path):
                log.info(f"Grading clip responses from {resp_path}...")
                summaries["clip"] = judge.grade_responses_file(
                    responses_path=resp_path,
                    scores_path=score_path,
                    summary_path=sum_path,
                    level="clip",
                    model_id=args.model_id,
                    tag=args.tag
                )
            else:
                log.error(f"Responses file not found: {resp_path}. Cannot perform offline clip grading.")
                
        # Full Video Evaluation
        if args.data_level in ("full", "both"):
            suffix = "_full"
            resp_path = os.path.join(args.output_dir, f"{args.tag}{suffix}_responses.jsonl")
            score_path = os.path.join(args.output_dir, f"{args.tag}{suffix}_scores.jsonl")
            sum_path = os.path.join(args.output_dir, f"{args.tag}{suffix}_summary.json")
            
            if os.path.exists(resp_path):
                log.info(f"Grading full video responses from {resp_path}...")
                summaries["full"] = judge.grade_responses_file(
                    responses_path=resp_path,
                    scores_path=score_path,
                    summary_path=sum_path,
                    level="full",
                    model_id=args.model_id,
                    tag=args.tag
                )
            else:
                log.error(f"Responses file not found: {resp_path}. Cannot perform offline full video grading.")
                
        print_summary_comparison(args.model_family, summaries)
        sys.exit(0)
        
    # 4. Load Dataset (only for 'all' and 'inference' modes)
    records = {}
    if args.data_level in ("clip", "both"):
        log.info(f"Loading clip-level dataset from {args.dataset_root}...")
        records["clip"] = dataset_loader.load_clip_records(
            dataset_root=args.dataset_root,
            splits=args.splits,
            validate_videos=True
        )
        
    if args.data_level in ("full", "both"):
        log.info(f"Loading full-video level dataset from {args.dataset_root}...")
        records["full"] = dataset_loader.load_full_video_records(
            dataset_root=args.dataset_root,
            splits=args.splits,
            validate_videos=True
        )
        
    # Handle dry-run mode
    if args.dry_run:
        log.info("Dry-run mode activated. Printing sample records and exiting.")
        if "clip" in records and records["clip"]:
            log.info(f"Clip sample count: {len(records['clip'])}")
            log.info(f"Clip record sample: {records['clip'][0]}")
        if "full" in records and records["full"]:
            log.info(f"Full-video sample count: {len(records['full'])}")
            log.info(f"Full-video record sample: {records['full'][0]}")
        sys.exit(0)
        
    # 5. Instantiate LLM Judge
    judge = LLMJudge(
        base_url=args.judge_base_url,
        api_key=judge_api_key,
        model=args.judge_model,
        retries=args.judge_retries
    )
    
    # 6. Dispatch model VLM inference runner
    summaries = {}
    if args.model_family == "hulumed":
        # Import dynamically inside venv to avoid Qwen dependency errors
        import hulumed_inference
        summaries = hulumed_inference.run(args, records, judge)
    elif args.model_family == "qwen3vl":
        import qwen3VL_inference
        summaries = qwen3VL_inference.run(args, records, judge)
        
    # 7. Print summary metrics (only in 'all' mode where judging was executed immediately)
    if args.mode == "all":
        print_summary_comparison(args.model_family, summaries)
    else:
        log.info(f"VLM inference run completed successfully. Raw output saved in: {args.output_dir}")
        log.info("To score the outputs, copy response files and run: python main.py --mode judge [args]")


if __name__ == "__main__":
    main()
