#!/usr/bin/env python3
# fairness_experiment.py
# Runs the LLM judge k times on identical model responses to quantify judge scoring stability

import os
import sys
import json
import time
import argparse
import logging
import threading
import statistics
import concurrent.futures
from collections import defaultdict

from tqdm import tqdm

from llm_judge import LLMJudge

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("fairness_experiment")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fairness experiment: run the LLM judge k times on identical model "
                    "responses and quantify judge scoring stability."
    )
    parser.add_argument("--tag", type=str, required=True,
                        help="Model tag, e.g. 'qwen3vl_qwen3_vl_8b_instruct' (no _clip/_full suffix).")
    parser.add_argument("--k", type=int, default=3,
                        help="Number of repeated judge runs per response.")
    parser.add_argument("-w", "--num-workers", type=int, default=1,
                        help="Number of concurrent judge worker threads (default: 1).")
    parser.add_argument("--output-dir", type=str, default="./results",
                        help="Directory holding <tag>_clip_responses.jsonl / <tag>_full_responses.jsonl.")
    parser.add_argument("--judge-base-url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--judge-model", type=str, default="qwen3.8-max")
    parser.add_argument("--judge-api-key-env", type=str, default="PROVIDER_API_KEY")
    parser.add_argument("--judge-retries", type=int, default=3)
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Sleep seconds between judge API calls per thread to avoid rate limits.")
    parser.add_argument("--skip-clip", action="store_true", help="Skip clip visual description scoring.")
    parser.add_argument("--skip-narration", action="store_true", help="Skip narration scoring.")
    return parser.parse_args()


def load_jsonl(path: str) -> list[dict]:
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def write_jsonl(fh, record: dict) -> None:
    fh.write(json.dumps(record) + "\n")
    fh.flush()


def is_valid_score(row: dict) -> bool:
    """
    Checks if a scored response record is valid.
    Filters out failed judge calls and fallback placeholders so they can be retried.
    """
    if not isinstance(row, dict):
        return False
    method = str(row.get("method", ""))
    justification = str(row.get("justification", ""))
    if method in ("llm_judge_failed", "llm_judge_fallback"):
        return False
    if "All LLM judge attempts failed" in justification or "LLM client not initialized" in justification:
        return False
    if "score" in row and row["score"] is None:
        return False
    if "overall_score" in row and row["overall_score"] is None:
        return False
    return True


def r4(x: float) -> float:
    return round(x, 4)


def describe(values: list[float]) -> dict:
    vals = [v for v in values if v is not None]
    if not vals:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None, "values": []}
    return {
        "n": len(vals),
        "mean": r4(statistics.mean(vals)),
        "std": r4(statistics.stdev(vals)) if len(vals) > 1 else 0.0,
        "min": min(vals),
        "max": max(vals),
        "values": vals,
    }


def aggregate(samples: list[dict], value_key: str = "values") -> dict:
    valid = [s for s in samples if s.get(value_key)]
    if not valid:
        return {"n_samples": 0}
    all_vals = [v for s in valid for v in s[value_key]]
    dist = defaultdict(int)
    for v in all_vals:
        dist[int(v)] += 1
    means = [statistics.mean(s[value_key]) for s in valid]
    stds = [statistics.stdev(s[value_key]) if len(s[value_key]) > 1 else 0.0 for s in valid]
    avg_mean = statistics.mean(means)
    avg_std = statistics.mean(stds)
    return {
        "n_samples": len(valid),
        "total_runs": len(all_vals),
        "avg_score": r4(avg_mean),
        "avg_std": r4(avg_std),
        "coefficient_of_variation": r4(avg_std / avg_mean) if avg_mean > 0 else None,
        "perfect_consistency_rate": r4(sum(1 for s in valid if len(set(s[value_key])) == 1) / len(valid)),
        "within_one_point_rate": r4(sum(1 for s in valid if (max(s[value_key]) - min(s[value_key])) <= 1) / len(valid)),
        "score_distribution": {str(k): dist[k] for k in sorted(dist)},
    }


def strip_values(obj):
    if isinstance(obj, dict):
        return {k: strip_values(v) for k, v in obj.items() if k != "values"}
    if isinstance(obj, list):
        return [strip_values(v) for v in obj]
    return obj


def run_clip_fairness(judge, args) -> dict:
    resp_path = os.path.join(args.output_dir, f"{args.tag}_clip_responses.jsonl")
    out_path = os.path.join(args.output_dir, f"{args.tag}_fairness_clip_scores.jsonl")

    records = [r for r in load_jsonl(resp_path) if r.get("reward_type") == "llm_judge"]
    if not records:
        log.warning(f"No clip responses with reward_type='llm_judge' in {resp_path}. Skipping clip fairness.")
        return {"n_samples": 0}

    # Load existing valid scores and prune failed records from disk
    existing_rows = load_jsonl(out_path)
    valid_scores = {}
    dirty = False
    for row in existing_rows:
        key = (row.get("sample_id"), row.get("run_index"))
        if is_valid_score(row):
            valid_scores[key] = row
        else:
            dirty = True

    if dirty or len(valid_scores) < len(existing_rows):
        log.info(f"Found {len(existing_rows) - len(valid_scores)} invalid/failed clip score record(s) in {out_path}. Pruning for retry...")
        tmp_path = f"{out_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            for row in valid_scores.values():
                write_jsonl(f, row)
        os.replace(tmp_path, out_path)

    tasks = []
    for record in records:
        resp_id = record.get("record_id") or record.get("clip_id")
        sample_id = f"{resp_id}::{record['question_type']}"
        for run_idx in range(args.k):
            if (sample_id, run_idx) not in valid_scores:
                tasks.append((record, run_idx))

    total_needed = len(records) * args.k
    log.info(f"Clip fairness: {len(records)} samples x {args.k} runs = {total_needed} tasks ({len(valid_scores)} cached, {len(tasks)} pending/retrying). Concurrency: {args.num_workers} worker(s).")
    
    n_calls = 0
    file_lock = threading.Lock()

    def process_task(task_item):
        nonlocal n_calls
        record, run_idx = task_item
        resp_id = record.get("record_id") or record.get("clip_id")
        sample_id = f"{resp_id}::{record['question_type']}"
        try:
            ref_desc = record.get("reference_description") or record.get("reference_reasoning", "")
            result = judge.score_description(
                reference_description=ref_desc,
                model_response=record["model_response"]
            )
        except Exception as e:
            log.error(f"Judge call failed for {sample_id} run {run_idx}: {e}")
            return None

        if not is_valid_score(result):
            log.warning(f"Judge returned invalid/failed response for {sample_id} run {run_idx}: {result.get('justification', 'unknown error')}. Will retry on next run.")
            return None

        out_record = {
            "sample_id": sample_id,
            "record_id": resp_id,
            "question_type": record["question_type"],
            "correct_answer": record.get("correct_answer", ""),
            "run_index": run_idx,
            "score": result["score"],
            "extracted_answer": result.get("extracted_answer", ""),
            "justification": result.get("justification", "")
        }

        with file_lock:
            with open(out_path, "a", encoding="utf-8") as f:
                write_jsonl(f, out_record)
            valid_scores[(sample_id, run_idx)] = out_record
            n_calls += 1

        if args.delay > 0:
            time.sleep(args.delay)

        return out_record

    if tasks:
        if args.num_workers > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
                futures = [executor.submit(process_task, t) for t in tasks]
                with tqdm(total=len(tasks), desc=f"Clip Fairness [{args.tag}]", dynamic_ncols=True) as pbar:
                    for future in concurrent.futures.as_completed(futures):
                        pbar.update(1)
        else:
            with tqdm(tasks, desc=f"Clip Fairness [{args.tag}]", dynamic_ncols=True) as pbar:
                for t in pbar:
                    process_task(t)

    per_sample = {}
    for row in valid_scores.values():
        sid = row["sample_id"]
        per_sample.setdefault(sid, {"scores": [], "answers": []})
        per_sample[sid]["scores"].append(row["score"])
        per_sample[sid]["answers"].append(row.get("extracted_answer", ""))

    samples = []
    for sid, agg in per_sample.items():
        desc = describe(agg["scores"])
        samples.append({
            "sample_id": sid,
            "n": desc["n"],
            "mean": desc["mean"],
            "std": desc["std"],
            "min": desc["min"],
            "max": desc["max"],
            "values": desc["values"],
            "answers": agg["answers"],
            "answer_letter_consistent": len(set(agg["answers"])) <= 1,
        })

    summary = {
        **aggregate(samples),
        "judge_calls_made": n_calls,
        "answer_letter_consistency_rate": (
            r4(sum(1 for s in samples if s["answer_letter_consistent"]) / len(samples))
            if samples else None
        ),
    }
    return summary


NARRATION_DIMS = [
    "step_coverage",
    "chronological_accuracy",
    "visual_technical_accuracy",
    "narrative_flow",
    "overall_score",
]


def run_narration_fairness(judge, args) -> dict:
    resp_path = os.path.join(args.output_dir, f"{args.tag}_full_responses.jsonl")
    out_path = os.path.join(args.output_dir, f"{args.tag}_fairness_narration_scores.jsonl")

    records = [r for r in load_jsonl(resp_path) if r.get("task_type") == "narration"]
    if not records:
        log.warning(f"No narration responses in {resp_path}. Skipping narration fairness.")
        return {"n_samples": 0}

    # Load existing valid scores and prune failed records from disk
    existing_rows = load_jsonl(out_path)
    valid_scores = {}
    dirty = False
    for row in existing_rows:
        key = (row.get("sample_id"), row.get("run_index"))
        if is_valid_score(row):
            valid_scores[key] = row
        else:
            dirty = True

    if dirty or len(valid_scores) < len(existing_rows):
        log.info(f"Found {len(existing_rows) - len(valid_scores)} invalid/failed narration score record(s) in {out_path}. Pruning for retry...")
        tmp_path = f"{out_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            for row in valid_scores.values():
                write_jsonl(f, row)
        os.replace(tmp_path, out_path)

    tasks = []
    for record in records:
        resp_id = record.get("record_id") or record.get("yt_id")
        sample_id = resp_id
        for run_idx in range(args.k):
            if (sample_id, run_idx) not in valid_scores:
                tasks.append((record, run_idx))

    total_needed = len(records) * args.k
    log.info(f"Narration fairness: {len(records)} samples x {args.k} runs = {total_needed} tasks ({len(valid_scores)} cached, {len(tasks)} pending/retrying). Concurrency: {args.num_workers} worker(s).")
    
    n_calls = 0
    file_lock = threading.Lock()

    def process_task(task_item):
        nonlocal n_calls
        record, run_idx = task_item
        resp_id = record.get("record_id") or record.get("yt_id")
        sample_id = resp_id
        try:
            result = judge.score_narration(
                reference_narration=record.get("reference_narration", ""),
                model_response=record["model_response"]
            )
        except Exception as e:
            log.error(f"Judge call failed for {sample_id} run {run_idx}: {e}")
            return None

        if not is_valid_score(result):
            log.warning(f"Judge returned invalid/failed response for {sample_id} run {run_idx}: {result.get('justification', 'unknown error')}. Will retry on next run.")
            return None

        out_record = {
            "sample_id": sample_id,
            "record_id": resp_id,
            "run_index": run_idx,
            **{d: result[d] for d in NARRATION_DIMS},
            "justification": result.get("justification", "")
        }

        with file_lock:
            with open(out_path, "a", encoding="utf-8") as f:
                write_jsonl(f, out_record)
            valid_scores[(sample_id, run_idx)] = out_record
            n_calls += 1

        if args.delay > 0:
            time.sleep(args.delay)

        return out_record

    if tasks:
        if args.num_workers > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
                futures = [executor.submit(process_task, t) for t in tasks]
                with tqdm(total=len(tasks), desc=f"Narr Fairness [{args.tag}]", dynamic_ncols=True) as pbar:
                    for future in concurrent.futures.as_completed(futures):
                        pbar.update(1)
        else:
            with tqdm(tasks, desc=f"Narr Fairness [{args.tag}]", dynamic_ncols=True) as pbar:
                for t in pbar:
                    process_task(t)

    per_sample = defaultdict(lambda: defaultdict(list))
    for row in valid_scores.values():
        for d in NARRATION_DIMS:
            per_sample[row["sample_id"]][d].append(row[d])

    samples = []
    for sid, dims in per_sample.items():
        samples.append({
            "sample_id": sid,
            "dimensions": {d: describe(vals) for d, vals in dims.items()},
            "identical_across_runs": all(len(set(vals)) == 1 for vals in dims.values()),
        })

    return {
        "n_samples": len(samples),
        "judge_calls_made": n_calls,
        "dimensions": {
            d: aggregate([{"values": s["dimensions"][d]["values"]} for s in samples])
            for d in NARRATION_DIMS
        },
        "perfect_consistency_rate": (
            r4(sum(1 for s in samples if s["identical_across_runs"]) / len(samples))
            if samples else None
        ),
    }


def print_summary(summary: dict) -> None:
    print("\n" + "=" * 60)
    print(f"  LLM-JUDGE FAIRNESS REPORT [{summary.get('tag')}]  k={summary.get('k')}")
    print("=" * 60)

    clip = summary.get("clip")
    if clip and clip.get("n_samples", 0) > 0:
        print("\n  CLIP VISUAL DESCRIPTION (0-5):")
        print(f"    Samples                 : {clip['n_samples']}")
        print(f"    Judge calls made        : {clip['judge_calls_made']}")
        print(f"    Avg score               : {clip['avg_score']}")
        print(f"    Avg per-sample std      : {clip['avg_std']}")
        print(f"    Coefficient of variation: {clip['coefficient_of_variation']}")
        print(f"    Perfect consistency     : {clip['perfect_consistency_rate']:.4f}")
        print(f"    Within 1 point          : {clip['within_one_point_rate']:.4f}")
        print(f"    Answer letter stable    : {clip.get('answer_letter_consistency_rate')}")
        print(f"    Run score distribution  : {clip['score_distribution']}")

    narr = summary.get("narration")
    if narr and narr.get("n_samples", 0) > 0:
        print("\n  NARRATION (0-5):")
        print(f"    Samples                      : {narr['n_samples']}")
        print(f"    Judge calls made             : {narr['judge_calls_made']}")
        print(f"    Perfect consistency (all dims): {narr['perfect_consistency_rate']:.4f}")
        for d, agg in narr["dimensions"].items():
            print(f"    {d:<31}: avg={agg['avg_score']}  std={agg['avg_std']}  "
                  f"CV={agg['coefficient_of_variation']}  perfect={agg['perfect_consistency_rate']:.4f}  "
                  f"dist={agg['score_distribution']}")

    print("=" * 60)


def main():
    args = parse_args()

    api_key = os.environ.get(args.judge_api_key_env, "").strip()
    # Local self-hosted endpoints (e.g. http://localhost:8000/v1) need no API key
    # — LLMJudge substitutes a placeholder key for loopback addresses.
    local_endpoint = "localhost" in args.judge_base_url or "127.0.0.1" in args.judge_base_url
    if not api_key and not local_endpoint:
        log.error(f"Judge API key env var '{args.judge_api_key_env}' is not set.")
        sys.exit(1)

    judge = LLMJudge(
        base_url=args.judge_base_url,
        api_key=api_key,
        model=args.judge_model,
        retries=args.judge_retries,
        num_workers=args.num_workers
    )

    log.info(f"Tag: {args.tag} | k={args.k} | workers={args.num_workers} | judge={args.judge_model} | output={args.output_dir}")

    summary = {
        "tag": args.tag,
        "k": args.k,
        "judge_model": args.judge_model,
        "judge_base_url": args.judge_base_url,
    }

    if not args.skip_clip:
        summary["clip"] = run_clip_fairness(judge, args)
    if not args.skip_narration:
        summary["narration"] = run_narration_fairness(judge, args)

    summary_path = os.path.join(args.output_dir, f"{args.tag}_fairness_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(strip_values(summary), f, indent=4)

    print_summary(summary)
    log.info(f"Fairness summary written to {summary_path}")


if __name__ == "__main__":
    main()
