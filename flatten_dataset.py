#!/usr/bin/env python3
"""
flatten_dataset.py
Flattens a Hugging Face format dataset (Parquet files + videos/ subfolder)
into the flat evaluation dataset structure (1004 single-line JSONL records + flat MP4 videos).

Can be used as a standalone CLI or imported as a library module.
Supports downloading directly from Hugging Face Hub if a repository ID is provided.
"""

import os
import sys
import json
import shutil
import pathlib
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

log = logging.getLogger("flatten_dataset")


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def download_hf_dataset(
    repo_id: str,
    local_dir: str,
    token: Optional[str] = None,
    allow_patterns: Optional[List[str]] = None,
    ignore_patterns: Optional[List[str]] = None
) -> str:
    """
    Downloads a dataset repository snapshot from Hugging Face Hub.

    Args:
        repo_id: Hugging Face dataset identifier, e.g., 'username/cataract_surgery_vlm_eval'.
        local_dir: Target local directory where the repository should be downloaded.
        token: Optional Hugging Face auth token (falls back to HF_TOKEN env var).
        allow_patterns: Optional list of glob patterns to download.
        ignore_patterns: Optional list of glob patterns to exclude.

    Returns:
        The path to the downloaded local directory.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required to download from Hugging Face. "
            "Install it via `pip install huggingface_hub`."
        )

    auth_token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    log.info(f"Downloading Hugging Face dataset '{repo_id}' to '{local_dir}'...")
    
    downloaded_path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=local_dir,
        token=auth_token,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
        resume_download=True
    )
    log.info(f"Hugging Face dataset successfully downloaded to '{downloaded_path}'.")
    return str(downloaded_path)


def _link_or_copy_file(src: Path, dst: Path, copy_only: bool = False) -> None:
    """Links (symlink or hardlink) or copies a file from src to dst."""
    if dst.exists():
        return

    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy_only:
        shutil.copy2(src, dst)
        return

    # Try symlink first
    try:
        os.symlink(src.resolve(), dst)
        return
    except (OSError, NotImplementedError):
        pass

    # Try hardlink second
    try:
        os.link(src.resolve(), dst)
        return
    except (OSError, NotImplementedError):
        pass

    # Fallback to copy
    shutil.copy2(src, dst)


def _parse_metadata(metadata_raw: Any) -> Dict[str, Any]:
    """Safely extracts dictionary metadata from stringified JSON or dict."""
    if isinstance(metadata_raw, dict):
        return metadata_raw
    if isinstance(metadata_raw, str) and metadata_raw.strip():
        try:
            return json.loads(metadata_raw)
        except json.JSONDecodeError:
            pass
    return {}


def _reconstruct_record(category: str, row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reconstructs the original JSONL envelope for a single evaluation record from a Parquet row.
    """
    record_id = row.get("record_id", "")
    question_type = row.get("question_type", "")
    reward_type = row.get("reward_type", "deterministic")
    split = row.get("split", "Test")
    track = row.get("track", "youtube")
    
    # Handle video column (which may be a string "videos/xxx.mp4" or a dict from datasets.Video)
    video_field = row.get("video", "")
    if isinstance(video_field, dict):
        raw_video_path = video_field.get("path") or video_field.get("bytes") or ""
    else:
        raw_video_path = str(video_field)
    video_filename = os.path.basename(raw_video_path)

    metadata = _parse_metadata(row.get("metadata_json") or row.get("metadata"))
    
    # If metadata was empty in metadata_json, reconstruct from row fields
    if not metadata:
        metadata = {}
        for k in [
            "parent_video_id", "parent_video_title", "parent_video_url", "clip_id",
            "phase_clip_id", "step_number", "step_title", "visual_description",
            "transcript_context", "instruments", "anatomy", "clip_duration_seconds",
            "timestamp_start_in_parent", "timestamp_end_in_parent", "phase_id",
            "phase_name", "site", "subclip_order", "phase_occurrence", "previous_phase",
            "next_phase", "context_bucket", "n_steps", "step_titles"
        ]:
            val = row.get(k)
            if val is not None and val != "" and val != []:
                metadata[k] = val

    prompt_text = row.get("prompt", "")

    if category == "visual_description":
        ref_desc = row.get("reference_description", "")
        return {
            "record_id": record_id,
            "task_category": "visual_description",
            "question_type": question_type,
            "reward_type": reward_type,
            "split": split,
            "track": track,
            "video": video_filename,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": video_filename},
                        {"type": "text", "text": prompt_text}
                    ]
                },
                {
                    "role": "assistant",
                    "content": ref_desc
                }
            ],
            "metadata": metadata
        }

    elif category == "mcq":
        return {
            "record_id": record_id,
            "task_category": "mcq",
            "question_type": question_type,
            "reward_type": reward_type,
            "split": split,
            "track": track,
            "video": video_filename,
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": video_filename},
                        {"type": "text", "text": prompt_text}
                    ]
                }
            ],
            "correct_answer": row.get("correct_answer", ""),
            "reference_reasoning": row.get("reference_reasoning", ""),
            "metadata": metadata
        }

    elif category == "phase_understanding" or category == "phase":
        # Reconstruct structured correct_answer dict
        correct_answer: Any = ""
        if question_type == "boundary_detection":
            ts = row.get("correct_timestamp")
            correct_answer = {"timestamp": float(ts)} if ts is not None else {}
        elif question_type == "temporal_localization":
            st = row.get("correct_start")
            et = row.get("correct_end")
            correct_answer = {
                "start": float(st) if st is not None else 0.0,
                "end": float(et) if et is not None else 0.0
            }
        elif question_type in ("timestamp_to_phase", "contextual_phase_recognition"):
            pid = row.get("correct_phase_id", "")
            pname = row.get("correct_phase_name", "")
            correct_answer = {"phase_id": pid, "phase_name": pname}
        elif "correct_answer" in row and row["correct_answer"]:
            ca = row["correct_answer"]
            correct_answer = json.loads(ca) if isinstance(ca, str) and ca.startswith("{") else ca

        return {
            "record_id": record_id,
            "task_category": "phase",
            "question_type": question_type,
            "reward_type": reward_type,
            "split": split,
            "track": track,
            "video": video_filename,
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": video_filename},
                        {"type": "text", "text": prompt_text}
                    ]
                }
            ],
            "correct_answer": correct_answer,
            "reference_reasoning": row.get("reference_reasoning", ""),
            "metadata": metadata
        }

    elif category == "narration":
        ref_narration = row.get("reference_narration", "")
        return {
            "record_id": record_id,
            "task_category": "narration",
            "question_type": question_type,
            "reward_type": reward_type,
            "split": split,
            "track": track,
            "video": video_filename,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": video_filename},
                        {"type": "text", "text": prompt_text}
                    ]
                },
                {
                    "role": "assistant",
                    "content": ref_narration
                }
            ],
            "metadata": metadata
        }

    else:
        raise ValueError(f"Unknown task category: {category}")


def flatten_hf_dataset(
    hf_source_dir: str | Path,
    output_dir: str | Path,
    video_dir: Optional[str | Path] = None,
    copy_videos: bool = False
) -> str:
    """
    Converts a Hugging Face formatted dataset directory into a flat evaluation dataset directory.

    Args:
        hf_source_dir: Path to directory containing Parquet files (under `data/`) and `videos/`.
        output_dir: Destination path for the flat evaluation dataset.
        video_dir: Optional explicit directory containing the .mp4 videos.
        copy_videos: If True, copies video files instead of attempting to create links.

    Returns:
        The absolute path to the flattened dataset directory.
    """
    src_path = Path(hf_source_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    try:
        import pyarrow.parquet as pq
    except ImportError:
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "Either `pyarrow` or `pandas` is required to read Parquet files. "
                "Install via `pip install pyarrow` or `pip install pandas`."
            )

    log.info(f"Flattening Hugging Face dataset from '{src_path}' to '{out_path}'...")

    # 1. Locate and process Parquet files
    data_dir = src_path / "data" if (src_path / "data").is_dir() else src_path
    parquet_files = list(data_dir.glob("**/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No Parquet files found under {data_dir}")

    total_records = 0
    records_per_category: Dict[str, int] = {}

    for pq_file in sorted(parquet_files):
        # Determine category from parent directory name or filename
        parent_name = pq_file.parent.name
        cat = parent_name if parent_name in (
            "visual_description", "mcq", "phase_understanding", "narration", "phase"
        ) else "unknown"

        if cat == "unknown":
            for candidate in ("visual_description", "mcq", "phase_understanding", "narration", "phase"):
                if candidate in pq_file.stem:
                    cat = candidate
                    break

        try:
            import pyarrow.parquet as pq
            table = pq.read_table(pq_file)
            rows = table.to_pylist()
        except Exception:
            import pandas as pd
            df = pd.read_parquet(pq_file)
            rows = df.to_dict(orient="records")

        for row in rows:
            record_obj = _reconstruct_record(cat, row)
            rec_id = record_obj["record_id"]
            out_jsonl = out_path / f"{rec_id}.jsonl"

            # Write single-line JSONL
            with open(out_jsonl, "w", encoding="utf-8") as f:
                f.write(json.dumps(record_obj, ensure_ascii=False) + "\n")

            total_records += 1
            records_per_category[cat] = records_per_category.get(cat, 0) + 1

    log.info(f"Reconstructed {total_records} JSONL records in '{out_path}'.")
    for c, cnt in sorted(records_per_category.items()):
        log.info(f"  - {c}: {cnt} records")

    # 2. Locate and link/copy video files (.mp4)
    # Search order: explicit video_dir -> videos/ subfolder -> src_path root -> sibling evaluation_dataset / videos
    video_sources = []
    if video_dir:
        video_sources.append(Path(video_dir))
    video_sources.extend([
        src_path / "videos",
        src_path / "evaluation_dataset",
        src_path,
        src_path.parent / "evaluation_dataset",
        src_path.parent / "videos"
    ])
    
    found_videos = []
    for v_dir in video_sources:
        if v_dir.is_dir():
            v_files = list(v_dir.glob("*.mp4"))
            if v_files:
                found_videos = v_files
                log.info(f"Found {len(v_files)} video files in '{v_dir}'.")
                break

    if not found_videos:
        found_videos = list(src_path.glob("**/*.mp4"))

    # Deduplicate video files by filename
    unique_videos: Dict[str, Path] = {}
    for v in found_videos:
        if v.parent != out_path:  # Avoid linking from destination into destination
            unique_videos[v.name] = v

    log.info(f"Linking/copying {len(unique_videos)} video files to '{out_path}'...")
    for v_name, v_src in unique_videos.items():
        dst_path = out_path / v_name
        _link_or_copy_file(v_src, dst_path, copy_only=copy_videos)

    # 3. Copy README.md if present
    readme_src = src_path / "README.md"
    if readme_src.is_file() and not (out_path / "README.md").exists():
        shutil.copy2(readme_src, out_path / "README.md")

    log.info(f"Dataset flattening complete at '{out_path}'.")
    return str(out_path.resolve())


def is_hf_dataset_dir(path: str | Path) -> bool:
    """Checks if a directory follows Hugging Face dataset layout (contains Parquet files or data/ subfolder)."""
    p = Path(path)
    if not p.is_dir():
        return False
    # If it contains .jsonl files directly, it is already a flat dataset
    if list(p.glob("*.jsonl")):
        return False
    # If it contains parquet files or data/ subfolder
    if list(p.glob("**/*.parquet")) or (p / "data").is_dir():
        return True
    return False


def is_flat_dataset_dir(path: str | Path) -> bool:
    """Checks if a directory follows the flat evaluation dataset format."""
    p = Path(path)
    if not p.is_dir():
        return False
    jsonl_count = len(list(p.glob("*.jsonl")))
    return jsonl_count > 0


def prepare_flat_dataset(
    dataset_root_or_repo: str,
    flatten_dir: Optional[str] = None,
    video_dir: Optional[str] = None,
    copy_videos: bool = False,
    token: Optional[str] = None
) -> str:
    """
    Ensures that a flat evaluation dataset directory is ready.
    
    - If `dataset_root_or_repo` is an existing flat directory (with .jsonl files), returns it directly.
    - If `dataset_root_or_repo` is an existing HF dataset directory (with parquet files), flattens it to `flatten_dir`.
    - If `dataset_root_or_repo` is a Hugging Face repo ID, downloads it first then flattens it.

    Returns:
        The path to the ready-to-use flat dataset folder.
    """
    path_obj = Path(dataset_root_or_repo)

    # Case 1: Existing local directory
    if path_obj.is_dir():
        if is_flat_dataset_dir(path_obj):
            log.info(f"Using existing flat evaluation dataset at '{path_obj.resolve()}'.")
            return str(path_obj.resolve())
        elif is_hf_dataset_dir(path_obj):
            target_dir = flatten_dir or (str(path_obj) + "_flattened")
            return flatten_hf_dataset(path_obj, target_dir, video_dir=video_dir, copy_videos=copy_videos)
        else:
            log.warning(f"Directory '{path_obj}' has neither JSONL nor Parquet files. Returning as-is.")
            return str(path_obj.resolve())

    # Case 2: Hugging Face Repository ID (e.g. 'username/repo_name')
    if "/" in dataset_root_or_repo and not path_obj.exists():
        cache_dir = os.path.join(flatten_dir or "./evaluation_dataset_hf_raw")
        downloaded = download_hf_dataset(
            repo_id=dataset_root_or_repo,
            local_dir=cache_dir,
            token=token
        )
        target_flat_dir = flatten_dir or "./evaluation_dataset"
        return flatten_hf_dataset(downloaded, target_flat_dir, video_dir=video_dir, copy_videos=copy_videos)

    return dataset_root_or_repo


def main():
    parser = argparse.ArgumentParser(
        description="Flatten a Hugging Face format dataset into the flat evaluation dataset structure."
    )
    parser.add_argument("--input-dir", type=str, default=None,
                        help="Path to local Hugging Face dataset folder (containing data/ and videos/).")
    parser.add_argument("--hf-repo", type=str, default=None,
                        help="Hugging Face repo ID (e.g., 'username/cataract_surgery_vlm_eval') to download.")
    parser.add_argument("--video-dir", type=str, default=None,
                        help="Optional explicit path to the directory containing .mp4 video files.")
    parser.add_argument("--output-dir", type=str, default="./evaluation_dataset",
                        help="Target output directory for the flat dataset (default: ./evaluation_dataset).")
    parser.add_argument("--copy-videos", action="store_true", default=False,
                        help="Copy MP4 videos instead of creating symlinks/hardlinks.")
    parser.add_argument("--token", type=str, default=None,
                        help="Hugging Face access token for private/gated datasets.")
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="Enable verbose debug logging.")

    args = parser.parse_args()
    setup_logging(args.verbose)

    if not args.input_dir and not args.hf_repo:
        parser.error("Must provide either --input-dir or --hf-repo.")

    source_path = args.input_dir
    if args.hf_repo:
        raw_download_dir = os.path.join(args.output_dir + "_raw_hf")
        source_path = download_hf_dataset(
            repo_id=args.hf_repo,
            local_dir=raw_download_dir,
            token=args.token
        )

    out = flatten_hf_dataset(
        hf_source_dir=source_path,
        output_dir=args.output_dir,
        video_dir=args.video_dir,
        copy_videos=args.copy_videos
    )
    print(f"\nFlat evaluation dataset successfully prepared at: {out}")


if __name__ == "__main__":
    main()
