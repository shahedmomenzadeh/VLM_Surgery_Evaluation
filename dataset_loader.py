# dataset_loader.py
# Module for loading the flat evaluation dataset (evaluation_dataset/).
#
# The dataset is a single flat folder where:
#   - every record is one .jsonl file containing a single JSON line
#   - every record has a common envelope:
#       record_id, task_category, question_type, reward_type, split, track, video, metadata
#   - clip-level categories: visual_description (llm_judge), mcq (deterministic), phase (deterministic)
#   - full-video category: narration (llm_judge)
#
# Automatically detects and flattens Hugging Face format datasets (Parquet + videos/)
# if a Hugging Face repository or Parquet dataset directory is supplied.

import os
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

from flatten_dataset import (
    is_flat_dataset_dir,
    is_hf_dataset_dir,
    flatten_hf_dataset,
    download_hf_dataset,
    prepare_flat_dataset
)

log = logging.getLogger("dataset_loader")

CLIP_CATEGORIES = {"visual_description", "mcq", "phase"}
FULL_CATEGORIES = {"narration"}


def ensure_flat_dataset(
    dataset_root_or_repo: str,
    flatten_dir: Optional[str] = None,
    video_dir: Optional[str] = None,
    copy_videos: bool = False,
    token: Optional[str] = None
) -> str:
    """
    Ensures that a flat evaluation dataset directory is ready for loading.

    Args:
        dataset_root_or_repo: Path to flat folder, HF dataset folder, or HF repo ID.
        flatten_dir: Custom target folder to store flattened dataset.
        video_dir: Optional explicit directory containing the .mp4 videos.
        copy_videos: If True, copies videos instead of creating links.
        token: Hugging Face authentication token.

    Returns:
        Absolute path to the validated flat dataset directory.
    """
    return prepare_flat_dataset(
        dataset_root_or_repo=dataset_root_or_repo,
        flatten_dir=flatten_dir,
        video_dir=video_dir,
        copy_videos=copy_videos,
        token=token
    )


def _iter_records(dataset_root: str, splits: list[str]):
    """Yields (jsonl_path, record) for every record whose split is in `splits`."""
    root_path = Path(dataset_root)
    if not root_path.is_dir():
        log.warning(f"Dataset root {root_path} not found.")
        return

    for jf in sorted(root_path.glob("*.jsonl")):
        try:
            with open(jf, "r", encoding="utf-8") as f:
                first_line = f.readline().strip()
            if not first_line:
                continue
            record = json.loads(first_line)
            if record.get("split") not in splits:
                continue
            yield jf, record
        except Exception as e:
            log.error(f"Error reading {jf}: {e}")


def _extract_user_text(record: dict) -> str:
    """Extracts the user question text from `messages[]` or `prompt[]`."""
    for key in ("messages", "prompt"):
        for m in record.get(key, []):
            if m.get("role") != "user":
                continue
            for block in m.get("content", []):
                if block.get("type") == "text":
                    return block.get("text", "")
    return ""


def _extract_reference(record: dict) -> str:
    """Extracts the assistant reference (description / narration) from `messages[]`."""
    for m in record.get("messages", []):
        if m.get("role") == "assistant":
            return str(m.get("content", "")).strip()
    return str(record.get("reference_reasoning", "")).strip()


def _resolve_video(dataset_root: str, video_name: str) -> str:
    """Flat resolution: <root>/<video filename>."""
    return str(Path(dataset_root) / video_name)


def load_clip_records(
    dataset_root: str,
    splits: list[str],
    validate_videos: bool = True,
    flatten_dir: Optional[str] = None,
    video_dir: Optional[str] = None,
    hf_token: Optional[str] = None
) -> list[dict]:
    """
    Loads clip-level evaluation records (visual_description, mcq, phase)
    from the flat evaluation dataset. Automatically flattens HF datasets if required.

    Args:
        dataset_root: Path to flat dataset folder, HF dataset folder, or HF repo ID.
        splits: List of splits to process, e.g., ["Test"].
        validate_videos: Whether to check that the referenced video file exists.
        flatten_dir: Custom output folder if flattening is needed.
        video_dir: Optional explicit directory containing the .mp4 videos.
        hf_token: Optional Hugging Face token for downloading.

    Returns:
        List of dicts representing clip evaluation tasks.
    """
    resolved_root = ensure_flat_dataset(
        dataset_root_or_repo=dataset_root,
        flatten_dir=flatten_dir,
        video_dir=video_dir,
        token=hf_token
    )

    records = []
    for jf, record in _iter_records(resolved_root, splits):
        category = record.get("task_category")
        if category not in CLIP_CATEGORIES:
            continue

        video_name = record.get("video", "")
        video_abs_path = _resolve_video(resolved_root, video_name)
        if validate_videos and (not video_name or not os.path.exists(video_abs_path)):
            log.warning(f"Video file not found at {video_abs_path}. Skipping {record.get('record_id')}.")
            continue

        records.append({
            "record_id": record.get("record_id", jf.stem),
            "task_category": category,
            "question_type": record.get("question_type", "unknown"),
            "reward_type": record.get("reward_type", "deterministic"),
            "split": record.get("split"),
            "track": record.get("track"),
            "video_path": video_abs_path,
            "relative_video_path": video_name,
            "question_text": _extract_user_text(record),
            "correct_answer": record.get("correct_answer", ""),
            "reference_reasoning": record.get("reference_reasoning", ""),
            "reference_description": _extract_reference(record) or record.get("reference_reasoning", ""),
            "metadata": record.get("metadata", {}),
        })

    log.info(f"Loaded {len(records)} clip-level records from split(s): {splits}")
    return records


def load_full_video_records(
    dataset_root: str,
    splits: list[str],
    validate_videos: bool = True,
    flatten_dir: Optional[str] = None,
    video_dir: Optional[str] = None,
    hf_token: Optional[str] = None
) -> list[dict]:
    """
    Loads full-video level evaluation records (narration only) from the flat dataset.
    Automatically flattens HF datasets if required.

    Args:
        dataset_root: Path to flat dataset folder, HF dataset folder, or HF repo ID.
        splits: List of splits to process, e.g., ["Test"].
        validate_videos: Whether to check that the full_video.mp4 exists.
        flatten_dir: Custom output folder if flattening is needed.
        video_dir: Optional explicit directory containing the .mp4 videos.
        hf_token: Optional Hugging Face token for downloading.

    Returns:
        List of dicts containing full-video narration questions and reference answers.
    """
    resolved_root = ensure_flat_dataset(
        dataset_root_or_repo=dataset_root,
        flatten_dir=flatten_dir,
        video_dir=video_dir,
        token=hf_token
    )

    records = []
    for jf, record in _iter_records(resolved_root, splits):
        if record.get("task_category") not in FULL_CATEGORIES:
            continue

        video_name = record.get("video", "")
        video_abs_path = _resolve_video(resolved_root, video_name)
        if validate_videos and (not video_name or not os.path.exists(video_abs_path)):
            log.warning(f"Video file not found at {video_abs_path}. Skipping {record.get('record_id')}.")
            continue

        metadata = record.get("metadata", {})
        records.append({
            "record_id": record.get("record_id", jf.stem),
            "yt_id": metadata.get("parent_video_id", record.get("record_id", jf.stem)),
            "task_category": "narration",
            "question_type": record.get("question_type", "narration"),
            "reward_type": record.get("reward_type", "llm_judge"),
            "split": record.get("split"),
            "video_path": video_abs_path,
            "relative_video_path": video_name,
            "narration_question": _extract_user_text(record),
            "narration_reference": _extract_reference(record),
            "metadata": metadata,
        })

    log.info(f"Loaded {len(records)} full-video records from split(s): {splits}")
    return records
