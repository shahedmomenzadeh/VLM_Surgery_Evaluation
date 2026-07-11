# dataset_loader.py
# Module for loading clip-level and full-video level surgical datasets

import os
import json
import logging
from pathlib import Path

log = logging.getLogger("dataset_loader")

def load_clip_records(dataset_root: str, splits: list[str], validate_videos: bool = True) -> list[dict]:
    """
    Loads clip-level evaluation records from clip_*_grpo.jsonl files.
    
    Args:
        dataset_root: Absolute path to the dataset directory (containing Train/Validation/Test).
        splits: List of splits to process, e.g., ["Test", "Validation", "Train"].
        validate_videos: Whether to check if the corresponding video file exists before loading.
        
    Returns:
        List of dicts representing GRPO test questions.
    """
    records = []
    root_path = Path(dataset_root)
    
    for split in splits:
        split_path = root_path / split
        if not split_path.is_dir():
            log.warning(f"Split directory {split_path} not found.")
            continue
            
        # Iterate over each YouTube ID folder in the split
        for yt_dir in sorted(p for p in split_path.iterdir() if p.is_dir()):
            # Find all clip_*_grpo.jsonl files
            grpo_files = sorted(yt_dir.glob("clip_*_grpo.jsonl"))
            for grpo_file in grpo_files:
                try:
                    with open(grpo_file, "r", encoding="utf-8") as f:
                        for line_idx, line in enumerate(f, 1):
                            line = line.strip()
                            if not line:
                                continue
                            record = json.loads(line)
                            
                            # Extract prompt parts
                            prompt_messages = record.get("prompt", [])
                            if not prompt_messages:
                                continue
                            user_content = prompt_messages[0].get("content", [])
                            
                            # Find video and text blocks
                            video_block = next((b for b in user_content if b.get("type") == "video"), None)
                            text_block = next((b for b in user_content if b.get("type") == "text"), None)
                            
                            if not video_block or not text_block:
                                log.warning(f"Malformed record in {grpo_file} at line {line_idx}. Missing video or text block.")
                                continue
                                
                            relative_video_path = video_block.get("video", "")
                            question_text = text_block.get("text", "")
                            
                            # Resolve video absolute path
                            # relative_video_path is e.g. "0xUbMicNy-w/clip_01.mp4"
                            video_abs_path = str(split_path / relative_video_path)
                            
                            if validate_videos and not os.path.exists(video_abs_path):
                                log.warning(f"Video file not found at {video_abs_path}. Skipping record.")
                                continue
                                
                            # Add split info to the record and normalize fields
                            # So that it is easy to parse later
                            records.append({
                                "clip_id": os.path.splitext(relative_video_path.replace("/", "_"))[0],
                                "yt_id": yt_dir.name,
                                "split": split,
                                "video_path": video_abs_path,
                                "relative_video_path": relative_video_path,
                                "question_text": question_text,
                                "correct_answer": record.get("correct_answer", ""),
                                "question_type": record.get("question_type", "unknown"),
                                "reference_reasoning": record.get("reference_reasoning", ""),
                                "reward_type": record.get("reward_type", "deterministic")
                            })
                except Exception as e:
                    log.error(f"Error reading grpo file {grpo_file}: {e}")
                    
    log.info(f"Loaded {len(records)} clip-level records from split(s): {splits}")
    return records


def load_full_video_records(dataset_root: str, splits: list[str], validate_videos: bool = True) -> list[dict]:
    """
    Loads full-video level evaluation records (narration and chronological step reordering).
    
    Args:
        dataset_root: Absolute path to the dataset directory.
        splits: List of splits to process, e.g., ["Test", "Validation"].
        validate_videos: Whether to check if full_video.mp4 exists.
        
    Returns:
        List of dicts containing full-video questions and reference answers.
    """
    records = []
    root_path = Path(dataset_root)
    
    for split in splits:
        split_path = root_path / split
        if not split_path.is_dir():
            log.warning(f"Split directory {split_path} not found.")
            continue
            
        for yt_dir in sorted(p for p in split_path.iterdir() if p.is_dir()):
            yt_id = yt_dir.name
            sft_path = yt_dir / "full_video_sft.jsonl"
            grpo_path = yt_dir / "full_video_grpo.jsonl"
            video_abs_path = str(yt_dir / "full_video.mp4")
            
            if validate_videos and not os.path.exists(video_abs_path):
                log.warning(f"Full video not found at {video_abs_path}. Skipping.")
                continue
                
            if not sft_path.exists() or not grpo_path.exists():
                log.warning(f"Missing full_video_sft.jsonl or full_video_grpo.jsonl for {yt_id}. Skipping.")
                continue
                
            # Parse Narration (first record in full_video_sft.jsonl)
            try:
                with open(sft_path, "r", encoding="utf-8") as f:
                    first_line = f.readline().strip()
                if not first_line:
                    continue
                sft_record = json.loads(first_line)
                messages = sft_record.get("messages", [])
                
                assistant_msg = next((m for m in messages if m.get("role") == "assistant"), None)
                user_msg = next((m for m in messages if m.get("role") == "user"), None)
                
                if not assistant_msg or not user_msg:
                    continue
                    
                user_content = user_msg.get("content", [])
                text_block = next((b for b in user_content if b.get("type") == "text"), None)
                
                if not text_block:
                    continue
                    
                narration_q = text_block.get("text", "")
                narration_ref = assistant_msg.get("content", "")
            except Exception as e:
                log.error(f"Error parsing SFT narration file for {yt_id}: {e}")
                continue
                
            # Parse Sequence Ordering (first record in full_video_grpo.jsonl)
            try:
                with open(grpo_path, "r", encoding="utf-8") as f:
                    grpo_line = f.readline().strip()
                if not grpo_line:
                    continue
                grpo_record = json.loads(grpo_line)
                prompt_messages = grpo_record.get("prompt", [])
                
                if not prompt_messages:
                    continue
                    
                ordering_user = prompt_messages[0].get("content", [])
                ordering_text_block = next((b for b in ordering_user if b.get("type") == "text"), None)
                
                if not ordering_text_block:
                    continue
                    
                ordering_q = ordering_text_block.get("text", "")
                correct_answer = grpo_record.get("correct_answer", "")
            except Exception as e:
                log.error(f"Error parsing GRPO ordering file for {yt_id}: {e}")
                continue
                
            records.append({
                "yt_id": yt_id,
                "split": split,
                "video_path": video_abs_path,
                "narration_question": narration_q,
                "narration_reference": narration_ref,
                "ordering_question": ordering_q,
                "correct_answer": correct_answer
            })
            
    log.info(f"Loaded {len(records)} full-video records from split(s): {splits}")
    return records