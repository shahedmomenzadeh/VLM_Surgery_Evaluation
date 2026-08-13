# dataset_loader.py
# Module for loading clip-level and full-video level surgical datasets

import os
import json
import logging
from pathlib import Path

log = logging.getLogger("dataset_loader")

def load_clip_records(dataset_root: str, splits: list[str], validate_videos: bool = True) -> list[dict]:
    """
    Loads clip-level evaluation records:
      1. Visual description records from line 1 of clip_*_sft.jsonl (reward_type: llm_judge)
      2. MCQ / Phase recognition records from clip_*_grpo.jsonl (reward_type: deterministic)
    
    Args:
        dataset_root: Absolute path to the dataset directory (containing Train/Validation/Test).
        splits: List of splits to process, e.g., ["Test", "Validation", "Train"].
        validate_videos: Whether to check if the corresponding video file exists before loading.
        
    Returns:
        List of dicts representing clip evaluation tasks.
    """
    records = []
    root_path = Path(dataset_root)
    
    for split in splits:
        split_path = root_path / split
        if not split_path.is_dir():
            log.warning(f"Split directory {split_path} not found.")
            continue
            
        # Iterate over each procedure directory (YouTube ID or PH_*) in the split
        for dir_entry in sorted(p for p in split_path.iterdir() if p.is_dir()):
            dir_name = dir_entry.name
            
            # 1. Load Visual Description from line 1 of all clip_*_sft.jsonl files
            sft_files = sorted(dir_entry.glob("clip_*_sft.jsonl"))
            for sft_file in sft_files:
                try:
                    clip_stem = sft_file.name.replace("_sft.jsonl", "")
                    with open(sft_file, "r", encoding="utf-8") as f:
                        first_line = f.readline().strip()
                    if not first_line:
                        continue
                    sft_record = json.loads(first_line)
                    messages = sft_record.get("messages", [])
                    user_msg = next((m for m in messages if m.get("role") == "user"), None)
                    assistant_msg = next((m for m in messages if m.get("role") == "assistant"), None)
                    
                    if not user_msg or not assistant_msg:
                        continue
                        
                    user_content = user_msg.get("content", [])
                    video_block = next((b for b in user_content if b.get("type") == "video"), None)
                    text_block = next((b for b in user_content if b.get("type") == "text"), None)
                    
                    relative_video_path = video_block.get("video", "") if video_block else f"{dir_name}/{clip_stem}.mp4"
                    question_text = text_block.get("text", "") if text_block else "Describe what is happening in this cataract surgical video clip."
                    reference_desc = assistant_msg.get("content", "").strip()
                    
                    video_abs_path = str(split_path / relative_video_path)
                    if validate_videos and not os.path.exists(video_abs_path):
                        log.warning(f"Video file not found at {video_abs_path}. Skipping visual description record.")
                        continue
                        
                    records.append({
                        "clip_id": f"{dir_name}_{clip_stem}",
                        "yt_id": dir_name,
                        "split": split,
                        "video_path": video_abs_path,
                        "relative_video_path": relative_video_path,
                        "question_text": question_text,
                        "correct_answer": "",
                        "question_type": "visual_description",
                        "reference_reasoning": reference_desc,
                        "reference_description": reference_desc,
                        "reward_type": "llm_judge"
                    })
                except Exception as e:
                    log.error(f"Error reading sft file {sft_file}: {e}")
                    
            # 2. Load MCQs / Phase recognition from all clip_*_grpo.jsonl files
            grpo_files = sorted(dir_entry.glob("clip_*_grpo.jsonl"))
            for grpo_file in grpo_files:
                try:
                    clip_stem = grpo_file.name.replace("_grpo.jsonl", "")
                    with open(grpo_file, "r", encoding="utf-8") as f:
                        for line_idx, line in enumerate(f, 1):
                            line = line.strip()
                            if not line:
                                continue
                            record = json.loads(line)
                            
                            prompt_messages = record.get("prompt", [])
                            if not prompt_messages:
                                continue
                            user_content = prompt_messages[0].get("content", [])
                            
                            video_block = next((b for b in user_content if b.get("type") == "video"), None)
                            text_block = next((b for b in user_content if b.get("type") == "text"), None)
                            
                            if not video_block or not text_block:
                                log.warning(f"Malformed record in {grpo_file} at line {line_idx}. Missing video or text block.")
                                continue
                                
                            relative_video_path = video_block.get("video", "")
                            question_text = text_block.get("text", "")
                            
                            video_abs_path = str(split_path / relative_video_path)
                            if validate_videos and not os.path.exists(video_abs_path):
                                log.warning(f"Video file not found at {video_abs_path}. Skipping record.")
                                continue
                                
                            qtype = record.get("question_type", "unknown")
                            
                            records.append({
                                "clip_id": f"{dir_name}_{clip_stem}",
                                "yt_id": dir_name,
                                "split": split,
                                "video_path": video_abs_path,
                                "relative_video_path": relative_video_path,
                                "question_text": question_text,
                                "correct_answer": record.get("correct_answer", ""),
                                "question_type": qtype,
                                "reference_reasoning": record.get("reference_reasoning", ""),
                                "reference_description": record.get("reference_reasoning", ""),
                                "reward_type": "deterministic"
                            })
                except Exception as e:
                    log.error(f"Error reading grpo file {grpo_file}: {e}")
                    
    log.info(f"Loaded {len(records)} clip-level records from split(s): {splits}")
    return records


def load_full_video_records(dataset_root: str, splits: list[str], validate_videos: bool = True) -> list[dict]:
    """
    Loads full-video level evaluation records (narration only).
    
    Args:
        dataset_root: Absolute path to the dataset directory.
        splits: List of splits to process, e.g., ["Test", "Validation"].
        validate_videos: Whether to check if full_video.mp4 exists.
        
    Returns:
        List of dicts containing full-video narration questions and reference answers.
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
            video_abs_path = str(yt_dir / "full_video.mp4")
            
            if validate_videos and not os.path.exists(video_abs_path):
                continue
                
            if not sft_path.exists():
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
                
            records.append({
                "yt_id": yt_id,
                "split": split,
                "video_path": video_abs_path,
                "narration_question": narration_q,
                "narration_reference": narration_ref
            })
            
    log.info(f"Loaded {len(records)} full-video records from split(s): {splits}")
    return records