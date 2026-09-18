# video_timestamping.py
# Burns an elapsed time code (MM:SS) onto the top-right corner of videos for VLM time-grounding.

import os
import sys
import shutil
import hashlib
import logging
import subprocess
from pathlib import Path
from typing import Optional

log = logging.getLogger("video_timestamping")

# Standard TrueType font candidates across Linux distributions
FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
]


def get_default_font_path() -> Optional[str]:
    """Finds the best available bold sans font on the system."""
    for font_path in FONT_CANDIDATES:
        if os.path.exists(font_path):
            return font_path
    return None


def get_video_hash(video_path: str, extra_params: str = "") -> str:
    """Computes a deterministic hash based on video file path, size, mtime, and parameters."""
    stat = os.stat(video_path)
    key_data = f"{os.path.abspath(video_path)}:{stat.st_size}:{stat.st_mtime}:{extra_params}"
    return hashlib.sha256(key_data.encode("utf-8")).hexdigest()[:16]


def build_drawtext_filter(
    font_path: Optional[str] = None,
    time_format: str = "mmss",
    fontsize: int = 48,
    fontcolor: str = "white",
    boxcolor: str = "black@0.85",
    boxborderw: int = 10,
    margin_x: int = 30,
    margin_y: int = 30
) -> str:
    """
    Constructs the ffmpeg drawtext filter expression.
    - mmss: MM:SS formatted string using trunc(t/60) and mod(t, 60)
    - hms:  HH:MM:SS format via ffmpeg %{pts:hms}
    """
    if font_path is None:
        font_path = get_default_font_path()

    font_arg = f"fontfile='{font_path}':" if font_path else ""

    if time_format.lower() == "mmss":
        # Format as 00:00 (MM:SS)
        text_expr = r"%{eif\:trunc(t/60)\:d\:2}\:%{eif\:mod(t\,60)\:d\:2}"
    else:
        # Fallback to HH:MM:SS
        text_expr = r"%{pts\:hms}"

    return (
        f"drawtext={font_arg}"
        f"text='{text_expr}':"
        f"fontcolor={fontcolor}:"
        f"fontsize={fontsize}:"
        f"box=1:"
        f"boxcolor={boxcolor}:"
        f"boxborderw={boxborderw}:"
        f"x=w-tw-{margin_x}:"
        f"y={margin_y}"
    )


def extract_preview_frame(
    video_path: str,
    timestamp_sec: float,
    output_image_path: str,
    time_format: str = "mmss",
    fontsize: int = 48,
    fontcolor: str = "white",
    boxcolor: str = "black@0.85",
    boxborderw: int = 10,
    margin_x: int = 30,
    margin_y: int = 30,
    font_path: Optional[str] = None
) -> str:
    """Extracts a single timestamped frame for visual inspection and verification."""
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg binary not found in system PATH.")

    os.makedirs(os.path.dirname(os.path.abspath(output_image_path)), exist_ok=True)
    vf = build_drawtext_filter(
        font_path=font_path,
        time_format=time_format,
        fontsize=fontsize,
        fontcolor=fontcolor,
        boxcolor=boxcolor,
        boxborderw=boxborderw,
        margin_x=margin_x,
        margin_y=margin_y
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-ss", str(timestamp_sec),
        "-vf", vf,
        "-vframes", "1",
        "-update", "1",
        output_image_path
    ]

    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg preview frame extraction failed:\n{res.stderr}")

    return output_image_path


def get_or_create_timestamped_video(
    video_path: str,
    cache_dir: str = "./timestamped_cache",
    time_format: str = "mmss",
    fontsize: int = 48,
    fontcolor: str = "white",
    boxcolor: str = "black@0.85",
    boxborderw: int = 10,
    margin_x: int = 30,
    margin_y: int = 30,
    font_path: Optional[str] = None,
    force_reencode: bool = False
) -> str:
    """
    Returns the path to a cached timestamped video, burning the timecode via ffmpeg if not already cached.
    """
    if not shutil.which("ffmpeg"):
        log.warning("ffmpeg not found! Falling back to un-timestamped video.")
        return video_path

    if not os.path.exists(video_path):
        log.error(f"Input video not found: {video_path}")
        return video_path

    os.makedirs(cache_dir, exist_ok=True)

    params_str = f"{time_format}_{fontsize}_{fontcolor}_{boxcolor}_{boxborderw}_{margin_x}_{margin_y}"
    vhash = get_video_hash(video_path, params_str)
    base_name = Path(video_path).stem
    cached_video_name = f"{base_name}_ts_{vhash}.mp4"
    cached_path = os.path.join(cache_dir, cached_video_name)

    # Check if cached video exists and is non-empty
    if not force_reencode and os.path.exists(cached_path) and os.path.getsize(cached_path) > 1024:
        log.debug(f"Using cached timestamped video: {cached_path}")
        return cached_path

    log.info(f"Burning {time_format.upper()} timestamp into video: {video_path} -> {cached_path}")
    vf = build_drawtext_filter(
        font_path=font_path,
        time_format=time_format,
        fontsize=fontsize,
        fontcolor=fontcolor,
        boxcolor=boxcolor,
        boxborderw=boxborderw,
        margin_x=margin_x,
        margin_y=margin_y
    )

    temp_path = f"{cached_path}.tmp.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vf", vf,
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "18",
        "-c:a", "copy",
        temp_path
    ]

    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        log.error(f"ffmpeg timestamp encoding failed for {video_path}:\n{res.stderr}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return video_path

    os.replace(temp_path, cached_path)
    log.info(f"Successfully generated timestamped video: {cached_path} ({os.path.getsize(cached_path)/1e6:.1f} MB)")
    return cached_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test and preview video timestamping.")
    parser.add_argument("--video", type=str, default="evaluation_dataset/0xUbMicNy-w_full_video.mp4",
                        help="Input video path.")
    parser.add_argument("--timestamp-sec", type=float, default=75.0,
                        help="Seek timestamp in seconds for preview frame.")
    parser.add_argument("--output-preview", type=str, default="/tmp/preview_timestamped.png",
                        help="Output path for preview image.")
    parser.add_argument("--format", type=str, default="mmss", choices=["mmss", "hms"],
                        help="Time format (mmss or hms).")
    parser.add_argument("--fontsize", type=int, default=48, help="Font size in pixels.")
    parser.add_argument("--boxcolor", type=str, default="black@0.85", help="Background box color and opacity.")
    parser.add_argument("--burn-full", action="store_true", help="Burn full video to cache.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    print(f"Extracting sample frame at {args.timestamp_sec}s from {args.video}...")
    img = extract_preview_frame(
        video_path=args.video,
        timestamp_sec=args.timestamp_sec,
        output_image_path=args.output_preview,
        time_format=args.format,
        fontsize=args.fontsize,
        boxcolor=args.boxcolor
    )
    print(f"Preview frame extracted to: {img}")

    if args.burn_full:
        print(f"Burning full video...")
        v = get_or_create_timestamped_video(
            video_path=args.video,
            time_format=args.format,
            fontsize=args.fontsize,
            boxcolor=args.boxcolor
        )
        print(f"Timestamped video ready at: {v}")
