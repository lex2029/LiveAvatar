"""Video enhancement module — face restoration + upscaling.

Supports two modes (configurable via LIVEAVATAR_ENHANCER env):
  - "codeformer" — CodeFormer face restoration + RealESRGAN background upscale
  - "realesrgan" — RealESRGAN only (upscale everything, no face-specific restore)
  - "" / "none" — disabled (default, use plain ffmpeg scale_cuda)

Usage in worker:
    from liveavatar.utils.video_enhancer import enhance_video
    enhance_video(input_path, output_path, output_size="720*1280")
"""

import os
import time
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch


_ENHANCER_CACHE = {}


def _get_realesrgan(model_path: str, scale: int = 2, device: str = "cuda"):
    key = f"realesrgan_{scale}"
    if key not in _ENHANCER_CACHE:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64,
            num_block=23, num_grow_ch=32, scale=scale,
        )
        upsampler = RealESRGANer(
            scale=scale,
            model_path=model_path,
            model=model,
            tile=512,
            tile_pad=10,
            pre_pad=0,
            half=True,
            device=device,
        )
        _ENHANCER_CACHE[key] = upsampler
    return _ENHANCER_CACHE[key]


def _get_codeformer(
    codeformer_path: str,
    realesrgan_path: str,
    device: str = "cuda",
):
    key = "codeformer"
    if key not in _ENHANCER_CACHE:
        from gfpgan import GFPGANer

        bg_upsampler = _get_realesrgan(realesrgan_path, scale=2, device=device)

        face_enhancer = GFPGANer(
            model_path=codeformer_path,
            upscale=2,
            arch="CodeFormer",
            channel_multiplier=2,
            bg_upsampler=bg_upsampler,
            device=device,
        )
        _ENHANCER_CACHE[key] = face_enhancer
    return _ENHANCER_CACHE[key]


def enhance_video(
    input_path: Path,
    output_path: Path,
    output_size: Optional[str] = None,
    fps: int = 25,
    enhancer_type: Optional[str] = None,
    progress_callback=None,
) -> None:
    """Enhance video with face restoration and/or upscaling.

    Args:
        input_path: Path to input video.
        output_path: Path to write enhanced video.
        output_size: Target size as "H*W" (e.g. "720*1280"). If None, uses 2x input.
        fps: Output framerate.
        enhancer_type: "codeformer", "realesrgan", or None/empty to auto-detect from env.
        progress_callback: Optional callback(progress_pct, processed_frames, total_frames).
    """
    if enhancer_type is None:
        enhancer_type = os.getenv("LIVEAVATAR_ENHANCER", "").strip().lower()
    if not enhancer_type or enhancer_type == "none":
        raise ValueError("No enhancer type specified")

    ckpt_dir = Path(os.getenv(
        "LIVEAVATAR_ENHANCER_CKPT_DIR",
        str(Path(__file__).resolve().parent.parent.parent / "ckpt" / "enhancers"),
    ))
    realesrgan_path = str(ckpt_dir / "RealESRGAN_x2plus.pth")
    codeformer_path = str(ckpt_dir / "codeformer.pth")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Read input video
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Determine output size
    if output_size:
        out_h, out_w = map(int, output_size.split("*"))
    else:
        out_w, out_h = in_w * 2, in_h * 2

    # Init enhancer
    started_at = time.perf_counter()
    if enhancer_type == "codeformer":
        enhancer = _get_codeformer(codeformer_path, realesrgan_path, device)
    elif enhancer_type == "realesrgan":
        enhancer = _get_realesrgan(realesrgan_path, scale=2, device=device)
    else:
        raise ValueError(f"Unknown enhancer type: {enhancer_type}")

    init_time = time.perf_counter() - started_at
    logging.info(f"Enhancer '{enhancer_type}' initialized in {init_time:.1f}s")

    # Process frames
    import subprocess
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{out_w}x{out_h}", "-r", str(fps),
        "-i", "pipe:0",
        "-i", str(input_path),  # for audio
        "-map", "0:v:0", "-map", "1:a:0?",
        "-c:v", "h264_nvenc", "-preset", "p4",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest", "-movflags", "+faststart",
        str(output_path),
    ]
    proc = subprocess.Popen(
        ffmpeg_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    processed = 0
    last_progress = -1
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Enhance frame
            if enhancer_type == "codeformer":
                fidelity = float(os.getenv("LIVEAVATAR_CODEFORMER_FIDELITY", "0.7"))
                _, _, enhanced = enhancer.enhance(
                    frame,
                    has_aligned=False,
                    only_center_face=False,
                    paste_back=True,
                    weight=fidelity,
                )
            elif enhancer_type == "realesrgan":
                enhanced, _ = enhancer.enhance(frame, outscale=2)

            # Resize to target
            if enhanced.shape[1] != out_w or enhanced.shape[0] != out_h:
                enhanced = cv2.resize(enhanced, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)

            proc.stdin.write(enhanced.tobytes())
            processed += 1

            if progress_callback and total_frames > 0:
                pct = int(100 * processed / total_frames)
                if pct != last_progress:
                    try:
                        progress_callback(pct, processed, total_frames)
                    except Exception:
                        pass
                    last_progress = pct

        proc.stdin.close()
    except Exception:
        proc.kill()
        raise
    finally:
        cap.release()

    proc.wait()
    if proc.returncode != 0:
        stderr = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
        raise RuntimeError(f"ffmpeg failed (exit {proc.returncode}): {stderr[:500]}")

    elapsed = time.perf_counter() - started_at
    logging.info(
        f"Enhanced {processed} frames in {elapsed:.1f}s "
        f"({processed/elapsed:.1f} fps, enhancer={enhancer_type})"
    )
