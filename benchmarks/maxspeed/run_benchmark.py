#!/usr/bin/env python3
import argparse
import traceback
import importlib.util
import inspect
import json
import os
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def load_worker_module():
    spec = importlib.util.spec_from_file_location(
        "smartblog_worker", REPO_ROOT / "smartblog_worker.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def parse_args():
    parser = argparse.ArgumentParser()
    # IMPORTANT:
    # Primary benchmark goal is generation speed in an already-running worker.
    # Cold start / pipeline load timings are diagnostic only and must not be used
    # as the main comparison metric between optimization variants.
    parser.add_argument("--image", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--size", default="832*480")
    parser.add_argument("--infer-frames", type=int, default=48)
    parser.add_argument("--sample-steps", type=int, default=4)
    parser.add_argument("--sample-solver", default="euler")
    parser.add_argument("--guide-scale", type=float, default=0.0)
    parser.add_argument("--base-seed", type=int, default=420)
    parser.add_argument("--enable-online-decode", action="store_true", default=None)
    parser.add_argument("--offload-model", action="store_true", default=None)
    parser.add_argument("--offload-kv-cache", action="store_true", default=None)
    parser.add_argument("--disable-cudnn-attn", action="store_true", default=None)
    parser.add_argument("--disable-flash-attn", action="store_true", default=None)
    parser.add_argument("--cross-attn-chunk-size", type=int, default=256)
    parser.add_argument("--rope-chunk-size", type=int, default=256)
    parser.add_argument("--attn-out-chunk-size", type=int, default=256)
    parser.add_argument("--qkv-chunk-size", type=int, default=256)
    parser.add_argument("--compile", choices=["true", "false"], default="true")
    parser.add_argument("--compile-mode", default=None)
    parser.add_argument("--compile-backend", default="inductor")
    parser.add_argument("--compile-dynamic", choices=["true", "false"], default=None)
    parser.add_argument("--capture-scalar-outputs", choices=["true", "false"], default=None)
    parser.add_argument("--simple-teacache-thresh", type=float, default=None)
    parser.add_argument("--simple-teacache-force-calc-steps", type=int, default=None)
    parser.add_argument("--simple-teacache-poly", default=None)
    parser.add_argument("--simple-adacache", action="store_true", default=False)
    parser.add_argument("--simple-adacache-codebook", default=None)
    parser.add_argument("--num-clip-safety-margin", type=int, default=None)
    parser.add_argument("--use-lightvae", action="store_true", default=False)
    parser.add_argument("--use-tae", action="store_true", default=False)
    parser.add_argument("--vae-path", default=None)
    parser.add_argument("--vae-pruning-rate", type=float, default=None)
    parser.add_argument("--tae-parallel", choices=["true", "false"], default=None)
    parser.add_argument("--full-online-postprocess", action="store_true", default=False)
    parser.add_argument("--relative-dist", default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--result-file", default=None)
    parser.add_argument("--warm-runs", type=int, default=0)
    parser.add_argument("--torch-profiler", action="store_true", default=False)
    parser.add_argument("--direct-final-encode", action="store_true", default=False)
    return parser.parse_args()


def probe_visual_stats(video_path: Path) -> dict:
    from PIL import Image
    import numpy as np

    if not video_path.exists():
        return {
            "exists": False,
            "valid": False,
            "reason": "missing_output",
        }

    def extract_frame(timestamp: float, out_png: Path) -> dict:
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{timestamp:.3f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            str(out_png),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        img = Image.open(out_png).convert("RGB")
        arr = np.array(img)
        return {
            "mean": float(arr.mean()),
            "min": int(arr.min()),
            "max": int(arr.max()),
            "center": arr[arr.shape[0] // 2, arr.shape[1] // 2].tolist(),
            "shape": list(arr.shape),
        }

    duration_cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    duration = float(
        subprocess.run(duration_cmd, check=True, text=True, capture_output=True).stdout.strip()
    )
    first_png = video_path.with_suffix(".first.png")
    mid_png = video_path.with_suffix(".mid.png")
    first = extract_frame(0.0, first_png)
    mid = extract_frame(max(0.0, duration * 0.5), mid_png)
    first_png.unlink(missing_ok=True)
    mid_png.unlink(missing_ok=True)
    valid = max(first["mean"], mid["mean"]) > 2.0 and max(first["max"], mid["max"]) > 8
    return {
        "exists": True,
        "valid": valid,
        "duration_s": duration,
        "first_frame": first,
        "mid_frame": mid,
    }


def main():
    args = parse_args()
    # Set benchmark-controlled env before importing worker/model code so
    # import-time knobs such as conditional torch.compile see the right values.
    os.environ["ENABLE_COMPILE"] = args.compile
    if args.compile_mode is not None:
        os.environ["LIVEAVATAR_COMPILE_MODE"] = args.compile_mode
    else:
        os.environ.pop("LIVEAVATAR_COMPILE_MODE", None)
    os.environ["LIVEAVATAR_COMPILE_BACKEND"] = args.compile_backend
    if args.compile_dynamic is not None:
        os.environ["LIVEAVATAR_COMPILE_DYNAMIC"] = args.compile_dynamic
    else:
        os.environ.pop("LIVEAVATAR_COMPILE_DYNAMIC", None)
    if args.capture_scalar_outputs is not None:
        os.environ["TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS"] = "1" if args.capture_scalar_outputs == "true" else "0"
    else:
        os.environ.pop("TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS", None)
    if args.simple_teacache_thresh is not None:
        os.environ["LIVEAVATAR_ENABLE_SIMPLE_TEACACHE"] = "true"
        os.environ["LIVEAVATAR_SIMPLE_TEACACHE_THRESH"] = str(args.simple_teacache_thresh)
    else:
        os.environ.pop("LIVEAVATAR_ENABLE_SIMPLE_TEACACHE", None)
        os.environ.pop("LIVEAVATAR_SIMPLE_TEACACHE_THRESH", None)
    if args.simple_teacache_force_calc_steps is not None:
        os.environ["LIVEAVATAR_SIMPLE_TEACACHE_FORCE_CALC_STEPS"] = str(
            args.simple_teacache_force_calc_steps
        )
    else:
        os.environ.pop("LIVEAVATAR_SIMPLE_TEACACHE_FORCE_CALC_STEPS", None)
    if args.simple_teacache_poly is not None:
        os.environ["LIVEAVATAR_SIMPLE_TEACACHE_POLY"] = args.simple_teacache_poly
    else:
        os.environ.pop("LIVEAVATAR_SIMPLE_TEACACHE_POLY", None)
    os.environ["LIVEAVATAR_ENABLE_SIMPLE_ADACACHE"] = "true" if args.simple_adacache else "false"
    if args.simple_adacache_codebook is not None:
        os.environ["LIVEAVATAR_SIMPLE_ADACACHE_CODEBOOK"] = args.simple_adacache_codebook
    else:
        os.environ.pop("LIVEAVATAR_SIMPLE_ADACACHE_CODEBOOK", None)
    if args.num_clip_safety_margin is not None:
        os.environ["LIVEAVATAR_NUM_CLIP_SAFETY_MARGIN"] = str(args.num_clip_safety_margin)
    else:
        os.environ.pop("LIVEAVATAR_NUM_CLIP_SAFETY_MARGIN", None)
    os.environ["LIVEAVATAR_INFER_FRAMES"] = str(args.infer_frames)
    os.environ["LIVEAVATAR_SAMPLE_STEPS"] = str(args.sample_steps)
    os.environ["LIVEAVATAR_SAMPLE_SOLVER"] = args.sample_solver
    os.environ["LIVEAVATAR_GUIDE_SCALE"] = str(args.guide_scale)
    if args.enable_online_decode is not None:
        os.environ["LIVEAVATAR_ENABLE_ONLINE_DECODE"] = (
            "true" if args.enable_online_decode else "false"
        )
    if args.offload_model is not None:
        os.environ["LIVEAVATAR_OFFLOAD_MODEL"] = "true" if args.offload_model else "false"
    if args.offload_kv_cache is not None:
        os.environ["LIVEAVATAR_OFFLOAD_KV_CACHE"] = (
            "true" if args.offload_kv_cache else "false"
        )
    if args.disable_cudnn_attn is not None:
        os.environ["LIVEAVATAR_DISABLE_CUDNN_ATTN"] = (
            "true" if args.disable_cudnn_attn else "false"
        )
    if args.disable_flash_attn is not None:
        os.environ["LIVEAVATAR_DISABLE_FLASH_ATTN"] = (
            "true" if args.disable_flash_attn else "false"
        )
    os.environ["LIVEAVATAR_CROSS_ATTN_CHUNK_SIZE"] = str(args.cross_attn_chunk_size)
    os.environ["LIVEAVATAR_ROPE_CHUNK_SIZE"] = str(args.rope_chunk_size)
    os.environ["LIVEAVATAR_ATTN_OUT_CHUNK_SIZE"] = str(args.attn_out_chunk_size)
    os.environ["LIVEAVATAR_QKV_CHUNK_SIZE"] = str(args.qkv_chunk_size)
    os.environ["LIVEAVATAR_USE_LIGHTVAE"] = "true" if args.use_lightvae else "false"
    os.environ["LIVEAVATAR_USE_TAE"] = "true" if args.use_tae else "false"
    if args.vae_path:
        os.environ["LIVEAVATAR_VAE_PATH"] = str(Path(args.vae_path).resolve())
    else:
        os.environ.pop("LIVEAVATAR_VAE_PATH", None)
    if args.vae_pruning_rate is not None:
        os.environ["LIVEAVATAR_VAE_PRUNING_RATE"] = str(args.vae_pruning_rate)
    else:
        os.environ.pop("LIVEAVATAR_VAE_PRUNING_RATE", None)
    if args.tae_parallel is not None:
        os.environ["LIVEAVATAR_TAE_PARALLEL"] = args.tae_parallel
    else:
        os.environ.pop("LIVEAVATAR_TAE_PARALLEL", None)
    os.environ["LIVEAVATAR_ENABLE_FULL_ONLINE_POSTPROCESS"] = (
        "true" if args.full_online_postprocess else "false"
    )
    if args.relative_dist is not None:
        os.environ["LIVEAVATAR_RELATIVE_DIST"] = str(args.relative_dist)
    else:
        os.environ.pop("LIVEAVATAR_RELATIVE_DIST", None)

    mod = load_worker_module()

    mod.load_config_file(REPO_ROOT / "worker_config.json")
    mod.load_env_file(REPO_ROOT / ".env")

    image_path = Path(args.image).resolve()
    audio_path = Path(args.audio).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_output = output_dir / "rendered_raw.mp4"
    final_output = output_dir / "rendered.mp4"
    live_events_path = output_dir / "live_events.json"
    heartbeat_path = output_dir / "heartbeat.json"

    audio_duration = mod.probe_audio_duration(audio_path)
    num_clip = mod.compute_num_clip(audio_duration, infer_frames=args.infer_frames, fps=25)

    runner_acquire_started = time.perf_counter()
    runner, cold_start, acquire_duration = mod.get_runner()
    runner_jobs_before = runner.jobs_processed
    prompt = args.prompt or mod.DEFAULT_PROMPT
    size_h, size_w = map(int, args.size.split("*"))
    output_size = "720*1280" if size_w >= size_h else "1280*720"

    def run_once(raw_target: Path, final_target: Path, enable_profiler: bool = False):
        profiler = None
        profiler_cpu_table_path = None
        profiler_cuda_table_path = None
        if enable_profiler:
            import torch
            profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
                profile_memory=True,
                with_stack=False,
            )
            profiler.__enter__()

        render_started_at = time.perf_counter()
        event_trace = []
        clip_generation_started_at = None
        clip_generation_finished_at = None
        postprocess_started_at = None
        postprocess_finished_at = None
        postprocess_decode_started_at = None
        postprocess_encode_started_at = None
        postprocess_decode_s = 0.0
        postprocess_encode_s = 0.0

        def flush_live_progress(extra: dict | None = None):
            payload = {
                "updated_at": time.time(),
                "render_elapsed_s": time.perf_counter() - render_started_at,
                "events_count": len(event_trace),
                "last_event": event_trace[-1] if event_trace else None,
                "extra": extra or {},
            }
            write_json(heartbeat_path, payload)
            write_json(live_events_path, event_trace)

        def on_render_progress(stage: str, clip_index: int, clip_total: int):
            nonlocal clip_generation_started_at
            nonlocal clip_generation_finished_at
            nonlocal postprocess_started_at
            nonlocal postprocess_finished_at
            nonlocal postprocess_decode_started_at
            nonlocal postprocess_encode_started_at
            nonlocal postprocess_decode_s
            nonlocal postprocess_encode_s
            now = time.perf_counter()
            event_trace.append(
                {
                    "t_rel_s": now - render_started_at,
                    "stage": stage,
                    "clip_index": clip_index,
                    "clip_total": clip_total,
                }
            )
            if stage == "clip_start" and clip_generation_started_at is None:
                clip_generation_started_at = now
            elif stage == "clip_complete":
                clip_generation_finished_at = now
            elif stage == "postprocess_start":
                postprocess_started_at = now
            elif stage == "postprocess_decode_start":
                postprocess_decode_started_at = now
            elif stage == "postprocess_decode_complete":
                if postprocess_decode_started_at is not None:
                    postprocess_decode_s += now - postprocess_decode_started_at
                    postprocess_decode_started_at = None
            elif stage == "postprocess_encode_start":
                postprocess_encode_started_at = now
            elif stage == "postprocess_encode_complete":
                if postprocess_encode_started_at is not None:
                    postprocess_encode_s += now - postprocess_encode_started_at
                    postprocess_encode_started_at = None
            elif stage == "postprocess_clip_complete":
                postprocess_finished_at = now
            elif stage == "postprocess_complete":
                postprocess_finished_at = now
            flush_live_progress(
                {
                    "kind": "render_progress",
                    "stage": stage,
                    "clip_index": clip_index,
                    "clip_total": clip_total,
                }
            )

        last_upscale = {"progress": 0, "processed": 0.0, "total": 0.0}

        def on_upscale_progress(progress: int, processed_seconds: float, total_seconds: float):
            last_upscale["progress"] = progress
            last_upscale["processed"] = processed_seconds
            last_upscale["total"] = total_seconds
            event_trace.append(
                {
                    "t_rel_s": time.perf_counter() - render_started_at,
                    "stage": "upscale_progress",
                    "progress": progress,
                    "processed_seconds": processed_seconds,
                    "total_seconds": total_seconds,
                }
            )
            flush_live_progress(
                {
                    "kind": "upscale_progress",
                    "progress": progress,
                    "processed_seconds": processed_seconds,
                    "total_seconds": total_seconds,
                }
            )

        try:
            flush_live_progress({"kind": "run_started"})
            if args.direct_final_encode:
                video = runner.render(
                    image_path=image_path,
                    audio_path=audio_path,
                    output_path=raw_target,
                    size=args.size,
                    prompt=prompt,
                    num_clip=num_clip,
                    base_seed=args.base_seed,
                    progress_callback=on_render_progress,
                    return_video_tensor=True,
                )
                render_finished_at = time.perf_counter()
                runner.jobs_processed += 1
                upscale_started_at = time.perf_counter()
                mod.save_video_final_nvenc(
                    tensor=video[None],
                    save_file=final_target,
                    audio_path=audio_path,
                    fps=25,
                    output_size=output_size,
                    normalize=True,
                    value_range=(-1, 1),
                )
                upscale_finished_at = time.perf_counter()
                on_upscale_progress(100, audio_duration, audio_duration)
                del video
                torch = __import__("torch")
                torch.cuda.empty_cache()
            else:
                runner.render(
                    image_path=image_path,
                    audio_path=audio_path,
                    output_path=raw_target,
                    size=args.size,
                    prompt=prompt,
                    num_clip=num_clip,
                    base_seed=args.base_seed,
                    progress_callback=on_render_progress,
                )
                render_finished_at = time.perf_counter()
                runner.jobs_processed += 1

                upscale_started_at = time.perf_counter()
                normalize_kwargs = {
                    "fps": 25,
                    "output_size": output_size,
                    "progress_callback": on_upscale_progress,
                }
                normalize_sig = inspect.signature(mod.normalize_video)
                if "audio_path" in normalize_sig.parameters:
                    normalize_kwargs["audio_path"] = audio_path
                mod.normalize_video(
                    raw_target,
                    final_target,
                    **normalize_kwargs,
                )
                upscale_finished_at = time.perf_counter()
        finally:
            if profiler is not None:
                profiler.__exit__(None, None, None)
                profiler_cpu_table_path = output_dir / f"{final_target.stem}_profiler_cpu.txt"
                profiler_cuda_table_path = output_dir / f"{final_target.stem}_profiler_cuda.txt"
                profiler_cpu_table_path.write_text(
                    profiler.key_averages().table(sort_by="self_cpu_time_total", row_limit=80) + "\n"
                )
                profiler_cuda_table_path.write_text(
                    profiler.key_averages().table(sort_by="self_cuda_time_total", row_limit=80) + "\n"
                )
        clip_durations = []
        postprocess_clip_durations = []
        clip_started_at = {}
        postprocess_clip_started_at = {}
        for event in event_trace:
            stage = event["stage"]
            clip_index = event.get("clip_index")
            key = (clip_index, event.get("clip_total"))
            if stage == "clip_start":
                clip_started_at[key] = event["t_rel_s"]
            elif stage == "clip_complete" and key in clip_started_at:
                clip_durations.append(
                    {
                        "clip_index": clip_index,
                        "clip_total": event.get("clip_total"),
                        "duration_s": event["t_rel_s"] - clip_started_at.pop(key),
                    }
                )
            elif stage == "postprocess_decode_start":
                postprocess_clip_started_at[key] = event["t_rel_s"]
            elif stage == "postprocess_clip_complete" and key in postprocess_clip_started_at:
                postprocess_clip_durations.append(
                    {
                        "clip_index": clip_index,
                        "clip_total": event.get("clip_total"),
                        "duration_s": event["t_rel_s"] - postprocess_clip_started_at.pop(key),
                    }
                )

        return {
            "render_s": render_finished_at - render_started_at,
            "clip_generation_s": (
                clip_generation_finished_at - clip_generation_started_at
                if clip_generation_started_at is not None and clip_generation_finished_at is not None
                else None
            ),
            "postprocess_s": (
                postprocess_finished_at - postprocess_started_at
                if postprocess_started_at is not None and postprocess_finished_at is not None
                else None
            ),
            "postprocess_decode_s": postprocess_decode_s if postprocess_decode_s > 0 else None,
            "postprocess_encode_s": postprocess_encode_s if postprocess_encode_s > 0 else None,
            "upscale_s": upscale_finished_at - upscale_started_at,
            "generation_total_s": upscale_finished_at - render_started_at,
            "generation_sec_per_audio_sec": (upscale_finished_at - render_started_at) / audio_duration,
            "upscale_progress": last_upscale,
            "event_trace": event_trace,
            "clip_durations": clip_durations,
            "postprocess_clip_durations": postprocess_clip_durations,
            "profiler_cpu_table": str(profiler_cpu_table_path) if profiler_cpu_table_path else None,
            "profiler_cuda_table": str(profiler_cuda_table_path) if profiler_cuda_table_path else None,
        }

    def write_json(path: Path, payload):
        path.write_text(json.dumps(payload, indent=2) + "\n")

    warmup_runs = []
    for warm_idx in range(args.warm_runs):
        warm_raw = output_dir / f"warmup_{warm_idx + 1}_raw.mp4"
        warm_final = output_dir / f"warmup_{warm_idx + 1}.mp4"
        warm_metrics = run_once(warm_raw, warm_final)
        warmup_runs.append(
            {
                "run": warm_idx + 1,
                "render_s": warm_metrics["render_s"],
                "clip_generation_s": warm_metrics["clip_generation_s"],
                "postprocess_s": warm_metrics["postprocess_s"],
                "postprocess_decode_s": warm_metrics["postprocess_decode_s"],
                "postprocess_encode_s": warm_metrics["postprocess_encode_s"],
                "upscale_s": warm_metrics["upscale_s"],
                "generation_total_s": warm_metrics["generation_total_s"],
                "generation_sec_per_audio_sec": warm_metrics["generation_sec_per_audio_sec"],
                "clip_durations": warm_metrics["clip_durations"],
                "postprocess_clip_durations": warm_metrics["postprocess_clip_durations"],
            }
        )
        write_json(
            output_dir / f"warmup_{warm_idx + 1}_metrics.json",
            warmup_runs[-1],
        )

    write_json(
        output_dir / "status.json",
        {
            "stage": "primary_run_started",
            "warm_runs_completed": len(warmup_runs),
            "image": str(image_path),
            "audio": str(audio_path),
            "size": args.size,
            "infer_frames": args.infer_frames,
            "compile": args.compile,
        },
    )
    try:
        primary_metrics = run_once(raw_output, final_output, enable_profiler=args.torch_profiler)
        finished_at = time.perf_counter()

        result = {
            "benchmark_policy": {
                "primary_metric": "generation_total_s",
                "primary_rate_metric": "generation_sec_per_audio_sec",
                "compare_warm_runs_not_cold_start": True,
                "notes": "Cold-start timings are diagnostic only. Compare generation timings from an already-loaded resident pipeline.",
            },
            "image": str(image_path),
            "audio": str(audio_path),
            "audio_duration": audio_duration,
            "size": args.size,
            "infer_frames": args.infer_frames,
            "sample_steps": args.sample_steps,
            "sample_solver": args.sample_solver,
            "guide_scale": args.guide_scale,
            "num_clip": num_clip,
            "enable_online_decode": args.enable_online_decode,
            "offload_model": args.offload_model,
            "offload_kv_cache": args.offload_kv_cache,
            "disable_cudnn_attn": args.disable_cudnn_attn,
            "disable_flash_attn": args.disable_flash_attn,
            "chunk_sizes": {
                "cross_attn": args.cross_attn_chunk_size,
                "rope": args.rope_chunk_size,
                "attn_out": args.attn_out_chunk_size,
                "qkv": args.qkv_chunk_size,
            },
            "compile": args.compile,
            "compile_mode": args.compile_mode,
            "compile_backend": args.compile_backend,
            "compile_dynamic": args.compile_dynamic,
            "capture_scalar_outputs": args.capture_scalar_outputs,
            "simple_teacache_thresh": args.simple_teacache_thresh,
            "simple_teacache_force_calc_steps": args.simple_teacache_force_calc_steps,
            "simple_teacache_poly": args.simple_teacache_poly,
            "simple_adacache": args.simple_adacache,
            "simple_adacache_codebook": args.simple_adacache_codebook,
            "num_clip_safety_margin": args.num_clip_safety_margin,
            "use_lightvae": args.use_lightvae,
            "use_tae": args.use_tae,
            "vae_path": str(Path(args.vae_path).resolve()) if args.vae_path else None,
            "vae_pruning_rate": args.vae_pruning_rate,
            "tae_parallel": args.tae_parallel,
            "full_online_postprocess": args.full_online_postprocess,
            "relative_dist": args.relative_dist,
            "warm_runs": args.warm_runs,
            "direct_final_encode": args.direct_final_encode,
            "warmup_runs": warmup_runs,
            "cold_start": cold_start,
            "runner_jobs_before": runner_jobs_before,
            "runner_acquire_s": acquire_duration,
            "pipeline_init_s": getattr(runner, "init_duration", 0.0) if cold_start else 0.0,
            "render_s": primary_metrics["render_s"],
            "clip_generation_s": primary_metrics["clip_generation_s"],
            "postprocess_s": primary_metrics["postprocess_s"],
            "postprocess_decode_s": primary_metrics["postprocess_decode_s"],
            "postprocess_encode_s": primary_metrics["postprocess_encode_s"],
            "upscale_s": primary_metrics["upscale_s"],
            "generation_total_s": primary_metrics["generation_total_s"],
            "generation_sec_per_audio_sec": primary_metrics["generation_sec_per_audio_sec"],
            "total_s_including_runner_acquire": finished_at - runner_acquire_started,
            "sec_per_audio_sec_including_runner_acquire": (finished_at - runner_acquire_started) / audio_duration,
            "raw_output": str(raw_output),
            "final_output": str(final_output),
            "raw_visual": probe_visual_stats(raw_output) if raw_output.exists() else None,
            "final_visual": probe_visual_stats(final_output) if final_output.exists() else None,
            "upscale_progress": primary_metrics["upscale_progress"],
            "event_trace": primary_metrics["event_trace"],
            "clip_durations": primary_metrics["clip_durations"],
            "postprocess_clip_durations": primary_metrics["postprocess_clip_durations"],
            "profiler_cpu_table": primary_metrics["profiler_cpu_table"],
            "profiler_cuda_table": primary_metrics["profiler_cuda_table"],
        }
        result_file = Path(args.result_file).resolve() if args.result_file else (output_dir / "benchmark.json")
        write_json(output_dir / "primary_metrics.json", primary_metrics)
        write_json(output_dir / "status.json", {"stage": "completed"})
        heartbeat_path.unlink(missing_ok=True)
        write_json(result_file, result)
        print(json.dumps(result, indent=2))
    except Exception as exc:
        failure = {
            "stage": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "image": str(image_path),
            "audio": str(audio_path),
            "size": args.size,
            "infer_frames": args.infer_frames,
            "compile": args.compile,
            "chunk_sizes": {
                "cross_attn": args.cross_attn_chunk_size,
                "rope": args.rope_chunk_size,
                "attn_out": args.attn_out_chunk_size,
                "qkv": args.qkv_chunk_size,
            },
            "direct_final_encode": args.direct_final_encode,
        }
        write_json(output_dir / "status.json", failure)
        write_json(output_dir / "failure.json", failure)
        flush_live_failure = {
            "updated_at": time.time(),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        write_json(heartbeat_path, flush_live_failure)
        raise


if __name__ == "__main__":
    main()
