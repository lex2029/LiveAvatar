#!/usr/bin/env python3
import json
import math
import os
import random
import shutil
import signal
import socket
import string
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from PIL import Image
import torch


REPO_ROOT = Path(__file__).resolve().parent
ENV_PATH = REPO_ROOT / ".env"
CONFIG_PATH = REPO_ROOT / "worker_config.json"
VENV_BIN = REPO_ROOT / ".venv" / "bin"
TORCHRUN = VENV_BIN / "torchrun"
PYTHON_BIN = VENV_BIN / "python"

DEFAULT_PROMPT = (
    "A natural talking-head video of the person in the reference image, "
    "looking at the camera, speaking clearly with expressive facial motion, "
    "stable identity, high detail, clean background, realistic motion, studio lighting."
)

STOP_REQUESTED = False
RUNNER: Optional["ResidentLiveAvatarRunner"] = None


class JobStoppedByServer(RuntimeError):
    pass


def log(message: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    print(f"[{ts}] {message}", flush=True)


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def load_config_file(path: Path) -> None:
    if not path.exists():
        return
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise RuntimeError(f"Config file must contain a JSON object: {path}")
    for key, value in data.items():
        if value is None:
            continue
        os.environ.setdefault(str(key), str(value))


def getenv_required(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def worker_headers() -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {getenv_required('WORKER_API_KEY')}",
        "Content-Type": "application/json",
    }


def worker_api_url() -> str:
    return f"{getenv_required('SUPABASE_URL').rstrip('/')}/functions/v1/worker-api"


def public_video_url(path: str) -> str:
    base = getenv_required("SUPABASE_URL").rstrip("/")
    return f"{base}/storage/v1/object/public/generated-assets/{path}"


def call_worker_api(body: Dict[str, Any]) -> Dict[str, Any]:
    response = requests.post(worker_api_url(), json=body, headers=worker_headers(), timeout=60)
    response.raise_for_status()
    data = response.json()
    if isinstance(data, dict) and data.get("error"):
        raise RuntimeError(f"Worker API error: {data['error']}")
    return data


def report_progress(job_id: str, progress: int) -> None:
    clamped = max(0, min(100, int(progress)))
    call_worker_api({"action": "progress", "job_id": job_id, "progress": clamped})


def download_file(url: str, destination: Path) -> None:
    with requests.get(url, stream=True, timeout=300) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)


def probe_audio_duration(audio_path: Path) -> float:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def probe_video_duration(video_path: Path) -> float:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def format_seconds(seconds: float) -> str:
    return f"{seconds:.1f}s"


def detect_orientation(image_path: Path) -> str:
    with Image.open(image_path) as image:
        width, height = image.size
    return "landscape" if width >= height else "portrait"


def orientation_to_render_size(orientation: str) -> str:
    if orientation == "landscape":
        return os.getenv("LIVEAVATAR_RENDER_LANDSCAPE_SIZE", "720*1280")
    return os.getenv("LIVEAVATAR_RENDER_PORTRAIT_SIZE", "832*480")


def orientation_to_free_render_size(orientation: str) -> str:
    if orientation == "landscape":
        return os.getenv("LIVEAVATAR_RENDER_LANDSCAPE_SIZE_FREE", "256*384")
    return os.getenv("LIVEAVATAR_RENDER_PORTRAIT_SIZE_FREE", "384*256")


def orientation_to_output_size(orientation: str) -> str:
    if orientation == "landscape":
        return os.getenv("LIVEAVATAR_OUTPUT_LANDSCAPE_SIZE", "720*1280")
    return os.getenv("LIVEAVATAR_OUTPUT_PORTRAIT_SIZE", "1280*720")


def choose_prompt(payload: Dict[str, Any]) -> str:
    for key in ("prompt", "video_prompt", "render_prompt", "visual_prompt", "description"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return os.getenv("LIVEAVATAR_DEFAULT_PROMPT", DEFAULT_PROMPT)


def plan_key_for_job(job: Dict[str, Any], payload: Dict[str, Any]) -> str:
    for value in (
        job.get("plan_key"),
        payload.get("plan_key"),
        payload.get("subscription_plan"),
        payload.get("plan"),
        payload.get("tariff"),
        payload.get("tier"),
    ):
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    return "pro"


def compute_num_clip(audio_duration: float, infer_frames: int, fps: int) -> int:
    clip_seconds = infer_frames / float(fps)
    safety_margin = 2
    raw = math.ceil(audio_duration / clip_seconds) + safety_margin
    minimum = int(os.getenv("LIVEAVATAR_MIN_NUM_CLIP", "1"))
    maximum = int(os.getenv("LIVEAVATAR_MAX_NUM_CLIP", "120"))
    return max(minimum, min(raw, maximum))


def random_suffix(length: int = 8) -> str:
    alphabet = string.ascii_lowercase + string.digits
    return "".join(random.choice(alphabet) for _ in range(length))


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def liveavatar_env(enable_compile: bool) -> Dict[str, str]:
    env = os.environ.copy()
    cuda_home = env.get("CUDA_HOME", "/usr/local/cuda")
    env["CUDA_HOME"] = cuda_home
    env["PATH"] = f"{cuda_home}/bin:{env.get('PATH', '')}"
    env["LD_LIBRARY_PATH"] = f"{cuda_home}/lib64:{env.get('LD_LIBRARY_PATH', '')}"
    env["ENABLE_COMPILE"] = "true" if enable_compile else "false"
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0")
    env["PYTHONUNBUFFERED"] = "1"
    return env


class ResidentLiveAvatarRunner:
    def __init__(self) -> None:
        self.init_started_at = time.perf_counter()
        self._setup_env()
        self._setup_dist()
        self._load_pipeline()
        self.init_finished_at = time.perf_counter()
        self.init_duration = self.init_finished_at - self.init_started_at
        self.jobs_processed = 0

    def _setup_env(self) -> None:
        cuda_home = os.getenv("CUDA_HOME", "/usr/local/cuda")
        os.environ["CUDA_HOME"] = cuda_home
        os.environ["PATH"] = f"{cuda_home}/bin:{os.getenv('PATH', '')}"
        os.environ["LD_LIBRARY_PATH"] = f"{cuda_home}/lib64:{os.getenv('LD_LIBRARY_PATH', '')}"
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(find_free_port()))
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("ENABLE_COMPILE", "true")
        os.environ.setdefault("PYTHONUNBUFFERED", "1")
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    def _setup_dist(self) -> None:
        import torch.distributed as dist

        self.dist = dist
        if not self.dist.is_initialized():
            torch.cuda.set_device(0)
            self.dist.init_process_group(
                backend="nccl",
                init_method="env://",
                rank=0,
                world_size=1,
            )

    def _load_pipeline(self) -> None:
        from liveavatar.models.wan.causal_s2v_pipeline import WanS2V
        from liveavatar.models.wan.wan_2_2.configs import MAX_AREA_CONFIGS, WAN_CONFIGS
        from liveavatar.utils.args_config import parse_args_for_training_config as training_config_parser
        from liveavatar.utils.fp8_linear import replace_linear_with_scaled_fp8

        self.MAX_AREA_CONFIGS = MAX_AREA_CONFIGS
        self.cfg = WAN_CONFIGS["s2v-14B"]
        self.training_settings = training_config_parser(str(REPO_ROOT / "liveavatar/configs/s2v_causal_sft.yaml"))
        self.MAX_AREA_CONFIGS.setdefault("256*384", 256 * 384)

        log("Loading LiveAvatar pipeline into resident worker memory")
        offload_kv_cache = os.getenv("LIVEAVATAR_OFFLOAD_KV_CACHE", "false").lower() == "true"

        self.pipeline = WanS2V(
            config=self.cfg,
            checkpoint_dir="ckpt/Wan2.2-S2V-14B/",
            device_id=0,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_sp=False,
            sp_size=1,
            t5_cpu=False,
            convert_model_dtype=True,
            single_gpu=True,
            offload_kv_cache=offload_kv_cache,
        )

        self.pipeline.noise_model = self.pipeline.add_lora_to_model(
            self.pipeline.noise_model,
            lora_rank=self.training_settings["lora_rank"],
            lora_alpha=self.training_settings["lora_alpha"],
            lora_target_modules=self.training_settings["lora_target_modules"],
            init_lora_weights=self.training_settings["init_lora_weights"],
            pretrained_lora_path="ckpt/LiveAvatar/liveavatar.safetensors",
            load_lora_weight_only=False,
        )

        if hasattr(torch, "_scaled_mm"):
            replace_linear_with_scaled_fp8(
                self.pipeline.noise_model,
                ignore_keys=[
                    "text_embedding",
                    "time_embedding",
                    "time_projection",
                    "head.head",
                    "casual_audio_encoder.encoder.final_linear",
                ],
            )
        log("Resident LiveAvatar pipeline loaded")

    def render(
        self,
        *,
        image_path: Path,
        audio_path: Path,
        output_path: Path,
        size: str,
        prompt: str,
        num_clip: int,
        base_seed: int,
        progress_callback=None,
    ) -> None:
        from liveavatar.models.wan.wan_2_2.utils.utils import merge_video_audio, save_video

        infer_frames = int(os.getenv("LIVEAVATAR_INFER_FRAMES", "48"))
        sample_steps = int(os.getenv("LIVEAVATAR_SAMPLE_STEPS", "4"))
        guide_scale = float(os.getenv("LIVEAVATAR_GUIDE_SCALE", "0"))
        sample_solver = os.getenv("LIVEAVATAR_SAMPLE_SOLVER", "euler")
        enable_online_decode = os.getenv("LIVEAVATAR_ENABLE_ONLINE_DECODE", "false").lower() == "true"
        offload_model = os.getenv("LIVEAVATAR_OFFLOAD_MODEL", "false").lower() == "true"

        output_path.parent.mkdir(parents=True, exist_ok=True)
        log(f"Starting resident render: size={size}, num_clip={num_clip}, output={output_path}")

        video, _dataset_info = self.pipeline.generate(
            input_prompt=prompt,
            ref_image_path=str(image_path),
            audio_path=str(audio_path),
            enable_tts=False,
            num_repeat=num_clip,
            pose_video=None,
            generate_size=size,
            max_area=self.MAX_AREA_CONFIGS[size],
            infer_frames=infer_frames,
            shift=3,
            sample_solver=sample_solver,
            sampling_steps=sample_steps,
            guide_scale=guide_scale,
            seed=base_seed,
            offload_model=offload_model,
            init_first_frame=False,
            use_dataset=False,
            dataset_sample_idx=0,
            drop_motion_noisy=False,
            num_gpus_dit=1,
            enable_vae_parallel=False,
            input_video_for_sam2=None,
            enable_online_decode=enable_online_decode,
            progress_callback=progress_callback,
        )

        save_video(
            tensor=video[None],
            save_file=str(output_path),
            fps=self.cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
        )
        merge_video_audio(video_path=str(output_path), audio_path=str(audio_path))
        del video
        torch.cuda.empty_cache()
        if not output_path.exists():
            raise FileNotFoundError(f"Resident renderer finished without output file: {output_path}")

    def release_gpu_for_ffmpeg(self) -> None:
        return


def get_runner() -> ResidentLiveAvatarRunner:
    global RUNNER
    cold_start = False
    acquire_started_at = time.perf_counter()
    if RUNNER is None:
        RUNNER = ResidentLiveAvatarRunner()
        cold_start = True
    acquire_duration = time.perf_counter() - acquire_started_at
    return RUNNER, cold_start, acquire_duration


def normalize_video(
    input_path: Path,
    output_path: Path,
    fps: int = 25,
    output_size: Optional[str] = None,
    progress_callback=None,
) -> None:
    gpu_filters = ["hwupload_cuda"]
    if output_size:
        out_h, out_w = map(int, output_size.split("*"))
        gpu_filters.append(f"scale_cuda={out_w}:{out_h}")
    gpu_command = [
        "ffmpeg",
        "-y",
        "-nostats",
        "-i",
        str(input_path),
        "-vf",
        ",".join(gpu_filters),
        "-c:v",
        "h264_nvenc",
        "-preset",
        "p4",
        "-r",
        str(fps),
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        "-progress",
        "pipe:1",
        str(output_path),
    ]
    try:
        total_duration = probe_video_duration(input_path)
        process = subprocess.Popen(
            gpu_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        stderr_lines: List[str] = []
        last_reported_progress = -1
        if process.stdout is not None:
            for line in process.stdout:
                stripped = line.strip()
                if not stripped or "=" not in stripped:
                    continue
                key, value = stripped.split("=", 1)
                if key == "out_time_ms" and total_duration > 0:
                    out_time_seconds = int(value) / 1_000_000.0
                    progress = max(81, min(99, 80 + int((20 * out_time_seconds) / total_duration)))
                    if progress_callback is not None and progress != last_reported_progress:
                        progress_callback(progress, out_time_seconds, total_duration)
                        last_reported_progress = progress
        if process.stderr is not None:
            stderr_lines = process.stderr.read().splitlines()
        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(
                returncode=return_code,
                cmd=gpu_command,
                stderr="\n".join(stderr_lines),
            )
        if progress_callback is not None:
            progress_callback(100, total_duration, total_duration)
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        raise RuntimeError(
            f"ffmpeg gpu normalize failed (exit {exc.returncode}): {stderr}"
        ) from exc
    if not output_path.exists():
        raise FileNotFoundError(f"ffmpeg did not create normalized output: {output_path}")


def upload_video(signed_url: str, video_path: Path) -> None:
    with video_path.open("rb") as handle:
        response = requests.put(
            signed_url,
            data=handle,
            headers={"Content-Type": "video/mp4"},
            timeout=1800,
        )
    response.raise_for_status()


def process_job(job_id: str) -> None:
    log(f"Claiming job {job_id}")
    claim = call_worker_api({"action": "claim", "job_id": job_id})
    if not claim.get("claimed"):
        log(f"Job {job_id} was not claimed: {claim.get('reason', 'unknown reason')}")
        return

    job = claim.get("job", {})
    payload = job.get("payload", {}) or {}
    assets = claim["assets"]
    upload = claim["upload"]
    job_started_at = time.perf_counter()

    with tempfile.TemporaryDirectory(prefix=f"smartblog_{job_id}_", dir=str(REPO_ROOT / "worker_runs")) as temp_dir:
        temp_root = Path(temp_dir)
        image_path = temp_root / "avatar.png"
        audio_path = temp_root / "audio.mp3"
        raw_video_path = temp_root / "rendered_raw.mp4"
        final_video_path = temp_root / "rendered.mp4"

        try:
            log(f"Downloading assets for {job_id}")
            download_file(assets["avatar_url"], image_path)
            download_file(assets["audio_url"], audio_path)

            orientation = payload.get("orientation")
            if orientation not in {"portrait", "landscape"}:
                orientation = detect_orientation(image_path)

            plan_key = plan_key_for_job(job, payload)
            is_free_plan = plan_key == "free"
            render_size = orientation_to_free_render_size(orientation) if is_free_plan else orientation_to_render_size(orientation)
            output_size = orientation_to_output_size(orientation)
            prompt = choose_prompt(payload)
            infer_frames = int(os.getenv("LIVEAVATAR_INFER_FRAMES", "48"))
            sample_fps = 25
            audio_duration = probe_audio_duration(audio_path)
            num_clip = compute_num_clip(audio_duration, infer_frames=infer_frames, fps=sample_fps)
            base_seed = int(payload.get("seed") or os.getenv("LIVEAVATAR_BASE_SEED", "420"))

            log(
                f"Processing job {job_id}: orientation={orientation}, plan_key={plan_key}, "
                f"render_size={render_size}, output_size={output_size}, "
                f"audio_duration={audio_duration:.2f}s, num_clip={num_clip}"
            )

            runner, runner_cold_start, runner_acquire_duration = get_runner()
            runner_jobs_before = runner.jobs_processed
            last_reported_progress = -1
            render_started_at = time.perf_counter()
            clip_generation_started_at: Optional[float] = None
            clip_generation_finished_at: Optional[float] = None
            postprocess_started_at: Optional[float] = None
            postprocess_finished_at: Optional[float] = None

            def on_render_progress(stage: str, clip_index: int, clip_total: int) -> None:
                nonlocal last_reported_progress, clip_generation_started_at
                nonlocal clip_generation_finished_at, postprocess_started_at, postprocess_finished_at
                progress = None

                if stage == "clip_start":
                    if clip_generation_started_at is None:
                        clip_generation_started_at = time.perf_counter()
                    if clip_total <= 0:
                        return
                    progress = 1 if clip_index == 1 else max(
                        1, min(40, (40 * (clip_index - 1)) // clip_total))
                elif stage == "clip_complete":
                    clip_generation_finished_at = time.perf_counter()
                    if clip_total <= 0:
                        return
                    progress = max(1, min(40, (40 * clip_index) // clip_total))
                elif stage == "postprocess_start":
                    postprocess_started_at = time.perf_counter()
                    progress = 41
                elif stage == "postprocess_clip_complete":
                    postprocess_finished_at = time.perf_counter()
                    if clip_total <= 0:
                        progress = 80
                    else:
                        progress = max(41, min(80, 40 + (40 * clip_index) // clip_total))
                elif stage == "postprocess_complete":
                    postprocess_finished_at = time.perf_counter()
                    progress = 80
                else:
                    return

                if progress == last_reported_progress:
                    return

                try:
                    ack = call_worker_api({"action": "progress", "job_id": job_id, "progress": max(0, min(100, int(progress)))})
                    if ack.get("stop"):
                        raise JobStoppedByServer(
                            f"Server stopped job: {ack.get('status', 'unknown')} / {ack.get('reason', 'no reason')}"
                        )
                    last_reported_progress = progress
                    log(f"Job {job_id} progress: {progress}% ({stage} {clip_index}/{clip_total})")
                except JobStoppedByServer:
                    raise
                except Exception as progress_exc:
                    log(
                        f"Job {job_id} progress update failed at {progress}% "
                        f"({stage} {clip_index}/{clip_total}): {progress_exc}"
                    )

            def on_upscale_progress(progress: int, processed_seconds: float, total_seconds: float) -> None:
                nonlocal last_reported_progress
                if progress == last_reported_progress:
                    return
                try:
                    ack = call_worker_api({"action": "progress", "job_id": job_id, "progress": max(0, min(100, int(progress)))})
                    if ack.get("stop"):
                        raise JobStoppedByServer(
                            f"Server stopped job: {ack.get('status', 'unknown')} / {ack.get('reason', 'no reason')}"
                        )
                    last_reported_progress = progress
                    if total_seconds > 0:
                        log(
                            f"Job {job_id} progress: {progress}% "
                            f"(upscale {processed_seconds:.1f}/{total_seconds:.1f}s)"
                        )
                    else:
                        log(f"Job {job_id} progress: {progress}% (upscale)")
                except JobStoppedByServer:
                    raise
                except Exception as progress_exc:
                    log(
                        f"Job {job_id} progress update failed at {progress}% "
                        f"(upscale {processed_seconds:.1f}/{total_seconds:.1f}s): {progress_exc}"
                    )

            log(
                f"Job {job_id} render started "
                f"(render_size={render_size}, output_size={output_size}, infer_frames={infer_frames}, "
                f"num_clip={num_clip}, plan_key={plan_key}, runner_cold_start={runner_cold_start}, "
                f"runner_acquire={format_seconds(runner_acquire_duration)}, runner_jobs_before={runner_jobs_before})"
            )
            runner.render(
                image_path=image_path,
                audio_path=audio_path,
                output_path=raw_video_path,
                size=render_size,
                prompt=prompt,
                num_clip=num_clip,
                base_seed=base_seed,
                progress_callback=on_render_progress,
            )
            render_finished_at = time.perf_counter()
            runner.jobs_processed += 1
            clip_generation_duration = (
                (clip_generation_finished_at - clip_generation_started_at)
                if clip_generation_started_at is not None and clip_generation_finished_at is not None
                else None
            )
            postprocess_duration = (
                (postprocess_finished_at - postprocess_started_at)
                if postprocess_started_at is not None and postprocess_finished_at is not None
                else None
            )
            log(
                f"Job {job_id} render finished in {format_seconds(render_finished_at - render_started_at)} "
                f"for {audio_duration:.1f}s of audio"
            )
            if runner_cold_start:
                log(
                    f"Job {job_id} runner cold start: pipeline_init={format_seconds(runner.init_duration)}, "
                    f"runner_acquire={format_seconds(runner_acquire_duration)}"
                )
            if clip_generation_duration is not None:
                log(
                    f"Job {job_id} clip generation finished in {format_seconds(clip_generation_duration)} "
                    f"for {num_clip} clips"
                )
            if postprocess_duration is not None:
                log(f"Job {job_id} postprocess finished in {format_seconds(postprocess_duration)}")
            runner.release_gpu_for_ffmpeg()
            upscale_started_at = time.perf_counter()
            log(f"Job {job_id} upscale started")
            normalize_video(
                raw_video_path,
                final_video_path,
                fps=25,
                output_size=output_size,
                progress_callback=on_upscale_progress,
            )
            upscale_finished_at = time.perf_counter()
            raw_duration = probe_video_duration(raw_video_path)
            log(
                f"Job {job_id} upscale finished in {format_seconds(upscale_finished_at - upscale_started_at)} "
                f"for {raw_duration:.1f}s of video"
            )
            upload_started_at = time.perf_counter()
            log(f"Job {job_id} upload started")
            upload_video(upload["signed_url"], final_video_path)
            video_url = public_video_url(upload["path"])
            call_worker_api({"action": "complete", "job_id": job_id, "video_url": video_url})
            upload_finished_at = time.perf_counter()
            total_elapsed = time.perf_counter() - job_started_at
            log(f"Job {job_id} upload finished")
            log(
                f"Completed job {job_id}: {video_url} "
                f"(total {format_seconds(total_elapsed)}, audio {audio_duration:.1f}s)"
            )
            log(
                f"Job {job_id} summary: orientation={orientation}, render_size={render_size}, "
                f"output_size={output_size}, plan_key={plan_key}, audio={audio_duration:.1f}s, infer_frames={infer_frames}, "
                f"num_clip={num_clip}, cold_start={runner_cold_start}, "
                f"runner_acquire={format_seconds(runner_acquire_duration)}, "
                f"pipeline_init={format_seconds(runner.init_duration) if runner_cold_start else '0.0s'}, "
                f"render={format_seconds(render_finished_at - render_started_at)}, "
                f"clip_generation={format_seconds(clip_generation_duration) if clip_generation_duration is not None else 'n/a'}, "
                f"postprocess={format_seconds(postprocess_duration) if postprocess_duration is not None else 'n/a'}, "
                f"upscale={format_seconds(upscale_finished_at - upscale_started_at)}, "
                f"upload={format_seconds(upload_finished_at - upload_started_at)}"
            )
        except JobStoppedByServer as exc:
            log(f"Job {job_id} canceled by server: {exc}")
            return
        except Exception as exc:
            error_text = str(exc)
            log(f"Job {job_id} failed: {error_text}")
            try:
                call_worker_api({"action": "fail", "job_id": job_id, "error_text": error_text[:1500]})
            except Exception as fail_exc:
                log(f"Could not mark job {job_id} as failed: {fail_exc}")
            raise


def initial_poll() -> List[str]:
    result = call_worker_api({"action": "poll"})
    jobs = result.get("jobs", []) or []
    return [job["id"] for job in jobs if isinstance(job, dict) and job.get("id")]


def handle_signal(signum: int, _frame: Any) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    log(f"Received signal {signum}, stopping after current iteration")


def main() -> int:
    load_config_file(CONFIG_PATH)
    load_env_file(ENV_PATH)
    (REPO_ROOT / "worker_runs").mkdir(exist_ok=True)

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, handle_signal)

    poll_interval = float(os.getenv("WORKER_POLL_INTERVAL_SECONDS", "10"))
    once = "--once" in sys.argv

    if not TORCHRUN.exists():
        raise RuntimeError(f"torchrun not found: {TORCHRUN}")
    if not PYTHON_BIN.exists():
        raise RuntimeError(f"python not found: {PYTHON_BIN}")

    log(
        "SmartBlog LiveAvatar worker started "
        f"(ENABLE_COMPILE={os.getenv('ENABLE_COMPILE', 'true')}, "
        f"portrait_render={os.getenv('LIVEAVATAR_RENDER_PORTRAIT_SIZE', '832*480')}, "
        f"landscape_render={os.getenv('LIVEAVATAR_RENDER_LANDSCAPE_SIZE', '480*832')})"
    )

    while not STOP_REQUESTED:
        try:
            job_ids = initial_poll()
            if job_ids:
                log(f"Polled {len(job_ids)} queued job(s)")
            for job_id in job_ids:
                if STOP_REQUESTED:
                    break
                try:
                    process_job(job_id)
                except Exception:
                    # The job failure is already reported back to the API.
                    continue
        except Exception as exc:
            log(f"Poll loop error: {exc}")

        if once:
            break
        time.sleep(poll_interval)

    log("SmartBlog LiveAvatar worker stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
