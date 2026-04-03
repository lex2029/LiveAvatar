#!/usr/bin/env python3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
import importlib
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
from urllib.parse import urlparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from PIL import Image, __version__ as PILLOW_VERSION
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
PROCESS_STARTED_AT = time.monotonic()


@dataclass(frozen=True)
class RuntimeProfile:
    name: str
    infer_frames: int
    sample_steps: int
    direct_final_encode: bool
    chunk_size: int = 512


class JobStoppedByServer(RuntimeError):
    pass


def log(message: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    print(f"[{ts}] {message}", flush=True)


def git_commit_short() -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def git_branch_name() -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--abbrev-ref", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def git_is_dirty() -> bool:
    try:
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        )
        return bool(result.stdout.strip())
    except Exception:
        return False


def parse_timestamp_seconds(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        pass
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        return None


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


def worker_api_host() -> str:
    url = os.getenv("SUPABASE_URL", "").strip()
    if not url:
        return "unknown"
    parsed = urlparse(url)
    return parsed.netloc or "unknown"


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


def resize_image_to_render_aspect(image_path: Path, render_size: str) -> None:
    target_height, target_width = map(int, render_size.split("*"))
    target_aspect = target_width / float(target_height)
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        src_width, src_height = image.size
        src_aspect = src_width / float(src_height)
        if abs(src_aspect - target_aspect) < 1e-4:
            return

        if src_aspect > target_aspect:
            new_width = max(1, int(round(src_height * target_aspect)))
            new_height = src_height
        else:
            new_width = src_width
            new_height = max(1, int(round(src_width / target_aspect)))

        resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        resized.save(image_path)
        log(
            f"Resized avatar to render aspect {render_size}: "
            f"{src_width}x{src_height} -> {new_width}x{new_height}"
        )


def save_video_fast_nvenc(
    tensor: torch.Tensor,
    save_file: Path,
    fps: int,
    normalize: bool = True,
    value_range=(-1, 1),
) -> None:
    if tensor.ndim != 5 or tensor.shape[0] != 1:
        raise ValueError(f"fast raw save expects tensor shape [1,C,T,H,W], got {tuple(tensor.shape)}")
    frames = tensor.detach()
    frame_count = int(frames.shape[2])
    height, width = int(frames.shape[3]), int(frames.shape[4])
    command = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "h264_nvenc",
        "-preset",
        "p4",
        "-movflags",
        "+faststart",
        str(save_file),
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdin is not None
    try:
        _stream_rgb24_tensor_to_ffmpeg(
            process.stdin,
            frames,
            normalize=normalize,
            value_range=value_range,
        )
        process.stdin.close()
    except Exception:
        process.kill()
        raise
    stderr = process.stderr.read() if process.stderr is not None else b""
    process.wait()
    if process.returncode != 0:
        raise RuntimeError(
            f"fast nvenc raw save failed (exit {process.returncode}): "
            f"{stderr.decode(errors='replace').strip()}"
        )


def save_video_final_nvenc(
    tensor: torch.Tensor,
    save_file: Path,
    audio_path: Path,
    fps: int,
    output_size: Optional[str] = None,
    normalize: bool = True,
    value_range=(-1, 1),
) -> None:
    if tensor.ndim != 5 or tensor.shape[0] != 1:
        raise ValueError(f"direct final save expects tensor shape [1,C,T,H,W], got {tuple(tensor.shape)}")
    frames = tensor.detach()
    audio_duration = probe_audio_duration(audio_path)
    max_frames = max(1, math.ceil(audio_duration * fps))
    if frames.shape[2] > max_frames:
        frames = frames[:, :, :max_frames]
    height, width = int(frames.shape[3]), int(frames.shape[4])
    vf = ["hwupload_cuda"]
    if output_size:
        out_h, out_w = map(int, output_size.split("*"))
        vf.append(f"scale_cuda={out_w}:{out_h}")
    command = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-i",
        str(audio_path),
        "-vf",
        ",".join(vf),
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
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-shortest",
        "-movflags",
        "+faststart",
        str(save_file),
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdin is not None
    try:
        _stream_rgb24_tensor_to_ffmpeg(
            process.stdin,
            frames,
            normalize=normalize,
            value_range=value_range,
        )
        process.stdin.close()
    except Exception:
        process.kill()
        raise
    stderr = process.stderr.read() if process.stderr is not None else b""
    process.wait()
    if process.returncode != 0:
        raise RuntimeError(
            f"direct final nvenc save failed (exit {process.returncode}): "
            f"{stderr.decode(errors='replace').strip()}"
        )


def _stream_rgb24_tensor_to_ffmpeg(
    stdin,
    tensor: torch.Tensor,
    normalize: bool = True,
    value_range=(-1, 1),
    batch_frames: int = 8,
) -> None:
    if tensor.ndim != 5 or tensor.shape[0] != 1:
        raise ValueError(f"expected tensor shape [1,C,T,H,W], got {tuple(tensor.shape)}")
    frame_count = int(tensor.shape[2])
    lo, hi = value_range
    scale = 255.0 / float(hi - lo)
    for start in range(0, frame_count, batch_frames):
        end = min(frame_count, start + batch_frames)
        chunk = tensor[:, :, start:end]
        if normalize:
            chunk = chunk.clamp(lo, hi)
            chunk = (chunk - lo) * scale
        else:
            chunk = chunk * 255.0
        chunk = (
            chunk[0]
            .permute(1, 2, 3, 0)
            .round()
            .to(torch.uint8)
            .cpu()
            .contiguous()
        )
        stdin.write(chunk.numpy().tobytes())


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


def choose_runtime_profile(audio_duration: float) -> RuntimeProfile:
    short_max_seconds = float(os.getenv("LIVEAVATAR_SHORT_AUDIO_MAX_SECONDS", "3.0"))
    if audio_duration <= short_max_seconds:
        return RuntimeProfile(
            name="short",
            infer_frames=int(os.getenv("LIVEAVATAR_SHORT_INFER_FRAMES", "64")),
            sample_steps=int(os.getenv("LIVEAVATAR_SHORT_SAMPLE_STEPS", "4")),
            direct_final_encode=os.getenv("LIVEAVATAR_SHORT_DIRECT_FINAL_ENCODE", "true").lower() == "true",
            chunk_size=int(os.getenv("LIVEAVATAR_SHORT_CHUNK_SIZE", "512")),
        )
    return RuntimeProfile(
        name="long",
        infer_frames=int(os.getenv("LIVEAVATAR_LONG_INFER_FRAMES", "48")),
        sample_steps=int(os.getenv("LIVEAVATAR_LONG_SAMPLE_STEPS", "4")),
        direct_final_encode=os.getenv("LIVEAVATAR_LONG_DIRECT_FINAL_ENCODE", "false").lower() == "true",
        chunk_size=int(os.getenv("LIVEAVATAR_LONG_CHUNK_SIZE", "512")),
    )


@contextmanager
def temporary_env(overrides: Dict[str, str]):
    old_values: Dict[str, Optional[str]] = {}
    for key, value in overrides.items():
        old_values[key] = os.environ.get(key)
        os.environ[key] = value
    try:
        yield
    finally:
        for key, old_value in old_values.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


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
        return_video_tensor: bool = False,
    ) -> Optional[torch.Tensor]:
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

        if return_video_tensor:
            return video

        direct_final_encode = os.getenv("LIVEAVATAR_DIRECT_FINAL_ENCODE", "false").lower() == "true"
        use_fast_raw_save = os.getenv("LIVEAVATAR_FAST_RAW_SAVE", "true").lower() == "true"
        if direct_final_encode:
            log("Saving final video via direct NVENC path")
            save_video_final_nvenc(
                tensor=video[None],
                save_file=output_path,
                audio_path=audio_path,
                fps=self.cfg.sample_fps,
                output_size=None,
                normalize=True,
                value_range=(-1, 1),
            )
        elif use_fast_raw_save:
            log("Saving raw video via fast NVENC path")
            save_video_fast_nvenc(
                tensor=video[None],
                save_file=output_path,
                fps=self.cfg.sample_fps,
                normalize=True,
                value_range=(-1, 1),
            )
            merge_video_audio(video_path=str(output_path), audio_path=str(audio_path))
        else:
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
        return None

    def release_gpu_for_ffmpeg(self) -> None:
        return

    def shutdown(self) -> None:
        try:
            if hasattr(self, "dist") and self.dist.is_initialized():
                self.dist.destroy_process_group()
        except Exception as exc:
            log(f"Resident runner shutdown warning: {exc}")


def get_runner() -> ResidentLiveAvatarRunner:
    global RUNNER
    cold_start = False
    acquire_started_at = time.perf_counter()
    if RUNNER is None:
        RUNNER = ResidentLiveAvatarRunner()
        cold_start = True
    acquire_duration = time.perf_counter() - acquire_started_at
    return RUNNER, cold_start, acquire_duration


def cleanup_runner() -> None:
    global RUNNER
    if RUNNER is None:
        log("Resident runner cleanup: runner_loaded=False")
        return
    try:
        log(f"Resident runner cleanup: {runner_state_summary()}")
        RUNNER.shutdown()
    finally:
        RUNNER = None


def runner_state_summary() -> str:
    if RUNNER is None:
        return "runner_loaded=False, runner_jobs=0"
    return (
        "runner_loaded=True, "
        f"runner_jobs={RUNNER.jobs_processed}, "
        f"runner_init={format_seconds(RUNNER.init_duration)}"
    )


def runner_state_info() -> Dict[str, Any]:
    if RUNNER is None:
        return {
            "runner_loaded": False,
            "runner_jobs": 0,
            "runner_init_s": None,
        }
    return {
        "runner_loaded": True,
        "runner_jobs": int(RUNNER.jobs_processed),
        "runner_init_s": round(float(RUNNER.init_duration), 3),
    }


def worker_uptime_seconds() -> float:
    return max(0.0, time.monotonic() - PROCESS_STARTED_AT)


def worker_pid() -> int:
    return os.getpid()


def worker_host() -> str:
    return socket.gethostname()


def worker_cuda_visible_devices() -> str:
    return os.getenv("CUDA_VISIBLE_DEVICES", "unset")


def worker_cuda_available() -> bool:
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def worker_cuda_device_count() -> int:
    try:
        return int(torch.cuda.device_count())
    except Exception:
        return 0


def worker_cuda_device_names() -> List[str]:
    try:
        return [str(torch.cuda.get_device_name(index)) for index in range(torch.cuda.device_count())]
    except Exception:
        return []


def worker_cuda_devices_info() -> List[Dict[str, Any]]:
    try:
        devices: List[Dict[str, Any]] = []
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            capability = f"{props.major}.{props.minor}"
            total_memory_gb = round(float(props.total_memory) / (1024**3), 2)
            devices.append(
                {
                    "index": index,
                    "name": str(props.name),
                    "capability": capability,
                    "total_memory_gb": total_memory_gb,
                }
            )
        return devices
    except Exception:
        return []


def worker_cuda_memory_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "free_gb": None,
        "total_gb": None,
        "allocated_gb": None,
        "reserved_gb": None,
    }
    try:
        if not torch.cuda.is_available():
            return info
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        info["free_gb"] = round(float(free_bytes) / (1024**3), 2)
        info["total_gb"] = round(float(total_bytes) / (1024**3), 2)
        info["allocated_gb"] = round(float(torch.cuda.memory_allocated()) / (1024**3), 3)
        info["reserved_gb"] = round(float(torch.cuda.memory_reserved()) / (1024**3), 3)
    except Exception as exc:
        info["error"] = str(exc)
    return info


def host_memory_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "total_gb": None,
        "available_gb": None,
    }
    try:
        meminfo: Dict[str, int] = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            number = value.strip().split()[0]
            meminfo[key] = int(number)
        if "MemTotal" in meminfo:
            info["total_gb"] = round(float(meminfo["MemTotal"]) / (1024**2), 2)
        if "MemAvailable" in meminfo:
            info["available_gb"] = round(float(meminfo["MemAvailable"]) / (1024**2), 2)
    except Exception as exc:
        info["error"] = str(exc)
    return info


def runtime_dependency_flags() -> Dict[str, bool]:
    wan_ckpt = (REPO_ROOT / "ckpt" / "Wan2.2-S2V-14B").exists()
    liveavatar_lora = (REPO_ROOT / "ckpt" / "LiveAvatar" / "liveavatar.safetensors").exists()
    ffmpeg_ready = shutil.which("ffmpeg") is not None
    ffprobe_ready = shutil.which("ffprobe") is not None
    supabase_url_ready = bool(os.getenv("SUPABASE_URL"))
    worker_api_key_ready = bool(os.getenv("WORKER_API_KEY"))
    return {
        "ffmpeg_ready": ffmpeg_ready,
        "ffprobe_ready": ffprobe_ready,
        "supabase_url_ready": supabase_url_ready,
        "worker_api_key_ready": worker_api_key_ready,
        "torchrun_ready": TORCHRUN.exists(),
        "python_ready": PYTHON_BIN.exists(),
        "wan_ckpt_ready": wan_ckpt,
        "lora_ready": liveavatar_lora,
    }


def command_version(command: str) -> Optional[str]:
    binary = shutil.which(command)
    if not binary:
        return None
    try:
        result = subprocess.run(
            [binary, "-version"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    first_line = result.stdout.splitlines()[0].strip() if result.stdout else ""
    return first_line or None


def ffmpeg_encoder_available(encoder_name: str) -> Optional[bool]:
    ffmpeg_binary = shutil.which("ffmpeg")
    if not ffmpeg_binary:
        return None
    try:
        result = subprocess.run(
            [ffmpeg_binary, "-hide_banner", "-encoders"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    pattern = f" {encoder_name}"
    return any(pattern in line for line in result.stdout.splitlines())


def nvidia_smi_info() -> Dict[str, Any]:
    nvidia_smi = shutil.which("nvidia-smi")
    info: Dict[str, Any] = {
        "available": nvidia_smi is not None,
        "driver_version": None,
        "cuda_version": None,
    }
    if not nvidia_smi:
        return info
    try:
        result = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        first_line = result.stdout.splitlines()[0].strip() if result.stdout else ""
        if first_line:
            info["driver_version"] = first_line.split(",")[0].strip()
        banner = subprocess.run(
            [nvidia_smi],
            check=True,
            capture_output=True,
            text=True,
        )
        banner_line = banner.stdout.splitlines()[0].strip() if banner.stdout else ""
        marker = "CUDA Version:"
        if marker in banner_line:
            info["cuda_version"] = banner_line.split(marker, 1)[1].strip().split()[0]
    except Exception as exc:
        info["error"] = str(exc)
    return info


def compile_runtime_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "enable_compile": os.getenv("ENABLE_COMPILE", "true"),
        "torch_compile_available": hasattr(torch, "compile"),
        "triton_available": False,
        "inductor_available": False,
    }
    try:
        import triton  # type: ignore

        info["triton_available"] = True
        info["triton_version"] = getattr(triton, "__version__", None)
    except Exception:
        pass
    try:
        importlib.import_module("torch._inductor")
        info["inductor_available"] = True
    except Exception:
        pass
    return info


def disk_space_gb(path: Path) -> Optional[float]:
    try:
        usage = shutil.disk_usage(path)
    except Exception:
        return None
    return round(float(usage.free) / (1024**3), 2)


def path_writable(path: Path) -> Optional[bool]:
    try:
        return os.access(path, os.W_OK)
    except Exception:
        return None


def runtime_storage_info() -> Dict[str, Optional[float]]:
    generated_assets_dir = Path(os.getenv("GENERATED_ASSETS_DIR", str(REPO_ROOT)))
    return {
        "repo_free_gb": disk_space_gb(REPO_ROOT),
        "tmp_free_gb": disk_space_gb(Path("/tmp")),
        "generated_assets_free_gb": disk_space_gb(generated_assets_dir),
        "repo_writable": path_writable(REPO_ROOT),
        "tmp_writable": path_writable(Path("/tmp")),
        "generated_assets_writable": path_writable(generated_assets_dir),
    }


def runtime_paths() -> Dict[str, str]:
    return {
        "repo_root": str(REPO_ROOT),
        "tmp_dir": "/tmp",
        "generated_assets_dir": os.getenv("GENERATED_ASSETS_DIR", str(REPO_ROOT)),
        "venv_bin": str(VENV_BIN),
        "torchrun": str(TORCHRUN),
        "python_bin": str(PYTHON_BIN),
    }


def runtime_tunables(poll_interval: float, idle_log_interval: float) -> Dict[str, float]:
    return {
        "poll_interval_s": poll_interval,
        "idle_log_interval_s": idle_log_interval,
        "worker_api_timeout_s": 60.0,
        "download_timeout_s": 300.0,
    }


def file_metadata(path: Path) -> Dict[str, Any]:
    exists = path.exists()
    metadata: Dict[str, Any] = {
        "path": str(path),
        "exists": exists,
        "is_dir": path.is_dir() if exists else False,
        "size_gb": None,
        "mtime_epoch_s": None,
    }
    if not exists:
        return metadata
    try:
        stat = path.stat()
        metadata["mtime_epoch_s"] = round(float(stat.st_mtime), 3)
        if path.is_dir():
            total_size = 0
            for child in path.rglob("*"):
                if child.is_file():
                    try:
                        total_size += child.stat().st_size
                    except Exception:
                        continue
            metadata["size_gb"] = round(float(total_size) / (1024**3), 3)
        else:
            metadata["size_gb"] = round(float(stat.st_size) / (1024**3), 3)
    except Exception:
        pass
    return metadata


def runtime_artifacts() -> Dict[str, Dict[str, Any]]:
    return {
        "wan_ckpt_dir": file_metadata(REPO_ROOT / "ckpt" / "Wan2.2-S2V-14B"),
        "lora_file": file_metadata(REPO_ROOT / "ckpt" / "LiveAvatar" / "liveavatar.safetensors"),
    }


def worker_api_dns_info() -> Dict[str, Any]:
    host = worker_api_host()
    info: Dict[str, Any] = {
        "host": host,
        "resolved": False,
        "ip_addresses": [],
    }
    if host == "unknown":
        return info
    try:
        addresses = sorted({entry[4][0] for entry in socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP)})
        info["resolved"] = True
        info["ip_addresses"] = addresses
    except Exception as exc:
        info["error"] = str(exc)
    return info


def worker_api_connectivity_info() -> Dict[str, Any]:
    host = worker_api_host()
    info: Dict[str, Any] = {
        "host": host,
        "port": 443,
        "reachable": False,
        "connect_s": None,
    }
    if host == "unknown":
        return info
    started_at = time.perf_counter()
    try:
        with socket.create_connection((host, 443), timeout=5.0):
            info["reachable"] = True
            info["connect_s"] = round(time.perf_counter() - started_at, 3)
    except Exception as exc:
        info["error"] = str(exc)
    return info


def worker_api_http_probe() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "url": worker_api_url() if os.getenv("SUPABASE_URL") else None,
        "ok": False,
        "status_code": None,
        "request_s": None,
    }
    if not os.getenv("SUPABASE_URL") or not os.getenv("WORKER_API_KEY"):
        info["error"] = "worker API env is incomplete"
        return info
    started_at = time.perf_counter()
    try:
        response = requests.post(
            worker_api_url(),
            json={"action": "poll", "job_type": "render_video"},
            headers=worker_headers(),
            timeout=60,
        )
        info["status_code"] = response.status_code
        info["request_s"] = round(time.perf_counter() - started_at, 3)
        info["ok"] = response.ok
    except Exception as exc:
        info["error"] = str(exc)
    return info


def runtime_dependency_summary() -> str:
    flags = runtime_dependency_flags()
    return (
        f"ffmpeg_ready={flags['ffmpeg_ready']}, "
        f"ffprobe_ready={flags['ffprobe_ready']}, "
        f"supabase_url_ready={flags['supabase_url_ready']}, "
        f"worker_api_key_ready={flags['worker_api_key_ready']}, "
        f"torchrun_ready={flags['torchrun_ready']}, "
        f"python_ready={flags['python_ready']}, "
        f"wan_ckpt_ready={flags['wan_ckpt_ready']}, "
        f"lora_ready={flags['lora_ready']}"
    )


def startup_summary(poll_interval: float, idle_log_interval: float) -> str:
    return (
        f"git_commit={git_commit_short()}, "
        f"git_branch={git_branch_name()}, "
        f"git_dirty={git_is_dirty()}, "
        f"worker_host={worker_host()}, "
        f"worker_pid={worker_pid()}, "
        f"cuda_visible_devices={worker_cuda_visible_devices()}, "
        f"cuda_available={worker_cuda_available()}, "
        f"cuda_device_count={worker_cuda_device_count()}, "
        f"worker_api_host={worker_api_host()}, "
        f"{runtime_dependency_summary()}, "
        f"ENABLE_COMPILE={os.getenv('ENABLE_COMPILE', 'true')}, "
        f"poll_interval={format_seconds(poll_interval)}, "
        f"idle_log_interval={format_seconds(idle_log_interval)}, "
        f"portrait_render={os.getenv('LIVEAVATAR_RENDER_PORTRAIT_SIZE', '832*480')}, "
        f"landscape_render={os.getenv('LIVEAVATAR_RENDER_LANDSCAPE_SIZE', '480*832')}, "
        f"short<= {os.getenv('LIVEAVATAR_SHORT_AUDIO_MAX_SECONDS', '3.0')}s:"
        f"if{os.getenv('LIVEAVATAR_SHORT_INFER_FRAMES', '64')}/"
        f"df={os.getenv('LIVEAVATAR_SHORT_DIRECT_FINAL_ENCODE', 'true')}/"
        f"chunk={os.getenv('LIVEAVATAR_SHORT_CHUNK_SIZE', '512')}, "
        f"long> {os.getenv('LIVEAVATAR_SHORT_AUDIO_MAX_SECONDS', '3.0')}s:"
        f"if{os.getenv('LIVEAVATAR_LONG_INFER_FRAMES', '48')}/"
        f"df={os.getenv('LIVEAVATAR_LONG_DIRECT_FINAL_ENCODE', 'false')}/"
        f"chunk={os.getenv('LIVEAVATAR_LONG_CHUNK_SIZE', '512')}"
    )


def render_size_config() -> Dict[str, str]:
    return {
        "portrait_render": os.getenv("LIVEAVATAR_RENDER_PORTRAIT_SIZE", "832*480"),
        "landscape_render": os.getenv("LIVEAVATAR_RENDER_LANDSCAPE_SIZE", "480*832"),
    }


def runtime_profile_config() -> Dict[str, Any]:
    short_audio_max_seconds = float(os.getenv("LIVEAVATAR_SHORT_AUDIO_MAX_SECONDS", "3.0"))
    return {
        "short": {
            "max_audio_s": short_audio_max_seconds,
            "infer_frames": int(os.getenv("LIVEAVATAR_SHORT_INFER_FRAMES", "64")),
            "sample_steps": int(os.getenv("LIVEAVATAR_SHORT_SAMPLE_STEPS", "4")),
            "direct_final_encode": os.getenv("LIVEAVATAR_SHORT_DIRECT_FINAL_ENCODE", "true").lower() == "true",
            "chunk_size": int(os.getenv("LIVEAVATAR_SHORT_CHUNK_SIZE", "512")),
        },
        "long": {
            "infer_frames": int(os.getenv("LIVEAVATAR_LONG_INFER_FRAMES", "48")),
            "sample_steps": int(os.getenv("LIVEAVATAR_LONG_SAMPLE_STEPS", "4")),
            "direct_final_encode": os.getenv("LIVEAVATAR_LONG_DIRECT_FINAL_ENCODE", "false").lower() == "true",
            "chunk_size": int(os.getenv("LIVEAVATAR_LONG_CHUNK_SIZE", "512")),
        },
    }


def attention_runtime_env() -> Dict[str, str]:
    return {
        "cross_attn_chunk_size": os.getenv("LIVEAVATAR_CROSS_ATTN_CHUNK_SIZE", "256"),
        "rope_chunk_size": os.getenv("LIVEAVATAR_ROPE_CHUNK_SIZE", "256"),
        "attn_out_chunk_size": os.getenv("LIVEAVATAR_ATTN_OUT_CHUNK_SIZE", "256"),
        "qkv_chunk_size": os.getenv("LIVEAVATAR_QKV_CHUNK_SIZE", "256"),
    }


def run_healthcheck(poll_interval: float, idle_log_interval: float) -> int:
    log(f"SmartBlog LiveAvatar worker healthcheck ({startup_summary(poll_interval, idle_log_interval)})")
    if not os.getenv("SUPABASE_URL") or not os.getenv("WORKER_API_KEY"):
        log("Healthcheck skipped poll: worker API env is incomplete")
        return 0
    try:
        poll_started_at = time.perf_counter()
        job_ids = initial_poll()
        poll_duration = time.perf_counter() - poll_started_at
        log(
            "Healthcheck poll "
            f"(jobs={len(job_ids)}, poll={format_seconds(poll_duration)}, worker_api_host={worker_api_host()})"
        )
        return 0
    except Exception as exc:
        log(f"Healthcheck poll failed: {exc}")
        return 1


def run_healthcheck_json(poll_interval: float, idle_log_interval: float) -> int:
    payload: Dict[str, Any] = {
        "git_commit": git_commit_short(),
        "git_branch": git_branch_name(),
        "git_dirty": git_is_dirty(),
        "worker_uptime_s": round(worker_uptime_seconds(), 3),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "requests_version": getattr(requests, "__version__", None),
        "pillow_version": PILLOW_VERSION,
        "compile_runtime": compile_runtime_info(),
        "nvidia_smi": nvidia_smi_info(),
        "worker_host": worker_host(),
        "worker_pid": worker_pid(),
        "cuda_visible_devices": worker_cuda_visible_devices(),
        "cuda_available": worker_cuda_available(),
        "cuda_device_count": worker_cuda_device_count(),
        "cuda_device_names": worker_cuda_device_names(),
        "cuda_devices": worker_cuda_devices_info(),
        "cuda_memory": worker_cuda_memory_info(),
        "host_memory": host_memory_info(),
        "worker_api_host": worker_api_host(),
        "runtime_dependencies": runtime_dependency_flags(),
        "paths": runtime_paths(),
        "storage": runtime_storage_info(),
        "tunables": runtime_tunables(poll_interval, idle_log_interval),
        "artifacts": runtime_artifacts(),
        "attention_runtime_env": attention_runtime_env(),
        "media_tools": {
            "ffmpeg_version": command_version("ffmpeg"),
            "ffprobe_version": command_version("ffprobe"),
            "h264_nvenc_available": ffmpeg_encoder_available("h264_nvenc"),
        },
        "render_sizes": render_size_config(),
        "profiles": runtime_profile_config(),
        "worker_api_dns": worker_api_dns_info(),
        "worker_api_connectivity": worker_api_connectivity_info(),
        "worker_api_http": worker_api_http_probe(),
        "runner_state": runner_state_info(),
    }
    runtime_dependencies = payload["runtime_dependencies"]
    storage = payload["storage"]
    api_dns = payload["worker_api_dns"]
    api_connectivity = payload["worker_api_connectivity"]
    api_http = payload["worker_api_http"]
    compile_runtime = payload["compile_runtime"]
    media_tools = payload["media_tools"]
    health_checks = {
        "api_env_ready": bool(
            runtime_dependencies["supabase_url_ready"] and runtime_dependencies["worker_api_key_ready"]
        ),
        "api_dns_ready": bool(api_dns["resolved"]),
        "api_tcp_ready": bool(api_connectivity["reachable"]),
        "api_http_ready": bool(api_http["ok"]),
        "media_ready": bool(runtime_dependencies["ffmpeg_ready"] and runtime_dependencies["ffprobe_ready"]),
        "nvenc_ready": bool(media_tools["h264_nvenc_available"]),
        "model_ready": bool(runtime_dependencies["wan_ckpt_ready"] and runtime_dependencies["lora_ready"]),
        "python_runtime_ready": bool(runtime_dependencies["python_ready"] and runtime_dependencies["torchrun_ready"]),
        "storage_writable": bool(
            storage["repo_writable"] and storage["tmp_writable"] and storage["generated_assets_writable"]
        ),
        "compile_runtime_ready": bool(
            compile_runtime["torch_compile_available"]
            and compile_runtime["inductor_available"]
            and compile_runtime["triton_available"]
        ),
    }
    payload["health_checks"] = health_checks
    payload["overall_ok"] = all(health_checks.values())
    warnings: List[str] = []
    if payload["git_dirty"]:
        warnings.append("git_dirty")
    if payload["git_branch"] != "main":
        warnings.append(f"non_main_branch:{payload['git_branch']}")
    if payload["nvidia_smi"]["available"] and payload["nvidia_smi"]["cuda_version"] is None:
        warnings.append("nvidia_smi_cuda_version_unavailable")
    if not payload["runner_state"]["runner_loaded"]:
        warnings.append("runner_not_loaded")
    payload["warnings"] = warnings
    if not os.getenv("SUPABASE_URL") or not os.getenv("WORKER_API_KEY"):
        payload["poll_skipped"] = True
        payload["poll_error"] = "worker API env is incomplete"
        print(json.dumps(payload, sort_keys=True), flush=True)
        return 0 if payload["overall_ok"] else 1
    try:
        poll_started_at = time.perf_counter()
        job_ids = initial_poll()
        poll_duration = time.perf_counter() - poll_started_at
        payload["poll_skipped"] = False
        payload["jobs"] = len(job_ids)
        payload["poll_s"] = round(poll_duration, 3)
        print(json.dumps(payload, sort_keys=True), flush=True)
        return 0 if payload["overall_ok"] else 1
    except Exception as exc:
        payload["poll_skipped"] = False
        payload["poll_error"] = str(exc)
        payload["health_checks"]["api_http_ready"] = False
        payload["overall_ok"] = False
        print(json.dumps(payload, sort_keys=True), flush=True)
        return 1


def run_healthcheck_json_only(poll_interval: float, idle_log_interval: float) -> int:
    return run_healthcheck_json(poll_interval, idle_log_interval)


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
    claim_started_at = time.perf_counter()
    claim = call_worker_api({"action": "claim", "job_id": job_id})
    claim_duration = time.perf_counter() - claim_started_at
    if not claim.get("claimed"):
        log(
            f"Job {job_id} was not claimed after {format_seconds(claim_duration)}: "
            f"{claim.get('reason', 'unknown reason')}"
        )
        return

    job = claim.get("job", {})
    if job.get("job_type") not in {None, "", "render_video"}:
        unsupported = f"Unsupported job_type for this worker: {job.get('job_type')}"
        log(f"Job {job_id} {unsupported}")
        try:
            call_worker_api({"action": "fail", "job_id": job_id, "error_text": unsupported})
        except Exception as fail_exc:
            log(f"Could not mark unsupported job {job_id} as failed: {fail_exc}")
        return
    payload = job.get("payload", {}) or {}
    assets = claim["assets"]
    upload = claim["upload"]
    job_started_at = time.perf_counter()
    queue_wait_duration: Optional[float] = None
    created_at_seconds = parse_timestamp_seconds(job.get("created_at"))
    if created_at_seconds is not None:
        queue_wait_duration = max(0.0, time.time() - created_at_seconds)
    try:
        ack = call_worker_api({"action": "progress", "job_id": job_id, "progress": 0})
        if ack.get("stop"):
            raise JobStoppedByServer(
                f"Server stopped job: {ack.get('status', 'unknown')} / {ack.get('reason', 'no reason')}"
            )
        log(f"Job {job_id} progress: 0% (claimed)")
    except JobStoppedByServer:
        raise
    except Exception as progress_exc:
        log(f"Job {job_id} initial progress update failed at 0%: {progress_exc}")

    with tempfile.TemporaryDirectory(prefix=f"smartblog_{job_id}_", dir=str(REPO_ROOT / "worker_runs")) as temp_dir:
        temp_root = Path(temp_dir)
        image_path = temp_root / "avatar.png"
        audio_path = temp_root / "audio.mp3"
        raw_video_path = temp_root / "rendered_raw.mp4"
        final_video_path = temp_root / "rendered.mp4"
        assets_download_duration: Optional[float] = None

        try:
            log(f"Downloading assets for {job_id}")
            assets_download_started_at = time.perf_counter()
            download_file(assets["avatar_url"], image_path)
            download_file(assets["audio_url"], audio_path)
            assets_download_duration = time.perf_counter() - assets_download_started_at

            orientation = payload.get("orientation")
            if orientation not in {"portrait", "landscape"}:
                orientation = detect_orientation(image_path)

            plan_key = plan_key_for_job(job, payload)
            render_size = orientation_to_render_size(orientation)
            output_size = orientation_to_output_size(orientation)
            resize_image_to_render_aspect(image_path, render_size)
            prompt = choose_prompt(payload)
            sample_fps = 25
            audio_duration = probe_audio_duration(audio_path)
            runtime_profile = choose_runtime_profile(audio_duration)
            infer_frames = runtime_profile.infer_frames
            num_clip = compute_num_clip(audio_duration, infer_frames=infer_frames, fps=sample_fps)
            base_seed = int(payload.get("seed") or os.getenv("LIVEAVATAR_BASE_SEED", "420"))
            runtime_env = {
                "LIVEAVATAR_INFER_FRAMES": str(runtime_profile.infer_frames),
                "LIVEAVATAR_SAMPLE_STEPS": str(runtime_profile.sample_steps),
                "LIVEAVATAR_DIRECT_FINAL_ENCODE": "true" if runtime_profile.direct_final_encode else "false",
                "LIVEAVATAR_CROSS_ATTN_CHUNK_SIZE": str(runtime_profile.chunk_size),
                "LIVEAVATAR_ROPE_CHUNK_SIZE": str(runtime_profile.chunk_size),
                "LIVEAVATAR_ATTN_OUT_CHUNK_SIZE": str(runtime_profile.chunk_size),
                "LIVEAVATAR_QKV_CHUNK_SIZE": str(runtime_profile.chunk_size),
            }

            log(
                f"Processing job {job_id}: orientation={orientation}, plan_key={plan_key}, "
                f"render_size={render_size}, output_size={output_size}, "
                f"audio_duration={audio_duration:.2f}s, num_clip={num_clip}, "
                f"queue_wait={format_seconds(queue_wait_duration) if queue_wait_duration is not None else 'n/a'}, "
                f"profile={runtime_profile.name}, infer_frames={runtime_profile.infer_frames}, "
                f"direct_final={runtime_profile.direct_final_encode}, chunk={runtime_profile.chunk_size}"
            )
            log(
                f"Job {job_id} output path: "
                f"{'direct_final_nvenc' if runtime_profile.direct_final_encode else 'standard_postprocess'} "
                f"(raw_save={'fast_nvenc' if not runtime_profile.direct_final_encode else 'n/a'})"
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
            with temporary_env(runtime_env):
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
                f"output_size={output_size}, plan_key={plan_key}, audio={audio_duration:.1f}s, "
                f"worker_host={worker_host()}, "
                f"worker_pid={worker_pid()}, "
                f"cuda_visible_devices={worker_cuda_visible_devices()}, "
                f"cuda_available={worker_cuda_available()}, "
                f"cuda_device_count={worker_cuda_device_count()}, "
                f"worker_uptime={format_seconds(worker_uptime_seconds())}, "
                f"queue_wait={format_seconds(queue_wait_duration) if queue_wait_duration is not None else 'n/a'}, "
                f"claim={format_seconds(claim_duration)}, "
                f"assets_download={format_seconds(assets_download_duration) if assets_download_duration is not None else 'n/a'}, "
                f"git_commit={git_commit_short()}, "
                f"git_branch={git_branch_name()}, "
                f"git_dirty={git_is_dirty()}, "
                f"worker_api_host={worker_api_host()}, "
                f"profile={runtime_profile.name}, infer_frames={infer_frames}, "
                f"sample_steps={runtime_profile.sample_steps}, "
                f"direct_final={runtime_profile.direct_final_encode}, "
                f"save_path={'direct_final_nvenc' if runtime_profile.direct_final_encode else 'standard_postprocess'}, "
                f"chunk={runtime_profile.chunk_size}, "
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
    result = call_worker_api({"action": "poll", "job_type": "render_video"})
    jobs = result.get("jobs", []) or []
    return [
        job["id"]
        for job in jobs
        if isinstance(job, dict)
        and job.get("id")
        and job.get("job_type") in {None, "", "render_video"}
    ]


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
    idle_log_interval = float(os.getenv("WORKER_IDLE_LOG_INTERVAL_SECONDS", "300"))
    once = "--once" in sys.argv
    healthcheck = "--healthcheck" in sys.argv
    healthcheck_json = "--healthcheck-json" in sys.argv
    healthcheck_json_only = "--healthcheck-json-only" in sys.argv
    last_idle_log_at = 0.0
    consecutive_poll_errors = 0
    last_poll_error_at: Optional[float] = None

    if not TORCHRUN.exists():
        raise RuntimeError(f"torchrun not found: {TORCHRUN}")
    if not PYTHON_BIN.exists():
        raise RuntimeError(f"python not found: {PYTHON_BIN}")

    if not healthcheck_json_only:
        log(f"SmartBlog LiveAvatar worker started ({startup_summary(poll_interval, idle_log_interval)})")

    if healthcheck_json_only:
        return run_healthcheck_json_only(poll_interval, idle_log_interval)
    if healthcheck_json:
        return run_healthcheck_json(poll_interval, idle_log_interval)
    if healthcheck:
        return run_healthcheck(poll_interval, idle_log_interval)

    while not STOP_REQUESTED:
        try:
            poll_started_at = time.perf_counter()
            job_ids = initial_poll()
            poll_duration = time.perf_counter() - poll_started_at
            if consecutive_poll_errors > 0:
                recovered_after = (
                    time.monotonic() - last_poll_error_at
                    if last_poll_error_at is not None
                    else 0.0
                )
                log(
                    f"Poll loop recovered after {consecutive_poll_errors} error(s) "
                    f"in {format_seconds(recovered_after)}; "
                    f"successful poll={format_seconds(poll_duration)}"
                )
                consecutive_poll_errors = 0
                last_poll_error_at = None
            if job_ids:
                log(f"Polled {len(job_ids)} queued job(s) in {format_seconds(poll_duration)}")
            else:
                now = time.monotonic()
                if now - last_idle_log_at >= idle_log_interval:
                    log(
                        "Worker idle heartbeat: "
                        f"queue empty (poll={format_seconds(poll_duration)}, "
                        f"worker_host={worker_host()}, "
                        f"worker_pid={worker_pid()}, "
                        f"cuda_visible_devices={worker_cuda_visible_devices()}, "
                        f"cuda_available={worker_cuda_available()}, "
                        f"cuda_device_count={worker_cuda_device_count()}, "
                        f"worker_uptime={format_seconds(worker_uptime_seconds())}, "
                        f"{runner_state_summary()})"
                    )
                    last_idle_log_at = now
            for job_id in job_ids:
                if STOP_REQUESTED:
                    break
                try:
                    process_job(job_id)
                except Exception:
                    # The job failure is already reported back to the API.
                    continue
        except Exception as exc:
            consecutive_poll_errors += 1
            last_poll_error_at = time.monotonic()
            log(f"Poll loop error #{consecutive_poll_errors}: {exc}")

        if once:
            break
        time.sleep(poll_interval)

    stop_runner_state = runner_state_summary()
    cleanup_runner()
    log(
        "SmartBlog LiveAvatar worker stopped "
        f"(worker_host={worker_host()}, "
        f"worker_pid={worker_pid()}, "
        f"cuda_visible_devices={worker_cuda_visible_devices()}, "
        f"cuda_available={worker_cuda_available()}, "
        f"cuda_device_count={worker_cuda_device_count()}, "
        f"worker_uptime={format_seconds(worker_uptime_seconds())}, {stop_runner_state})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
