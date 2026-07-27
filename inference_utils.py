import os
import sys
import shutil
import subprocess
import torch
from typing import Dict, Any, Optional, List
from huggingface_hub import hf_hub_download


def configure_cpu_threadmaxing() -> int:
    """Configures environment variables and PyTorch CPU threads for threadmaxing performance on multi-core runner."""
    num_cpus: int = os.cpu_count() or 4
    os.environ["OMP_NUM_THREADS"] = str(num_cpus)
    os.environ["MKL_NUM_THREADS"] = str(num_cpus)
    os.environ["OPENBLAS_NUM_THREADS"] = str(num_cpus)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(num_cpus)
    os.environ["NUMEXPR_NUM_THREADS"] = str(num_cpus)
    os.environ["TORCH_NUM_THREADS"] = str(num_cpus)

    torch.set_num_threads(num_cpus)
    try:
        torch.set_num_interop_threads(num_cpus)
    except Exception as e:
        print(f"[LatentSync] Note: interop threads config: {e}")

    print(f"[LatentSync] CPU Threadmaxing configured with {num_cpus} threads.")
    return num_cpus


def download_latentsync_checkpoints(checkpoints_dir: str = "checkpoints") -> bool:
    """Downloads LatentSync 1.5 checkpoints (latentsync_unet.pt and whisper/tiny.pt) from Hugging Face if missing."""
    os.makedirs(checkpoints_dir, exist_ok=True)
    whisper_dir: str = os.path.join(checkpoints_dir, "whisper")
    os.makedirs(whisper_dir, exist_ok=True)

    unet_path: str = os.path.join(checkpoints_dir, "latentsync_unet.pt")
    whisper_path: str = os.path.join(whisper_dir, "tiny.pt")

    repo_id: str = "ByteDance/LatentSync"

    if not os.path.exists(unet_path):
        print(f"[LatentSync] Downloading latentsync_unet.pt to {unet_path}...")
        try:
            hf_hub_download(repo_id=repo_id, filename="latentsync_unet.pt", local_dir=checkpoints_dir)
            print("[LatentSync] Downloaded latentsync_unet.pt successfully.")
        except Exception as e:
            print(f"[LatentSync] Failed downloading latentsync_unet.pt via huggingface_hub: {e}")
            return False

    if not os.path.exists(whisper_path):
        print(f"[LatentSync] Downloading whisper/tiny.pt to {whisper_path}...")
        try:
            hf_hub_download(repo_id=repo_id, filename="whisper/tiny.pt", local_dir=checkpoints_dir)
            print("[LatentSync] Downloaded whisper/tiny.pt successfully.")
        except Exception as e:
            print(f"[LatentSync] Failed downloading whisper/tiny.pt via huggingface_hub: {e}")
            return False

    return os.path.exists(unet_path) and os.path.exists(whisper_path)


def ensure_latentsync_repository(target_dir: str = "LatentSync") -> str:
    """Ensures the official ByteDance LatentSync repository is present for model configs and pipeline utilities."""
    if not os.path.exists(target_dir):
        print(f"[LatentSync] Cloning ByteDance LatentSync repository into {target_dir}...")
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", "https://github.com/bytedance/LatentSync.git", target_dir],
                check=True
            )
            print("[LatentSync] LatentSync repository cloned successfully.")
        except Exception as e:
            print(f"[LatentSync] Error cloning LatentSync repository: {e}")

    if os.path.exists(target_dir) and target_dir not in sys.path:
        sys.path.insert(0, os.path.abspath(target_dir))

    return target_dir


def generate_synthetic_video(video_path: str = "clip.mp4", duration_sec: int = 3, fps: int = 25) -> bool:
    """Generates a default test presenter clip (25fps, 256x256) using ffmpeg if clip.mp4 does not exist."""
    print(f"[LatentSync] Generating synthetic test video {video_path} ({duration_sec}s @ {fps}fps)...")
    cmd: List[str] = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"testsrc=size=256x256:rate={fps}",
        "-t", str(duration_sec),
        "-pix_fmt", "yuv420p",
        video_path
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return os.path.exists(video_path)
    except Exception as e:
        print(f"[LatentSync] Failed to generate synthetic video using ffmpeg: {e}")
        return False


def generate_synthetic_audio(audio_path: str = "audio.wav", duration_sec: int = 3, sample_rate: int = 16000) -> bool:
    """Generates a default test audio file (16kHz WAV) using ffmpeg if audio.wav does not exist."""
    print(f"[LatentSync] Generating synthetic test audio {audio_path} ({duration_sec}s @ {sample_rate}Hz)...")
    cmd: List[str] = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"sine=frequency=440:sample_rate={sample_rate}",
        "-t", str(duration_sec),
        audio_path
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return os.path.exists(audio_path)
    except Exception as e:
        print(f"[LatentSync] Failed to generate synthetic audio using ffmpeg: {e}")
        return False


def ensure_sample_inputs(video_path: str = "clip.mp4", audio_path: str = "audio.wav") -> bool:
    """Ensures input clip.mp4 and audio.wav exist; generates fallback synthetic inputs for testing if missing."""
    video_ok: bool = os.path.exists(video_path) or generate_synthetic_video(video_path)
    audio_ok: bool = os.path.exists(audio_path) or generate_synthetic_audio(audio_path)
    return video_ok and audio_ok
