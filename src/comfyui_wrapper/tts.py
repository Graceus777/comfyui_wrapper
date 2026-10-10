"""Local SOAR voice cloning through an isolated, short-lived GPU worker."""
from __future__ import annotations

import json
import math
import os
import queue
import secrets
import shutil
import subprocess
import threading
import time
import wave
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from comfyui_wrapper.client import ComfyClient, ComfyError
from comfyui_wrapper.config import WrapperConfig
from comfyui_wrapper.paths import project_root, resolve_path

EVENT_PREFIX = "VOICE_CLONE_JSON "
MODEL_ID = "dots-studio/dots.tts-soar"
REQUIRED_MODEL_FILES = (
    "config.json", "llm_config.json", "model.safetensors", "vocoder.safetensors",
    "speaker_encoder.safetensors", "latent_stats.pt", "tokenizer.json",
)


class VoiceCloneError(RuntimeError):
    pass


@dataclass(frozen=True)
class VoiceCloneResult:
    audio: Path
    metadata: Path
    log: Path
    seed: int
    duration: float
    sample_rate: int
    peak_vram_gb: float


def runtime_paths(cfg: WrapperConfig) -> tuple[Path, Path, Path]:
    options = cfg.raw.get("voice_clone", {})
    root = project_root()
    python = resolve_path(options.get("python", ".local/tts/venv/Scripts/python.exe"), root=root)
    model = resolve_path(options.get("model_dir", "models/tts/dots.tts-soar"), root=root)
    worker = resolve_path(options.get("worker", "scripts/dots_tts_worker.py"), root=root)
    return python, model, worker


def installation_status(cfg: WrapperConfig) -> str:
    python, model, worker = runtime_paths(cfg)
    if not python.is_file() or not worker.is_file():
        return "Voice Clone needs setup. Run scripts/setup_voice_clone.ps1 first."
    missing = [name for name in REQUIRED_MODEL_FILES if not (model / name).is_file()]
    if missing:
        return "SOAR download incomplete: " + ", ".join(missing)
    return "Ready — dots.tts-SOAR, local voice cloning."


def prepare_gpu(cfg: WrapperConfig, on_progress: Callable[[str], None]) -> None:
    """Release idle ComfyUI caches, never interrupt or discard queued jobs."""
    if cfg.comfyui.host not in {"127.0.0.1", "localhost", "::1"}:
        return
    client = ComfyClient(cfg=cfg)
    try:
        state = client.get("/queue", timeout=2)
    except ComfyError:
        return  # Local TTS also works when ComfyUI is not running.
    if not isinstance(state, dict):
        raise VoiceCloneError("Could not check the ComfyUI queue before voice cloning.")
    if state.get("queue_running") or state.get("queue_pending"):
        raise VoiceCloneError("ComfyUI has active or queued jobs. Wait for them to finish before cloning.")
    on_progress("Releasing idle image/video models to make room for the voice model...")
    client.post("/free", {"unload_models": True, "free_memory": True}, timeout=10)


def clone_voice(
    text: str,
    reference_audio: str | Path,
    *,
    cfg: WrapperConfig,
    reference_text: str = "",
    mode: str = "continuation",
    language: str = "EN",
    steps: int = 10,
    guidance_scale: float = 1.2,
    seed: int = -1,
    max_seconds: float = 25,
    stop_event: threading.Event | None = None,
    on_progress: Callable[[str], None] | None = None,
    output: str | Path | None = None,
) -> VoiceCloneResult:
    text = str(text or "").strip()
    reference_text = str(reference_text or "").strip()
    if not text:
        raise VoiceCloneError("Enter the words to speak.")
    if len(text) > 1500:
        raise VoiceCloneError("Use a shorter passage (up to 1,500 characters) for one clip.")
    if mode not in {"continuation", "voice_only"}:
        raise VoiceCloneError("Unknown clone mode.")
    if mode == "continuation" and not reference_text:
        raise VoiceCloneError("Best match needs the exact transcript of the reference clip.")
    if mode == "voice_only":
        reference_text = ""
    if not reference_audio:
        raise VoiceCloneError("Upload a short reference recording.")
    reference = Path(reference_audio).expanduser().resolve()
    if not reference.is_file():
        raise VoiceCloneError(f"Reference recording not found: {reference}")
    if reference.stat().st_size > 50 * 1024 * 1024:
        raise VoiceCloneError("Use a short reference recording smaller than 50 MB.")
    if not 10 <= int(steps) <= 32:
        raise VoiceCloneError("Quality steps must be between 10 and 32.")
    guidance_scale = float(guidance_scale)
    if not math.isfinite(guidance_scale) or not 0 <= guidance_scale <= 3:
        raise VoiceCloneError("CFG must be between 0 and 3.")
    if not math.isfinite(float(max_seconds)) or not 3 <= float(max_seconds) <= 30:
        raise VoiceCloneError("Clip limit must be between 3 and 30 seconds.")
    seed = int(seed)
    if seed < 0:
        seed = secrets.randbelow(2**31)
    if seed >= 2**32:
        raise VoiceCloneError("Seed must be below 4,294,967,296.")
    explicit_output = Path(output).expanduser().resolve() if output else None
    if explicit_output is not None:
        if explicit_output.suffix.lower() != ".wav":
            raise VoiceCloneError("Output must be a WAV file.")
        if explicit_output.exists():
            raise VoiceCloneError(f"Output already exists: {explicit_output}")
    python, model, worker = runtime_paths(cfg)
    status = installation_status(cfg)
    if not status.startswith("Ready"):
        raise VoiceCloneError(status)
    progress = on_progress or (lambda message: None)
    stop = stop_event or threading.Event()
    if stop.is_set():
        raise VoiceCloneError("Voice clone stopped.")
    prepare_gpu(cfg, progress)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = cfg.paths.output / "voice-clone" / f"{stamp}-{secrets.token_hex(4)}"
    run_dir.mkdir(parents=True, exist_ok=False)
    audio = explicit_output if explicit_output is not None else run_dir / "voice.wav"
    audio.parent.mkdir(parents=True, exist_ok=True)
    saved_reference = run_dir / f"reference{reference.suffix.lower()}"
    shutil.copy2(reference, saved_reference)
    request = {
        "model": str(model.resolve()), "text": text,
        "reference_audio": str(saved_reference.resolve()), "reference_text": reference_text,
        "mode": mode, "language": language, "steps": int(steps), "seed": seed,
        "guidance_scale": guidance_scale,
        "max_seconds": float(max_seconds), "output": str(audio.resolve()),
    }
    request_path = run_dir / "request.json"
    request_path.write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")
    log_path = run_dir / "worker.log"
    env = os.environ.copy()
    env.update({
        "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1",
        "HF_HOME": str(project_root() / ".local/tts/hf-cache"),
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    })
    mailbox: queue.Queue[str | None] = queue.Queue()
    done = None
    error = None
    start = time.monotonic()
    process = subprocess.Popen(
        [str(python), "-u", str(worker), "--request", str(request_path.resolve())],
        cwd=project_root(), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )

    def read_output() -> None:
        try:
            for line in process.stdout:
                mailbox.put(line)
        finally:
            mailbox.put(None)

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    try:
        with log_path.open("w", encoding="utf-8") as log:
            while True:
                if stop.is_set():
                    raise VoiceCloneError("Voice clone stopped. GPU memory has been released.")
                if time.monotonic() - start > 1800:
                    raise VoiceCloneError("Voice cloning exceeded the 30-minute limit.")
                try:
                    line = mailbox.get(timeout=0.5)
                except queue.Empty:
                    continue
                if line is None:
                    break
                log.write(line)
                log.flush()
                if not line.startswith(EVENT_PREFIX):
                    continue
                event = json.loads(line[len(EVENT_PREFIX):])
                if event.get("event") == "status":
                    progress(str(event["message"]))
                elif event.get("event") == "done":
                    done = event
                elif event.get("event") == "error":
                    error = str(event["message"])
        process.wait(timeout=10)
        if process.returncode != 0 or done is None:
            raise VoiceCloneError(f"{error or 'Voice worker failed.'}\nLog: {log_path}")
        if not audio.is_file():
            raise VoiceCloneError(f"Voice worker did not save the audio. Log: {log_path}")
        with wave.open(str(audio), "rb") as wav:
            sample_rate = wav.getframerate()
            duration = wav.getnframes() / sample_rate
            if sample_rate != 48000 or wav.getnchannels() != 1 or duration < 0.1:
                raise VoiceCloneError("Voice worker produced an invalid audio file.")
        metadata = run_dir / "generation.json"
        record = {
            **request, **{key: value for key, value in done.items() if key != "event"},
            "model_id": MODEL_ID, "duration": duration, "sample_rate": sample_rate,
            "elapsed_seconds": round(time.monotonic() - start, 3),
            "log": str(log_path.resolve()),
        }
        metadata.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return VoiceCloneResult(
            audio=audio.resolve(), metadata=metadata.resolve(), log=log_path.resolve(),
            seed=seed, duration=duration, sample_rate=sample_rate,
            peak_vram_gb=float(done.get("peak_vram_gb", 0)),
        )
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        reader.join(timeout=2)
        process.stdout.close()
