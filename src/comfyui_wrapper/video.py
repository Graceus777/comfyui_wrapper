"""MiniMax H3 image-to-video loops (same graph as sd-comic-ext ambient loops)."""

from __future__ import annotations

import random
import hashlib
import json
import math
import shutil
import subprocess
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Callable

from comfyui_wrapper.client import ComfyClient, ComfyError
from comfyui_wrapper.config import WrapperConfig, load_config
from comfyui_wrapper.generate import GenerateResult, _save_locally
from comfyui_wrapper.h3 import (
    CHAR_LORA,
    CHAR_SAMPLER,
    CHAR_STEPS,
    CHAR_UNET,
    CHAR_VAE,
    HEAD_SWAP_UNET,
    RECIPES,
    apply_character_swap,
    apply_h3_graph,
    apply_h3_latent_upscale,
    apply_head_swap,
    apply_reference_recipe,
    character_swap_prompt,
    clear_first_frame,
    coerce_video_stack,
    detect_upscale_kind,
    ensure_save_latent,
    finish_swap_video,
    h3_frame_count,
    h3_trunk,
    head_swap_prompt,
    is_character_swap_mode,
    is_head_swap_mode,
    latent_upscale_graph,
    normalize_ref_image_size,
    normalize_video_mode,
    pick_model_name,
    reference_video_prompt,
    recipe_settings,
    rtx_vsr_graph,
    set_last_frame,
    set_r2v_refs,
    set_reference_frame_guides,
    sibling_latent_pair,
    video_model_choices,
    video_prompt_default,
)
from comfyui_wrapper.history import append_history, compute_generation_hash
from comfyui_wrapper.models import infer_comfyui_root, list_checkpoints, list_loras, list_pdd_acc, list_text_encoders, list_unets, list_vaes
from comfyui_wrapper.workflow import apply_settings, dump_workflow, load_workflow, merge_bindings

# MiniMax H3 sizes are a 32px grid. "Size from image" keeps the still's aspect
# and fits ~0.52 MP (H3 Studio "balanced" 960×544). Native max is 1344×768.
H3_ALIGN = 32
H3_SIZE_MP = 0.52
H3_MAX_MP = 1.03
H3_MIN_MP = 0.15


def align_h3(value: int | float) -> int:
    return max(H3_ALIGN, int(round(float(value) / H3_ALIGN) * H3_ALIGN))


def h3_size(
    width: int | float,
    height: int | float,
    *,
    megapixels: float | None = None,
    fit: bool = True,
) -> tuple[int, int]:
    """Return an H3-legal (width, height).

    ``fit=True`` (Size from image) preserves aspect and fills ``megapixels``.
    ``fit=False`` only snaps the given canvas onto the 32px grid.
    """
    src_w = max(1, int(width))
    src_h = max(1, int(height))
    if not fit:
        return align_h3(src_w), align_h3(src_h)
    ratio = src_w / src_h
    mp = H3_SIZE_MP if megapixels is None else float(megapixels)
    if not (mp > 0):
        mp = H3_SIZE_MP
    mp = min(H3_MAX_MP, max(H3_MIN_MP, mp))
    target = mp * 1_000_000
    out_w = align_h3((target * ratio) ** 0.5)
    out_h = align_h3((target / ratio) ** 0.5)
    area = out_w * out_h
    cap = H3_MAX_MP * 1_000_000
    if area > cap:
        scale = (cap / area) ** 0.5
        out_w = max(H3_ALIGN, int(out_w * scale) // H3_ALIGN * H3_ALIGN)
        out_h = max(H3_ALIGN, int(out_h * scale) // H3_ALIGN * H3_ALIGN)
    return out_w, out_h


def target_dimensions(
    source: str | Path,
    *,
    megapixels: float | None = None,
) -> tuple[int, int]:
    """Derive H3 ImageScale + latent size from the still's aspect ratio."""
    try:
        from PIL import Image
    except ImportError:
        return h3_size(832, 480, megapixels=megapixels)
    with Image.open(source) as image:
        return h3_size(image.width, image.height, megapixels=megapixels)


def video_frame_size(source: str | Path) -> tuple[int, int] | None:
    """Read the first video stream's pixel size. None when ffprobe cannot."""
    probe = shutil.which("ffprobe")
    if not probe:
        return None
    try:
        result = subprocess.run(
            [
                probe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "csv=p=0:s=x",
                str(source),
            ],
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode:
        return None
    line = (result.stdout or "").strip().splitlines()
    if not line or "x" not in line[0]:
        return None
    width_s, _, height_s = line[0].partition("x")
    try:
        width, height = int(width_s), int(height_s)
    except ValueError:
        return None
    if width < 1 or height < 1:
        return None
    return width, height


def ffmpeg_bin() -> str | None:
    return shutil.which("ffmpeg")


def _run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode:
        raise ComfyError(result.stderr.strip() or result.stdout.strip() or "ffmpeg failed")


def video_frame_count(source: str | Path) -> int:
    probe = shutil.which("ffprobe")
    if not probe:
        raise ComfyError("ffprobe is required to preserve source-video timing")
    result = subprocess.run(
        [probe, "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=nb_frames", "-of", "json", str(source)],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode:
        raise ComfyError(result.stderr.strip() or "Could not read source-video frames")
    streams = json.loads(result.stdout).get("streams", [])
    if not streams or not str(streams[0].get("nb_frames", "")).isdigit():
        raise ComfyError("Prepared source video has no readable frame count")
    return int(streams[0]["nb_frames"])


def prepare_swap_video(
    source: str | Path,
    destination: str | Path,
    *,
    width: int,
    height: int,
    start_seconds: float = 0.0,
    end_seconds: float = 0.0,
) -> tuple[int, int]:
    """Resample a source trim to 24 fps, then pad its last frame for H3."""
    ff = ffmpeg_bin()
    if not ff:
        raise ComfyError("ffmpeg is required to preserve source-video timing")
    source, destination = Path(source), Path(destination)
    start, end = float(start_seconds), float(end_seconds)
    if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end < 0:
        raise ValueError("Source start and end must be finite, non-negative seconds")
    if end and end <= start:
        raise ValueError("Source end must be after source start (or 0 for the whole clip)")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="h3source_") as tmp:
        normalized = Path(tmp) / "normalized.mp4"
        command = [ff, "-y", "-hide_banner", "-loglevel", "error", "-ss", str(start), "-i", str(source)]
        if end:
            command += ["-t", str(end - start)]
        command += [
            "-map", "0:v:0", "-map", "0:a:0?",
            "-vf", f"fps=24,scale={int(width)}:{int(height)}:force_original_aspect_ratio=increase:flags=lanczos,crop={int(width)}:{int(height)},setsar=1,setpts=PTS-STARTPTS",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-movflags", "+faststart", str(normalized),
        ]
        _run(command)
        delivered_frames = video_frame_count(normalized)
        if delivered_frames < 1:
            raise ValueError("The selected source trim contains no frames")
        sampling_frames = h3_frame_count(delivered_frames)
        if sampling_frames == delivered_frames:
            shutil.copy2(normalized, destination)
        else:
            _run([
                ff, "-y", "-hide_banner", "-loglevel", "error", "-i", str(normalized),
                "-map", "0:v:0", "-map", "0:a:0?",
                "-vf", "tpad=stop_mode=clone:stop=-1",
                "-frames:v", str(sampling_frames),
                "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
                "-c:a", "copy", "-movflags", "+faststart", str(destination),
            ])
        if video_frame_count(destination) != sampling_frames:
            raise ComfyError("Prepared source frame count does not match H3's sampling duration")
        return delivered_frames, sampling_frames


def cyclic_crossfade(
    raw_path: str | Path,
    final_path: str | Path,
    *,
    fps: int = 24,
    crossfade_frames: int = 12,
) -> Path:
    """Close a loop: middle = frames[N:-N], seam = fade(last N → first N)."""
    raw_path = Path(raw_path)
    final_path = Path(final_path)
    ff = ffmpeg_bin()
    if not ff:
        raise ComfyError("ffmpeg is not on PATH; cannot close the loop")
    cf = max(1, int(crossfade_frames))
    fade_s = cf / float(fps)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="h3loop_") as tmp:
        tdir = Path(tmp)
        head = tdir / "head.mp4"
        tail = tdir / "tail.mp4"
        mid = tdir / "mid.mp4"
        seam = tdir / "seam.mp4"
        concat = tdir / "list.txt"
        _run([
            ff, "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(raw_path),
            "-vf", f"select='lte(n\\,{cf - 1})',setpts=PTS-STARTPTS",
            "-an", str(head),
        ])
        _run([
            ff, "-y", "-hide_banner", "-loglevel", "error",
            "-sseof", f"-{fade_s:.4f}",
            "-i", str(raw_path),
            "-an", str(tail),
        ])
        _run([
            ff, "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(raw_path),
            "-vf", f"select='gte(n\\,{cf})',setpts=PTS-STARTPTS",
            "-an", str(mid),
        ])
        # Drop the last N frames of `mid` (they become the fade-from side).
        mid_cut = tdir / "mid_cut.mp4"
        _run([
            ff, "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(mid),
            "-vf", f"reverse,select='gte(n\\,{cf})',reverse,setpts=PTS-STARTPTS",
            "-an", str(mid_cut),
        ])
        _run([
            ff, "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(tail), "-i", str(head),
            "-filter_complex", f"[0][1]xfade=transition=fade:duration={fade_s:.4f}:offset=0[v]",
            "-map", "[v]", "-an", str(seam),
        ])
        concat.write_text(
            f"file '{mid_cut.resolve().as_posix()}'\nfile '{seam.resolve().as_posix()}'\n",
            encoding="utf-8",
        )
        silent = tdir / "silent.mp4"
        _run([
            ff, "-y", "-hide_banner", "-loglevel", "error",
            "-f", "concat", "-safe", "0", "-i", str(concat),
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-pix_fmt", "yuv420p", "-an",
            str(silent),
        ])
        # Keep native H3 audio from the raw mux, trimmed to the looped picture.
        mux = subprocess.run(
            [
                ff, "-y", "-hide_banner", "-loglevel", "error",
                "-i", str(silent), "-i", str(raw_path),
                "-map", "0:v:0", "-map", "1:a:0?",
                "-c:v", "copy", "-c:a", "aac", "-shortest",
                "-movflags", "+faststart",
                str(final_path),
            ],
            capture_output=True,
            text=True,
        )
        if mux.returncode:
            silent.replace(final_path)
    return final_path


def stage_input_video(
    source: str | Path,
    *,
    client: ComfyClient,
    cfg: WrapperConfig,
    subfolder: str = "wrapper/rtx",
) -> str:
    """Put a local mp4 where LoadVideo can see it; return the Comfy input key."""
    source = Path(source)
    try:
        return client.upload_image(source, subfolder)
    except ComfyError:
        root = infer_comfyui_root(cfg)
        if root is None:
            raise
        dest_dir = root / "input" / subfolder
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / source.name
        shutil.copy2(source, dest)
        return f"{subfolder}/{dest.name}".replace("\\", "/")


def stage_latent_file(
    source: str | Path,
    *,
    client: ComfyClient,
    cfg: WrapperConfig,
    subfolder: str = "wrapper/latents",
) -> str:
    """Put a local .latent where LoadLatent can see it; return the Comfy input key.

    ``/upload/image`` accepts raw bytes (no extension check), so the normal
    upload path works for latents. Falls back to a direct copy into
    ``<comfyui>/input/<subfolder>/`` when the server is reachable but upload
    fails (mirrors :func:`stage_input_video`).
    """
    source = Path(source)
    if not source.is_file():
        # Already a ComfyUI input key (e.g. from a previous upload)?
        text = str(source).strip()
        if text and Path(text).suffix.lower() in {".latent", ".safetensors"}:
            return text.replace("\\", "/")
        raise ComfyError(f"Latent not found: {source}")
    suffix = source.suffix.lower()
    if suffix not in {".latent", ".safetensors"}:
        raise ValueError(
            f"Not a latent file: {source.name} (expected .latent, got {suffix or 'no suffix'})"
        )
    try:
        return client.upload_image(source, subfolder)
    except ComfyError:
        root = infer_comfyui_root(cfg)
        if root is None:
            raise
        dest_dir = root / "input" / subfolder
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / source.name
        shutil.copy2(source, dest)
        return f"{subfolder}/{dest.name}".replace("\\", "/")


def upscale_latent_file(
    source: str | Path,
    destination: str | Path,
    *,
    audio_source: str | Path | None = None,
    client: ComfyClient,
    cfg: WrapperConfig,
    model_name: str | None = None,
    scale: float | None = None,
    fps: float | None = None,
    vae: str | None = None,
    audio_vae: str | None = None,
    filename_prefix: str = "h3_latent_up",
    on_progress: Callable[[dict], None] | None = None,
) -> Path:
    """Decode-only 3D 2x upscale of saved split latents; copy the result locally.

    ``source`` is the dropped ``*_video.latent`` (or any single ``.latent``).
    The ``*_audio.latent`` sibling in the same folder is picked up
    automatically unless ``audio_source`` is given explicitly. Without an
    audio half the upscaled clip is silent.
    """
    source_p = Path(source)
    destination = Path(destination)
    vid = cfg.video
    explicit_audio = audio_source is not None and str(audio_source).strip() != ""
    if explicit_audio:
        video_p, audio_p = source_p, Path(str(audio_source))
        if not audio_p.is_file():
            raise ComfyError(f"Audio latent not found: {audio_p}")
    else:
        video_p, audio_p = sibling_latent_pair(source_p)
        if video_p is None:
            raise ComfyError(f"Latent not found: {source_p}")
    key = stage_latent_file(video_p, client=client, cfg=cfg)
    audio_key: str | None = None
    if audio_p is not None and Path(audio_p).is_file():
        try:
            audio_key = stage_latent_file(audio_p, client=client, cfg=cfg)
        except (ComfyError, ValueError) as exc:
            if explicit_audio:
                raise ComfyError(f"Could not stage audio latent {audio_p.name}: {exc}") from exc
            audio_key = None
    wf = latent_upscale_graph(
        key,
        filename_prefix,
        audio_latent_key=audio_key,
        vae=vae or vid.vae,
        audio_vae=audio_vae or vid.audio_vae,
        model_name=model_name or vid.latent_upscale_model,
        scale=float(scale if scale is not None else vid.latent_upscale_scale),
        fps=float(fps if fps is not None else vid.fps),
    )
    prompt_id = client.queue_prompt(wf)
    result = client.wait(prompt_id, on_progress=on_progress)
    saved = _save_locally(client, result.images, destination.parent, Path(filename_prefix).name)
    if not saved:
        raise ComfyError("Latent upscale completed without a video output")
    produced = Path(saved[0])
    if produced.resolve() != destination.resolve():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(produced, destination)
        if produced.parent == destination.parent and produced.name != destination.name:
            try:
                produced.unlink()
            except OSError:
                pass
    return destination


def upscale_dropped_file(
    source: str | Path,
    destination: str | Path | None = None,
    *,
    mode: str | None = None,
    audio_source: str | Path | None = None,
    client: ComfyClient,
    cfg: WrapperConfig,
    on_progress: Callable[[dict], None] | None = None,
) -> tuple[Path, str]:
    """Upscale one drag-and-dropped file; ``mode`` is auto/latent/rtx.

    Returns ``(output_path, kind)`` where kind is ``latent`` (3D 2x
    decode-only) or ``clip`` (RTX VSR 2x post pass). ``auto`` picks from the
    file suffix so a single radio/check in the UI is enough.

    ``audio_source`` is the explicit ``*_audio.latent`` half. The WebUI
    uploads to an isolated temp dir, so the sibling auto-pickup in
    :func:`upscale_latent_file` cannot see the audio half unless both files
    are dropped — pass it explicitly to keep audio in the upscaled clip.
    """
    source_p = Path(source)
    want = str(mode or "auto").strip().lower()
    if want in {"", "auto", "detect", "single"}:
        kind = detect_upscale_kind(source_p)
    elif want in {"latent", "3d", "3d_latent", "latent_upscale", "h3"}:
        kind = "latent"
    elif want in {"rtx", "rtx_vsr", "clip", "video"}:
        kind = "clip"
    else:
        raise ValueError(f"Unknown upscale mode {mode!r} (expected auto, latent, or rtx)")
    vid = cfg.video
    if destination is None:
        stem = source_p.stem
        if kind == "latent":
            destination = source_p.with_name(f"{stem}__latent2x.mp4")
        else:
            destination = source_p.with_name(f"{stem}__rtx.mp4")
    else:
        destination = Path(destination)
    if kind == "latent":
        out = upscale_latent_file(
            source_p,
            destination,
            audio_source=audio_source,
            client=client,
            cfg=cfg,
            model_name=vid.latent_upscale_model,
            scale=vid.latent_upscale_scale,
            fps=vid.fps,
            vae=vid.vae,
            audio_vae=vid.audio_vae,
            filename_prefix=f"{destination.stem or 'h3_latent_up'}",
            on_progress=on_progress,
        )
    else:
        out = apply_rtx_vsr(
            source_p,
            destination,
            client=client,
            cfg=cfg,
            scale=vid.rtx_vsr_scale,
            quality=vid.rtx_vsr_quality,
            filename_prefix=f"{destination.stem or 'h3_loop_rtx'}",
            on_progress=on_progress,
        )
    return out, kind


def apply_rtx_vsr(
    source: str | Path,
    destination: str | Path,
    *,
    client: ComfyClient,
    cfg: WrapperConfig,
    scale: float = 2.0,
    quality: str = "ULTRA",
    filename_prefix: str = "h3_loop_rtx",
    on_progress: Callable[[dict], None] | None = None,
) -> Path:
    """Queue RTXVideoSuperResolution on a saved clip and copy the result locally."""
    source = Path(source)
    destination = Path(destination)
    key = stage_input_video(source, client=client, cfg=cfg)
    wf = rtx_vsr_graph(key, filename_prefix, scale=scale, quality=quality)
    prompt_id = client.queue_prompt(wf)
    result = client.wait(prompt_id, on_progress=on_progress)
    saved = _save_locally(client, result.images, destination.parent, Path(filename_prefix).name)
    if not saved:
        raise ComfyError("RTX VSR completed without a video output")
    produced = Path(saved[0])
    if produced.resolve() != destination.resolve():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(produced, destination)
        if produced.parent == destination.parent and produced.name != destination.name:
            try:
                produced.unlink()
            except OSError:
                pass
    return destination


def generate_video(
    prompt: str | None = None,
    *,
    image: str | Path | None = None,
    last_frame: str | Path | None = None,
    mode: str | None = None,
    ref_images: list[str | Path] | None = None,
    ref_videos: list[str | Path] | None = None,
    ref_audios: list[str | Path] | None = None,
    ref_image_size: str | None = None,
    include_reference_video_audio: bool | None = None,
    seed: int | None = None,
    steps: int | None = None,
    width: int | None = None,
    height: int | None = None,
    frames: int | None = None,
    fps: int | None = None,
    sampler: str | None = None,
    scheduler: str | None = None,
    unet: str | None = None,
    clip: str | None = None,
    vae: str | None = None,
    audio_vae: str | None = None,
    recipe: str | None = None,
    acc_file: str | None = None,
    loop: bool | None = None,
    crossfade_frames: int | None = None,
    size_megapixels: float | None = None,
    latent_upscale: bool | None = None,
    rtx_vsr: bool | None = None,
    save_latents: bool | None = None,
    sam_target: str | None = None,
    sam3_checkpoint: str | None = None,
    source_start: float | None = None,
    source_end: float | None = None,
    reference_description: str | None = None,
    filename_prefix: str | None = None,
    cfg: WrapperConfig | None = None,
    config: WrapperConfig | str | Path | None = None,
    client: ComfyClient | None = None,
    wait: bool = True,
    dump_path: str | Path | None = None,
    on_progress: Callable[[dict], None] | None = None,
) -> GenerateResult:
    """Queue a matched H3 workflow: anchored FL2VA, Ref2VA, or a swap preset.

    R2V keeps semantic appearance references separate from timeline anchors.
    ``r2v_frames`` uses Add Guide nodes for optional start/end frames.

    Head swap loads the regular w4a8 UNET and separate rank-32 face LoRA,
    then PDD Acc. The face still is ``<Picture 1>``. The source clip is an
    aligned ``MiniMaxH3AddGuide`` at frame 0 (``<Video 1>``).

    Character swap tracks a SAM3 phrase, inverts colors inside that mask,
    and feeds the treated frames to ``MiniMaxH3ReferenceToVideo`` as
    ``<Video 1>``. The replacement still is ``<Picture 1>``. That mode loads
    Ref2VA int8 plus the Studio V4 turbo LoRA. It does not use the FL2VA
    UNET or PDD Acc.

    Swap sources are trimmed in seconds and resampled to 24 fps. Their final
    frame is padded for H3, then removed from the output; source audio is kept.
    An explicit ``frames`` without ``source_end`` selects that many 24-fps frames.
    """
    if cfg is None:
        if isinstance(config, WrapperConfig):
            cfg = config
        else:
            cfg = load_config(config)
    vid = cfg.video
    used_seed = cfg.generation.seed if seed is None else seed
    if used_seed is None or int(used_seed) < 0:
        used_seed = random.randint(0, 2**32 - 1)
    used_seed = int(used_seed)

    ref_img_list = [ref_images] if isinstance(ref_images, (str, Path)) else list(ref_images or [])
    ref_vid_list = [ref_videos] if isinstance(ref_videos, (str, Path)) else list(ref_videos or [])
    ref_aud_list = [ref_audios] if isinstance(ref_audios, (str, Path)) else list(ref_audios or [])
    raw_mode = normalize_video_mode(mode if mode is not None else vid.mode)
    if mode is None and raw_mode == "i2v" and (ref_img_list or ref_vid_list or ref_aud_list):
        raw_mode = "r2v_frames" if last_frame else "r2v"
    use_char = is_character_swap_mode(raw_mode)
    use_head = is_head_swap_mode(raw_mode)
    use_r2v = raw_mode in {"r2v", "r2v_frames"}
    use_guides = raw_mode == "r2v_frames"
    use_text = raw_mode == "t2v"
    requested_prompt = str(prompt if prompt is not None else video_prompt_default(vid, raw_mode)).strip()
    if (use_r2v or use_text) and not requested_prompt:
        raise ValueError("Describe the scene and action to generate")
    if raw_mode == "i2v" and (ref_img_list or ref_vid_list or ref_aud_list):
        raise ValueError("Appearance/motion/audio references belong in the Reference workflow, not Animate a still")
    if use_text and (image or last_frame or ref_img_list or ref_vid_list or ref_aud_list):
        raise ValueError("Text-to-video does not use images or references; select the matching image/reference workflow")
    if (use_head or use_char) and (last_frame or ref_aud_list or len(ref_vid_list) != 1):
        raise ValueError("Swap needs exactly one source video, appearance images, and no end frame or separate audio reference")
    face_src = None
    char_src = None
    body_src = None
    sam_target_n = ""
    sam3_ckpt_n = ""
    if use_head:
        face_src = image or (ref_img_list[0] if ref_img_list else None)
        body_src = ref_vid_list[0] if ref_vid_list else None
        if face_src is None:
            raise ValueError("Head swap needs a face still (--image or --ref-image) for <Picture 1>")
        if body_src is None:
            raise ValueError("Head swap needs a source clip (--ref-video) for <Video 1>")
    if use_char:
        char_src = image or (ref_img_list[0] if ref_img_list else None)
        body_src = ref_vid_list[0] if ref_vid_list else None
        sam_target_n = str(sam_target if sam_target is not None else vid.sam3_target or "").strip()
        sam3_ckpt_n = str(
            sam3_checkpoint if sam3_checkpoint is not None else vid.sam3_checkpoint or ""
        ).strip()
        if char_src is None:
            raise ValueError("Character swap needs a character still (--image or --ref-image) for <Picture 1>")
        if body_src is None:
            raise ValueError("Character swap needs a source clip (--ref-video) for <Video 1>")
        if not sam_target_n:
            raise ValueError("Character swap needs a SAM3 target (--sam-target), for example \"the person\"")
        if not sam3_ckpt_n:
            raise ValueError(
                "Character swap needs video.sam3_checkpoint (a SAM3.1 file in models/checkpoints)"
            )
    if use_r2v and not use_guides and image is not None:
        ref_img_list = [image, *ref_img_list]
    if use_r2v and not (ref_img_list or ref_vid_list or ref_aud_list):
        raise ValueError("Reference generation needs an appearance, motion or audio reference; use Text to video for text alone")
    if use_guides and not (image or last_frame):
        raise ValueError("Reference + frame guides needs a start frame or an end frame")
    if not use_guides and use_r2v and last_frame:
        raise ValueError("An end-frame anchor belongs in Reference + start/end frames")
    if raw_mode == "i2v" and not (image or last_frame):
        raise ValueError("Animate a still needs a start frame or an end frame")
    if use_head and len(ref_img_list) > (0 if image else 1):
        raise ValueError("Head swap uses one face picture; use Character swap for multiple appearance references")
    extra_char_refs = ref_img_list if image else ref_img_list[1:]
    if (use_r2v and len(ref_img_list) > 9) or (use_char and len(extra_char_refs) > 8):
        raise ValueError("H3 accepts at most nine appearance reference images in total")
    if use_r2v and (len(ref_vid_list) > 3 or len(ref_aud_list) > 3):
        raise ValueError("H3 accepts at most three reference videos and three reference audio files")
    image_for_size = face_src if use_head else char_src if use_char else (image or last_frame)
    if image_for_size is None and ref_img_list:
        image_for_size = ref_img_list[0]
    mp = float(size_megapixels if size_megapixels is not None else
               (vid.character_swap_size_megapixels if use_char else vid.size_megapixels))
    if not math.isfinite(mp) or mp <= 0:
        raise ValueError("Resolution in megapixels must be a positive finite number")
    if (use_head or use_char) and width is None and height is None:
        dims = video_frame_size(body_src)
        if dims:
            width, height = h3_size(
                dims[0], dims[1],
                megapixels=mp,
            )
        else:
            still = char_src if use_char else face_src
            try:
                width, height = target_dimensions(still, megapixels=mp)
            except Exception:
                width, height = h3_size(vid.width, vid.height, fit=False)
    elif image_for_size is not None and width is None and height is None:
        try:
            width, height = target_dimensions(image_for_size, megapixels=mp)
        except Exception:
            width, height = h3_size(vid.width, vid.height, fit=False)
    else:
        width, height = h3_size(width or vid.width, height or vid.height, fit=False)
    frames_n = h3_frame_count(frames or vid.frames)
    if fps is not None and int(fps) != 24:
        raise ValueError("H3 generates at 24 fps; changing playback fps would change action timing")
    fps_n = 24
    rec = recipe_settings(recipe if recipe is not None else (vid.ref_recipe if use_r2v else vid.recipe))
    if use_r2v and rec.key not in {"ref_native", "ref_turbo4", "ref_turbo8"}:
        if recipe is not None:
            raise ValueError("Ref2VA uses a reference quality/Turbo preset, not an FL2VA Acc preset")
        rec = RECIPES["ref_native"]
    if not use_r2v and rec.key.startswith("ref_") and not use_char:
        raise ValueError("Reference Turbo presets require the Reference workflow")
    if use_head:
        rec = RECIPES["acc"]
    client = client or ComfyClient(cfg=cfg)
    live = client if client.ping() else None
    unet_names = list_unets(cfg, live)
    compatible = video_model_choices(unet_names, raw_mode)
    preferred_unet = (vid.character_swap_unet if use_char else vid.head_swap_unet if use_head
                      else vid.r2v_unet if use_r2v else vid.unet)
    required_trunk = "ref2va" if (use_r2v or use_char) else "fl2va"
    if unet and h3_trunk(unet) != required_trunk:
        raise ValueError(f"This workflow requires {required_trunk.upper()}; the selected model belongs to another family")
    if h3_trunk(preferred_unet) != required_trunk:
        raise ValueError(f"Configure the {required_trunk.upper()} model for this workflow")
    unet_selected = unet or (pick_model_name(compatible, preferred_unet) if compatible else preferred_unet)
    if live and unet_selected not in compatible:
        raise ValueError(f"Required {required_trunk.upper()} model is not installed: {unet_selected}")
    wanted_acc = (acc_file if acc_file is not None else vid.acc_file) if rec.acc else ""
    acc_files = list_pdd_acc(cfg, live)
    vae_files = list_vaes(cfg, live)
    stack = coerce_video_stack(
        unet=unet_selected,
        clip=clip or vid.clip,
        vae=vae or vid.vae,
        audio_vae=audio_vae or vid.audio_vae,
        acc_file=wanted_acc,
        recipe=rec,
        defaults=vid,
        unets=compatible,
        clips=list_text_encoders(cfg, live),
        vaes=vae_files,
        acc_files=acc_files,
    )
    rec = stack["recipe"]
    unet_n = stack["unet"]
    clip_n = stack["clip"]
    vae_n = stack["vae"]
    audio_vae_n = stack["audio_vae"]
    acc_n = stack["acc_file"] if rec.acc else ""
    if use_head:
        unet_n = unet_selected
    if use_char:
        unet_n = unet_selected
        vae_n = str(vid.character_swap_vae or CHAR_VAE).strip() or CHAR_VAE
        rec = RECIPES["native"]
        acc_n = ""
        steps_n = CHAR_STEPS
        sampler_n = CHAR_SAMPLER
        scheduler_n = "simple"
    elif use_r2v:
        steps_n = int(steps if steps is not None else vid.ref_steps) if rec.key == "ref_native" else rec.steps
        sampler_n, scheduler_n = rec.sampler, rec.scheduler
        acc_n = ""
    elif rec.acc:
        steps_n = int(stack["steps"])
        sampler_n = stack["sampler"]
        scheduler_n = stack["scheduler"]
    else:
        steps_n = int(steps if steps is not None else rec.steps)
        sampler_n, scheduler_n = rec.sampler, rec.scheduler
    reference_lora = None
    def installed_name(wanted: str, names: list[str], description: str) -> str:
        leaf = Path(wanted.replace("\\", "/")).name.lower()
        matches = [n for n in names if Path(n.replace("\\", "/")).name.lower() == leaf]
        if not matches:
            raise ValueError(f"{description} is not installed: {wanted}")
        return pick_model_name(matches, wanted)

    if use_r2v and rec.key != "ref_native":
        reference_lora = vid.ref_turbo4_lora if rec.key == "ref_turbo4" else vid.ref_turbo8_lora
        reference_lora = installed_name(reference_lora, list_loras(cfg, live), "The selected reference Turbo LoRA")
    swap_lora = (vid.character_swap_lora or CHAR_LORA) if use_char else vid.head_swap_lora
    if live:
        if use_head or use_char:
            swap_lora = installed_name(swap_lora, list_loras(cfg, live), "The swap preset's LoRA")
        if use_char:
            sam3_ckpt_n = installed_name(sam3_ckpt_n, list_checkpoints(cfg, live), "The character swap SAM3 checkpoint")
            vae_n = installed_name(vae_n, vae_files, "The character swap video VAE")
        if rec.acc:
            acc_n = installed_name(acc_n, acc_files, "The selected Acc weight")
    if steps_n < 1:
        raise ValueError("Quality steps must be positive")
    do_loop = False if (use_head or use_char or use_guides or last_frame) else (vid.loop if loop is None else bool(loop))
    fade_n = int(crossfade_frames if crossfade_frames is not None else vid.crossfade_frames)
    do_latent = vid.latent_upscale if latent_upscale is None else bool(latent_upscale)
    do_rtx = vid.rtx_vsr if rtx_vsr is None else bool(rtx_vsr)
    do_save_latents = vid.save_latents if save_latents is None else bool(save_latents)
    ref_size_n = normalize_ref_image_size(ref_image_size if ref_image_size is not None else vid.ref_image_size)
    ref_video_audio = vid.ref_video_audio if include_reference_video_audio is None else bool(include_reference_video_audio)
    prefix = filename_prefix or (
        "h3_char_swap" if use_char else "h3_head_swap" if use_head else "h3_r2v" if use_r2v else "h3_loop"
    )
    if use_head:
        positive = head_swap_prompt(requested_prompt)
    elif use_char:
        positive = character_swap_prompt(
            requested_prompt,
            sam_target_n,
            reference_description,
            reference_image_count=1 + len(extra_char_refs),
        )
    elif use_r2v:
        positive = reference_video_prompt(requested_prompt,
                                          image_count=len(ref_img_list), reference_description=reference_description,
                                          guided=use_guides)
    else:
        positive = requested_prompt or cfg.generation.positive

    if use_head or use_char:
        fps_n = 24
        start_n = float(vid.swap_source_start if source_start is None else source_start)
        end_n = float(vid.swap_source_end if source_end is None else source_end)
        if frames is not None and source_end is None:
            end_n = start_n + int(frames) / 24
        body_path = Path(body_src)
        if not body_path.is_file():
            comfy_root = infer_comfyui_root(cfg)
            if comfy_root is not None:
                body_path = comfy_root / "input" / body_path
        source_stat = body_path.stat()
        source_key = hashlib.sha256(json.dumps([
            str(body_path.resolve()), source_stat.st_size, source_stat.st_mtime_ns,
            start_n, end_n, width, height,
        ]).encode()).hexdigest()[:20]
        if on_progress:
            on_progress({"type": "executing", "data": {"node": "Preparing source trim at 24 fps"}})
        with tempfile.TemporaryDirectory(prefix="h3swap_") as tmp:
            prepared = Path(tmp) / f"h3_source_{source_key}.mp4"
            delivered_frames, frames_n = prepare_swap_video(
                body_path, prepared, width=width, height=height,
                start_seconds=start_n, end_seconds=end_n,
            )
            body_key = stage_input_video(prepared, client=client, cfg=cfg, subfolder="wrapper/h3")
        image_key = client.upload_image(face_src if use_head else char_src, "wrapper/h3")
        last_key = None
        ref_image_keys = [client.upload_image(p, "wrapper/h3/refs") for p in extra_char_refs] if use_char else []
        ref_video_keys = []
        ref_audio_keys = []
    else:
        body_key = None
        image_key = client.upload_image(image, "wrapper/h3") if image is not None and (not use_r2v or use_guides) else None
        last_key = client.upload_image(last_frame, "wrapper/h3") if last_frame is not None else None
        ref_image_keys = [client.upload_image(p, "wrapper/h3/refs") for p in ref_img_list] if use_r2v else []
        ref_video_keys = []
        if use_r2v:
            for source_ref in ref_vid_list:
                source_path = Path(source_ref)
                if not source_path.is_file():
                    root = infer_comfyui_root(cfg)
                    if root:
                        source_path = root / "input" / source_path
                dims = video_frame_size(source_path)
                rw, rh = h3_size(*(dims or (width, height)), megapixels=mp)
                with tempfile.TemporaryDirectory(prefix="h3reference_") as tmp:
                    prepared = Path(tmp) / f"reference_{uuid.uuid4().hex}.mp4"
                    prepare_swap_video(source_path, prepared, width=rw, height=rh,
                                       end_seconds=frames_n / 24)
                    ref_video_keys.append(stage_input_video(prepared, client=client, cfg=cfg, subfolder="wrapper/h3/refs"))
        ref_audio_keys = (
            [stage_input_video(p, client=client, cfg=cfg) for p in ref_aud_list] if use_r2v else []
        )

    if use_char:
        wf = load_workflow(cfg.character_swap_workflow_path())
    elif use_head:
        wf = load_workflow(cfg.head_swap_workflow_path())
    elif use_r2v:
        wf = load_workflow(cfg.r2v_workflow_path())
    else:
        wf = load_workflow(cfg.video_workflow_path())
    settings = {
        "positive": positive,
        "seed": used_seed,
        "steps": steps_n,
        "sampler": sampler_n,
        "scheduler": scheduler_n,
        "denoise": 1.0,
        "width": width,
        "height": height,
        "length": frames_n,
        "fps": float(fps_n),
        "checkpoint": unet_n,
        "clip": clip_n,
        "vae": vae_n,
        "audio_vae": audio_vae_n,
        "image": image_key,
        "ref_image_size": ref_size_n,
        "filename_prefix": prefix,
    }
    wf = apply_settings(wf, settings, bindings=merge_bindings(wf, cfg.workflow_map))
    if not use_r2v and not use_head and not use_char and not image_key:
        wf = clear_first_frame(wf)
    if use_head:
        wf = apply_head_swap(
            wf,
            face_image=image_key,
            source_video=body_key,
            length=frames_n,
            unet_name=unet_n,
            lora_name=swap_lora,
            lora_strength=vid.head_swap_strength,
        )
    if not use_r2v and not use_head and not use_char and last_key:
        wf = set_last_frame(wf, last_key)
    if use_r2v:
        wf = set_r2v_refs(
            wf,
            ref_images=ref_image_keys,
            ref_videos=ref_video_keys,
            ref_audios=ref_audio_keys,
            ref_image_size=ref_size_n,
            include_video_audio=ref_video_audio,
        )
        if use_guides:
            wf = set_reference_frame_guides(wf, first_frame=image_key, last_frame=last_key)
        wf = apply_reference_recipe(wf, recipe=rec, lora_name=reference_lora, steps=steps_n)
    else:
        wf = apply_h3_graph(
            wf, recipe="native" if use_char else rec,
            acc_file=None if use_char else (acc_n if rec.acc else None), unet=unet_n, steps=steps_n,
        )
    if use_char:
        wf = apply_character_swap(
            wf,
            character_image=image_key,
            source_video=body_key,
            length=frames_n,
            sam_target=sam_target_n,
            sam3_checkpoint=sam3_ckpt_n,
            unet_name=unet_n,
            lora_name=swap_lora,
            lora_strength=vid.character_swap_lora_strength,
            vae_name=vae_n,
            shift_video=vid.character_swap_shift_video,
            steps=steps_n,
            reference_images=ref_image_keys,
        )
    if do_latent:
        wf = apply_h3_latent_upscale(
            wf,
            model_name=vid.latent_upscale_model,
            scale=vid.latent_upscale_scale,
        )
    if do_save_latents:
        wf = ensure_save_latent(wf, f"{prefix}_latent")
    if use_head or use_char:
        wf = finish_swap_video(wf, frames=delivered_frames)
    if dump_path:
        dump_workflow(wf, dump_path)

    prompt_id = client.queue_prompt(wf)
    images: list[dict] = []
    saved: list[str] = []
    loop_error = ""
    if wait:
        result = client.wait(prompt_id, on_progress=on_progress)
        images = [artifact for artifact in result.images if artifact.get("type") != "input"]
        if (cfg.batch.save_locally or do_rtx or do_save_latents) and images:
            saved = _save_locally(client, images, cfg.paths.output, prefix)

        def _first_video(paths: list[str]) -> Path | None:
            for p in paths:
                if Path(p).suffix.lower() in {".mp4", ".webm", ".mov", ".mkv"}:
                    return Path(p)
            return None

        if do_loop and saved:
            raw = _first_video(saved)
            if raw is not None:
                looped = raw.with_name(raw.stem + "_loop" + raw.suffix)
                try:
                    cyclic_crossfade(raw, looped, fps=fps_n, crossfade_frames=fade_n)
                    saved.append(str(looped))
                except Exception as exc:
                    loop_error = str(exc)
        if do_rtx and saved:
            source = _first_video(saved)
            if source is not None:
                rtx_dest = source.with_name(source.stem + "__rtx" + source.suffix)
                try:
                    if on_progress:
                        on_progress({"type": "executing", "data": {"node": "rtx_vsr"}})
                    apply_rtx_vsr(
                        source,
                        rtx_dest,
                        client=client,
                        cfg=cfg,
                        scale=vid.rtx_vsr_scale,
                        quality=vid.rtx_vsr_quality,
                        filename_prefix=prefix + "_rtx",
                        on_progress=on_progress,
                    )
                    saved.append(str(rtx_dest))
                except Exception as exc:
                    loop_error = (loop_error + "; " if loop_error else "") + f"rtx_vsr: {exc}"
        append_history(
            cfg.paths.history,
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "prompt_id": prompt_id,
                "kind": "video",
                "mode": raw_mode,
                "model": unet_n,
                "recipe": rec.key,
                "acc_file": acc_n if rec.acc else "",
                "steps": steps_n,
                "frames": delivered_frames if (use_head or use_char) else frames_n,
                "sampling_frames": frames_n,
                "fps": fps_n,
                "source_start": start_n if (use_head or use_char) else None,
                "source_end": start_n + delivered_frames / 24 if (use_head or use_char) else None,
                "source_audio": bool(use_head or use_char),
                "ref_images": 1 + len(ref_image_keys) if use_char else 1 if use_head else len(ref_image_keys),
                "ref_videos": 1 if (use_head or use_char) else len(ref_video_keys),
                "ref_audios": len(ref_audio_keys),
                "ref_video_audio": bool(use_r2v and ref_video_keys and ref_video_audio),
                "ref_image_size": ref_size_n if (use_r2v or use_head or use_char) else "",
                "last_frame": bool(last_key),
                "first_frame": bool(image_key) if not (use_head or use_char) else False,
                "frame_guides": bool(use_guides),
                "latent_upscale": do_latent,
                "rtx_vsr": do_rtx,
                "save_latents": do_save_latents,
                "seed": used_seed,
                "prompt_preview": (positive or "")[:200],
                "files": [Path(p).name for p in saved],
                "loop_error": loop_error,
            },
        )

    return GenerateResult(
        prompt_id=prompt_id,
        workflow=wf,
        images=images,
        saved_files=saved,
        seed=used_seed,
        positive=positive or "",
        negative="",
        loras=[],
        gen_hash=compute_generation_hash(
            positive or "",
            "",
            steps=steps_n,
            sampler=sampler_n,
            cfg=1.0,
            width=width,
            height=height,
            checkpoint=unet_n,
        ),
    )
