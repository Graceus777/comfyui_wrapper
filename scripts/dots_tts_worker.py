"""One offline voice-cloning request; exit releases the model's GPU memory."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path

PREFIX = "VOICE_CLONE_JSON "


def emit(event: str, **values) -> None:
    print(PREFIX + json.dumps({"event": event, **values}, ensure_ascii=False), flush=True)


def run(request: dict) -> None:
    emit("status", message="Starting the local voice model...")
    import numpy as np
    import soundfile as sf
    import torch
    from dots_tts.runtime import DotsTtsRuntime
    from dots_tts.utils.util import seed_everything

    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the voice environment. Run scripts/setup_voice_clone.ps1.")
    reference = Path(request["reference_audio"])
    info = sf.info(reference)
    ref_seconds = info.frames / info.samplerate
    if not 0.5 <= ref_seconds <= 12:
        raise ValueError("Use a reference between 0.5 and 12 seconds; 3–10 seconds is recommended.")
    if info.channels > 2:
        raise ValueError("Use a mono or stereo reference recording.")
    model_dir = Path(request["model"])
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Local SOAR model not found: {model_dir}")
    seed_everything(int(request["seed"]))
    precision = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
    torch.cuda.reset_peak_memory_stats()
    # One latent patch is 160 ms. Leave headroom so the explicit duration
    # check fires before the model exhausts its schedule and clips a sentence.
    conditioning_seconds = ref_seconds if request["reference_text"] else 0
    max_patches = math.ceil((conditioning_seconds + request["max_seconds"] + 1) / 0.16) + 1
    emit("status", message="Loading SOAR and its vocoder...")
    runtime = DotsTtsRuntime.from_pretrained(
        str(model_dir), precision=precision, optimize=False,
        max_generate_length=max_patches, max_sequence_length=2048,
    )
    emit("status", message="Generating the cloned voice...")
    started = time.monotonic()
    chunks = []
    samples = 0
    for chunk in runtime.generate_stream(
        text=request["text"], prompt_audio_path=str(reference),
        prompt_text=request["reference_text"] or None,
        language=request["language"] or None,
        num_steps=int(request["steps"]),
        guidance_scale=float(request.get("guidance_scale", 1.2)),
        normalize_text=False,
    ):
        chunk = chunk.detach().float().cpu().reshape(-1)
        if not bool(torch.isfinite(chunk).all()):
            raise RuntimeError("The model produced non-finite audio.")
        chunks.append(chunk)
        samples += chunk.numel()
        if samples / runtime.sample_rate > request["max_seconds"] + 0.16:
            raise RuntimeError("The passage exceeded the clip limit. Increase the limit or shorten the text.")
        if len(chunks) % 5 == 0:
            emit("status", message=f"Generating the cloned voice... {samples / runtime.sample_rate:.1f}s of audio")
    if not chunks:
        raise RuntimeError("The model produced no audio.")
    audio = torch.cat(chunks).numpy()
    if float(np.max(np.abs(audio))) < 0.001:
        raise RuntimeError("The model produced silent audio.")
    output = Path(request["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output, audio, runtime.sample_rate, subtype="PCM_16")
    emit(
        "done", sample_rate=runtime.sample_rate, samples=int(audio.size),
        reference_seconds=ref_seconds, precision=precision,
        device=torch.cuda.get_device_name(), synthesis_seconds=round(time.monotonic() - started, 3),
        peak_vram_gb=round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        peak_reserved_vram_gb=round(torch.cuda.max_memory_reserved() / 1024**3, 3),
        rms=float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))),
        peak=float(np.max(np.abs(audio))),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(json.loads(args.request.read_text(encoding="utf-8")))
    except Exception as error:
        emit("error", message=str(error))
        traceback.print_exc(file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
