# Voice Clone

The wrapper's **Voice Clone** tab uses the official
[dots.tts-SOAR](https://huggingface.co/dots-studio/dots.tts-soar) checkpoint.
It runs locally in a separate worker rather than through a ComfyUI custom node.

## Setup

From this repository in PowerShell, with `uv` installed:

```powershell
.\scripts\setup_voice_clone.ps1
python webui.py
```

Setup installs Python 3.12, CUDA 12.8 PyTorch 2.8, and dots.tts 0.3.1 under
`.local/tts/`. It downloads the complete model package, including the vocoder
and speaker encoder, to `models/tts/dots.tts-soar/`. The pinned Hugging Face
revision is `2f9b3e18d70d670d4c701da2dc55ded5755815ce`, totaling
5,164,925,338 bytes. `download-manifest.json` records file sizes and verifies
the SHA-256 hashes supplied for large files. Model and runtime directories
are ignored by Git.

The inference worker does not need the upstream Gradio application or its
optional WeTextProcessing/Pynini text normalization. Setup makes those
normalizer imports lazy for native Windows; inference retains upstream's
default `normalize_text=False`. Spell numbers and abbreviations as you want
them spoken. The wrapper keeps its existing Gradio installation.

Optional `voice_clone.python` and `voice_clone.model_dir` configuration keys
can point to another compatible local runtime or model folder.

## Use

1. Open **Voice Clone** and upload or record one clean speaker for 3–10 seconds.
2. Leave **Best match** selected and enter the exact words in that reference.
3. Enter **Words to speak** and click **Generate cloned voice**.
4. Play or download the WAV. Generation details include the chosen seed.

**Voice only** accepts audio without a transcript and conditions on speaker
features. **Best match** also conditions on the reference audio sequence and
its transcript for stronger voice and speaking-style adherence.

Output is mono, 48 kHz, PCM16 WAV. Reference recordings up to 12 seconds are
accepted. Start with one or two sentences; the output limit is 3–30 seconds.
**Quality steps (NFE)** is the flow sampling step count, adjustable from 10
to 32 (default 10). **CFG (guidance scale)** controls conditioning strength,
adjustable from 0 to 3 in 0.05 increments (default 1.2). Both settings and the
seed are saved in the generation details.

For an electronic-voice experiment, compare CFG 1.2, 1.35, and 1.45 at the
same seed and 10–12 steps. Keep the reference and text identical so you can
hear the effect of the setting. Higher CFG or fewer steps do not guarantee
a closer match to the reference's processing; choose by listening.

The worker loads weights on CPU before placing the inference model on the
GPU, uses BF16 when supported, and uses eager inference on Windows. It exits
after every job, releasing GPU allocations. Before cloning, the wrapper
releases idle ComfyUI caches when reachable locally. It rejects a new clone
while ComfyUI has active or queued jobs. **Stop voice clone** terminates only
the voice worker. Generation reads local model files with network downloads
disabled.

## Command line

```powershell
python -m comfyui_wrapper tts --check
python -m comfyui_wrapper tts --reference C:\audio\reference.wav `
  --reference-text "The words spoken in the reference recording." `
  --text "The new words to synthesize." --seed 4081
```

Use `--voice-only` to omit the transcript, `--steps 12 --guidance-scale 1.35`
to adjust sampling, or `--max-seconds 30` for a longer clip.
`--output` accepts a new WAV path;
existing output files are preserved. Default runs are saved under
`outputs/voice-clone/<run>/` with `voice.wav`, `reference.*`, `request.json`,
`generation.json`, and `worker.log`.

Published benchmark results do not establish cloning quality for every
recording. Assess identity, pronunciation, prosody, and stability with your
own short reference.
