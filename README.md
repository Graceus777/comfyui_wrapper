# comfyui_wrapper

Config-driven Python client for a running [ComfyUI](https://github.com/comfyanonymous/ComfyUI) server.

Ways in:

1. **WebUI** — A1111-style Gradio app (`python webui.py`): SDXL, Anima, Krea 2, Flux GGUF, MiniMax H3 loops.
2. **`generate`** — one still prompt, every sampler/size/checkpoint setting in `config.yaml`.
3. **`video`** — MiniMax H3 image-to-video (same graph as `sd-comic-ext` ambient loops).
4. **`prep` + `batch`** — combinator-style zone expansion.

Zone and prompt JSON from [sd-combinator-ext](https://github.com/Graceus777/sd-combinator-ext) load as-is. LoRAs are applied as ComfyUI `LoraLoader` nodes, not as `<lora:...>` tags in the prompt.

For the complete capability inventory, see [FEATURES.md](FEATURES.md).

## Setup

ComfyUI must already be running (`http://127.0.0.1:8188` by default).

```powershell
git clone https://github.com/Graceus777/comfyui_wrapper.git
cd comfyui_wrapper
Copy-Item config.example.yaml config.yaml
python -m pip install -e .
# or, without installing:
python -m pip install -r requirements.txt
```

Edit `config.yaml`:

- `comfyui.host` / `port`
- `generation.*` — checkpoint, prompt, steps, cfg, sampler, size, seed
- `workflow.file` — API-format graph (export from ComfyUI with **File → Export (API)**)
- `paths.combinator_root` — points at `sd-combinator-ext` so existing zone/prompt configs and `lora_texts/` resolve without copying
- `paths.webui_root` — A1111 install (`…/sd.webui/webui`). Checkpoints, LoRAs, and VAE are used **from that folder**; nothing is copied into ComfyUI
- `paths.comfyui_root` — ComfyUI tree, so the wrapper can keep `extra_model_paths.yaml` in sync and add junctions if ComfyUI cannot see the A1111 files yet

Confirm the graph bindings:

```powershell
python -m comfyui_wrapper bindings
```

## WebUI

A1111-style Gradio interface: txt2img on the left, gallery + infotext on the right, extra-network LoRAs, combinator batch, and an outputs browser.

ComfyUI must already be running.

```powershell
python -m pip install -e ".[web]"
python webui.py
```

Opens `http://127.0.0.1:7860`. Same thing via the CLI: `python -m comfyui_wrapper ui` or `comfywrap ui`.

For Tailscale access, double-click [`launch_tailscale.bat`](launch_tailscale.bat).
It starts the WebUI with `--listen` on port 8000 and prints your Tailscale URL.
Connect Tailscale and start ComfyUI first, then keep the launcher's window open.
The launcher uses `.venv` or `venv` when present, otherwise Python on your PATH.

**Workflow tools** adds separate panels for H3 memory saver video, the dedicated
character-swap LoRA, OmniChar, guided clip upscale, Qwen 2.1 reference control,
consistent edits, outfit swaps, outpainting, and local Wan keyframe animation.
Each runs its own preset. See [setup, recipes and verification](workflows/containers/README.md).

**Voice Clone** uses local dots.tts-SOAR with a 3–10 second reference recording.
Choose **Best match**, enter the exact reference transcript, and enter the new
words to speak. The tab saves a playable 48 kHz WAV and generation details;
its separate worker releases GPU memory after each job. Setup and CLI usage:
[Voice Clone](docs/voice-clone.md).

| Flag | Meaning |
|------|---------|
| `--port 7861` | WebUI port (default 7860) |
| `--listen` | Bind `0.0.0.0` |
| `--share` | Gradio public tunnel |
| `--no-browser` | Do not open a browser |
| `--host` / `--comfy-port` | Override the ComfyUI server from `config.yaml` |

- **txt2img** — prompt / negative, sampling, size, seed, batch count, **img2img**, **Inpaint**, **Hires. fix**, **Anima LLLite control**, and **ADetailer**. The img2img tab takes an init image (resized to Width x Height, VAE-encoded into the first KSampler; denoise is the strength) and works on SDXL, Anima, Krea 2, and Flux GGUF. The Inpaint tab is A1111-style: drop the init image, paint the mask with the brush (white = repaint), tune denoise / mask expand / mask blur / invert; a mask upload overrides the drawing. Inpaint uses whole-picture `SetLatentNoiseMask`, so no inpaint model is needed. The Anima section accepts a preprocessed depth, pose, or lineart image plus a model patch, strength, and timestep schedule. Extra networks cover LoRA (`<lora:name:weight>` → ComfyUI LoRA nodes), textual inversion (`embedding:name` from A1111 `embeddings/`), and hypernetworks. Checkpoints, LoRAs, VAE, embeddings, and ControlNet are used from `paths.webui_root` without copying.
- **Qwen Image** — Qwen Image 2.1 text-to-image and reference-image edits on the Q6 graph (`qwen_image_2.1-Q6_K.gguf`, `qwen3vl_8b_w4a8.safetensors`, `qwen_image_2.1_vae_bf16.safetensors`). Text-to-image uses Width × Height. Image edit sends up to 10 references into `TextEncodeQwenImage21`; **Match first image** samples at that image's size. Start ComfyUI with `run_qwen_image_2_1_q6_lowvram.bat` on a 12 GB GPU. CLI: `comfywrap qwen --prompt "..."` or `comfywrap qwen --image ref.png --prompt "wear the blue shirt"`.
- **Batch** — combinator zone + prompt config, plus its own steps / CFG / resolution / sampler, and ADetailer. Preview / Prep / Run, or run an existing `configs/jobs/*.json` file.
- **History** — recent files in `paths.output`.

Generation info is printed in A1111 infotext form (`Steps: …, Sampler: Euler a, Seed: …`) so you can recycle the last seed.

## Single prompt

```powershell
python -m comfyui_wrapper generate
python -m comfyui_wrapper generate --prompt "1girl, looking at viewer" --steps 28 --seed 42
python -m comfyui_wrapper generate --lora example_char_a:0.8 --lora "example_style:1"
python -m comfyui_wrapper generate --init-image C:\stills\base.png --denoise 0.6
python -m comfyui_wrapper generate --init-image C:\stills\base.png --inpaint-mask C:\stills\mask.png --inpaint-denoise 0.75
python -m comfyui_wrapper generate --dump-workflow patched.json
```

img2img (`--init-image` + `--denoise`, or `generation.init_image` in `config.yaml`) injects `LoadImage -> ImageScale -> VAEEncode` ahead of the first KSampler and rewires `latent_image`. The init is resized to the run Width x Height, so it covers SDXL, Anima (`qwen_image_vae`), Krea 2, and Flux GGUF without a separate graph. `bindings` lists it as `init_image -> img2img_loader.image`.

Python:

```python
from comfyui_wrapper import generate, load_config

cfg = load_config()
result = generate(
    "1girl, from side, white background",
    negative="low quality, bad anatomy",
    steps=28,
    loras=["example_char_a:0.8"],
    cfg=cfg,
)
print(result.prompt_id, result.saved_files, result.seed)
```

Every field under `generation:` in `config.yaml` is the default; CLI flags and `generate()` kwargs override one run.

## Batch prep and run

Same zone model as sd-combinator-ext:

| Zone | Behavior |
|------|----------|
| **always** | LoRA + activation text on every job |
| **loop** | One job per LoRA (least-used first when history exists) |
| **random** | Pick N per job, weighted toward LoRAs that appear less in history |

Plus prompt **slots** — `{pose}` / `{view}` / anything — expanded as a cartesian product and frozen into the job file.

```powershell
# see what would run, including full prompts
python -m comfyui_wrapper preview --zone example --prompt-config example

# write an editable job list (random picks and skip-exists already applied)
python -m comfyui_wrapper prep --zone example --prompt-config example --out configs/jobs/run.json --show

# run it
python -m comfyui_wrapper batch --jobs configs/jobs/run.json

# or prep+run in one step
python -m comfyui_wrapper batch --zone example --prompt-config example --limit 4
```

Filter, slots, resume:

```powershell
python -m comfyui_wrapper prep --zone example --prompt-config example `
    --slot pose=standing,kneeling --only example_char_a,example_char_b --limit 10

python -m comfyui_wrapper batch --jobs configs/jobs/run.json --resume
```

`prep` is the difference from the A1111 tab: you get a JSON job list with the exact prompt, LoRA files, seed, and `gen_hash` for every image *before* the GPU starts. Edit or delete jobs, then `batch --jobs`.

Skip-exists uses `generation_history.jsonl` (same idea as the extension). Loop order is least-used-first from that history.

## Commands

| Command | Purpose |
|---------|---------|
| `generate` | One prompt |
| `preview` | Print expanded jobs |
| `prep` | Write `configs/jobs/*.json` |
| `batch` | Run a job file, or `--zone` to prep+run |
| `configs` | List zone / prompt names (wrapper + combinator_root) |
| `loras` / `models` | List what ComfyUI can see |
| `bindings` | Show which graph nodes map to prompt/seed/size/... |
| `status` / `interrupt` | Queue / stop |
| `ui` | A1111-style Gradio WebUI |
| `video` | MiniMax H3 I2V loop (`--image still.png`) |

`comfywrap` is installed as a console script if you used `pip install -e .`.

## Workflows

Bundled graphs under `workflows/`:

| File | Stack |
|------|--------|
| `sdxl_txt2img.api.json` | Illustrious / SDXL checkpoint |
| `anima_txt2img.api.json` | Anima UNET + Qwen3 0.6B CLIP (`qwen_3_06b_base.safetensors`) + `qwen_image_vae.safetensors` |
| `krea2_txt2img.api.json` | Krea 2 Turbo UNET + Qwen CLIP |
| `flux_gguf_txt2img.api.json` | Flux dev GGUF + Dual CLIP + `ae.safetensors` |
| `minimax_h3_i2v_loop.api.json` | MiniMax H3 I2V + native audio mux (ambient loop) |

Pick the workflow in the WebUI; model / CLIP / VAE dropdowns list ComfyUI `checkpoints`, `diffusion_models`, `unet` (GGUF), `text_encoders`, and `vae`.

**Anima** is a split graph (not an SDXL checkpoint): UNET from `models/Stable-diffusion` (hardlinked into ComfyUI `models/diffusion_models`), text encoder `qwen_3_06b_base.safetensors` in `models/text_encoders`, VAE `qwen_image_vae.safetensors` in `models/vae`. CLIPLoader type is `stable_diffusion` so ComfyUI auto-detects the Qwen3 0.6B Anima encoder. WAI-ANIMA defaults: Euler a, CFG 4.5, 28 steps, 1024×1344. Turbo checkpoints want CFG 1 and ~8–12 steps.

Anima LLLite uses ComfyUI's built-in `ModelPatchLoader` and
`AnimaLLLiteApply` nodes. Put the LLLite weights in `models/model_patches`,
then pass a preprocessed control image either in `generation.anima_lllite` or
on the CLI:

```powershell
python -m comfyui_wrapper generate `
  --workflow anima_txt2img.api.json `
  --checkpoint anima_turboV11.safetensors `
  --steps 8 --cfg-scale 1 --sampler euler --scheduler simple `
  --anima-control-image C:\maps\p0101-depth.png `
  --anima-lllite-model anima-lllite-depth-1.safetensors `
  --anima-control-strength 0.8
```

`generation.anima_lllite` is a list, so depth, lineart, or pose controls can
be stacked in order. Each item accepts `image`, `model_patch`, `strength`,
`start_percent`, and `end_percent`. The wrapper uploads local image paths;
an existing ComfyUI input key can be supplied instead. Preprocessing is kept
outside the wrapper so API runs consume the exact saved control map.

To add another graph:

1. In ComfyUI: **File → Export (API)**
2. Save under `workflows/`
3. Set `workflow.file` in `config.yaml`
4. Run `python -m comfyui_wrapper bindings` and, if auto-detect misses a node, set `workflow.map`:

```yaml
workflow:
  file: my_flux.api.json
  map:
    positive: "11.text"
    seed: "13.noise_seed"
    steps: "15.steps"
    width: "12.width"
    height: "12.height"
```

UI-format graphs (`nodes` + `links`) are rejected with an error pointing at Export (API).

LoRA injection walks the graph, chains `LoraLoader` (or `LoraLoaderModelOnly` when `generation.lora_mode: model_only`) onto the MODEL/CLIP loaders, and rewires CLIP encode + sampler inputs. Activation text is still concatenated onto the positive prompt.

## Video workflow selection

FL2VA and Ref2VA both accept frame guides. FL2VA specializes in animating the
supplied scene; Ref2VA also accepts separate appearance, motion and audio
references. Ref2VA is not a stronger image-to-video setting: an appearance
reference supplies reusable content rather than forcing the opening pose.

| Workflow in the Video tab | Model | Inputs and roles |
|---|---|---|
| Animate a still / start & end frames | FL2VA | Start and/or end pictures are frames of the generated scene. |
| Reference appearance → new video | Ref2VA | Appearance pictures define the subject; the prompt describes a new scene and action. Optional video/audio references supply additional guidance. |
| Reference appearance + start / end frames | Ref2VA | Appearance pictures are separate from start/end scene guides. |
| Character swap in a source video | Ref2VA + Studio Turbo V4 | Replacement appearance, one source clip, and who to replace. |
| Head swap in a source video | FL2VA + head LoRA + Acc | One replacement face and one source performance. |
| Text → video | FL2VA | Scene/action prompt without reference inputs. |

The dropdown changes labels, required inputs, compatible models and sampling
presets. Switching an appearance input into a frame input clears it; when
entering the combined reference/frame workflow, it is retained as an appearance
reference. Fixed Acc/Turbo/swap steps are hidden. Base quality exposes Steps;
reference Turbo presets appear only when their matching LoRAs are installed.
Resolution controls match the input aspect ratio; disable automatic sizing to
set width and height. Generation uses 24 fps, with duration rounded to H3's
supported frame counts. Loop blending is optional and disabled for frame-guided
references, an end-frame anchor, and swaps.

```powershell
comfywrap video --mode i2v --image start.png --last-frame end.png --prompt "The character walks across the room."
comfywrap video --mode r2v --image character-sheet.png --prompt "The character walks across a moonlit courtyard."
comfywrap video --mode r2v_frames --ref-image character-sheet.png --image opening-scene.png --last-frame ending-scene.png --prompt "The character walks from the opening position to the ending position."
```

`video.r2v_unet`, `video.ref_recipe` and `video.ref_steps` configure reference
generation. `video.reference_prompt`, `video.swap_prompt` and `video.text_prompt`
provide separate prompt defaults; `video.prompt` supplies the still-animation
default. The CLI rejects an explicitly selected incompatible model/workflow.

## Character replacement

Select **Character swap in a source video** in the Video tab. Upload the new
character under **Replacement character appearance**, the performance clip under
**Source video to edit**, and describe **Who to replace in the source video**.
Use source start/end seconds to select the shot; end `0` uses the remaining clip.
The default preset edits the first three seconds at 0.8 MP with Turbo V4 Step600.

Source video is resampled to 24 fps at its original speed. H3 frame padding is
removed from the final video, which retains the selected source audio and
duration. The picture supplies identity; it is not an init frame. Describe the
edit in the scene prompt and put optional appearance details in the separate
character field. A T-pose or character sheet belongs in appearance references,
and does not become an opening frame. Up to eight additional pictures can show
the replacement from other angles; all nine pictures describe the same subject.
The generated prompt follows the source pose and timing from the first frame.

```powershell
comfywrap video --mode charswap --image character.png --ref-video source.mp4 --sam-target "the person" --source-start 1 --source-end 4 --prompt "Replace the person with the character in <Picture 1>. Keep the source performance."
```

This matches the reference-based [Studio workflow](https://github.com/BiggerFishy/comfyui-minimax-h3-studio).
Visual identity and motion adherence remain subject to generation quality.

## Config map

| Key | Role |
|-----|------|
| `comfyui.*` | Host, port, timeouts |
| `generation.*` | Prompt, checkpoint, sampler, size, seed, default LoRAs |
| `generation.lora_mode` | `full` (MODEL+CLIP) or `model_only` |
| `batch.skip_exists` | Drop jobs whose hash is already in history |
| `batch.sample_random_at` | `prep` freezes random-zone picks into the job file |
| `batch.save_locally` | Copy ComfyUI outputs into `paths.output` |
| `paths.combinator_root` | Reuse the A1111 extension's `configs/` and `lora_texts/` |
| `paths.webui_root` | A1111 install; checkpoints, LoRAs, VAE, embeddings, ControlNet used in place (no copy) |
| `paths.comfyui_root` | ComfyUI install, for `extra_model_paths.yaml` + junctions |

## Tests

```powershell
python -m pip install pytest
python -m pytest -q
```

The test suite covers CLI/config parsing, model resolution, workflow binding
and graph injection, H3 video graph wiring, batch preparation, and WebUI
construction. A live GPU render still depends on the installed ComfyUI nodes
and model files; see [FEATURES.md](FEATURES.md) for those operational
boundaries.

## First-run checks and recovery

Use an editable installation from this checkout; bundled API graphs remain in
`workflows/`. The source-only `requirements.txt` alternative also requires
`PYTHONPATH=src` when using `python -m comfyui_wrapper`; the top-level
`webui.py` launcher adds that path itself.

```powershell
comfywrap status
comfywrap models
comfywrap bindings
```

Replace the checkpoint in `config.yaml` and the placeholder LoRA names in
`configs/zones/example.json` with names returned by your server. Start with
SDXL and one image before enabling optional nodes. Krea, Anima, Flux GGUF,
MiniMax H3, Hires, ADetailer, video/audio, latent upscaling, and RTX VSR each
require their matching model files and ComfyUI node classes.

`paths.webui_root` and `paths.comfyui_root` are optional local integration
settings. When configured, model discovery can update
`extra_model_paths.yaml` and create model links/junctions; restart ComfyUI
after changing its model search paths. Leave both blank when managing server
models yourself or using a remote server.

`prep`/`preview` expand jobs without queueing generation. `batch --resume`
uses local history; preserve it together with the job file and outputs. If a
request times out or a process is interrupted, check `status` and server
history before resubmitting. `interrupt` affects the server's current job and
does not erase local outputs.

The WebUI is optional; CLI-only use needs no Gradio installation. The
`--dump-workflow` option writes the resolved graph as part of generation; use
`bindings` or inspect the API JSON for graph inspection without queueing work.
