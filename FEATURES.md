# Feature inventory

`comfyui_wrapper` is a config-driven Python client and local WebUI for a
running ComfyUI server. It keeps the authored workflow graph in API-format
JSON and applies run-specific settings by patching a copy of that graph. It
does not download models or install ComfyUI custom nodes.

This document describes the capabilities implemented in the current source.
The matching ComfyUI nodes, model files, and optional command-line tools must
already be available in the environment where a job runs.

## Core ComfyUI client

- HTTP client for ComfyUI's queue, history, system statistics, object/model
  listings, image upload, output download, and interrupt endpoints.
- WebSocket progress monitoring with polling fallback when a WebSocket is not
  available.
- Queue submission with client-supplied prompt IDs and structured
  `JobResult`/`GenerateResult` values.
- Local output saving and generation-history records that retain prompt IDs,
  seeds, output names, and generation metadata.
- Explicit health checks before UI and generation work, with readable errors
  when the ComfyUI server is unavailable.

## Workflow loading and graph patching

- Loads ComfyUI API-format workflow JSON exported with **File → Export (API)**.
- Rejects UI-format graphs (`nodes`/`links`) with an actionable export error.
- Auto-detects common bindings for positive and negative text, checkpoint or
  UNET, CLIP/text encoders, VAE, seed, steps, CFG, sampler, scheduler, width,
  height, batch size, image, video length, FPS, audio VAE, and H3 reference
  image sizing.
- Supports explicit `workflow.map` bindings for custom graphs when automatic
  detection is not sufficient.
- Handles standard `KSampler`/`KSamplerAdvanced` graphs and the custom sampler
  patterns used by Flux GGUF and MiniMax H3 workflows.
- Patches a deep copy of the graph, leaving the bundled/source graph unchanged.
- `bindings` reports the resolved node/input mapping and the img2img injection
  point available for a graph.
- Dumps the fully resolved graph for inspection with `--dump-workflow`.

## Still-image generation

- One-shot txt2img through the CLI, Python API, or WebUI.
- Per-run overrides for prompt, negative prompt, checkpoint/UNET, CLIP(s),
  VAE, steps, CFG, sampler, scheduler, width, height, batch size, seed, and
  filename prefix.
- `-1`/negative seeds become a generated seed that is reported in the result,
  infotext, and history.
- Local or ComfyUI-input-key init images for img2img. A local init is uploaded
  automatically, resized to the requested canvas, VAE-encoded, optionally
  repeated for batch generation, and connected to the first sampler with the
  requested denoise strength.
- Idempotent img2img graph injection and a helper to clear the injected graph
  back to its original empty-latent input.
- Img2img support for the bundled SDXL, Anima, Krea 2, and Flux GGUF graph
  shapes, including graphs whose VAE comes from a standalone `VAELoader`.
- A1111-style Hires. fix as a second sampling pass, using either latent
  upscaling or an installed pixel upscaler model.
- ADetailer/Impact Pack `FaceDetailer` injection with detector, guide size,
  steps, CFG, sampler, scheduler, denoise, and seed settings.
- Dedicated VAE loader injection so a selected VAE can remain independently
  cached instead of being repeatedly unpacked from a checkpoint.
- Generation hashes and output-prefix checks used by batch skip/resume logic.

## LoRA and extra networks

- Accepts LoRA names, filenames, weighted references, and repeatable CLI
  `--lora` options.
- Parses A1111 `<lora:name:weight>` prompt tags, removes them from the text,
  resolves the corresponding file, and injects ComfyUI `LoraLoader` nodes.
- Chains multiple LoRAs in order and preserves their activation text in the
  positive prompt.
- Supports full MODEL+CLIP LoRA injection and `model_only` injection for
  graphs such as Flux where CLIP should not be patched.
- Reads activation text from the existing `lora_texts/` layout used by
  `sd-combinator-ext`.
- WebUI selectors and prompt insertion for LoRAs, textual-inversion embeddings,
  and hypernetworks, including weighted embedding tags.
- Model discovery across ComfyUI and an optional A1111 installation, with
  duplicate names merged without copying model files.

## A1111 model sharing and discovery

- Uses `paths.webui_root` as an optional source of checkpoints, LoRAs, VAEs,
  text encoders, embeddings, hypernetworks, ControlNet models, and upscalers.
- Uses `paths.comfyui_root` to create or extend `extra_model_paths.yaml` while
  preserving existing mappings.
- Creates hardlinks, directory junctions, or symlinks when needed; model data
  is not copied into a second installation.
- Links Anima UNETs, Qwen text encoder/VAE assets, ADetailer detectors, and
  other unseen A1111 model files into the folders scanned by ComfyUI.
- Lists checkpoints, diffusion-model/UNET files, text encoders, VAEs, PDD Acc
  weights, model patches, LoRAs, embeddings, hypernetworks, ADetailer models,
  and upscalers from both local and live-server sources.
- Can operate with only a live ComfyUI server when local A1111 integration is
  not configured.

## Anima and Anima LLLite controls

- Bundled Anima graph support for an Anima UNET, Qwen3 text encoder, and Qwen
  image VAE rather than treating Anima as a conventional SDXL checkpoint.
- Anima LLLite control injection using ComfyUI's built-in
  `ModelPatchLoader` and `AnimaLLLiteApply` nodes.
- Stacks multiple controls in list order; each control has an image, model
  patch, strength, and start/end timestep schedule.
- Accepts a saved preprocessed depth, pose, or lineart map unchanged so the
  wrapper consumes the exact authored control image.
- Optional in-graph Canny preprocessing for a raw control photo, with bounded
  low/high thresholds for lineart-style control patches.
- Local control maps are uploaded automatically and their source hashes are
  retained in generation metadata.
- CLI flags and WebUI controls expose the same Anima LLLite settings.
- Validates missing files, missing model patches, invalid schedules, finite
  strengths, and Canny threshold ranges before queueing work.

## Combinator-style batch generation

- Loads zone and prompt JSON from the wrapper or an external
  `sd-combinator-ext` root without requiring a copy.
- Three zone modes:
  - **always** applies its LoRAs and activation text to every job.
  - **loop** creates one job per LoRA, optionally least-used first.
  - **random** samples a configured number of LoRAs with history-aware
    weighting toward less-used entries.
- Expands arbitrary prompt slots such as `{pose}` and `{view}` as a Cartesian
  product.
- Resolves LoRA filenames, weights, activation text, seeds, output prefixes,
  prompts, and generation hashes before the GPU starts.
- `preview` prints the expanded jobs without queueing them.
- `prep` writes an editable JSON job list; random choices can be frozen at
  prep time.
- `batch` runs an existing job file or prepares and runs a zone in one command.
- Supports LoRA filtering, slot overrides, limits, cooldowns, batch counts,
  local output copies, and Ctrl+C interruption.
- `--resume` skips completed jobs and preserves per-job status, error, output,
  and prompt-ID fields.
- Skip-exists checks both `generation_history.jsonl` hashes and matching output
  prefixes before queueing duplicate work.

## MiniMax H3 video

- Bundled H3 image-to-video workflow with synchronized video and audio output.
- I2V mode requires a first-frame still and supports automatic aspect-preserving
  sizing on H3's 32-pixel grid.
- Optional first-plus-last-frame interpolation (FLF2V) for the H3 I2V graph;
  the last frame is resized to the same canvas as the first frame.
- R2V-lite mode uses the native `MiniMaxH3ReferenceToVideo` node with the
  existing FL2VA UNET, avoiding a separate Ref2VA model download.
- R2V-lite accepts up to 9 still references, 3 reference videos, and 3
  reference audio files. Video references expose both frames and their
  soundtrack to the conditioning node.
- R2V reference sizing supports `match` for faster input matching and `max` for
  the larger identity-preserving reference size.
- The R2V prompt can use `<Picture i>`, `<Video k>`, and `<Audio j>` references
  to describe identity, motion, and voice sources.
- Automatically coerces the selected H3 UNET, Qwen text encoder, video VAE,
  audio VAE, and acceleration weight into a compatible stack.
- Supports native sampling, PDD Acc 8-step, and Acc 4-step when the matching
  weight is available; the Acc path uses `MiniMaxH3PDDAccApply`, not a normal
  LoRA loader.
- Applies H3 sampler, step, sigma-shift, video/audio VAE, frame-count, FPS,
  and output-prefix settings to a copied graph.
- Optional decode-only latent upscaling through
  `MinimaxH3LatentUpscaler3D`, without a second sampling pass.
- Optional NVIDIA RTX Video Super Resolution post-processing through ComfyUI.
- Optional cyclic ffmpeg crossfade that closes the generated clip into an
  ambient loop while retaining the native audio track.
- Stages local reference videos/audio into ComfyUI's input tree when direct
  upload is unavailable.
- Records video mode, recipe, references, seed, output files, loop errors,
  latent-upscale state, and RTX-VSR state in generation history.

## WebUI

- Optional Gradio WebUI (`python webui.py` or `comfywrap ui`) that resembles an
  A1111 workflow while sending API-format graphs to ComfyUI.
- Txt2img tab with prompt and negative prompt, model selectors, dimensions,
  sampler/scheduler, steps, CFG, seed, batch count, filename prefix, and
  live progress.
- Txt2img extras for img2img, Hires. fix, Anima LLLite, ADetailer, LoRA,
  embedding, and hypernetwork insertion.
- Video/H3 tab with I2V, FLF2V, and R2V-lite mode selection, still/last-frame
  uploads, reference image/video/audio uploads, H3 recipe selection, model
  stack selectors, auto-size, latent upscale, RTX-VSR, looping, and progress.
- Batch tab with zone/prompt selectors, slot editor, LoRA filter, preview,
  prep, run, resume, existing job-file execution, and batch-specific settings.
- History tab that browses recent locally saved images.
- Refresh action that re-reads live model, workflow, sampler, scheduler,
  control, zone, prompt, job, and H3 choices.
- Stop/interrupt control and a serialized worker path that keeps the UI
  responsive while a generation is running.
- A1111-style infotext containing prompts, negative prompt, sampler, steps,
  CFG, seed, size, model, LoRAs, img2img, Hires, ADetailer, and Anima control
  metadata.
- Configurable bind address, port, browser launch, and optional Gradio share
  tunnel.

## Command-line interface

The installed `comfywrap` command and `python -m comfyui_wrapper` expose:

| Command | Capability |
| --- | --- |
| `generate` | Queue one still-image generation with per-run overrides. |
| `preview` | Expand and print batch jobs without queueing them. |
| `prep` | Write an editable, deterministic job file. |
| `batch` | Run a job file or prepare and run a zone. |
| `configs` | List available zone and prompt configuration names. |
| `loras` | List resolved LoRA files and activation sources. |
| `models` | List a selected ComfyUI/A1111 model folder. |
| `bindings` | Inspect workflow bindings and supported injection points. |
| `status` | Show ComfyUI queue/system status. |
| `interrupt` | Request interruption of the server's current job. |
| `video` | Queue H3 I2V/FLF2V or R2V-lite video generation. |
| `ui` | Launch the optional Gradio WebUI. |

Global CLI options can select a config file and override the ComfyUI host or
port. Generation options include prompt, negative prompt, model, sampler,
scheduler, size, seed, init image, denoise, LoRA, Anima LLLite controls, and
workflow output. Video options include mode, last frame, repeatable reference
images/videos/audio, reference sizing, H3 model/recipe settings, and post-
processing toggles.

## Bundled workflows

- `sdxl_txt2img.api.json` — standard Illustrious/SDXL checkpoint graph.
- `anima_txt2img.api.json` — Anima UNET + Qwen text encoder + Qwen image VAE.
- `krea2_txt2img.api.json` — Krea 2 Turbo graph.
- `flux_gguf_txt2img.api.json` — Flux dev GGUF with dual CLIP inputs.
- `minimax_h3_i2v_loop.api.json` — MiniMax H3 I2V with native audio mux.
- `minimax_h3_r2v_loop.api.json` — MiniMax H3 FL2VA R2V-lite graph.

Any additional ComfyUI API-format graph can be used by placing it under
`workflows/`, selecting it in `config.yaml`, and checking `comfywrap bindings`.

## Python API

The package exports the primary operations directly:

```python
from comfyui_wrapper import (
    ComfyClient,
    generate,
    generate_video,
    load_config,
    prep_jobs,
    preview_jobs,
    run_jobs,
)
```

This makes the wrapper usable as a library for custom runners while retaining
the same configuration, graph patching, history, and output behavior as the
CLI/WebUI paths.

## Operational boundaries

- ComfyUI must be running and must have the custom nodes required by the
  selected graph. Selecting a workflow does not install its dependencies.
- Model filenames in the example config are placeholders or expected local
  assets; the wrapper does not fetch model weights.
- `--share` exposes the Gradio UI through Gradio's public tunnel and should be
  treated as an explicit network exposure choice.
- `interrupt` affects the ComfyUI server's current job, which may belong to
  another client; it does not delete local outputs.
- H3 loop closing requires `ffmpeg`; RTX-VSR requires a compatible ComfyUI
  node and NVIDIA runtime; ADetailer requires the Impact Pack/Ultralytics
  nodes; Anima LLLite requires the built-in node support and model patches.
- The tests validate configuration, graph transformation, parsing, and UI
  construction. A successful test run does not replace a live GPU render or
  prove that a particular model/node revision is compatible.
