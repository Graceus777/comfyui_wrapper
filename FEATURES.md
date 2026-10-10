# Feature inventory

`comfyui_wrapper` is a config-driven Python client, CLI, and local WebUI for
ComfyUI image/video jobs, plus a separate local voice-cloning worker. It
keeps authored workflow graphs in API-format JSON and applies run-specific
settings by patching a copy. Generation does not automatically download
models or install ComfyUI custom nodes. Explicit, opt-in setup scripts install
the [dedicated workflow dependencies](workflows/containers/README.md) or the
[voice runtime and model package](docs/voice-clone.md).

This document describes the capabilities implemented in the current source.
The matching ComfyUI nodes, model files, and optional command-line tools must
already be available in the environment where a job runs.

## Local voice cloning

- Dedicated **Voice Clone** tab and `comfywrap tts` CLI with dots.tts-SOAR.
- Best-match continuation from reference audio plus its exact transcript, or
  audio-only speaker cloning. Clean 3–10 second recordings are recommended.
- Reference recording through the WebUI microphone or file upload; reference
  duration is checked by the worker before inference.
- Language selectors for English, Chinese, Spanish, French, German, Japanese,
  Korean, or automatic detection. The worker passes the selected language to
  SOAR; quality depends on the reference and model.
- NFE sampling steps (10–32), CFG guidance (0–3), seed, and output-duration
  limit (3–30 seconds), with these settings retained in generation details.
- Local mono 48 kHz WAV output, retained reference, generation settings,
  timing, GPU memory measurements, and worker log.
- Isolated Python 3.12/CUDA runtime, offline generation after explicit setup,
  cancellation, and GPU model release at the end of each request.
- Idle ComfyUI model caches can be released before cloning; active or queued
  ComfyUI jobs block a new clone until they finish.
- [Setup and limitations](docs/voice-clone.md).

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
- A1111-style whole-picture inpainting: white mask = repaint via
  `LoadImage -> ImageScale -> ImageToMask -> GrowMaskWithBlur ->
  SetLatentNoiseMask` (with optional `InvertMask` for inpaint-not-masked).
  Idempotent `inpaint_*` nodes reuse the img2img init chain and work on SDXL,
  Anima, Krea 2, and Flux GGUF without an inpaint model.
- Idempotent img2img/inpaint graph injection and helpers to clear the injected
  graph back to its original empty-latent input.
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

## Qwen Image 2.1

- Dedicated Qwen Image tab, `comfywrap qwen` command, and `generate_qwen()`
  Python API for text-to-image and reference-image editing.
- Bundled Q6 GGUF base, Qwen3-VL 8B W4A8 encoder, and native Qwen 2.1 VAE;
  the graph builder also supports the standard UNET loader for safetensors.
- Up to 10 ordered reference images, automatically uploaded for an edit.
- Optional first-reference latent sizing, independent output width/height,
  reference resize limit, and batch generation from the encoder latent.
- Prompt/negative, seed, steps, CFG, sampler, scheduler, model-stack, and
  filename-prefix controls.
- Optional KV-cache device selection (`auto`, `gpu`, `cpu`, `off`) and dtype
  (`default`, `int8`, `int4`).
- Missing-image, mode, reference-count, and size validation before submission.
- Dedicated controlled, consistent, outfit, and outpaint recipes are separate
  panels under **Workflow tools**, described below.

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
- I2V mode requires a start and/or end still and supports automatic aspect-preserving
  sizing on H3's 32-pixel grid.
- Optional first-plus-last-frame interpolation (FLF2V) for the H3 I2V graph;
  the last frame is resized to the same canvas as the first frame.
- Reference generation uses `MiniMaxH3ReferenceToVideo` with the Ref2VA UNET.
  Appearance images do not become start frames. A separate reference-plus-frame
  workflow chains `MiniMaxH3AddGuide` at frame 0 and/or -1 while retaining
  appearance references and the base latent.
- Reference generation accepts up to 9 still references, 3 reference videos, and 3
  reference audio files. Video references expose both frames and their
  soundtrack to the conditioning node.
- R2V reference sizing supports `match` for faster input matching and `max` for
  the larger identity-preserving reference size.
- Head swap mode loads the regular `minimax_h3_fl2va_pruned_w4a8_mixed` UNET
  and Alissonerdx `minimax_h3_head_swap_v1.0_r32` as a separate LoRA (trigger
  `head_swap:`). `video.head_swap_lora` and `video.head_swap_strength` select
  the LoRA and its strength, which defaults to 1.0. The face still is
  `<Picture 1>` on `MiniMaxH3ReferenceToVideo`. The source clip is an aligned
  `MiniMaxH3AddGuide` at frame 0 (`<Video 1>`). The LoRA precedes sigma shift
  and PDD Acc; turbo and other distill LoRAs are stripped with Acc enabled.
- Character swap loads Ref2VA int8
  (`minimax_h3_ref2va_pruned_int8_convrot`) and Turbo V4 Step600 pruned, matching
  the [published Studio preset](https://github.com/BiggerFishy/comfyui-minimax-h3-studio).
  SAM3 tracks the named subject, `ImageInvert` and `ImageCompositeMasked` flip
  colors inside that mask, and `MiniMaxH3ReferenceToVideo` takes the
  replacement still as `<Picture 1>` and the treated frames as `<Video 1>`.
  Sigma shift is 6, the sampler is 8-step `res_multistep`, and decode is tiled.
  This mode does not use the FL2VA UNET or PDD Acc. Head swap still does.
- Swap source clips are selected by start/end seconds and converted to 24 fps
  without changing playback speed. The last frame is padded to H3's `17n+5`
  grid, then removed from the delivered output. The original trimmed audio
  is retained and cyclic crossfade is disabled. The default trim is the first
  three seconds; end `0` uses the whole remaining clip. Character swap follows
  the source aspect at 0.8 MP, with separate scene instructions and optional
  replacement appearance details. Only generated outputs are downloaded.
- The R2V prompt can use `<Picture i>`, `<Video k>`, and `<Audio j>` references
  to describe identity, motion, and voice sources.
- Automatically coerces the selected H3 UNET, Qwen text encoder, video VAE,
  audio VAE, and acceleration weight into a compatible stack.
- Supports native sampling, PDD Acc 8-step, and Acc 4-step when the matching
  weight is available; the Acc path uses `MiniMaxH3PDDAccApply`, not a normal
  LoRA loader.
- Reference presets use base quality with adjustable steps or installed
  Ref2V Turbo 4/8-step LoRAs with fixed steps. Character swap fixes the Studio
  preset at eight steps. The UI offers models only from the selected workflow's
  checkpoint family and relabels media inputs according to their role.
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

## Saved latents and video upscaling

- H3 generation can save video and audio latent halves next to the delivered
  clip. `video.save_latents` defaults to `true`; per-run CLI flags override it.
- **Upscale from latents** tab and `comfywrap upscale` command accept saved
  `.latent`/`.safetensors` files for a separate 3D upscale and decode-only pass.
  This path does not run a sampler again.
- Audio is decoded when the matching audio latent is supplied. CLI input can
  resolve the sibling audio file; browser uploads must include both halves.
  A video-only latent produces a silent output.
- Rendered `.mp4`, `.webm`, `.mov`, or `.mkv` input uses the RTX VSR path.
  Automatic mode selects the path from the extension; explicit `latent` and
  `rtx` modes are also available.
- VAE, audio VAE, upscaler model, scale, FPS, and RTX quality use `video.*`
  configuration; the CLI accepts an explicit output path and audio latent.
- The dedicated guided-upscale tool below performs a light redraw and is a
  different workflow from saved-latent decoding or RTX VSR.

## Dedicated workflow tools

Each panel builds its own API graph and preset rather than using the global
workflow/model selector. The existing configurable image and video tabs keep
their own settings.

| Panel | Implemented capability |
| --- | --- |
| H3 Memory saver video | Native FL2VA W4A8 sampling from a first frame and optional last frame, with H3 memory optimization. |
| H3 Character swap LoRA | Ref2VA with a dedicated character-swap adapter and source-performance conditioning; optional Turbo 8-step adapter, adjustable swap strength, and explicit step override. |
| H3 OmniChar video | `.char` package loading with up to four balanced references, available voice, and a compatible H3 adapter; falls back to references when an adapter is absent. |
| H3 Guided clip upscale | Trim a finished clip, enlarge it with tiled latent upscaling and a light guided redraw, and retain source audio. |
| Qwen Reference control | Up to three references with independent strength, phrase weighting, and an optional mask selecting details from a chosen reference. |
| Qwen Consistent edit | A separate consistency adapter for small edits with adjustable strength. |
| Qwen Outfit swap | A person image plus outfit reference, edit instruction, seed, and outfit-adapter strength. |
| Qwen Outpaint | Left/top/right/bottom canvas extension with source-pixel restoration at the original size and position; grid padding and a 2 MP canvas limit. |
| Keyframe animation | Local Wan 2.2 FLF2V for two to five consecutive pictures, per-pair movement/duration, AniSora high-noise and Lightx2v low-noise branches, and a joined 24 fps output. |

- Clip tools normalize source trims to 24 fps and supported H3 frame counts.
  Swap and guided-upscale delivery removes sampling padding and retains the
  selected source audio.
- Qwen reference masks control what the reference contributes; they do not
  freeze output pixels. Outpaint restores source pixels with compositing.
- OmniChar enforces the compatible-model and minimum-key-coverage checks;
  foreign-model adapters are not applied to H3.
- Preflight validates installed node schemas, required model choices, graph
  links, socket types, uploaded inputs, and tool-specific input limits.
- Panels share the wrapper's busy lock and refuse to queue behind an active
  ComfyUI job. Tool cancellation targets its owned prompt.
- Each run retains its media, actual API graph, result record, and generation
  history under the configured local output tree.
- Opt-in setup uses `configs/selected_workflows.assets.json`, pinned additive
  node snapshots, and the specified OmniChar SDK. Model downloads resume,
  verify publisher size/hash, check safetensors headers, and refuse to replace
  mismatched existing files.
- Export, live preflight, and model-header inspection scripts are provided.
  These checks do not queue inference or establish visual quality.

See [recipes, installation, and requirements](workflows/containers/README.md).

## WebUI

- Optional Gradio WebUI (`python webui.py` or `comfywrap ui`) that resembles an
  A1111 workflow while sending API-format graphs to ComfyUI.
- Txt2img tab with prompt and negative prompt, model selectors, dimensions,
  sampler/scheduler, steps, CFG, seed, batch count, filename prefix, and
  live progress.
- Txt2img extras for img2img, inpaint (ImageEditor draw-mask tab, A1111-style
  grow/blur/invert), Hires. fix, Anima LLLite, ADetailer, LoRA,
  embedding, and hypernetwork insertion.
- Qwen Image tab for Qwen Image 2.1 text-to-image and reference-image edits
  (up to 10 images, first-image latent size, optional KV-cache device/dtype).
- Video/H3 tab with frame animation, appearance references, references plus
  start/end guides, character/head swaps, and text generation; still/last-frame
  uploads, reference image/video/audio uploads, H3 recipe selection, model
  stack selectors, auto-size, latent upscale, RTX-VSR, looping, and progress.
- Upscale from latents tab for saved AV latent halves or a rendered clip.
- Workflow tools tab with the nine dedicated H3, Qwen, and Wan panels above.
- Voice Clone tab with reference recording/upload, exact transcript, clone
  mode, language, sampling controls, playable output, and worker stop.
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
- `launch_tailscale.bat` starts the WebUI on port 8000 and prints the current
  tailnet address, using the checkout's virtual environment or Python on PATH.

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
| `video` | Queue H3 frame animation, Ref2VA references/frame guides, swaps, or text generation. |
| `qwen` | Queue Qwen Image 2.1 text-to-image or a reference-image edit. |
| `upscale` | Upscale/decode a saved H3 latent or apply RTX VSR to a rendered clip. |
| `tts` | Clone a voice locally with dots.tts-SOAR, or report installation status with `--check`. |
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
- `qwen_image_2_1.api.json` — Qwen Image 2.1 Q6 GGUF text-to-image. Edits add reference images at queue time.
- `minimax_h3_i2v_loop.api.json` — MiniMax H3 I2V with native audio mux.
- `minimax_h3_r2v_loop.api.json` — MiniMax H3 Ref2VA reference graph.
- `minimax_h3_character_swap.api.json` — Ref2VA + Studio Turbo V4 + SAM3 character swap.
- `minimax_h3_head_swap.api.json` — regular FL2VA with separate head-swap LoRA and compatible Acc.

Dedicated tool graphs under `workflows/containers/`:

- `h3_memory_i2v.api.json` — H3 memory saver video.
- `h3_lora_swap.api.json` — dedicated character-swap LoRA, optional Turbo.
- `h3_omnichar.api.json` — OmniChar references, voice, and compatible adapter.
- `h3_guided_upscale.api.json` — tiled guided clip upscale with source audio.
- `qwen_controlled.api.json` — reference influence, phrase weights, and optional mask.
- `qwen_consistent.api.json` — small edits with the consistency adapter.
- `qwen_outfit.api.json` — person/outfit reference swap.
- `qwen_outpaint.api.json` — canvas extension and source-pixel restoration.
- `wan_keyframes.api.json` — local Wan 2.2 keyframe transitions.

`workflows/containers/upstream/` also preserves the original
`comfy-cloud-animation5.api.json` and `comfy-cloud-animation5.ui.json` exports.
The UI export is for the ComfyUI editor; the wrapper loads API graphs.

Any additional ComfyUI API-format graph can be used by placing it under
`workflows/`, selecting it in `config.yaml`, and checking `comfywrap bindings`.

## Python API

The package exports the primary operations directly:

```python
from comfyui_wrapper import (
    ComfyClient,
    generate,
    generate_video,
    generate_qwen,
    load_config,
    prep_jobs,
    preview_jobs,
    run_jobs,
)
```

This makes the wrapper usable as a library for custom runners while retaining
the same configuration, graph patching, history, and output behavior as the
CLI/WebUI paths.

Additional module APIs expose the dedicated operations:

```python
from comfyui_wrapper.container_runner import run_container
from comfyui_wrapper.tts import clone_voice
from comfyui_wrapper.video import upscale_dropped_file
```

## Operational boundaries

- ComfyUI image/video/upscale jobs need a running server with the nodes and
  models required by the selected graph. Selecting a workflow does not install
  its dependencies. Voice cloning uses its separate local runtime.
- Model filenames in the example config are placeholders or expected local
  assets. Generation does not fetch weights; the explicit workflow and voice
  setup scripts can download their dependencies.
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
