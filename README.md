# comfyui_wrapper

Config-driven Python client for a running [ComfyUI](https://github.com/comfyanonymous/ComfyUI) server.

Ways in:

1. **WebUI** — A1111-style Gradio app (`python webui.py`): SDXL, Anima, Krea 2, Flux GGUF, MiniMax H3 loops.
2. **`generate`** — one still prompt, every sampler/size/checkpoint setting in `config.yaml`.
3. **`video`** — MiniMax H3 image-to-video (same graph as `sd-comic-ext` ambient loops).
4. **`prep` + `batch`** — combinator-style zone expansion.

Zone and prompt JSON from [sd-combinator-ext](https://github.com/Graceus777/sd-combinator-ext) load as-is. LoRAs are applied as ComfyUI `LoraLoader` nodes, not as `<lora:...>` tags in the prompt.

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

| Flag | Meaning |
|------|---------|
| `--port 7861` | WebUI port (default 7860) |
| `--listen` | Bind `0.0.0.0` |
| `--share` | Gradio public tunnel |
| `--no-browser` | Do not open a browser |
| `--host` / `--comfy-port` | Override the ComfyUI server from `config.yaml` |

- **txt2img** — prompt / negative, sampling, size, seed, batch count, **Hires. fix**, and **ADetailer**. Extra networks cover LoRA (`<lora:name:weight>` → ComfyUI LoRA nodes), textual inversion (`embedding:name` from A1111 `embeddings/`), and hypernetworks. Checkpoints, LoRAs, VAE, embeddings, and ControlNet are used from `paths.webui_root` without copying.
- **Batch** — combinator zone + prompt config, plus its own steps / CFG / resolution / sampler, and ADetailer. Preview / Prep / Run, or run an existing `configs/jobs/*.json` file.
- **History** — recent files in `paths.output`.

Generation info is printed in A1111 infotext form (`Steps: …, Sampler: Euler a, Seed: …`) so you can recycle the last seed.

## Single prompt

```powershell
python -m comfyui_wrapper generate
python -m comfyui_wrapper generate --prompt "1girl, looking at viewer" --steps 28 --seed 42
python -m comfyui_wrapper generate --lora example_char_a:0.8 --lora "YOUR_STYLE_LORA:1"
python -m comfyui_wrapper generate --dump-workflow patched.json
```

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

## First-run checks and recovery

Use an editable installation from this checkout; bundled API graphs remain in `workflows/`. The source-only `requirements.txt` alternative also requires `PYTHONPATH=src` when using `python -m comfyui_wrapper`; `webui.py` adds that path itself.

```powershell
comfywrap status
comfywrap models
comfywrap bindings
```

Replace the checkpoint in `config.yaml` and the placeholder LoRA names in `configs/zones/example.json` with names returned by your server. The example files describe a format; they are not bundled models. Start with SDXL and one image before enabling additional nodes.

Krea, Anima, Flux GGUF and MiniMax H3 graphs require the matching model files and node classes on the server. Hires, face detailing, GGUF loading, video/audio, latent upscaling and RTX VSR add their own node dependencies. Inspect a graph's `class_type` values and compare them with ComfyUI's `/object_info`. Selecting a workflow does not install its dependencies. Model and node revisions can change graph compatibility.

`paths.webui_root` and `paths.comfyui_root` are optional local integration settings. When configured, model discovery can update `extra_model_paths.yaml` and create model links/junctions; restart ComfyUI after changing its model search paths. Leave both blank when managing server models yourself or using a remote server.

`prep`/`preview` expand jobs without queueing generation. `batch --resume` uses local history; preserve it together with the job file and outputs. If a request times out or a process is interrupted, check `status` and server history before resubmitting. The `interrupt` command interrupts the server's current work, which may belong to another client. It does not erase local outputs.

The still-image `--dump-workflow` option writes the resolved graph as part of generation; it is not a dry-run flag. For graph inspection without generation use `bindings` or inspect the API JSON directly.

The WebUI is optional; CLI-only use needs no Gradio installation. If the UI fails after a Gradio upgrade, capture the traceback and check compatibility before changing an active production environment.
