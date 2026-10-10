# Dedicated workflow tools

Open **Workflow tools** in the wrapper. Each panel runs a fixed graph with its
own inputs and preset. The global workflow/model selectors are hidden here.
The configurable Qwen, Video, Character Swap and Head Swap paths keep their
existing graphs and settings.

| Panel | Inputs | Recipe |
| --- | --- | --- |
| H3 Memory saver video | First frame, scene/action, optional last frame | Regular FL2VA W4A8, native 20 steps, H3 memory optimization |
| H3 Character swap LoRA | Character picture, source clip, target person | Ref2VA INT8, dedicated swap LoRA, optional Turbo 8-step, original source audio |
| H3 OmniChar video | `.char`, scene/action/dialogue | Ref2VA, up to four balanced references, available voice, optional compatible H3 adapter |
| H3 Guided clip upscale | Finished clip and source trim | Plain Ref2VA, latent upscale plus guided light redraw, 25,000-token tiles |
| Qwen Reference control | Source, up to two extra references, instruction | Independent reference strengths, optional reference mask, phrase weights |
| Qwen Consistent edit | Source and small edit instruction | Dedicated consistency LoRA |
| Qwen Outfit swap | Person, outfit reference, instruction | Dedicated outfit consistency LoRA |
| Qwen Outpaint | Source, extension margins, optional scene description | Outpaint v2, gray canvas, exact source pixels restored |
| Keyframe animation | Two to five ordered pictures, movement and duration per pair | Local Wan 2.2 FLF2V, AniSora high-noise + Lightx2v low-noise |

Use a short clip first. H3 starts at about 0.5 MP and 24 fps; video trims are
limited to 15 seconds. Source clips are normalized to 24 fps, padded to H3's
17n+5 sampling grid, and trimmed back for delivery. Source audio is retained
in the swap and guided-upscale outputs. Padding never becomes an extra frozen
tail in those delivered clips.

Qwen uses the installed Q6 base, Qwen3-VL 8B W4A8 encoder and native 2.1 VAE.
Edits use 25 steps, CFG 1, Euler/simple, full denoise, and the encoder's output
latent. References are limited to about 1 MP each. The reference cache stays
in RAM at its native precision. Masks select reference details and do not
freeze output pixels. Consistency is intended for small edits; it can resist
pose changes, as described by its [author](https://huggingface.co/ausboss/Qwen-Image-2.1-Consistency-LoRA).

Outpainting preserves the source size and pastes its exact RGB pixels back.
Right/bottom padding gains at most 31 pixels to make the canvas a multiple of
32. The canvas limit is 2 MP. This uses the author's gray-canvas and encoder
latent approach without a latent noise mask. The implementation restores
source pixels with core compositing nodes and a hard boundary; seam appearance
still needs visual evaluation. [Outpaint recipe](https://huggingface.co/ausboss/Qwen-Image-2.1-Outpaint-LoRA).

The LoRA character swap is a parallel addition. It uses the original source
clip as its reference and does not use the SAM3/inverted-color treatment in
the existing swap graph. It targets one person and starts at strength 1.
The panel enables the installed Ref2VA Turbo 8-step LoRA by default, chained
after the swap adapter at strength 1, with Euler/simple and video/audio shifts
6/3. Disable **Use Turbo 8-step LoRA** for the original native 20-step recipe
(res_multistep/simple, shifts 12/3). **Override steps** reveals a 1–40 slider
for either mode; when disabled, the slider is ignored and the mode's 8/20
steps apply. Overriding steps keeps the selected adapters and sampler recipe.
Exact expression and motion matching remain experimental.
[Publisher](https://huggingface.co/akatz-ai/MiniMax-H3-Character-Swap-LoRA).

OmniChar reads `.char` files with the pinned SDK and stages them under
`ComfyUI/models/characters/wrapper`. It automatically falls back to reference
conditioning when a portable MiniMax H3 adapter is absent. A foreign model
adapter is never applied to H3. An available H3 adapter must pass the node's
90% key-coverage gate; the wrapper does not lower that gate.
[OmniChar nodes](https://github.com/omnichar/ComfyUI-Omnichar).

Guided upscale is an enlarged and lightly redrawn delivery, not native
high-resolution generation. The output target defaults to 2 MP; the source
is fitted to at most about 0.9 MP for the H3 pass. This container feeds a
plain base model and lets the upsampler apply its own adapter and tiling.
It does not combine those forward patches with the memory optimization node.
[Upsampler](https://github.com/dntpi/ComfyUI-H3-Video-Upsampler).

## Downloadable keyframe workflow

The [Comfy Cloud keyframe share](https://cloud.comfy.org/?share=77e905e3fc4a)
is downloadable. The exported UI and API graphs are preserved in `upstream/`.
It runs local Wan 2.2 nodes rather than an API-only Cloud service.

The wrapper retains the two model/adapter branches, shift 5, CFG 1,
Euler/simple and the four-step split (high-noise steps 0–2, low-noise 2–4).
It uses one shared loader set, tiled VAE decode, about 0.4 MP at the first
picture's aspect ratio, and two to five keyframes. Each adjacent pair is
generated once. Joining removes a duplicated endpoint at each boundary.
Durations round up to Wan's 4n+1 grid with a 17-frame minimum; they are
approximate. Unlike the original example's different frame-skip rates, this
panel delivers every selected transition at 24 fps. Unused branches are not
included in the graph.

## Installation and stability

The opt-in installer reads `configs/selected_workflows.assets.json`:

```powershell
python scripts/setup_selected_workflows.py --comfy-root C:\ComfyUI_windows_portable\ComfyUI
```

It installs pinned additive snapshots of H3-Optimizations, OmniChar, the Qwen
reference enhancer and the H3 upsampler. It installs only
`omnichar-sdk==0.1.1` with `--no-deps` in ComfyUI's embedded Python. It never
upgrades ComfyUI, Torch, CUDA, or other node packs. Existing unrecognized
folders are refused rather than overwritten. Node snapshots include a
`.wrapper-source.json` marker with their exact revision.

H3 memory optimization uses **Preserve native**, QKV streaming **Auto**,
4,096 activation rows and standard attention memory. Forced quantization,
forced attention replacement, and sparse attention are not enabled. Veda is
not installed. [H3 memory optimization](https://github.com/Zironic/H3-Optimizations).

Models go directly into ComfyUI's `models/diffusion_models`, `text_encoders`,
`vae` and `loras`. Downloads resume `.part` files, verify publisher size and
SHA256, check the safetensors header, and rename only after verification.
The installer records resolved model revisions, hashes and paths in
`outputs/selected-workflows-install.json`. It refuses to overwrite a model
with a different size or hash.

Preflight checks the live node schema, model names and socket types before
queueing. The panels share the wrapper's busy lock and also refuse to add a
job to an already busy ComfyUI queue. Stop only interrupts a matching owned
prompt. Each successful run saves media, the actual API graph and a result
record under `outputs/containers/<tool>/<run>/`, plus a history entry.

Default API files here use placeholder filenames. Upload/select valid inputs
before queueing one directly in ComfyUI. To regenerate these examples:

```powershell
python scripts/export_selected_workflows.py
python scripts/check_selected_workflows.py
C:\ComfyUI_windows_portable\python_embeded\python.exe -s scripts/check_selected_model_headers.py --comfy-root C:\ComfyUI_windows_portable\ComfyUI --report outputs/selected-workflows/model-headers.json
```

The setup check does not queue inference. The header check verifies adapter
target names and matrix shapes; the live preflight checks default and
optional-input graph variants against your installed nodes and models. These
checks do not prove peak VRAM/RAM, sampler execution or visual quality.

To disable an addition, rename just its installed custom-node folder with a
`.disabled` suffix and restart ComfyUI. Keep the downloaded weights for a
later retry. The old wrapper workflows do not contain these new nodes.
