"""Check adapter target names and matrix shapes without allocating GPU models.

Run with ComfyUI's embedded Python (it already has gguf). Header coverage proves
structural compatibility, not sampler execution or output quality.
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path


def header(path):
    with path.open("rb") as source:
        length = struct.unpack("<Q", source.read(8))[0]
        return json.loads(source.read(length))


def targets(path):
    return {key: tuple(value["shape"]) for key,value in header(path).items() if key != "__metadata__"}


def coverage(adapter, base):
    state = targets(adapter)
    rows = []
    for key, a_shape in state.items():
        suffix = next((s for s in (".lora_A.weight", ".lora_down.weight") if key.endswith(s)), None)
        if suffix is None:
            continue
        module = key.removeprefix("diffusion_model.").removesuffix(suffix)
        up_suffix = ".lora_B.weight" if suffix == ".lora_A.weight" else ".lora_up.weight"
        b_shape = state.get(key.removesuffix(suffix) + up_suffix)
        weight = base.get(module + ".weight")
        expected = (b_shape[0], a_shape[1]) if b_shape and len(a_shape) == len(b_shape) == 2 else None
        rows.append({"module": module, "matched": weight == expected,
                     "base_shape": weight, "adapter_shape": expected})
    deltas = []
    for key, shape in state.items():
        suffix = next((s for s in (".diff", ".diff_b") if key.endswith(s)), None)
        if suffix:
            target = key.removeprefix("diffusion_model.").removesuffix(suffix) + (".bias" if suffix == ".diff_b" else ".weight")
            deltas.append({"module":target,"matched":base.get(target) == shape,
                           "base_shape":base.get(target),"adapter_shape":shape})
    return {"modules":len(rows), "matched":sum(row["matched"] for row in rows),
            "delta_targets":len(deltas), "delta_matched":sum(row["matched"] for row in deltas),
            "mismatches":[row for row in rows + deltas if not row["matched"]]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy-root", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    root = args.comfy_root / "models"
    from gguf import GGUFReader
    reader = GGUFReader(str(root/"diffusion_models/qwen_image_2.1-Q6_K.gguf"))
    qwen = {t.name:tuple(int(n) for n in reversed(t.shape)) for t in reader.tensors}
    h3 = targets(root/"diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors")
    wan_high = targets(root/"diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors")
    wan_low = targets(root/"diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors")
    pairs = {
        "qwen-image-2.1-consistency.safetensors":qwen,
        "qwen-image-2.1-outfit-swap.safetensors":qwen,
        "qwen-image-2.1-outpaint-v2.safetensors":qwen,
        "h3_character_swap_pro4500_1000.safetensors":h3,
        "h3upscale_v2_x2.safetensors":h3,
        "Wan2_2_I2V_AniSora_3_2_HIGH_rank_64_fp16.safetensors":wan_high,
        "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors":wan_low,
    }
    report = {name:coverage(root/"loras"/name, base) for name,base in pairs.items()}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    failed = False
    for name,row in report.items():
        print(f"{name}: {row['matched']}/{row['modules']} module shapes match")
        failed |= bool(row["mismatches"] or not row["modules"])
        if row["mismatches"]:
            print(json.dumps(row["mismatches"][:2]))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
