"""Export reviewable default API graphs for the dedicated wrapper panels."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from comfyui_wrapper import workflow_containers as r


def examples():
    return {
        "qwen_controlled": r.build_qwen_controlled(["source.png"], "Change the jacket to (dark green:1.2)."),
        "qwen_consistent": r.build_qwen_consistent("source.png", "Change the jacket to dark green."),
        "qwen_outfit": r.build_qwen_outfit("person.png", "outfit.png", "Use the outfit from image 2 on the person in image 1. Preserve the scene."),
        "qwen_outpaint": r.build_qwen_outpaint("source.png", (1024,768)),
        "h3_memory_i2v": r.build_h3_memory_i2v("first.png", "A continuous shot with natural movement."),
        "h3_lora_swap": r.build_h3_lora_swap("character.png", "source.mp4", "the main person", delivered_frames=120, turbo=True),
        "h3_omnichar": r.build_h3_omnichar("wrapper/character.char", "A continuous shot of the character speaking softly.", 1),
        "h3_guided_upscale": r.build_h3_guided_upscale("source.mp4", delivered_frames=120),
        "wan_keyframes": r.build_wan_keyframes(["first.png","last.png"], ["Smooth continuous motion between the keyframes."], [1.5]),
    }


if __name__ == "__main__":
    directory = ROOT / "workflows" / "containers"
    directory.mkdir(parents=True, exist_ok=True)
    for name, graph in examples().items():
        path = directory / f"{name}.api.json"
        path.write_text(json.dumps(graph, indent=2) + "\n", encoding="utf-8")
        print(path.name)
