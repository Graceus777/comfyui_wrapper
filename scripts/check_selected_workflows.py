"""Read-only live node/model preflight; never queue inference.

Example API files use placeholder inputs. Their file selectors are skipped in
this setup check; the real UI run validates uploaded input choices as well.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from comfyui_wrapper.client import ComfyClient
from comfyui_wrapper.config import load_config
from comfyui_wrapper.container_runner import validate_graph
from export_selected_workflows import examples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    args = parser.parse_args()
    client = ComfyClient(cfg=load_config(args.config))
    live = client.get("/object_info", timeout=60)
    schemas = copy.deepcopy(live)
    for name, field in (("LoadImage","image"),("LoadVideo","file"),("OmnicharLoadCharacter","char")):
        schemas[name]["input"]["required"][field] = ["STRING", {}]
    graphs = examples()
    # Exercise dynamic mask, voice/adapter and all five keyframes in addition to defaults.
    from comfyui_wrapper import workflow_containers as r
    graphs.update({
        "qwen_controlled_masked": r.build_qwen_controlled(["one.png","two.png","three.png"], "Change the (coat:1.2)",
                                                            mask_key="mask.png", mask_index=3, isolate_mask=True),
        "h3_omnichar_voice_adapter": r.build_h3_omnichar("character.char", "She speaks", 4, use_voice=True, use_adapter=True),
        "wan_five_keyframes": r.build_wan_keyframes([f"{i}.png" for i in range(5)], ["Movement"]*4, [1.5]*4),
    })
    report = {name: validate_graph(graph, schemas) for name, graph in graphs.items()}
    path = ROOT / "outputs" / "selected-workflows" / "preflight.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"graphs":report,"system_stats":client.system_stats()}, indent=2), encoding="utf-8")
    failed = False
    for name, errors in report.items():
        print(f"{name}: {'FAIL' if errors else 'PASS'}")
        for error in errors:
            failed = True
            print("  " + error)
    if failed:
        raise SystemExit(1)
    print(f"Read-only preflight: {len(graphs)} graphs passed; no prompts queued.")


if __name__ == "__main__":
    main()
