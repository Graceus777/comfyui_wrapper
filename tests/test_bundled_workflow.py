from pathlib import Path

from comfyui_wrapper.workflow import detect_bindings, load_workflow


def test_bundled_sdxl_workflow_loads():
    path = Path(__file__).resolve().parents[1] / "workflows" / "sdxl_txt2img.api.json"
    wf = load_workflow(path)
    b = detect_bindings(wf)
    assert b["positive"][0] == "6"
    assert b["negative"][0] == "7"
    assert b["checkpoint"][0] == "4"
    assert b["seed"][0] == "3"


def test_bundled_anima_workflow_loads():
    path = Path(__file__).resolve().parents[1] / "workflows" / "anima_txt2img.api.json"
    wf = load_workflow(path)
    b = detect_bindings(wf)
    assert b["positive"][0] == "4"
    assert b["negative"][0] == "5"
    assert b["checkpoint"][0] == "1"
    assert b["clip"][0] == "2"
    assert b["vae"][0] == "3"
