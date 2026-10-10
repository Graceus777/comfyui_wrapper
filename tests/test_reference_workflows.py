from comfyui_wrapper.config import from_dict
from comfyui_wrapper.h3 import (
    CHAR_UNET, DEFAULT_UNET, RECIPES, REF_TURBO8,
    apply_character_swap, apply_reference_recipe, character_swap_prompt,
    clear_first_frame, set_r2v_refs, set_reference_frame_guides,
    video_model_choices, video_prompt_default, video_workflow_options,
)
from comfyui_wrapper.workflow import load_workflow


def test_guides_do_not_replace_appearance_or_initialize_latent(repo_root):
    graph = load_workflow(repo_root / "workflows/minimax_h3_r2v_loop.api.json")
    graph = set_r2v_refs(graph, ref_images=["sheet.png", "profile.png"])
    guided = set_reference_frame_guides(graph, first_frame="start.png", last_frame="end.png")
    assert guided["6"]["inputs"]["ref_images.ref_image_0"] == ["h3_ref_img_0", 0]
    assert guided["h3_ref_img_0"]["inputs"]["image"] == "sheet.png"
    assert guided["h3_ref_img_1"]["inputs"]["image"] == "profile.png"
    assert "first_frame" not in guided["6"]["inputs"]
    assert guided["h3_frame_start_guide"]["inputs"]["frame_idx"] == 0
    assert guided["h3_frame_end_guide"]["inputs"]["frame_idx"] == -1
    assert guided["h3_frame_end_guide"]["inputs"]["positive"] == ["h3_frame_start_guide", 0]
    assert guided["7"]["inputs"]["conditioning"] == ["h3_frame_end_guide", 0]
    assert guided["11"]["inputs"]["latent_image"] == ["6", 1]
    replaced = set_reference_frame_guides(guided, last_frame="other-end.png")
    assert "h3_frame_start_guide" not in replaced
    assert replaced["h3_frame_end_guide"]["inputs"]["positive"] == ["6", 0]
    assert replaced["h3_ref_img_0"]["inputs"]["image"] == "sheet.png"


def test_workflow_model_families_and_fixed_steps(tmp_path, repo_root):
    cfg = from_dict({}, root=tmp_path)
    names = [DEFAULT_UNET, CHAR_UNET, "minimax_h3_ref2va_turbo_fused.safetensors", "sdxl.safetensors"]
    assert video_model_choices(names, "i2v") == [DEFAULT_UNET]
    assert video_model_choices(names, "r2v_frames") == [CHAR_UNET]
    opt = video_workflow_options("r2v", cfg.video, unets=names, loras=[REF_TURBO8], recipe="ref_turbo8")
    assert opt["model"] == CHAR_UNET
    assert opt["steps"] == 8 and opt["steps_visible"] is False
    assert not any(key == "ref_turbo4" for _label, key in opt["recipes"])
    graph = load_workflow(repo_root / "workflows/minimax_h3_r2v_loop.api.json")
    graph = apply_reference_recipe(graph, recipe=RECIPES["ref_turbo8"], lora_name=REF_TURBO8, steps=50)
    assert graph["10"]["inputs"]["steps"] == 8
    assert graph["9"]["inputs"]["sampler_name"] == "euler"
    assert graph["16"]["inputs"]["shift_video"] == 6
    quality = apply_reference_recipe(graph, recipe=RECIPES["ref_native"], steps=25)
    assert "h3_reference_turbo" not in quality
    assert quality["10"]["inputs"]["steps"] == 25
    assert quality["10"]["inputs"]["scheduler"] == "normal"


def test_swap_appearance_references_never_become_frame_anchors(repo_root):
    graph = load_workflow(repo_root / "workflows/minimax_h3_character_swap.api.json")
    graph = apply_character_swap(graph, character_image="sheet.png", source_video="posed.mp4",
                                 length=73, sam_target="the person", reference_images=["profile.png"], steps=50)
    assert graph["6"]["inputs"]["ref_images.ref_image_1"] == ["h3_ref_img_1", 0]
    assert graph["h3_ref_img_1"]["inputs"]["image"] == "profile.png"
    assert graph["11"]["inputs"]["latent_image"] == ["6", 1]
    assert graph["10"]["inputs"]["steps"] == 8
    assert not any(node["class_type"] == "MiniMaxH3AddGuide" for node in graph.values())
    prompt = character_swap_prompt(target="the person", reference_image_count=2)
    assert "appearance comes from <Picture 1> and <Picture 2>" in prompt
    assert "Do not begin in a T-pose" in prompt


def test_prompt_defaults_and_text_graph_do_not_use_still_initialization(tmp_path, repo_root):
    cfg = from_dict({"video": {"prompt": "Animate this still", "reference_prompt": "A new scene"}}, root=tmp_path)
    assert video_prompt_default(cfg.video, "i2v") == "Animate this still"
    assert video_prompt_default(cfg.video, "r2v") == "A new scene"
    assert video_prompt_default(cfg.video, "charswap") == ""
    graph = load_workflow(repo_root / "workflows/minimax_h3_i2v_loop.api.json")
    graph = clear_first_frame(graph)
    assert "first_frame" not in graph["6"]["inputs"]
    assert not any(node["class_type"] == "LoadImage" for node in graph.values())
