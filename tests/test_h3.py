from copy import deepcopy

from comfyui_wrapper.config import apply_overrides, from_dict
from comfyui_wrapper.h3 import (
    acc_nfe,
    apply_h3_graph,
    apply_h3_latent_upscale,
    coerce_video_stack,
    match_acc_file,
    match_h3_unet,
    normalize_recipe,
    recipe_choices,
    recipe_settings,
    resolve_recipe,
    rtx_vsr_graph,
    video_recipe_ui,
)
from comfyui_wrapper.workflow import load_workflow


def _bundled(repo_root):
    return load_workflow(repo_root / "workflows" / "minimax_h3_i2v_loop.api.json")


def test_recipe_aliases():
    assert normalize_recipe("acc") == "acc"
    assert normalize_recipe("Acc 8-step") == "acc"
    assert normalize_recipe("pdd_acc") == "acc"
    assert normalize_recipe("acc4") == "acc4"
    assert normalize_recipe("native") == "native"
    assert normalize_recipe("12-step") == "native"
    assert recipe_settings("acc").steps == 8
    assert recipe_settings("acc").sampler == "euler"
    assert recipe_settings("native").acc is False
    assert resolve_recipe("acc4").key == "acc"
    assert resolve_recipe("acc4").nfe == 8
    assert resolve_recipe("acc4", acc_file="MiniMax-H3-FL2VA-Acc-4Step.safetensors").key == "acc4"


def test_recipe_choices_hide_4step_until_weight_exists():
    labels = [label for label, _key in recipe_choices()]
    assert "Acc 8-step" in labels
    assert "Acc 4-step" not in labels
    shown = recipe_choices(["MiniMax-H3-FL2VA-Acc-8Step.safetensors"])
    assert "acc4" not in {key for _label, key in shown}
    later = recipe_choices(["MiniMax-H3-FL2VA-Acc-4Step.safetensors"])
    assert "acc4" in {key for _label, key in later}


def test_acc_nfe_follows_filename():
    assert acc_nfe("MiniMax-H3-FL2VA-Acc-8Step.safetensors") == 8
    assert acc_nfe("minimax_h3_fl2va_pdd_acc_8step_comfyui.safetensors") == 8
    assert acc_nfe("MiniMax-H3-FL2VA-Acc-4Step.safetensors") == 4


def test_match_acc_file_swaps_wrong_trunk():
    files = [
        "MiniMax-H3-FL2VA-Acc-8Step.safetensors",
        "MiniMax-H3-Ref2VA-Acc-8Step.safetensors",
    ]
    assert match_acc_file("minimax_h3_fl2va_pruned_w4a8_mixed.safetensors", files).startswith("MiniMax-H3-FL2VA")
    assert match_acc_file("minimax_h3_ref2va_pruned_int8_convrot.safetensors", files).startswith("MiniMax-H3-Ref2VA")
    swapped = match_acc_file(
        "minimax_h3_fl2va_pruned_w4a8_mixed.safetensors",
        files,
        "MiniMax-H3-Ref2VA-Acc-8Step.safetensors",
    )
    assert "FL2VA" in swapped
    assert "Ref2VA" not in swapped


def test_match_h3_unet_prefers_w4a8_for_acc():
    unets = [
        "wai.safetensors",
        "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        "minimax_h3_fl2va_pruned_w4a8_mixed.safetensors",
    ]
    acc = match_h3_unet(unets, "acc")
    assert "w4a8" in acc
    native = match_h3_unet(unets, "native", "minimax_h3_fl2va_pruned_int8_convrot.safetensors")
    assert "int8_convrot" in native


def test_coerce_swaps_non_h3_stack():
    stack = coerce_video_stack(
        unet="krea2_turbo_fp8_scaled.safetensors",
        clip="clip_l.safetensors",
        vae="ae.safetensors",
        audio_vae="ae.safetensors",
        acc_file="MiniMax-H3-Ref2VA-Acc-8Step.safetensors",
        recipe="acc",
    )
    assert "minimax_h3" in stack["unet"]
    assert "fl2va" in stack["unet"]
    assert "minimax" in stack["clip"]
    assert "video" in stack["vae"]
    assert "audio" in stack["audio_vae"]
    assert "FL2VA" in stack["acc_file"]
    assert stack["sampler"] == "euler"
    assert stack["steps"] == 8
    snapped = coerce_video_stack(
        unet="minimax_h3_fl2va_pruned_w4a8_mixed.safetensors",
        clip="qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        vae="minimax_h3_video_vae_fp16.safetensors",
        audio_vae="minimax_h3_audio_vae_fp32.safetensors",
        acc_file="MiniMax-H3-FL2VA-Acc-8Step.safetensors",
        recipe="acc4",
        acc_files=["MiniMax-H3-FL2VA-Acc-8Step.safetensors"],
    )
    assert snapped["recipe"].key == "acc"
    assert snapped["steps"] == 8
    assert snapped["acc_file"].endswith("Acc-8Step.safetensors")


def test_acc_injects_apply_not_lora(repo_root):
    wf = apply_h3_graph(
        _bundled(repo_root),
        recipe="acc",
        acc_file="MiniMax-H3-FL2VA-Acc-8Step.safetensors",
        unet="minimax_h3_fl2va_pruned_w4a8_mixed.safetensors",
    )
    acc = wf["h3_acc"]
    assert acc["class_type"] == "MiniMaxH3PDDAccApply"
    assert acc["inputs"]["pdd_file"] == "MiniMax-H3-FL2VA-Acc-8Step.safetensors"
    assert acc["inputs"]["nfe"] == "8"
    assert acc["inputs"]["enabled"] is True
    assert acc["inputs"]["lora_strength"] == 1.0
    assert acc["inputs"]["on_off_grid"] == "error"
    assert acc["inputs"]["model"] == ["16", 0]
    assert acc["inputs"]["bypass_sigmas"] == ["10", 0]
    assert wf["7"]["inputs"]["model"] == ["h3_acc", 0]
    assert wf["11"]["inputs"]["sigmas"] == ["h3_acc", 1]
    assert wf["10"]["inputs"]["model"] == ["16", 0]
    assert wf["9"]["inputs"]["sampler_name"] == "euler"
    assert wf["10"]["inputs"]["steps"] == 8
    assert wf["16"]["inputs"]["shift_video"] == 12.0
    assert wf["16"]["inputs"]["shift_audio"] == 3.0
    assert not any(n.get("class_type") in {"LoraLoader", "LoraLoaderModelOnly"} for n in wf.values() if isinstance(n, dict))


def test_acc_no_fault_swaps_sampler_steps_shift_and_file(repo_root):
    src = _bundled(repo_root)
    src["9"]["inputs"]["sampler_name"] = "res_multistep"
    src["10"]["inputs"]["steps"] = 12
    src["16"]["inputs"]["shift_video"] = 6.0
    src["16"]["inputs"]["shift_audio"] = 1.0
    wf = apply_h3_graph(
        src,
        recipe="acc",
        acc_file="MiniMax-H3-Ref2VA-Acc-8Step.safetensors",
        unet="minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    )
    assert wf["9"]["inputs"]["sampler_name"] == "euler"
    assert wf["10"]["inputs"]["steps"] == 8
    assert wf["16"]["inputs"]["shift_video"] == 12.0
    assert wf["16"]["inputs"]["shift_audio"] == 3.0
    assert "FL2VA" in wf["h3_acc"]["inputs"]["pdd_file"]


def test_native_does_not_inject_acc(repo_root):
    wf = apply_h3_graph(_bundled(repo_root), recipe="native")
    assert "h3_acc" not in wf
    assert not any(n.get("class_type") == "MiniMaxH3PDDAccApply" for n in wf.values() if isinstance(n, dict))
    assert wf["7"]["inputs"]["model"] == ["16", 0]
    assert wf["11"]["inputs"]["sigmas"] == ["10", 0]
    assert wf["9"]["inputs"]["sampler_name"] == "res_multistep"
    assert wf["10"]["inputs"]["steps"] == 12
    assert wf["16"]["inputs"]["shift_video"] == 12.0


def test_native_disables_existing_acc(repo_root):
    acced = apply_h3_graph(_bundled(repo_root), recipe="acc")
    wf = apply_h3_graph(acced, recipe="native")
    assert wf["h3_acc"]["inputs"]["enabled"] is False
    assert wf["h3_acc"]["inputs"]["bypass_sigmas"] == ["10", 0]
    assert wf["11"]["inputs"]["sigmas"] == ["10", 0]
    assert wf["9"]["inputs"]["sampler_name"] == "res_multistep"
    assert wf["10"]["inputs"]["steps"] == 12


def test_h3recipe_object_does_not_snap_native_to_acc(repo_root):
    rec = recipe_settings("native")
    assert "acc=False" in str(rec)
    assert normalize_recipe(rec) == "native"
    assert recipe_settings(rec).key == "native"
    assert resolve_recipe(rec).key == "native"
    stack = coerce_video_stack(
        unet="minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        clip="qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        vae="minimax_h3_video_vae_fp16.safetensors",
        audio_vae="minimax_h3_audio_vae_fp32.safetensors",
        acc_file="MiniMax-H3-FL2VA-Acc-8Step.safetensors",
        recipe=rec,
    )
    assert stack["recipe"].key == "native"
    assert stack["steps"] == 12
    assert stack["sampler"] == "res_multistep"
    wf = apply_h3_graph(_bundled(repo_root), recipe=stack["recipe"], steps=12)
    assert wf["10"]["inputs"]["steps"] == 12
    assert wf["9"]["inputs"]["sampler_name"] == "res_multistep"
    assert not any(
        n.get("class_type") == "MiniMaxH3PDDAccApply" and n.get("inputs", {}).get("enabled")
        for n in wf.values()
        if isinstance(n, dict)
    )


def test_video_recipe_ui_clears_acc_on_native():
    files = ["MiniMax-H3-FL2VA-Acc-8Step.safetensors"]
    rec, steps, acc = video_recipe_ui(
        "native",
        "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        files,
        "MiniMax-H3-FL2VA-Acc-8Step.safetensors",
    )
    assert rec.key == "native"
    assert steps == 12
    assert acc == ""
    rec, steps, acc = video_recipe_ui(
        "acc",
        "minimax_h3_fl2va_pruned_w4a8_mixed.safetensors",
        files,
        "MiniMax-H3-FL2VA-Acc-8Step.safetensors",
    )
    assert rec.key == "acc"
    assert steps == 8
    assert acc.endswith("Acc-8Step.safetensors")


def test_latent_upscale_splits_av_and_rewires_decode(repo_root):
    wf = apply_h3_latent_upscale(apply_h3_graph(_bundled(repo_root), recipe="native"))
    assert wf["h3_av_split"]["class_type"] == "LTXVSeparateAVLatent"
    assert wf["h3_av_split"]["inputs"]["av_latent"] == ["11", 0]
    assert wf["h3_latent_up"]["class_type"] == "MinimaxH3LatentUpscaler3D"
    assert wf["h3_latent_up"]["inputs"]["latent"] == ["h3_av_split", 0]
    assert wf["h3_latent_up"]["inputs"]["mode"] == "scale by multiplier"
    assert wf["h3_latent_up"]["inputs"]["mode.scale"] == 2.0
    assert wf["12"]["inputs"]["samples"] == ["h3_latent_up", 0]
    assert wf["17"]["inputs"]["samples"] == ["h3_av_split", 1]


def test_rtx_vsr_graph_is_load_then_upscale():
    wf = rtx_vsr_graph("wrapper/rtx/clip.mp4", "h3_loop_rtx", scale=2.0)
    assert wf["1"]["class_type"] == "LoadVideo"
    assert wf["1"]["inputs"]["file"] == "wrapper/rtx/clip.mp4"
    assert wf["3"]["class_type"] == "RTXVideoSuperResolution"
    assert wf["3"]["inputs"]["resize_type"] == "scale by multiplier"
    assert wf["3"]["inputs"]["resize_type.scale"] == 2.0
    assert wf["4"]["inputs"]["audio"] == ["2", 1]
    assert wf["5"]["inputs"]["filename_prefix"] == "h3_loop_rtx"


def test_acc_strips_turbo_lora_not_character(repo_root):
    src = _bundled(repo_root)
    src["lora_c"] = {
        "class_type": "LoraLoaderModelOnly",
        "inputs": {
            "model": ["1", 0],
            "lora_name": "AsheIL-21.safetensors",
            "strength_model": 0.8,
        },
    }
    src["16"]["inputs"]["model"] = ["lora_c", 0]
    src["lora_t"] = {
        "class_type": "LoraLoaderModelOnly",
        "inputs": {
            "model": ["16", 0],
            "lora_name": "minimax_h3_fl2v_lightx2v_turbo_8step_v1.0.safetensors",
            "strength_model": 1.0,
        },
    }
    src["7"]["inputs"]["model"] = ["lora_t", 0]
    wf = apply_h3_graph(src, recipe="acc", unet="minimax_h3_fl2va_pruned_w4a8_mixed.safetensors")
    assert "lora_t" not in wf
    assert wf["lora_c"]["inputs"]["model"] == ["1", 0]
    assert wf["16"]["inputs"]["model"] == ["lora_c", 0]
    assert wf["h3_acc"]["inputs"]["model"] == ["16", 0]
    assert wf["7"]["inputs"]["model"] == ["h3_acc", 0]


def test_acc_disables_spectrum_and_drops_cfg_guider(repo_root):
    src = _bundled(repo_root)
    src["spec"] = {
        "class_type": "SpectrumApplyMiniMaxH3",
        "inputs": {"model": ["16", 0], "enabled": True, "blend_weight": 0.5},
    }
    src["7"]["class_type"] = "CFGGuider"
    src["7"]["inputs"] = {
        "model": ["spec", 0],
        "positive": ["6", 0],
        "negative": ["6", 0],
        "cfg": 3.5,
    }
    src["sage"] = {
        "class_type": "SolAttnPatch",
        "inputs": {"model": ["spec", 0], "tau": 1.3},
    }
    wf = apply_h3_graph(src, recipe="acc")
    assert wf["spec"]["inputs"]["enabled"] is False
    assert wf["spec"]["inputs"]["model"] == ["h3_acc", 0]
    assert wf["7"]["class_type"] == "BasicGuider"
    assert "cfg" not in wf["7"]["inputs"]
    assert "sage" not in wf


def test_acc4_does_not_rewrite_nfe_on_8step_file(repo_root):
    wf = apply_h3_graph(
        _bundled(repo_root),
        recipe="acc4",
        acc_file="MiniMax-H3-FL2VA-Acc-8Step.safetensors",
        unet="minimax_h3_fl2va_pruned_w4a8_mixed.safetensors",
    )
    assert wf["h3_acc"]["inputs"]["pdd_file"].endswith("Acc-8Step.safetensors")
    assert wf["h3_acc"]["inputs"]["nfe"] == "8"
    assert wf["10"]["inputs"]["steps"] == 8
    assert wf["9"]["inputs"]["sampler_name"] == "euler"


def test_acc4_uses_4_only_when_4step_weight_is_named(repo_root):
    wf = apply_h3_graph(
        _bundled(repo_root),
        recipe="acc4",
        acc_file="MiniMax-H3-FL2VA-Acc-4Step.safetensors",
        unet="minimax_h3_fl2va_pruned_w4a8_mixed.safetensors",
    )
    assert wf["h3_acc"]["inputs"]["nfe"] == "4"
    assert wf["h3_acc"]["inputs"]["pdd_file"].endswith("Acc-4Step.safetensors")


def test_video_config_recipe_override(tmp_path):
    cfg = from_dict({}, root=tmp_path)
    assert cfg.video.recipe == "acc"
    assert cfg.video.steps == 8
    cfg2 = apply_overrides(cfg, video_recipe="native")
    assert cfg2.video.recipe == "native"
    cfg3 = apply_overrides(cfg, latent_upscale=True, rtx_vsr=True)
    assert cfg3.video.latent_upscale is True
    assert cfg3.video.rtx_vsr is True


def test_apply_h3_does_not_mutate_source(repo_root):
    src = _bundled(repo_root)
    original = deepcopy(src)
    apply_h3_graph(src, recipe="acc")
    assert src == original
