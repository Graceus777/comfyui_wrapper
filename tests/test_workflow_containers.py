"""Recipe contracts that protect the old tabs and catch dangerous miswiring."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from comfyui_wrapper import workflow_containers as r
from comfyui_wrapper.container_runner import (
    bounded_image_size, cancel_owned, consecutive, run_container, stage_image,
    validate_graph,
)


def test_qwen_adapters_have_separate_recipes_and_reach_sampler():
    graphs = [
        (r.build_qwen_consistent("source.png", "Change the hat"), r.QWEN_CONSISTENCY),
        (r.build_qwen_outfit("person.png", "outfit.png", "Change the outfit"), r.QWEN_OUTFIT),
        (r.build_qwen_outpaint("source.png", (1024, 768)), r.QWEN_OUTPAINT),
    ]
    for graph, name in graphs:
        assert graph["adapter"]["inputs"]["lora_name"] == name
        assert graph["adapter"]["inputs"]["model"] == ["model", 0]
        assert graph["cache"]["inputs"]["model"] == ["adapter", 0]
        assert graph["sample"]["inputs"]["model"] == ["cache", 0]
        assert graph["sample"]["inputs"]["latent_image"] == ["encode", 2]
        assert graph["encode"]["inputs"]["resolution"] == 0
        assert graph["sample"]["inputs"]["steps"] == 25
        assert graph["rgb"]["class_type"] == "SplitImageWithAlpha"
    assert graphs[1][0]["encode"]["inputs"]["images.image_2"] == ["image_2", 0]


def test_reference_mask_uses_reference_number_and_explicit_white_black_mask():
    graph = r.build_qwen_controlled(["person.png", "coat.png"], "Change the (coat:1.2)",
                                    strengths=[1, 1.5], mask_key="mask.png", mask_index=2,
                                    mask_strength=2, isolate_mask=True)
    assert graph["encode"]["class_type"] == "QwenImage21ConditioningPhraseEncoder"
    assert graph["strength_2"]["inputs"]["reference_index"] == 2
    assert graph["mask"]["inputs"] == {"image": ["mask_image", 0], "channel": "red"}
    assert graph["mask_strength"]["inputs"]["mask_behavior"] == "zero_unmasked_tokens"
    assert graph["mask_strength"]["inputs"]["model"] == ["strength_2", 0]
    assert graph["cache"]["inputs"]["model"] == ["mask_strength", 0]
    with pytest.raises(ValueError, match="supplied image"):
        r.build_qwen_controlled(["source.png"], "Edit", mask_key="mask.png", mask_index=2)


def test_outpaint_keeps_source_coordinates_and_rounds_canvas_without_scaling_source():
    graph = r.build_qwen_outpaint("source.png", (1001, 777), margins=[128, 0, 0, 128])
    pad = graph["pad"]["inputs"]
    assert (1001 + pad["left"] + pad["right"]) % 32 == 0
    assert (777 + pad["top"] + pad["bottom"]) % 32 == 0
    assert pad["left"] == 128 and pad["top"] == 0
    assert pad["feathering"] == 0
    assert graph["encode"]["inputs"]["images.image_1"] == ["pad", 0]
    assert graph["restore"]["inputs"]["source"] == ["pad", 0]
    assert graph["restore"]["inputs"]["mask"] == ["keep_mask", 0]
    assert graph["keep_mask"]["inputs"]["mask"] == ["pad", 1]
    assert graph["restore"]["inputs"]["resize_source"] is False
    assert not any(n["class_type"] == "SetLatentNoiseMask" for n in graph.values())
    with pytest.raises(ValueError, match="Extend"):
        r.build_qwen_outpaint("source.png", (1024, 1024), margins=[0]*4)
    with pytest.raises(ValueError, match="size limit"):
        r.build_qwen_outpaint("source.png", (2048, 2048))


def test_lora_swap_is_native_ref2va_separate_from_turbo_sam3_swap():
    graph = r.build_h3_lora_swap("character.png", "source.mp4", "the person in red",
                                frames=124, delivered_frames=120)
    assert graph["model"]["inputs"]["unet_name"] == r.H3_REF2VA
    assert graph["adapter"]["inputs"]["lora_name"] == r.H3_SWAP
    assert graph["memory"]["inputs"]["model"] == ["adapter", 0]
    assert graph["memory"]["inputs"]["precision_mode"] == "Preserve native"
    assert graph["schedule"]["inputs"]["steps"] == 20
    assert graph["encode"]["inputs"]["ref_videos.ref_video_0"] == ["source_parts", 0]
    assert graph["decode"]["inputs"]["samples"] == ["sample", 1]
    assert graph["trim"]["inputs"]["length"] == 120
    assert graph["video"]["inputs"]["audio"] == ["source_parts", 1]
    assert not any("SAM3" in n["class_type"] for n in graph.values())
    assert not any("turbo" in str(n).lower() for n in graph.values())


def test_memory_i2v_preserves_regular_w4a8_and_optional_last_frame():
    graph = r.build_h3_memory_i2v("first.png", "Natural movement", frames=120, last_image_key="last.png")
    assert graph["model"]["inputs"]["unet_name"] == r.H3_FL2VA
    assert graph["encode"]["inputs"]["length"] == 124
    assert graph["encode"]["inputs"]["last_frame"] == ["last_fit", 0]
    assert graph["memory"]["inputs"]["qkv_streaming_mode"] == "Auto"
    assert graph["guider"]["inputs"]["model"] == graph["schedule"]["inputs"]["model"]


def test_omnichar_reference_count_voice_and_adapter_are_explicit():
    graph = r.build_h3_omnichar("wrapper/character.char", "She speaks softly.", 2,
                               use_voice=True, use_adapter=True)
    assert graph["character_decode"]["inputs"]["style"] == "token"
    assert graph["encode"]["inputs"]["prompt"] == ["character_decode", 4]
    assert graph["encode"]["inputs"]["ref_audios.ref_audio_0"] == ["character_decode", 5]
    assert "character_ref_2" not in graph
    assert graph["adapter"]["inputs"]["arch"] == "minimax-h3"
    assert graph["adapter"]["inputs"]["min_key_coverage"] == 0.9
    reference_only = r.build_h3_omnichar("wrapper/character.char", "She waves.", 1)
    assert "adapter" not in reference_only
    assert "ref_audios.ref_audio_0" not in reference_only["encode"]["inputs"]


def test_guided_upscale_uses_plain_model_own_lora_and_small_tiles():
    graph = r.build_h3_guided_upscale("source.mp4", delivered_frames=120)
    assert graph["upscale"]["class_type"] == "H3VideoUpsampler"
    assert graph["upscale"]["inputs"]["model"] == ["model", 0]
    assert graph["upscale"]["inputs"]["lora"] == r.H3_UPSCALE_LORA
    assert graph["upscale"]["inputs"]["tile_tokens"] == 25000
    assert graph["upscale"]["inputs"]["denoise"] == 0.1
    assert "memory" not in graph and "adapter" not in graph
    assert graph["video"]["inputs"]["images"] == ["trim", 0]
    assert graph["video"]["inputs"]["audio"] == ["source_parts", 1]


@pytest.mark.parametrize("count", [2,3,4,5])
def test_keyframes_use_each_adjacent_pair_once_and_keep_publisher_split(count):
    keys = [f"image{i}.png" for i in range(count)]
    graph = r.build_wan_keyframes(keys, ["Move smoothly"]*(count-1), [1.5]*(count-1), seed=42)
    assert graph["high_lora"]["inputs"]["lora_name"] == r.WAN_ANISORA
    assert graph["low_lora"]["inputs"]["lora_name"] == r.WAN_LIGHT
    for i in range(count-1):
        pair = f"pair_{i}"
        encode = graph[pair+"_encode"]["inputs"]
        assert encode["start_image"] == [f"keyframe_{i}", 0]
        assert encode["end_image"] == [f"keyframe_{i+1}", 0]
        assert encode["length"] == 37
        assert graph[pair+"_high"]["inputs"]["end_at_step"] == 2
        assert graph[pair+"_low"]["inputs"]["start_at_step"] == 2
        assert graph[pair+"_low"]["inputs"]["add_noise"] == "disable"
        assert graph[pair+"_trim"]["inputs"]["batch_index"] == (0 if i == 0 else 1)
    assert f"pair_{count-1}_encode" not in graph
    assert sum(v["inputs"]["length"] for k,v in graph.items() if k.endswith("_trim")) == 36*(count-1)+1


def test_input_gaps_and_limits_fail_before_sampling():
    with pytest.raises(ValueError, match="without empty slots"):
        consecutive(["one.png", None, "three.png"], minimum=2, label="keyframes")
    with pytest.raises(ValueError, match="Describe"):
        r.build_qwen_consistent("source.png", " ")
    with pytest.raises(ValueError, match="size limit"):
        r.build_wan_keyframes(["1.png","2.png"], ["Move"], [1], width=1280, height=736)
    with pytest.raises(ValueError, match="0.5"):
        r.wan_frame_count(float("nan"))
    assert bounded_image_size((4096, 2048)) == (1440, 704)


def test_staged_outpaint_preserves_original_pixels(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    source, destination = tmp_path/"original.png", tmp_path/"staged.png"
    image = Image.new("RGB", (65, 97), (19, 131, 202))
    image.putpixel((32, 48), (255, 0, 123))
    image.save(source)
    assert stage_image(source, destination, resize=False) == (65, 97)
    with Image.open(destination) as staged:
        assert staged.tobytes() == image.tobytes()


def test_preflight_understands_dynamic_ports_and_rejects_missing_models_and_wrong_link_types():
    info = {
        "Loader": {"input":{"required":{"name":[["installed"],{}]}},"output":["IMAGE"]},
        "Encoder": {"input":{"required":{
            "images":["COMFY_AUTOGROW_V3",{"template":{"input":{"required":{"image":["IMAGE",{}]}},"prefix":"image_","min":1,"max":3}}],
        }},"output":["CONDITIONING"]},
    }
    graph = {"source":r.node("Loader", name="installed"), "encode":r.node("Encoder", **{"images.image_0":["source",0]})}
    assert not validate_graph(graph, info)
    graph["source"]["inputs"]["name"] = "missing"
    assert any("missing" in e for e in validate_graph(graph, info))
    graph["encode"]["inputs"]["images.image_0"] = ["encode", 0]
    assert any("cannot feed IMAGE" in e for e in validate_graph(graph, info))


def test_busy_comfy_queue_prevents_staging_or_submission(tmp_path):
    class Client:
        def ping(self): return True
        def queue(self): return {"queue_running":[[1,"other-job"]]}
        def queue_prompt(self, graph): pytest.fail("Should not queue a second job")
    cfg = SimpleNamespace(paths=SimpleNamespace(output=tmp_path))
    with pytest.raises(Exception, match="already running"):
        run_container("qwen_consistent", {}, cfg=cfg, client=Client())
    assert list(tmp_path.iterdir()) == []


def test_uploaded_input_paths_need_not_appear_in_top_level_file_dropdown():
    # Native LoadVideo/LoadImage VALIDATE_INPUTS checks path existence; the
    # server's UI dropdown lists only its top-level input files.
    info = {
        "LoadVideo":{"input":{"required":{"file":["COMBO",{"options":["top-level.mp4"]}]}},"output":["VIDEO"]},
        "LoadImage":{"input":{"required":{"image":[["top-level.png"],{}]}},"output":["IMAGE","MASK"]},
        "UNETLoader":{"input":{"required":{"unet_name":[["installed.safetensors"],{}]}},"output":["MODEL"]},
    }
    graph = {"source":r.node("LoadVideo", file="wrapper/containers/source.mp4"),
             "image":r.node("LoadImage", image="wrapper/containers/reference.png")}
    assert not validate_graph(graph, info)
    graph["model"] = r.node("UNETLoader", unet_name="missing.safetensors")
    assert validate_graph(graph, info) == ["model.unet_name: required choice/model is unavailable: missing.safetensors"]


def test_cancel_only_interrupts_owned_job():
    class Client:
        def __init__(self, running): self.running, self.calls = running, []
        def queue(self): return {"queue_running":[[1,self.running]],"queue_pending":[[2,"queued-owned"]]}
        def interrupt(self): self.calls.append("interrupt")
        def post(self, path, payload): self.calls.append((path,payload))
    client = Client("someone-else")
    cancel_owned(client, "finished-owned")
    assert not client.calls
    cancel_owned(client, "queued-owned")
    assert client.calls == [("/queue",{"delete":["queued-owned"]})]
    client.running = "owned"
    cancel_owned(client, "owned")
    assert client.calls[-1] == "interrupt"


def test_existing_swap_and_qwen_assets_are_not_mutated_by_container_builders():
    root = Path(__file__).resolve().parents[1]/"workflows"
    paths = [root/"minimax_h3_character_swap.api.json", root/"qwen_image_2_1.api.json"]
    before = [p.read_bytes() for p in paths]
    r.build_h3_lora_swap("image.png", "video.mp4", "the main person")
    r.build_qwen_consistent("image.png", "Change hat")
    assert before == [p.read_bytes() for p in paths]
    old_swap = json.loads(before[0])
    assert old_swap["h3_treat"]["inputs"]["source"] == ["h3_invert",0]
