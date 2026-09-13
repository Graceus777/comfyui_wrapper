from pathlib import Path

from comfyui_wrapper.config import DEFAULTS, apply_overrides, deep_merge, from_dict, load_config
from comfyui_wrapper.lora import LoRAResolver, parse_lora_flag, refs_from_config_list


def test_deep_merge_nested():
    out = deep_merge({"a": {"b": 1, "c": 2}, "d": 3}, {"a": {"c": 9}, "e": 4})
    assert out == {"a": {"b": 1, "c": 9}, "d": 3, "e": 4}


def test_from_dict_applies_defaults(tmp_path: Path):
    cfg = from_dict({"generation": {"steps": 12}}, root=tmp_path)
    assert cfg.generation.steps == 12
    assert cfg.generation.sampler == DEFAULTS["generation"]["sampler"]
    assert cfg.comfyui.port == 8188


def test_apply_overrides(tmp_path: Path):
    cfg = from_dict({}, root=tmp_path)
    cfg2 = apply_overrides(cfg, steps=40, host="10.0.0.2", cfg_scale=3.5)
    assert cfg2.generation.steps == 40
    assert cfg2.generation.cfg == 3.5
    assert cfg2.comfyui.host == "10.0.0.2"
    assert cfg.generation.steps == DEFAULTS["generation"]["steps"]


def test_hires_adetailer_overrides(tmp_path: Path):
    cfg = from_dict({}, root=tmp_path)
    cfg2 = apply_overrides(
        cfg,
        hires_enable=True,
        hires_scale=2.0,
        adetailer_enable=True,
        adetailer_model="bbox/face_yolov8n.pt",
    )
    assert cfg2.generation.hires_enable is True
    assert cfg2.generation.hires_scale == 2.0
    assert cfg2.generation.adetailer_enable is True
    assert cfg2.generation.adetailer_model == "bbox/face_yolov8n.pt"
    assert cfg.generation.hires_enable is False


def test_load_config_missing(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = load_config(tmp_path / "nope.yaml")
    assert cfg.generation.steps == 28


def test_lora_resolver_stem_and_ambiguous():
    r = LoRAResolver(files=["AsheIL-21.safetensors", "Characters/Ahri.safetensors", "style/Ahri.safetensors"])
    assert r.resolve_file("AsheIL-21") == "AsheIL-21.safetensors"
    assert r.resolve_file("AsheIL-21.safetensors") == "AsheIL-21.safetensors"
    try:
        r.resolve_file("Ahri")
        assert False, "expected ambiguity error"
    except ValueError as exc:
        assert "Ambiguous" in str(exc)


def test_lora_resolver_offline_suffix():
    r = LoRAResolver(files=[])
    assert r.resolve_file("foo") == "foo.safetensors"


def test_extract_lora_tags():
    from comfyui_wrapper.lora import extract_lora_tags
    cleaned, found = extract_lora_tags("lazypos, <lora:sdt-hair1:1> sdchan, 1girl")
    assert found == [("sdt-hair1", 1.0)]
    assert "<lora:" not in cleaned
    assert "sdchan" in cleaned
    assert "lazypos" in cleaned


def test_parse_lora_flag():
    assert parse_lora_flag("name") == ("name", 1.0)
    assert parse_lora_flag("name:0.8") == ("name", 0.8)


def test_refs_from_config_list():
    r = LoRAResolver(files=["a.safetensors"])
    refs = refs_from_config_list(["a:0.7", {"name": "a", "weight": 0.5, "activation": "hi"}], r)
    assert refs[0].weight == 0.7
    assert refs[1].activation == "hi"


def test_activation_from_lora_texts(tmp_path: Path):
    texts = tmp_path / "lora_texts"
    texts.mkdir()
    (texts / "char.json").write_text('{"activation text": "foo bar"}', encoding="utf-8")
    cfg = from_dict({}, root=tmp_path)
    r = LoRAResolver(files=["char.safetensors"], cfg=cfg)
    assert r.activation_text("char") == "foo bar"
