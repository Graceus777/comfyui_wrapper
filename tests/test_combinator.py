from pathlib import Path

from comfyui_wrapper.combinator import (
    Job,
    LoRAEntry,
    PromptConfig,
    ZoneConfig,
    cartesian_slots,
    fill_slots,
    load_jobs,
    pick_random,
    prep_jobs,
    preview_jobs,
    save_jobs,
    weighted_sample,
)
from comfyui_wrapper.config import from_dict
from comfyui_wrapper.history import compute_generation_hash, existing_hashes, prefix_has_output
from comfyui_wrapper.lora import LoRAResolver


def test_cartesian_slots_only_uses_placeholders():
    template = "1girl, {pose}, white background"
    combos = cartesian_slots(
        {"pose": ["standing", "kneeling"], "view": ["side", "front"]},
        template,
    )
    assert combos == [{"pose": "standing"}, {"pose": "kneeling"}]


def test_cartesian_slots_multi():
    combos = cartesian_slots(
        {"pose": ["a", "b"], "view": ["x"]},
        "{pose} {view}",
    )
    assert len(combos) == 2
    assert {"pose": "a", "view": "x"} in combos


def test_fill_slots_leaves_unknown():
    assert fill_slots("a {pose} {missing}", {"pose": "sit"}) == "a sit {missing}"


def test_weighted_sample_without_replacement():
    pool = ["a", "b", "c"]
    picked = weighted_sample(pool, [10, 1, 1], k=2)
    assert len(picked) == 2
    assert len(set(picked)) == 2
    assert set(picked) <= set(pool)


def test_prep_jobs_loop_and_slots(tmp_path: Path):
    cfg = from_dict(
        {
            "paths": {
                "history": str(tmp_path / "history.jsonl"),
                "output": str(tmp_path / "out"),
            },
            "generation": {"steps": 20, "cfg": 5, "sampler": "euler", "width": 1024, "height": 1280},
            "batch": {"batch_count": 1, "skip_exists": False, "random_count": 0},
        },
        root=tmp_path,
    )
    zone = ZoneConfig(
        loop={
            "char_a": LoRAEntry("char_a", 0.8, "tag_a"),
            "char_b": LoRAEntry("char_b", 0.8, "tag_b"),
        }
    )
    prompt = PromptConfig(
        positive="1girl, {pose}",
        negative="bad",
        slots={"pose": ["standing", "kneeling"]},
    )
    resolver = LoRAResolver(files=["char_a.safetensors", "char_b.safetensors"])
    jobs = prep_jobs(cfg, zone=zone, prompt=prompt, resolver=resolver)
    assert len(jobs) == 4
    prefixes = {j.filename_prefix for j in jobs}
    assert any("char_a" in p and "standing" in p for p in prefixes)
    assert any("char_b" in p and "kneeling" in p for p in prefixes)
    a_stand = next(j for j in jobs if j.loop_name == "char_a" and j.slots.get("pose") == "standing")
    assert "tag_a" in a_stand.positive
    assert "standing" in a_stand.positive
    assert a_stand.loras[0]["file"] == "char_a.safetensors"
    assert "<lora:" not in a_stand.positive


def test_prep_strips_prompt_lora_tags(tmp_path: Path):
    cfg = from_dict(
        {
            "paths": {"history": str(tmp_path / "h.jsonl")},
            "generation": {"steps": 20, "cfg": 5, "sampler": "euler", "width": 8, "height": 8},
            "batch": {"skip_exists": False, "batch_count": 1},
        },
        root=tmp_path,
    )
    zone = ZoneConfig(loop={"char_a": LoRAEntry("char_a", 0.8, "tag_a")})
    prompt = PromptConfig(positive="1girl, <lora:sdt-hair1:1> sdchan", negative="bad")
    resolver = LoRAResolver(files=["char_a.safetensors", "sdt-hair1.safetensors"])
    jobs = prep_jobs(cfg, zone=zone, prompt=prompt, resolver=resolver)
    assert "<lora:" not in jobs[0].positive
    names = {x["name"] for x in jobs[0].loras}
    assert names == {"char_a", "sdt-hair1"}
    assert "sdchan" in jobs[0].positive


def test_prep_skip_exists_by_output_file(tmp_path: Path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "char_a_b1_00001_.png").write_bytes(b"png")
    zone = ZoneConfig(loop={"char_a": LoRAEntry("char_a", 0.8, "tag_a")})
    prompt = PromptConfig(positive="1girl", negative="bad")
    resolver = LoRAResolver(files=["char_a.safetensors"])
    cfg = from_dict(
        {
            "paths": {"history": str(tmp_path / "history.jsonl"), "output": str(out)},
            "generation": {"steps": 20, "cfg": 5, "sampler": "euler", "width": 8, "height": 8},
            "batch": {"skip_exists": True, "batch_count": 1, "random_count": 0},
        },
        root=tmp_path,
    )
    jobs = prep_jobs(cfg, zone=zone, prompt=prompt, resolver=resolver)
    assert jobs[0].filename_prefix == "char_a_b1"
    assert jobs[0].status == "skipped"


def test_prep_skip_exists_duplicate_hash_in_batch(tmp_path: Path):
    zone = ZoneConfig(loop={"char_a": LoRAEntry("char_a", 0.8, "tag_a")})
    prompt = PromptConfig(positive="1girl", negative="bad")
    resolver = LoRAResolver(files=["char_a.safetensors"])
    cfg = from_dict(
        {
            "paths": {"history": str(tmp_path / "history.jsonl"), "output": str(tmp_path / "out")},
            "generation": {"steps": 20, "cfg": 5, "sampler": "euler", "width": 8, "height": 8},
            "batch": {"skip_exists": True, "batch_count": 2, "random_count": 0},
        },
        root=tmp_path,
    )
    jobs = prep_jobs(cfg, zone=zone, prompt=prompt, resolver=resolver)
    assert len(jobs) == 2
    assert jobs[0].status == "pending"
    assert jobs[1].status == "skipped"
    assert jobs[0].gen_hash == jobs[1].gen_hash


def test_run_jobs_skips_existing_output(tmp_path: Path, monkeypatch):
    from comfyui_wrapper.combinator import run_jobs

    out = tmp_path / "out"
    out.mkdir()
    (out / "char_a_b1_00001_.png").write_bytes(b"png")
    cfg = from_dict(
        {
            "paths": {"history": str(tmp_path / "history.jsonl"), "output": str(out)},
            "batch": {"skip_exists": True, "cooldown_s": 0},
        },
        root=tmp_path,
    )
    jobs = [
        Job(
            id="0001",
            positive="1girl",
            negative="bad",
            loras=[{"name": "char_a", "file": "char_a.safetensors", "weight": 0.8}],
            filename_prefix="char_a_b1",
            gen_hash="deadbeefdeadbeef",
            status="pending",
        )
    ]
    called = {"n": 0}

    def fake_generate(**_kwargs):
        called["n"] += 1
        raise AssertionError("generate should not run for an existing output")

    monkeypatch.setattr("comfyui_wrapper.combinator.generate", fake_generate)
    run_jobs(jobs, cfg, on_log=lambda _m: None)
    assert jobs[0].status == "skipped"
    assert called["n"] == 0


def test_prep_skip_exists(tmp_path: Path):
    history = tmp_path / "history.jsonl"
    zone = ZoneConfig(loop={"char_a": LoRAEntry("char_a", 0.8, "tag_a")})
    prompt = PromptConfig(positive="1girl", negative="bad")
    resolver = LoRAResolver(files=["char_a.safetensors"])
    cfg = from_dict(
        {
            "paths": {"history": str(history)},
            "generation": {"steps": 20, "cfg": 5, "sampler": "euler", "width": 8, "height": 8},
            "batch": {"skip_exists": True, "batch_count": 1},
        },
        root=tmp_path,
    )
    jobs = prep_jobs(cfg, zone=zone, prompt=prompt, resolver=resolver)
    assert jobs[0].status == "pending"
    digest = jobs[0].gen_hash
    history.write_text('{"gen_hash": "%s"}\n' % digest, encoding="utf-8")
    jobs2 = prep_jobs(cfg, zone=zone, prompt=prompt, resolver=resolver)
    assert jobs2[0].status == "skipped"
    assert jobs2[0].gen_hash == digest


def test_job_roundtrip(tmp_path: Path):
    path = tmp_path / "jobs.json"
    jobs = [
        Job(
            id="0001",
            positive="hello",
            negative="no",
            loras=[{"name": "a", "file": "a.safetensors", "weight": 0.8, "activation": "aa"}],
            loop_name="a",
            gen_hash="abc",
        )
    ]
    save_jobs(jobs, path, extra={"zone": "z"})
    loaded, data = load_jobs(path)
    assert loaded[0].positive == "hello"
    assert loaded[0].loras[0]["name"] == "a"
    assert data["count"] == 1


def test_preview_contains_counts():
    jobs = [
        Job(id="1", positive="p", negative="", loras=[], status="pending"),
        Job(id="2", positive="p", negative="", loras=[], status="skipped"),
    ]
    text = preview_jobs(jobs)
    assert "2 job(s)" in text
    assert "1 pending" in text


def test_prefix_has_output_does_not_match_b10(tmp_path: Path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "char_a_b10_00001_.png").write_bytes(b"png")
    assert prefix_has_output("char_a_b10", out) is True
    assert prefix_has_output("char_a_b1", out) is False


def test_hash_stable_excludes_seed():
    a = compute_generation_hash("p", "n", steps=20, sampler="euler", cfg=5, width=8, height=8, loras=["a"])
    b = compute_generation_hash("p", "n", steps=20, sampler="euler", cfg=5, width=8, height=8, loras=["a"])
    c = compute_generation_hash("p", "n", steps=21, sampler="euler", cfg=5, width=8, height=8, loras=["a"])
    assert a == b
    assert a != c
    assert len(a) == 16
    assert existing_hashes([{"gen_hash": a}]) == {a}


def test_pick_random_respects_count():
    zone = ZoneConfig(
        random={
            "a": LoRAEntry("a"),
            "b": LoRAEntry("b"),
            "c": LoRAEntry("c"),
        }
    )
    picks = pick_random(zone, 2, freq={})
    assert len(picks) == 2
    assert len({p.name for p in picks}) == 2
