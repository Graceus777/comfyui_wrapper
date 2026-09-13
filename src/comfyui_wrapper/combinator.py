"""Zone combinator, prompt-slot cartesian expansion, job prep and batch run."""

from __future__ import annotations

import json
import random
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Callable

from comfyui_wrapper.config import WrapperConfig
from comfyui_wrapper.generate import GenerateResult, generate
from comfyui_wrapper.history import (
    compute_generation_hash,
    existing_hashes,
    frequency_map,
    load_history,
    prefix_has_output,
)
from comfyui_wrapper.lora import LoRARef, LoRAResolver, extract_lora_tags

SLOT_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


@dataclass
class LoRAEntry:
    name: str
    weight: float = 1.0
    activation: str = ""


@dataclass
class ZoneConfig:
    always: dict[str, LoRAEntry] = field(default_factory=dict)
    loop: dict[str, LoRAEntry] = field(default_factory=dict)
    random: dict[str, LoRAEntry] = field(default_factory=dict)
    name: str = ""


@dataclass
class PromptConfig:
    positive: str = ""
    negative: str = ""
    slots: dict[str, list[str]] = field(default_factory=dict)
    name: str = ""


@dataclass
class Job:
    id: str
    positive: str
    negative: str
    loras: list[dict]
    loop_name: str | None = None
    slots: dict[str, str] = field(default_factory=dict)
    seed: int = -1
    filename_prefix: str = "wrapper"
    gen_hash: str = ""
    status: str = "pending"  # pending | skipped | running | done | error
    error: str = ""
    files: list[str] = field(default_factory=list)
    prompt_id: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Job":
        kwargs = {}
        for name, fdef in cls.__dataclass_fields__.items():
            if name in data and data[name] is not None:
                kwargs[name] = data[name]
        kwargs.setdefault("id", "0000")
        kwargs.setdefault("positive", "")
        kwargs.setdefault("negative", "")
        kwargs.setdefault("loras", [])
        return cls(**kwargs)  # type: ignore[arg-type]


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _entries(raw: dict | None) -> dict[str, LoRAEntry]:
    out: dict[str, LoRAEntry] = {}
    for name, data in (raw or {}).items():
        if isinstance(data, dict):
            out[name] = LoRAEntry(
                name=name,
                weight=float(data.get("weight", 1.0)),
                activation=str(data.get("activation") or ""),
            )
        else:
            out[name] = LoRAEntry(name=name, weight=float(data), activation="")
    return out


def find_named_json(dirs: list[Path], name: str) -> Path | None:
    stem = name[:-5] if name.endswith(".json") else name
    for folder in dirs:
        if not folder.exists():
            continue
        direct = folder / f"{stem}.json"
        if direct.is_file():
            return direct
    return None


def list_named_json(dirs: list[Path]) -> list[str]:
    names: set[str] = set()
    for folder in dirs:
        if not folder.exists():
            continue
        for path in folder.glob("*.json"):
            if path.is_file():
                names.add(path.stem)
    return sorted(names)


def load_zone_config(name: str, cfg: WrapperConfig) -> ZoneConfig:
    path = find_named_json(cfg.paths.zone_dirs(), name)
    if path is None:
        available = ", ".join(list_named_json(cfg.paths.zone_dirs())) or "(none)"
        raise FileNotFoundError(f"Zone config {name!r} not found. Available: {available}")
    data = _load_json(path)
    return ZoneConfig(
        always=_entries(data.get("always")),
        loop=_entries(data.get("loop")),
        random=_entries(data.get("random")),
        name=path.stem,
    )


def load_prompt_config(name: str, cfg: WrapperConfig) -> PromptConfig:
    path = find_named_json(cfg.paths.prompt_dirs(), name)
    if path is None:
        available = ", ".join(list_named_json(cfg.paths.prompt_dirs())) or "(none)"
        raise FileNotFoundError(f"Prompt config {name!r} not found. Available: {available}")
    data = _load_json(path)
    slots = data.get("slots") or {}
    slots = {str(k): [str(x) for x in v] for k, v in slots.items()}
    return PromptConfig(
        positive=str(data.get("positive_prompt") or data.get("positive") or ""),
        negative=str(data.get("negative_prompt") or data.get("negative") or ""),
        slots=slots,
        name=path.stem,
    )


def weighted_sample(pool: list, weights: list[float], k: int) -> list:
    pool = list(pool)
    weights = list(weights)
    selected = []
    for _ in range(min(k, len(pool))):
        if not pool:
            break
        total = sum(weights)
        if total <= 0:
            idx = random.randrange(len(pool))
        else:
            r = random.uniform(0, total)
            cumulative = 0.0
            idx = 0
            for i, w in enumerate(weights):
                cumulative += w
                if cumulative >= r:
                    idx = i
                    break
        selected.append(pool[idx])
        pool.pop(idx)
        weights.pop(idx)
    return selected


def cartesian_slots(slots: dict[str, list[str]], template: str) -> list[dict[str, str]]:
    """Expand only placeholders that actually appear in the template."""
    used = [m for m in SLOT_RE.findall(template) if m in slots and slots[m]]
    if not used:
        return [{}]
    keys = used
    values = [slots[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in product(*values)]


def fill_slots(template: str, values: dict[str, str]) -> str:
    def repl(match: re.Match) -> str:
        key = match.group(1)
        return values[key] if key in values else match.group(0)
    return SLOT_RE.sub(repl, template)


def safe_basename(value: str) -> str:
    invalid = set('<>:"/|?*') | {"\\", " "}
    cleaned = "".join("_" if ch in invalid else ch for ch in str(value or "image"))
    cleaned = re.sub(r"_+", "_", cleaned)
    return cleaned.strip(" ._")[:80] or "image"


def pick_random(
    zone: ZoneConfig,
    count: int,
    freq: dict[str, int],
) -> list[LoRAEntry]:
    pool = list(zone.random.values())
    if not pool or count <= 0:
        return []
    k = min(int(count), len(pool))
    if freq:
        max_f = max((freq.get(e.name, 0) for e in pool), default=0) + 1
        weights = [float(max_f - freq.get(e.name, 0) + 1) for e in pool]
        return weighted_sample(pool, weights, k)
    return random.sample(pool, k)


def _compose_prompt(
    base_positive: str,
    zone: ZoneConfig,
    loop_entry: LoRAEntry | None,
    random_picks: list[LoRAEntry],
    slot_values: dict[str, str],
    *,
    include_lora_tags: bool,
    resolver: LoRAResolver,
) -> tuple[str, list[LoRARef]]:
    entries: list[LoRAEntry] = list(zone.always.values())
    if loop_entry is not None:
        entries.append(loop_entry)
    entries.extend(random_picks)

    refs: list[LoRARef] = []
    activations: list[str] = []
    tags: list[str] = []
    for entry in entries:
        ref = resolver.resolve(entry.name, entry.weight, entry.activation)
        refs.append(ref)
        if ref.activation:
            activations.append(ref.activation)
        if include_lora_tags:
            tags.append(f"<lora:{ref.name}:{ref.weight}>")

    positive = fill_slots(base_positive, slot_values)
    positive, tagged = extract_lora_tags(positive)
    seen = {r.name.lower() for r in refs}
    for name, weight in tagged:
        if name.lower() in seen:
            continue
        refs.append(resolver.resolve(name, weight))
        seen.add(name.lower())
        if include_lora_tags:
            tags.append(f"<lora:{name}:{weight}>")
    parts = tags + activations
    if positive:
        parts.append(positive)
    return ", ".join(parts), refs


def prep_jobs(
    cfg: WrapperConfig,
    *,
    zone_name: str | None = None,
    prompt_name: str | None = None,
    zone: ZoneConfig | None = None,
    prompt: PromptConfig | None = None,
    resolver: LoRAResolver | None = None,
    extra_slots: dict[str, list[str]] | None = None,
    only: list[str] | None = None,
    exclude: list[str] | None = None,
    limit: int | None = None,
    offset: int = 0,
    seed: int | None = None,
) -> list[Job]:
    zone = zone or (load_zone_config(zone_name, cfg) if zone_name else ZoneConfig())
    prompt = prompt or (
        load_prompt_config(prompt_name, cfg)
        if prompt_name
        else PromptConfig(positive=cfg.generation.positive, negative=cfg.generation.negative)
    )
    resolver = resolver or LoRAResolver(cfg=cfg)

    slots = dict(prompt.slots)
    if extra_slots:
        slots.update(extra_slots)

    history = load_history(cfg.paths.history)
    hashes = existing_hashes(history)
    freq = frequency_map(history)

    loop_items: list[LoRAEntry | None]
    if zone.loop:
        items = list(zone.loop.values())
        if only:
            allow = {n.lower() for n in only}
            items = [e for e in items if e.name.lower() in allow]
        if exclude:
            deny = {n.lower() for n in exclude}
            items = [e for e in items if e.name.lower() not in deny]
        if cfg.batch.least_used_first:
            items = sorted(items, key=lambda e: freq.get(e.name, 0))
        loop_items = items or []
    else:
        loop_items = [None]

    if not loop_items and not zone.always and not zone.random:
        raise ValueError("No LoRAs configured. Pass --zone or put LoRAs in always/loop/random.")

    sample_at_prep = str(cfg.batch.sample_random_at).lower() != "run"
    slot_combos = cartesian_slots(slots, prompt.positive)
    gen = cfg.generation
    jobs: list[Job] = []
    job_index = 0
    cap = (offset + int(limit)) if limit is not None else None

    for loop_entry in loop_items:
        for slot_values in slot_combos:
            for batch_idx in range(int(cfg.batch.batch_count)):
                random_picks = (
                    pick_random(zone, cfg.batch.random_count, freq)
                    if sample_at_prep
                    else []
                )
                # If sampling at run, still reserve a placeholder job; picks happen later.
                positive, refs = _compose_prompt(
                    prompt.positive,
                    zone,
                    loop_entry,
                    random_picks,
                    slot_values,
                    include_lora_tags=cfg.generation.include_lora_tags,
                    resolver=resolver,
                )
                negative = fill_slots(prompt.negative, slot_values)
                name_parts = []
                if loop_entry:
                    name_parts.append(loop_entry.name)
                if slot_values:
                    name_parts.extend(str(v) for v in slot_values.values())
                name_parts.append(f"b{batch_idx + 1}")
                prefix = safe_basename("_".join(name_parts) if name_parts else cfg.generation.filename_prefix)
                gen_hash = compute_generation_hash(
                    positive,
                    negative,
                    steps=gen.steps,
                    sampler=gen.sampler,
                    cfg=gen.cfg,
                    width=gen.width,
                    height=gen.height,
                    checkpoint=gen.checkpoint,
                    loras=[r.name for r in refs],
                )
                exists = cfg.batch.skip_exists and (
                    gen_hash in hashes or prefix_has_output(prefix, cfg.paths.output, history)
                )
                status = "skipped" if exists else "pending"
                if cfg.batch.skip_exists:
                    hashes.add(gen_hash)
                job_index += 1
                jobs.append(
                    Job(
                        id=f"{job_index:04d}",
                        positive=positive,
                        negative=negative,
                        loras=[r.to_dict() for r in refs],
                        loop_name=loop_entry.name if loop_entry else None,
                        slots=dict(slot_values),
                        seed=gen.seed if seed is None else seed,
                        filename_prefix=prefix,
                        gen_hash=gen_hash,
                        status=status,
                    )
                )
                if cap is not None and len(jobs) >= cap:
                    break
            if cap is not None and len(jobs) >= cap:
                break
        if cap is not None and len(jobs) >= cap:
            break

    if offset:
        jobs = jobs[offset:]
    if limit is not None:
        jobs = jobs[: int(limit)]
    return jobs


def preview_jobs(jobs: list[Job]) -> str:
    pending = [j for j in jobs if j.status == "pending"]
    skipped = [j for j in jobs if j.status == "skipped"]
    lines = [
        f"=== {len(jobs)} job(s): {len(pending)} pending, {len(skipped)} skipped ===",
        "",
    ]
    for job in jobs:
        lora_s = ", ".join(f"{x['name']}:{x['weight']}" for x in job.loras) or "(none)"
        slot_s = ", ".join(f"{k}={v}" for k, v in job.slots.items())
        lines.append(f"[{job.id}] {job.status}  {job.filename_prefix}")
        if slot_s:
            lines.append(f"    slots: {slot_s}")
        lines.append(f"    loras: {lora_s}")
        lines.append(f"    prompt: {job.positive[:160]}{'...' if len(job.positive) > 160 else ''}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def save_jobs(jobs: list[Job], path: str | Path, extra: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "count": len(jobs),
        "pending": sum(1 for j in jobs if j.status == "pending"),
        "skipped": sum(1 for j in jobs if j.status == "skipped"),
        "meta": extra or {},
        "jobs": [j.to_dict() for j in jobs],
    }
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    return path


def load_jobs(path: str | Path) -> tuple[list[Job], dict]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, list):
        return [Job.from_dict(x) for x in data], {}
    jobs = [Job.from_dict(x) for x in data.get("jobs") or []]
    return jobs, data


def run_jobs(
    jobs: list[Job],
    cfg: WrapperConfig,
    *,
    client=None,
    resolver: LoRAResolver | None = None,
    on_log: Callable[[str], None] | None = None,
    on_job: Callable[[Job], None] | None = None,
    resume: bool = True,
    stop_flag: Callable[[], bool] | None = None,
) -> list[Job]:
    """Run pending jobs sequentially against ComfyUI. Mutates and returns jobs."""

    def log(msg: str) -> None:
        if on_log:
            on_log(msg)

    pending = [j for j in jobs if j.status == "pending" or (not resume and j.status in {"error", "running"})]
    history = load_history(cfg.paths.history)
    hashes = existing_hashes(history)
    total = len(pending)
    log(f"Running {total} pending job(s) of {len(jobs)} total")
    if cfg.batch.skip_exists:
        log(f"Skip exists: ON ({len(hashes)} history hashes)")

    for i, job in enumerate(pending, start=1):
        if stop_flag and stop_flag():
            log("Stop requested; remaining jobs left pending.")
            break
        if cfg.batch.skip_exists and (
            (job.gen_hash and job.gen_hash in hashes)
            or prefix_has_output(job.filename_prefix, cfg.paths.output, history)
        ):
            job.status = "skipped"
            log(f"\n--- [{job.id}] {i}/{total} {job.filename_prefix} SKIPPED (already exists) ---")
            if on_job:
                try:
                    on_job(job)
                except Exception as cb_exc:
                    log(f"on_job callback error: {cb_exc}")
            continue
        job.status = "running"
        log(f"\n--- [{job.id}] {i}/{total} {job.filename_prefix} ---")
        log(f"Prompt: {job.positive[:200]}")
        try:
            result: GenerateResult = generate(
                prompt=job.positive,
                negative=job.negative,
                seed=job.seed,
                filename_prefix=job.filename_prefix,
                loras=job.loras,
                cfg=cfg,
                client=client,
                resolver=resolver,
                record_history=True,
                wait=True,
            )
            job.status = "done"
            job.prompt_id = result.prompt_id
            job.files = result.saved_files or [img.get("filename", "") for img in result.images]
            job.gen_hash = result.gen_hash or job.gen_hash
            if job.gen_hash:
                hashes.add(job.gen_hash)
            history.append(
                {
                    "gen_hash": job.gen_hash,
                    "filename_prefix": job.filename_prefix,
                    "files": [Path(p).name for p in job.files],
                }
            )
            log(f"Done: {', '.join(job.files) or result.prompt_id}")
        except Exception as exc:
            job.status = "error"
            job.error = str(exc)
            log(f"ERROR: {exc}")
        if on_job:
            try:
                on_job(job)
            except Exception as cb_exc:
                log(f"on_job callback error: {cb_exc}")
        if cfg.batch.cooldown_s > 0 and i < total:
            log(f"Cooldown {cfg.batch.cooldown_s}s")
            time.sleep(float(cfg.batch.cooldown_s))
    return jobs
