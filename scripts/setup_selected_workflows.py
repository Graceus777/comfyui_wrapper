"""Install pinned, additive workflow dependencies; never upgrade ComfyUI/Torch.

Model downloads resume .part files, verify publisher SHA256, then rename atomically.
Existing complete models and node packs are left in place. No generation is queued.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import io
import json
import shutil
import struct
import subprocess
import tempfile
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def request(url, *, headers=None, method=None):
    return urllib.request.urlopen(urllib.request.Request(
        url, headers={"User-Agent": "ComfyUI-Wrapper-Setup", **(headers or {})}, method=method
    ), timeout=120)


def metadata(asset):
    repo = asset["repo"]
    with request(f"https://huggingface.co/api/models/{repo}") as response:
        revision = json.load(response)["sha"]
    folder, _, name = asset["file"].rpartition("/")
    url = f"https://huggingface.co/api/models/{repo}/tree/{revision}"
    if folder:
        url += "/" + urllib.parse.quote(folder, safe="/")
    with request(url) as response:
        rows = json.load(response)
    row = next((item for item in rows if item.get("path") == asset["file"]), None)
    if row is None:
        raise RuntimeError(f"Publisher file not found: {repo}/{asset['file']}")
    lfs = row.get("lfs") or {}
    return {**asset, "revision": revision, "size": row["size"], "sha256": lfs.get("oid"),
            "url": f"https://huggingface.co/{repo}/resolve/{revision}/{urllib.parse.quote(asset['file'], safe='/')}"}


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def validate_safetensors(path):
    with path.open("rb") as source:
        length = struct.unpack("<Q", source.read(8))[0]
        if length < 2 or length > min(path.stat().st_size - 8, 100 * 1024 * 1024):
            raise RuntimeError(f"Invalid safetensors header: {path}")
        header = json.loads(source.read(length))
        if not isinstance(header, dict) or not any(key != "__metadata__" for key in header):
            raise RuntimeError(f"Empty tensor file: {path}")


def download_model(asset, comfy):
    destination = comfy / "models" / asset["folder"] / Path(asset["file"]).name
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected_hash = asset.get("sha256")
    if destination.exists():
        if destination.stat().st_size != asset["size"]:
            raise RuntimeError(f"Existing model has a different size; refusing overwrite: {destination}")
        validate_safetensors(destination)
        if expected_hash and digest(destination) != expected_hash:
            raise RuntimeError(f"Existing model hash mismatch; refusing overwrite: {destination}")
        print(f"VERIFIED {destination.name}", flush=True)
        return {**asset, "path": str(destination), "status": "verified"}
    part = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(4):
        offset = part.stat().st_size if part.exists() else 0
        if offset == asset["size"]:
            break
        if offset > asset["size"]:
            raise RuntimeError(f"Oversized partial download: {part}")
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            with request(asset["url"], headers=headers) as response:
                append = offset > 0 and response.status == 206
                if append and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise RuntimeError("Server returned an unexpected byte range")
                total = offset if append else 0
                report_at = total
                with part.open("ab" if append else "wb") as output:
                    while block := response.read(8 * 1024 * 1024):
                        output.write(block)
                        total += len(block)
                        if total - report_at >= 512 * 1024 * 1024:
                            print(f"DOWNLOAD {destination.name}: {total / 1e9:.2f}/{asset['size'] / 1e9:.2f} GB", flush=True)
                            report_at = total
            if part.stat().st_size == asset["size"]:
                break
        except Exception as exc:
            print(f"RETRY {destination.name} ({attempt + 1}/4): {exc}", flush=True)
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))
    if not part.exists() or part.stat().st_size != asset["size"]:
        raise RuntimeError(f"Incomplete model: {destination}")
    validate_safetensors(part)
    actual_hash = digest(part)
    if expected_hash and actual_hash != expected_hash:
        raise RuntimeError(f"SHA256 mismatch: {part}; expected {expected_hash}, got {actual_hash}")
    part.rename(destination)
    print(f"COMPLETE {destination.name}: SHA256 {actual_hash}", flush=True)
    return {**asset, "path": str(destination), "sha256": actual_hash, "status": "downloaded"}


def install_node(asset, comfy):
    target = comfy / "custom_nodes" / asset["directory"]
    marker = target / ".wrapper-source.json"
    if target.exists():
        if marker.is_file() and json.loads(marker.read_text())["revision"] == asset["revision"]:
            print(f"PINNED {asset['directory']}", flush=True)
            return {**asset, "path": str(target), "status": "existing"}
        raise RuntimeError(f"Node pack already exists; refusing to replace it: {target}")
    with request(f"https://codeload.github.com/{asset['repo']}/zip/{asset['revision']}") as response:
        archive = response.read()
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="workflow_nodes_", dir=target.parent) as tmp:
        staging = Path(tmp)
        with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
            for entry in bundle.infolist():
                resolved = (staging / entry.filename).resolve()
                if not resolved.is_relative_to(staging.resolve()):
                    raise RuntimeError("Unsafe node archive path")
            bundle.extractall(staging)
        roots = list(staging.iterdir())
        if len(roots) != 1 or not (roots[0] / "__init__.py").is_file():
            raise RuntimeError(f"Unexpected node package layout: {asset['repo']}")
        (roots[0] / ".wrapper-source.json").write_text(json.dumps(asset, indent=2), encoding="utf-8")
        roots[0].rename(target)
    print(f"INSTALLED {asset['directory']} @ {asset['revision']}", flush=True)
    return {**asset, "path": str(target), "status": "installed"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy-root", required=True, type=Path)
    parser.add_argument("--nodes-only", action="store_true")
    parser.add_argument("--models-only", action="store_true")
    args = parser.parse_args()
    comfy = args.comfy_root.resolve()
    if not (comfy / "main.py").is_file():
        parser.error("--comfy-root must be the ComfyUI directory containing main.py")
    spec = json.loads((ROOT / "configs" / "selected_workflows.assets.json").read_text())
    report = ROOT / "outputs" / "selected-workflows-install.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    state = json.loads(report.read_text()) if report.exists() else {"nodes": [], "models": []}
    try:
        if not args.models_only:
            for node in spec["nodes"]:
                installed = install_node(node, comfy)
                state["nodes"] = [n for n in state["nodes"] if n["directory"] != node["directory"]] + [installed]
                report.write_text(json.dumps(state, indent=2), encoding="utf-8")
            embedded = comfy.parent / "python_embeded" / "python.exe"
            subprocess.run([str(embedded), "-s", "-m", "pip", "install", "--no-deps", *spec["python_packages"]], check=True)
        if not args.nodes_only:
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as workers:
                assets = list(workers.map(metadata, spec["models"]))
            needed = sum(a["size"] for a in assets if not (comfy / "models" / a["folder"] / Path(a["file"]).name).exists())
            free = shutil.disk_usage(comfy).free
            print(f"MODELS {needed / 1e9:.2f} GB needed; {free / 1e9:.2f} GB free", flush=True)
            if needed + 5 * 1024**3 > free:
                raise RuntimeError("Not enough free disk space for the selected models plus 5 GB reserve")
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as workers:
                futures = [workers.submit(download_model, asset, comfy) for asset in assets]
                for future in concurrent.futures.as_completed(futures):
                    installed = future.result()
                    state["models"] = [m for m in state["models"] if m["file"] != installed["file"]] + [installed]
                    report.write_text(json.dumps(state, indent=2), encoding="utf-8")
    finally:
        report.write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(f"Setup complete: {report}", flush=True)


if __name__ == "__main__":
    main()
