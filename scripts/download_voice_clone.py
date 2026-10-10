"""Download the complete pinned SOAR package and verify Hugging Face hashes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

MODEL_ID = "dots-studio/dots.tts-soar"
MODEL_REVISION = "2f9b3e18d70d670d4c701da2dc55ded5755815ce"
ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models/tts/dots.tts-soar")
    args = parser.parse_args()
    from huggingface_hub import HfApi, snapshot_download

    model_dir = args.model_dir.resolve()
    info = HfApi().model_info(MODEL_ID, revision=MODEL_REVISION, files_metadata=True)
    expected_bytes = sum(entry.size or 0 for entry in info.siblings)
    print(f"Downloading {MODEL_ID} at {MODEL_REVISION}: {expected_bytes:,} bytes", flush=True)
    snapshot_download(
        repo_id=MODEL_ID, revision=MODEL_REVISION, local_dir=model_dir, max_workers=2,
    )
    records = []
    for entry in info.siblings:
        path = model_dir / entry.rfilename
        size = path.stat().st_size
        if entry.size is not None and size != entry.size:
            raise RuntimeError(f"Incomplete download: {path} ({size} != {entry.size})")
        digest = None
        if entry.lfs is not None:
            print(f"Verifying SHA-256: {entry.rfilename}", flush=True)
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != entry.lfs.sha256:
                raise RuntimeError(f"SHA-256 mismatch: {path}")
        records.append({"file": entry.rfilename, "bytes": size, "sha256": digest})
    manifest = {
        "model_id": MODEL_ID, "revision": MODEL_REVISION,
        "total_bytes": sum(record["bytes"] for record in records), "files": records,
    }
    (model_dir / "download-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8",
    )
    print(f"Verified {len(records)} files ({manifest['total_bytes']:,} bytes): {model_dir}", flush=True)


if __name__ == "__main__":
    main()
