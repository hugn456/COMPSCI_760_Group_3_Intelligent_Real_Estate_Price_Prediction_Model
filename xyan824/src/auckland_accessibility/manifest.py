from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .paths import METADATA_DIR, ensure_directories, relative_to_root


MANIFEST_PATH = METADATA_DIR / "manifest.json"


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        return {"generated_at": None, "artifacts": {}}
    with MANIFEST_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def record_artifact(key: str, path: Path, **metadata: Any) -> None:
    ensure_directories()
    manifest = load_manifest()
    artifacts = manifest.setdefault("artifacts", {})
    item: dict[str, Any] = {
        "path": relative_to_root(path),
        "recorded_at": utc_now(),
        **metadata,
    }
    if path.exists() and path.is_file():
        item["bytes"] = path.stat().st_size
        item["sha256"] = sha256_file(path)
    artifacts[key] = item
    manifest["generated_at"] = utc_now()
    tmp = MANIFEST_PATH.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    tmp.replace(MANIFEST_PATH)

