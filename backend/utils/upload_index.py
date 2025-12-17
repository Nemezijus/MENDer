from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional


INDEX_FILENAME = ".mender_upload_index.json"
INDEX_VERSION = 1


def _index_path(upload_dir: str) -> str:
    return os.path.join(upload_dir, INDEX_FILENAME)


def _now_ts() -> float:
    return time.time()


def load_index(upload_dir: str) -> Dict[str, Any]:
    """Load the upload index JSON (best-effort)."""
    path = _index_path(upload_dir)
    if not os.path.exists(path):
        return {"version": INDEX_VERSION, "files": {}}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {"version": INDEX_VERSION, "files": {}}
        if "files" not in data or not isinstance(data["files"], dict):
            data["files"] = {}
        if "version" not in data:
            data["version"] = INDEX_VERSION
        return data
    except (OSError, json.JSONDecodeError):
        # If the index is temporarily unreadable/corrupt, fall back safely.
        return {"version": INDEX_VERSION, "files": {}}


def save_index(upload_dir: str, data: Dict[str, Any]) -> None:
    """Atomically write the index to disk."""
    os.makedirs(upload_dir, exist_ok=True)

    path = _index_path(upload_dir)
    tmp_path = f"{path}.tmp"

    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)

    os.replace(tmp_path, path)


def _key(digest: str, ext: str) -> str:
    # Keep per-extension entries just in case same bytes appear under different file types.
    return f"{digest}:{ext.lower()}"


def update_index(
    upload_dir: str,
    *,
    digest: str,
    ext: str,
    saved_name: str,
    original_name: str,
    size_bytes: Optional[int] = None,
) -> None:
    """Upsert an index entry for a stored file.

    We store:
      - names: list of original filenames ever used for this content
      - last_seen_name: most recent original name
      - created_at / last_seen_at
    """
    data = load_index(upload_dir)
    files = data.setdefault("files", {})

    k = _key(digest, ext)
    now = _now_ts()

    entry = files.get(k)
    if not isinstance(entry, dict):
        entry = {
            "digest": digest,
            "ext": ext.lower(),
            "saved_name": saved_name,
            "created_at": now,
            "last_seen_at": now,
            "names": [],
            "last_seen_name": original_name,
        }

    # Update mutable fields
    entry["saved_name"] = saved_name
    entry["last_seen_at"] = now
    entry["last_seen_name"] = original_name

    if size_bytes is not None:
        entry["size_bytes"] = int(size_bytes)

    names = entry.get("names")
    if not isinstance(names, list):
        names = []
    if original_name and original_name not in names:
        names.append(original_name)
    entry["names"] = names

    files[k] = entry
    data["version"] = data.get("version", INDEX_VERSION)

    save_index(upload_dir, data)


def lookup_display_name(upload_dir: str, *, digest: str, ext: str) -> Optional[str]:
    """Return the best display/original name for a given digest/ext, if present."""
    data = load_index(upload_dir)
    entry = data.get("files", {}).get(_key(digest, ext))
    if isinstance(entry, dict):
        name = entry.get("last_seen_name")
        if isinstance(name, str) and name.strip():
            return name.strip()
        # fallback: first known name
        names = entry.get("names")
        if isinstance(names, list) and names:
            first = names[0]
            if isinstance(first, str) and first.strip():
                return first.strip()
    return None


def get_index_filename() -> str:
    return INDEX_FILENAME
