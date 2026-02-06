"""Model artifact serialization utilities.

We persist a dict package via joblib with the structure:

{
  "__mender_artifact__": true,
  "schema_version": "1",
  "meta": <dict compatible with backend ModelArtifactMeta>,
  "pipeline": <fitted sklearn Pipeline/BaseEstimator>,
}

This module is Business Layer safe: it has no backend dependencies and no global caches.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, Tuple, Union
import hashlib
from datetime import datetime

try:
    import joblib
except ImportError as e:
    raise RuntimeError("joblib is required for model persistence") from e

try:
    from sklearn.base import BaseEstimator  # type: ignore
except Exception:
    BaseEstimator = object  # fallback

SCHEMA_VERSION = "1"
MAGIC_KEY = "__mender_artifact__"

# --- Choose a compression backend safely ---
_HAS_LZ4 = False
try:
    import lz4.frame  # noqa: F401
    _HAS_LZ4 = True
except Exception:
    _HAS_LZ4 = False


def _joblib_compress_arg():
    """Choose a compression backend safely.

    If LZ4 is available, use it; otherwise fall back to zlib (built-in).
    """

    if _HAS_LZ4:
        return ("lz4", 3)
    return 3


@dataclass
class SaveResult:
    content_bytes: bytes
    size: int
    sha256: str


def _ensure_pipeline_is_serializable(pipeline: Any) -> None:
    # Supervised models typically expose predict(). Unsupervised estimators may
    # expose fit_predict() and/or transform() but not predict(). We still want
    # to persist such pipelines.
    if not hasattr(pipeline, "fit"):
        raise ValueError("Pipeline must expose a fit() method.")

    has_infer = any(
        hasattr(pipeline, name)
        for name in (
            "predict",
            "fit_predict",
            "transform",
        )
    )
    if not has_infer:
        raise ValueError(
            "Pipeline must expose at least one inference method: predict(), fit_predict(), or transform()."
        )

    if BaseEstimator is not object and not isinstance(pipeline, BaseEstimator):
        # non-fatal; we prefer sklearn-compatible estimators but don't require it
        pass


def _validate_meta(meta: Dict[str, Any]) -> None:
    required = ["uid", "created_at", "kind", "model", "split", "eval", "pipeline"]
    missing = [k for k in required if k not in meta]
    if missing:
        raise ValueError(f"Artifact meta missing required keys: {missing}")

    if not isinstance(meta["created_at"], datetime):
        raise ValueError("Artifact meta 'created_at' must be a datetime object")

    pl = meta.get("pipeline")
    if not isinstance(pl, list):
        raise ValueError("Artifact meta 'pipeline' must be a list")
    for i, step in enumerate(pl):
        if not isinstance(step, dict) or "name" not in step:
            raise ValueError(f"Artifact meta 'pipeline[{i}]' must include 'name'")

    for k in ("model", "split", "eval"):
        if not isinstance(meta.get(k), dict):
            raise ValueError(f"Artifact meta '{k}' must be a dict")


def _hash_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def save_model_artifact(pipeline: Any, meta: Dict[str, Any]) -> SaveResult:
    """Serialize a fitted pipeline + meta to joblib bytes."""

    _ensure_pipeline_is_serializable(pipeline)
    _validate_meta(meta)

    package = {
        MAGIC_KEY: True,
        "schema_version": SCHEMA_VERSION,
        "meta": meta,
        "pipeline": pipeline,
    }

    buf = BytesIO()
    joblib.dump(package, buf, compress=_joblib_compress_arg())
    data = buf.getvalue()
    digest = _hash_bytes(data)

    meta.setdefault("payload_hash", digest)

    return SaveResult(content_bytes=data, size=len(data), sha256=digest)


def load_model_artifact(payload: Union[bytes, BytesIO]) -> Tuple[Any, Dict[str, Any]]:
    """Deserialize an artifact payload and validate."""

    buf = BytesIO(payload) if isinstance(payload, bytes) else payload
    package = joblib.load(buf)

    if not isinstance(package, dict) or not package.get(MAGIC_KEY):
        raise ValueError("Not a valid MENDer artifact package")

    if str(package.get("schema_version")) != SCHEMA_VERSION:
        raise ValueError(
            f"Incompatible schema_version: {package.get('schema_version')}, expected {SCHEMA_VERSION}"
        )

    meta = package.get("meta")
    pipeline = package.get("pipeline")
    if meta is None or pipeline is None:
        raise ValueError("Corrupt artifact: missing 'meta' or 'pipeline'")

    _validate_meta(meta)
    _ensure_pipeline_is_serializable(pipeline)

    return pipeline, meta
