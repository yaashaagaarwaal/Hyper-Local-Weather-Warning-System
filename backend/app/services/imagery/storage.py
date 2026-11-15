"""Safe, id-based file storage for the imagery/video processing module.

Two distinct roots:
- RUNTIME_DIR (`backend/app/data/imagery_runtime/`): user uploads and
  processing outputs generated while the server runs. Not checked in as
  curated sample data -- purely a scratch/working area, gitignored.
- DATASET_DIR (`dataset/imagery/`): the small, real, pre-acquired NASA GIBS
  sample set (see dataset/scripts/fetch_gibs_imagery.py and
  dataset/imagery/metadata/SOURCES.md). Read-only from this module's point
  of view -- the API can list/serve it, never write into it.

Safety rules actually enforced here (not just documented):
- Every stored file gets a server-generated UUID name -- a client-supplied
  filename is never used as a path component, so there is no path-traversal
  surface.
- Uploaded bytes are size-capped *before* being decoded.
- "Is this really an image/video" is decided by actually decoding the bytes
  with OpenCV, not by trusting the extension or the client's Content-Type.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from app.core.exceptions import AppError, NotFoundError

BACKEND_ROOT = Path(__file__).resolve().parents[3]
REPO_ROOT = BACKEND_ROOT.parent

RUNTIME_DIR = BACKEND_ROOT / "app" / "data" / "imagery_runtime"
UPLOADS_DIR = RUNTIME_DIR / "uploads"
PROCESSED_DIR = RUNTIME_DIR / "processed"
DATASET_DIR = REPO_ROOT / "dataset" / "imagery"
DATASET_SATELLITE_DIR = DATASET_DIR / "satellite"
DATASET_METADATA_DIR = DATASET_DIR / "metadata"
DATASET_SAMPLES_DIR = DATASET_DIR / "samples"

for _d in (UPLOADS_DIR, PROCESSED_DIR):
    _d.mkdir(parents=True, exist_ok=True)

MAX_IMAGE_BYTES = 15 * 1024 * 1024  # 15 MB
MAX_VIDEO_BYTES = 60 * 1024 * 1024  # 60 MB
ALLOWED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
ALLOWED_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}

SAMPLE_PREFIX = "sample:"


class UploadTooLargeError(AppError):
    def __init__(self, limit_bytes: int):
        super().__init__(f"File exceeds the {limit_bytes // (1024 * 1024)} MB limit for this upload type.", status_code=413)


class UnsupportedFileTypeError(AppError):
    def __init__(self, detail: str):
        super().__init__(detail, status_code=415)


def _safe_ext(filename: str, allowed: set[str]) -> str:
    ext = Path(filename or "").suffix.lower()
    if ext not in allowed:
        raise UnsupportedFileTypeError(
            f"Unsupported file extension {ext!r}. Allowed: {sorted(allowed)}"
        )
    return ext


@dataclass
class StoredAsset:
    asset_id: str
    kind: Literal["image", "video"]
    path: Path
    original_filename: str
    size_bytes: int
    metadata: dict


def _write_metadata(meta_path: Path, metadata: dict) -> None:
    meta_path.write_text(json.dumps(metadata, indent=2))


def save_upload(data: bytes, original_filename: str, kind: Literal["image", "video"]) -> StoredAsset:
    """Persist raw upload bytes under a fresh UUID name. Caller is
    responsible for having already validated `data` actually decodes as the
    claimed kind (see processing.decode_image / processing.probe_video) --
    this function only handles safe naming, size limits and bookkeeping.
    """
    allowed = ALLOWED_IMAGE_EXTS if kind == "image" else ALLOWED_VIDEO_EXTS
    limit = MAX_IMAGE_BYTES if kind == "image" else MAX_VIDEO_BYTES
    ext = _safe_ext(original_filename, allowed)

    if len(data) > limit:
        raise UploadTooLargeError(limit)
    if len(data) == 0:
        raise UnsupportedFileTypeError("Uploaded file is empty.")

    asset_id = uuid.uuid4().hex
    safe_name = Path(original_filename or "upload").name  # strip any path components, display-only
    stored_path = UPLOADS_DIR / f"{asset_id}{ext}"
    stored_path.write_bytes(data)

    metadata = {
        "assetId": asset_id,
        "kind": kind,
        "originalFilename": safe_name,
        "storedFilename": stored_path.name,
        "sizeBytes": len(data),
    }
    _write_metadata(UPLOADS_DIR / f"{asset_id}.json", metadata)
    return StoredAsset(asset_id, kind, stored_path, safe_name, len(data), metadata)


def get_upload_path(asset_id: str) -> tuple[Path, dict]:
    """Like load_upload_bytes but returns the on-disk Path instead of the
    bytes -- needed for cv2.VideoCapture, which has no in-memory API.
    """
    meta_path = UPLOADS_DIR / f"{asset_id}.json"
    if not meta_path.exists():
        raise NotFoundError(f"Unknown uploaded asset id: {asset_id!r}")
    metadata = json.loads(meta_path.read_text())
    stored_path = UPLOADS_DIR / metadata["storedFilename"]
    if not stored_path.exists():
        raise NotFoundError(f"Unknown uploaded asset id: {asset_id!r}")
    return stored_path, metadata


def register_derived_image(image_bytes: bytes, original_filename: str, extra_metadata: dict) -> StoredAsset:
    """Registers a server-generated image (e.g. a frame extracted from an
    uploaded video) as a first-class, reusable image asset -- so it can be
    passed back into /process, /compare, /motion, /video/generate by id,
    the same as a directly-uploaded image.
    """
    asset_id = uuid.uuid4().hex
    stored_path = UPLOADS_DIR / f"{asset_id}.png"
    stored_path.write_bytes(image_bytes)
    metadata = {
        "assetId": asset_id,
        "kind": "image",
        "originalFilename": original_filename,
        "storedFilename": stored_path.name,
        "sizeBytes": len(image_bytes),
        "derived": True,
        **extra_metadata,
    }
    _write_metadata(UPLOADS_DIR / f"{asset_id}.json", metadata)
    return StoredAsset(asset_id, "image", stored_path, original_filename, len(image_bytes), metadata)


def save_processed_image(image_bytes: bytes, out_dir: Path, name: str) -> Path:
    path = out_dir / f"{name}.png"
    path.write_bytes(image_bytes)
    return path


def load_upload_bytes(asset_id: str) -> tuple[bytes, dict]:
    for meta_path in UPLOADS_DIR.glob(f"{asset_id}.json"):
        metadata = json.loads(meta_path.read_text())
        stored_path = UPLOADS_DIR / metadata["storedFilename"]
        if not stored_path.exists():
            break
        return stored_path.read_bytes(), metadata
    raise NotFoundError(f"Unknown uploaded asset id: {asset_id!r}")


def update_upload_metadata(asset_id: str, patch: dict) -> dict:
    meta_path = UPLOADS_DIR / f"{asset_id}.json"
    if not meta_path.exists():
        raise NotFoundError(f"Unknown uploaded asset id: {asset_id!r}")
    metadata = json.loads(meta_path.read_text())
    metadata.update(patch)
    _write_metadata(meta_path, metadata)
    return metadata


def list_dataset_samples() -> list[dict]:
    """Every real, pre-acquired GIBS asset in dataset/imagery/satellite/,
    joined with its metadata sidecar from dataset/imagery/metadata/. Used by
    the frontend's satellite viewer so a demo works with zero uploads.
    """
    out = []
    if not DATASET_SATELLITE_DIR.exists():
        return out
    for img_path in sorted(DATASET_SATELLITE_DIR.glob("*.png")):
        meta_path = DATASET_METADATA_DIR / (img_path.stem + ".json")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        out.append({
            "id": f"{SAMPLE_PREFIX}{img_path.stem}",
            "filename": img_path.name,
            "sizeBytes": img_path.stat().st_size,
            "url": dataset_media_url(img_path),
            **meta,
        })
    return out


def load_sample_bytes(sample_id: str) -> tuple[bytes, dict]:
    if not sample_id.startswith(SAMPLE_PREFIX):
        raise NotFoundError(f"Not a sample id: {sample_id!r}")
    stem = sample_id[len(SAMPLE_PREFIX):]
    img_path = DATASET_SATELLITE_DIR / f"{stem}.png"
    if not img_path.exists():
        raise NotFoundError(f"Unknown sample id: {sample_id!r}")
    meta_path = DATASET_METADATA_DIR / f"{stem}.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return img_path.read_bytes(), meta


def load_image_bytes_by_id(image_id: str) -> tuple[bytes, dict]:
    """Resolve either an uploaded image id or a `sample:...` curated-dataset
    id through one call, so processing endpoints don't need to care which
    kind of id they were given.
    """
    if image_id.startswith(SAMPLE_PREFIX):
        return load_sample_bytes(image_id)
    return load_upload_bytes(image_id)


def new_processed_dir(label: str) -> tuple[str, Path]:
    result_id = uuid.uuid4().hex
    out_dir = PROCESSED_DIR / f"{label}_{result_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return result_id, out_dir


def save_result(out_dir: Path, result: dict) -> None:
    _write_metadata(out_dir / "result.json", result)


def read_result(result_id: str) -> dict:
    matches = list(PROCESSED_DIR.glob(f"*_{result_id}"))
    if not matches:
        raise NotFoundError(f"Unknown result id: {result_id!r}")
    result_path = matches[0] / "result.json"
    if not result_path.exists():
        raise NotFoundError(f"Unknown result id: {result_id!r}")
    return json.loads(result_path.read_text())


def runtime_media_url(path: Path) -> str:
    return f"/media/imagery/{path.relative_to(RUNTIME_DIR).as_posix()}"


def dataset_media_url(path: Path) -> str:
    return f"/media/imagery-dataset/{path.relative_to(DATASET_DIR).as_posix()}"
