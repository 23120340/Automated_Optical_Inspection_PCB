"""Fetch a small, deterministic VisA PCB reference set.

The authoritative VisA dataset is published by Amazon as one monolithic tar
archive.  For the small manual audit planned for AOI step 5.5/6.2, this command
uses the ``BrachioLab/visa`` Hugging Face Parquet mirror to fetch exactly 30
normal images without downloading that whole archive::

    python scripts/fetch_visa_reference_set.py \
        --output datasets/visa_pcb2_reference

The mirror is a preprocessed distribution channel, not the authority for the
dataset's identity or licence.  ``manifest.json`` therefore records both the
mirror coordinates and Amazon's official VisA provenance.  Expiring image URLs
returned by the Dataset Viewer are deliberately never written to disk.

Images are validated and staged before the output directory is published.  An
existing valid output is an idempotent no-op; an incomplete, changed or
otherwise incompatible output is refused rather than overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import shutil
import sys
import tempfile
import urllib.parse
import urllib.request
import warnings
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping, Protocol

from PIL import Image, UnidentifiedImageError

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


DATASET_ID = "BrachioLab/visa"
CONFIG = "default"
DEFAULT_SPLIT = "pcb2.train"
REFERENCE_COUNT = 30
SOURCE_KIND = "dataset_preprocessed"
SCHEMA_VERSION = "aoi-visa-reference-set/1.0"
SELECTION_ALGORITHM = "inclusive_integer_linspace_floor_v1"

DATASET_VIEWER_BASE = "https://datasets-server.huggingface.co"
MIRROR_URL = "https://huggingface.co/datasets/BrachioLab/visa"
OFFICIAL_REGISTRY_URL = "https://registry.opendata.aws/visa/"
OFFICIAL_REPOSITORY_URL = "https://github.com/amazon-science/spot-diff"
OFFICIAL_ARCHIVE_URL = (
    "https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar"
)
OFFICIAL_PAPER_URL = (
    "https://assets.amazon.science/e6/b4/e3510d084be1bc785515fe05b2d2/"
    "spot-the-difference-self-supervised-pre-training-for-anomaly-detection-"
    "and-segmentation.pdf"
)
LICENSE_NAME = "CC BY 4.0"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"

_MAX_RESPONSE_BYTES = 64 * 1024 * 1024
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_SPLIT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SIGNED_URL_MARKERS = ("expires=", "signature=", "key-pair-id=", "x-amz-")
_ALLOWED_IMAGE_HOSTS = frozenset({"datasets-server.huggingface.co"})
_IMAGE_EXTENSIONS = {
    "BMP": ".bmp",
    "JPEG": ".jpg",
    "PNG": ".png",
    "TIFF": ".tiff",
    "WEBP": ".webp",
}
_PROCESSING_NOTE = (
    "The Hugging Face mirror exposes Parquet/Dataset Viewer image assets. "
    "Files below are the exact downloaded mirror bytes and are not asserted "
    "to be byte-identical to Amazon's official tar archive."
)


class ReferenceSetError(RuntimeError):
    """The requested reference set could not be verified safely."""


class Fetcher(Protocol):
    """Minimal injectable transport used by the command and offline tests."""

    def fetch(self, url: str) -> bytes:
        """Return the response body for ``url`` without transforming it."""


@dataclass(frozen=True)
class URLFetcher:
    """Small standard-library HTTPS fetcher with bounded responses."""

    timeout_seconds: float = 30.0

    def fetch(self, url: str) -> bytes:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "AOI-PCB-VisA-reference-fetcher/1.0"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            status = getattr(response, "status", 200)
            if status != 200:
                raise ReferenceSetError(f"HTTP response was not successful (status {status})")
            payload = response.read(_MAX_RESPONSE_BYTES + 1)
        if len(payload) > _MAX_RESPONSE_BYTES:
            raise ReferenceSetError(
                f"Response exceeds the {_MAX_RESPONSE_BYTES}-byte safety limit"
            )
        return payload


@dataclass(frozen=True)
class FetchResult:
    """Outcome returned by :func:`fetch_reference_set`."""

    output: Path
    created: bool
    manifest: Mapping[str, Any]


@dataclass(frozen=True)
class _Row:
    index: int
    label: int
    width: int
    height: int
    image_url: str
    total_rows: int


def evenly_spaced_indices(total_rows: int, count: int = REFERENCE_COUNT) -> list[int]:
    """Return ``count`` deterministic indices spanning the complete split.

    Integer arithmetic makes the selection stable across Python/NumPy versions.
    Both endpoints are included.  Adjacent gaps can differ by at most one row.
    """

    if isinstance(total_rows, bool) or not isinstance(total_rows, int) or total_rows <= 0:
        raise ReferenceSetError("Split row count must be a positive integer")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ReferenceSetError("Reference image count must be a positive integer")
    if count > total_rows:
        raise ReferenceSetError(
            f"Cannot choose {count} unique rows from a split containing {total_rows} rows"
        )
    if count == 1:
        return [0]
    return [index * (total_rows - 1) // (count - 1) for index in range(count)]


def fetch_reference_set(
    output: str | Path,
    *,
    split: str = DEFAULT_SPLIT,
    fetcher: Fetcher | None = None,
) -> FetchResult:
    """Fetch and atomically publish the 30-image VisA reference set.

    ``fetcher`` is intentionally injectable.  Tests and callers with their own
    transport can provide an object implementing :class:`Fetcher`; no network
    package or global monkeypatch is required.
    """

    split = str(split).strip()
    if not _SPLIT_NAME.fullmatch(split):
        raise ReferenceSetError("Split must be a non-empty portable Dataset Viewer name")

    requested_destination = Path(output).expanduser()
    if requested_destination.is_symlink():
        raise ReferenceSetError("Output must not be a symbolic link")
    destination = requested_destination.resolve()
    if destination.exists():
        if not destination.is_dir():
            raise ReferenceSetError("Output exists and is not a directory")
        manifest = _validate_existing_output(destination, split=split)
        return FetchResult(output=destination, created=False, manifest=manifest)

    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=str(parent)))
    transport = fetcher or URLFetcher()
    published = False
    try:
        images_dir = stage / "images"
        images_dir.mkdir()

        first_payload = _fetch_json(transport, _row_url(split, 0), context="row 0")
        first_row = _parse_row(first_payload, expected_index=0)
        indices = evenly_spaced_indices(first_row.total_rows)

        file_entries: list[dict[str, Any]] = []
        digests: set[str] = set()
        for row_index in indices:
            if row_index == 0:
                row = first_row
            else:
                payload = _fetch_json(
                    transport,
                    _row_url(split, row_index),
                    context=f"row {row_index}",
                )
                row = _parse_row(
                    payload,
                    expected_index=row_index,
                    expected_total=first_row.total_rows,
                )
            image_bytes = _fetch_image(transport, row)
            image_format = _inspect_image(
                image_bytes,
                expected_width=row.width,
                expected_height=row.height,
                context=f"row {row.index}",
            )
            digest = hashlib.sha256(image_bytes).hexdigest()
            if digest in digests:
                raise ReferenceSetError(
                    f"Selected row {row.index} duplicates another image by SHA-256"
                )
            digests.add(digest)

            extension = _IMAGE_EXTENSIONS.get(image_format)
            if extension is None:
                raise ReferenceSetError(
                    f"Row {row.index} uses unsupported image format {image_format!r}"
                )
            relative_path = f"images/row_{row.index:06d}{extension}"
            (stage / PurePosixPath(relative_path)).write_bytes(image_bytes)
            file_entries.append(
                {
                    "row_index": row.index,
                    "path": relative_path,
                    "label": row.label,
                    "width": row.width,
                    "height": row.height,
                    "byte_size": len(image_bytes),
                    "sha256": digest,
                }
            )

        manifest = _build_manifest(
            split=split,
            total_rows=first_row.total_rows,
            indices=indices,
            files=file_entries,
        )
        _assert_manifest_portable(manifest)
        (stage / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        # Re-read every staged byte through the same verifier used on a later
        # idempotent invocation.  Nothing is published if this final gate fails.
        verified = _validate_existing_output(stage, split=split)
        if destination.exists() or destination.is_symlink():
            raise ReferenceSetError("Output appeared while the reference set was downloading")
        stage.rename(destination)
        published = True
        return FetchResult(output=destination, created=True, manifest=verified)
    finally:
        if not published:
            shutil.rmtree(stage, ignore_errors=True)


def _row_url(split: str, index: int) -> str:
    query = urllib.parse.urlencode(
        {
            "dataset": DATASET_ID,
            "config": CONFIG,
            "split": split,
            "offset": index,
            "length": 1,
        }
    )
    return f"{DATASET_VIEWER_BASE}/rows?{query}"


def _fetch_json(fetcher: Fetcher, url: str, *, context: str) -> Mapping[str, Any]:
    try:
        payload = fetcher.fetch(url)
    except ReferenceSetError:
        raise
    except Exception as exc:
        # The exception text from a HTTP client can contain the signed URL.  Do
        # not echo it into logs or CI artifacts.
        raise ReferenceSetError(
            f"Could not fetch Dataset Viewer metadata for {context} ({type(exc).__name__})"
        ) from exc
    if not isinstance(payload, bytes):
        raise ReferenceSetError(f"Fetcher returned non-byte metadata for {context}")
    if len(payload) > _MAX_RESPONSE_BYTES:
        raise ReferenceSetError(f"Dataset Viewer metadata is too large for {context}")
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReferenceSetError(f"Dataset Viewer returned invalid JSON for {context}") from exc
    if not isinstance(decoded, Mapping):
        raise ReferenceSetError(f"Dataset Viewer returned a non-object for {context}")
    return decoded


def _parse_row(
    payload: Mapping[str, Any],
    *,
    expected_index: int,
    expected_total: int | None = None,
) -> _Row:
    if payload.get("partial") is not False:
        raise ReferenceSetError(f"Dataset Viewer response for row {expected_index} is partial")
    total_rows = _positive_int(payload.get("num_rows_total"), "num_rows_total")
    if expected_total is not None and total_rows != expected_total:
        raise ReferenceSetError(
            f"Split row count changed from {expected_total} to {total_rows} during download"
        )

    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != 1:
        raise ReferenceSetError(
            f"Dataset Viewer must return exactly one record for row {expected_index}"
        )
    record = rows[0]
    row_index = record.get("row_idx") if isinstance(record, Mapping) else None
    if (
        not isinstance(record, Mapping)
        or isinstance(row_index, bool)
        or not isinstance(row_index, int)
        or row_index != expected_index
    ):
        raise ReferenceSetError(f"Dataset Viewer returned the wrong row for {expected_index}")
    if record.get("truncated_cells") not in (None, []):
        raise ReferenceSetError(f"Dataset Viewer truncated row {expected_index}")
    values = record.get("row")
    if not isinstance(values, Mapping):
        raise ReferenceSetError(f"Dataset Viewer row {expected_index} has no row object")

    label = values.get("label")
    if isinstance(label, bool) or not isinstance(label, int) or label != 0:
        raise ReferenceSetError(
            f"Row {expected_index} is not a normal VisA sample (expected label=0)"
        )
    image = values.get("image")
    if not isinstance(image, Mapping):
        raise ReferenceSetError(f"Row {expected_index} has no image object")
    image_url = image.get("src")
    if not isinstance(image_url, str) or not image_url:
        raise ReferenceSetError(f"Row {expected_index} has no downloadable image URL")
    _validate_image_url(image_url, row_index=expected_index)

    return _Row(
        index=expected_index,
        label=label,
        width=_positive_int(image.get("width"), f"row {expected_index} width"),
        height=_positive_int(image.get("height"), f"row {expected_index} height"),
        image_url=image_url,
        total_rows=total_rows,
    )


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ReferenceSetError(f"{name} must be a positive integer")
    return value


def _validate_image_url(url: str, *, row_index: int) -> None:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in _ALLOWED_IMAGE_HOSTS:
        raise ReferenceSetError(
            f"Row {row_index} image URL is outside the trusted Hugging Face asset host"
        )


def _fetch_image(fetcher: Fetcher, row: _Row) -> bytes:
    try:
        payload = fetcher.fetch(row.image_url)
    except ReferenceSetError:
        raise
    except Exception as exc:
        raise ReferenceSetError(
            f"Could not download image bytes for row {row.index} ({type(exc).__name__})"
        ) from exc
    if not isinstance(payload, bytes):
        raise ReferenceSetError(f"Fetcher returned non-byte image data for row {row.index}")
    if not payload:
        raise ReferenceSetError(f"Downloaded image for row {row.index} is empty")
    if len(payload) > _MAX_RESPONSE_BYTES:
        raise ReferenceSetError(f"Downloaded image for row {row.index} is too large")
    return payload


def _inspect_image(
    payload: bytes,
    *,
    expected_width: int,
    expected_height: int,
    context: str,
) -> str:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(payload)) as image:
                image_format = str(image.format or "").upper()
                actual_width, actual_height = image.size
                image.verify()
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombWarning) as exc:
        raise ReferenceSetError(f"Downloaded bytes are not a valid image for {context}") from exc
    if (actual_width, actual_height) != (expected_width, expected_height):
        raise ReferenceSetError(
            f"Image dimensions for {context} are {actual_width}x{actual_height}, "
            f"not declared {expected_width}x{expected_height}"
        )
    if image_format not in _IMAGE_EXTENSIONS:
        raise ReferenceSetError(f"Unsupported image format {image_format!r} for {context}")
    return image_format


def _build_manifest(
    *,
    split: str,
    total_rows: int,
    indices: list[int],
    files: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "source_kind": SOURCE_KIND,
        "dataset_id": DATASET_ID,
        "config": CONFIG,
        "split": split,
        "mirror_url": MIRROR_URL,
        "license": LICENSE_NAME,
        "license_url": LICENSE_URL,
        "official_provenance": {
            "dataset_name": "VisA",
            "publisher": "Amazon",
            "registry_url": OFFICIAL_REGISTRY_URL,
            "repository_url": OFFICIAL_REPOSITORY_URL,
            "archive_url": OFFICIAL_ARCHIVE_URL,
            "paper_url": OFFICIAL_PAPER_URL,
        },
        "processing_note": _PROCESSING_NOTE,
        "selection": {
            "algorithm": SELECTION_ALGORITHM,
            "requested_count": REFERENCE_COUNT,
            "num_rows_total": total_rows,
            "expected_label": 0,
            "row_indices": indices,
        },
        "files": files,
    }


def _validate_existing_output(destination: Path, *, split: str) -> Mapping[str, Any]:
    manifest_path = destination / "manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ReferenceSetError("Existing output has no regular manifest.json")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReferenceSetError("Existing manifest.json is unreadable or invalid") from exc
    if not isinstance(manifest, Mapping):
        raise ReferenceSetError("Existing manifest.json is not a JSON object")
    _assert_manifest_portable(manifest)

    expected_root_keys = {
        "schema_version",
        "source_kind",
        "dataset_id",
        "config",
        "split",
        "mirror_url",
        "license",
        "license_url",
        "official_provenance",
        "processing_note",
        "selection",
        "files",
    }
    if set(manifest) != expected_root_keys:
        raise ReferenceSetError("Existing manifest has missing or unexpected fields")

    expected_scalars = {
        "schema_version": SCHEMA_VERSION,
        "source_kind": SOURCE_KIND,
        "dataset_id": DATASET_ID,
        "config": CONFIG,
        "split": split,
        "mirror_url": MIRROR_URL,
        "license": LICENSE_NAME,
        "license_url": LICENSE_URL,
    }
    for key, expected in expected_scalars.items():
        if manifest.get(key) != expected:
            raise ReferenceSetError(f"Existing manifest has incompatible {key!r}")
    if manifest.get("processing_note") != _PROCESSING_NOTE:
        raise ReferenceSetError("Existing manifest has an incompatible processing note")

    provenance = manifest.get("official_provenance")
    expected_provenance = {
        "dataset_name": "VisA",
        "publisher": "Amazon",
        "registry_url": OFFICIAL_REGISTRY_URL,
        "repository_url": OFFICIAL_REPOSITORY_URL,
        "archive_url": OFFICIAL_ARCHIVE_URL,
        "paper_url": OFFICIAL_PAPER_URL,
    }
    if provenance != expected_provenance:
        raise ReferenceSetError("Existing manifest has incompatible official provenance")

    selection = manifest.get("selection")
    if not isinstance(selection, Mapping):
        raise ReferenceSetError("Existing manifest has no valid selection object")
    if set(selection) != {
        "algorithm",
        "requested_count",
        "num_rows_total",
        "expected_label",
        "row_indices",
    }:
        raise ReferenceSetError("Existing manifest selection has incompatible fields")
    total_rows = _positive_int(selection.get("num_rows_total"), "selection num_rows_total")
    expected_indices = evenly_spaced_indices(total_rows)
    if selection.get("algorithm") != SELECTION_ALGORITHM:
        raise ReferenceSetError("Existing manifest uses a different selection algorithm")
    if selection.get("requested_count") != REFERENCE_COUNT:
        raise ReferenceSetError("Existing manifest has a different requested count")
    if selection.get("expected_label") != 0 or isinstance(
        selection.get("expected_label"), bool
    ):
        raise ReferenceSetError("Existing manifest does not require label=0")
    row_indices = selection.get("row_indices")
    if (
        not isinstance(row_indices, list)
        or any(isinstance(index, bool) or not isinstance(index, int) for index in row_indices)
        or row_indices != expected_indices
    ):
        raise ReferenceSetError("Existing manifest row indices are not deterministic")

    files = manifest.get("files")
    if not isinstance(files, list) or len(files) != REFERENCE_COUNT:
        raise ReferenceSetError(f"Existing manifest must contain {REFERENCE_COUNT} files")
    seen_paths: set[str] = set()
    seen_digests: set[str] = set()
    for expected_index, entry in zip(expected_indices, files, strict=True):
        if not isinstance(entry, Mapping) or set(entry) != {
            "row_index",
            "path",
            "label",
            "width",
            "height",
            "byte_size",
            "sha256",
        }:
            raise ReferenceSetError("Existing manifest file entry has incompatible fields")
        entry_index = entry.get("row_index")
        if (
            isinstance(entry_index, bool)
            or not isinstance(entry_index, int)
            or entry_index != expected_index
        ):
            raise ReferenceSetError("Existing manifest file rows are missing or out of order")
        if entry.get("label") != 0 or isinstance(entry.get("label"), bool):
            raise ReferenceSetError(f"Existing row {expected_index} does not have label=0")
        relative_path = _safe_relative_path(entry.get("path"), row_index=expected_index)
        relative_text = relative_path.as_posix()
        if relative_text in seen_paths:
            raise ReferenceSetError("Existing manifest contains duplicate file paths")
        seen_paths.add(relative_text)
        file_path = destination.joinpath(*relative_path.parts)
        if not file_path.is_file() or file_path.is_symlink():
            raise ReferenceSetError(f"Existing image is missing for row {expected_index}")
        try:
            payload = file_path.read_bytes()
        except OSError as exc:
            raise ReferenceSetError(f"Existing image is unreadable for row {expected_index}") from exc
        if entry.get("byte_size") != len(payload):
            raise ReferenceSetError(f"Existing image byte size changed for row {expected_index}")
        expected_digest = entry.get("sha256")
        if not isinstance(expected_digest, str) or not _HEX64.fullmatch(expected_digest):
            raise ReferenceSetError(f"Existing SHA-256 is invalid for row {expected_index}")
        if hashlib.sha256(payload).hexdigest() != expected_digest:
            raise ReferenceSetError(f"Existing image SHA-256 changed for row {expected_index}")
        if expected_digest in seen_digests:
            raise ReferenceSetError("Existing reference images are not unique by SHA-256")
        seen_digests.add(expected_digest)
        width = _positive_int(entry.get("width"), f"row {expected_index} width")
        height = _positive_int(entry.get("height"), f"row {expected_index} height")
        image_format = _inspect_image(
            payload,
            expected_width=width,
            expected_height=height,
            context=f"existing row {expected_index}",
        )
        if relative_path.suffix.lower() != _IMAGE_EXTENSIONS[image_format]:
            raise ReferenceSetError(f"Existing image extension changed for row {expected_index}")
    return manifest


def _safe_relative_path(value: Any, *, row_index: int) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ReferenceSetError(f"Manifest path is not portable for row {row_index}")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ReferenceSetError(f"Manifest path is not relative for row {row_index}")
    if path.as_posix() != value or len(path.parts) != 2 or path.parts[0] != "images":
        raise ReferenceSetError(f"Manifest path is not canonical for row {row_index}")
    return path


def _assert_manifest_portable(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ReferenceSetError("Manifest object keys must be strings")
            _assert_manifest_portable(item)
        return
    if isinstance(value, list):
        for item in value:
            _assert_manifest_portable(item)
        return
    if not isinstance(value, str):
        return
    lowered = value.lower()
    if any(marker in lowered for marker in _SIGNED_URL_MARKERS):
        raise ReferenceSetError("Manifest must not contain an expiring signed URL")
    if "://" not in value and (
        PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute()
    ):
        raise ReferenceSetError("Manifest must not contain absolute workstation paths")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch 30 deterministic normal images from the BrachioLab/visa "
            "mirror and record official Amazon VisA provenance."
        )
    )
    parser.add_argument("--output", required=True, help="Reference-set directory to create.")
    parser.add_argument(
        "--split",
        default=DEFAULT_SPLIT,
        help=f"Hugging Face split (default: {DEFAULT_SPLIT}).",
    )
    return parser


def main(argv: list[str] | None = None, *, fetcher: Fetcher | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = fetch_reference_set(args.output, split=args.split, fetcher=fetcher)
    except (ReferenceSetError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    action = "Downloaded and verified" if result.created else "Already verified"
    print(
        f"{action}: {REFERENCE_COUNT} normal VisA rows from {args.split} -> {result.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
