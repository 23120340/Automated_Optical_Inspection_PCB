"""Fetch 30 aligned, unmodified images of one gas-pump PCB from MPI-PCB.

The upstream ``original_aligned.zip`` is larger than 5 GiB.  This script never
downloads that archive as a whole: :class:`RemoteRangeReader` exposes exact,
validated HTTP byte ranges to :class:`zipfile.ZipFile`, which reads the ZIP64
central directory and only decompresses the 30 selected members.

The selected JPEG bytes are preserved exactly as published.  They are input
records, not a production Golden recipe: Golden enrollment must still choose a
frame, save the canonical Golden losslessly, and pass human/alignment review.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from http.client import IncompleteRead, RemoteDisconnected
import io
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import shutil
import statistics
import sys
import tempfile
import time
from typing import Any, BinaryIO, Callable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4
from zipfile import BadZipFile, ZIP_DEFLATED, ZIP_STORED, ZipFile, ZipInfo
import zlib

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402


DATASET_SCHEMA_VERSION = "aoi-reference-source-set/2.0"
DATASET_NAME = "MPI-PCB Dataset"
DATASET_DOI = "10.5281/zenodo.8213098"
DATASET_URL = "https://zenodo.org/records/8213098"
PAPER_DOI = "10.3390/s23031353"
PAPER_URL = "https://doi.org/10.3390/s23031353"
LICENSE = "CC BY 4.0"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"

REFERENCE_COUNT = 30
EXPECTED_WIDTH = 4096
EXPECTED_HEIGHT = 2816
EXPECTED_TRAIN_GOOD_COUNT = 1687
TRAIN_GOOD_PREFIX = "original_aligned/train/good/"
_TRAIN_GOOD_PATTERN = re.compile(
    rf"^{re.escape(TRAIN_GOOD_PREFIX)}(?P<index>\d{{4}})\.jpg$"
)
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_HEX8 = re.compile(r"^[0-9a-f]{8}$")
_SIGNED_URL_MARKERS = ("expires=", "signature=", "key-pair-id=", "x-amz-")
_MAX_IMAGE_BYTES = 64 * 1024 * 1024


class FetchError(RuntimeError):
    """Raised when source identity, ZIP bytes, or local output fail closed."""


@dataclass(frozen=True, slots=True)
class ArchiveSpec:
    filename: str
    url: str
    size: int
    md5: str


ARCHIVE = ArchiveSpec(
    filename="original_aligned.zip",
    url=(
        "https://zenodo.org/records/8213098/files/"
        "original_aligned.zip?download=1"
    ),
    size=5_451_602_482,
    md5="c49e03b709b2cddfd6c6344017a1afea",
)


@dataclass(frozen=True, slots=True)
class FetchResult:
    output: Path
    created: bool
    manifest: Mapping[str, Any]


RangeFetcher = Callable[[int, int], bytes]
ArchiveFactory = Callable[[ArchiveSpec], BinaryIO]


class RemoteRangeReader(io.RawIOBase):
    """Seekable, bounded HTTP Range reader with exact response validation.

    A small LRU block cache avoids repeated requests while ``ZipFile`` parses
    local headers.  Offline tests may inject a range fetcher and total size.
    Production validates status 206, ``Content-Range``, content length, archive
    size, and a stable ETag/Last-Modified validator when the server supplies it.
    """

    def __init__(
        self,
        url: str,
        *,
        block_size: int = 1024 * 1024,
        cache_blocks: int = 8,
        timeout: float = 90.0,
        retries: int = 4,
        total_size: int | None = None,
        fetcher: RangeFetcher | None = None,
    ) -> None:
        super().__init__()
        if int(block_size) <= 0 or int(cache_blocks) <= 0:
            raise ValueError("block_size and cache_blocks must be positive")
        if float(timeout) <= 0 or int(retries) <= 0:
            raise ValueError("timeout and retries must be positive")
        self.url = str(url)
        self.block_size = int(block_size)
        self.cache_blocks = int(cache_blocks)
        self.timeout = float(timeout)
        self.retries = int(retries)
        self._position = 0
        self._cache: OrderedDict[int, bytes] = OrderedDict()
        self._fetcher = fetcher
        self._http_validator: tuple[str, str] | None = None
        if fetcher is not None:
            if total_size is None or int(total_size) < 0:
                raise ValueError("Injected fetcher requires a non-negative total_size")
            self.size = int(total_size)
        else:
            self.size = self._discover_size()

    @property
    def http_validator(self) -> Mapping[str, str] | None:
        if self._http_validator is None:
            return None
        key, value = self._http_validator
        return {key: value}

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            position = int(offset)
        elif whence == io.SEEK_CUR:
            position = self._position + int(offset)
        elif whence == io.SEEK_END:
            position = self.size + int(offset)
        else:
            raise ValueError(f"Unsupported seek whence: {whence}")
        if position < 0:
            raise ValueError("Negative seek position")
        self._position = position
        return position

    def read(self, size: int = -1) -> bytes:
        if self.closed:
            raise ValueError("I/O operation on closed range reader")
        if self._position >= self.size:
            return b""
        end = (
            self.size
            if size is None or int(size) < 0
            else min(self.size, self._position + int(size))
        )
        chunks: list[bytes] = []
        while self._position < end:
            block_index = self._position // self.block_size
            block = self._get_block(block_index)
            block_start = block_index * self.block_size
            offset = self._position - block_start
            take = min(end - self._position, len(block) - offset)
            if take <= 0:
                raise FetchError("Remote range block was unexpectedly short")
            chunks.append(block[offset : offset + take])
            self._position += take
        return b"".join(chunks)

    def readinto(self, buffer: Any) -> int:
        payload = self.read(len(buffer))
        buffer[: len(payload)] = payload
        return len(payload)

    def _discover_size(self) -> int:
        payload, total = self._http_range(0, 0)
        if len(payload) != 1 or total <= 0:
            raise FetchError("Could not discover a valid remote archive size")
        return total

    def _get_block(self, index: int) -> bytes:
        cached = self._cache.pop(index, None)
        if cached is not None:
            self._cache[index] = cached
            return cached
        start = index * self.block_size
        end = min(self.size - 1, start + self.block_size - 1)
        if start > end:
            return b""
        if self._fetcher is not None:
            payload = self._fetcher(start, end)
        else:
            payload, total = self._http_range(start, end)
            if total != self.size:
                raise FetchError("Remote archive size changed during transfer")
        expected = end - start + 1
        if len(payload) != expected:
            raise FetchError(
                f"Remote range {start}-{end} returned {len(payload)} bytes; "
                f"expected {expected}"
            )
        self._cache[index] = bytes(payload)
        while len(self._cache) > self.cache_blocks:
            self._cache.popitem(last=False)
        return self._cache[index]

    def _http_range(self, start: int, end: int) -> tuple[bytes, int]:
        last_error: Exception | None = None
        for attempt in range(self.retries):
            headers = {
                "Range": f"bytes={start}-{end}",
                "Accept-Encoding": "identity",
                "User-Agent": "AOI-PCB-MPI-reference-fetcher/1.0",
            }
            if self._http_validator is not None:
                validator_name, validator_value = self._http_validator
                if validator_name == "etag":
                    headers["If-Range"] = validator_value
            request = Request(self.url, headers=headers)
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    status = getattr(response, "status", response.getcode())
                    content_range = response.headers.get("Content-Range", "")
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
                    if status != 206 or match is None:
                        raise FetchError(
                            "Server did not honour the exact HTTP Range request"
                        )
                    actual_start, actual_end, total = (
                        int(value) for value in match.groups()
                    )
                    if (actual_start, actual_end) != (start, end):
                        raise FetchError("Server returned a different byte range")
                    encoding = response.headers.get("Content-Encoding", "").strip()
                    if encoding and encoding.lower() != "identity":
                        raise FetchError("Ranged archive response was content-encoded")
                    expected = end - start + 1
                    declared_length = response.headers.get("Content-Length")
                    if declared_length is not None and int(declared_length) != expected:
                        raise FetchError("Ranged response declared the wrong length")
                    validator = self._response_validator(response.headers)
                    self._check_validator(validator)
                    payload = response.read(expected + 1)
                    if len(payload) != expected:
                        raise FetchError("Ranged response body had the wrong length")
                    return payload, total
            except (
                HTTPError,
                URLError,
                TimeoutError,
                ConnectionError,
                IncompleteRead,
                RemoteDisconnected,
                FetchError,
                OSError,
                ValueError,
            ) as exc:
                last_error = exc
                if attempt + 1 < self.retries:
                    time.sleep(min(8.0, 2.0**attempt))
        raise FetchError(f"HTTP Range failed for {start}-{end}: {last_error}")

    @staticmethod
    def _response_validator(headers: Mapping[str, str]) -> tuple[str, str] | None:
        etag = str(headers.get("ETag", "")).strip()
        if etag:
            return ("etag", etag)
        modified = str(headers.get("Last-Modified", "")).strip()
        if modified:
            return ("last_modified", modified)
        return None

    def _check_validator(self, received: tuple[str, str] | None) -> None:
        if self._http_validator is None:
            self._http_validator = received
        elif received != self._http_validator:
            raise FetchError("Remote archive validator changed during transfer")


def evenly_spaced_indices(total: int, count: int = REFERENCE_COUNT) -> list[int]:
    """Return deterministic integer indices spanning the complete source pool."""

    total = int(total)
    count = int(count)
    if total <= 0 or count <= 0 or count > total:
        raise ValueError("count must be positive and no larger than total")
    if count == 1:
        return [0]
    denominator = count - 1
    return [
        (index * (total - 1) + denominator // 2) // denominator
        for index in range(count)
    ]


def select_stratified_quality_indices(
    members: Mapping[int, ZipInfo],
    count: int = REFERENCE_COUNT,
) -> list[int]:
    """Pick one high-detail JPEG from each contiguous source-sequence stratum.

    MPI-PCB filenames contain only a numeric sequence; they do not encode a
    calibrated pose or focus value.  The central directory does expose each
    JPEG's uncompressed byte size without downloading it.  Within each of
    ``count`` contiguous sequence strata, the largest JPEG is selected as a
    weak, deterministic detail/sharpness proxy.  Sequence stratification keeps
    coverage across the acquisition instead of clustering all choices in one
    short interval.  Real Laplacian focus is still measured after download and
    recorded for human review; byte size is not presented as calibrated focus.
    """

    count = int(count)
    total = len(members)
    if total <= 0 or count <= 0 or count > total:
        raise ValueError("count must be positive and no larger than member count")
    if set(members) != set(range(total)):
        raise FetchError("Quality selection requires a contiguous source sequence")
    selected: list[int] = []
    for stratum in range(count):
        start = stratum * total // count
        end = (stratum + 1) * total // count
        if start >= end:
            raise FetchError("Quality-selection stratum is empty")
        # Prefer the larger source JPEG, then the larger compressed member, then
        # the earliest source index for an explicit deterministic tie-break.
        source_index = max(
            range(start, end),
            key=lambda index: (
                members[index].file_size,
                members[index].compress_size,
                -index,
            ),
        )
        selected.append(source_index)
    return selected


def _pool_metadata_sha256(members: Mapping[int, ZipInfo]) -> str:
    rows = [
        (
            f"{index}|{members[index].filename}|{members[index].file_size}|"
            f"{members[index].compress_size}|{members[index].CRC:08x}|"
            f"{members[index].header_offset}\n"
        )
        for index in range(len(members))
    ]
    return sha256("".join(rows).encode("utf-8")).hexdigest()


def _validate_archive_structure(
    archive: ZipFile,
    *,
    expected_pool_count: int,
    require_large_zip64: bool,
) -> dict[int, ZipInfo]:
    """Validate the ZIP/ZIP64 central directory and resolve train/good members."""

    infos = archive.infolist()
    names = [info.filename for info in infos]
    if len(names) != len(set(names)):
        raise FetchError("ZIP central directory contains duplicate member names")
    if archive.start_dir <= 0:
        raise FetchError("ZIP central directory has an invalid start offset")
    if require_large_zip64 and archive.start_dir <= 0xFFFFFFFF:
        raise FetchError("Expected the 5+ GiB source to resolve through ZIP64 metadata")

    members: dict[int, ZipInfo] = {}
    for info in infos:
        match = _TRAIN_GOOD_PATTERN.fullmatch(info.filename)
        if match is None:
            continue
        source_index = int(match.group("index"))
        if source_index in members:
            raise FetchError(f"Duplicate train/good source index: {source_index}")
        if info.is_dir() or not (0 < info.file_size <= _MAX_IMAGE_BYTES):
            raise FetchError(f"Invalid image member size: {info.filename}")
        if info.compress_size <= 0 or info.compress_type not in {
            ZIP_STORED,
            ZIP_DEFLATED,
        }:
            raise FetchError(f"Unsupported image compression: {info.filename}")
        if info.flag_bits & 0x1:
            raise FetchError(f"Encrypted image member is not accepted: {info.filename}")
        if info.header_offset < 0 or info.header_offset >= archive.start_dir:
            raise FetchError(f"Invalid local-header offset: {info.filename}")
        members[source_index] = info

    expected = set(range(int(expected_pool_count)))
    actual = set(members)
    if actual != expected:
        missing = sorted(expected - actual)[:5]
        unexpected = sorted(actual - expected)[:5]
        raise FetchError(
            "train/good member sequence does not match the declared pool "
            f"(missing={missing}, unexpected={unexpected})"
        )
    return members


def _decode_and_measure(payload: bytes, *, context: str) -> tuple[np.ndarray, float, float]:
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None or image.size == 0:
        raise FetchError(f"Could not decode source image: {context}")
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise FetchError(f"Source image is not RGB/RGBA: {context}")
    height, width = image.shape[:2]
    if (width, height) != (EXPECTED_WIDTH, EXPECTED_HEIGHT):
        raise FetchError(
            f"Unexpected dimensions for {context}: {width}x{height}; "
            f"expected {EXPECTED_WIDTH}x{EXPECTED_HEIGHT}"
        )
    conversion = cv2.COLOR_BGRA2GRAY if image.shape[2] == 4 else cv2.COLOR_BGR2GRAY
    gray = cv2.cvtColor(image, conversion)
    focus = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    if not math.isfinite(focus) or focus <= 0.0:
        raise FetchError(f"Source image has no measurable focus detail: {context}")
    if not math.isfinite(brightness):
        raise FetchError(f"Source image has invalid luminance: {context}")
    return image, focus, brightness


def _read_member(archive: ZipFile, info: ZipInfo) -> bytes:
    try:
        payload = archive.read(info)
    except (BadZipFile, EOFError, NotImplementedError, OSError, RuntimeError) as exc:
        raise FetchError(f"CRC/decompression failed for {info.filename}") from exc
    if len(payload) != info.file_size:
        raise FetchError(f"Decompressed size mismatch for {info.filename}")
    actual_crc = zlib.crc32(payload) & 0xFFFFFFFF
    if actual_crc != info.CRC:
        raise FetchError(f"CRC32 mismatch for {info.filename}")
    return payload


def _entry_for_member(
    *,
    row_index: int,
    source_index: int,
    info: ZipInfo,
    payload: bytes,
    focus: float,
    brightness: float,
) -> dict[str, Any]:
    output_name = f"mpi_pcb_train_good_{source_index:04d}.jpg"
    return {
        "row_index": int(row_index),
        "source_index": int(source_index),
        "path": f"images/{output_name}",
        "label": 0,
        "source_label": "good",
        "source_state": "unmodified",
        "split": "train",
        "source_member": info.filename,
        "width": EXPECTED_WIDTH,
        "height": EXPECTED_HEIGHT,
        "byte_size": len(payload),
        "sha256": sha256(payload).hexdigest(),
        "zip_crc32": f"{info.CRC:08x}",
        "zip_compressed_size": int(info.compress_size),
        "focus_laplacian_variance": round(focus, 6),
        "mean_luminance": round(brightness, 6),
    }


def _attribution_text() -> str:
    return f"""# Attribution — {DATASET_NAME}

The images in this directory are selected byte-for-byte from the aligned,
unmodified (`train/good`) portion of the **{DATASET_NAME}**.

- Dataset record: {DATASET_URL}
- Dataset DOI: https://doi.org/{DATASET_DOI}
- Source archive: `{ARCHIVE.filename}`
- Source archive MD5 declared by Zenodo: `{ARCHIVE.md5}`
- Associated paper: {PAPER_URL}
- Authors: Diulhio Candido de Oliveira, Bogdan Tomoyuki Nassu, and Marco
  Aurelio Wehrmeister
- License declared by Zenodo: **{LICENSE}** ({LICENSE_URL})

The archive-level MD5 is recorded but cannot be verified by a selective Range
download. Each selected member is instead checked against its ZIP central-
directory CRC32 after decompression and receives a local SHA-256 in
`manifest.json`.

The upstream record describes these aligned images as repeated views of an
unmodified PCB from a gas pump. This supports same-layout enrollment, but the
public `good` label does not by itself approve a production Golden. Review all
frames and save the chosen canonical Golden as lossless PNG/TIFF.
"""


def _manifest(
    entries: Sequence[Mapping[str, Any]],
    *,
    selected_indices: Sequence[int],
    http_validator: Mapping[str, str] | None,
    central_directory_offset: int,
    pool_metadata_sha256: str,
) -> dict[str, Any]:
    focus_scores = [float(entry["focus_laplacian_variance"]) for entry in entries]
    luminance = [float(entry["mean_luminance"]) for entry in entries]
    return {
        "schema_version": DATASET_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_kind": "dataset_preprocessed_aligned",
        "dataset_name": DATASET_NAME,
        "dataset_doi": DATASET_DOI,
        "dataset_url": DATASET_URL,
        "paper_doi": PAPER_DOI,
        "license": LICENSE,
        "license_url": LICENSE_URL,
        "archive": {
            "filename": ARCHIVE.filename,
            "stable_url": ARCHIVE.url,
            "byte_size_declared_zenodo": ARCHIVE.size,
            "md5_declared_zenodo": ARCHIVE.md5,
            "full_archive_md5_verified": False,
            "access_method": "validated_http_range_zip64",
            "central_directory_offset": int(central_directory_offset),
            "http_validator": dict(http_validator) if http_validator else None,
        },
        "board": {
            "identity": "one gas-pump PCB described by upstream",
            "layout_relation": "same_layout_same_upstream_board",
            "side": "component_side",
            "upstream_registration": "aligned by the procedure in the paper",
        },
        "selection": {
            "requested_count": len(entries),
            "source_pool": "original_aligned/train/good",
            "source_pool_count": EXPECTED_TRAIN_GOOD_COUNT,
            "source_pool_central_metadata_sha256": pool_metadata_sha256,
            "source_indices": [int(value) for value in selected_indices],
            "strategy": "sequence_stratified_max_source_jpeg_byte_size",
            "strategy_reason": (
                "one candidate per contiguous numeric-sequence stratum; largest "
                "source JPEG is a weak detail proxy, not a calibrated focus or pose label"
            ),
            "same_layout_consensus_allowed": True,
            "different_layout_mixing_allowed": False,
            "upstream_label_required": "good/unmodified",
        },
        "quality": {
            "focus_metric": "variance_of_grayscale_laplacian",
            "focus_min": min(focus_scores),
            "focus_median": statistics.median(focus_scores),
            "focus_max": max(focus_scores),
            "mean_luminance_min": min(luminance),
            "mean_luminance_median": statistics.median(luminance),
            "mean_luminance_max": max(luminance),
            "human_visual_review_required": True,
        },
        "production_status": {
            "golden_recipe_created": False,
            "production_eligible": False,
            "reason": (
                "public aligned JPEG inputs still require operator review, lossless "
                "Golden export, alignment gates, and real-camera validation"
            ),
        },
        "files": [dict(entry) for entry in entries],
    }


def _assert_manifest_portable(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise FetchError("Manifest object keys must be strings")
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
        raise FetchError("Manifest must not contain an expiring signed URL")
    if "://" not in value and (
        PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute()
    ):
        raise FetchError("Manifest must not contain absolute workstation paths")


def _safe_image_path(output: Path, value: Any, *, row_index: int) -> Path:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise FetchError(f"Manifest path is not portable for row {row_index}")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or "." in relative.parts
        or relative.as_posix() != value
        or len(relative.parts) != 2
        or relative.parts[0] != "images"
    ):
        raise FetchError(f"Manifest path is not canonical for row {row_index}")
    path = output.joinpath(*relative.parts)
    try:
        path.resolve().relative_to(output.resolve())
    except ValueError as exc:
        raise FetchError(f"Manifest image escapes output for row {row_index}") from exc
    return path


def validate_existing_output(
    output_dir: str | Path,
    *,
    count: int = REFERENCE_COUNT,
) -> Mapping[str, Any]:
    """Revalidate an existing published set without touching the network."""

    output = Path(output_dir).expanduser().resolve()
    try:
        manifest_bytes = (output / "manifest.json").read_bytes()
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FetchError("Existing output has no readable manifest.json") from exc
    if not isinstance(manifest, Mapping):
        raise FetchError("Existing manifest must be a JSON object")
    if manifest.get("schema_version") != DATASET_SCHEMA_VERSION:
        raise FetchError("Existing manifest schema does not match")
    if manifest.get("dataset_doi") != DATASET_DOI or manifest.get("license") != LICENSE:
        raise FetchError("Existing manifest source identity does not match MPI-PCB")
    _assert_manifest_portable(manifest)
    files = manifest.get("files")
    if not isinstance(files, list) or len(files) != int(count):
        raise FetchError(f"Existing manifest must contain exactly {count} files")

    selection = manifest.get("selection")
    if not isinstance(selection, Mapping):
        raise FetchError("Existing manifest has no selection object")
    selected = selection.get("source_indices")
    if (
        not isinstance(selected, list)
        or len(selected) != int(count)
        or any(isinstance(value, bool) or not isinstance(value, int) for value in selected)
        or len(set(selected)) != len(selected)
    ):
        raise FetchError("Existing manifest has invalid selected source indices")
    for stratum, source_index in enumerate(selected):
        start = stratum * EXPECTED_TRAIN_GOOD_COUNT // count
        end = (stratum + 1) * EXPECTED_TRAIN_GOOD_COUNT // count
        if not start <= source_index < end:
            raise FetchError("Existing source selection violates sequence stratification")
    seen_hashes: set[str] = set()
    for row_index, (entry, source_index) in enumerate(zip(files, selected)):
        if not isinstance(entry, Mapping) or entry.get("row_index") != row_index:
            raise FetchError("Existing file rows are missing or out of order")
        if entry.get("source_index") != source_index:
            raise FetchError("Existing source selection does not match deterministic policy")
        if entry.get("label") != 0 or isinstance(entry.get("label"), bool):
            raise FetchError(f"Existing row {row_index} is not label=0")
        expected_member = f"{TRAIN_GOOD_PREFIX}{source_index:04d}.jpg"
        if entry.get("source_member") != expected_member:
            raise FetchError(f"Existing source member changed for row {row_index}")
        path = _safe_image_path(output, entry.get("path"), row_index=row_index)
        if not path.is_file() or path.is_symlink():
            raise FetchError(f"Existing image is missing for row {row_index}")
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise FetchError(f"Existing image is unreadable for row {row_index}") from exc
        if entry.get("byte_size") != len(payload):
            raise FetchError(f"Existing image byte size changed for row {row_index}")
        digest = entry.get("sha256")
        if not isinstance(digest, str) or not _HEX64.fullmatch(digest):
            raise FetchError(f"Existing SHA-256 is invalid for row {row_index}")
        if sha256(payload).hexdigest() != digest:
            raise FetchError(f"Existing image SHA-256 changed for row {row_index}")
        if digest in seen_hashes:
            raise FetchError("Existing images are not unique by SHA-256")
        seen_hashes.add(digest)
        crc_text = entry.get("zip_crc32")
        if not isinstance(crc_text, str) or not _HEX8.fullmatch(crc_text):
            raise FetchError(f"Existing CRC32 is invalid for row {row_index}")
        if f"{zlib.crc32(payload) & 0xFFFFFFFF:08x}" != crc_text:
            raise FetchError(f"Existing image CRC32 changed for row {row_index}")
        _decode_and_measure(payload, context=f"existing row {row_index}")
        if (entry.get("width"), entry.get("height")) != (
            EXPECTED_WIDTH,
            EXPECTED_HEIGHT,
        ):
            raise FetchError(f"Existing dimensions changed for row {row_index}")
    if not (output / "ATTRIBUTION.md").is_file():
        raise FetchError("Existing output is missing ATTRIBUTION.md")
    return manifest


def _safe_remove_staging(stage: Path, parent: Path) -> None:
    try:
        resolved_stage = stage.resolve()
        resolved_parent = parent.resolve()
        resolved_stage.relative_to(resolved_parent)
    except (OSError, ValueError):
        return
    if resolved_stage.parent != resolved_parent or not resolved_stage.name.startswith("."):
        return
    shutil.rmtree(resolved_stage, ignore_errors=True)


def fetch_reference_set(
    output_dir: str | Path,
    *,
    count: int = REFERENCE_COUNT,
    archive_factory: ArchiveFactory | None = None,
    timeout: float = 90.0,
    retries: int = 4,
    block_size: int = 1024 * 1024,
) -> FetchResult:
    """Fetch and atomically publish a deterministic same-layout reference set."""

    count = int(count)
    if count <= 0 or count > EXPECTED_TRAIN_GOOD_COUNT:
        raise ValueError("count must be within the configured train/good pool")
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        manifest = validate_existing_output(output, count=count)
        return FetchResult(output=output, created=False, manifest=manifest)

    output.parent.mkdir(parents=True, exist_ok=True)
    prefix = f".{output.name}.tmp-{uuid4().hex[:10]}-"
    stage = Path(tempfile.mkdtemp(prefix=prefix, dir=output.parent)).resolve()
    try:
        images_dir = stage / "images"
        images_dir.mkdir()
        if archive_factory is None:
            reader: BinaryIO = RemoteRangeReader(
                ARCHIVE.url,
                timeout=timeout,
                retries=retries,
                block_size=block_size,
            )
            if not isinstance(reader, RemoteRangeReader) or reader.size != ARCHIVE.size:
                raise FetchError(
                    f"Remote archive size is not the declared {ARCHIVE.size} bytes"
                )
            require_large_zip64 = True
        else:
            reader = archive_factory(ARCHIVE)
            require_large_zip64 = False

        entries: list[dict[str, Any]] = []
        seen_hashes: set[str] = set()
        http_validator: Mapping[str, str] | None = None
        selected_indices: list[int] = []
        pool_metadata_sha256 = ""
        with reader:
            try:
                archive = ZipFile(reader)
            except (BadZipFile, OSError) as exc:
                raise FetchError("Remote source is not a readable ZIP/ZIP64 archive") from exc
            with archive:
                members = _validate_archive_structure(
                    archive,
                    expected_pool_count=EXPECTED_TRAIN_GOOD_COUNT,
                    require_large_zip64=require_large_zip64,
                )
                selected_indices = select_stratified_quality_indices(members, count)
                pool_metadata_sha256 = _pool_metadata_sha256(members)
                central_directory_offset = int(archive.start_dir)
                for row_index, source_index in enumerate(selected_indices):
                    info = members[source_index]
                    print(
                        f"[{row_index + 1:02d}/{count:02d}] {info.filename}",
                        flush=True,
                    )
                    payload = _read_member(archive, info)
                    _image, focus, brightness = _decode_and_measure(
                        payload, context=info.filename
                    )
                    entry = _entry_for_member(
                        row_index=row_index,
                        source_index=source_index,
                        info=info,
                        payload=payload,
                        focus=focus,
                        brightness=brightness,
                    )
                    digest = str(entry["sha256"])
                    if digest in seen_hashes:
                        raise FetchError(
                            f"Selected member duplicates another image: {info.filename}"
                        )
                    seen_hashes.add(digest)
                    target = _safe_image_path(
                        stage, entry["path"], row_index=row_index
                    )
                    target.write_bytes(payload)
                    entries.append(entry)
                if isinstance(reader, RemoteRangeReader):
                    http_validator = reader.http_validator

        manifest = _manifest(
            entries,
            selected_indices=selected_indices,
            http_validator=http_validator,
            central_directory_offset=central_directory_offset,
            pool_metadata_sha256=pool_metadata_sha256,
        )
        _assert_manifest_portable(manifest)
        (stage / "ATTRIBUTION.md").write_text(
            _attribution_text(), encoding="utf-8", newline="\n"
        )
        (stage / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        validate_existing_output(stage, count=count)
        if output.exists():
            raise FetchError("Output appeared during download; refusing to overwrite")
        os.replace(stage, output)
        return FetchResult(output=output, created=True, manifest=manifest)
    except BaseException:
        _safe_remove_staging(stage, output.parent)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Selectively fetch 30 aligned train/good MPI-PCB images from the "
            "5.45 GB Zenodo ZIP64 archive without downloading it in full."
        )
    )
    parser.add_argument("--output", required=True, help="Reference-set directory to create.")
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--block-size-mib", type=float, default=1.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    block_size = int(float(args.block_size_mib) * 1024 * 1024)
    try:
        result = fetch_reference_set(
            args.output,
            timeout=args.timeout,
            retries=args.retries,
            block_size=block_size,
        )
    except (FetchError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    action = "Downloaded and verified" if result.created else "Already verified"
    print(f"{action}: {REFERENCE_COUNT} MPI-PCB same-layout images -> {result.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
