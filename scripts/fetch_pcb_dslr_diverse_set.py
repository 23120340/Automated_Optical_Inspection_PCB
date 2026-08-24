"""Fetch 30 distinct high-resolution PCBs from the TU Wien PCB DSLR dataset.

Only ``rec1`` plus its board mask and IC annotation are extracted for
``pcb1`` through ``pcb30``.  The two source ZIPs are not downloaded in full:
``zipfile`` reads them through a seekable HTTP Range cache, so the command
transfers the selected members and the small central-directory regions only.

The dataset source text limits use to non-commercial research.  This command
preserves that restriction and full per-file provenance in ``manifest.json``;
it does not relicense the images under the zlib license of the companion code.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import dataclass
from hashlib import sha256
from http.client import IncompleteRead, RemoteDisconnected
import io
import json
import math
from pathlib import Path
import re
import sys
import time
from typing import Any, Callable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZipFile, ZipInfo

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402


DATASET_SCHEMA_VERSION = "aoi-reference-source-set/2.0"
DATASET_NAME = "TU Wien PCB DSLR"
DATASET_DOI = "10.5281/zenodo.3886553"
DATASET_URL = "https://zenodo.org/records/3886553"
OFFICIAL_PAGE = "https://cvl.tuwien.ac.at/research/cvl-databases/pcb-dslr-dataset/"
COMPANION_CODE = "https://github.com/cpra/mva15"
USAGE_RESTRICTION = "noncommercial_research_only_due_to_source_text"
EXPECTED_WIDTH = 4928
EXPECTED_HEIGHT = 3280


class FetchError(RuntimeError):
    """Raised when remote bytes, ZIP members or local provenance fail closed."""


@dataclass(frozen=True, slots=True)
class ArchiveSpec:
    part: int
    filename: str
    url: str
    md5: str
    first_board: int
    last_board: int


ARCHIVES = (
    ArchiveSpec(
        part=1,
        filename="cvl_pcb_dslr_1.zip",
        url="https://zenodo.org/records/3886553/files/cvl_pcb_dslr_1.zip?download=1",
        md5="480cbb572441f2b6fb013db965d3d4a1",
        first_board=1,
        last_board=20,
    ),
    ArchiveSpec(
        part=2,
        filename="cvl_pcb_dslr_2.zip",
        url="https://zenodo.org/records/3886553/files/cvl_pcb_dslr_2.zip?download=1",
        md5="8df69ce61a65b92aee058197453b594d",
        first_board=21,
        last_board=40,
    ),
)


RangeFetcher = Callable[[int, int], bytes]


class RemoteRangeReader(io.RawIOBase):
    """A bounded, seekable HTTP Range reader suitable for :class:`ZipFile`.

    Blocks are cached by index.  A caller may inject ``total_size`` and
    ``fetcher`` for deterministic offline tests; production discovers size
    with a one-byte range request and validates every ``Content-Range``.
    """

    def __init__(
        self,
        url: str,
        *,
        block_size: int = 4 * 1024 * 1024,
        cache_blocks: int = 8,
        timeout: float = 60.0,
        retries: int = 4,
        total_size: int | None = None,
        fetcher: RangeFetcher | None = None,
    ) -> None:
        super().__init__()
        if block_size <= 0 or cache_blocks <= 0:
            raise ValueError("block_size and cache_blocks must be positive")
        self.url = str(url)
        self.block_size = int(block_size)
        self.cache_blocks = int(cache_blocks)
        self.timeout = float(timeout)
        self.retries = int(retries)
        self._position = 0
        self._cache: OrderedDict[int, bytes] = OrderedDict()
        self._injected_fetcher = fetcher
        if fetcher is not None:
            if total_size is None or int(total_size) < 0:
                raise ValueError("Injected fetcher requires a non-negative total_size")
            self.size = int(total_size)
        else:
            self.size = self._discover_size()

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
        return self._position

    def read(self, size: int = -1) -> bytes:
        if self.closed:
            raise ValueError("I/O operation on closed range reader")
        if self._position >= self.size:
            return b""
        if size is None or int(size) < 0:
            end = self.size
        else:
            end = min(self.size, self._position + int(size))
        chunks: list[bytes] = []
        while self._position < end:
            block_index = self._position // self.block_size
            block = self._block(block_index)
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

    def _block(self, index: int) -> bytes:
        cached = self._cache.pop(index, None)
        if cached is not None:
            self._cache[index] = cached
            return cached
        start = index * self.block_size
        end = min(self.size - 1, start + self.block_size - 1)
        if start > end:
            return b""
        if self._injected_fetcher is not None:
            payload = self._injected_fetcher(start, end)
        else:
            payload, total = self._http_range(start, end)
            if total != self.size:
                raise FetchError("Remote archive size changed during download")
        expected = end - start + 1
        if len(payload) != expected:
            raise FetchError(
                f"Remote range {start}-{end} returned {len(payload)} bytes, expected {expected}"
            )
        self._cache[index] = payload
        while len(self._cache) > self.cache_blocks:
            self._cache.popitem(last=False)
        return payload

    def _http_range(self, start: int, end: int) -> tuple[bytes, int]:
        last_error: Exception | None = None
        for attempt in range(max(1, self.retries)):
            request = Request(
                self.url,
                headers={
                    "Range": f"bytes={start}-{end}",
                    "Accept-Encoding": "identity",
                    "User-Agent": "AOI-PCB-reference-fetcher/1.0",
                },
            )
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    status = getattr(response, "status", response.getcode())
                    content_range = response.headers.get("Content-Range", "")
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
                    if status != 206 or match is None:
                        raise FetchError(
                            "Server did not honour the exact HTTP Range request"
                        )
                    actual_start, actual_end, total = (int(value) for value in match.groups())
                    if (actual_start, actual_end) != (start, end):
                        raise FetchError("Server returned a different byte range")
                    payload = response.read()
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
            ) as exc:
                last_error = exc
                if attempt + 1 < max(1, self.retries):
                    time.sleep(min(8.0, 2.0**attempt))
        raise FetchError(f"HTTP Range failed for {start}-{end}: {last_error}")


def _member_names(board_number: int, recording: int = 1) -> tuple[str, str, str]:
    prefix = f"pcb{int(board_number)}/rec{int(recording)}"
    return (f"{prefix}.jpg", f"{prefix}-mask.png", f"{prefix}-annot.txt")


def _zip_info(archive: ZipFile, name: str, *, allow_empty: bool = False) -> ZipInfo:
    try:
        info = archive.getinfo(name)
    except KeyError as exc:
        raise FetchError(f"Archive is missing required member: {name}") from exc
    if info.is_dir() or (info.file_size <= 0 and not allow_empty):
        raise FetchError(f"Archive member is empty or not a file: {name}")
    return info


def select_members(
    archive: ZipFile, board_numbers: Sequence[int]
) -> dict[int, tuple[ZipInfo, ZipInfo, ZipInfo]]:
    """Resolve the exact image/mask/annotation triplet for each board."""

    selected: dict[int, tuple[ZipInfo, ZipInfo, ZipInfo]] = {}
    for board_number in board_numbers:
        names = _member_names(board_number)
        selected[int(board_number)] = (
            _zip_info(archive, names[0]),
            _zip_info(archive, names[1]),
            _zip_info(archive, names[2], allow_empty=True),
        )
    return selected


def _sha(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def _validate_annotation(payload: bytes, *, board_id: str) -> int:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise FetchError(f"Annotation is not UTF-8 for {board_id}") from exc
    count = 0
    for line_number, raw in enumerate(text.splitlines(), start=1):
        parts = raw.strip().split()
        if not parts:
            continue
        if len(parts) < 5:
            raise FetchError(f"Invalid annotation {board_id}, line {line_number}")
        try:
            values = [float(value) for value in parts[:5]]
        except ValueError as exc:
            raise FetchError(
                f"Non-numeric annotation {board_id}, line {line_number}"
            ) from exc
        if not all(math.isfinite(value) for value in values) or values[2] <= 0 or values[3] <= 0:
            raise FetchError(f"Invalid annotation geometry {board_id}, line {line_number}")
        count += 1
    return count


def _decode_image(payload: bytes, flags: int) -> np.ndarray:
    array = np.frombuffer(payload, dtype=np.uint8)
    image = cv2.imdecode(array, flags)
    if image is None or image.size == 0:
        raise FetchError("Downloaded image could not be decoded")
    return image


def _extract_board(
    archive: ZipFile,
    archive_spec: ArchiveSpec,
    board_number: int,
    infos: tuple[ZipInfo, ZipInfo, ZipInfo],
    boards_root: Path,
) -> dict[str, Any]:
    image_info, mask_info, annotation_info = infos
    try:
        image_payload = archive.read(image_info)
        mask_payload = archive.read(mask_info)
        annotation_payload = archive.read(annotation_info)
    except (BadZipFile, OSError, RuntimeError) as exc:
        raise FetchError(f"CRC/decompression failed for pcb{board_number}") from exc
    image = _decode_image(image_payload, cv2.IMREAD_COLOR)
    mask = _decode_image(mask_payload, cv2.IMREAD_GRAYSCALE)
    if image.shape[:2] != (EXPECTED_HEIGHT, EXPECTED_WIDTH):
        raise FetchError(
            f"pcb{board_number} has unexpected image dimensions {image.shape[1]}x{image.shape[0]}"
        )
    if mask.shape[:2] != image.shape[:2]:
        raise FetchError(f"pcb{board_number} mask dimensions do not match image")
    mask_values = set(int(value) for value in np.unique(mask))
    if not mask_values.issubset({0, 255}) or 255 not in mask_values:
        raise FetchError(f"pcb{board_number} mask is not a non-empty 0/255 mask")
    annotation_count = _validate_annotation(
        annotation_payload, board_id=f"pcb{board_number}"
    )

    board_id = f"pcb_dslr_{board_number:03d}"
    board_dir = boards_root / board_id
    board_dir.mkdir(parents=True)
    image_path = board_dir / "rec1.jpg"
    mask_path = board_dir / "rec1-mask.png"
    annotation_path = board_dir / "rec1-annot.txt"
    def member_metadata(info: ZipInfo, payload: bytes) -> dict[str, Any]:
        return {
            "source_member": info.filename,
            "zip_crc32": f"{info.CRC:08x}",
            "byte_size": len(payload),
            "sha256": _sha(payload),
        }

    record = {
        "board_id": board_id,
        "upstream_board_id": f"pcb{board_number}",
        "recording_id": "rec1",
        "source_archive": archive_spec.filename,
        "source_archive_md5_declared_upstream": archive_spec.md5,
        "full_archive_md5_verified": False,
        "image_path": f"boards/{board_id}/rec1.jpg",
        "mask_path": f"boards/{board_id}/rec1-mask.png",
        "annotation_path": f"boards/{board_id}/rec1-annot.txt",
        "width": EXPECTED_WIDTH,
        "height": EXPECTED_HEIGHT,
        "image_sha256": _sha(image_payload),
        "image_byte_size": len(image_payload),
        "mask_sha256": _sha(mask_payload),
        "mask_byte_size": len(mask_payload),
        "annotation_sha256": _sha(annotation_payload),
        "annotation_byte_size": len(annotation_payload),
        "upstream_ic_count": annotation_count,
        "upstream": {
            "image": member_metadata(image_info, image_payload),
            "mask": member_metadata(mask_info, mask_payload),
            "annotation": member_metadata(annotation_info, annotation_payload),
        },
        "quality_status": "unknown_requires_human_review",
        "side": "unknown",
    }
    image_path.write_bytes(image_payload)
    mask_path.write_bytes(mask_payload)
    annotation_path.write_bytes(annotation_payload)
    (board_dir / "source_record.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return record


def _cached_board_record(
    boards_root: Path, board_number: int
) -> dict[str, Any] | None:
    """Return a fully revalidated partial download, or ``None`` if absent."""

    board_id = f"pcb_dslr_{board_number:03d}"
    board_dir = boards_root / board_id
    if not board_dir.exists():
        return None
    record_path = board_dir / "source_record.json"
    if not record_path.is_file():
        raise FetchError(
            f"Incomplete partial cache for {board_id}; remove only {board_dir} and retry"
        )
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FetchError(f"Unreadable partial cache record for {board_id}") from exc
    if not isinstance(record, dict) or record.get("board_id") != board_id:
        raise FetchError(f"Invalid partial cache identity for {board_id}")
    paths = {
        "image": board_dir / "rec1.jpg",
        "mask": board_dir / "rec1-mask.png",
        "annotation": board_dir / "rec1-annot.txt",
    }
    expected = {
        "image": str(record.get("image_sha256", "")),
        "mask": str(record.get("mask_sha256", "")),
        "annotation": str(record.get("annotation_sha256", "")),
    }
    for name, path in paths.items():
        try:
            actual = _sha(path.read_bytes())
        except OSError as exc:
            raise FetchError(f"Partial cache is missing {name} for {board_id}") from exc
        if not expected[name] or actual != expected[name]:
            raise FetchError(f"Partial cache {name} failed SHA-256 for {board_id}")
    image = cv2.imread(str(paths["image"]), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(paths["mask"]), cv2.IMREAD_GRAYSCALE)
    if image is None or mask is None or image.shape[:2] != mask.shape[:2]:
        raise FetchError(f"Partial cache assets do not decode consistently for {board_id}")
    if image.shape[:2] != (EXPECTED_HEIGHT, EXPECTED_WIDTH):
        raise FetchError(f"Partial cache dimensions are invalid for {board_id}")
    _validate_annotation(paths["annotation"].read_bytes(), board_id=board_id)
    return record


def _attribution_text() -> str:
    return f"""# Attribution and use restriction

These files are selected, source-as-received records from the **{DATASET_NAME}**
dataset ({DATASET_DOI}).

- Official dataset page: {OFFICIAL_PAGE}
- Stable archive: {DATASET_URL}
- Companion API/code: {COMPANION_CODE}
- Camera/resolution reported upstream: Nikon D4, f/2.8 lens, polarization
  filter; 4928×3280 pixels.
- Selection: `rec1` from 30 distinct upstream boards, `pcb1` through `pcb30`.

## Use restriction

The official dataset page and the Zenodo description say the dataset is freely
available for **non-commercial research use**. Treat that source text as the
controlling restriction for these image files. Zenodo machine metadata may
display CC BY 4.0, while the companion code repository is zlib-licensed; the
zlib license applies to the code, not automatically to the images. Obtain
permission from the authors before commercial use or redistribution.

Please cite:

> C. Pramerdorfer and M. Kampel, “A Dataset for Computer-Vision-Based PCB
> Analysis,” 14th IAPR International Conference on Machine Vision Applications
> (MVA), 2015, pp. 378–381. DOI: 10.1109/MVA.2015.7153209.

Every local file is bound to its upstream ZIP member, CRC32 and SHA-256 in
`manifest.json`.
"""


def fetch_reference_set(
    output_dir: str | Path,
    *,
    count: int = 30,
    archive_factory: Callable[[ArchiveSpec], io.RawIOBase] | None = None,
) -> Path:
    """Fetch a deterministic set of distinct boards without overwriting data."""

    if int(count) < 1 or int(count) > 40:
        raise ValueError("count must be between 1 and 40 for the configured archives")
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    allowed_existing = {"NHUNG_VIEC_BAN_CAN_LAM.md"}
    unexpected = [item.name for item in output.iterdir() if item.name not in allowed_existing]
    if unexpected:
        raise FetchError(
            f"Output contains existing data; refusing to overwrite: {sorted(unexpected)}"
        )

    partial = (output.parent / f".{output.name}.partial").resolve()
    try:
        partial.relative_to(output.parent)
    except ValueError as exc:
        raise FetchError("Partial-download path escapes the output parent") from exc
    partial.mkdir(exist_ok=True)
    boards_root = partial / "boards"
    boards_root.mkdir(exist_ok=True)
    files: list[dict[str, Any]] = []
    desired = list(range(1, int(count) + 1))
    cached: dict[int, dict[str, Any]] = {}
    for board_number in desired:
        record = _cached_board_record(boards_root, board_number)
        if record is not None:
            cached[board_number] = record
            print(f"  reusing verified partial pcb{board_number}/rec1", flush=True)

    for spec in ARCHIVES:
        board_numbers = [
            value
            for value in desired
            if spec.first_board <= value <= spec.last_board and value not in cached
        ]
        if not board_numbers:
            continue
        print(
            f"Reading {spec.filename}: pcb{board_numbers[0]}..pcb{board_numbers[-1]}",
            flush=True,
        )
        reader = (
            archive_factory(spec)
            if archive_factory is not None
            else RemoteRangeReader(spec.url)
        )
        try:
            with reader, ZipFile(reader) as archive:
                selected = select_members(archive, board_numbers)
                for board_number in board_numbers:
                    print(f"  extracting pcb{board_number}/rec1", flush=True)
                    record = _extract_board(
                        archive,
                        spec,
                        board_number,
                        selected[board_number],
                        boards_root,
                    )
                    cached[board_number] = record
        except BadZipFile as exc:
            raise FetchError(f"Remote archive is not a valid ZIP: {spec.filename}") from exc

    files = [cached[board_number] for board_number in desired if board_number in cached]
    files.sort(key=lambda item: int(str(item["upstream_board_id"])[3:]))
    if len(files) != int(count):
        raise FetchError(f"Extracted {len(files)} boards, expected {count}")
    manifest = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "source_dataset": DATASET_NAME,
        "source_kind": "dataset_preprocessed_source_as_received",
        "dataset_doi": DATASET_DOI,
        "dataset_url": DATASET_URL,
        "official_page": OFFICIAL_PAGE,
        "companion_code": COMPANION_CODE,
        "usage_restriction": USAGE_RESTRICTION,
        "download_method": (
            "selective_http_range; selected ZIP members CRC-checked; "
            "full archive MD5 not recomputed"
        ),
        "license_note": (
            "Source text says non-commercial research use; companion code zlib "
            "does not relicense images. See ATTRIBUTION.md."
        ),
        "selection": {
            "policy": "one rec1 image from each distinct upstream PCB",
            "board_numbers": desired,
            "recording_id": "rec1",
            "distinct_layout_count": len(files),
            "same_sku_consensus_allowed": False,
        },
        "files": files,
    }
    (partial / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (partial / "ATTRIBUTION.md").write_text(
        _attribution_text(), encoding="utf-8"
    )
    (partial / "README.md").write_text(
        "# 30 distinct PCB DSLR references\n\n"
        "This set contains 30 different PCB layouts, not repeated captures of one SKU.\n"
        "Do not align or vote across the boards. Each board needs its own Golden/PnP draft.\n"
        "See `NHUNG_VIEC_BAN_CAN_LAM.md` and `ATTRIBUTION.md`.\n",
        encoding="utf-8",
    )
    (partial / "boards").replace(output / "boards")
    for name in ("manifest.json", "ATTRIBUTION.md", "README.md"):
        (partial / name).replace(output / name)
    try:
        partial.rmdir()
    except OSError as exc:
        raise FetchError(f"Completed output, but partial state is not empty: {partial}") from exc
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "datasets"
        / "reference_sets"
        / "pcb_dslr_30_diverse",
    )
    parser.add_argument("--count", type=int, default=30)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = fetch_reference_set(args.output, count=args.count)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Fetched {args.count} distinct PCB references -> {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
