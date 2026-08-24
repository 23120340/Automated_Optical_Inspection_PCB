"""Offline contract tests for the selective MPI-PCB ZIP64 fetcher."""

from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import struct
import sys
from zipfile import ZIP_DEFLATED, ZipFile

import cv2
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_reference_bundle as bundle  # noqa: E402
import fetch_mpi_pcb_same_layout_set as mpi  # noqa: E402


def _jpeg(index: int, width: int, height: int) -> bytes:
    rng = np.random.default_rng(index + 191)
    image = rng.integers(15, 45, size=(height, width, 3), dtype=np.uint8)
    cv2.rectangle(
        image,
        (4 + index % 3, 3),
        (width - 5, height - 4),
        (30 + index, 145, 65),
        -1,
    )
    cv2.line(image, (7, 7), (width - 8, height - 8), (235, 235, 235), 2)
    cv2.circle(
        image,
        (width // 2 + index % 2, height // 2),
        max(2, min(width, height) // 8),
        (10, 10, 10),
        -1,
    )
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 94])
    assert ok
    return encoded.tobytes()


def _archive_bytes(
    pool_count: int,
    width: int,
    height: int,
    *,
    missing_index: int | None = None,
    duplicate_index: int | None = None,
    bad_dimensions_index: int | None = None,
    force_zip64_eocd: bool = False,
) -> tuple[bytes, dict[int, bytes]]:
    members: dict[int, bytes] = {}
    handle = BytesIO()
    with ZipFile(handle, "w", compression=ZIP_DEFLATED, allowZip64=True) as archive:
        archive.writestr("original_aligned/", b"")
        archive.writestr("original_aligned/train/", b"")
        archive.writestr("original_aligned/train/good/", b"")
        for index in range(pool_count):
            if index == missing_index:
                continue
            source = 0 if index == duplicate_index else index
            payload = _jpeg(
                source,
                width + 1 if index == bad_dimensions_index else width,
                height,
            )
            members[index] = payload
            archive.writestr(
                f"original_aligned/train/good/{index:04d}.jpg", payload
            )
        archive.writestr("original_aligned/test/good/0000.jpg", _jpeg(500, width, height))
    payload = handle.getvalue()
    if force_zip64_eocd:
        payload = _with_zip64_end_records(payload)
    return payload, members


def _with_zip64_end_records(payload: bytes) -> bytes:
    """Replace a small ZIP EOCD with valid ZIP64 EOCD + locator records."""

    signature = b"PK\x05\x06"
    offset = payload.rfind(signature)
    assert offset >= 0
    fields = struct.unpack_from("<4s4H2LH", payload, offset)
    (
        _signature,
        disk_number,
        central_disk,
        entries_on_disk,
        entries_total,
        central_size,
        central_offset,
        comment_size,
    ) = fields
    assert disk_number == central_disk == 0
    comment = payload[offset + 22 : offset + 22 + comment_size]
    assert offset + 22 + comment_size == len(payload)
    zip64_offset = offset
    zip64_eocd = struct.pack(
        "<4sQ2H2L4Q",
        b"PK\x06\x06",
        44,
        45,
        45,
        0,
        0,
        entries_on_disk,
        entries_total,
        central_size,
        central_offset,
    )
    locator = struct.pack("<4sLQL", b"PK\x06\x07", 0, zip64_offset, 1)
    sentinel_eocd = struct.pack(
        "<4s4H2LH",
        signature,
        0,
        0,
        0xFFFF,
        0xFFFF,
        0xFFFFFFFF,
        0xFFFFFFFF,
        len(comment),
    )
    return payload[:offset] + zip64_eocd + locator + sentinel_eocd + comment


def _corrupt_member_compressed_bytes(payload: bytes, member_name: str) -> bytes:
    data = bytearray(payload)
    with ZipFile(BytesIO(payload)) as archive:
        info = archive.getinfo(member_name)
    (
        signature,
        _version,
        _flags,
        _compression,
        _time,
        _date,
        _crc,
        _compressed_size,
        _file_size,
        name_size,
        extra_size,
    ) = struct.unpack_from("<4s5H3L2H", data, info.header_offset)
    assert signature == b"PK\x03\x04"
    start = info.header_offset + 30 + name_size + extra_size
    assert info.compress_size > 8
    data[start + info.compress_size // 2] ^= 0x5A
    return bytes(data)


def _configure_small_source(
    monkeypatch: pytest.MonkeyPatch,
    *,
    pool_count: int,
    width: int,
    height: int,
) -> None:
    monkeypatch.setattr(mpi, "EXPECTED_TRAIN_GOOD_COUNT", pool_count)
    monkeypatch.setattr(mpi, "EXPECTED_WIDTH", width)
    monkeypatch.setattr(mpi, "EXPECTED_HEIGHT", height)


def test_remote_range_reader_is_seekable_bounded_and_block_cached() -> None:
    payload = bytes(range(251)) * 17
    requests: list[tuple[int, int]] = []

    def get_range(start: int, end: int) -> bytes:
        requests.append((start, end))
        return payload[start : end + 1]

    reader = mpi.RemoteRangeReader(
        "memory://zip64",
        block_size=64,
        cache_blocks=2,
        total_size=len(payload),
        fetcher=get_range,
    )
    reader.seek(50)
    assert reader.read(40) == payload[50:90]
    reader.seek(55)
    assert reader.read(5) == payload[55:60]
    assert requests.count((0, 63)) == 1
    assert requests.count((64, 127)) == 1
    reader.seek(-17, 2)
    assert reader.read() == payload[-17:]


def test_default_selection_is_exact_and_spans_the_complete_good_pool() -> None:
    assert mpi.evenly_spaced_indices(1687, 30) == [
        0,
        58,
        116,
        174,
        233,
        291,
        349,
        407,
        465,
        523,
        581,
        640,
        698,
        756,
        814,
        872,
        930,
        988,
        1046,
        1105,
        1163,
        1221,
        1279,
        1337,
        1395,
        1453,
        1512,
        1570,
        1628,
        1686,
    ]


def test_fetch_reads_zip64_central_directory_and_publishes_portable_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pool_count, count = 9, 3
    width, height = 72, 48
    _configure_small_source(
        monkeypatch, pool_count=pool_count, width=width, height=height
    )
    archive_bytes, upstream = _archive_bytes(
        pool_count, width, height, force_zip64_eocd=True
    )
    with ZipFile(BytesIO(archive_bytes)) as archive:
        assert archive.getinfo("original_aligned/train/good/0008.jpg").file_size > 0
        expected_members = mpi._validate_archive_structure(
            archive, expected_pool_count=pool_count, require_large_zip64=False
        )
        expected_selected = mpi.select_stratified_quality_indices(
            expected_members, count
        )

    output = tmp_path / "mpi-reference"
    result = mpi.fetch_reference_set(
        output,
        count=count,
        archive_factory=lambda _spec: BytesIO(archive_bytes),
    )
    assert result.created is True
    manifest_text = (output / "manifest.json").read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    assert manifest["dataset_doi"] == "10.5281/zenodo.8213098"
    assert manifest["license"] == "CC BY 4.0"
    assert manifest["archive"]["access_method"] == "validated_http_range_zip64"
    assert manifest["archive"]["full_archive_md5_verified"] is False
    assert manifest["selection"]["source_indices"] == expected_selected
    assert manifest["selection"]["strategy"] == (
        "sequence_stratified_max_source_jpeg_byte_size"
    )
    assert manifest["selection"]["same_layout_consensus_allowed"] is True
    assert manifest["selection"]["different_layout_mixing_allowed"] is False
    assert manifest["production_status"]["production_eligible"] is False
    assert len(manifest["files"]) == count

    for entry in manifest["files"]:
        source_index = entry["source_index"]
        path = output / entry["path"]
        assert path.read_bytes() == upstream[source_index]
        assert entry["label"] == 0
        assert entry["source_state"] == "unmodified"
        assert entry["width"] == width and entry["height"] == height
        assert entry["focus_laplacian_variance"] > 0
        assert not Path(entry["path"]).is_absolute()
    assert str(tmp_path).lower() not in manifest_text.lower()
    assert "x-amz-" not in manifest_text.lower()
    assert (output / "ATTRIBUTION.md").is_file()

    inputs, loaded_manifest, _digest = bundle.load_reference_inputs(
        output, expected_count=count
    )
    assert len(inputs) == count
    assert loaded_manifest["selection"]["source_indices"] == expected_selected


def test_verified_existing_output_is_an_offline_idempotent_noop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pool_count, count = 7, 3
    width, height = 64, 44
    _configure_small_source(
        monkeypatch, pool_count=pool_count, width=width, height=height
    )
    archive_bytes, _ = _archive_bytes(pool_count, width, height)
    output = tmp_path / "mpi-reference"
    first = mpi.fetch_reference_set(
        output,
        count=count,
        archive_factory=lambda _spec: BytesIO(archive_bytes),
    )
    snapshot = {
        path.relative_to(output).as_posix(): path.read_bytes()
        for path in output.rglob("*")
        if path.is_file()
    }

    def exploding_factory(_spec: mpi.ArchiveSpec) -> BytesIO:
        raise AssertionError("idempotent verification must not touch the network")

    second = mpi.fetch_reference_set(
        output, count=count, archive_factory=exploding_factory
    )
    assert first.created is True
    assert second.created is False
    assert snapshot == {
        path.relative_to(output).as_posix(): path.read_bytes()
        for path in output.rglob("*")
        if path.is_file()
    }


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("missing", "member sequence does not match"),
        ("dimensions", "Unexpected dimensions"),
        ("duplicate", "duplicates another image"),
        ("crc", "CRC/decompression failed"),
    ],
)
def test_source_failures_publish_nothing_and_remove_atomic_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
    message: str,
) -> None:
    pool_count, count = 3, 3
    width, height = 68, 46
    _configure_small_source(
        monkeypatch, pool_count=pool_count, width=width, height=height
    )
    # One member per stratum makes index 1 selected for every content failure.
    archive_bytes, _ = _archive_bytes(
        pool_count,
        width,
        height,
        missing_index=1 if failure == "missing" else None,
        duplicate_index=1 if failure == "duplicate" else None,
        bad_dimensions_index=1 if failure == "dimensions" else None,
    )
    if failure == "crc":
        archive_bytes = _corrupt_member_compressed_bytes(
            archive_bytes, "original_aligned/train/good/0001.jpg"
        )
    output = tmp_path / "mpi-reference"

    with pytest.raises(mpi.FetchError, match=message):
        mpi.fetch_reference_set(
            output,
            count=count,
            archive_factory=lambda _spec: BytesIO(archive_bytes),
        )

    assert not output.exists()
    assert list(tmp_path.glob(".mpi-reference.tmp-*")) == []


def test_tampered_existing_output_is_refused_without_overwrite_or_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pool_count, count = 5, 3
    width, height = 62, 42
    _configure_small_source(
        monkeypatch, pool_count=pool_count, width=width, height=height
    )
    archive_bytes, _ = _archive_bytes(pool_count, width, height)
    output = tmp_path / "mpi-reference"
    mpi.fetch_reference_set(
        output,
        count=count,
        archive_factory=lambda _spec: BytesIO(archive_bytes),
    )
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    first_image = output / manifest["files"][0]["path"]
    first_image.write_bytes(b"tampered")

    def exploding_factory(_spec: mpi.ArchiveSpec) -> BytesIO:
        raise AssertionError("invalid existing output must fail before network access")

    with pytest.raises(mpi.FetchError, match="byte size changed"):
        mpi.fetch_reference_set(
            output, count=count, archive_factory=exploding_factory
        )
    assert first_image.read_bytes() == b"tampered"
