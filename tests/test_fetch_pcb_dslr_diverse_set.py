from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import sys
from zipfile import ZIP_DEFLATED, ZipFile

import cv2
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import fetch_pcb_dslr_diverse_set as fetcher  # noqa: E402


def _zip_bytes(
    board_numbers: list[int],
    width: int,
    height: int,
    *,
    empty_annotation: bool = False,
    invalid_annotation_board: int | None = None,
) -> bytes:
    handle = BytesIO()
    with ZipFile(handle, "w", compression=ZIP_DEFLATED) as archive:
        for board_number in board_numbers:
            image = np.full((height, width, 3), (30, 90 + board_number, 40), np.uint8)
            cv2.rectangle(image, (5, 4), (width - 6, height - 5), (80, 170, 120), -1)
            mask = np.zeros((height, width), np.uint8)
            cv2.rectangle(mask, (5, 4), (width - 6, height - 5), 255, -1)
            ok_image, encoded_image = cv2.imencode(".jpg", image)
            ok_mask, encoded_mask = cv2.imencode(".png", mask)
            assert ok_image and ok_mask
            prefix = f"pcb{board_number}/rec1"
            archive.writestr(f"{prefix}.jpg", encoded_image.tobytes())
            archive.writestr(f"{prefix}-mask.png", encoded_mask.tobytes())
            archive.writestr(
                f"{prefix}-annot.txt",
                (
                    "broken\n"
                    if board_number == invalid_annotation_board
                    else ""
                    if empty_annotation
                    else f"{width / 2} {height / 2} 8 6 -12.5 CHIP TEXT\n"
                ),
            )
    return handle.getvalue()


def test_remote_range_reader_is_seekable_and_block_cached() -> None:
    payload = bytes(range(251)) * 11
    requests: list[tuple[int, int]] = []

    def get_range(start: int, end: int) -> bytes:
        requests.append((start, end))
        return payload[start : end + 1]

    reader = fetcher.RemoteRangeReader(
        "memory://archive",
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
    reader.seek(-12, 2)
    assert reader.read() == payload[-12:]


def test_select_members_requires_complete_triplet() -> None:
    payload = _zip_bytes([1], 40, 30)
    with ZipFile(BytesIO(payload)) as archive:
        selected = fetcher.select_members(archive, [1])
        assert [item.filename for item in selected[1]] == [
            "pcb1/rec1.jpg",
            "pcb1/rec1-mask.png",
            "pcb1/rec1-annot.txt",
        ]
        with pytest.raises(fetcher.FetchError, match="missing required member"):
            fetcher.select_members(archive, [2])


def test_empty_ic_annotation_is_a_valid_zero_ic_board() -> None:
    payload = _zip_bytes([21], 40, 30, empty_annotation=True)
    with ZipFile(BytesIO(payload)) as archive:
        selected = fetcher.select_members(archive, [21])
        assert selected[21][2].file_size == 0


def test_fetch_builds_portable_manifest_and_preserves_guidance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    width, height = 64, 48
    monkeypatch.setattr(fetcher, "EXPECTED_WIDTH", width)
    monkeypatch.setattr(fetcher, "EXPECTED_HEIGHT", height)
    archive_bytes = _zip_bytes([1, 2], width, height)
    output = tmp_path / "refs"
    output.mkdir()
    guidance = output / "NHUNG_VIEC_BAN_CAN_LAM.md"
    guidance.write_text("keep me\n", encoding="utf-8")

    result = fetcher.fetch_reference_set(
        output,
        count=2,
        archive_factory=lambda _spec: BytesIO(archive_bytes),
    )
    assert result == output.resolve()
    assert guidance.read_text(encoding="utf-8") == "keep me\n"
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["selection"]["distinct_layout_count"] == 2
    assert manifest["selection"]["same_sku_consensus_allowed"] is False
    assert manifest["usage_restriction"] == fetcher.USAGE_RESTRICTION
    assert [item["board_id"] for item in manifest["files"]] == [
        "pcb_dslr_001",
        "pcb_dslr_002",
    ]
    assert all(not Path(item["image_path"]).is_absolute() for item in manifest["files"])
    assert (output / "boards" / "pcb_dslr_001" / "rec1.jpg").is_file()
    assert (output / "ATTRIBUTION.md").is_file()

    with pytest.raises(fetcher.FetchError, match="refusing to overwrite"):
        fetcher.fetch_reference_set(
            output,
            count=2,
            archive_factory=lambda _spec: BytesIO(archive_bytes),
        )


def test_failed_later_board_keeps_verified_partial_for_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    width, height = 64, 48
    monkeypatch.setattr(fetcher, "EXPECTED_WIDTH", width)
    monkeypatch.setattr(fetcher, "EXPECTED_HEIGHT", height)
    output = tmp_path / "refs"
    bad = _zip_bytes(
        [1, 2], width, height, invalid_annotation_board=2
    )
    with pytest.raises(fetcher.FetchError, match="Invalid annotation"):
        fetcher.fetch_reference_set(
            output,
            count=2,
            archive_factory=lambda _spec: BytesIO(bad),
        )
    partial = tmp_path / ".refs.partial" / "boards"
    assert (partial / "pcb_dslr_001" / "source_record.json").is_file()
    assert not (partial / "pcb_dslr_002").exists()

    good = _zip_bytes([1, 2], width, height)
    fetcher.fetch_reference_set(
        output,
        count=2,
        archive_factory=lambda _spec: BytesIO(good),
    )
    assert not (tmp_path / ".refs.partial").exists()
    assert (output / "boards" / "pcb_dslr_001" / "source_record.json").is_file()
