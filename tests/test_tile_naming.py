"""Hai mặt của một bo đều có ``1.jpg`` — tile của chúng không được đè nhau.

Đây là kiểu hỏng không kêu: ``front_side/1.jpg`` và ``back_side/1.jpg`` cùng cho
stem ``1``, nên tile của mặt cắt sau ghi đè tile của mặt cắt trước, manifest vẫn
đủ số dòng, và chỉ khi ai đó mở ảnh ra khoanh mới thấy nhãn không khớp ảnh.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "tile_test_images",
    Path(__file__).resolve().parents[1] / "scripts" / "tile_test_images.py",
)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
source_stems = _MODULE.source_stems


def test_flat_sources_keep_their_plain_name() -> None:
    """Không trùng thì giữ nguyên tên cũ.

    Bộ tile PCB-DSLR đã dựng theo tên này và bộ khoanh nhãn vòng 2 trỏ vào nó;
    đổi tên khi không cần là làm đứt liên kết với công đã làm.
    """

    root = Path("/anh")
    sources = [root / "a" / "pcb_dslr_001__rec1.jpg", root / "b" / "pcb31__rec1.jpg"]
    assert source_stems(sources, root) == {
        sources[0]: "pcb_dslr_001__rec1",
        sources[1]: "pcb31__rec1",
    }


def test_two_sides_with_the_same_file_name_get_different_stems() -> None:
    root = Path("/bo")
    front = root / "front_side" / "1.jpg"
    back = root / "back_side" / "1.jpg"
    stems = source_stems([front, back], root)
    assert stems[front] != stems[back]
    assert stems[front] == "front_side__1"


def test_one_collision_renames_every_source_not_just_the_clashing_pair() -> None:
    """Đặt tên nửa kiểu này nửa kiểu kia thì sau không ai đoán được tên file."""

    root = Path("/bo")
    sources = [
        root / "front_side" / "1.jpg",
        root / "back_side" / "1.jpg",
        root / "front_side" / "9.jpg",
    ]
    stems = source_stems(sources, root)
    assert set(stems.values()) == {"front_side__1", "back_side__1", "front_side__9"}


def test_every_source_still_gets_exactly_one_name() -> None:
    root = Path("/bo")
    sources = [root / side / f"{i}.jpg" for side in ("front_side", "back_side")
               for i in range(1, 14)]
    stems = source_stems(sources, root)
    assert len(stems) == len(sources)
    assert len(set(stems.values())) == len(sources)


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_nested_folders_are_flattened_into_the_name(depth: int) -> None:
    root = Path("/bo")
    a = root.joinpath(*[f"d{i}" for i in range(depth)], "1.jpg")
    b = root.joinpath(*[f"e{i}" for i in range(depth)], "1.jpg")
    stems = source_stems([a, b], root)
    assert "/" not in stems[a] and "\\" not in stems[a]
    assert stems[a] != stems[b]
