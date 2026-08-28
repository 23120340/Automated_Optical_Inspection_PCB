"""The test-image collector must not quietly admit the wrong kind of picture.

"High quality" for this project is a pixel-scale claim, not an aesthetic one,
and the two ways of getting it wrong have both already cost the project a
measurement: SolDef_AI's macro photography returned zero boxes on a real board,
and a 640 px board export leaves a chip part on ten pixels.

The gate here is therefore a number, and these tests hold two things about it:
it is applied to every candidate, and it is not the *only* filter -- a binary
board mask sits in the same folder as the photograph it belongs to, is the same
resolution, and would sail through a megapixel test on its own.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.fetch_test_board_images import (  # noqa: E402
    MIN_MEGAPIXELS,
    gather_local,
)
import scripts.fetch_test_board_images as collector  # noqa: E402


def _write(path: Path, size: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (20, 90, 40)).save(path)


@pytest.fixture
def sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "repo"
    boards = root / "boards"
    _write(boards / "b1" / "rec1.jpg", (4000, 3000))     # 12.0 MP  giữ
    _write(boards / "b2" / "rec1.jpg", (1200, 900))      #  1.1 MP  loại
    _write(boards / "b1" / "rec1-mask.png", (4000, 3000))  # đủ MP nhưng là mask
    _write(boards / "b2" / "rec1-annot.png", (4000, 3000))
    monkeypatch.setattr(collector, "PROJECT_ROOT", root)
    monkeypatch.setattr(collector, "LOCAL_SOURCES", {"boards": Path("boards")})
    return root


def test_a_low_resolution_photo_is_refused(sources: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    gather_local(out, min_megapixels=MIN_MEGAPIXELS)
    kept = {p.name for p in (out / "boards").iterdir()}
    assert "b1__rec1.jpg" in kept
    assert not any(name.startswith("b2__rec1.jpg") for name in kept), (
        "một ảnh 1.1 MP đã lọt qua cổng"
    )


def test_a_mask_at_full_resolution_is_still_refused(sources: Path, tmp_path: Path) -> None:
    """Cổng đo ĐỘ PHÂN GIẢI, không đo "đây có phải ảnh chụp không". Mask nhị
    phân nằm cùng thư mục với ảnh nó thuộc về và cùng kích thước, nên chỉ đo
    megapixel là copy luôn cả nó -- đã xảy ra thật: 30 file mask lọt vào thư
    mục test trước khi bộ lọc theo tên được thêm."""

    out = tmp_path / "out"
    gather_local(out, min_megapixels=MIN_MEGAPIXELS)
    kept = {p.name for p in (out / "boards").iterdir()}
    assert not [n for n in kept if "mask" in n or "annot" in n], (
        f"mask/nhãn lọt vào thư mục ảnh test: {sorted(kept)}"
    )


def test_the_flat_name_keeps_boards_apart(sources: Path, tmp_path: Path) -> None:
    """Nguồn DSLR lưu MỌI ảnh là ``rec1.jpg`` trong thư mục riêng của từng board.
    Gộp phẳng mà không mang tên thư mục cha theo thì 40 board còn đúng 1 file."""

    out = tmp_path / "out"
    gather_local(out, min_megapixels=0.0)
    kept = sorted(p.name for p in (out / "boards").iterdir())
    assert "b1__rec1.jpg" in kept and "b2__rec1.jpg" in kept
    assert len({n for n in kept if n.endswith(".jpg")}) == 2


def test_the_manifest_records_what_was_measured(sources: Path, tmp_path: Path) -> None:
    """Một thư mục ảnh không tự nói được vì sao từng ảnh có mặt ở đó."""

    out = tmp_path / "out"
    gather_local(out, min_megapixels=MIN_MEGAPIXELS)
    rows = json.loads((out / "local_sources_manifest.json").read_text(encoding="utf-8"))
    assert rows and all(
        {"file", "source", "width", "height", "megapixels"} <= set(row) for row in rows
    )
    assert all(row["megapixels"] >= MIN_MEGAPIXELS for row in rows)


def test_a_dry_run_writes_nothing(sources: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    gather_local(out, dry_run=True)
    assert not out.exists() or not any(out.rglob("*.jpg"))


def test_the_threshold_is_documented_where_it_is_defined() -> None:
    """Ngưỡng này quyết định ảnh nào vào bộ test, nên nó phải mang theo lý do —
    con số trần không cho ai sửa nó một cách có căn cứ."""

    source = Path(collector.__file__).read_text(encoding="utf-8")
    head = source.split("MIN_MEGAPIXELS")[0]
    assert "46 um/px" in head or "46 um/px" in source
    assert MIN_MEGAPIXELS >= 6.0
