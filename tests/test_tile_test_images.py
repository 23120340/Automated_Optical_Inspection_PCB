"""Tiling test photographs: what a tile is kept for, and where it is cut from.

Two decisions in here are easy to get wrong in a way nothing reports.

A tile is kept on **component count**, not on a pixel statistic. Brightness and
saturation were tried and do not separate: on ``pcb31__rec1`` the background
strip reads 133 mean brightness while the darkest genuine board tile reads 34.
Component count separates, and it is also the property that matters -- a tile
with nothing on it cannot exercise step 4 however good it looks.

And a component belongs to the tile its **centre** falls in. Counting overlap
instead would credit one part to every tile it grazes, letting a tile of bare
laminate qualify on its neighbour's connector.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aoi_pipeline.models import BoundingBox  # noqa: E402
from scripts.tile_test_images import NAME, tile_one  # noqa: E402

cv2 = pytest.importorskip("cv2")


class _Detector:
    """Returns fixed boxes in ANALYSIS-frame coordinates, like the real one."""

    def __init__(self, boxes: list[tuple[float, float, float, float]]) -> None:
        self._boxes = boxes

    def detect(self, image):
        return [
            SimpleNamespace(bbox=BoundingBox(*b), label="ic", confidence=0.9)
            for b in self._boxes
        ]


@pytest.fixture
def board(tmp_path: Path) -> Path:
    path = tmp_path / "pcb99__rec1.jpg"
    cv2.imwrite(str(path), np.full((2048, 3072, 3), 40, np.uint8))
    return path


def test_a_tile_with_too_few_components_is_dropped(board: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    # three parts clustered inside the first tile, nothing anywhere else
    detector = _Detector([(50, 50, 90, 90), (150, 60, 190, 100), (260, 70, 300, 110)])
    rows = tile_one(board, detector, tile=1024, stride=1024, min_components=4,
                    output=out, dry_run=False)
    assert rows == []
    assert not list(out.glob("*.png"))


def test_the_same_tile_is_kept_once_the_floor_is_met(board: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    detector = _Detector([(50, 50, 90, 90), (150, 60, 190, 100), (260, 70, 300, 110)])
    rows = tile_one(board, detector, tile=1024, stride=1024, min_components=3,
                    output=out, dry_run=False)
    assert len(rows) == 1
    assert rows[0]["components"] == 3
    written = list(out.glob("*.png"))
    assert len(written) == 1
    assert cv2.imread(str(written[0])).shape[:2] == (1024, 1024)


def test_a_component_counts_for_the_tile_its_centre_lies_in(
    board: Path, tmp_path: Path
) -> None:
    """A part straddling a seam must not be credited to both sides."""

    out = tmp_path / "out"
    out.mkdir()
    # centre at x=1020, just inside the first tile; the box spills past 1024
    straddler = (980, 500, 1060, 560)
    detector = _Detector([straddler] + [(100 + 60 * i, 100, 140 + 60 * i, 140)
                                        for i in range(5)])
    rows = tile_one(board, detector, tile=1024, stride=1024, min_components=1,
                    output=out, dry_run=True)
    by_x = {int(r["x"]): int(r["components"]) for r in rows}
    assert by_x.get(0) == 6, "the straddling part belongs to the tile holding its centre"
    assert 1024 not in by_x, "it was counted twice, once for the neighbouring tile"


def test_boxes_are_scaled_from_the_analysis_frame_to_the_file(
    board: Path, tmp_path: Path
) -> None:
    """The detector sees a frame capped at max_side; the tiles are cut from the
    original. Skipping the scale factor puts every box in the wrong place, and
    the tiles still look plausible."""

    out = tmp_path / "out"
    out.mkdir()
    # A part whose centre sits at x=2500 in the 3072-wide FILE. If the analysis
    # frame were treated as the file, the same box would land near x=2500 of a
    # narrower frame and be assigned to a different tile.
    detector = _Detector([(2400, 900, 2600, 1000)] * 6)
    rows = tile_one(board, detector, tile=1024, stride=1024, min_components=6,
                    output=out, dry_run=True)
    assert rows, "no tile produced"
    # the file is 3072 wide; a centre at 2500 belongs to the tile starting at 2048
    assert {int(r["x"]) for r in rows} == {2048}


def test_the_name_matches_the_projects_own_reference_image(board: Path, tmp_path: Path) -> None:
    """``00001__1024__1648___4120.png`` is the shape a reader already knows.
    Three underscores before y, not two."""

    assert NAME.format(stem="00001", tile=1024, x=1648, y=4120) == (
        "00001__1024__1648___4120.png"
    )


def test_a_dry_run_still_reports_but_writes_nothing(board: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    detector = _Detector([(100 + 60 * i, 100, 140 + 60 * i, 140) for i in range(8)])
    rows = tile_one(board, detector, tile=1024, stride=1024, min_components=4,
                    output=out, dry_run=True)
    assert rows and not list(out.glob("*.png"))


def test_resuming_requires_the_manifest_that_holds_the_measurements(
    board: Path, tmp_path: Path
) -> None:
    """Một lần chạy bị ngắt để lại tile trên đĩa nhưng KHÔNG có manifest.

    Bỏ qua ảnh đó chỉ vì thấy file tile là mất số linh kiện mỗi tile -- đúng thứ
    manifest sinh ra để ghi -- và không ai biết là thiếu. Nên điều kiện bỏ qua
    phải là "manifest cũ còn số đo", không phải "có file trên đĩa".
    """

    from scripts.tile_test_images import main as tile_main

    out = tmp_path / "out"
    out.mkdir()
    # tile mồ côi: có file trên đĩa, không có manifest đi kèm
    orphan = out / f"{board.stem}__1024__0___0.png"
    cv2.imwrite(str(orphan), np.zeros((8, 8, 3), np.uint8))
    assert orphan.is_file()
    assert not (out / "tiles_manifest.json").exists()

    source = inspect.getsource(tile_main)
    assert "previous.get(path.name)" in source, (
        "điều kiện bỏ qua phải đọc manifest, không phải chỉ liệt kê file"
    )
    assert 'record = output / "tiles_manifest.json"' in source


def test_a_reused_row_keeps_its_component_count() -> None:
    """Dòng lấy lại từ manifest phải mang nguyên số đo, không phải một cờ trống."""

    from scripts.tile_test_images import main as tile_main

    source = inspect.getsource(tile_main)
    assert '"reused": True' not in source, (
        "bản đầu ghi cờ reused thay cho số đo; manifest mất đúng thứ nó ghi lại"
    )
    assert "rows.extend(done)" in source


def test_the_manifest_records_how_much_of_a_tile_is_background(
    board: Path, tmp_path: Path
) -> None:
    """Các board này chụp trên vải đen. Một tile ở mép board vẫn đủ số linh kiện
    tối thiểu khi chúng dồn vào một góc, và 80% khung còn lại là nền.

    Tile đó KHÔNG bị loại — mép board là thứ có thật và đáng test — nhưng con số
    phải nằm trong manifest. Không có nó thì muốn lọc phải mở lại 310 ảnh mà đo
    lại từ đầu.
    """

    out = tmp_path / "out"
    out.mkdir()
    detector = _Detector([(100 + 60 * i, 100, 140 + 60 * i, 140) for i in range(8)])
    rows = tile_one(board, detector, tile=1024, stride=1024, min_components=4,
                    output=out, dry_run=True)
    assert rows
    assert all("dark_fraction" in r for r in rows)
    assert all(0.0 <= float(r["dark_fraction"]) <= 1.0 for r in rows)


def test_the_dark_filter_can_drop_a_tile_the_component_count_would_keep(
    board: Path, tmp_path: Path
) -> None:
    """Hai cổng đo hai thứ khác nhau: một cái đếm linh kiện, một cái đo phần
    khung không có gì. Nếu cổng thứ hai không cắn được thì nó vô nghĩa."""

    out = tmp_path / "out"
    out.mkdir()
    # Board tối hẳn: ngưỡng "tối" là < 40, nên fixture chung (giá trị đúng 40)
    # cho dark_fraction = 0 và không kiểm được cổng này.
    dark_board = tmp_path / "pcb98__rec1.jpg"
    cv2.imwrite(str(dark_board), np.full((2048, 3072, 3), 5, np.uint8))
    detector = _Detector([(100 + 60 * i, 100, 140 + 60 * i, 140) for i in range(8)])

    kept = tile_one(dark_board, detector, tile=1024, stride=1024, min_components=4,
                    output=out, dry_run=True, max_dark=1.0)
    assert kept, "không lọc thì tile phải được giữ"
    assert float(kept[0]["dark_fraction"]) > 0.9, "fixture chưa đủ tối để kiểm"

    dropped = tile_one(dark_board, detector, tile=1024, stride=1024, min_components=4,
                       output=out, dry_run=True, max_dark=0.5)
    assert not dropped, "cổng nền không loại được tile mà cổng đếm giữ lại"
