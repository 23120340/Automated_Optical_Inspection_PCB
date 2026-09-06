"""Kho lịch sử kiểm tra phải ghi được, đọc lại đúng, và không mất bằng chứng.

Bốn hợp đồng dưới đây là **yêu cầu nghiệp vụ** trong
``ke_hoach_so_hoa_du_lieu_va_truy_xuat_aoi.md``, không phải chi tiết kỹ thuật:
gửi lại một sự kiện không tạo bản ghi mới, toạ độ luôn đi kèm hệ quy chiếu, ảnh
còn được tham chiếu thì không xoá, và một mặt đạt không làm cả PCB đạt.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from aoi_pipeline.golden.inspector import InspectionRun
from aoi_pipeline.imaging.alignment import StrictAlignmentResult
from aoi_pipeline.storage import DefectRecord, InspectionStore


def _run(side: str = "top", status: str = "ng", *, enforced: bool = True) -> InspectionRun:
    return InspectionRun(
        status=status,
        reason="",
        alignment=StrictAlignmentResult(
            status="ok", image=None, transform=None, residual_px=None,
            matched_anchors=4, inliers=4, inlier_ratio=1.0, scale=1.0,
            rotation_deg=0.0, canvas_overlap_ratio=1.0, valid_mask=None, reason="",
        ),
        slots=(), extras=(),
        recipe_schema_version="1", recipe_sha256="a" * 64, golden_sha256="b" * 64,
        model_identifiers={"component_detector": "det.onnx:sha"},
        runtime_detector="Mock", runtime_detector_identifier="det.onnx:sha",
        production_gates_enforced=enforced,
        board_id="B1", side=side,
    )


def _png(path: Path, payload: bytes = b"anh-loi") -> Path:
    path.write_bytes(payload)
    return path


def test_an_inspection_survives_a_restart_with_its_defects_and_images(tmp_path: Path) -> None:
    """Hợp đồng nền: ghi xong, đóng, mở lại, đọc ra đúng cái đã ghi.

    Kể cả đường dẫn ảnh — yêu cầu tối thiểu của dây chuyền là *vị trí lỗi kèm
    ảnh lỗi*, nên một bản ghi đọc lại được mà không mở được ảnh là chưa đạt.
    """

    crop = _png(tmp_path / "loi.png")
    with InspectionStore(tmp_path / "kho") as store:
        board = store.ensure_board()
        inspection_id = store.save_inspection(
            _run(), board_id=board, event_id="EV-1",
            defects=[DefectRecord(
                slot_id="U12", status="ng", reason="thieu_linh_kien",
                bbox=(10.0, 20.0, 40.0, 60.0), coordinate_space="golden",
                image_path=crop,
            )],
        )

    with InspectionStore(tmp_path / "kho") as store:
        stored = store.load_inspection(inspection_id)
        assert stored.side == "top"
        assert stored.board_id == board
        assert len(stored.defects) == 1
        defect = stored.defects[0]
        assert (defect["x1"], defect["y1"], defect["x2"], defect["y2"]) == (10, 20, 40, 60)
        assert defect["coordinate_space"] == "golden"
        assert store.image_file(defect["image_sha256"]).read_bytes() == b"anh-loi"
        # Toàn bộ bản ghi gốc giữ nguyên, không chỉ vài cột được trích ra.
        assert stored.run["recipe_sha256"] == "a" * 64


def test_resending_the_same_event_does_not_create_a_second_record(tmp_path: Path) -> None:
    """§8.2: gửi lại cùng một sự kiện trả về bản ghi cũ.

    Đây là ca hay gặp nhất khi mạng chập chờn: bên gửi không nhận được xác nhận
    nên gửi lại. Tạo bản ghi thứ hai làm hỏng mọi thống kê đếm theo lần kiểm tra.
    """

    with InspectionStore(tmp_path / "kho") as store:
        board = store.ensure_board()
        first = store.save_inspection(_run(), board_id=board, event_id="EV-1")
        second = store.save_inspection(_run(), board_id=board, event_id="EV-1")
        assert first == second
        assert len(store.inspections_for_board(board)) == 1


def test_a_defect_position_without_a_coordinate_space_is_refused(tmp_path: Path) -> None:
    """§6.2: toạ độ không kèm hệ quy chiếu là con số không đọc lại được.

    Ảnh thô, ảnh đã nắn và ảnh Golden không cùng một hệ; lưu bbox mà quên hệ thì
    sau này không ai biết nó thuộc ảnh nào.
    """

    with pytest.raises(ValueError, match="hệ quy chiếu"):
        DefectRecord(slot_id="U1", status="ng", bbox=(1.0, 2.0, 3.0, 4.0))


def test_an_image_two_inspections_share_is_not_deleted_with_one_of_them(
    tmp_path: Path,
) -> None:
    """§8.2: đếm tham chiếu trước khi thu hồi, không xoá theo tuổi.

    Ca thật: ảnh gốc dùng chung hoặc ảnh Golden được nhiều lần phân tích trỏ tới.
    Xoá theo tuổi là mất bằng chứng của những lần vẫn còn hiệu lực.
    """

    crop = _png(tmp_path / "chung.png")
    with InspectionStore(tmp_path / "kho") as store:
        board = store.ensure_board()
        keep = DefectRecord(slot_id="U1", status="ng", image_path=crop)
        first = store.save_inspection(_run(), board_id=board, event_id="EV-1",
                                      defects=[keep])
        store.save_inspection(_run(side="bottom"), board_id=board, event_id="EV-2",
                              defects=[keep])
        sha = store.load_inspection(first).defects[0]["image_sha256"]

        assert store.delete_unreferenced_images() == ()
        assert store.image_file(sha).exists()

        store.connection.execute("DELETE FROM inspection WHERE inspection_id = ?",
                                 (first,))
        store.connection.commit()
        assert store.delete_unreferenced_images() == (), (
            "vẫn còn lần kiểm tra thứ hai trỏ tới ảnh này"
        )
        assert store.image_file(sha).exists()


def test_one_passing_side_does_not_clear_the_whole_board(tmp_path: Path) -> None:
    """§4.4, hỏi trên DỮ LIỆU ĐÃ LƯU chứ không trên đối tượng trong bộ nhớ.

    Sau khi khởi động lại thì các ``InspectionRun`` không còn, nhưng câu hỏi
    "PCB này đủ điều kiện chưa" vẫn phải trả lời được.
    """

    with InspectionStore(tmp_path / "kho") as store:
        board = store.ensure_board()
        store.save_inspection(_run(side="top", status="pass"), board_id=board,
                              event_id="EV-1")
        assert store.sides_still_missing(board) == ("bottom",)
        store.save_inspection(_run(side="bottom", status="pass"), board_id=board,
                              event_id="EV-2")
        assert store.sides_still_missing(board) == ()


def test_a_relaxed_run_never_satisfies_a_side(tmp_path: Path) -> None:
    """§9.3: một lần PASS ở chế độ chạy thử không đủ tư cách làm căn cứ."""

    with InspectionStore(tmp_path / "kho") as store:
        board = store.ensure_board()
        store.save_inspection(_run(side="top", status="pass", enforced=False),
                              board_id=board, event_id="EV-1")
        assert store.sides_still_missing(board, required=("top",)) == ("top",)


def test_an_internal_id_is_never_mistaken_for_a_real_serial(tmp_path: Path) -> None:
    """§4.2b / §13: chưa đọc được serial thì phải NÓI RA, không bịa một chuỗi.

    Gắn serial thật sau vẫn giữ nguyên ``board_id`` — đổi khoá chính là cách
    chắc chắn nhất để mất liên kết với lịch sử đã lưu.
    """

    with InspectionStore(tmp_path / "kho") as store:
        board = store.ensure_board()
        row = store.connection.execute(
            "SELECT serial, identity_source FROM board WHERE board_id = ?", (board,)
        ).fetchone()
        assert row["serial"] is None
        assert row["identity_source"] == "internal"

        store.save_inspection(_run(), board_id=board, event_id="EV-1")
        store.attach_serial(board, "SN-0001")
        assert store.inspections_for_board(board), "lịch sử phải còn nguyên"

        with pytest.raises(ValueError, match="đổi danh tính"):
            store.attach_serial(board, "SN-KHAC")


def test_foreign_keys_are_actually_enforced(tmp_path: Path) -> None:
    """``PRAGMA foreign_keys`` mặc định TẮT trong SQLite và chỉ theo từng kết nối.

    Quên bật thì mọi ``REFERENCES`` chỉ là chú thích, và ràng buộc "không xoá ảnh
    còn được tham chiếu" thành lời hứa suông. Chốt bằng test chứ không bằng niềm
    tin.
    """

    with InspectionStore(tmp_path / "kho") as store:
        with pytest.raises(sqlite3.IntegrityError):
            store.connection.execute(
                "INSERT INTO inspection (inspection_id, board_id, side, event_id,"
                " status, started_at, received_at, production_gates_enforced,"
                " recipe_sha256, golden_sha256, runtime_detector_identifier,"
                " coordinate_space, run_json)"
                " VALUES ('X','KHONG-CO','top','E',"
                "'pass','t','t',1,'a','b','c','d','{}')"
            )


# ------------------------------------------- phiên kiểm: giữ danh tính khi lật

def test_a_station_can_only_have_one_open_session(tmp_path: Path) -> None:
    """§4.3: mỗi trạm một phiên đang mở.

    Ép bằng chỉ mục một phần trong lược đồ, không bằng kiểm tra ở tầng ứng dụng
    — hai tiến trình cùng mở thì tầng ứng dụng thua, cơ sở dữ liệu thì không.
    """

    with InspectionStore(tmp_path / "kho") as store:
        a, b = store.ensure_board(), store.ensure_board()
        first = store.open_session(board_id=a, station="T1")
        with pytest.raises(RuntimeError, match="đang có phiên mở"):
            store.open_session(board_id=b, station="T1")
        # Trạm khác thì không ảnh hưởng.
        store.open_session(board_id=b, station="T2")
        store.cancel_session(first, reason="đổi bo")
        assert store.open_session(board_id=b, station="T1")


def test_closing_a_session_with_a_side_missing_needs_a_reason(tmp_path: Path) -> None:
    """Bỏ dở IM LẶNG là cách một tấm bo chưa kiểm đủ đi tiếp mà không ai biết."""

    with InspectionStore(tmp_path / "kho") as store:
        board = store.ensure_board()
        session = store.open_session(board_id=board, station="T1")
        store.save_inspection(_run(side="top", status="pass"), board_id=board,
                              event_id="EV-1", session_id=session)

        assert store.session_sides_missing(session) == ("bottom",)
        with pytest.raises(ValueError, match="nêu lý do"):
            store.close_session(session)
        assert store.close_session(session, reason="bo cong, dừng kiểm") == ("bottom",)


def test_a_finished_session_closes_without_a_reason(tmp_path: Path) -> None:
    with InspectionStore(tmp_path / "kho") as store:
        board = store.ensure_board()
        session = store.open_session(board_id=board, station="T1")
        for index, side in enumerate(("top", "bottom")):
            store.save_inspection(_run(side=side, status="pass"), board_id=board,
                                  event_id=f"EV-{index}", session_id=session)
        assert store.session_sides_missing(session) == ()
        assert store.close_session(session) == ()


def test_a_one_sided_board_is_declared_not_assumed(tmp_path: Path) -> None:
    """``required_sides`` ghi vào phiên, không lấy mặc định lúc đọc.

    Một bo một mặt và một bo hai mặt phải phân biệt được sau nhiều tháng.
    """

    with InspectionStore(tmp_path / "kho") as store:
        board = store.ensure_board()
        session = store.open_session(board_id=board, station="T1",
                                     required_sides=("top",))
        store.save_inspection(_run(side="top", status="pass"), board_id=board,
                              event_id="EV-1", session_id=session)
        assert store.session_sides_missing(session) == ()


def test_a_result_arriving_after_cancel_is_kept_but_flagged(tmp_path: Path) -> None:
    """§4.3: ghi vào ĐÚNG phiên sinh ra nó, không hồi sinh, không gán sang phiên khác.

    Ca thật: máy gửi kết quả chậm, người vận hành đã huỷ phiên và bắt đầu bo
    khác. Vứt kết quả đi là mất dữ liệu; gán sang phiên đang chạy là tráo danh
    tính bo. Cả hai đều sai, nên phải giữ lại và đánh dấu.
    """

    with InspectionStore(tmp_path / "kho") as store:
        board = store.ensure_board()
        session = store.open_session(board_id=board, station="T1")
        store.cancel_session(session, reason="người vận hành huỷ")

        late = store.save_inspection(_run(side="top", status="pass"), board_id=board,
                                     event_id="EV-late", session_id=session)
        row = store.connection.execute(
            "SELECT session_id, arrived_after_close FROM inspection"
            " WHERE inspection_id = ?", (late,)
        ).fetchone()
        assert row["session_id"] == session, "phải nằm ở đúng phiên sinh ra nó"
        assert row["arrived_after_close"] == 1

        # Phiên vẫn huỷ, và kết quả về muộn KHÔNG được tính là đã kiểm mặt đó.
        status = store.connection.execute(
            "SELECT status FROM scan_session WHERE session_id = ?", (session,)
        ).fetchone()["status"]
        assert status == "cancelled"
        assert "top" in store.session_sides_missing(session)


def test_an_inspection_cannot_be_filed_under_another_boards_session(tmp_path: Path) -> None:
    """Gán chéo phiên là tráo danh tính bo — phải chặn ngay, không ghi rồi sửa."""

    with InspectionStore(tmp_path / "kho") as store:
        a, b = store.ensure_board(), store.ensure_board()
        session = store.open_session(board_id=a, station="T1")
        with pytest.raises(ValueError, match="tráo danh tính"):
            store.save_inspection(_run(), board_id=b, event_id="EV-1",
                                  session_id=session)
