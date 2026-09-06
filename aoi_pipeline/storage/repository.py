"""Ghi và đọc lại một lần kiểm tra, kèm ảnh lỗi.

Yêu cầu tối thiểu của dây chuyền (chốt 06/09): **lưu vị trí lỗi kèm ảnh lỗi**.
Nên ``save_inspection`` nhận đúng hai thứ đó cùng lúc — tách làm hai lời gọi thì
sẽ có lúc lưu được vị trí mà mất ảnh, và ngược lại.

Không dùng ORM. Mô hình dữ liệu nhỏ, câu hỏi thì cụ thể, và một lớp trừu tượng
nữa ở đây chỉ làm khó việc đọc ra chính xác cái gì chạm vào đĩa.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..models import utc_now_iso
from .schema import connect, migrate

__all__ = ["DefectRecord", "InspectionStore", "StoredInspection"]

_MEDIA_BY_SUFFIX = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}


@dataclass(frozen=True, slots=True)
class DefectRecord:
    """Một vị trí lỗi, và ảnh của chính nó nếu có.

    ``bbox`` và ``coordinate_space`` đi cùng nhau hoặc cùng vắng. Một toạ độ
    không kèm hệ quy chiếu là con số không đọc được (kế hoạch §6.2): ảnh thô,
    ảnh đã nắn và ảnh Golden không cùng một hệ.
    """

    slot_id: str
    status: str
    reason: str = ""
    bbox: tuple[float, float, float, float] | None = None
    coordinate_space: str | None = None
    image_path: str | Path | None = None

    def __post_init__(self) -> None:
        if (self.bbox is None) != (self.coordinate_space is None):
            raise ValueError(
                "bbox và coordinate_space phải cùng có hoặc cùng không: toạ độ "
                "không kèm hệ quy chiếu thì không đọc lại được"
            )


@dataclass(frozen=True, slots=True)
class StoredInspection:
    inspection_id: str
    board_id: str
    side: str
    status: str
    started_at: str
    production_gates_enforced: bool
    production_gate_findings: tuple[str, ...]
    run: Mapping[str, Any]
    defects: tuple[Mapping[str, Any], ...]


class InspectionStore:
    """Kho lịch sử kiểm tra: một file SQLite + một thư mục ảnh."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.images_root = self.root / "images"
        self.images_root.mkdir(parents=True, exist_ok=True)
        self.connection = connect(self.root / "aoi_history.sqlite3")
        migrate(self.connection)

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> InspectionStore:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------ bo

    def ensure_board(
        self,
        *,
        serial: str | None = None,
        board_type: str | None = None,
        revision: str | None = None,
    ) -> str:
        """Trả về ``board_id``; tạo mới nếu chưa có.

        Chưa đọc được serial thì sinh ID nội bộ và ghi rõ ``identity_source =
        'internal'``. Không bịa một chuỗi trông giống serial: sau này không ai
        phân biệt được ID tự sinh với serial thật nữa (§4.2b, và §13 "không tự
        bịa serial").
        """

        if serial:
            row = self.connection.execute(
                "SELECT board_id FROM board WHERE serial = ?", (serial,)
            ).fetchone()
            if row is not None:
                return str(row["board_id"])
            board_id = f"S-{serial}"
            source = "serial"
        else:
            board_id = f"INT-{uuid.uuid4().hex[:12]}"
            source = "internal"
        self.connection.execute(
            "INSERT INTO board (board_id, serial, identity_source, board_type,"
            " revision, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (board_id, serial, source, board_type, revision, utc_now_iso()),
        )
        self.connection.commit()
        return board_id

    def attach_serial(self, board_id: str, serial: str) -> None:
        """Gắn serial thật cho một bo trước đó chỉ có ID nội bộ.

        Giữ nguyên ``board_id`` để mọi inspection đã lưu không mất liên kết —
        đổi khoá chính ở đây là cách chắc chắn nhất để mất lịch sử.
        """

        row = self.connection.execute(
            "SELECT identity_source, serial FROM board WHERE board_id = ?", (board_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"không có bo {board_id}")
        if row["serial"] is not None and row["serial"] != serial:
            raise ValueError(
                f"bo {board_id} đã mang serial {row['serial']}; đổi serial là đổi "
                "danh tính, phải đi qua quy trình đối soát chứ không sửa tại chỗ"
            )
        self.connection.execute(
            "UPDATE board SET serial = ?, identity_source = 'serial' WHERE board_id = ?",
            (serial, board_id),
        )
        self.connection.commit()

    # ------------------------------------------------------------ ghi/đọc

    def save_inspection(
        self,
        run: Any,
        *,
        board_id: str,
        event_id: str,
        defects: Sequence[DefectRecord] = (),
        session_id: str | None = None,
    ) -> str:
        """Lưu một lần chạy cùng vị trí lỗi và ảnh lỗi. Trả về ``inspection_id``.

        **Bất biến (idempotent) theo ``event_id``.** Gửi lại cùng một sự kiện trả
        về đúng bản ghi cũ chứ không tạo bản mới (§8.2). Người gọi quyết định
        ``event_id`` vì chỉ họ biết hai lần gọi có phải cùng một sự kiện sản xuất
        hay không — sinh ngẫu nhiên ở đây là tự tay bỏ mất chống trùng.

        Ảnh được chép vào kho **trước** khi ghi hàng lỗi, và cả hai nằm trong một
        transaction: không bao giờ có hàng lỗi trỏ tới ảnh chưa tồn tại.

        Kết quả về **sau khi phiên đã đóng hoặc huỷ** vẫn được ghi vào đúng phiên
        sinh ra nó, đánh dấu ``arrived_after_close`` (§4.3). Không hồi sinh phiên
        và tuyệt đối không gán sang phiên đang chạy — cả hai đều làm tráo danh
        tính bo. Bản ghi đó dùng để đối soát, không dùng làm căn cứ chuyển công
        đoạn.
        """

        existing = self.connection.execute(
            "SELECT inspection_id FROM inspection WHERE event_id = ?", (event_id,)
        ).fetchone()
        if existing is not None:
            return str(existing["inspection_id"])

        late = 0
        if session_id is not None:
            row = self.connection.execute(
                "SELECT status, board_id FROM scan_session WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"không có phiên {session_id}")
            if str(row["board_id"]) != board_id:
                raise ValueError(
                    f"phiên {session_id} thuộc bo {row['board_id']}, không phải "
                    f"{board_id}; gán chéo là tráo danh tính bo"
                )
            late = 0 if row["status"] == "open" else 1

        payload = run.to_dict()
        inspection_id = f"INS-{uuid.uuid4().hex[:12]}"
        stored_images = [
            None if item.image_path is None else self._store_image(item.image_path)
            for item in defects
        ]
        try:
            with self.connection:
                self.connection.execute(
                    "INSERT INTO inspection (inspection_id, board_id, side, event_id,"
                    " status, reason, started_at, received_at,"
                    " production_gates_enforced, production_gate_findings,"
                    " recipe_sha256, golden_sha256, runtime_detector_identifier,"
                    " coordinate_space, run_json, session_id, arrived_after_close)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        inspection_id,
                        board_id,
                        str(payload.get("side") or ""),
                        event_id,
                        str(payload.get("status") or ""),
                        str(payload.get("reason") or ""),
                        str(payload.get("started_at") or utc_now_iso()),
                        utc_now_iso(),
                        1 if payload.get("production_gates_enforced", True) else 0,
                        json.dumps(
                            list(payload.get("production_gate_findings") or []),
                            ensure_ascii=False,
                        ),
                        str(payload.get("recipe_sha256") or ""),
                        str(payload.get("golden_sha256") or ""),
                        str(payload.get("runtime_detector_identifier") or ""),
                        str(payload.get("coordinate_space") or ""),
                        json.dumps(payload, ensure_ascii=False),
                        session_id,
                        late,
                    ),
                )
                for item, sha in zip(defects, stored_images):
                    box = item.bbox or (None, None, None, None)
                    self.connection.execute(
                        "INSERT INTO defect (inspection_id, slot_id, status, reason,"
                        " x1, y1, x2, y2, coordinate_space, image_sha256)"
                        " VALUES (?,?,?,?,?,?,?,?,?,?)",
                        (inspection_id, item.slot_id, item.status, item.reason,
                         *box, item.coordinate_space, sha),
                    )
        except sqlite3.IntegrityError:
            # Hai tiến trình cùng gửi một sự kiện: bên thua đọc lại bản của bên
            # thắng thay vì báo lỗi -- vẫn đúng nghĩa "một sự kiện, một bản ghi".
            row = self.connection.execute(
                "SELECT inspection_id FROM inspection WHERE event_id = ?", (event_id,)
            ).fetchone()
            if row is None:
                raise
            return str(row["inspection_id"])
        return inspection_id

    def load_inspection(self, inspection_id: str) -> StoredInspection:
        row = self.connection.execute(
            "SELECT * FROM inspection WHERE inspection_id = ?", (inspection_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"không có inspection {inspection_id}")
        defects = self.connection.execute(
            "SELECT d.*, a.rel_path FROM defect d"
            " LEFT JOIN image_asset a ON a.sha256 = d.image_sha256"
            " WHERE d.inspection_id = ? ORDER BY d.defect_id",
            (inspection_id,),
        ).fetchall()
        return StoredInspection(
            inspection_id=str(row["inspection_id"]),
            board_id=str(row["board_id"]),
            side=str(row["side"]),
            status=str(row["status"]),
            started_at=str(row["started_at"]),
            production_gates_enforced=bool(row["production_gates_enforced"]),
            production_gate_findings=tuple(json.loads(row["production_gate_findings"])),
            run=json.loads(row["run_json"]),
            defects=tuple(dict(item) for item in defects),
        )

    def inspections_for_board(self, board_id: str) -> tuple[Mapping[str, Any], ...]:
        rows = self.connection.execute(
            "SELECT inspection_id, side, status, started_at,"
            " production_gates_enforced FROM inspection"
            " WHERE board_id = ? ORDER BY started_at, inspection_id",
            (board_id,),
        ).fetchall()
        return tuple(dict(row) for row in rows)

    def sides_still_missing(
        self, board_id: str, *, required: Sequence[str] = ("top", "bottom")
    ) -> tuple[str, ...]:
        """Mặt bắt buộc nào chưa có lần chạy ĐẠT ở chế độ production.

        Cùng quy tắc với ``golden.inspector.missing_required_sides`` nhưng hỏi
        trên dữ liệu đã lưu: sau khi khởi động lại thì câu hỏi vẫn phải trả lời
        được, mà các đối tượng ``InspectionRun`` trong bộ nhớ thì không còn.
        """

        rows = self.connection.execute(
            "SELECT DISTINCT side FROM inspection WHERE board_id = ?"
            " AND status = 'pass' AND production_gates_enforced = 1",
            (board_id,),
        ).fetchall()
        passed = {str(row["side"]) for row in rows}
        return tuple(side for side in required if side not in passed)

    # ------------------------------------------------------------ phiên kiểm

    def open_session(
        self,
        *,
        board_id: str,
        station: str,
        operator: str | None = None,
        required_sides: Sequence[str] = ("top", "bottom"),
    ) -> str:
        """Mở một phiên kiểm cho một tấm bo tại một trạm.

        **Mỗi trạm chỉ một phiên đang mở** (§4.3), ép bằng chỉ mục một phần
        trong lược đồ chứ không bằng kiểm tra ở tầng ứng dụng — hai tiến trình
        cùng mở thì tầng ứng dụng thua, cơ sở dữ liệu thì không.

        ``required_sides`` được **ghi vào phiên**, không lấy mặc định lúc đọc:
        một bo một mặt và một bo hai mặt phải phân biệt được sau nhiều tháng.
        """

        session_id = f"SES-{uuid.uuid4().hex[:12]}"
        try:
            with self.connection:
                self.connection.execute(
                    "INSERT INTO scan_session (session_id, board_id, station,"
                    " operator, required_sides, status, opened_at)"
                    " VALUES (?, ?, ?, ?, ?, 'open', ?)",
                    (session_id, board_id, station, operator,
                     json.dumps(list(required_sides)), utc_now_iso()),
                )
        except sqlite3.IntegrityError as exc:
            raise RuntimeError(
                f"trạm {station} đang có phiên mở; đóng hoặc huỷ nó trước"
            ) from exc
        return session_id

    def session_sides_missing(self, session_id: str) -> tuple[str, ...]:
        """Mặt bắt buộc của CHÍNH phiên này chưa có lần chạy đạt.

        Khác ``sides_still_missing`` ở chỗ nó hỏi trong phạm vi một phiên và
        dùng ``required_sides`` đã ghi lúc mở, chứ không lấy mặc định.
        """

        row = self.connection.execute(
            "SELECT required_sides FROM scan_session WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"không có phiên {session_id}")
        required = json.loads(row["required_sides"])
        rows = self.connection.execute(
            "SELECT DISTINCT side FROM inspection WHERE session_id = ?"
            " AND status = 'pass' AND production_gates_enforced = 1"
            " AND arrived_after_close = 0",
            (session_id,),
        ).fetchall()
        passed = {str(item["side"]) for item in rows}
        return tuple(side for side in required if side not in passed)

    def close_session(
        self, session_id: str, *, reason: str | None = None
    ) -> tuple[str, ...]:
        """Đóng phiên. Trả về các mặt còn thiếu tại thời điểm đóng.

        Còn thiếu mặt mà không nêu lý do thì **từ chối**: bỏ dở im lặng là cách
        một tấm bo chưa kiểm đủ đi tiếp mà không ai biết. Nêu lý do thì cho
        đóng, và lý do được lưu lại.
        """

        missing = self.session_sides_missing(session_id)
        if missing and not reason:
            raise ValueError(
                f"phiên còn thiếu mặt {', '.join(missing)}; muốn đóng thì phải "
                "nêu lý do, không bỏ dở im lặng"
            )
        with self.connection:
            self.connection.execute(
                "UPDATE scan_session SET status = 'closed', closed_at = ?,"
                " close_reason = ? WHERE session_id = ? AND status = 'open'",
                (utc_now_iso(), reason, session_id),
            )
        return missing

    def cancel_session(self, session_id: str, *, reason: str) -> None:
        """Huỷ phiên. Lý do là bắt buộc — phiên huỷ không lý do không đối soát được."""

        if not reason:
            raise ValueError("huỷ phiên phải nêu lý do")
        with self.connection:
            self.connection.execute(
                "UPDATE scan_session SET status = 'cancelled', closed_at = ?,"
                " close_reason = ? WHERE session_id = ? AND status = 'open'",
                (utc_now_iso(), reason, session_id),
            )

    def open_session_at(self, station: str) -> str | None:
        row = self.connection.execute(
            "SELECT session_id FROM scan_session WHERE station = ? AND status = 'open'",
            (station,),
        ).fetchone()
        return None if row is None else str(row["session_id"])

    # ------------------------------------------------------------- kho ảnh

    def image_file(self, sha256: str) -> Path:
        row = self.connection.execute(
            "SELECT rel_path FROM image_asset WHERE sha256 = ?", (sha256,)
        ).fetchone()
        if row is None:
            raise KeyError(f"không có ảnh {sha256}")
        return self.root / str(row["rel_path"])

    def delete_unreferenced_images(self) -> tuple[str, ...]:
        """Thu hồi ảnh KHÔNG còn hàng lỗi nào trỏ tới. Trả về sha đã xoá.

        Đếm tham chiếu trước khi xoá (§8.2): một ảnh gốc dùng chung hoặc ảnh
        Golden có thể được nhiều lần phân tích trỏ tới, xoá theo tuổi là mất
        bằng chứng của những lần còn hiệu lực.
        """

        rows = self.connection.execute(
            "SELECT a.sha256, a.rel_path FROM image_asset a"
            " WHERE NOT EXISTS (SELECT 1 FROM defect d WHERE d.image_sha256 = a.sha256)"
        ).fetchall()
        removed: list[str] = []
        with self.connection:
            for row in rows:
                path = self.root / str(row["rel_path"])
                path.unlink(missing_ok=True)
                self.connection.execute(
                    "DELETE FROM image_asset WHERE sha256 = ?", (row["sha256"],)
                )
                removed.append(str(row["sha256"]))
        return tuple(removed)

    def _store_image(self, source: str | Path) -> str:
        """Chép ảnh vào kho địa chỉ-theo-nội dung, trả về sha256."""

        path = Path(source)
        data = path.read_bytes()
        if not data:
            raise ValueError(f"ảnh rỗng: {path}")
        sha = hashlib.sha256(data).hexdigest()
        row = self.connection.execute(
            "SELECT rel_path FROM image_asset WHERE sha256 = ?", (sha,)
        ).fetchone()
        if row is not None:
            return sha
        suffix = path.suffix.lower() or ".png"
        rel = f"images/{sha[:2]}/{sha}{suffix}"
        target = self.root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        with self.connection:
            self.connection.execute(
                "INSERT INTO image_asset (sha256, rel_path, byte_size, media_type,"
                " created_at) VALUES (?, ?, ?, ?, ?)",
                (sha, rel, len(data),
                 _MEDIA_BY_SUFFIX.get(suffix, "application/octet-stream"),
                 utc_now_iso()),
            )
        return sha
