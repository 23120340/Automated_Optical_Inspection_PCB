"""Lược đồ CSDL lịch sử kiểm tra, có phiên bản và migration.

Dùng ``sqlite3`` của thư viện chuẩn: bản đầu chạy một trạm, không thêm phụ thuộc
nào. Mọi câu lệnh ở đây là SQL thuần nên đổi sang Postgres sau này là đổi driver,
không phải viết lại mô hình dữ liệu.

**Ảnh không nằm trong CSDL.** Bảng ``image_asset`` chỉ giữ *siêu dữ liệu* và
đường dẫn tương đối; byte ảnh nằm trong kho ảnh địa chỉ-theo-nội dung. Nhét ảnh
vào SQLite làm file phình nhanh và làm sao lưu/khôi phục khó hơn hẳn.

Bốn ràng buộc dưới đây là **yêu cầu nghiệp vụ**, không phải trang trí; kế hoạch
``ke_hoach_so_hoa_du_lieu_va_truy_xuat_aoi.md`` nêu từng cái:

* mỗi ``event_id`` chỉ sinh **một** inspection (§8.2 — gửi lại không tạo bản ghi
  mới);
* lưu **cả** kết quả đạt, không chỉ lỗi (§6.4 — thiếu nó thì không tính được tỉ
  lệ đạt lần đầu);
* mỗi lỗi giữ **toạ độ + hệ quy chiếu** của chính nó (§6.2 — toạ độ không kèm hệ
  quy chiếu là con số vô nghĩa);
* ảnh còn được tham chiếu thì **không được xoá** (§8.2).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

#: Tăng khi thêm migration. ``PRAGMA user_version`` của SQLite giữ số này ngay
#: trong file, nên không cần bảng riêng và không lệch được khỏi dữ liệu.
SCHEMA_VERSION = 1

_MIGRATION_1 = """
CREATE TABLE board (
    board_id        TEXT PRIMARY KEY,
    -- Serial THẬT đọc từ bo. NULL khi chưa biết -- xem kế hoạch §4.2b: bản đầu
    -- dùng ID nội bộ tự sinh và gắn serial sau bằng một sự kiện liên kết.
    -- Phải phân biệt được "chưa có serial" với "serial là chuỗi rỗng".
    serial          TEXT UNIQUE,
    -- Nguồn của định danh: 'serial' (đọc từ bo) hay 'internal' (tự sinh).
    -- Thiếu trường này thì sau không biết ID nào truy ngược được ra bo thật.
    identity_source TEXT NOT NULL CHECK (identity_source IN ('serial', 'internal')),
    board_type      TEXT,
    revision        TEXT,
    created_at      TEXT NOT NULL
);

CREATE TABLE inspection (
    inspection_id   TEXT PRIMARY KEY,
    board_id        TEXT NOT NULL REFERENCES board(board_id),
    -- Hai mặt là hai lần kiểm tra riêng của cùng một PCB vật lý (§4.4).
    side            TEXT NOT NULL CHECK (side IN ('top', 'bottom')),
    -- Khoá chống trùng: gửi lại cùng một sự kiện không được tạo bản ghi mới.
    event_id        TEXT NOT NULL UNIQUE,
    status          TEXT NOT NULL,
    reason          TEXT NOT NULL DEFAULT '',
    started_at      TEXT NOT NULL,
    received_at     TEXT NOT NULL,
    -- Chế độ chạy nằm NGAY TRONG bản ghi, không suy từ cấu hình lúc đọc (§9.3).
    production_gates_enforced INTEGER NOT NULL CHECK (production_gates_enforced IN (0, 1)),
    production_gate_findings  TEXT NOT NULL DEFAULT '[]',
    recipe_sha256   TEXT NOT NULL,
    golden_sha256   TEXT NOT NULL,
    runtime_detector_identifier TEXT NOT NULL,
    coordinate_space TEXT NOT NULL,
    run_json        TEXT NOT NULL
);

CREATE INDEX inspection_by_board ON inspection(board_id, side, started_at);

CREATE TABLE image_asset (
    -- Địa chỉ theo NỘI DUNG: hai lần lưu cùng một ảnh chỉ tốn một bản.
    sha256      TEXT PRIMARY KEY,
    rel_path    TEXT NOT NULL UNIQUE,
    byte_size   INTEGER NOT NULL CHECK (byte_size > 0),
    media_type  TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE TABLE defect (
    defect_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    inspection_id TEXT NOT NULL REFERENCES inspection(inspection_id) ON DELETE CASCADE,
    slot_id       TEXT NOT NULL,
    status        TEXT NOT NULL,
    reason        TEXT NOT NULL DEFAULT '',
    -- Toạ độ KHÔNG tách rời hệ quy chiếu (§6.2). Cho phép NULL trọn bộ khi lỗi
    -- không gắn với một vùng cụ thể, nhưng không cho phép có toạ độ mà thiếu hệ.
    x1 REAL, y1 REAL, x2 REAL, y2 REAL,
    coordinate_space TEXT,
    image_sha256  TEXT REFERENCES image_asset(sha256),
    CHECK (
        (x1 IS NULL AND y1 IS NULL AND x2 IS NULL AND y2 IS NULL)
        OR (x1 IS NOT NULL AND y1 IS NOT NULL AND x2 IS NOT NULL
            AND y2 IS NOT NULL AND coordinate_space IS NOT NULL)
    )
);

CREATE INDEX defect_by_inspection ON defect(inspection_id);
CREATE INDEX defect_by_image ON defect(image_sha256);
"""

MIGRATIONS: tuple[str, ...] = (_MIGRATION_1,)


def connect(path: str | Path) -> sqlite3.Connection:
    """Mở kết nối đã bật khoá ngoại và trả về hàng dạng ``sqlite3.Row``.

    ``PRAGMA foreign_keys`` mặc định **TẮT** trong SQLite và chỉ có tác dụng
    theo từng kết nối. Quên bật nó thì mọi ``REFERENCES`` ở trên chỉ là chú
    thích, và ràng buộc "không xoá ảnh còn được tham chiếu" trở thành lời hứa
    suông.
    """

    connection = sqlite3.connect(str(path))
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def migrate(connection: sqlite3.Connection) -> int:
    """Đưa lược đồ lên ``SCHEMA_VERSION``; trả về phiên bản trước khi chạy.

    Chạy lại nhiều lần không sao — mỗi migration chỉ chạy khi ``user_version``
    còn thấp hơn nó.
    """

    current = int(connection.execute("PRAGMA user_version").fetchone()[0])
    if current > SCHEMA_VERSION:
        raise RuntimeError(
            f"CSDL ở phiên bản {current}, mã nguồn chỉ biết tới {SCHEMA_VERSION}. "
            "Bản cũ mở file mới là đường ngắn nhất để ghi hỏng dữ liệu."
        )
    for index in range(current, SCHEMA_VERSION):
        connection.executescript(MIGRATIONS[index])
        connection.execute(f"PRAGMA user_version = {index + 1}")
    connection.commit()
    return current
