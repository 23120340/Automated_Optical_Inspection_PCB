"""Lưu trữ lịch sử kiểm tra: siêu dữ liệu trong SQLite, ảnh trong kho riêng.

Xem ``docs/plans/ke_hoach_so_hoa_du_lieu_va_truy_xuat_aoi.md``.
"""

from .bridge import PASSING_SLOT_STATUSES, defects_from_run
from .repository import DefectRecord, InspectionStore, StoredInspection
from .schema import SCHEMA_VERSION, connect, migrate

__all__ = [
    "PASSING_SLOT_STATUSES",
    "defects_from_run",
    "DefectRecord",
    "InspectionStore",
    "StoredInspection",
    "SCHEMA_VERSION",
    "connect",
    "migrate",
]
