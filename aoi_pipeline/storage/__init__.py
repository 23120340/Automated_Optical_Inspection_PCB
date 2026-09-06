"""Lưu trữ lịch sử kiểm tra: siêu dữ liệu trong SQLite, ảnh trong kho riêng.

Xem ``Docs/ke_hoach/ke_hoach_so_hoa_du_lieu_va_truy_xuat_aoi.md``.
"""

from .repository import DefectRecord, InspectionStore, StoredInspection
from .schema import SCHEMA_VERSION, connect, migrate

__all__ = [
    "DefectRecord",
    "InspectionStore",
    "StoredInspection",
    "SCHEMA_VERSION",
    "connect",
    "migrate",
]
