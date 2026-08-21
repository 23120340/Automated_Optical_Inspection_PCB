"""Khoá các chốt giữ model 6.2 khỏi tự mình ra quyết định.

Đo được 2026-08-22 trên board thật (119 ROI mối hàn):

    `bridge` trên ROI thật              61.3%
    `bridge` trên nhiễu ngẫu nhiên      70.0%
    `bridge` trên mảnh board bất kỳ     68.3%
    phần chồng lấn hai phân bố          80.4%
    vượt ngưỡng accept 0.85             11/119 (9.2%)

Model KHÔNG hoàn toàn suy biến — chi-square 32.91 (dof 6, ngưỡng 12.59) bác
được giả thuyết "hai phân bố là một". Nhưng 80% đầu ra của nó giống hệt nhau
dù đưa mối hàn thật hay một mảnh board bất kỳ, và tập val của chính nó ghi
review_rate 0.4569 tại ngưỡng đang chạy.

Vì thế ba chốt dưới đây phải giữ nguyên. Chúng không phải sở thích cấu hình:
tắt bất kỳ cái nào cũng biến một model gọi 61% mối hàn là chập thành thẩm
quyền, và mọi board sẽ thành phế phẩm. Sửa được các test này — nhưng phải kèm
số đo mới chứng minh model đã khác.

Đầy đủ ở `Docs/danh_gia_model_6_2.md`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aoi_pipeline.config import SolderGradingConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = PROJECT_ROOT / "models" / "active" / "solder" / "model_manifest.json"

#: Đo trên board thật, xem docstring. Một model thay thế phải khá hơn con số này.
MEASURED_BRIDGE_RATE_ON_REAL_ROIS = 0.613
MEASURED_BRIDGE_RATE_ON_NOISE = 0.700
MEASURED_DISTRIBUTION_OVERLAP = 0.804


def test_the_model_can_never_accept_a_joint_on_its_own_below_high_confidence() -> None:
    config = SolderGradingConfig()
    assert config.model_accept_probability >= 0.80, (
        "Chỉ 9.2% ROI thật vượt được 0.85. Hạ ngưỡng này là mở cửa cho phần "
        "đuôi mà model đoán mò."
    )


def test_the_physical_floor_stays_on() -> None:
    """Sàn vật lý là thứ duy nhất chặn được escape khi model tự tin sai."""

    config = SolderGradingConfig()
    assert config.escape_guard_enabled is True
    assert config.escape_guard_solder_ratio > 0.0


def test_disagreement_goes_to_review_not_to_the_model() -> None:
    """Chốt quan trọng nhất. Tắt cái này là để một model gọi 61% mối hàn là
    `bridge` tự quyết, và cả lô hàng thành phế phẩm."""

    config = SolderGradingConfig()
    assert config.disagreement_is_review is True


def test_the_pipeline_still_works_with_no_model_at_all() -> None:
    """Lớp luật vật lý là thứ đáng tin nhất hiện nay, nên nó phải đứng một mình
    được — nếu không, "gỡ model ra" sẽ không còn là một lựa chọn."""

    config = SolderGradingConfig()
    assert config.enabled is True
    assert config.model_path is None
    assert config.manifest_path is None


@pytest.mark.skipif(not MANIFEST_PATH.is_file(), reason="chưa có artifact 6.2")
def test_the_shipped_operating_point_is_recorded_so_a_replacement_can_beat_it() -> None:
    """Manifest tự ghi lại: 46% mối hàn phải xem tay ở ngưỡng đang chạy.

    Test này không đòi model tốt hơn — nó ghim con số lại. Ai thay model mà
    review_rate tệ hơn sẽ thấy test đỏ và biết mình đi lùi.
    """

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    accept = manifest["decision_thresholds"]["accept"]
    sweep = manifest["training"]["threshold_sweep"]
    here = next((row for row in sweep if abs(row["accept"] - accept) < 1e-9), None)
    assert here is not None, f"sweep không có điểm làm việc {accept}"

    assert here["review_rate"] <= 0.46, (
        f"review_rate {here['review_rate']:.4f} tại ngưỡng {accept} — model mới "
        "phải không tệ hơn bản đã đo (0.4569)."
    )
    assert here["escape"] <= 0.010, (
        f"escape {here['escape']:.4f} — lỗi lọt lưới không được xấu đi so với "
        "bản đã đo (0.0098)."
    )


@pytest.mark.skipif(not MANIFEST_PATH.is_file(), reason="chưa có artifact 6.2")
def test_the_manifest_admits_which_classes_came_from_a_single_source() -> None:
    """Ba lớp học từ một dataset duy nhất là lý do chính khiến model học đặc
    trưng của *nguồn ảnh* thay vì của *khuyết tật*. Manifest phải nói ra điều
    đó — mất trường này là mất mất luôn cảnh báo."""

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    training = manifest["training"]
    assert "single_source_classes" in training
    assert "sources" in training
    assert "class_counts" in training

    counts = training["class_counts"]
    imbalance = max(counts.values()) / min(counts.values())
    assert imbalance <= 11.0, (
        f"mất cân bằng {imbalance:.1f}x — bản đã đo là 10.6x, đừng để tệ hơn"
    )
