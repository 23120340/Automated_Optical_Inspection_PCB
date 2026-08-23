"""Mục đánh giá model, chạy thật qua giao diện.

`tests/test_model_feedback.py` giữ tầng lưu trữ. Ở đây kiểm những thứ chỉ hỏng
khi ráp vào app: mục có hiện ở đúng trang của bước không, có **không** lọt vào
sidebar không, và toạ độ ghi xuống có đúng bằng toạ độ của mục được chọn không.

Chỗ dễ hỏng lặng lẽ nhất là bước 6: `return` sớm của nó nằm trong `with
content:` và thoát khỏi cả hàm render, nên trước khi tách hàm thì bất cứ thứ gì
thêm sau khối cột đều là code chết.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from aoi_pipeline.model_feedback import load_feedback

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")
DIGEST = "d" * 64


@pytest.fixture(autouse=True)
def feedback_dir(monkeypatch, tmp_path):
    """Không bao giờ ghi vào `feedback/` của repo khi chạy test."""

    monkeypatch.setenv("AOI_FEEDBACK_DIR", str(tmp_path))
    return tmp_path


def _crop(index: int):
    from app.pipeline_bridge import CropRecord

    return CropRecord(
        crop_id=f"c{index}", label="resistor",
        image=np.zeros((20, 20, 3), np.uint8),
        bbox=(10 * index, 10, 10 * index + 18, 28),
        confidence=0.7, source="cv",
    )


def _detection(index: int):
    from app.pipeline_bridge import DetectionRecord

    return DetectionRecord(
        detection_id=f"d{index}", label="capacitor", confidence=0.61,
        bbox=(20 * index, 30, 20 * index + 16, 54), source="model",
    )


def _app(step: int, *, detections=None) -> AppTest:
    instance = AppTest.from_file(APP, default_timeout=180)
    instance.session_state["workspace_mode"] = "pipeline_lab"
    instance.run()
    instance.session_state["input_image"] = np.random.default_rng(1).integers(
        0, 255, (600, 800, 3), dtype=np.uint8)
    instance.session_state["input_digest"] = DIGEST
    instance.session_state["input_name"] = "board.png"
    instance.session_state["crops"] = [_crop(i) for i in range(3)]
    if detections is not None:
        from app.pipeline_bridge import DetectionResult

        instance.session_state["detection_result"] = DetectionResult(
            image=instance.session_state["input_image"], mode="model",
            message="", metrics={}, detections=detections,
        )
    instance.session_state["active_step"] = step
    instance.run()
    return instance


def _radio_keys(app: AppTest) -> list[str]:
    return [item.key for item in app.radio if item.key and item.key.startswith("fb_")]


# --------------------------------------------------------------------------
# Có mặt ở đúng chỗ
# --------------------------------------------------------------------------


@pytest.mark.parametrize("step, expected", [
    (4, "fb_detection_mode"),
    (6, "fb_classification_mode"),
    (7, "fb_solder_mode"),
])
def test_the_section_renders_on_the_page_of_its_own_step(step, expected) -> None:
    app = _app(step)
    assert not app.exception, [str(e.value) for e in app.exception]
    assert expected in _radio_keys(app)


def test_step_six_shows_it_even_before_the_classifier_has_run() -> None:
    """Đây là test ghim việc tách hàm. Trước khi tách, `return` sớm ở
    `with content:` thoát khỏi cả hàm render bước 6, nên mục đánh giá không
    bao giờ được vẽ khi chưa có kết quả phân loại — đúng lúc cần nó nhất."""

    app = _app(6)
    assert app.session_state["classification_result"] is None
    assert "fb_classification_mode" in _radio_keys(app)


def test_the_section_is_not_in_the_sidebar() -> None:
    """Yêu cầu nói rõ: đặt trong trang làm việc của model, không phải sidebar."""

    app = _app(4)
    sidebar_keys = [
        item.key for item in app.sidebar.radio if item.key and item.key.startswith("fb_")
    ]
    assert sidebar_keys == []


def test_it_stays_quiet_when_there_is_no_image_at_all() -> None:
    """Không có khung ảnh thì không có không gian toạ độ, nên một khung ghi
    xuống sẽ vô nghĩa."""

    app = AppTest.from_file(APP, default_timeout=180)
    app.session_state["workspace_mode"] = "pipeline_lab"
    app.session_state["active_step"] = 4
    app.run()
    assert not app.exception
    assert _radio_keys(app) == []


# --------------------------------------------------------------------------
# Ghi được, và ghi đúng toạ độ
# --------------------------------------------------------------------------


def test_choosing_a_result_row_records_that_row_s_exact_box(feedback_dir) -> None:
    detections = [_detection(0), _detection(1)]
    app = _app(4, detections=detections)

    app.selectbox(key="fb_detection_target").select(
        f"{detections[1].detection_id} · capacitor 0.61").run()
    submit = [b for b in app.button if "Ghi nhận" in (b.label or "")]
    assert submit, "không thấy nút Ghi nhận"
    submit[0].click().run()

    entries, problems = load_feedback(root=feedback_dir)
    assert problems == []
    assert len(entries) == 1
    entry = entries[0]
    assert entry.bbox == tuple(detections[1].bbox), "toạ độ phải bằng đúng box đã chọn"
    assert entry.origin == "result_row"
    assert entry.target.record_id == "d1"
    assert entry.target.model_label == "capacitor"
    assert entry.stage == "detection" and entry.step == 4


def test_the_magnifier_records_the_box_it_previewed(feedback_dir) -> None:
    app = _app(4)
    app.radio(key="fb_detection_mode").set_value("Kính lúp (model bỏ sót)").run()
    app.slider(key="fb_detection_cx").set_value(200).run()
    app.slider(key="fb_detection_cy").set_value(150).run()

    submit = [b for b in app.button if "Ghi nhận" in (b.label or "")]
    submit[0].click().run()

    entries, _ = load_feedback(root=feedback_dir)
    assert len(entries) == 1
    entry = entries[0]
    assert entry.origin == "magnifier"
    assert entry.target.record_id is None
    # Ô mặc định 96 px quanh tâm (200, 150).
    assert entry.bbox == (152, 102, 248, 198)


def test_the_entry_carries_the_digest_of_the_model_it_judges(feedback_dir) -> None:
    """Phần quan trọng nhất của định dạng. App tự nạp model trong
    ``models/active/``, và bản ghi phải gắn với **file trọng số** đó — tên file
    luôn là ``best.onnx``, chỉ sha256 mới phân biệt được hai bản."""

    from aoi_pipeline.model_registry import find_active

    app = _app(4, detections=[_detection(0)])
    submit = [b for b in app.button if "Ghi nhận" in (b.label or "")]
    submit[0].click().run()

    entries, _ = load_feedback(root=feedback_dir)
    identity = entries[0].model
    assert identity.loaded is True
    assert identity.sha256 == find_active("detector").summary().sha256
    assert identity.compare_key == identity.sha256
    assert entries[0].runtime_mode == "MODEL"


def test_a_recording_made_with_no_model_says_so(feedback_dir) -> None:
    """Bước 4 chạy CV demo vẫn đáng ghi nhận; ngữ cảnh phải đi kèm để sau này
    lọc ra được."""

    app = _app(4, detections=[_detection(0)])
    app.session_state["component_model_path"] = None
    app.session_state["component_model_name"] = None
    app.run()

    submit = [b for b in app.button if "Ghi nhận" in (b.label or "")]
    submit[0].click().run()

    entries, _ = load_feedback(root=feedback_dir)
    assert entries[0].model.loaded is False
    assert entries[0].model.compare_key == "no-model"
    assert entries[0].runtime_mode == "CV DEMO"


# --------------------------------------------------------------------------
# Xem lại
# --------------------------------------------------------------------------


def test_a_saved_entry_comes_back_in_the_history(feedback_dir) -> None:
    app = _app(4, detections=[_detection(0)])
    submit = [b for b in app.button if "Ghi nhận" in (b.label or "")]
    submit[0].click().run()

    app.run()
    body = " ".join(item.value for item in app.markdown if isinstance(item.value, str))
    assert "Đã ghi nhận cho ảnh này" in body


def test_an_entry_from_another_image_is_not_shown_for_this_one(feedback_dir) -> None:
    """Toạ độ chỉ có nghĩa với đúng tấm ảnh đã đo trên nó."""

    from aoi_pipeline.model_feedback import (
        FeedbackEntry, ModelIdentity, append_feedback,
    )

    append_feedback(
        FeedbackEntry(
            stage="detection", step=4, bbox=(1, 1, 9, 9), error_kind="missed",
            model=ModelIdentity(slot="component", kind="detector"),
            source_name="anh-khac.png", source_sha256="e" * 64,
            analysis_width=800, analysis_height=600,
        ),
        path=feedback_dir / "log.jsonl",
    )
    app = _app(4)
    heading = [
        item.label for item in app.get("expander")
        if "Đánh giá model" in (item.label or "")
    ]
    assert heading and "(0 đã ghi" in heading[0], heading
