"""Streamlit dashboard for Golden inspection and AOI steps 0 through 6.2.

Run from the repository root:

    streamlit run app/streamlit_app.py

All images in session state and at the pipeline boundary use OpenCV BGR order.
"""

from __future__ import annotations

import collections
from collections.abc import Mapping, Sequence
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import html
import io
import json
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Callable
from uuid import uuid4
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from aoi_pipeline.solder.cad import CadError, load_cad, register_cad  # noqa: E402
from aoi_pipeline.bom import (  # noqa: E402
    BillOfMaterials,
    BomError,
    load_bom,
    reconcile_bom,
)
from streamlit_image_coordinates import streamlit_image_coordinates  # noqa: E402
from aoi_pipeline.models import BoundingBox  # noqa: E402
from aoi_pipeline.model_feedback import (  # noqa: E402
    ERROR_KINDS,
    FeedbackEntry,
    FeedbackError,
    FeedbackTargetRef,
    ModelIdentity,
    append_feedback,
    entries_for_source,
    error_label,
    feedback_root,
    load_feedback,
    preprocess_identity,
)
from aoi_pipeline.model_registry import (  # noqa: E402
    ModelEntry,
    discover_models,
    find_active,
)
from app.pipeline_bridge import (  # noqa: E402
    BoardResult,
    ClassificationRecord,
    ClassificationResult,
    CropRecord,
    DetectionRecord,
    DetectionResult,
    InspectionRecipeRecord,
    InspectionResult,
    PipelineBridge,
    SolderCropRecord,
    SolderResult,
    SolderVerdictRecord,
    StageResult,
)


APP_TITLE = "AOI PCB · Workbench"
APP_VERSION = "0.6.0"
MAX_IMAGE_BYTES = 64 * 1024 * 1024
MAX_IMAGE_PIXELS = 50_000_000
MIN_SOURCE_LONG_SIDE = 1280
MIN_SOURCE_SHORT_SIDE = 960
MIN_SOURCE_PIXELS = MIN_SOURCE_LONG_SIDE * MIN_SOURCE_SHORT_SIDE
# The resolution figure is still measured and still shown; it just no longer
# blocks the import. Set this back to True for a production line, where running
# a board nobody can grade is worse than refusing it. During development it only
# stops you from feeding the pipeline the images you have.
ENFORCE_SOURCE_RESOLUTION = False
MAX_CALIBRATION_PROFILE_BYTES = 256 * 1024
# Mirror of aoi_pipeline.grading.classifier.MANIFEST_SCHEMA and of
# aoi_pipeline.classification.MANIFEST_SCHEMA. Kept as literals so the UI module
# does not import the core at load time; a drift test asserts they still match.
CLASSIFIER_MANIFEST_SCHEMA = "pcb-component-classifier/1.0"
SOLDER_MANIFEST_SCHEMA = "pcb-solder-defect-classifier/1.0"
#: (chỉ số nội bộ, tên, mô tả, mã, số hiển thị)
#:
#: Chỉ số nội bộ là khoá của `renderers` và của `statuses`; nó KHÔNG đổi khi
#: thứ tự hiển thị đổi. Số hiển thị mới là thứ người dùng thấy, và dự án vốn
#: đã dùng số lẻ ở chỗ khác (`5.5 · ROI chân hàn`, `6.1`, `6.2`), nên Golden
#: Inspection là **3.5**: nó chỉ cần ảnh đã căn và vùng board, không cần
#: detect hay phân loại. Đặt nó ở cuối chỉ vì nó được thêm sau cùng là sai
#: thứ tự công việc.
STEP_DEFINITIONS = (
    (0, "Thu thập ảnh", "Import ảnh PCB", "IN", "0"),
    (1, "Tiền xử lý", "Undistort và chuẩn hóa", "FX", "1"),
    (2, "Căn chỉnh PCB", "Golden image + homography", "AL", "2"),
    (3, "Khoanh vùng PCB", "Xác định board ROI", "ROI", "3"),
    (8, "Golden Inspection", "Recipe + so sánh với board chuẩn", "GLD", "3.5"),
    (4, "Phát hiện linh kiện", "Detector từ Kaggle", "AI", "4"),
    (5, "Cắt linh kiện", "Crop + normalize + export", "CUT", "5"),
    (6, "Phân loại linh kiện", "Family + accept/review/unknown", "CLS", "6.1"),
    (7, "Kiểm tra mối hàn", "ROI chân hàn + chấm lỗi", "SLD", "6.2"),
)

#: Chỉ số nội bộ -> hàng của nó. Tra theo VỊ TRÍ trong tuple sẽ sai ngay khi
#: thứ tự hiển thị khác thứ tự chỉ số.
STEP_BY_INDEX = {row[0]: row for row in STEP_DEFINITIONS}

#: Thứ tự công việc thật, theo chỉ số nội bộ. `_invalidate_after` dùng nó chứ
#: không dùng `range()`: sau khi Golden thành 3.5, "các bước sau" không còn
#: trùng với "chỉ số lớn hơn".
STEP_ORDER = tuple(row[0] for row in STEP_DEFINITIONS)
SOLDER_ROI_COLORS = {
    "joint": (0, 200, 255),  # BGR amber
    "body": (255, 170, 0),   # BGR blue
}

SOLDER_SOURCE_COLORS = {
    "cad+derived": (80, 220, 80),   # BGR green: both sources agreed
    "cad": (255, 120, 255),         # BGR magenta: CAD only
    "derived": (0, 200, 255),       # BGR amber: detector geometry only
}

CAD_SEVERITY_ICONS = {"defect": "🔴", "review": "🟠", "info": "🔵"}

SOLDER_MIN_READABLE_PX = 24

VERDICT_DECISION_COLORS = {
    "accept": (80, 200, 80),   # BGR green
    "review": (0, 170, 255),   # BGR orange
    "reject": (40, 40, 230),   # BGR red
}

VERDICT_DECISION_ICONS = {"accept": "🟢", "review": "🟠", "reject": "🔴"}

STATUS_LABELS = {
    "pending": "Chờ chạy",
    "running": "Đang chạy",
    "done": "Hoàn tất",
    "demo": "CV demo",
    "warning": "Cảnh báo",
    "skipped": "Bỏ qua",
    "error": "Có lỗi",
}


st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _default_config() -> dict[str, Any]:
    return {
        "preprocess": {
            "undistort": False,
            "calibration_profile": None,
            "undistort_alpha": 0.0,
            "calibration_aspect_tolerance": 0.01,
            "resize_enabled": True,
            "max_side": 4096,
            "denoise": "Bilateral",
            "denoise_strength": 5,
            "white_balance": True,
            "clahe": True,
            "clahe_clip": 2.0,
            "normalize": True,
            "sharpen": 0.35,
        },
        "alignment": {
            "method": "ORB",
            "features": 3000,
            "match_ratio": 0.75,
            "ransac_threshold": 4.0,
        },
        "board": {
            "min_area_ratio": 0.08,
            "min_rectangularity": 0.45,
            "padding_ratio": 0.005,
        },
        "components": {
            "confidence": 0.25,
            "iou": 0.45,
            "max_candidates": 2000,
            "min_area_ratio": 0.00004,
            "max_area_ratio": 0.035,
            "device": "auto",
            # Kaggle notebook validates/exports the YOLO26 one-to-many head so
            # IoU/NMS behavior is identical in Kaggle and in this local app.
            "end2end": False,
            "tiling_mode": "auto",
            "tile_size": 1280,
            "min_tile_size": 640,
            "detail_window_ratio": 0.64,
            "tile_overlap": 0.20,
            "tile_trigger_scale": 1.25,
            "full_image_pass": True,
            "tile_confidence": 0.20,
            "tile_led_confidence": 0.35,
            "merge_iou": 0.45,
            "seam_ios": 0.50,
            "containment_ios": 0.80,
            "cross_class_iou": 0.70,
            "show_tile_grid": False,
        },
        "crops": {
            "padding": 6,
            "padding_ratio": 0.0,
            "square": False,
            "normalize": True,
            "target_size": 224,
            "image_format": "png",
        },
        "classification": {
            "batch_size": 32,
            "top_k": 3,
            "device": "cpu",
            "accept_threshold": None,
            "review_threshold": None,
            "temperature": None,
        },
        # Steps 5.5 and 6.2. Declared here rather than created on demand: the
        # sidebar writes the 6.2 artifact paths into ``solder_grading`` before
        # anything has rendered the step-7 panel, and an absent section made
        # that a KeyError that killed the whole app on upload.
        "solder": {
            "enabled": True,
            "refine_to_metal": True,
            "split_pins": False,
            "include_body_view": True,
            "terminal_outer_ratio": 0.45,
            "lead_outer_ratio": 0.26,
            "target_size": 128,
        },
        # Hồ sơ CAD / pick-and-place. `PipelineConfig` đã đọc mục này từ trước;
        # chỉ có giao diện là chưa bao giờ lộ nó ra.
        "cad": {
            "path": None,
            "fmt": "auto",
            "units": "mm",
            "side": "top",
        },
        "solder_grading": {
            "enabled": True,
            "model_path": None,
            "manifest_path": None,
        },
    }


def _adopt_active_models() -> None:
    """Nạp sẵn model trong ``models/active/`` khi phiên mới bắt đầu.

    ``models/active/`` được định nghĩa là "cái app tự nạp", nhưng trước đây
    không ai gọi :func:`find_active`, nên mọi phiên đều mở ra với bộ chọn ở
    "— không dùng —" và bước 4 báo đang chạy CV demo. Người dùng phải chọn tay
    đúng cái model đã được đặt làm mặc định.

    Chạy **một lần cho mỗi phiên**, không phải mỗi lần chạy lại. Khác biệt này
    quan trọng: bấm "Gỡ model" rồi mà mỗi rerun lại nạp về thì nút gỡ trông như
    hỏng. Đây đúng là lý do lần trước việc gán sẵn model bị bỏ đi -- xem
    ``tests/test_solder_model_upload.py::test_no_model_artifact_is_seeded_from_disk``.

    Lần này khác ở chỗ: đường dẫn lấy từ :func:`find_active` nên file chắc chắn
    tồn tại, và chỉ ``.onnx`` mới được liệt kê nên không có gì phải xác nhận
    pickle.
    """

    if st.session_state.get("active_models_adopted"):
        return
    st.session_state.active_models_adopted = True

    for slot, (kind, path_key, name_key, manifest_key) in _MODEL_SLOTS.items():
        if st.session_state.get(path_key):
            continue
        entry = find_active(kind)
        if entry is None:
            continue
        # Thiếu manifest thì 6.1/6.2 từ chối nạp; điền vào đây chỉ dời thất bại
        # sang lúc chạy. Detector không cần manifest nên vẫn nạp được.
        if manifest_key is not None and not entry.has_manifest:
            continue
        st.session_state[path_key] = str(entry.model_path)
        st.session_state[name_key] = entry.name
        if manifest_key is not None and entry.manifest_path is not None:
            st.session_state[manifest_key] = str(entry.manifest_path)
            st.session_state[f"{manifest_key.rsplit('_', 1)[0]}_name"] = (
                entry.manifest_path.name
            )
        if slot == "solder":
            grading = st.session_state.config["solder_grading"]
            grading["model_path"] = str(entry.model_path)
            if entry.manifest_path is not None:
                grading["manifest_path"] = str(entry.manifest_path)
        if slot == "component":
            # .onnx không mang pickle nên không có gì phải xác nhận.
            st.session_state.pt_model_trusted = (
                entry.model_path.suffix.lower() != ".pt"
            )


def _init_state() -> None:
    # No model is seeded from disk, the detector included. Every artifact enters
    # through the sidebar uploader, which is the only thing that records a
    # digest and that decides ``pt_model_trusted``; a path planted here would
    # skip both, could not be restored after "Gỡ model", and named a file the
    # repo no longer has now that the detectors live under kaggle/ver*.
    defaults: dict[str, Any] = {
        "active_step": 0,
        "pending_navigation": None,
        "input_image": None,
        "input_name": None,
        "input_digest": None,
        "reference_image": None,
        "reference_name": None,
        "reference_digest": None,
        "calibration_profile_name": None,
        "calibration_profile_digest": None,
        "board_model_path": None,
        "board_model_name": None,
        "board_model_digest": None,
        # Model trong models/active/ được nạp MỘT LẦN mỗi phiên; cờ này nhớ
        # rằng đã nạp rồi, để "Gỡ model" không bị rerun nạp lại.
        "active_models_adopted": False,
        "component_model_path": None,
        "component_model_name": None,
        "component_model_digest": None,
        "classifier_model_path": None,
        "classifier_model_name": None,
        "classifier_model_digest": None,
        "classifier_manifest_path": None,
        "classifier_manifest_name": None,
        "classifier_manifest_digest": None,
        "classifier_manifest_quality_warning": None,
        "pt_model_trusted": False,
        "preprocess_result": None,
        "alignment_result": None,
        "board_result": None,
        "detection_result": None,
        "crops": [],
        "classification_result": None,
        # Bước 5.5/6.2. Thiếu các key này thì mở tab 6.2 là AttributeError ngay,
        # vì Streamlit không tự sinh thuộc tính chưa khai báo.
        "solder_result": None,
        "cad_summary": None,
        "solder_model_path": None,
        "solder_model_name": None,
        "solder_model_digest": None,
        "solder_manifest_path": None,
        "solder_manifest_name": None,
        "solder_manifest_digest": None,
        "inspection_recipe": None,
        "inspection_run": None,
        "inspection_session_id": uuid4().hex,
        "statuses": {step: "pending" for step, *_ in STEP_DEFINITIONS},
        "latencies": {},
        "messages": [],
        "last_backend_mode": "CHƯA CHẠY",
        "last_backend_detail": "Import ảnh để bắt đầu.",
        "config": _default_config(),
        "ignored_uploads": {
            "calibration": None,
            "reference": None,
            "board": None,
            "component": None,
            "classifier": None,
            "classifier_manifest": None,
            "solder_model": None,
            "solder_manifest": None,
        },
        # Đánh giá model: chỗ người vận hành ghi nhận model sai ở từng bước.
        # ``feedback_reload_token`` làm mất hiệu lực cache đọc sau mỗi lần ghi.
        "feedback_reload_token": uuid4().hex,
        "feedback_canvas": None,        # ảnh thu nhỏ để bấm, dựng 1 lần/board
        "feedback_canvas_key": None,    # digest:kích thước nó được dựng cho
        # BOM: hợp đồng lắp ráp. ``None`` = chưa nạp, và mọi đối chiếu bị bỏ qua.
        "bom": None,
        "bom_name": None,
        "bom_complete": True,
        # Hồ sơ board: CAD / pick-and-place. `cad_summary` là kết quả hợp nhất
        # ở bước 6.2, không phải file nguồn — nên tên riêng cho nguồn.
        "cad_name": None,
        "cad_digest": None,
        "cad_components": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    _adopt_active_models()


def _load_css() -> None:
    css_path = Path(__file__).with_name("assets") / "styles.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _decode_image(data: bytes) -> np.ndarray:
    if not data:
        raise ValueError("Ảnh rỗng.")
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError(
            f"Ảnh vượt giới hạn {MAX_IMAGE_BYTES // (1024 * 1024)} MB cho mỗi file."
        )
    buffer = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Không đọc được ảnh. Hãy thử PNG/JPG/BMP/TIFF hợp lệ.")
    if int(image.shape[0]) * int(image.shape[1]) > MAX_IMAGE_PIXELS:
        raise ValueError(
            "Ảnh vượt giới hạn 50 megapixel. Hãy resize hoặc chia ảnh trước khi import."
        )
    return image


def _source_resolution_issue(image: np.ndarray) -> str | None:
    """Return a user-facing quality gate error for a complete-board import."""

    height, width = (int(value) for value in image.shape[:2])
    long_side, short_side = max(width, height), min(width, height)
    pixel_count = width * height
    if (
        long_side < MIN_SOURCE_LONG_SIDE
        or short_side < MIN_SOURCE_SHORT_SIDE
        or pixel_count < MIN_SOURCE_PIXELS
    ):
        return (
            f"Ảnh {width} × {height}px ({pixel_count / 1_000_000:.2f} MP) không đạt "
            f"ngưỡng ảnh toàn PCB tối thiểu {MIN_SOURCE_LONG_SIDE} × "
            f"{MIN_SOURCE_SHORT_SIDE}px ({MIN_SOURCE_PIXELS / 1_000_000:.2f} MP). "
            "Pipeline đã khóa để tránh bỏ sót linh kiện nhỏ. Hãy chụp hoặc gửi "
            "ảnh khác có độ phân giải cao hơn; không nội suy/upscale ảnh cũ."
        )
    return None


def _require_source_resolution(image: np.ndarray) -> None:
    """Raise only when the gate is enforced; otherwise the caller warns instead."""

    if not ENFORCE_SOURCE_RESOLUTION:
        return
    issue = _source_resolution_issue(image)
    if issue:
        raise ValueError(issue)


def _safe_name(name: str, fallback: str = "asset") -> str:
    stem = Path(name).stem
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")
    return cleaned[:80] or fallback


def _materialize_upload(name: str, data: bytes) -> str:
    """Persist an uploaded model in a deterministic OS temp location."""

    suffix = Path(name).suffix.lower()
    digest = _digest(data)
    upload_dir = Path(tempfile.gettempdir()) / "aoi-pcb-workbench" / "models"
    upload_dir.mkdir(parents=True, exist_ok=True)
    output_path = upload_dir / f"{_safe_name(name, 'model')}-{digest[:12]}{suffix}"
    if not output_path.exists() or output_path.stat().st_size != len(data):
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=".model-upload-",
                suffix=".tmp",
                dir=upload_dir,
                delete=False,
            ) as stream:
                stream.write(data)
                temporary_path = Path(stream.name)
            temporary_path.replace(output_path)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink(missing_ok=True)
    return str(output_path)


def _invalidate_after(step: int) -> None:
    result_keys = {
        0: "input_image",
        1: "preprocess_result",
        2: "alignment_result",
        3: "board_result",
        4: "detection_result",
        5: "crops",
        6: "classification_result",
        7: "solder_result",
        # Golden Inspection kiểm CHÍNH tấm ảnh của bước 0, nên ảnh mới thì kết
        # quả cũ là kết quả của board khác. `inspection_recipe` KHÔNG nằm ở đây:
        # recipe dựng từ ảnh Golden riêng và sống lâu hơn từng board.
        8: "inspection_run",
    }
    # Đi theo STEP_ORDER, không theo `range()`: từ khi Golden Inspection thành
    # bước 3.5 thì "các bước sau" không còn trùng với "chỉ số lớn hơn". Một số
    # cứng ở đây từng làm bước 6.2 vô hình với việc reset trạng thái.
    position = STEP_ORDER.index(step)
    for candidate in STEP_ORDER[position + 1:]:
        st.session_state[result_keys[candidate]] = [] if candidate == 5 else None
        st.session_state.statuses[candidate] = "pending"
        st.session_state.latencies.pop(candidate, None)


def _set_source(name: str, data: bytes) -> None:
    digest = _digest(data)
    if digest == st.session_state.input_digest:
        return
    image = _decode_image(data)
    _require_source_resolution(image)
    st.session_state.input_image = image
    st.session_state.input_name = name
    st.session_state.input_digest = digest
    st.session_state.statuses[0] = "done"
    st.session_state.inspection_run = None
    _invalidate_after(0)
    st.session_state.messages.append(f"Đã nạp ảnh: {name}")


def _set_reference(upload: Any) -> None:
    if upload is None:
        return
    data = upload.getvalue()
    digest = _digest(data)
    if digest == st.session_state.ignored_uploads.get("reference"):
        return
    if digest == st.session_state.reference_digest:
        return
    st.session_state.reference_image = _decode_image(data)
    st.session_state.reference_name = upload.name
    st.session_state.reference_digest = digest
    st.session_state.ignored_uploads["reference"] = None
    st.session_state.inspection_recipe = None
    st.session_state.inspection_run = None
    _invalidate_after(1)
    st.session_state.messages.append(f"Đã nạp Golden Image: {upload.name}")


def _set_calibration_profile(upload: Any) -> None:
    if upload is None:
        return
    data = upload.getvalue()
    if not data:
        raise ValueError("Calibration profile rỗng.")
    if len(data) > MAX_CALIBRATION_PROFILE_BYTES:
        raise ValueError("Calibration profile vượt giới hạn 256 KB.")
    digest = _digest(data)
    if digest == st.session_state.ignored_uploads.get("calibration"):
        return
    if digest == st.session_state.calibration_profile_digest:
        return
    from aoi_pipeline import CameraCalibrationProfile

    profile = CameraCalibrationProfile.from_json(data)
    preprocess_config = st.session_state.config["preprocess"]
    preprocess_config["calibration_profile"] = profile.to_dict()
    preprocess_config["undistort"] = True
    st.session_state.calibration_profile_name = upload.name
    st.session_state.calibration_profile_digest = digest
    st.session_state.ignored_uploads["calibration"] = None
    _invalidate_after(0)
    st.session_state.messages.append(f"Đã nạp camera calibration: {upload.name}")


def _remove_calibration_profile() -> None:
    st.session_state.ignored_uploads["calibration"] = (
        st.session_state.calibration_profile_digest
    )
    st.session_state.calibration_profile_name = None
    st.session_state.calibration_profile_digest = None
    preprocess_config = st.session_state.config["preprocess"]
    preprocess_config["calibration_profile"] = None
    preprocess_config["undistort"] = False
    _invalidate_after(0)


def _set_model(upload: Any, kind: str) -> None:
    if upload is None:
        return
    data = upload.getvalue()
    digest = _digest(data)
    if digest == st.session_state.ignored_uploads.get(kind):
        return
    digest_key = f"{kind}_model_digest"
    if digest == st.session_state[digest_key]:
        return
    path = _materialize_upload(upload.name, data)
    st.session_state[f"{kind}_model_path"] = path
    st.session_state[f"{kind}_model_name"] = upload.name
    st.session_state[digest_key] = digest
    st.session_state.ignored_uploads[kind] = None
    if kind == "board":
        _invalidate_after(2)
    elif kind == "component":
        st.session_state.pt_model_trusted = Path(upload.name).suffix.lower() != ".pt"
        st.session_state.inspection_recipe = None
        st.session_state.inspection_run = None
        _invalidate_after(3)
    else:
        _invalidate_after(5)
    st.session_state.messages.append(f"Đã nạp {kind} model: {upload.name}")


def _remove_model(kind: str) -> None:
    st.session_state.ignored_uploads[kind] = st.session_state[f"{kind}_model_digest"]
    st.session_state[f"{kind}_model_path"] = None
    st.session_state[f"{kind}_model_name"] = None
    st.session_state[f"{kind}_model_digest"] = None
    # Ô chọn nhớ giá trị của nó qua các lần chạy lại. Không xoá thì nó vẫn hiện
    # tên model vừa gỡ, tức giao diện nói một đằng còn trạng thái một nẻo.
    for key in (f"{kind}_model_choice", f"{kind}_model_choice_applied"):
        st.session_state.pop(key, None)
    if kind == "board":
        _invalidate_after(2)
    elif kind == "component":
        st.session_state.pt_model_trusted = False
        st.session_state.inspection_recipe = None
        st.session_state.inspection_run = None
        _invalidate_after(3)
    else:
        _invalidate_after(5)


def _set_classifier_manifest(upload: Any) -> None:
    if upload is None:
        return
    data = upload.getvalue()
    if not data or len(data) > 1024 * 1024:
        raise ValueError("model_manifest.json rỗng hoặc vượt quá 1 MB.")
    digest = _digest(data)
    if digest == st.session_state.ignored_uploads.get("classifier_manifest"):
        return
    try:
        manifest = json.loads(data.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"model_manifest.json không hợp lệ: {exc}") from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != CLASSIFIER_MANIFEST_SCHEMA
    ):
        raise ValueError(
            f"Manifest không đúng schema {CLASSIFIER_MANIFEST_SCHEMA} của bước 6.1."
        )
    quality_warning = _classifier_manifest_quality_warning(manifest)
    if digest == st.session_state.classifier_manifest_digest:
        # Recompute after a hot reload so an already-uploaded manifest also
        # receives quality diagnostics introduced by a newer app version.
        st.session_state.classifier_manifest_quality_warning = quality_warning
        return
    path = _materialize_upload(upload.name, data)
    st.session_state.classifier_manifest_path = path
    st.session_state.classifier_manifest_name = upload.name
    st.session_state.classifier_manifest_digest = digest
    st.session_state.classifier_manifest_quality_warning = quality_warning
    st.session_state.ignored_uploads["classifier_manifest"] = None
    _invalidate_after(5)
    st.session_state.messages.append(f"Đã nạp classifier manifest: {upload.name}")


def _remove_classifier_manifest() -> None:
    st.session_state.ignored_uploads["classifier_manifest"] = (
        st.session_state.classifier_manifest_digest
    )
    st.session_state.classifier_manifest_path = None
    st.session_state.classifier_manifest_name = None
    st.session_state.classifier_manifest_digest = None
    st.session_state.classifier_manifest_quality_warning = None
    _invalidate_after(5)


def _classifier_manifest_quality_warning(manifest: Mapping[str, Any]) -> str | None:
    metrics = manifest.get("metrics")
    if not isinstance(metrics, Mapping):
        return None
    failures: list[str] = []
    for key, minimum in (("accuracy", 0.50), ("weighted_f1", 0.50)):
        value = metrics.get(key)
        if isinstance(value, (int, float)) and float(value) < minimum:
            failures.append(f"{key}={float(value):.3f} < {minimum:.2f}")
    if not failures:
        return None
    return (
        "Classifier artifact không đạt quality gate ("
        + "; ".join(failures)
        + "). Kết quả sẽ chủ yếu review/unknown; cần retrain trước production."
    )


def _pt_model_blocked() -> bool:
    name = st.session_state.component_model_name or ""
    return (
        st.session_state.component_model_path is not None
        and Path(name).suffix.lower() == ".pt"
        and not bool(st.session_state.pt_model_trusted)
    )


@st.cache_resource(show_spinner=False, max_entries=1)
def _cached_bridge(
    config_json: str,
    component_model_path: str | None,
    board_model_path: str | None,
    classifier_model_path: str | None,
    classifier_manifest_path: str | None,
) -> PipelineBridge:
    config = json.loads(config_json)
    component_config = dict(config.get("components", {}))
    # The core facade keeps CV proposal and trained-model settings separate;
    # the UI intentionally exposes one coherent detector control panel.
    config["cv_detector"] = {
        "min_area_ratio": component_config.get("min_area_ratio"),
        "max_area_ratio": component_config.get("max_area_ratio"),
        "max_detections": component_config.get("max_candidates"),
        "nms_iou_threshold": component_config.get("iou"),
    }
    config["model_detector"] = {
        "confidence": component_config.get("confidence"),
        "iou": component_config.get("iou"),
        "max_detections": component_config.get("max_candidates"),
        "device": None if component_config.get("device") == "auto" else component_config.get("device"),
        "end2end": component_config.get("end2end"),
    }
    config["tiling"] = {
        "mode": component_config.get("tiling_mode", "auto"),
        "tile_size": component_config.get("tile_size", 1280),
        "min_tile_size": component_config.get("min_tile_size", 640),
        "detail_window_ratio": component_config.get("detail_window_ratio", 0.64),
        "overlap_ratio": component_config.get("tile_overlap", 0.20),
        "auto_trigger_scale": component_config.get("tile_trigger_scale", 1.25),
        "include_full_image": component_config.get("full_image_pass", True),
        "detail_confidence": component_config.get("tile_confidence", 0.20),
        "detail_class_confidence": {
            "led": component_config.get("tile_led_confidence", 0.35),
        },
        "merge_iou_threshold": component_config.get("merge_iou", 0.45),
        "seam_ios_threshold": component_config.get("seam_ios", 0.50),
        "containment_ios_threshold": component_config.get("containment_ios", 0.80),
        "cross_class_iou_threshold": component_config.get("cross_class_iou", 0.70),
    }
    config.setdefault("models", {})
    config["models"]["board_path"] = board_model_path
    config["models"]["component_path"] = component_model_path
    return PipelineBridge(
        config=config,
        model_path=component_model_path,
        board_model_path=board_model_path,
        classifier_model_path=classifier_model_path,
        classifier_manifest_path=classifier_manifest_path,
    )


def _engine_config(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The config as the engine should see it, not as the sidebar recorded it.

    ``create_solder_classifier`` refuses half a contract: a 6.2 model whose
    class order is unknown would have to be guessed, and guessing wrong maps
    every defect onto ``good``. That refusal is raised inside
    ``AOIPipeline.__init__``, so it does not fail step 6.2 alone -- it leaves
    the bridge with no engine at all and quietly drops step 4 back to the CV
    demo. Hold an incomplete pair back until the user supplies the other half,
    the same way the 6.1 classifier pair is held back below.
    """

    config = dict(st.session_state.config if config is None else config)
    grading = dict(config.get("solder_grading") or {})
    if not (grading.get("model_path") and grading.get("manifest_path")):
        grading["model_path"] = None
        grading["manifest_path"] = None
    config["solder_grading"] = grading
    return config


def _get_bridge() -> PipelineBridge:
    config_json = json.dumps(_engine_config(), sort_keys=True, ensure_ascii=False)
    classifier_ready = bool(
        st.session_state.classifier_model_path
        and st.session_state.classifier_manifest_path
    )
    bridge = _cached_bridge(
        config_json,
        st.session_state.component_model_path,
        st.session_state.board_model_path,
        st.session_state.classifier_model_path if classifier_ready else None,
        st.session_state.classifier_manifest_path if classifier_ready else None,
    )
    st.session_state.last_backend_mode = bridge.backend_mode
    st.session_state.last_backend_detail = bridge.backend_detail
    return bridge


def _analysis_image() -> np.ndarray | None:
    alignment = st.session_state.alignment_result
    if isinstance(alignment, StageResult):
        return alignment.image
    preprocess = st.session_state.preprocess_result
    if isinstance(preprocess, StageResult):
        return preprocess.image
    return st.session_state.input_image


def _analysis_coordinate_space() -> dict[str, Any]:
    """Describe the pixel canvas used by board/component bounding boxes."""

    image = _analysis_image()
    if isinstance(st.session_state.alignment_result, StageResult):
        stage, image_role = 2, "aligned"
    elif isinstance(st.session_state.preprocess_result, StageResult):
        stage, image_role = 1, "preprocessed"
    else:
        stage, image_role = 0, "input"
    return {
        "id": "analysis_image_pixels",
        "stage": stage,
        "image_role": image_role,
        "width": int(image.shape[1]) if image is not None else None,
        "height": int(image.shape[0]) if image is not None else None,
        "origin": "top_left",
        "bbox_format": "xyxy",
        "right_bottom": "exclusive",
    }


def _mode_to_status(mode: str) -> str:
    normalized = mode.upper()
    if normalized == "SKIPPED":
        return "skipped"
    if "FALLBACK" in normalized:
        return "warning"
    if "DEMO" in normalized:
        return "demo"
    return "done"


def _record_stage(step: int, result: StageResult) -> None:
    st.session_state.statuses[step] = _mode_to_status(result.mode)
    elapsed = result.metrics.get("elapsed_ms")
    if elapsed is not None:
        st.session_state.latencies[step] = float(elapsed)
    if result.message:
        st.session_state.messages.append(f"Bước {step}: {result.message}")
    if step == 4:
        # Detection source is more authoritative than merely having a model
        # filename selected.  This prevents OpenCV candidates from receiving a
        # MODEL badge when the backend fell back internally.
        st.session_state.last_backend_mode = result.mode
        st.session_state.last_backend_detail = result.message


def _execute_preprocess(bridge: PipelineBridge) -> StageResult:
    source = st.session_state.input_image
    if source is None:
        raise RuntimeError("Chưa có ảnh đầu vào.")
    _require_source_resolution(source)
    st.session_state.statuses[1] = "running"
    result = bridge.preprocess(source)
    st.session_state.preprocess_result = result
    _invalidate_after(1)
    _record_stage(1, result)
    return result


def _execute_alignment(bridge: PipelineBridge) -> StageResult:
    source = (
        st.session_state.preprocess_result.image
        if isinstance(st.session_state.preprocess_result, StageResult)
        else st.session_state.input_image
    )
    if source is None:
        raise RuntimeError("Chưa có ảnh đầu vào.")
    st.session_state.statuses[2] = "running"
    reference = st.session_state.reference_image
    # Keep the current board and Golden Image in the same image domain.  The
    # full facade does this in ``AOIPipeline.run`` as well; direct UI stage
    # execution needs to mirror it explicitly.
    if reference is not None and isinstance(st.session_state.preprocess_result, StageResult):
        reference = bridge.preprocess(reference).image
    result = bridge.align(source, reference=reference)
    st.session_state.alignment_result = result
    _invalidate_after(2)
    _record_stage(2, result)
    return result


def _skip_alignment(_: PipelineBridge) -> StageResult:
    source = (
        st.session_state.preprocess_result.image
        if isinstance(st.session_state.preprocess_result, StageResult)
        else st.session_state.input_image
    )
    if source is None:
        raise RuntimeError("Chưa có ảnh đầu vào.")
    result = StageResult(
        image=source.copy(),
        mode="SKIPPED",
        message="Người dùng chọn bỏ qua căn chỉnh; Golden Image vẫn được giữ trong phiên.",
        metrics={"elapsed_ms": 0.0},
    )
    st.session_state.alignment_result = result
    _invalidate_after(2)
    _record_stage(2, result)
    return result


def _execute_board(bridge: PipelineBridge) -> BoardResult:
    source = _analysis_image()
    if source is None:
        raise RuntimeError("Chưa có ảnh đầu vào.")
    st.session_state.statuses[3] = "running"
    result = bridge.detect_board(source)
    st.session_state.board_result = result
    _invalidate_after(3)
    _record_stage(3, result)
    return result


def _execute_components(bridge: PipelineBridge) -> DetectionResult:
    source = _analysis_image()
    if source is None:
        raise RuntimeError("Chưa có ảnh đầu vào.")
    if _pt_model_blocked():
        raise RuntimeError("Cần xác nhận tin cậy file .pt trong sidebar trước khi nạp model.")
    st.session_state.statuses[4] = "running"
    result = bridge.detect_components(source, board_region=st.session_state.board_result)
    st.session_state.detection_result = result
    _invalidate_after(4)
    _record_stage(4, result)
    return result


def _execute_crops(bridge: PipelineBridge) -> list[CropRecord]:
    source = _analysis_image()
    detection_result = st.session_state.detection_result
    if source is None:
        raise RuntimeError("Chưa có ảnh đầu vào.")
    if not isinstance(detection_result, DetectionResult):
        raise RuntimeError("Chưa có kết quả bước 4.")
    st.session_state.statuses[5] = "running"
    started = datetime.now(timezone.utc)
    crops = bridge.make_crops(
        source,
        detection_result.detections,
        **st.session_state.config["crops"],
    )
    elapsed_ms = (datetime.now(timezone.utc) - started).total_seconds() * 1000
    st.session_state.crops = crops
    _invalidate_after(5)
    st.session_state.statuses[5] = "demo" if "DEMO" in detection_result.mode.upper() else "done"
    st.session_state.latencies[5] = round(elapsed_ms, 2)
    st.session_state.messages.append(f"Bước 5: tạo {len(crops)} crop.")
    return crops


def _execute_classification(bridge: PipelineBridge) -> ClassificationResult:
    crops: list[CropRecord] = st.session_state.crops
    if not crops:
        raise RuntimeError("Chưa có crop từ bước 5.")
    if not st.session_state.classifier_model_path or not st.session_state.classifier_manifest_path:
        raise RuntimeError(
            "Chưa nạp đủ best.onnx và model_manifest.json do notebook bước 6.1 xuất."
        )
    st.session_state.statuses[6] = "running"
    result = bridge.classify_components(crops)
    st.session_state.classification_result = result
    _record_stage(6, result)
    return result


def _run_stage(step: int, callback: Callable[[PipelineBridge], Any]) -> None:
    try:
        with st.spinner(f"Đang xử lý bước {step}…"):
            callback(_get_bridge())
        st.toast(f"Bước {step} đã xử lý xong.", icon="✅")
    except Exception as exc:
        st.session_state.statuses[step] = "error"
        st.session_state.messages.append(f"Bước {step} lỗi: {type(exc).__name__}: {exc}")
        st.error(f"Bước {step} thất bại: {exc}")


def _run_all() -> None:
    if st.session_state.input_image is None:
        st.error("Hãy import ít nhất một ảnh PCB ở bước 0 trước khi chạy toàn bộ.")
        return
    if _pt_model_blocked():
        st.error("Hãy xác nhận file .pt là đáng tin cậy trong sidebar trước khi chạy.")
        return
    bridge = _get_bridge()
    progress = st.progress(0, text="Khởi tạo pipeline…")
    stages: tuple[tuple[int, str, Callable[[PipelineBridge], Any]], ...] = (
        (1, "Tiền xử lý ảnh", _execute_preprocess),
        (2, "Căn chỉnh PCB", _execute_alignment),
        (3, "Khoanh vùng PCB", _execute_board),
        (4, "Phát hiện linh kiện", _execute_components),
        (5, "Cắt linh kiện", _execute_crops),
    )
    classifier_ready = bool(
        st.session_state.classifier_model_path
        and st.session_state.classifier_manifest_path
    )
    if classifier_ready:
        stages += ((6, "6.1 Phân loại linh kiện", _execute_classification),)
    # 6.2 chạy luôn: nó là bước riêng chứ không còn là tab con của bước 4, và
    # tầng luật chấm được ngay cả khi chưa nạp model nào -- nên "chạy toàn bộ"
    # mà dừng ở 6.1 sẽ để người dùng tưởng bước 6.2 hỏng.
    solder_enabled = bool(st.session_state.config.get("solder", {}).get("enabled", True))
    if solder_enabled:
        stages += ((7, "6.2 Kiểm tra mối hàn", _execute_solder),)
    for index, (step, label, callback) in enumerate(stages, start=1):
        progress.progress((index - 1) / len(stages), text=f"Bước {step} · {label}")
        try:
            callback(bridge)
        except Exception as exc:
            st.session_state.statuses[step] = "error"
            st.session_state.messages.append(f"Bước {step} lỗi: {type(exc).__name__}: {exc}")
            progress.empty()
            st.error(f"Dừng ở bước {step}: {exc}")
            return
    final_step = stages[-1][0]
    titles = {index: title for index, title, *_ in STEP_DEFINITIONS}
    label = titles.get(final_step, str(final_step))
    progress.progress(1.0, text=f"Hoàn tất workflow 0 → {label}")
    st.session_state.active_step = final_step
    st.session_state.pending_navigation = final_step
    st.toast(f"Đã chạy xong workflow đến {label}.", icon="✅")
    if not classifier_ready:
        # After the loop, not before it: step 5 invalidates every later step,
        # which would reset this straight back to "pending". Say it in the
        # navigation too -- "Chờ chạy" on a step the run deliberately passed
        # over reads as a stall.
        st.session_state.statuses[6] = "skipped"
        st.info(
            "Bước 6.1 bị bỏ qua vì chưa có best.onnx và model_manifest.json từ "
            "notebook train; các bước còn lại đã chạy."
        )


def _status_dot(status: str) -> str:
    return {
        "pending": "○",
        "running": "◉",
        "done": "●",
        "demo": "◆",
        "warning": "▲",
        "skipped": "–",
        "error": "!",
    }.get(status, "○")


#: Dấu trạng thái đứng trước tên bước. Ký tự thay cho màu vì sidebar tối và
#: một chấm màu nhỏ ở đó rất khó phân biệt, nhất là với người nhìn màu kém.
_STATUS_GLYPH = {
    "done": "✓",
    "running": "⟳",
    "error": "✕",
    "warning": "!",
    "demo": "~",
    "pending": "·",
}


def _render_stepper() -> None:
    """Danh sách bước — vừa hiện trạng thái vừa bấm được.

    Trước đây sidebar vẽ danh sách này hai lần: một khối HTML có trạng thái
    nhưng không bấm được, và một radio nhạt bên dưới để điều hướng. Người dùng
    phải đọc ở bảng trên rồi tìm lại đúng dòng đó ở bảng dưới, và nửa màn hình
    sidebar mất vào việc lặp lại cùng một thông tin.

    Nút của Streamlit bấm được, nên mỗi bước chỉ còn một dòng. Mô tả ngắn chuyển
    vào tooltip: nó có ích lúc học quy trình và chỉ chiếm chỗ sau đó.
    """

    for step, name, description, code, shown in STEP_DEFINITIONS:
        status = st.session_state.statuses[step]
        active = step == st.session_state.active_step
        glyph = _STATUS_GLYPH.get(status, "·")
        if st.button(
            f"{glyph}  {shown}. {name}",
            key=f"stepnav_{step}",
            width="stretch",
            type="primary" if active else "secondary",
            help=f"{code} — {description}",
        ):
            st.session_state.active_step = step
            st.rerun()


#: Bốn nguồn nói về cùng một board nhưng mỗi nguồn một khía cạnh, và chúng bổ
#: sung chứ không thay thế nhau: Golden nói board TRÔNG thế nào, BOM nói board
#: PHẢI CÓ những gì, CAD/pick-and-place nói mỗi linh kiện NẰM ĐÂU.
_REFERENCE_SOURCES = {
    "golden": (
        "Golden image",
        "Ảnh một board đã biết là tốt. Dùng để căn ảnh ở bước 2 và so sánh ở "
        "bước 3.5.",
    ),
    "bom": (
        "BOM — danh sách linh kiện",
        "Board phải có những linh kiện nào. Dùng để đối chiếu ở bước 4: linh "
        "kiện ở toạ độ BOM không có là một lỗi.",
    ),
    "cad": (
        "CAD / pick-and-place",
        "Toạ độ và góc xoay từng linh kiện. Dùng ở bước 6.2 để đặt ROI chân "
        "hàn theo land thật thay vì suy ra từ hình học.",
    ),
}


def _render_reference_golden() -> None:
    reference_upload = st.file_uploader(
        "Ảnh board chuẩn",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        key="reference_uploader",
    )
    if reference_upload is not None:
        try:
            _set_reference(reference_upload)
        except ValueError as exc:
            st.error(str(exc))
    if st.session_state.reference_name:
        st.success(st.session_state.reference_name)
        if st.button("Gỡ reference", width="stretch"):
            st.session_state.ignored_uploads["reference"] = st.session_state.reference_digest
            st.session_state.reference_image = None
            st.session_state.reference_name = None
            st.session_state.reference_digest = None
            _invalidate_after(1)
            st.rerun()


def _render_reference_bom() -> None:
    bom_upload = st.file_uploader(
        "File BOM (.csv)",
        type=["csv"],
        key="bom_uploader",
        help=(
            "Nhận cả hai dạng: một dòng mỗi linh kiện kèm toạ độ và kích "
            "thước, hoặc một dòng mỗi loại với danh sách designator "
            '("R1, R2, R5") và cột Quantity.'
        ),
    )
    complete = st.checkbox(
        "BOM này liệt kê đủ mọi linh kiện của board",
        key="bom_complete",
        help=(
            "Bật (mặc định): linh kiện tìm thấy ở chỗ BOM không có sẽ bị "
            "báo LỖI — linh kiện thừa, đặt nhầm chỗ, hoặc vật lạ. Tắt nếu "
            "file chỉ liệt kê một phần; khi đó nó chỉ là ghi nhận."
        ),
    )
    if bom_upload is not None:
        try:
            _set_bom(bom_upload, complete)
        except BomError as exc:
            st.error(str(exc))
    bom = st.session_state.bom
    if bom is not None:
        st.success(f"{st.session_state.bom_name} · {len(bom)} linh kiện")
        st.caption(
            ("có toạ độ" if bom.has_positions else "không có toạ độ")
            + (" · đủ board" if bom.complete else " · một phần")
        )
        for warning in bom.warnings[:3]:
            st.caption(f"⚠ {warning}")
        if st.button("Gỡ BOM", width="stretch"):
            _remove_bom()
            st.rerun()


def _render_reference_cad() -> None:
    upload = st.file_uploader(
        "CAD / pick-and-place (.csv, .json, IPC-356)",
        type=["csv", "json", "ipc", "net", "txt"],
        key="cad_uploader",
        help="Nhận file centroid/pick-and-place, danh sách pad, IPC-D-356A hoặc "
             "CAD JSON. Định dạng được đoán từ nội dung, không theo đuôi file.",
    )
    if upload is not None:
        try:
            _set_cad(upload)
        except CadError as exc:
            st.error(str(exc))
    if st.session_state.cad_name:
        st.success(f"{st.session_state.cad_name} · "
                   f"{st.session_state.cad_components} linh kiện")
        if st.button("Gỡ CAD", key="remove_cad", width="stretch"):
            _remove_cad()
            st.rerun()


_RESOURCE_RENDERERS = {
    "golden": _render_reference_golden,
    "bom": _render_reference_bom,
    "cad": _render_reference_cad,
}


def _render_sidebar() -> bool:
    with st.sidebar:
        pending_navigation = st.session_state.pending_navigation
        if pending_navigation is not None:
            st.session_state.active_step = pending_navigation
            st.session_state.pending_navigation = None
        st.markdown(
            """
            <div class="sidebar-brand">
              <div class="brand-mark">AOI</div>
              <div><strong>PCB Workbench</strong><span>LOCAL INSPECTION LAB</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="sidebar-section-label">WORKFLOW · 0 → 6.2</div>',
                    unsafe_allow_html=True)
        _render_stepper()

        st.markdown('<div class="sidebar-section-label">HỒ SƠ BOARD</div>',
                    unsafe_allow_html=True)
        with st.expander("Dữ liệu tham chiếu — board này phải trông thế nào",
                         expanded=False):
            st.caption(
                "Ba nguồn, mỗi nguồn nói một khía cạnh khác nhau về board. Có "
                "cái nào thì bật cái đó — không cái nào bắt buộc, và chúng bổ "
                "sung chứ không thay thế nhau."
            )
            already = [
                key for key, present in (
                    ("golden", bool(st.session_state.reference_name)),
                    ("bom", st.session_state.bom is not None),
                    ("cad", bool(st.session_state.cad_name)),
                ) if present
            ]
            chosen = st.multiselect(
                "Nguồn đang dùng",
                options=list(_REFERENCE_SOURCES),
                default=already,
                format_func=lambda key: _REFERENCE_SOURCES[key][0],
                key="reference_sources",
                help="Bỏ chọn chỉ ẩn ô nạp đi, KHÔNG gỡ dữ liệu đã nạp — dùng "
                     "nút Gỡ cho việc đó.",
            )
            for position, key in enumerate(chosen):
                if position:
                    st.divider()
                st.caption(_REFERENCE_SOURCES[key][1])
                _RESOURCE_RENDERERS[key]()
            if not chosen:
                st.caption("Chưa bật nguồn nào. Đường ống vẫn chạy, chỉ là không "
                           "có gì để đối chiếu.")

        st.markdown('<div class="sidebar-section-label">CAMERA</div>',
                    unsafe_allow_html=True)
        with st.expander("Camera calibration", expanded=False):
            calibration_upload = st.file_uploader(
                "Profile hiệu chỉnh (.json)",
                type=["json"],
                key="calibration_profile_uploader",
                help="Profile tạo bởi scripts/calibrate_camera.py cho đúng camera, lens và tiêu cự.",
            )
            if calibration_upload is not None:
                try:
                    _set_calibration_profile(calibration_upload)
                except ValueError as exc:
                    st.error(str(exc))
            profile = st.session_state.config["preprocess"].get("calibration_profile")
            if st.session_state.calibration_profile_name and isinstance(profile, dict):
                st.success(st.session_state.calibration_profile_name)
                rms = profile.get("rms_reprojection_error")
                summary = f"{profile.get('image_width')}×{profile.get('image_height')}"
                if rms is not None:
                    summary += f" · RMS {float(rms):.4f}px"
                st.caption(summary)
                if st.button("Gỡ calibration profile", width="stretch"):
                    _remove_calibration_profile()
                    st.rerun()

        st.markdown(
            '<div class="pipeline-asset"><span>BOARD ROI</span><b>Contour pipeline</b>'
            '<small>Model hook được để dành cho giai đoạn sau</small></div>',
            unsafe_allow_html=True,
        )

        with st.expander("Model phát hiện linh kiện", expanded=True):
            if _render_model_picker("component"):
                st.rerun()
            component_upload = st.file_uploader(
                "Component detector (.onnx/.pt)",
                type=["onnx", "pt"],
                key="component_model_uploader",
                help="File export sau khi train trên Kaggle. Ưu tiên .onnx để chạy local.",
            )
            if component_upload is not None:
                _set_model(component_upload, "component")
            _render_model_asset("component")

        with st.expander("Model phân loại 6.1", expanded=True):
            if _render_model_picker("classifier"):
                st.rerun()
            classifier_upload = st.file_uploader(
                "Classifier (best.onnx)",
                type=["onnx"],
                key="classifier_model_uploader",
                help="ONNX raw-logit do notebook phân loại trong training/kaggle xuất.",
            )
            if classifier_upload is not None:
                _set_model(classifier_upload, "classifier")
            _render_model_asset("classifier")
            manifest_upload = st.file_uploader(
                "Contract (model_manifest.json)",
                type=["json"],
                key="classifier_manifest_uploader",
                help="Bắt buộc để app biết đúng class order, preprocessing và confidence policy.",
            )
            if manifest_upload is not None:
                try:
                    _set_classifier_manifest(manifest_upload)
                except ValueError as exc:
                    st.error(str(exc))
            if st.session_state.classifier_manifest_name:
                st.success(f"Manifest: {st.session_state.classifier_manifest_name}")
                quality_warning = st.session_state.classifier_manifest_quality_warning
                if quality_warning:
                    st.warning(quality_warning)
                if st.button("Gỡ classifier manifest", width="stretch"):
                    _remove_classifier_manifest()
                    st.rerun()
            else:
                st.caption("Chưa có manifest · bước 6.1 chưa thể chạy")

        with st.expander("Model kiểm tra mối hàn 6.2", expanded=False):
            if _render_model_picker("solder"):
                st.rerun()
            st.caption(
                "Tùy chọn. Không có model thì bước 6.2 vẫn chấm bằng tầng luật đo "
                "hình học; nạp model chỉ thêm một tầng nữa vào hợp nhất."
            )
            solder_upload = st.file_uploader(
                "Model 6.2 (best.onnx)",
                type=["onnx"],
                key="solder_model_uploader",
                help="ONNX raw-logit do notebook pcb_solder_defect_kaggle xuất.",
            )
            if solder_upload is not None:
                try:
                    _set_solder_model(solder_upload)
                except ValueError as exc:
                    st.error(str(exc))
            solder_manifest_upload = st.file_uploader(
                "Contract 6.2 (model_manifest.json)",
                type=["json"],
                key="solder_manifest_uploader",
                help=(
                    "Bắt buộc đi kèm model: nó ghim class order và ngưỡng. Thiếu nó "
                    "thì thứ tự lớp phải đoán, mà đoán sai là mọi lỗi thành 'đạt'."
                ),
            )
            if solder_manifest_upload is not None:
                try:
                    _set_solder_manifest(solder_manifest_upload)
                except ValueError as exc:
                    st.error(str(exc))
            solder_model_name = st.session_state.solder_model_name
            solder_manifest_name = st.session_state.solder_manifest_name
            if solder_model_name:
                st.success(f"Model: {solder_model_name}")
            if solder_manifest_name:
                st.success(f"Manifest: {solder_manifest_name}")
            if solder_model_name and solder_manifest_name:
                st.caption("Đủ cặp · bước 6.2 hợp nhất luật đo với model")
            elif solder_model_name or solder_manifest_name:
                # Deliberately not an error: the run still produces verdicts.
                st.warning(
                    "Mới có một nửa cặp; bước 6.2 vẫn chấm bằng luật đo và chưa "
                    "dùng model cho tới khi có đủ cả hai."
                )
            if solder_model_name or solder_manifest_name:
                if st.button("Gỡ model 6.2", width="stretch"):
                    _remove_solder_model()
                    st.rerun()
            else:
                st.caption("Chưa có model · bước 6.2 chấm bằng luật đo")

        st.markdown('<div class="security-note"><b>Lưu ý model</b><br>.pt có thể chứa pickle. Chỉ mở weight do bạn tự train hoặc nguồn tin cậy; ưu tiên ONNX khi trao đổi.</div>', unsafe_allow_html=True)
        quick_run = st.button(
            "▶  Chạy pipeline 0–6.2",
            type="primary",
            width="stretch",
            disabled=st.session_state.input_image is None or _pt_model_blocked(),
        )
        if st.session_state.input_image is None:
            st.caption("Import ảnh ở bước 0 để bật chạy nhanh.")
        elif _pt_model_blocked():
            st.caption("Xác nhận tin cậy file .pt để bật chạy nhanh.")
        st.markdown(f'<div class="sidebar-version">LOCAL · v{APP_VERSION}</div>', unsafe_allow_html=True)
    return quick_run


#: Which session keys each pipeline stage stores its chosen artifact under, and
#: which folder under ``models/active`` holds its default.
_MODEL_SLOTS = {
    "component": ("detector", "component_model_path", "component_model_name", None),
    "classifier": ("classifier", "classifier_model_path", "classifier_model_name",
                   "classifier_manifest_path"),
    "solder": ("solder", "solder_model_path", "solder_model_name",
               "solder_manifest_path"),
}


def _use_model_entry(slot: str, entry: ModelEntry) -> None:
    """Point one stage at a model already on disk.

    No copy into the temp workbench: the file is already somewhere stable, and
    duplicating a 40 MB artifact per session buys nothing.
    """

    _, path_key, name_key, manifest_key = _MODEL_SLOTS[slot]
    st.session_state[path_key] = str(entry.model_path)
    st.session_state[name_key] = entry.name
    if manifest_key and entry.manifest_path is not None:
        st.session_state[manifest_key] = str(entry.manifest_path)
        st.session_state[f"{manifest_key.rsplit('_', 1)[0]}_name"] = entry.manifest_path.name
    if slot == "solder":
        st.session_state.config["solder_grading"]["model_path"] = str(entry.model_path)
        if entry.manifest_path is not None:
            st.session_state.config["solder_grading"]["manifest_path"] = str(entry.manifest_path)
    if slot == "component":
        # An .onnx carries no pickle, so nothing to confirm; a .pt still does.
        st.session_state.pt_model_trusted = entry.model_path.suffix.lower() != ".pt"
    st.session_state.messages.append(f"Đã chọn model {slot}: {entry.name} ({entry.origin})")


def _render_model_picker(slot: str) -> bool:
    """Choose a model that is already on disk. Returns True if one was picked.

    Uploading works and stays, but it makes someone re-supply the same file
    every session. The models the project ships live in ``models/active`` and
    anything dropped in ``models/library`` shows up here beside them.
    """

    kind, path_key, _, _ = _MODEL_SLOTS[slot]
    entries = discover_models(kind)
    if not entries:
        st.caption(
            f"Không thấy model nào trong `models/active/{kind}/` hay "
            "`models/library/`. Tải lên bên dưới, hoặc bỏ file `.onnx` kèm "
            "`model_manifest.json` vào `models/library/`."
        )
        return False

    current = st.session_state.get(path_key)
    labels = ["— không dùng —"] + [entry.label for entry in entries]
    index = 0
    for offset, entry in enumerate(entries, start=1):
        if current and Path(current) == entry.model_path:
            index = offset
            break
    chosen = st.selectbox(
        "Chọn từ thư mục models/",
        labels,
        index=index,
        key=f"{slot}_model_choice",
        help=(
            "active = model dự án đang dùng · của bạn = models/library/ · "
            "bản cũ = models/archive/, không tự nạp"
        ),
    )

    # Chỉ hành động khi NGƯỜI DÙNG vừa đổi ô chọn.
    #
    # Widget có ``key`` thì Streamlit nhớ giá trị của nó qua các lần chạy lại và
    # bỏ qua ``index``. Nếu chỉ so ô chọn với đường dẫn trong session rồi thấy
    # lệch là áp đặt, thì bất kỳ thứ gì khác đổi đường dẫn đó sẽ bị ô chọn giật
    # về ngay lần chạy lại kế tiếp. Cụ thể: tải model lên bằng nút upload rồi
    # thì lần rerun sau ô chọn vẫn còn nhớ model cũ, và nó **âm thầm vứt bỏ**
    # file vừa tải.
    #
    # Nên mốc so sánh là "lần cuối CHÍNH Ô CHỌN NÀY áp cái gì", không phải
    # trạng thái hiện tại của session.
    applied_key = f"{slot}_model_choice_applied"
    previously_applied = st.session_state.get(applied_key)
    if chosen == previously_applied:
        return False
    st.session_state[applied_key] = chosen

    if chosen == labels[0]:
        return False
    entry = entries[labels.index(chosen) - 1]
    if current and Path(current) == entry.model_path:
        return False
    _use_model_entry(slot, entry)
    return True


def _set_bom(upload: Any, complete: bool) -> None:
    """Nạp BOM từ file người dùng tải lên."""

    data = upload.getvalue()
    if not data:
        raise BomError("File BOM rỗng.")
    path = Path(_materialize_upload(upload.name, data))
    bom = load_bom(path, complete=complete)
    st.session_state.bom = bom
    st.session_state.bom_name = upload.name
    st.session_state.messages.append(
        f"Đã nạp BOM {upload.name}: {len(bom)} linh kiện"
        + (", có toạ độ" if bom.has_positions else ", KHÔNG có toạ độ")
    )


def _set_cad(upload: Any) -> None:
    """Nạp CAD hoặc pick-and-place. Định dạng được đoán từ nội dung.

    Nạp thử ngay lúc này thay vì đợi bước 6.2: một file sai định dạng phát
    hiện lúc bấm upload thì sửa được ngay, còn phát hiện lúc chạy hợp nhất ROI
    thì nó hiện ra như "CAD không khớp" và người dùng đi tìm nhầm chỗ.
    """

    data = upload.getvalue()
    if not data:
        raise CadError("File CAD rỗng.")
    digest = _digest(data)
    if digest == st.session_state.cad_digest:
        return
    path = _materialize_upload(upload.name, data)
    board = load_cad(Path(path))          # ném CadError nếu không đọc được
    st.session_state.config["cad"]["path"] = path
    st.session_state.cad_name = upload.name
    st.session_state.cad_digest = digest
    st.session_state.cad_components = len(board.components)
    _invalidate_after(6)
    st.session_state.messages.append(
        f"Đã nạp {board.source_format}: {upload.name} · {len(board.components)} linh kiện"
    )


def _remove_cad() -> None:
    st.session_state.config["cad"]["path"] = None
    st.session_state.cad_name = None
    st.session_state.cad_digest = None
    st.session_state.cad_components = 0
    _invalidate_after(6)


def _remove_bom() -> None:
    st.session_state.bom = None
    st.session_state.bom_name = None


@st.cache_resource(show_spinner=False)
def _bom_registration(
    bom_signature: str,
    points_mm: tuple[tuple[float, float], ...],
    detection_signature: str,
    _board: Any,
    _detections: Any,
    image_size: tuple[int, int],
) -> Any:
    """Căn BOM vào ảnh bằng chính các detection, một lần cho mỗi bộ đầu vào.

    BOM cho toạ độ mm, detector cho toạ độ pixel; `register_cad` bỏ phiếu
    RANSAC trên các cặp tương ứng để tìm phép biến đổi giữa hai không gian đó.
    Nghĩa là BOM tự căn được ngay ở bước 4, không phải đợi bước 7 nạp CAD.

    RANSAC tốn vài trăm mili giây, và Streamlit chạy lại cả script mỗi lần bấm
    nút. Không cache thì mỗi lần tick một checkbox là căn lại từ đầu.

    Tham số có tiền tố ``_`` không tham gia khoá cache (Streamlit không băm
    được chúng); ``bom_signature`` và ``detection_signature`` mới là khoá.
    """

    del bom_signature, points_mm, detection_signature   # chỉ để làm khoá cache
    try:
        return register_cad(_board, _detections, image_size)
    except CadError:
        return None


def _bom_projection(detections: Sequence[Any]) -> Any | None:
    """Hàm chiếu mm -> pixel cho BOM, hoặc None nếu không căn được.

    Trả None thì đối chiếu tự chuyển sang đếm theo lớp. Đó là hành vi đúng:
    ghép theo toạ độ mà không biết board nằm đâu trong ảnh sẽ cho ra một bảng
    trông rất thuyết phục và sai toàn bộ.
    """

    bom = st.session_state.bom
    image = _analysis_image()
    if bom is None or not bom.has_positions or not detections or image is None:
        return None

    board = bom.to_board_cad()
    if not board.components:
        return None

    points = tuple(
        (float(component.x), float(component.y)) for component in board.components
    )
    detection_signature = _digest(
        "|".join(
            f"{d.detection_id}:{d.label}:{d.bbox.x1:.1f},{d.bbox.y1:.1f}"
            for d in detections
        ).encode("utf-8")
    )
    registration = _bom_registration(
        str(bom.source), points, detection_signature,
        board, list(detections), (image.shape[1], image.shape[0]),
    )
    if registration is None:
        return None
    return registration.to_image


def _bom_findings_frame(findings: Sequence[Any]) -> "pd.DataFrame":
    severity_order = {"error": 0, "warning": 1, "info": 2}
    rows = [
        {
            "mức": {"error": "LỖI", "warning": "cảnh báo", "info": "ghi nhận"}.get(
                item.severity, item.severity),
            "loại": {
                "missing": "thiếu linh kiện",
                "unexpected": "không có trong BOM",
                "class_mismatch": "sai loại",
                "bom_inconsistent": "BOM tự mâu thuẫn",
            }.get(item.kind, item.kind),
            "designator": item.designator or "—",
            "BOM ghi": item.expected_class or "—",
            "camera thấy": item.observed_class or "—",
            "chi tiết": item.message,
        }
        for item in sorted(findings, key=lambda f: severity_order.get(f.severity, 9))
    ]
    return pd.DataFrame(rows)


def _render_bom_reconciliation(detections: Sequence[Any]) -> None:
    """Bảng đối chiếu BOM cho bước 4."""

    bom = st.session_state.bom
    if bom is None:
        _render_empty(
            "Chưa nạp BOM",
            "Tải file BOM ở sidebar để đối chiếu linh kiện tìm được với danh "
            "sách linh kiện board phải có.",
        )
        return
    if not detections:
        _render_empty("Chưa có detection", "Chạy detector trước khi đối chiếu.")
        return

    project = _bom_projection(detections)
    result = reconcile_bom(bom, detections, project)

    if project is None and bom.has_positions:
        st.info(
            "BOM có toạ độ nhưng **chưa căn được vào ảnh** — RANSAC không tìm "
            "đủ cặp linh kiện khớp nhau. Đang đối chiếu bằng cách **đếm theo "
            "lớp**: nói được board thiếu hay thừa mấy con, không chỉ được con "
            "nào. Thường là do detector bỏ sót quá nhiều, hoặc BOM và ảnh khác "
            "mặt board."
        )
    elif project is None:
        st.caption(
            "BOM không có cột toạ độ, nên đang đối chiếu bằng cách đếm theo lớp."
        )

    columns = st.columns(4)
    columns[0].metric("BOM", len(bom))
    columns[1].metric("Khớp", result.stats["matched"])
    columns[2].metric("Thiếu", result.stats["missing"])
    columns[3].metric("Không có trong BOM", result.stats["unexpected"])

    if result.passed:
        st.success("Board khớp BOM: không thiếu linh kiện, không có linh kiện lạ.")
    else:
        st.error(f"{len(result.errors)} lỗi so với BOM.")

    if result.findings:
        st.dataframe(_bom_findings_frame(result.findings),
                     hide_index=True, width="stretch")


# --------------------------------------------------------------------------
# Đánh giá model: người vận hành ghi lại chỗ model sai, ngay trong trang của
# bước đó. Lưu toạ độ chứ không lưu ảnh -- xem aoi_pipeline/model_feedback.py.
# --------------------------------------------------------------------------

#: Bước nào đánh giá model nào, và tiêu đề hiển thị.
_FEEDBACK_STAGES = {
    "detection": (4, "component", "model phát hiện linh kiện"),
    "classification": (6, "classifier", "model phân loại 6.1"),
    "solder": (7, "solder", "model chấm mối hàn 6.2"),
}


@dataclass(frozen=True, slots=True)
class _FeedbackTarget:
    """Một dòng kết quả chọn được, đã làm phẳng.

    Ba bước trả về ba kiểu bản ghi chẳng liên quan gì nhau; hàm render dùng
    chung không nên phải biết cả ba.
    """

    record_id: str
    record_type: str
    display: str
    bbox: tuple[int, int, int, int]
    model_label: str | None = None
    model_decision: str | None = None
    model_probability: float | None = None


def _detection_targets(result: Any) -> list[_FeedbackTarget]:
    if not isinstance(result, DetectionResult):
        return []
    return [
        _FeedbackTarget(
            record_id=item.detection_id, record_type="detection",
            display=f"{item.detection_id} · {item.label}"
                    + (f" {item.confidence:.2f}" if item.confidence is not None else ""),
            bbox=tuple(int(v) for v in item.bbox),
            model_label=item.label, model_probability=item.confidence,
        )
        for item in result.detections
    ]


def _classification_targets(result: Any, crops: Sequence[Any]) -> list[_FeedbackTarget]:
    """``ClassificationRecord`` không mang bbox, phải nối qua ``crop_id``."""

    if not isinstance(result, ClassificationResult):
        return []
    crop_by_id = {crop.crop_id: crop for crop in crops}
    targets = []
    for item in result.classifications:
        crop = crop_by_id.get(item.crop_id)
        if crop is None:
            continue
        targets.append(_FeedbackTarget(
            record_id=item.crop_id, record_type="classification",
            display=f"{item.crop_id} · {item.family} {item.probability:.2f} · {item.decision}",
            bbox=tuple(int(v) for v in crop.bbox),
            model_label=item.family, model_decision=item.decision,
            model_probability=item.probability,
        ))
    return targets


def _solder_targets(result: Any) -> list[_FeedbackTarget]:
    if not isinstance(result, SolderResult):
        return []
    return [
        _FeedbackTarget(
            record_id=item.joint_id, record_type="solder_verdict",
            display=f"{item.joint_id} · {item.label} · {item.decision}",
            bbox=tuple(int(v) for v in item.bbox),
            model_label=item.label, model_decision=item.decision,
            model_probability=item.probability,
        )
        for item in result.verdicts
    ]


def _model_identity(slot: str) -> ModelIdentity:
    """Model nào đang trả lời ở bước này, đủ để nhận ra nó sau khi bị thay.

    Đọc sha256 từ MANIFEST, không phải từ ``*_model_digest``: khoá đó chỉ được
    đặt ở đường upload, còn model nạp từ ``models/active/`` -- nay là mặc định
    -- luôn để nó là None.
    """

    kind, path_key, name_key, _ = _MODEL_SLOTS[slot]
    raw_path = st.session_state.get(path_key)
    if not raw_path:
        return ModelIdentity(slot=slot, kind=kind, loaded=False)

    path = Path(raw_path)
    entry = next(
        (candidate for candidate in discover_models(kind, require_manifest=False)
         if candidate.model_path == path),
        None,
    )
    if entry is None:
        # Model tải lên nằm ở thư mục tạm, không thuộc registry.
        entry = ModelEntry(name=path.name, kind=kind, model_path=path,
                           manifest_path=None, origin="upload")
    summary = entry.summary()
    return ModelIdentity(
        slot=slot, kind=kind, loaded=True,
        name=st.session_state.get(name_key) or entry.name,
        origin=entry.origin, sha256=summary.sha256, version=summary.version,
        architecture=summary.architecture, created=summary.created,
    )


def _feedback_signature() -> str:
    """Chữ ký rẻ để biết log có đổi không: tên + mtime + cỡ, cộng token ghi."""

    try:
        files = sorted(feedback_root().glob("**/*.jsonl"))
        stamp = ";".join(
            f"{item.name}:{item.stat().st_mtime_ns}:{item.stat().st_size}"
            for item in files
        )
    except OSError:
        stamp = "khong-doc-duoc"
    return f"{stamp}|{st.session_state.feedback_reload_token}"


@st.cache_data(show_spinner=False)
def _feedback_entries(signature: str) -> tuple[list[FeedbackEntry], list[str]]:
    del signature                    # chỉ để làm khoá cache
    return load_feedback()


def _feedback_crop(entry: FeedbackEntry) -> np.ndarray | None:
    """Pixel đằng sau một bản ghi, cắt lại tại chỗ.

    Trong phiên thì không cần file nào: khi digest khớp ảnh đang mở và khung
    phân tích vẫn đúng cỡ đã đo, khung đã nằm sẵn trong bộ nhớ. Khi không khớp
    thì trả None chứ **không** cắt đại -- người vận hành nhìn nhầm chỗ còn tệ
    hơn không nhìn gì.
    """

    frame = _analysis_image()
    if frame is None:
        return None
    if entry.source_sha256 != st.session_state.input_digest:
        return None
    if frame.shape[:2] != (entry.analysis_height, entry.analysis_width):
        return None
    x1, y1, x2, y2 = entry.clamped_bbox(frame.shape[1], frame.shape[0])
    patch = frame[y1:y2, x1:x2]
    return patch if patch.size else None


#: Chiều ngang ảnh chọn vùng. Đủ to để bấm trúng một linh kiện 60 px, đủ nhỏ
#: để JPEG của nó chỉ vài chục KB — ảnh này gửi lại mỗi lần rerun.
_FEEDBACK_CANVAS_WIDTH = 760


def _feedback_canvas(
    targets: Sequence[_FeedbackTarget],
    highlight: int | None,
) -> tuple[np.ndarray, float] | None:
    """Ảnh board thu nhỏ có vẽ sẵn mọi box, kèm tỉ lệ thu nhỏ.

    Phần đắt là thu nhỏ khung 14 MP; làm một lần cho mỗi board rồi giữ trong
    session. Mỗi lần bấm chỉ tốn một bản sao ảnh 760 px cộng vài hình chữ nhật.
    """

    frame = _analysis_image()
    if frame is None:
        return None
    key = f"{st.session_state.input_digest}:{frame.shape[0]}x{frame.shape[1]}"
    if st.session_state.feedback_canvas_key != key:
        scale = min(1.0, _FEEDBACK_CANVAS_WIDTH / max(1, frame.shape[1]))
        st.session_state.feedback_canvas = (
            cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            if scale < 1.0 else frame.copy()
        )
        st.session_state.feedback_canvas_key = key

    base = st.session_state.feedback_canvas
    if base is None:
        return None
    scale = base.shape[1] / frame.shape[1]
    canvas = base.copy()
    for index, target in enumerate(targets):
        x1, y1, x2, y2 = (int(round(value * scale)) for value in target.bbox)
        chosen = index == highlight
        cv2.rectangle(canvas, (x1, y1), (x2, y2),
                      (0, 235, 255) if chosen else (150, 150, 150),
                      3 if chosen else 1)
    return canvas, scale


def _feedback_click(key: str, canvas: np.ndarray, scale: float) -> tuple[int, int] | None:
    """Toạ độ vừa bấm, quy về pixel của ảnh phân tích. None nếu chưa bấm mới.

    Component trả toạ độ theo ảnh **hiển thị** kèm luôn chiều ngang hiển thị,
    nên quy đổi là chính xác chứ không phải đoán.

    `unix_time` phân biệt một cú bấm MỚI với giá trị cũ mà component trả lại ở
    mọi lần chạy lại. Không lọc theo nó thì mỗi lần gõ một ký tự bình luận là
    một lần "bấm" lặp lại, và ô chọn nhảy về chỗ cũ.
    """

    value = streamlit_image_coordinates(
        canvas, width=canvas.shape[1], key=key,
        image_format="JPEG", jpeg_quality=80, cursor="crosshair",
    )
    if not value:
        return None
    stamp_key = f"{key}_stamp"
    stamp = value.get("unix_time")
    if stamp is not None and st.session_state.get(stamp_key) == stamp:
        return None                      # giá trị cũ, không phải cú bấm mới
    st.session_state[stamp_key] = stamp
    return _click_to_source(value, canvas.shape[1], scale)


def _click_to_source(
    value: Mapping[str, Any], canvas_width: int, scale: float
) -> tuple[int, int]:
    """Toạ độ bấm -> pixel của ảnh phân tích.

    Hai lần thu nhỏ chồng lên nhau, và quên một trong hai là lệch chỗ:

    1. ảnh phân tích -> canvas (hệ số ``scale``, do ta chọn)
    2. canvas -> kích thước thật trên màn hình (trình duyệt co theo bề rộng cột)

    Component trả kèm bề rộng đã hiển thị, nên bậc thứ hai đo được chứ không
    phải đoán.
    """

    shown_width = float(value.get("width") or canvas_width)
    ratio = canvas_width / max(shown_width, 1.0) / max(scale, 1e-9)
    return int(round(float(value["x"]) * ratio)), int(round(float(value["y"]) * ratio))


def _target_under(targets: Sequence[_FeedbackTarget], point: tuple[int, int]) -> int | None:
    """Box nào nằm dưới điểm vừa bấm.

    Chọn box NHỎ NHẤT chứa điểm: linh kiện nhỏ hay nằm lọt trong box của một
    connector hay IC lớn, và người bấm vào con nhỏ thì ý họ là con nhỏ.
    """

    x, y = point
    best, best_area = None, None
    for index, target in enumerate(targets):
        x1, y1, x2, y2 = target.bbox
        if x1 <= x <= x2 and y1 <= y <= y2:
            area = (x2 - x1) * (y2 - y1)
            if best_area is None or area < best_area:
                best, best_area = index, area
    return best


def _feedback_runtime_mode(stage: str) -> str:
    """Ngữ cảnh chạy lúc ghi, để sau này lọc được ra."""

    if stage == "detection":
        return "MODEL" if st.session_state.component_model_path else "CV DEMO"
    if stage == "solder":
        result = st.session_state.solder_result
        if isinstance(result, SolderResult):
            return "MODEL" if result.graded_by_model else "rules-only"
    return "MODEL" if st.session_state.get(_MODEL_SLOTS[
        _FEEDBACK_STAGES[stage][1]][1]) else "chưa nạp model"


def _render_model_feedback(stage: str, targets: Sequence[_FeedbackTarget]) -> None:
    """Mục "model sai ở đây", đặt cuối trang của chính bước đó.

    Một hàm dùng chung cho cả ba bước: chúng chỉ khác nhau ở danh sách dòng
    chọn được, từ vựng lỗi và model đang bị đánh giá. Ba bản sao sẽ là ba chỗ
    để sửa cùng một lỗi trạng thái widget và bỏ sót hai.
    """

    step, slot, title = _FEEDBACK_STAGES[stage]
    frame = _analysis_image()
    if frame is None:
        return

    identity = _model_identity(slot)
    entries, problems = _feedback_entries(_feedback_signature())
    mine = entries_for_source(entries, st.session_state.input_digest or "", stage=stage)

    st.divider()
    with st.expander(f"🔎 Đánh giá model — ghi nhận chỗ sai ({len(mine)} đã ghi cho ảnh này)"):
        st.caption(
            f"Đang đánh giá **{title}**: {identity.display}. "
            "Bản ghi chỉ giữ toạ độ và ghi chú, ảnh được cắt lại khi cần xem."
        )
        if problems:
            st.caption(f"⚠ {len(problems)} dòng trong log đọc không được, đã bỏ qua.")

        mode = st.radio(
            "Cách đánh dấu",
            ["Bấm vào box bị sai", "Bấm vào chỗ model bỏ sót"],
            horizontal=True, key=f"fb_{stage}_mode",
            help="Cách 1 dùng khi model ĐÃ khoanh nhưng khoanh sai. Cách 2 dùng "
                 "khi model không khoanh gì cả — bấm vào giữa chỗ bị sót.",
        )
        by_click = mode == "Bấm vào box bị sai"

        picked: _FeedbackTarget | None = None
        box: tuple[int, int, int, int] | None = None
        box_size: int | None = None
        height, width = frame.shape[:2]

        if by_click and not targets:
            st.info(
                "Bước này chưa khoanh được gì để mà bấm. Chuyển sang **Bấm vào "
                "chỗ model bỏ sót**."
            )
            return

        if not by_click:
            box_size = st.select_slider(
                "Kích thước ô", options=(32, 48, 64, 96, 128, 192, 256, 384),
                value=96, key=f"fb_{stage}_size",
                help="Chỉnh trước, rồi bấm vào ảnh — chỗ bấm là TÂM ô. Thang nhân "
                     "vì thanh trượt tuyến tính tiêu phần lớn hành trình vào "
                     "những cỡ không ai dùng.",
            )

        selected_key = f"fb_{stage}_selected"
        drawn = targets if by_click else ()
        prepared = _feedback_canvas(drawn, st.session_state.get(selected_key))
        if prepared is None:
            return
        canvas, scale = prepared

        left, right = st.columns([1.6, 1.0], gap="large")
        with left:
            st.caption(
                "Bấm vào box bị sai." if by_click
                else f"Bấm vào giữa chỗ bị sót — ô {box_size}×{box_size} px sẽ đặt quanh điểm bấm."
            )
            point = _feedback_click(f"fb_{stage}_canvas", canvas, scale)
            if point is not None:
                if by_click:
                    found = _target_under(targets, point)
                    st.session_state[selected_key] = found
                    if found is None:
                        st.warning("Chỗ vừa bấm không nằm trong box nào. Bấm lại vào một box.")
                else:
                    st.session_state[f"fb_{stage}_point"] = point
                st.rerun()

        if by_click:
            index = st.session_state.get(selected_key)
            if index is None or index >= len(targets):
                with right:
                    st.info("Chưa chọn box nào. Bấm vào một box trong ảnh bên trái.")
                return
            picked = targets[index]
            box = picked.bbox
        else:
            point = st.session_state.get(f"fb_{stage}_point")
            if point is None:
                with right:
                    st.info("Chưa bấm chỗ nào. Bấm vào ảnh bên trái.")
                return
            half = (box_size or 96) // 2
            box = BoundingBox(
                float(point[0] - half), float(point[1] - half),
                float(point[0] + half), float(point[1] + half),
            ).clamp(width, height).to_int()

        with right:
            x1, y1, x2, y2 = box
            patch = frame[y1:y2, x1:x2]
            if patch.size:
                _show_image(patch, f"Vùng đã chọn · {x2 - x1}×{y2 - y1} px")
            if picked is not None:
                st.caption(f"**{picked.display}**")
            elif box_size is not None:
                st.caption(f"Ô {box_size}×{box_size} px quanh điểm bấm")

        with st.form(f"fb_{stage}_form"):
            kinds = ERROR_KINDS[stage]
            kind_code = st.selectbox(
                "Loại lỗi", [code for code, _ in kinds],
                format_func=lambda code: error_label(stage, code),
            )
            expected = st.text_input(
                "Đáng lẽ phải là", value="",
                help="Nhãn đúng, nếu bạn biết. Đây là thứ biến log thành một tập "
                     "đánh giá dùng được, không chỉ là ghi chú.",
            )
            default_note = (
                f"Model nói: {picked.model_label} "
                f"({picked.model_probability:.2f})" if picked and picked.model_label
                and picked.model_probability is not None else ""
            )
            comment = st.text_area(
                "Ghi chú", value=default_note, max_chars=1000,
                help="Sai như thế nào, và vì sao bạn cho là sai.",
            )
            if st.form_submit_button("Ghi nhận lỗi này", type="primary"):
                space = _analysis_coordinate_space()
                try:
                    entry = FeedbackEntry(
                        stage=stage, step=step, bbox=box, error_kind=kind_code,
                        model=identity,
                        source_name=st.session_state.input_name or "",
                        source_sha256=st.session_state.input_digest or "",
                        analysis_width=int(space["width"] or frame.shape[1]),
                        analysis_height=int(space["height"] or frame.shape[0]),
                        analysis_stage=int(space["stage"]),
                        image_role=str(space["image_role"]),
                        preprocess=preprocess_identity(st.session_state.config.get("preprocess")),
                        origin="result_row" if picked else "magnifier",
                        box_size=box_size,
                        target=FeedbackTargetRef(
                            record_type=picked.record_type if picked else None,
                            record_id=picked.record_id if picked else None,
                            model_label=picked.model_label if picked else None,
                            model_decision=picked.model_decision if picked else None,
                            model_probability=picked.model_probability if picked else None,
                        ),
                        expected_label=expected.strip() or None,
                        comment=comment.strip(),
                        runtime_mode=_feedback_runtime_mode(stage),
                    )
                    written = append_feedback(entry)
                except FeedbackError as exc:
                    st.error(str(exc))
                except OSError as exc:
                    st.error(
                        f"Không ghi được vào {feedback_root()}: {exc}. "
                        "Đặt biến môi trường AOI_FEEDBACK_DIR để ghi ra chỗ khác."
                    )
                else:
                    st.session_state.feedback_reload_token = uuid4().hex
                    st.session_state.messages.append(
                        f"Đã ghi nhận lỗi {stage}: {error_label(stage, kind_code)}"
                    )
                    st.success(f"Đã ghi vào {written.name}.")
                    st.rerun()

        _render_feedback_history(stage, mine, identity)


def _render_feedback_history(
    stage: str, mine: Sequence[FeedbackEntry], identity: ModelIdentity
) -> None:
    """Những gì đã ghi cho ảnh này, kèm nút xem lại crop."""

    if not mine:
        return
    st.markdown("##### Đã ghi nhận cho ảnh này")
    for entry in sorted(mine, key=lambda item: item.recorded_at, reverse=True)[:20]:
        other_model = entry.model.compare_key != identity.compare_key
        head = (
            f"{entry.recorded_at[:16].replace('T', ' ')} · "
            f"**{error_label(entry.stage, entry.error_kind)}**"
        )
        if other_model:
            head += " · ⚠ ghi nhận trên model khác"
        with st.expander(head):
            columns = st.columns([1.0, 1.6])
            with columns[0]:
                patch = _feedback_crop(entry)
                if patch is not None:
                    _show_image(patch)
                else:
                    st.caption(
                        "Không dựng lại được vùng ảnh: ảnh đang mở khác ảnh lúc "
                        f"ghi (digest {entry.source_sha256[:12]}…), hoặc cấu hình "
                        "tiền xử lý đã đổi."
                    )
            with columns[1]:
                st.caption(f"Khung {entry.bbox} · {entry.origin}")
                if entry.expected_label:
                    st.markdown(f"Đáng lẽ là: **{entry.expected_label}**")
                if entry.comment:
                    st.markdown(entry.comment)
                st.caption(f"Model: {entry.model.display} · chạy ở chế độ {entry.runtime_mode}")
    if len(mine) > 20:
        st.caption(f"Hiện 20 bản ghi mới nhất trong tổng số {len(mine)}.")


def _render_model_asset(kind: str) -> None:
    name = st.session_state[f"{kind}_model_name"]
    if not name:
        if kind == "classifier":
            st.caption("Chưa có model · bước 6.1 sẽ chờ, không sinh nhãn giả")
        else:
            st.caption("Chưa có model · sẽ dùng CV demo")
        return
    suffix = Path(name).suffix.lower()
    st.success(f"Đã nạp: {name}")
    if suffix == ".pt":
        st.warning("File .pt có thể chứa pickle; chỉ nạp weight do bạn tự train hoặc nguồn tin cậy.")
        st.checkbox(
            "Tôi xác nhận file .pt này đáng tin cậy",
            key="pt_model_trusted",
        )
    if st.button("Gỡ model", key=f"remove_{kind}_model", width="stretch"):
        _remove_model(kind)
        st.rerun()


def _render_header() -> None:
    image = st.session_state.input_image
    detections = (
        st.session_state.detection_result.detections
        if isinstance(st.session_state.detection_result, DetectionResult)
        else []
    )
    total_latency = sum(float(value) for value in st.session_state.latencies.values())
    dimensions = "—" if image is None else f"{image.shape[1]} × {image.shape[0]}"
    mode = html.escape(str(st.session_state.last_backend_mode))
    mode_class = "model" if mode in {"MODEL", "PIPELINE"} else "demo"
    st.markdown(
        f"""
        <div class="hero">
          <div>
            <div class="eyebrow">AUTOMATED OPTICAL INSPECTION · LOCAL</div>
            <h1>PCB Vision Workbench</h1>
            <p>Kiểm thử trực quan workflow từ ảnh đầu vào đến crop linh kiện.</p>
          </div>
          <div class="mode-pill {mode_class}"><span></span>{mode}</div>
        </div>
        <div class="metric-grid">
          <div class="metric-card"><span>INPUT</span><strong>{dimensions}</strong><small>W × H pixels</small></div>
          <div class="metric-card"><span>DETECTIONS</span><strong>{len(detections)}</strong><small>component regions</small></div>
          <div class="metric-card"><span>CROPS</span><strong>{len(st.session_state.crops)}</strong><small>ready to export</small></div>
          <div class="metric-card"><span>PIPELINE TIME</span><strong>{total_latency:.0f} ms</strong><small>current session</small></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_step_heading(step: int) -> None:
    _, name, description, code, shown = STEP_BY_INDEX[step]
    status = st.session_state.statuses[step]
    st.markdown(
        f"""
        <div class="section-heading">
          <div class="section-index">{code}</div>
          <div><span>BƯỚC {step} / 6.1</span><h2>{html.escape(name)}</h2><p>{html.escape(description)}</p></div>
          <div class="status-chip {status}">{STATUS_LABELS[status]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_empty(title: str, detail: str) -> None:
    st.markdown(
        f"""
        <div class="empty-state">
          <div class="empty-icon">＋</div>
          <h3>{html.escape(title)}</h3>
          <p>{html.escape(detail)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _show_image(image: np.ndarray, caption: str | None = None) -> None:
    st.image(image, channels="BGR", caption=caption, width="stretch")


def _render_result_notice(result: StageResult) -> None:
    if "DEMO" in result.mode.upper() or "FALLBACK" in result.mode.upper():
        st.warning(f"{result.mode} · {result.message}")
    elif result.mode == "SKIPPED":
        st.info(result.message)
    else:
        st.success(f"{result.mode} · {result.message}")


def _render_metrics(metrics: dict[str, Any]) -> None:
    if not metrics:
        return
    items = list(metrics.items())[:5]
    columns = st.columns(len(items))
    for column, (key, value) in zip(columns, items):
        if isinstance(value, float):
            rendered = f"{value:.2f}"
        else:
            rendered = str(value)
        column.metric(key.replace("_", " ").title(), rendered)


def _render_step_zero() -> None:
    _render_step_heading(0)
    st.info("Chế độ hiện tại: **Import ảnh local**. Adapter camera sẽ được nối khi có camera tại lab.")
    uploaded_file = st.file_uploader(
        "Kéo thả ảnh PCB vào đây",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        accept_multiple_files=False,
        key="source_uploads",
        help=(
            "Yêu cầu ảnh toàn PCB tối thiểu 1280×960 px (1,23 MP). "
            "Prototype xử lý một ảnh mỗi lần để giới hạn RAM."
        ),
    )
    if uploaded_file is not None:
        try:
            payload = uploaded_file.getvalue()
            candidate = _decode_image(payload)
            left, right = st.columns([1.65, 1], gap="large")
            with left:
                _show_image(candidate, uploaded_file.name)
            with right:
                size_mb = len(payload) / (1024 * 1024)
                st.markdown("#### Thông tin ảnh")
                st.metric("Kích thước", f"{candidate.shape[1]} × {candidate.shape[0]}")
                st.metric("Dung lượng", f"{size_mb:.2f} MB")
                st.metric("Kênh màu", "BGR · 8-bit")
                resolution_issue = _source_resolution_issue(candidate)
                if resolution_issue:
                    (st.error if ENFORCE_SOURCE_RESOLUTION else st.warning)(resolution_issue)
                blocked = bool(resolution_issue) and ENFORCE_SOURCE_RESOLUTION
                if not blocked and st.button(
                    "Nạp ảnh này vào pipeline", type="primary", width="stretch"
                ):
                    _set_source(uploaded_file.name, payload)
                    st.toast("Đã nạp ảnh vào workspace.", icon="✅")
                    st.rerun()
        except ValueError as exc:
            st.error(str(exc))
    elif st.session_state.input_image is None:
        _render_empty("Chưa có ảnh PCB", "Dùng vùng upload phía trên để bắt đầu workflow 0–5.")

    if st.session_state.input_image is not None:
        st.markdown("#### Input hiện tại")
        active_left, active_right = st.columns([1.7, 1], gap="large")
        with active_left:
            _show_image(st.session_state.input_image, st.session_state.input_name)
        with active_right:
            st.code(st.session_state.input_digest[:16], language=None)
            active_issue = _source_resolution_issue(st.session_state.input_image)
            if active_issue:
                (st.error if ENFORCE_SOURCE_RESOLUTION else st.warning)(active_issue)
            else:
                st.success("Ảnh đã sẵn sàng cho bước 1.")
            active_blocked = bool(active_issue) and ENFORCE_SOURCE_RESOLUTION
            if not active_blocked and st.button("Đi đến bước 1 →", width="stretch"):
                st.session_state.active_step = 1
                st.session_state.pending_navigation = 1
                st.rerun()


def _render_step_one() -> None:
    _render_step_heading(1)
    if st.session_state.input_image is None:
        _render_empty("Thiếu ảnh đầu vào", "Quay lại bước 0 và import một ảnh PCB.")
        return
    config = st.session_state.config["preprocess"]
    control_col, preview_col = st.columns([0.85, 2.15], gap="large")
    with control_col:
        st.markdown("#### Recipe tiền xử lý")
        with st.form("preprocess_form"):
            has_calibration = isinstance(config.get("calibration_profile"), dict)
            undistort = st.checkbox(
                "Sửa méo ống kính",
                value=bool(config.get("undistort", False) and has_calibration),
                disabled=not has_calibration,
                help="Cần tải Camera calibration profile trong sidebar; chạy trước resize.",
            )
            undistort_alpha = st.slider(
                "Giữ vùng biên sau undistort",
                0.0,
                1.0,
                float(config.get("undistort_alpha", 0.0)),
                0.05,
                disabled=not undistort,
                help="0: ít viền đen nhất; 1: giữ tối đa trường nhìn.",
            )
            resize_enabled = st.checkbox("Resize cạnh dài", value=config["resize_enabled"])
            max_side = st.number_input("Cạnh dài tối đa", 640, 8192, int(config["max_side"]), 160)
            denoise = st.selectbox(
                "Khử nhiễu",
                ["None", "Bilateral", "Gaussian", "NLMeans"],
                index=["None", "Bilateral", "Gaussian", "NLMeans"].index(config["denoise"]),
            )
            denoise_strength = st.slider("Mức khử nhiễu", 1, 15, int(config["denoise_strength"]))
            white_balance = st.checkbox("White balance", value=config["white_balance"])
            clahe = st.checkbox("CLAHE", value=config["clahe"])
            clahe_clip = st.slider("CLAHE clip", 1.0, 5.0, float(config["clahe_clip"]), 0.1)
            normalize = st.checkbox("Normalize min/max", value=config["normalize"])
            sharpen = st.slider("Sharpen", 0.0, 1.5, float(config["sharpen"]), 0.05)
            submitted = st.form_submit_button("Áp dụng tiền xử lý", type="primary", width="stretch")
        if submitted:
            config.update(
                {
                    "undistort": undistort,
                    "undistort_alpha": undistort_alpha,
                    "resize_enabled": resize_enabled,
                    "max_side": max_side,
                    "denoise": denoise,
                    "denoise_strength": denoise_strength,
                    "white_balance": white_balance,
                    "clahe": clahe,
                    "clahe_clip": clahe_clip,
                    "normalize": normalize,
                    "sharpen": sharpen,
                }
            )
            _run_stage(1, _execute_preprocess)
    with preview_col:
        before_tab, after_tab, compare_tab = st.tabs(["Before", "After", "So sánh"])
        with before_tab:
            _show_image(st.session_state.input_image, "Ảnh gốc")
        result = st.session_state.preprocess_result
        with after_tab:
            if isinstance(result, StageResult):
                _show_image(result.image, "Sau tiền xử lý")
            else:
                _render_empty("Chưa có output", "Điều chỉnh recipe rồi nhấn Áp dụng tiền xử lý.")
        with compare_tab:
            if isinstance(result, StageResult):
                left, right = st.columns(2)
                with left:
                    _show_image(st.session_state.input_image, "Before")
                with right:
                    _show_image(result.image, "After")
            else:
                st.caption("Chạy bước 1 để xem so sánh.")
    if isinstance(st.session_state.preprocess_result, StageResult):
        _render_result_notice(st.session_state.preprocess_result)
        _render_metrics(st.session_state.preprocess_result.metrics)


def _render_step_two() -> None:
    _render_step_heading(2)
    if st.session_state.input_image is None:
        _render_empty("Thiếu ảnh đầu vào", "Quay lại bước 0 và import một ảnh PCB.")
        return
    source = (
        st.session_state.preprocess_result.image
        if isinstance(st.session_state.preprocess_result, StageResult)
        else st.session_state.input_image
    )
    config = st.session_state.config["alignment"]
    if st.session_state.reference_image is None:
        st.warning("Chưa có Golden Image. Upload reference trong sidebar; nếu chạy nhanh, bước này sẽ được đánh dấu Bỏ qua.")
    controls, preview = st.columns([0.85, 2.15], gap="large")
    with controls:
        st.markdown("#### Feature matching")
        with st.form("alignment_form"):
            st.selectbox(
                "Phương pháp",
                ["ORB + ECC fallback"],
                disabled=True,
                help="Core hiện hỗ trợ ORB/homography và ECC fallback; SIFT chưa được nối.",
            )
            features = st.slider("Số feature tối đa", 500, 8000, int(config["features"]), 500)
            match_ratio = st.slider("Lowe ratio", 0.50, 0.95, float(config["match_ratio"]), 0.01)
            ransac = st.slider("RANSAC threshold", 1.0, 10.0, float(config["ransac_threshold"]), 0.5)
            submitted = st.form_submit_button(
                "Căn chỉnh với reference",
                type="primary",
                width="stretch",
                disabled=st.session_state.reference_image is None,
            )
        if submitted:
            config.update(
                {
                    "method": "ORB",
                    "features": features,
                    "match_ratio": match_ratio,
                    "ransac_threshold": ransac,
                }
            )
            _run_stage(2, _execute_alignment)
        if st.button("Bỏ qua căn chỉnh", width="stretch"):
            _run_stage(2, _skip_alignment)
    with preview:
        if st.session_state.reference_image is not None:
            source_tab, reference_tab, aligned_tab, diff_tab = st.tabs(["Input", "Reference", "Aligned", "Difference"])
        else:
            source_tab, aligned_tab = st.tabs(["Input", "Output"])
            reference_tab = diff_tab = None
        with source_tab:
            _show_image(source, "Input căn chỉnh")
        if reference_tab is not None:
            with reference_tab:
                _show_image(st.session_state.reference_image, st.session_state.reference_name)
        result = st.session_state.alignment_result
        with aligned_tab:
            if isinstance(result, StageResult):
                _show_image(result.image, "Ảnh đã căn chỉnh")
            else:
                _render_empty("Chưa căn chỉnh", "Chạy alignment hoặc chọn bỏ qua để tiếp tục.")
        if diff_tab is not None:
            with diff_tab:
                if isinstance(result, StageResult):
                    reference = st.session_state.reference_image
                    output = result.image
                    if reference.shape[:2] != output.shape[:2]:
                        reference = cv2.resize(reference, (output.shape[1], output.shape[0]))
                    difference = cv2.absdiff(output, reference)
                    difference = cv2.applyColorMap(cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_TURBO)
                    _show_image(difference, "Absolute difference map")
                else:
                    st.caption("Chạy bước 2 để xem difference map.")
    if isinstance(st.session_state.alignment_result, StageResult):
        _render_result_notice(st.session_state.alignment_result)
        _render_metrics(st.session_state.alignment_result.metrics)


def _render_step_three() -> None:
    _render_step_heading(3)
    source = _analysis_image()
    if source is None:
        _render_empty("Thiếu ảnh đầu vào", "Hoàn thành bước 0 trước khi khoanh vùng PCB.")
        return
    config = st.session_state.config["board"]
    st.info(
        "Bước 3 hiện dùng **contour-based PCB localization** trong AOIPipeline "
        "(có full-image fallback), chưa dùng YOLO board detector."
    )
    controls, preview = st.columns([0.8, 2.2], gap="large")
    with controls:
        st.markdown("#### PCB contour locator")
        with st.form("board_form"):
            min_area_ratio = st.slider(
                "Diện tích board tối thiểu",
                0.01,
                0.50,
                float(config["min_area_ratio"]),
                0.01,
                help="Tỷ lệ diện tích contour PCB so với toàn ảnh.",
            )
            min_rectangularity = st.slider(
                "Độ chữ nhật tối thiểu",
                0.10,
                0.95,
                float(config["min_rectangularity"]),
                0.05,
                help="Lọc contour quá méo hoặc không giống mặt board.",
            )
            padding_ratio = st.slider(
                "Padding quanh board",
                0.0,
                0.05,
                float(config["padding_ratio"]),
                0.001,
                format="%.3f",
            )
            submitted = st.form_submit_button("Khoanh vùng PCB", type="primary", width="stretch")
        if submitted:
            config.update(
                {
                    "min_area_ratio": min_area_ratio,
                    "min_rectangularity": min_rectangularity,
                    "padding_ratio": padding_ratio,
                }
            )
            _run_stage(3, _execute_board)
        st.caption("Backend: PCBLocalizer · contour geometry")
    with preview:
        result = st.session_state.board_result
        raw_tab, roi_tab = st.tabs(["Board detection", "PCB crop"])
        with raw_tab:
            _show_image(result.image if isinstance(result, BoardResult) else source, "PCB ROI")
        with roi_tab:
            if isinstance(result, BoardResult):
                x1, y1, x2, y2 = result.bbox
                _show_image(source[y1:y2, x1:x2], f"ROI [{x1}, {y1}, {x2}, {y2}]")
            else:
                _render_empty("Chưa có ROI", "Nhấn Khoanh vùng PCB để tạo board crop.")
    if isinstance(st.session_state.board_result, BoardResult):
        _render_result_notice(st.session_state.board_result)
        _render_metrics(st.session_state.board_result.metrics)


def _detections_frame(detections: list[DetectionRecord]) -> pd.DataFrame:
    rows = []
    for item in detections:
        x1, y1, x2, y2 = item.bbox
        rows.append(
            {
                "id": item.detection_id,
                "class": item.label,
                "confidence": item.confidence,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "width": item.width,
                "height": item.height,
                "source": item.source,
                "pass": item.metadata.get("inference_pass", ""),
                "tile": item.metadata.get("tile_id", ""),
                "owner": item.metadata.get("center_in_tile_ownership", pd.NA),
            }
        )
    frame = pd.DataFrame(rows)
    if "owner" in frame.columns:
        # Full-image detections have no tile owner. A nullable Boolean keeps
        # that missing value without mixing strings and bools in Arrow.
        frame["owner"] = frame["owner"].astype("boolean")
    return frame


def _render_step_four() -> None:
    _render_step_heading(4)
    source = _analysis_image()
    if source is None:
        _render_empty("Thiếu ảnh đầu vào", "Hoàn thành bước 0 trước khi chạy detector.")
        return
    config = st.session_state.config["components"]
    input_image = st.session_state.input_image
    if isinstance(input_image, np.ndarray):
        source_long_side = max(source.shape[:2])
        input_long_side = max(input_image.shape[:2])
        if source_long_side > 0 and input_long_side / source_long_side >= 1.75:
            st.warning(
                f"Ảnh phân tích đã bị thu từ cạnh dài {input_long_side}px xuống "
                f"{source_long_side}px. Tiling không thể khôi phục chi tiết đã mất; "
                "hãy quay lại bước 1, tăng Cạnh dài tối đa hoặc tắt resize."
            )
    has_model = st.session_state.component_model_path is not None
    model_blocked = _pt_model_blocked()
    if has_model:
        if model_blocked:
            st.warning("File .pt chưa được xác nhận tin cậy; detector đang bị khóa.")
        else:
            st.success(f"Model đã chọn: **{st.session_state.component_model_name}** · backend sẽ xác minh khả năng nạp khi chạy.")
    else:
        st.warning("**CV candidate demo – không phải nhận dạng đáng tin cậy.** Upload model export từ Kaggle ở sidebar để nhận diện thật.")
    controls, preview = st.columns([0.8, 2.2], gap="large")
    with controls:
        st.markdown("#### Component detector")
        with st.form("components_form"):
            confidence = st.slider(
                "Confidence (model)",
                0.05,
                0.95,
                float(config["confidence"]),
                0.05,
                disabled=not has_model,
                help=(
                    "Ngưỡng confidence chỉ áp dụng cho model .pt/.onnx. "
                    "CV demo không dùng tham số này."
                ),
            )
            iou = st.slider("IoU / NMS", 0.10, 0.90, float(config["iou"]), 0.05)
            max_candidates = st.slider(
                "Max detections",
                50,
                2000,
                int(config["max_candidates"]),
                50,
                help="Giữ 2000 cho board dày linh kiện; có thể giảm để tăng tốc độ.",
            )
            model_suffix = Path(st.session_state.component_model_name or "").suffix.lower()
            device_options = ["auto", "cpu", "cuda"] if model_suffix == ".pt" else ["auto", "cpu"]
            if config["device"] not in device_options:
                config["device"] = "auto"
            device = st.selectbox(
                "Device",
                device_options,
                index=device_options.index(config["device"]),
                disabled=not has_model,
                help="Chỉ áp dụng cho model đã nạp; ONNX local mặc định dùng CPU.",
            )
            st.markdown("##### Ảnh lớn / adaptive tiling")
            tiling_mode = st.selectbox(
                "Chế độ chia ảnh",
                ["auto", "on", "off"],
                index=["auto", "on", "off"].index(config.get("tiling_mode", "auto")),
                format_func={"auto": "Tự động", "on": "Luôn bật", "off": "Tắt"}.get,
                disabled=not has_model,
                help=(
                    "Auto tự chọn cửa sổ chi tiết từ 640 đến kích thước tile tối đa. "
                    "Ảnh 1000px sẽ dùng tile khoảng 640px thay vì bị bỏ qua."
                ),
            )
            tile_size = st.number_input(
                "Kích thước tile tối đa",
                640,
                2048,
                int(config.get("tile_size", 1280)),
                64,
                disabled=not has_model,
            )
            tile_overlap = st.slider(
                "Overlap giữa tile",
                0.0,
                0.40,
                float(config.get("tile_overlap", 0.20)),
                0.05,
                disabled=not has_model,
            )
            full_image_pass = st.checkbox(
                "Chạy thêm một lượt toàn board",
                value=bool(config.get("full_image_pass", True)),
                disabled=not has_model or tiling_mode == "off",
                help="Giữ khả năng bắt linh kiện lớn; sẽ tăng thời gian inference.",
            )
            tile_confidence = st.slider(
                "Confidence cho detail tile",
                0.05,
                0.95,
                float(config.get("tile_confidence", 0.20)),
                0.05,
                disabled=not has_model or tiling_mode == "off",
                help=(
                    "Nên thấp hơn confidence toàn board để ưu tiên recall linh kiện nhỏ; "
                    "global NMS và classifier sẽ xử lý kết quả tiếp theo."
                ),
            )
            tile_led_confidence = st.slider(
                "Confidence LED trong detail tile",
                0.20,
                0.90,
                float(config.get("tile_led_confidence", 0.35)),
                0.05,
                disabled=not has_model or tiling_mode == "off",
                help=(
                    "LED dễ bị nhầm với mối hàn sáng. Ngưỡng riêng này giữ recall "
                    "cho resistor/capacitor nhưng loại LED confidence thấp."
                ),
            )
            merge_iou = st.slider(
                "IoU gộp detection giữa tile",
                0.10,
                0.90,
                float(config.get("merge_iou", 0.45)),
                0.05,
                disabled=not has_model or tiling_mode == "off",
            )
            seam_ios = st.slider(
                "IoS gộp box bị cắt tại biên tile",
                0.30,
                0.90,
                float(config.get("seam_ios", 0.50)),
                0.05,
                disabled=not has_model or tiling_mode == "off",
                help=(
                    "Chỉ áp dụng cho hai box chạm biên từ hai tile khác nhau. "
                    "Giảm ngưỡng nếu box của cùng linh kiện vẫn bị đếm hai lần."
                ),
            )
            containment_ios = st.slider(
                "IoS loại box nhỏ nằm trong box lớn",
                0.60,
                0.95,
                float(config.get("containment_ios", 0.80)),
                0.05,
                disabled=not has_model or tiling_mode == "off",
                help=(
                    "Áp dụng cho cùng class từ full-image/hai tile khác nhau; "
                    "ưu tiên box đầy đủ không chạm biên."
                ),
            )
            cross_class_iou = st.slider(
                "IoU xung đột khác class",
                0.50,
                0.95,
                float(config.get("cross_class_iou", 0.70)),
                0.05,
                disabled=not has_model or tiling_mode == "off",
                help=(
                    "Hai box khác nhãn nhưng gần như trùng nhau được coi là hai giả thuyết "
                    "cho cùng linh kiện; chỉ giữ box có ưu tiên cao hơn."
                ),
            )
            show_tile_grid = st.checkbox(
                "Hiện lưới tile trên kết quả",
                value=bool(config.get("show_tile_grid", False)),
                disabled=not has_model or tiling_mode == "off",
            )
            submitted = st.form_submit_button(
                "Phát hiện linh kiện" if has_model else "Tạo candidate boxes (demo)",
                type="primary",
                width="stretch",
                disabled=model_blocked,
            )
        if submitted:
            config.update(
                {
                    "confidence": confidence,
                    "iou": iou,
                    "max_candidates": max_candidates,
                    "device": device,
                    "tiling_mode": tiling_mode,
                    "tile_size": int(tile_size),
                    "tile_overlap": tile_overlap,
                    "full_image_pass": full_image_pass,
                    "tile_confidence": tile_confidence,
                    "tile_led_confidence": tile_led_confidence,
                    "merge_iou": merge_iou,
                    "seam_ios": seam_ios,
                    "containment_ios": containment_ios,
                    "cross_class_iou": cross_class_iou,
                    "show_tile_grid": show_tile_grid,
                }
            )
            _run_stage(4, _execute_components)
        st.markdown("##### Model hand-off từ Kaggle")
        st.caption("Cần: `best.onnx` (ưu tiên), `classes.yaml`/`data.yaml`, metrics và ảnh test. Có thể dùng `best.pt` nếu chính bạn export.")
    with preview:
        result = st.session_state.detection_result
        overlay_tab, table_tab, stats_tab, bom_tab = st.tabs(
            ["Detection overlay", "Detection table", "Class stats", "Đối chiếu BOM"]
        )
        with overlay_tab:
            _show_image(result.image if isinstance(result, DetectionResult) else source, "Component detections")
        with table_tab:
            if isinstance(result, DetectionResult) and result.detections:
                frame = _detections_frame(result.detections)
                st.dataframe(
                    frame,
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "confidence": st.column_config.NumberColumn(format="%.3f"),
                    },
                )
            else:
                _render_empty("Chưa có detection", "Chạy detector để xem tọa độ và nhãn.")
        with stats_tab:
            if isinstance(result, DetectionResult) and result.detections:
                frame = _detections_frame(result.detections)
                counts = frame.groupby("class").size().sort_values(ascending=False)
                st.bar_chart(counts)
            else:
                st.caption("Chưa có dữ liệu thống kê.")
        with bom_tab:
            _render_bom_reconciliation(
                result.detections
                if isinstance(result, DetectionResult) else []
            )
    if isinstance(st.session_state.detection_result, DetectionResult):
        _render_result_notice(st.session_state.detection_result)
        _render_metrics(st.session_state.detection_result.metrics)
        annotated = _encode_png(st.session_state.detection_result.image)
        st.download_button(
            "Tải ảnh annotated PNG",
            annotated,
            file_name=f"{_safe_name(st.session_state.input_name or 'pcb')}_annotated.png",
            mime="image/png",
        )
    # Ngoài khối `if` ở trên, có chủ ý: "detector không tìm thấy gì ở đây" là
    # điều đáng ghi nhận nhất, và nó chỉ xảy ra khi KHÔNG có detection nào.
    _render_model_feedback(
        "detection", _detection_targets(st.session_state.detection_result)
    )


def _render_step_five() -> None:
    _render_step_heading(5)
    result = st.session_state.detection_result
    if not isinstance(result, DetectionResult):
        _render_empty("Chưa có detection", "Hoàn thành bước 4 trước khi cắt từng linh kiện.")
        return
    if "DEMO" in result.mode.upper():
        st.warning("Các crop dưới đây kế thừa **CV candidate demo** và chưa phải dataset nhãn đáng tin cậy.")
    config = st.session_state.config["crops"]
    settings, content = st.columns([0.75, 2.25], gap="large")
    with settings:
        st.markdown("#### Crop recipe")
        with st.form("crop_form"):
            padding = st.slider("Padding (px)", 0, 64, int(config["padding"]), 2)
            normalize = st.checkbox("Letterbox vuông", value=config["normalize"])
            target_options = [96, 128, 160, 224, 256, 320, 384]
            current_size = int(config["target_size"])
            target_index = target_options.index(current_size) if current_size in target_options else 3
            target_size = st.selectbox("Output size", target_options, index=target_index)
            submitted = st.form_submit_button("Tạo lại crop", type="primary", width="stretch")
        if submitted:
            config.update({"padding": padding, "normalize": normalize, "target_size": target_size})
            _run_stage(5, _execute_crops)
        st.metric("Detection input", len(result.detections))
        st.metric("Crop output", len(st.session_state.crops))
    with content:
        crops: list[CropRecord] = st.session_state.crops
        if not crops:
            if st.button("Tạo crop từ detection", type="primary"):
                _run_stage(5, _execute_crops)
                st.rerun()
            _render_empty("Chưa có crop", "Nhấn Tạo crop để cắt các detection hiện tại.")
        else:
            gallery_tab, export_tab = st.tabs(["Crop gallery", "Export package"])
            with gallery_tab:
                search = st.text_input("Lọc theo class/id", placeholder="resistor, det_0001…")
                filtered = [
                    crop
                    for crop in crops
                    if not search or search.lower() in f"{crop.crop_id} {crop.label}".lower()
                ]
                st.caption(f"Hiển thị {min(len(filtered), 60)}/{len(filtered)} crop (giới hạn 60 để UI mượt).")
                for offset in range(0, min(len(filtered), 60), 5):
                    columns = st.columns(5)
                    for column, crop in zip(columns, filtered[offset : offset + 5]):
                        with column:
                            _show_image(crop.image)
                            confidence = "—" if crop.confidence is None else f"{crop.confidence:.2f}"
                            st.caption(f"**{crop.label}** · {crop.crop_id}\n\nconf {confidence}")
            with export_tab:
                _render_exports()


def _classifications_frame(items: list[ClassificationRecord]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "crop_id": item.crop_id,
                "detection_id": item.detection_id,
                "family": item.family,
                "probability": item.probability,
                "unknown_score": item.unknown_score,
                "decision": item.decision,
                "detector_hint": item.detector_hint,
                "top_k": " | ".join(
                    f"{score['label']}:{float(score['probability']):.3f}"
                    for score in item.top_k
                ),
                "model_version": item.model_version,
            }
            for item in items
        ]
    )


def _render_classification_content(crops: Sequence[Any]) -> None:
    """Cột phải của bước 6.1, tách khỏi phần bố cục.

    `return` sớm ở đây trước kia thoát khỏi CẢ hàm render bước 6, nên mọi
    thứ đặt sau khối cột không bao giờ được vẽ khi classifier chưa chạy --
    đúng lúc mục đánh giá model cần có mặt nhất.
    """

    result = st.session_state.classification_result
    if not isinstance(result, ClassificationResult):
        _render_empty(
            "Chưa có kết quả phân loại",
            "Nạp model và manifest rồi chạy bước 6.1. Nhãn detector không được dùng thay thế.",
        )
        return
    items = result.classifications
    table_tab, stats_tab, review_tab = st.tabs(
        ["Classification table", "Family stats", "Review queue"]
    )
    with table_tab:
        if items:
            st.dataframe(
                _classifications_frame(items),
                hide_index=True,
                width="stretch",
                column_config={
                    "probability": st.column_config.NumberColumn(format="%.4f"),
                    "unknown_score": st.column_config.NumberColumn(format="%.4f"),
                },
            )
        else:
            st.warning("Model không trả về classification nào.")
    with stats_tab:
        if items:
            frame = _classifications_frame(items)
            left, right = st.columns(2)
            with left:
                st.markdown("##### Theo family")
                st.bar_chart(frame.groupby("family").size().sort_values(ascending=False))
            with right:
                st.markdown("##### Theo quyết định")
                st.bar_chart(frame.groupby("decision").size())
    with review_tab:
        crop_by_id = {crop.crop_id: crop for crop in crops}
        queue = [item for item in items if item.decision != "accept"]
        if not queue:
            st.success("Không có crop trong hàng đợi review/unknown.")
        for offset in range(0, min(len(queue), 60), 4):
            columns = st.columns(4)
            for column, item in zip(columns, queue[offset : offset + 4]):
                with column:
                    crop = crop_by_id.get(item.crop_id)
                    if crop is not None:
                        _show_image(crop.image)
                    st.caption(
                        f"**{item.family}** · {item.decision}\n\n"
                        f"p={item.probability:.3f} · {item.crop_id}"
                    )
    _render_result_notice(result)
    _render_metrics(result.metrics)


def _render_step_six() -> None:
    _render_step_heading(6)
    crops: list[CropRecord] = st.session_state.crops
    if not crops:
        _render_empty("Chưa có crop", "Hoàn thành bước 5 trước khi phân loại linh kiện.")
        return
    detection_result = st.session_state.detection_result
    if isinstance(detection_result, DetectionResult) and "DEMO" in detection_result.mode.upper():
        st.warning(
            "Các crop đầu vào đến từ CV candidate demo. Classifier có thể chạy để thử "
            "luồng, nhưng không được coi kết quả này là AOI production."
        )

    model_ready = bool(
        st.session_state.classifier_model_path
        and st.session_state.classifier_manifest_path
    )
    settings, content = st.columns([0.8, 2.2], gap="large")
    with settings:
        st.markdown("#### Classifier hand-off")
        st.caption("Runtime an toàn chỉ nhận `best.onnx` cùng `model_manifest.json`.")
        st.metric("Crop input", len(crops))
        if not model_ready:
            st.info(
                "Khung bước 6.1 đã sẵn sàng nhưng chưa có model. Hãy chạy notebook "
                "`training/kaggle/pcb_component_classification_kaggle.ipynb`, rồi nạp "
                "hai artifact trong sidebar."
            )
        submitted = st.button(
            "Phân loại linh kiện",
            type="primary",
            width="stretch",
            disabled=not model_ready,
        )
        if submitted:
            _run_stage(6, _execute_classification)
        st.caption(
            "`accept` có thể đi tiếp; `review` cần người kiểm tra; `unknown` không được "
            "ép thành một family đã biết."
        )

    with content:
        _render_classification_content(crops)
    _render_model_feedback(
        "classification",
        _classification_targets(st.session_state.classification_result, crops),
    )


def _encode_png(image: np.ndarray) -> bytes:
    success, buffer = cv2.imencode(".png", image)
    if not success:
        raise ValueError("Không encode được PNG.")
    return buffer.tobytes()


def _manifest() -> dict[str, Any]:
    result = st.session_state.detection_result
    detections = result.detections if isinstance(result, DetectionResult) else []
    board = st.session_state.board_result
    coordinate_space = _analysis_coordinate_space()
    preprocess_config = st.session_state.config.get("preprocess", {})
    calibration_profile = preprocess_config.get("calibration_profile")
    classification_result = getattr(st.session_state, "classification_result", None)
    classifications = (
        classification_result.classifications
        if isinstance(classification_result, ClassificationResult)
        else []
    )
    return {
        "schema_version": "aoi-pcb-workbench/0.3",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "name": st.session_state.input_name,
            "sha256": st.session_state.input_digest,
            "width": int(st.session_state.input_image.shape[1]) if st.session_state.input_image is not None else None,
            "height": int(st.session_state.input_image.shape[0]) if st.session_state.input_image is not None else None,
        },
        "reference": {
            "name": st.session_state.reference_name,
            "sha256": st.session_state.reference_digest,
        },
        "calibration": {
            "name": getattr(st.session_state, "calibration_profile_name", None),
            "sha256": getattr(st.session_state, "calibration_profile_digest", None),
            "enabled": bool(preprocess_config.get("undistort", False)),
            "profile": calibration_profile if isinstance(calibration_profile, dict) else None,
        },
        "models": {
            "board": {
                "name": st.session_state.board_model_name,
                "sha256": st.session_state.board_model_digest,
            },
            "component": {
                "name": st.session_state.component_model_name,
                "sha256": st.session_state.component_model_digest,
            },
            "classifier": {
                "name": getattr(st.session_state, "classifier_model_name", None),
                "sha256": getattr(st.session_state, "classifier_model_digest", None),
                "manifest_name": getattr(
                    st.session_state, "classifier_manifest_name", None
                ),
                "manifest_sha256": getattr(
                    st.session_state, "classifier_manifest_digest", None
                ),
            },
        },
        "backend": {
            "mode": st.session_state.last_backend_mode,
            "detail": st.session_state.last_backend_detail,
        },
        "coordinate_space": coordinate_space,
        "board_bbox": list(board.bbox) if isinstance(board, BoardResult) else None,
        "statuses": {str(key): value for key, value in st.session_state.statuses.items()},
        "latencies_ms": {str(key): value for key, value in st.session_state.latencies.items()},
        "config": st.session_state.config,
        "detections": [
            {
                "id": item.detection_id,
                "label": item.label,
                "confidence": item.confidence,
                "bbox_xyxy": list(item.bbox),
                "coordinate_space": coordinate_space["id"],
                "source": item.source,
                "metadata": item.metadata,
            }
            for item in detections
        ],
        "crop_count": len(st.session_state.crops),
        "classifications": [
            {
                "crop_id": item.crop_id,
                "detection_id": item.detection_id,
                "family": item.family,
                "probability": item.probability,
                "unknown_score": item.unknown_score,
                "decision": item.decision,
                "top_k": item.top_k,
                "detector_hint": item.detector_hint,
                "model_version": item.model_version,
            }
            for item in classifications
        ],
        "classification_count": len(classifications),
        "warning": (
            "CV candidate demo; not reliable recognition."
            if isinstance(result, DetectionResult) and "DEMO" in result.mode.upper()
            else None
        ),
    }


def _detections_csv_bytes() -> bytes:
    result = st.session_state.detection_result
    detections = result.detections if isinstance(result, DetectionResult) else []
    coordinate_space = _analysis_coordinate_space()
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=[
            "id",
            "class",
            "confidence",
            "x1",
            "y1",
            "x2",
            "y2",
            "width",
            "height",
            "source",
            "frame_id",
            "inference_pass",
            "tile_id",
            "touches_tile_border",
            "center_in_tile_ownership",
            "coordinate_space",
            "image_width",
            "image_height",
            "bbox_format",
        ],
    )
    writer.writeheader()
    for item in detections:
        x1, y1, x2, y2 = item.bbox
        writer.writerow(
            {
                "id": _csv_cell(item.detection_id),
                "class": _csv_cell(item.label),
                "confidence": "" if item.confidence is None else f"{item.confidence:.6f}",
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "width": item.width,
                "height": item.height,
                "source": _csv_cell(item.source),
                "frame_id": _csv_cell(item.metadata.get("frame_id", "")),
                "inference_pass": _csv_cell(item.metadata.get("inference_pass", "")),
                "tile_id": _csv_cell(item.metadata.get("tile_id", "")),
                "touches_tile_border": bool(item.metadata.get("touches_tile_border", False)),
                "center_in_tile_ownership": item.metadata.get(
                    "center_in_tile_ownership", ""
                ),
                "coordinate_space": coordinate_space["id"],
                "image_width": coordinate_space["width"],
                "image_height": coordinate_space["height"],
                "bbox_format": "xyxy_right_bottom_exclusive",
            }
        )
    return stream.getvalue().encode("utf-8-sig")


def _csv_cell(value: Any) -> str:
    """Prevent spreadsheet formula execution when CSV is opened interactively."""

    text = str(value)
    return f"'{text}" if text.startswith(("=", "+", "-", "@", "\t", "\r")) else text


def _classifications_csv_bytes() -> bytes:
    result = getattr(st.session_state, "classification_result", None)
    items = result.classifications if isinstance(result, ClassificationResult) else []
    stream = io.StringIO(newline="")
    fieldnames = [
        "crop_id",
        "detection_id",
        "family",
        "probability",
        "unknown_score",
        "decision",
        "detector_hint",
        "top_k",
        "model_version",
    ]
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    for item in items:
        writer.writerow(
            {
                "crop_id": _csv_cell(item.crop_id),
                "detection_id": _csv_cell(item.detection_id),
                "family": _csv_cell(item.family),
                "probability": f"{item.probability:.8f}",
                "unknown_score": f"{item.unknown_score:.8f}",
                "decision": _csv_cell(item.decision),
                "detector_hint": _csv_cell(item.detector_hint or ""),
                "top_k": json.dumps(item.top_k, ensure_ascii=False),
                "model_version": _csv_cell(item.model_version),
            }
        )
    return stream.getvalue().encode("utf-8-sig")


def _build_zip() -> bytes:
    output = io.BytesIO()
    base = _safe_name(st.session_state.input_name or "pcb")
    manifest = _manifest()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        archive.writestr("detections.csv", _detections_csv_bytes())
        archive.writestr("classifications.csv", _classifications_csv_bytes())
        if st.session_state.input_image is not None:
            archive.writestr(f"stages/00_{base}_input.png", _encode_png(st.session_state.input_image))
        calibration_profile = st.session_state.config.get("preprocess", {}).get(
            "calibration_profile"
        )
        if isinstance(calibration_profile, dict):
            archive.writestr(
                "calibration/profile.json",
                json.dumps(calibration_profile, ensure_ascii=False, indent=2),
            )
        preprocess = st.session_state.preprocess_result
        if isinstance(preprocess, StageResult):
            archive.writestr(f"stages/01_{base}_preprocessed.png", _encode_png(preprocess.image))
        alignment = st.session_state.alignment_result
        if isinstance(alignment, StageResult):
            archive.writestr(f"stages/02_{base}_aligned.png", _encode_png(alignment.image))
        board = st.session_state.board_result
        if isinstance(board, BoardResult):
            archive.writestr(f"stages/03_{base}_board_annotated.png", _encode_png(board.image))
            board_source = _analysis_image()
            if board_source is not None:
                x1, y1, x2, y2 = board.bbox
                board_crop = board_source[y1:y2, x1:x2]
                if board_crop.size:
                    archive.writestr(f"stages/03_{base}_board_crop.png", _encode_png(board_crop))
        detections = st.session_state.detection_result
        if isinstance(detections, DetectionResult):
            archive.writestr(f"stages/04_{base}_annotated.png", _encode_png(detections.image))
        for crop in st.session_state.crops:
            filename = f"crops/{_safe_name(crop.label, 'component')}/{crop.crop_id}.png"
            archive.writestr(filename, _encode_png(crop.image))
    return output.getvalue()


def _render_exports() -> None:
    result = st.session_state.detection_result
    if not isinstance(result, DetectionResult):
        return
    manifest_bytes = json.dumps(_manifest(), ensure_ascii=False, indent=2).encode("utf-8")
    csv_bytes = _detections_csv_bytes()
    annotated = _encode_png(result.image)
    zip_bytes = _build_zip()
    base = _safe_name(st.session_state.input_name or "pcb")
    st.markdown("#### Gói kết quả")
    st.caption(
        "ZIP gồm ảnh từng stage, detections, classifications, manifest và toàn bộ crop."
    )
    col_a, col_b = st.columns(2)
    with col_a:
        st.download_button(
            "Tải toàn bộ ZIP",
            zip_bytes,
            file_name=f"{base}_aoi_steps_0_6_1.zip",
            mime="application/zip",
            type="primary",
            width="stretch",
        )
        st.download_button(
            "Tải detections.csv",
            csv_bytes,
            file_name=f"{base}_detections.csv",
            mime="text/csv",
            width="stretch",
        )
    with col_b:
        st.download_button(
            "Tải manifest.json",
            manifest_bytes,
            file_name=f"{base}_manifest.json",
            mime="application/json",
            width="stretch",
        )
        st.download_button(
            "Tải annotated.png",
            annotated,
            file_name=f"{base}_annotated.png",
            mime="image/png",
            width="stretch",
        )
    st.code(
        json.dumps(
            {
                "detections": len(result.detections),
                "crops": len(st.session_state.crops),
                "mode": result.mode,
                "zip_size_mb": round(len(zip_bytes) / (1024 * 1024), 2),
            },
            indent=2,
        ),
        language="json",
    )


def _inspection_recipe_output_dir(*, build_id: str | None = None) -> Path:
    """Return a session/build-unique path; bridge publishes it atomically."""

    digest = st.session_state.reference_digest
    if not digest:
        raise ValueError("Chưa có Golden Image để xác định recipe output.")
    return (
        Path(tempfile.gettempdir())
        / "aoi-pcb-workbench"
        / "inspection-recipes"
        / str(st.session_state.inspection_session_id)
        / digest[:16]
        / (build_id or uuid4().hex)
    )


def _inspection_recipe_asset_names(record: InspectionRecipeRecord) -> list[str]:
    recipe = record.raw
    names = {"recipe.json", str(recipe.golden_asset_path)}
    for anchor in recipe.alignment.anchors:
        names.add(str(anchor.template_path))
        if anchor.mask_path is not None:
            names.add(str(anchor.mask_path))
    for slot in recipe.slots:
        names.update(
            {
                str(slot.template_path),
                str(slot.component_mask_path),
                str(slot.compare_mask_path),
            }
        )
        if slot.ignore_mask_path is not None:
            names.add(str(slot.ignore_mask_path))
    return sorted(names)


def _inspection_recipe_zip_bytes(record: InspectionRecipeRecord) -> bytes:
    output = io.BytesIO()
    root = record.recipe_root.resolve()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative_name in _inspection_recipe_asset_names(record):
            path = (root / relative_name).resolve()
            if root not in path.parents:
                raise ValueError(f"Recipe asset thoát khỏi thư mục gốc: {relative_name}")
            if not path.is_file():
                raise ValueError(f"Thiếu recipe asset: {relative_name}")
            archive.write(path, arcname=relative_name)
    return output.getvalue()


def _inspection_position_rows(result: InspectionResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for slot in result.slots:
        position = slot["position"]
        rows.append(
            {
                "slot_id": slot["slot_id"],
                "slot_status": slot["status"],
                "position_status": position["status"],
                "dx_px": position["dx_px"],
                "dy_px": position["dy_px"],
                "dx_mm": position["dx_mm"],
                "dy_mm": position["dy_mm"],
                "angle_deg": position["angle_deg"],
                "score": position["score"],
                "peak_margin": position["peak_margin"],
                "psr": position["psr"],
                "reason": position["reason"],
            }
        )
    return rows


def _inspection_appearance_rows(result: InspectionResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for slot in result.slots:
        appearance = slot["appearance"]
        rows.append(
            {
                "slot_id": slot["slot_id"],
                "slot_status": slot["status"],
                "appearance_status": appearance["status"],
                "ssim": appearance["ssim"],
                "diff_ratio": appearance["diff_ratio"],
                "edge_diff_ratio": appearance["edge_diff_ratio"],
                "max_blob_area_px": appearance["max_blob_area_px"],
                "anomaly_blob_count": appearance["anomaly_blob_count"],
                "valid_overlap_ratio": appearance["valid_overlap_ratio"],
                "defect_label": appearance["defect_label"],
                "reason": appearance["reason"],
            }
        )
    return rows


def _render_inspection_header() -> None:
    recipe = st.session_state.inspection_recipe
    run = st.session_state.inspection_run
    board_status = run.status.upper() if isinstance(run, InspectionResult) else "—"
    slot_count = recipe.slot_count if isinstance(recipe, InspectionRecipeRecord) else 0
    anchor_count = recipe.anchor_count if isinstance(recipe, InspectionRecipeRecord) else 0
    st.markdown(
        f"""
        <div class="hero">
          <div>
            <div class="eyebrow">GOLDEN RECIPE · POSITION · APPEARANCE</div>
            <h1>Golden Inspection</h1>
            <p>Build recipe từ Golden và kiểm tra board bằng contract Phase 1–5.</p>
          </div>
          <div class="mode-pill model"><span></span>CORE INSPECTOR</div>
        </div>
        <div class="metric-grid">
          <div class="metric-card"><span>RECIPE</span><strong>{'READY' if recipe else '—'}</strong><small>validated assets</small></div>
          <div class="metric-card"><span>SLOTS</span><strong>{slot_count}</strong><small>fixed Golden ROIs</small></div>
          <div class="metric-card"><span>ANCHORS</span><strong>{anchor_count}</strong><small>strict alignment</small></div>
          <div class="metric-card"><span>BOARD</span><strong>{html.escape(board_status)}</strong><small>core decision</small></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_build_recipe_mode() -> None:
    golden = st.session_state.reference_image
    if golden is None:
        _render_empty(
            "Chưa có Golden Image",
            "Nạp Image #1 ở sidebar. Recipe sẽ dùng ảnh native, không letterbox/resize.",
        )
        return
    preview, settings = st.columns([1.25, 1.0], gap="large")
    with preview:
        st.markdown("#### Golden measurement image")
        _show_image(golden, st.session_state.reference_name)
        st.caption(
            f"{golden.shape[1]} × {golden.shape[0]}px · canonical `golden_board_pixels`"
        )
    with settings:
        st.markdown("#### Recipe settings")
        with st.form("build_inspection_recipe_form"):
            identity_col, side_col = st.columns(2)
            with identity_col:
                board_id = st.text_input(
                    "Board ID", value="demo_board", help="Định danh SKU/board của recipe."
                )
            with side_col:
                side = st.selectbox("Side", ["top", "bottom"], index=0)
            scale_x, scale_y = st.columns(2)
            with scale_x:
                pixels_per_mm_x = st.number_input(
                    "Pixels/mm X", min_value=0.001, value=1.0, step=0.1
                )
            with scale_y:
                pixels_per_mm_y = st.number_input(
                    "Pixels/mm Y", min_value=0.001, value=1.0, step=0.1
                )
            calibration_verified = st.checkbox(
                "Calibration này đã được đo và xác minh",
                value=False,
                help="Để trống cho demo. Không check nếu đang dùng 1 px/mm tạm.",
            )
            roi_col, search_col = st.columns(2)
            with roi_col:
                roi_padding_px = st.number_input(
                    "ROI padding (px)", min_value=0, value=12, step=1
                )
            with search_col:
                search_margin_px = st.number_input(
                    "Search margin (px)", min_value=0, value=16, step=1
                )
            tolerance_x, tolerance_y = st.columns(2)
            with tolerance_x:
                max_dx = st.number_input(
                    "Tolerance |dx| (mm)", min_value=0.0, value=0.20, step=0.05
                )
            with tolerance_y:
                max_dy = st.number_input(
                    "Tolerance |dy| (mm)", min_value=0.0, value=0.20, step=0.05
                )
            rotation_label = st.selectbox(
                "Rotation measurement",
                ["Không đo góc", "180° periodic", "360°"],
            )
            max_angle = st.number_input(
                "Tolerance góc (deg)",
                min_value=0.0,
                value=3.0,
                step=0.5,
                disabled=rotation_label == "Không đo góc",
            )
            with st.expander("Appearance thresholds", expanded=False):
                min_ssim = st.number_input(
                    "Min local SSIM", min_value=0.0, max_value=1.0, value=0.88, step=0.01
                )
                max_diff = st.number_input(
                    "Max diff ratio", min_value=0.0, max_value=1.0, value=0.08, step=0.01
                )
                max_edge = st.number_input(
                    "Max edge diff ratio", min_value=0.0, max_value=1.0, value=0.10, step=0.01
                )
                max_blob = st.number_input(
                    "Max anomaly blob (px)", min_value=0, value=45, step=1
                )
                min_overlap = st.number_input(
                    "Min valid overlap", min_value=0.0, max_value=1.0, value=0.88, step=0.01
                )
            submitted = st.form_submit_button(
                "Build Golden Recipe",
                type="primary",
                width="stretch",
                disabled=_pt_model_blocked(),
            )
        if submitted:
            rotation_period = {
                "Không đo góc": None,
                "180° periodic": 180.0,
                "360°": 360.0,
            }[rotation_label]
            if _pt_model_blocked():
                st.error("Cần xác nhận file .pt đáng tin cậy trước khi build Golden Recipe.")
                return
            try:
                with st.spinner("Đang chạy detector và tạo fixed ROI/anchor lossless…"):
                    record = _get_bridge().build_inspection_recipe(
                        golden,
                        _inspection_recipe_output_dir(),
                        board_id=board_id,
                        side=side,
                        pixels_per_mm_x=float(pixels_per_mm_x),
                        pixels_per_mm_y=float(pixels_per_mm_y),
                        calibration_verified=bool(calibration_verified),
                        roi_padding_px=int(roi_padding_px),
                        search_margin_px=int(search_margin_px),
                        max_abs_dx_mm=float(max_dx),
                        max_abs_dy_mm=float(max_dy),
                        max_abs_angle_deg=(
                            None if rotation_period is None else float(max_angle)
                        ),
                        rotation_period_deg=rotation_period,
                        min_ssim=float(min_ssim),
                        max_diff_ratio=float(max_diff),
                        max_edge_diff_ratio=float(max_edge),
                        max_blob_area_px=int(max_blob),
                        min_valid_overlap_ratio=float(min_overlap),
                        allow_trusted_pt=bool(st.session_state.pt_model_trusted),
                    )
                st.session_state.inspection_recipe = record
                st.session_state.inspection_run = None
                st.session_state.messages.append(
                    f"Đã build Golden recipe: {record.slot_count} slots, {record.anchor_count} anchors"
                )
                st.success("Golden recipe đã được tạo và validate đầy đủ asset.")
            except Exception as exc:
                st.error(f"Build recipe thất bại: {exc}")

    record = st.session_state.inspection_recipe
    if isinstance(record, InspectionRecipeRecord):
        st.markdown("#### Recipe contract")
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Slots", record.slot_count)
        col_b.metric("Anchors", record.anchor_count)
        col_c.metric("Rejected", record.rejected_count)
        col_d.metric(
            "Eligibility", "PRODUCTION" if record.production_eligible else "DEMO"
        )
        if not record.production_eligible:
            st.info(
                "Recipe đang ở chế độ demo, thường do calibration chưa verified hoặc detector demo. "
                "Inspect vẫn chạy khi tắt production gate nhưng kết quả không phải acceptance production."
            )
        recipe_json = record.recipe_path.read_bytes()
        recipe_zip = _inspection_recipe_zip_bytes(record)
        left, right = st.columns(2)
        with left:
            st.download_button(
                "Tải recipe package (.zip)",
                recipe_zip,
                file_name="golden_inspection_recipe.zip",
                mime="application/zip",
                type="primary",
                width="stretch",
            )
        with right:
            st.download_button(
                "Tải recipe.json",
                recipe_json,
                file_name="recipe.json",
                mime="application/json",
                width="stretch",
            )
        st.caption(f"Local recipe: {record.recipe_path}")


def _render_inspection_result(result: InspectionResult) -> None:
    if result.status == "pass":
        st.success("BOARD PASS")
    elif result.status == "ng":
        st.error("BOARD NG")
    elif result.status == "review":
        st.warning("BOARD REVIEW")
    else:
        st.error(f"INSPECTION INVALID · {result.raw.reason}")
    _render_result_notice(result)
    _show_image(result.image, "Aligned Golden coordinates · core decision overlay")

    alignment_tab, position_tab, appearance_tab, json_tab = st.tabs(
        ["Alignment", "Position", "Appearance", "Result JSON"]
    )
    with alignment_tab:
        alignment = result.alignment
        columns = st.columns(5)
        columns[0].metric("Status", str(alignment.get("status", "—")).upper())
        columns[1].metric("Matched", alignment.get("matched_anchors", 0))
        columns[2].metric("Inliers", alignment.get("inliers", 0))
        columns[3].metric(
            "Residual px",
            "—" if alignment.get("residual_px") is None else f"{alignment['residual_px']:.4f}",
        )
        columns[4].metric(
            "Canvas overlap",
            "—"
            if alignment.get("canvas_overlap_ratio") is None
            else f"{alignment['canvas_overlap_ratio']:.3f}",
        )
        if alignment.get("reason"):
            st.error(alignment["reason"])
        st.json(alignment, expanded=False)
    with position_tab:
        rows = _inspection_position_rows(result)
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        else:
            st.info("Không có Position result vì inspection đã dừng trước slot stage.")
    with appearance_tab:
        rows = _inspection_appearance_rows(result)
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        else:
            st.info("Không có Appearance result vì inspection đã dừng trước slot stage.")
        if result.extras:
            st.markdown("##### Extra candidates")
            st.dataframe(pd.DataFrame(result.extras), hide_index=True, width="stretch")
    with json_tab:
        st.download_button(
            "Tải inspection_result.json",
            result.json_payload.encode("utf-8"),
            file_name="inspection_result.json",
            mime="application/json",
            type="primary",
        )
        st.code(result.json_payload, language="json")


def _render_inspect_board_mode() -> None:
    record = st.session_state.inspection_recipe
    st.markdown("#### Recipe source")
    recipe_path = st.text_input(
        "Hoặc nhập đường dẫn local tới recipe.json",
        placeholder="/path/to/recipe/recipe.json",
    )
    if st.button("Load và validate recipe", disabled=not recipe_path.strip()):
        try:
            record = _get_bridge().load_inspection_recipe(recipe_path)
            st.session_state.inspection_recipe = record
            st.session_state.inspection_run = None
            st.success("Recipe và toàn bộ lossless assets hợp lệ.")
        except Exception as exc:
            st.error(f"Không load được recipe: {exc}")
    if isinstance(record, InspectionRecipeRecord):
        st.caption(
            f"{record.schema_version} · {record.slot_count} slots · "
            f"{record.anchor_count} anchors · {record.coordinate_space}"
        )
    else:
        st.info("Build recipe ở tab đầu hoặc load một recipe local trước khi inspect.")

    test_image = st.session_state.input_image
    if test_image is None:
        _render_empty("Chưa có ảnh test", "Nạp Image #2 ở sidebar.")
        return
    left, right = st.columns([1.25, 1.0], gap="large")
    with left:
        _show_image(test_image, st.session_state.input_name)
        st.caption(f"Measurement image · {test_image.shape[1]} × {test_image.shape[0]}px")
    with right:
        require_production = st.checkbox(
            "Bật production eligibility gate",
            value=False,
            help="Demo recipe sẽ fail closed nếu bật gate này.",
        )
        st.caption(
            "Detector chỉ cung cấp presence/missing/extra và class hint. Position/Appearance "
            "luôn dùng fixed ROI từ Golden recipe."
        )
        inspect_clicked = st.button(
            "Inspect Board",
            type="primary",
            width="stretch",
            disabled=(
                not isinstance(record, InspectionRecipeRecord) or _pt_model_blocked()
            ),
        )
        if inspect_clicked and isinstance(record, InspectionRecipeRecord):
            if _pt_model_blocked():
                st.error("Cần xác nhận file .pt đáng tin cậy trước khi inspect board.")
                return
            try:
                with st.spinner("Đang strict-align, đo pose và Golden Compare…"):
                    result = _get_bridge().inspect_board(
                        test_image,
                        record,
                        require_production_eligible=bool(require_production),
                        allow_trusted_pt=bool(st.session_state.pt_model_trusted),
                    )
                st.session_state.inspection_run = result
                st.session_state.messages.append(
                    f"Inspection board: {result.status.upper()} · {len(result.slots)} slots"
                )
            except Exception as exc:
                st.error(f"Inspect Board thất bại: {exc}")

    result = st.session_state.inspection_run
    if isinstance(result, InspectionResult):
        _render_inspection_result(result)


def _render_step_eight() -> None:
    """Golden Inspection, nay là một bước của đường ống chứ không phải một
    workspace riêng.

    Nó vốn đã kiểm **chính tấm ảnh của bước 0** (`_render_inspect_board_mode`
    đọc `st.session_state.input_image`), nên tách nó ra một chế độ riêng chỉ
    làm người dùng phải nạp lại đúng những thứ đã nạp: ảnh test là ảnh bước 0,
    ảnh Golden là mục "Golden Image / Reference" ở sidebar, và detector là mục
    model ở sidebar. Ba khối tài nguyên riêng của workspace cũ vì thế đã bỏ
    hẳn chứ không chuyển chỗ.
    """

    _render_step_heading(8)
    _render_inspection_header()
    build_tab, inspect_tab = st.tabs(["Build Recipe", "Inspect Board"])
    with build_tab:
        _render_build_recipe_mode()
    with inspect_tab:
        _render_inspect_board_mode()


def _render_activity_log() -> None:
    with st.expander("Nhật ký phiên xử lý", expanded=False):
        if not st.session_state.messages:
            st.caption("Chưa có hoạt động.")
        else:
            for item in reversed(st.session_state.messages[-20:]):
                st.markdown(f"<div class='log-row'>{html.escape(item)}</div>", unsafe_allow_html=True)
        st.caption(st.session_state.last_backend_detail)


def _render_footer() -> None:
    st.markdown(
        """
        <div class="app-footer">
          <span>AOI PCB WORKBENCH · STEPS 0–6.2 · GOLDEN</span>
          <span>Local session · dữ liệu không được upload ra ngoài bởi UI</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _set_solder_model(upload: Any) -> None:
    """Accept the step-6.2 ONNX. Validation waits for the manifest.

    The two are only meaningful together: an ONNX whose class order is unknown
    cannot be checked, and guessing that order maps every defect onto a pass.
    """

    if upload is None:
        return
    data = upload.getvalue()
    if not data or len(data) > 256 * 1024 * 1024:
        raise ValueError("File model rỗng hoặc vượt quá 256 MB.")
    digest = _digest(data)
    if digest in (
        st.session_state.ignored_uploads.get("solder_model"),
        st.session_state.solder_model_digest,
    ):
        return
    path = _materialize_upload(upload.name, data)
    st.session_state.solder_model_path = path
    st.session_state.solder_model_name = upload.name
    st.session_state.solder_model_digest = digest
    st.session_state.config["solder_grading"]["model_path"] = path
    st.session_state.ignored_uploads["solder_model"] = None
    _invalidate_after(6)
    st.session_state.messages.append(f"Đã nạp model 6.2: {upload.name}")


def _set_solder_manifest(upload: Any) -> None:
    """Accept and validate the step-6.2 contract before the run needs it."""

    if upload is None:
        return
    data = upload.getvalue()
    if not data or len(data) > 1024 * 1024:
        raise ValueError("model_manifest.json rỗng hoặc vượt quá 1 MB.")
    digest = _digest(data)
    if digest in (
        st.session_state.ignored_uploads.get("solder_manifest"),
        st.session_state.solder_manifest_digest,
    ):
        return
    try:
        manifest = json.loads(data.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        st.session_state.ignored_uploads["solder_manifest"] = digest
        raise ValueError(f"model_manifest.json không hợp lệ: {exc}") from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != SOLDER_MANIFEST_SCHEMA
    ):
        st.session_state.ignored_uploads["solder_manifest"] = digest
        raise ValueError(
            f"Manifest không đúng schema {SOLDER_MANIFEST_SCHEMA} của bước 6.2."
        )
    path = _materialize_upload(upload.name, data)
    st.session_state.solder_manifest_path = path
    st.session_state.solder_manifest_name = upload.name
    st.session_state.solder_manifest_digest = digest
    st.session_state.config["solder_grading"]["manifest_path"] = path
    st.session_state.ignored_uploads["solder_manifest"] = None
    _invalidate_after(6)
    st.session_state.messages.append(f"Đã nạp manifest 6.2: {upload.name}")


def _remove_solder_model() -> None:
    for key in (
        "solder_model_path", "solder_model_name", "solder_model_digest",
        "solder_manifest_path", "solder_manifest_name", "solder_manifest_digest",
    ):
        st.session_state[key] = None
    st.session_state.config["solder_grading"]["model_path"] = None
    st.session_state.config["solder_grading"]["manifest_path"] = None
    _invalidate_after(6)
    st.session_state.messages.append(
        "Đã gỡ model 6.2; bước này quay về chấm bằng luật đo."
    )


def _render_solder_settings() -> None:
    """Controls for step 5.5. The bridge is cached on the config JSON, so
    writing the new values is enough to rebuild the engine on the next call."""

    config = st.session_state.config.setdefault("solder", {})
    st.markdown("#### ROI mối hàn")
    st.caption(
        "Box của detector chỉ ôm thân linh kiện. Các ROI dưới đây được suy ra từ "
        "box cộng topology chân của class, không phải do detector tìm ra."
    )
    with st.form("solder_form"):
        enabled = st.checkbox("Bật bước 5.5", value=bool(config.get("enabled", True)))
        split_pins = st.checkbox(
            "Tách từng chân (IC/connector)",
            value=bool(config.get("split_pins", False)),
            help=(
                "Tắt thì mỗi cạnh là một ROI dải. Lỗi bridge nằm giữa hai chân nên "
                "ROI dải thường là đơn vị kiểm tra tốt hơn."
            ),
        )
        include_body = st.checkbox(
            "Kèm ảnh toàn linh kiện + chân",
            value=bool(config.get("include_body_view", True)),
        )
        refine = st.checkbox(
            "Siết ROI về vùng kim loại thật",
            value=bool(config.get("refine_to_metal", True)),
            help=(
                "Các tỉ lệ ở trên nói ROI nằm ĐÂU; chúng không biết land rộng "
                "bao nhiêu. Bật thì ROI được thu về đúng vệt kim loại bên trong "
                "nó. Tắt để so sánh — bảng ở tab 'Bảng nhãn 6.2' có cột "
                "`refined` và `shrink_pct` cho biết nó đã làm gì."
            ),
        )
        terminal_outer = st.slider(
            "Nới đầu trục dài (nhân cạnh dài)",
            0.10,
            0.80,
            float(config.get("terminal_outer_ratio", 0.45)),
            0.05,
            help="Tăng nếu ROI chưa với tới hết pad của điện trở/tụ.",
        )
        lead_outer = st.slider(
            "Nới cạnh nhiều chân (nhân cạnh ngắn)",
            0.10,
            0.60,
            float(config.get("lead_outer_ratio", 0.26)),
            0.02,
        )
        current_size = int(config.get("target_size", 128))
        size_options = [64, 96, 128, 160, 224]
        size_index = (
            size_options.index(current_size) if current_size in size_options else 2
        )
        joint_size = st.selectbox("Kích thước crop", size_options, index=size_index)
        submitted = st.form_submit_button(
            "Tạo lại ROI", type="primary", width="stretch"
        )
    if submitted:
        config.update(
            {
                "enabled": enabled,
                "refine_to_metal": refine,
                "split_pins": split_pins,
                "include_body_view": include_body,
                "terminal_outer_ratio": terminal_outer,
                "lead_outer_ratio": lead_outer,
                "target_size": joint_size,
            }
        )
        _execute_solder(_get_bridge())
        st.rerun()


def _execute_solder(bridge: PipelineBridge) -> SolderResult | None:
    """Step 5.5: derive the solder-joint ROIs that step 6.2 needs.

    Failure here must not invalidate the component crops, which are the input
    to step 6.1; it is reported and the run continues.
    """

    source = _analysis_image()
    detection_result = st.session_state.detection_result
    if source is None or not isinstance(detection_result, DetectionResult):
        return None
    if not st.session_state.config.get("solder", {}).get("enabled", True):
        st.session_state.solder_result = None
        st.session_state.statuses[7] = "skipped"
        return None
    st.session_state.statuses[7] = "running"
    try:
        result = bridge.make_solder_crops(source, detection_result.detections)
    except Exception as exc:
        st.session_state.solder_result = None
        st.session_state.statuses[7] = "error"
        st.session_state.messages.append(
            f"Bước 5.5 lỗi: {type(exc).__name__}: {exc}"
        )
        return None
    st.session_state.solder_result = result
    # Not ``_record_stage``: this one step covers both 5.5 and 6.2, and the log
    # is more useful naming them than repeating "Bước 7" twice.
    st.session_state.statuses[7] = _mode_to_status(result.mode)
    elapsed = result.metrics.get("elapsed_ms")
    if elapsed is not None:
        st.session_state.latencies[7] = float(elapsed)
    st.session_state.messages.append(f"Bước 5.5: {result.message}")
    if result.verdicts:
        layer = "model + luật đo" if result.graded_by_model else "luật đo"
        st.session_state.messages.append(
            f"Bước 6.2: {len(result.verdicts)} kết quả ({layer})."
        )
    return result


def _draw_solder_overlay(
    image: np.ndarray,
    crops: list[SolderCropRecord],
    show_body: bool,
    by_source: bool = False,
) -> np.ndarray:
    overlay = image.copy()
    for crop in crops:
        if crop.kind == "body" and not show_body:
            continue
        x1, y1, x2, y2 = crop.bbox
        if by_source:
            color = SOLDER_SOURCE_COLORS.get(crop.source, (200, 200, 200))
        else:
            color = SOLDER_ROI_COLORS.get(crop.kind, (200, 200, 200))
        thickness = 1 if crop.kind == "body" else 2
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)
    return overlay


def _shrink_percent(crop: SolderCropRecord) -> float | None:
    """How much of the derived ROI ``refine_to_metal`` cut away, as a percent.

    Measured across the refine step alone, from the box before it to the box
    right after it -- NOT to the final bbox. De-confliction runs again after
    fusion, so the final box has been through two stages; comparing against it
    once produced shrink figures as low as -11%, which reads as "shrinking made
    it bigger" and is really "the later cut took less off a smaller box".

    ``None`` means the ROI was left alone -- either the stage is off, or the
    evidence inside was too weak to act on. Both are meaningful answers and
    neither should read as 0%.
    """

    before, after = crop.roi_before_refine, crop.roi_after_refine
    if not crop.refined or before is None or after is None:
        return None
    before_area = max(0, before[2] - before[0]) * max(0, before[3] - before[1])
    if before_area <= 0:
        return None
    after_area = max(0, after[2] - after[0]) * max(0, after[3] - after[1])
    return round(100.0 * (1.0 - after_area / before_area), 1)


def _solder_frame(
    crops: list[SolderCropRecord],
    verdicts: list[SolderVerdictRecord] | None = None,
) -> pd.DataFrame:
    """One row per ROI, carrying what step 6.2 called it.

    ``defect_class`` used to be hard-coded empty because this table doubled as
    the labelling sheet. That made the panel look like step 6.2 had produced
    nothing, when in fact every row already had a verdict. The machine's call
    goes in ``defect_class`` where a reader expects it, and the blank column
    for a human to fill is now named ``label_manual`` so the two are never
    confused with each other.
    """

    by_joint = {item.joint_id: item for item in (verdicts or [])}
    rows = []
    for crop in crops:
        verdict = by_joint.get(crop.joint_id)
        rows.append(
            {
                "joint_id": crop.joint_id,
                "detection_id": crop.detection_id,
                "component_label": crop.label,
                "kind": crop.kind,
                "position": crop.position,
                "pin_index": crop.pin_index,
                "terminal_geometry": crop.terminal_geometry,
                "x1": crop.bbox[0],
                "y1": crop.bbox[1],
                "x2": crop.bbox[2],
                "y2": crop.bbox[3],
                "roi_width_px": crop.bbox[2] - crop.bbox[0],
                "roi_height_px": crop.bbox[3] - crop.bbox[1],
                "detector_confidence": crop.confidence,
                "source": crop.source,
                # --- what refine_to_metal did to this ROI -----------------
                "refined": bool(crop.refined),
                "shrink_pct": _shrink_percent(crop),
                "designator": crop.designator or "",
                "pin": crop.pin or "",
                "net": crop.net or "",
                # --- what step 6.2 decided -------------------------------
                "defect_class": verdict.label if verdict else "",
                "decision": verdict.decision if verdict else "",
                "verdict_source": verdict.source if verdict else "",
                "rule_label": (verdict.rule_label or "") if verdict else "",
                "model_label": (verdict.model_label or "") if verdict else "",
                "model_prob": verdict.model_probability if verdict else None,
                "reasons": " | ".join(verdict.reasons) if verdict else "",
                # --- and the column a person fills in --------------------
                "label_manual": "",
            }
        )
    return pd.DataFrame(rows)


def _roi_pixels_for_display(
    source: np.ndarray | None, crop: SolderCropRecord
) -> np.ndarray | None:
    """The ROI's pixels, cut on demand rather than carried around.

    Records deliberately do not hold their own image; on a full board that cost
    292 MB against 0.70 MB for the coordinates. Cutting one 30x15 patch out of
    a frame that is already in memory is far cheaper than having kept it.
    """

    if crop.image is not None:
        return crop.image
    if source is None:
        return None
    height, width = source.shape[:2]
    x1, y1, x2, y2 = crop.bbox
    x1 = max(0, min(int(x1), width))
    y1 = max(0, min(int(y1), height))
    x2 = max(x1, min(int(x2), width))
    y2 = max(y1, min(int(y2), height))
    patch = source[y1:y2, x1:x2]
    return patch if patch.size else None


def _draw_verdict_overlay(
    image: np.ndarray, verdicts: list[SolderVerdictRecord], show_component: bool
) -> np.ndarray:
    overlay = image.copy()
    for verdict in verdicts:
        if verdict.scope == "component" and not show_component:
            continue
        x1, y1, x2, y2 = verdict.bbox
        color = VERDICT_DECISION_COLORS.get(verdict.decision, (200, 200, 200))
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        if verdict.decision != "accept":
            cv2.putText(
                overlay, verdict.label, (x1, max(11, y1 - 3)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1, cv2.LINE_AA,
            )
    return overlay


def _verdict_frame(verdicts: list[SolderVerdictRecord]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "joint_id": item.joint_id,
                "designator": item.designator or "",
                "pin": item.pin or "",
                "component": item.component_label,
                "scope": item.scope,
                "label": item.label,
                "decision": item.decision,
                "source": item.source,
                "rule_label": item.rule_label or "",
                "model_label": item.model_label or "",
                "model_prob": item.model_probability,
                "solder_ratio": item.features.get("solder_ratio"),
                "span_ratio": item.features.get("span_ratio"),
                "specular_ratio": item.features.get("specular_ratio"),
                "reasons": " | ".join(item.reasons),
            }
            for item in verdicts
        ]
    )


def _render_solder_grading(result: SolderResult) -> None:
    """Step 6.2: what each ROI was called, and on what evidence."""

    verdicts = result.verdicts
    if not verdicts:
        if result.grading_error:
            # A failure and a stage that was never run look identical from an
            # empty panel; say which one this is.
            st.error(f"Bước 6.2 chạy nhưng lỗi: {result.grading_error}")
            st.caption(
                "ROI ở tab 5.5 vẫn dùng được để gán nhãn. Nếu vừa nạp model 6.2, "
                "hãy kiểm tra model và manifest có cùng bộ lớp không."
            )
        else:
            _render_empty(
                "Chưa chấm được mối hàn",
                "Bước 6.2 chạy cùng bước 5.5. Nếu trống, hãy chạy lại bước 5.",
            )
        return

    if result.grading_error:
        # The model can fail per board while staying loaded; say which board
        # this was, rather than leaving the panel claiming a model verdict.
        st.warning(f"Bước 6.2: {result.grading_error}")
    if result.graded_by_model:
        st.success(
            f"Đang chấm bằng **model + luật đo** (model: "
            f"{verdicts[0].model_version}). Bất đồng giữa hai tầng được đưa vào "
            "hàng đợi kiểm tra thay vì chọn bên nào thắng."
        )
    else:
        st.info(
            "**Đang chấm bằng luật đo hình học, chưa có model** — đây là trạng "
            "thái bình thường, không phải lỗi. Nạp `.onnx` + `model_manifest.json` "
            "ở sidebar **Model kiểm tra mối hàn 6.2** để bật thêm tầng phân loại.\n\n"
            "Ngưỡng mặc định chỉ là số khởi đầu. Chạy "
            "`scripts/calibrate_solder_thresholds.py` trên các board bạn đã chấp "
            "nhận để đo ngưỡng từ chính dây chuyền của bạn."
        )

    joints = [item for item in verdicts if item.scope == "joint"]
    counts = collections.Counter(item.decision for item in joints)
    columns = st.columns(4)
    columns[0].metric("ROI mối hàn", len(joints))
    columns[1].metric("Đạt", counts.get("accept", 0))
    columns[2].metric("Cần kiểm tra", counts.get("review", 0))
    columns[3].metric("Loại", counts.get("reject", 0))

    conflicts = [item for item in joints if item.source == "conflict"]
    guarded = [item for item in joints if item.source == "escape_guard"]
    if conflicts:
        st.warning(
            f"{len(conflicts)} ROI có model và luật bất đồng. Đây chính là chỗ "
            "đáng xem trước tiên khi hiệu chỉnh ngưỡng hoặc đánh giá model."
        )
    if guarded:
        st.error(
            f"{len(guarded)} ROI bị chốt chặn giữ lại: model kết luận đạt nhưng "
            "lượng thiếc đo được dưới sàn vật lý."
        )

    source = _analysis_image()
    overlay_tab, table_tab, detail_tab = st.tabs(
        ["Verdict overlay", "Bảng kết quả", "Chi tiết theo ROI"]
    )
    with overlay_tab:
        if source is None:
            _render_empty("Chưa có ảnh", "Hoàn thành bước 1 đến 4 trước.")
        else:
            show_component = st.checkbox(
                "Hiện cả ROI mức linh kiện", value=False, key="verdict_show_component"
            )
            _show_image(
                _draw_verdict_overlay(source, verdicts, show_component),
                "Xanh: đạt · Cam: cần kiểm tra · Đỏ: loại",
            )
    with table_tab:
        frame = _verdict_frame(verdicts)
        st.dataframe(frame, width="stretch", height=340)
        st.download_button(
            "Tải solder_verdicts.csv",
            frame.to_csv(index=False).encode("utf-8-sig"),
            file_name="solder_verdicts.csv",
            mime="text/csv",
        )
    with detail_tab:
        flagged = [item for item in joints if item.decision != "accept"]
        pool = flagged or joints
        st.caption(
            f"{len(flagged)} ROI cần chú ý trong tổng {len(joints)}."
            if flagged
            else "Không ROI nào bị gắn cờ; hiển thị toàn bộ."
        )
        for item in pool[:40]:
            icon = VERDICT_DECISION_ICONS.get(item.decision, "·")
            title = f"{icon} {item.label} — {item.designator or item.component_label}"
            with st.expander(f"{title} ({item.joint_id})", expanded=False):
                st.markdown(
                    f"**Quyết định:** {item.decision} · **Nguồn:** {item.source}  \n"
                    f"**Luật:** {item.rule_label or '—'} · **Model:** "
                    f"{item.model_label or '—'}"
                    + (
                        f" ({item.model_probability:.2f})"
                        if item.model_probability is not None
                        else ""
                    )
                )
                for reason in item.reasons:
                    st.markdown(f"- {reason}")
                if item.features:
                    st.json(item.features, expanded=False)


def _render_solder_rois() -> None:
    """Step 5.5 view: the ROIs that make solder joints visible for step 6.2.

    The detector box stops at the component body, so these ROIs are derived
    from that box plus the class terminal topology rather than detected.
    """

    source = _analysis_image()
    settings, content = st.columns([0.75, 2.25], gap="large")
    with settings:
        _render_solder_settings()

    result = st.session_state.solder_result
    with content:
        if not isinstance(result, SolderResult):
            _render_empty(
                "Chưa có ROI mối hàn",
                "Chạy lại bước 5 hoặc bấm Tạo lại ROI để sinh vùng kiểm tra mối hàn.",
            )
            return
        if result.mode == "UNAVAILABLE" or not result.crops:
            _render_empty("Chưa sinh được ROI", result.message)
            return

        joints = [crop for crop in result.crops if crop.kind == "joint"]
        bodies = [crop for crop in result.crops if crop.kind == "body"]
        metric_a, metric_b, metric_c = st.columns(3)
        metric_a.metric("ROI mối hàn", len(joints))
        metric_b.metric("Ảnh linh kiện kèm chân", len(bodies))
        smallest = min(
            (
                min(crop.bbox[2] - crop.bbox[0], crop.bbox[3] - crop.bbox[1])
                for crop in joints
            ),
            default=0,
        )
        metric_c.metric("ROI nhỏ nhất (px)", smallest)
        if joints and smallest < SOLDER_MIN_READABLE_PX:
            st.warning(
                f"ROI nhỏ nhất chỉ {smallest} px ở cạnh ngắn. Ở kích thước đó không "
                "đọc được hình dạng fillet; cần chụp độ phân giải cao hơn hoặc thu "
                "hẹp trường nhìn trước khi gán nhãn 6.2."
            )

        # Step 6.2 lives one tab up, not in here. Say so: from inside this view
        # the ROIs look like the end of the road, and someone who never opens
        # the sibling tab concludes the grading stage was never built.
        if result.verdicts:
            st.caption(
                f"Đã chấm {len(result.verdicts)} ROI ở tab **6.2 · Chấm lỗi hàn** "
                "bên cạnh"
                + (
                    " — đang dùng model + luật đo."
                    if result.graded_by_model
                    else " — đang dùng luật đo. Nạp model 6.2 ở sidebar để thêm "
                    "một tầng phân loại."
                )
            )
        elif result.grading_error:
            st.warning(f"Bước 6.2 lỗi: {result.grading_error}")

        # No grading tab here: step 6.2 has its own tab one level up. Rendering
        # it in both places drew the same widget twice and Streamlit refused the
        # duplicate key, which took the whole step-7 view down.
        overlay_tab, gallery_tab, table_tab, cad_tab = st.tabs(
            [
                "ROI overlay",
                "Joint gallery",
                "Bảng nhãn 6.2",
                "Đối chiếu CAD",
            ]
        )
        with overlay_tab:
            if source is None:
                _render_empty("Chưa có ảnh", "Hoàn thành bước 1 đến 4 trước.")
            else:
                show_body = st.checkbox("Hiện khung toàn linh kiện", value=True)
                by_source = False
                if result.used_cad:
                    by_source = st.checkbox(
                        "Tô màu theo nguồn ROI",
                        value=True,
                        help="Xanh lá: CAD và detector cùng đồng ý · Hồng: chỉ CAD · "
                        "Vàng: chỉ suy ra từ detector.",
                    )
                _show_image(
                    _draw_solder_overlay(source, result.crops, show_body, by_source),
                    (
                        "Xanh lá: CAD + detector - Hồng: chỉ CAD - Vàng: chỉ suy ra"
                        if by_source
                        else "Vàng: ROI mối hàn - Xanh: linh kiện kèm chân"
                    ),
                )
        with gallery_tab:
            kind = st.radio("Loại ROI", ["Mối hàn", "Linh kiện kèm chân"], horizontal=True)
            selected = joints if kind == "Mối hàn" else bodies
            st.caption(
                f"Hiển thị {min(len(selected), 60)}/{len(selected)} ROI "
                "(giới hạn 60 để UI mượt)."
            )
            for offset in range(0, min(len(selected), 60), 6):
                columns = st.columns(6)
                for column, crop in zip(columns, selected[offset : offset + 6]):
                    with column:
                        pixels = _roi_pixels_for_display(source, crop)
                        if pixels is not None:
                            _show_image(pixels)
                        tag = f" · {crop.designator}" if crop.designator else ""
                        st.caption(f"**{crop.label}**{tag}\n\n{crop.position}")
        with table_tab:
            frame = _solder_frame(result.crops, result.verdicts)
            st.dataframe(frame, width="stretch", height=320)
            if result.verdicts:
                st.caption(
                    "`defect_class` là kết luận của bước 6.2 cho từng ROI; "
                    "`rule_label` và `model_label` cho thấy mỗi tầng nói gì, và "
                    "`reasons` là số đo dẫn tới kết luận đó. Cột `label_manual` "
                    "để trống là chỗ bạn gán nhãn thật khi xuất dữ liệu train."
                )
            else:
                st.caption(
                    "Chưa có kết luận 6.2 nên `defect_class` để trống. Cột "
                    "`label_manual` là chỗ gán nhãn tay: hình học đã xong nên "
                    "gán nhãn chỉ còn là phán quyết theo từng dòng."
                )
            st.download_button(
                "Tải solder_joints.csv",
                frame.to_csv(index=False).encode("utf-8-sig"),
                file_name="solder_joints.csv",
                mime="text/csv",
            )
        with cad_tab:
            _render_cad_panel(result)


def _findings_frame(findings: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "kind": item.get("kind"),
                "severity": item.get("severity"),
                "designator": item.get("designator") or "",
                "expected_class": item.get("expected_class") or "",
                "observed_class": item.get("observed_class") or "",
                "shift_mm": item.get("shift_mm"),
                "message": item.get("message"),
            }
            for item in findings
        ]
    )


def _render_cad_panel(result: SolderResult) -> None:
    """What the CAD comparison found, and how far to trust the alignment."""

    if not st.session_state.cad_summary:
        st.info(
            "Chưa nạp sơ đồ CAD. ROI hiện tại được suy ra từ box của detector cộng "
            "topology chân của class. Nạp file CAD ở sidebar để hợp nhất với toạ độ "
            "land thật; pipeline không đổi gì khác."
        )
        return
    if not result.used_cad:
        st.warning(
            "Đã nạp CAD nhưng chưa áp dụng được cho ảnh này; bước 5.5 dùng ROI suy ra."
        )
        for warning in result.cad_warnings:
            st.caption(f"· {warning}")
        return

    registration = result.registration or {}
    stats = result.cad_stats or {}
    if registration.get("ambiguous"):
        st.error(
            "Căn CAD **mơ hồ**: có phép căn khác khớp không kém. Kiểm tra overlay "
            "trước khi dùng crop, hoặc chốt bằng fiducial / file registration."
        )
    for warning in result.cad_warnings:
        st.warning(warning)

    columns = st.columns(4)
    columns[0].metric(
        "Khớp CAD", f"{stats.get('matched', 0)}/{stats.get('cad_components', 0)}"
    )
    columns[1].metric("Thiếu linh kiện", stats.get("missing", 0))
    columns[2].metric("Lệch vị trí", stats.get("shifted", 0))
    columns[3].metric("px / mm", f"{registration.get('scale_px_per_mm', 0.0):.2f}")
    st.caption(
        f"Phương pháp: {registration.get('method', '—')} · "
        f"residual {registration.get('residual_px', 0.0):.2f} px · "
        f"inlier {registration.get('inlier_ratio', 0.0):.0%} · "
        f"class khớp {stats.get('class_agreements', 0)}/{stats.get('class_comparable', 0)}"
    )

    if result.findings:
        for item in sorted(
            result.findings,
            key=lambda entry: {"defect": 0, "review": 1}.get(entry.get("severity"), 2),
        )[:6]:
            icon = CAD_SEVERITY_ICONS.get(item.get("severity"), "·")
            st.markdown(f"{icon} {item.get('message')}")
        frame = _findings_frame(result.findings)
        st.dataframe(frame, width="stretch", height=220)
        st.download_button(
            "Tải cad_findings.csv",
            frame.to_csv(index=False).encode("utf-8-sig"),
            file_name="cad_findings.csv",
            mime="text/csv",
        )
    else:
        st.success("CAD và ảnh khớp nhau: không có linh kiện thiếu, thừa hay lệch.")

    if registration:
        st.download_button(
            "Tải registration.json",
            json.dumps(registration, indent=2).encode("utf-8"),
            file_name="cad_registration.json",
            mime="application/json",
            help="Nạp lại ở sidebar để mọi lần chạy sau dùng đúng phép căn này.",
        )


def _render_step_seven() -> None:
    """Step 6.2 as a section of its own, not a tab tucked inside step 4.

    Solder inspection has its own ROIs, its own model contract and its own
    verdict vocabulary; burying it under component detection made it look like
    a detail of that stage rather than the stage that decides whether a board
    ships. The two halves stay together here because they are one question --
    where the joints are (5.5) and what they are (6.2).
    """

    _render_step_heading(7)
    crops: list[CropRecord] = st.session_state.crops
    if not crops and not isinstance(st.session_state.solder_result, SolderResult):
        _render_empty(
            "Chưa có dữ liệu mối hàn",
            "Chạy tới bước 5 trước; ROI mối hàn được suy ra từ box linh kiện.",
        )
        return

    roi_tab, grading_tab = st.tabs(["5.5 · ROI chân hàn", "6.2 · Chấm lỗi hàn"])
    with roi_tab:
        _render_solder_rois()
    with grading_tab:
        result = st.session_state.solder_result
        if not isinstance(result, SolderResult):
            _render_empty(
                "Chưa có ROI để chấm",
                "Tạo ROI ở tab 5.5 trước khi chấm lỗi mối hàn.",
            )
        else:
            _render_solder_grading(result)

    # Ngoài cả hai tab: mục đánh giá thuộc về cả bước, không riêng tab nào.
    _render_model_feedback(
        "solder", _solder_targets(st.session_state.solder_result)
    )


def main() -> None:
    _init_state()
    _load_css()
    quick_run = _render_sidebar()
    _render_header()
    if quick_run:
        _run_all()

    renderers: dict[int, Callable[[], None]] = {
        0: _render_step_zero,
        1: _render_step_one,
        2: _render_step_two,
        3: _render_step_three,
        4: _render_step_four,
        5: _render_step_five,
        6: _render_step_six,
        7: _render_step_seven,
        8: _render_step_eight,
    }
    renderers[st.session_state.active_step]()
    _render_activity_log()
    _render_footer()


if __name__ == "__main__":
    main()
