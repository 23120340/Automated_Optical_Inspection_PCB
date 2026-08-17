"""Streamlit dashboard for AOI PCB steps 0 through 5.

Run from the repository root:

    streamlit run app/streamlit_app.py

All images in session state and at the pipeline boundary use OpenCV BGR order.
"""

from __future__ import annotations

import csv
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
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app.pipeline_bridge import (  # noqa: E402
    BoardResult,
    CropRecord,
    DetectionRecord,
    DetectionResult,
    PipelineBridge,
    StageResult,
)


APP_TITLE = "AOI PCB · Workbench"
APP_VERSION = "0.2.0"
MAX_IMAGE_BYTES = 64 * 1024 * 1024
MAX_IMAGE_PIXELS = 50_000_000
MAX_CALIBRATION_PROFILE_BYTES = 256 * 1024
STEP_DEFINITIONS = (
    (0, "Thu thập ảnh", "Import ảnh PCB", "IN"),
    (1, "Tiền xử lý", "Undistort và chuẩn hóa", "FX"),
    (2, "Căn chỉnh PCB", "Golden image + homography", "AL"),
    (3, "Khoanh vùng PCB", "Xác định board ROI", "ROI"),
    (4, "Phát hiện linh kiện", "Detector từ Kaggle", "AI"),
    (5, "Cắt linh kiện", "Crop + normalize + export", "CUT"),
)
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
            "max_side": 1600,
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
            "confidence": 0.35,
            "iou": 0.45,
            "max_candidates": 2000,
            "min_area_ratio": 0.00004,
            "max_area_ratio": 0.035,
            "device": "auto",
            # Kaggle notebook validates/exports the YOLO26 one-to-many head so
            # IoU/NMS behavior is identical in Kaggle and in this local app.
            "end2end": False,
        },
        "crops": {
            "padding": 6,
            "padding_ratio": 0.0,
            "square": False,
            "normalize": True,
            "target_size": 224,
            "image_format": "png",
        },
    }


def _init_state() -> None:
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
        "component_model_path": None,
        "component_model_name": None,
        "component_model_digest": None,
        "pt_model_trusted": False,
        "preprocess_result": None,
        "alignment_result": None,
        "board_result": None,
        "detection_result": None,
        "crops": [],
        "statuses": {step: "pending" for step in range(6)},
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
        },
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


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
    }
    for candidate in range(step + 1, 6):
        st.session_state[result_keys[candidate]] = [] if candidate == 5 else None
        st.session_state.statuses[candidate] = "pending"
        st.session_state.latencies.pop(candidate, None)


def _set_source(name: str, data: bytes) -> None:
    digest = _digest(data)
    if digest == st.session_state.input_digest:
        return
    image = _decode_image(data)
    st.session_state.input_image = image
    st.session_state.input_name = name
    st.session_state.input_digest = digest
    st.session_state.statuses[0] = "done"
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
    else:
        st.session_state.pt_model_trusted = Path(upload.name).suffix.lower() != ".pt"
        _invalidate_after(3)
    st.session_state.messages.append(f"Đã nạp {kind} model: {upload.name}")


def _remove_model(kind: str) -> None:
    st.session_state.ignored_uploads[kind] = st.session_state[f"{kind}_model_digest"]
    st.session_state[f"{kind}_model_path"] = None
    st.session_state[f"{kind}_model_name"] = None
    st.session_state[f"{kind}_model_digest"] = None
    if kind == "board":
        _invalidate_after(2)
    else:
        st.session_state.pt_model_trusted = False
        _invalidate_after(3)


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
    config.setdefault("models", {})
    config["models"]["board_path"] = board_model_path
    config["models"]["component_path"] = component_model_path
    return PipelineBridge(
        config=config,
        model_path=component_model_path,
        board_model_path=board_model_path,
    )


def _get_bridge() -> PipelineBridge:
    config_json = json.dumps(st.session_state.config, sort_keys=True, ensure_ascii=False)
    bridge = _cached_bridge(
        config_json,
        st.session_state.component_model_path,
        st.session_state.board_model_path,
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
    st.session_state.statuses[5] = "demo" if "DEMO" in detection_result.mode.upper() else "done"
    st.session_state.latencies[5] = round(elapsed_ms, 2)
    st.session_state.messages.append(f"Bước 5: tạo {len(crops)} crop.")
    return crops


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
    for index, (step, label, callback) in enumerate(stages, start=1):
        progress.progress((index - 1) / len(stages), text=f"Bước {step}/5 · {label}")
        try:
            callback(bridge)
        except Exception as exc:
            st.session_state.statuses[step] = "error"
            st.session_state.messages.append(f"Bước {step} lỗi: {type(exc).__name__}: {exc}")
            progress.empty()
            st.error(f"Dừng ở bước {step}: {exc}")
            return
    progress.progress(1.0, text="Hoàn tất workflow 0–5")
    st.session_state.active_step = 5
    st.session_state.pending_navigation = 5
    st.toast("Đã chạy xong workflow 0–5.", icon="✅")


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


def _render_sidebar() -> bool:
    with st.sidebar:
        pending_navigation = st.session_state.pending_navigation
        if pending_navigation is not None:
            st.session_state.sidebar_step_navigation = pending_navigation
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
        st.markdown('<div class="sidebar-section-label">WORKFLOW · 0–5</div>', unsafe_allow_html=True)
        step_markup: list[str] = ['<div class="stepper">']
        for step, name, description, code in STEP_DEFINITIONS:
            status = st.session_state.statuses[step]
            active = " active" if step == st.session_state.active_step else ""
            step_markup.append(
                f'<div class="step-row {status}{active}">'
                f'<div class="step-code">{html.escape(code)}</div>'
                f'<div class="step-copy"><strong>{step}. {html.escape(name)}</strong>'
                f'<span>{html.escape(description)}</span></div>'
                f'<div class="step-state">{_status_dot(status)}</div></div>'
            )
        step_markup.append("</div>")
        st.markdown("".join(step_markup), unsafe_allow_html=True)

        step_options = [step for step, *_ in STEP_DEFINITIONS]
        selected = st.radio(
            "Mở bước",
            options=step_options,
            index=step_options.index(st.session_state.active_step),
            format_func=lambda value: f"{value}. {STEP_DEFINITIONS[value][1]}",
            horizontal=False,
            label_visibility="collapsed",
            key="sidebar_step_navigation",
        )
        st.session_state.active_step = selected

        st.markdown('<div class="sidebar-section-label">TÀI NGUYÊN</div>', unsafe_allow_html=True)
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

        with st.expander("Golden Image / Reference", expanded=False):
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

        st.markdown(
            '<div class="pipeline-asset"><span>BOARD ROI</span><b>Contour pipeline</b>'
            '<small>Model hook được để dành cho giai đoạn sau</small></div>',
            unsafe_allow_html=True,
        )

        with st.expander("Model phát hiện linh kiện", expanded=True):
            component_upload = st.file_uploader(
                "Component detector (.onnx/.pt)",
                type=["onnx", "pt"],
                key="component_model_uploader",
                help="File export sau khi train trên Kaggle. Ưu tiên .onnx để chạy local.",
            )
            if component_upload is not None:
                _set_model(component_upload, "component")
            _render_model_asset("component")

        st.markdown('<div class="security-note"><b>Lưu ý model</b><br>.pt có thể chứa pickle. Chỉ mở weight do bạn tự train hoặc nguồn tin cậy; ưu tiên ONNX khi trao đổi.</div>', unsafe_allow_html=True)
        quick_run = st.button(
            "▶  Chạy toàn bộ 0–5",
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


def _render_model_asset(kind: str) -> None:
    name = st.session_state[f"{kind}_model_name"]
    if not name:
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
    _, name, description, code = STEP_DEFINITIONS[step]
    status = st.session_state.statuses[step]
    st.markdown(
        f"""
        <div class="section-heading">
          <div class="section-index">{code}</div>
          <div><span>BƯỚC {step} / 5</span><h2>{html.escape(name)}</h2><p>{html.escape(description)}</p></div>
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
        help="Prototype xử lý một ảnh mỗi lần để giới hạn RAM; có thể thay file bất kỳ lúc nào.",
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
                if st.button("Nạp ảnh này vào pipeline", type="primary", width="stretch"):
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
            st.success("Ảnh đã sẵn sàng cho bước 1.")
            st.code(st.session_state.input_digest[:16], language=None)
            if st.button("Đi đến bước 1 →", width="stretch"):
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
            max_side = st.number_input("Cạnh dài tối đa", 640, 4096, int(config["max_side"]), 160)
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
            }
        )
    return pd.DataFrame(rows)


def _render_step_four() -> None:
    _render_step_heading(4)
    source = _analysis_image()
    if source is None:
        _render_empty("Thiếu ảnh đầu vào", "Hoàn thành bước 0 trước khi chạy detector.")
        return
    config = st.session_state.config["components"]
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
                }
            )
            _run_stage(4, _execute_components)
        st.markdown("##### Model hand-off từ Kaggle")
        st.caption("Cần: `best.onnx` (ưu tiên), `classes.yaml`/`data.yaml`, metrics và ảnh test. Có thể dùng `best.pt` nếu chính bạn export.")
    with preview:
        result = st.session_state.detection_result
        overlay_tab, table_tab, stats_tab = st.tabs(["Detection overlay", "Detection table", "Class stats"])
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
    return {
        "schema_version": "aoi-pcb-workbench/0.2",
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
            }
            for item in detections
        ],
        "crop_count": len(st.session_state.crops),
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


def _build_zip() -> bytes:
    output = io.BytesIO()
    base = _safe_name(st.session_state.input_name or "pcb")
    manifest = _manifest()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        archive.writestr("detections.csv", _detections_csv_bytes())
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
    st.caption("ZIP gồm ảnh từng stage, annotated image, detections CSV/JSON và toàn bộ crop.")
    col_a, col_b = st.columns(2)
    with col_a:
        st.download_button(
            "Tải toàn bộ ZIP",
            zip_bytes,
            file_name=f"{base}_aoi_steps_0_5.zip",
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
          <span>AOI PCB WORKBENCH · STEPS 0–5</span>
          <span>Local session · dữ liệu không được upload ra ngoài bởi UI</span>
        </div>
        """,
        unsafe_allow_html=True,
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
    }
    renderers[st.session_state.active_step]()
    _render_activity_log()
    _render_footer()


if __name__ == "__main__":
    main()
