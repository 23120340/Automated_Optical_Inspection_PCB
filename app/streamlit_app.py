"""Streamlit dashboard for AOI PCB steps 0 through 6.1.

Run from the repository root:

    streamlit run app/streamlit_app.py

All images in session state and at the pipeline boundary use OpenCV BGR order.
"""

from __future__ import annotations

import collections
from collections.abc import Mapping
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

from aoi_pipeline.inspection.cad import (  # noqa: E402
    CadError,
    CadRegistration,
    load_cad,
)
from aoi_pipeline.grading.classifier import (  # noqa: E402
    MANIFEST_SCHEMA as SOLDER_MANIFEST_SCHEMA,
)

from app.pipeline_bridge import (  # noqa: E402
    BoardResult,
    ClassificationRecord,
    ClassificationResult,
    CropRecord,
    DetectionRecord,
    DetectionResult,
    PipelineBridge,
    SolderCropRecord,
    SolderResult,
    SolderVerdictRecord,
    StageResult,
)


APP_TITLE = "AOI PCB · Workbench"
APP_VERSION = "0.5.0"
MAX_IMAGE_BYTES = 64 * 1024 * 1024
MAX_IMAGE_PIXELS = 50_000_000
MIN_SOURCE_LONG_SIDE = 1280
MIN_SOURCE_SHORT_SIDE = 960
MIN_SOURCE_PIXELS = MIN_SOURCE_LONG_SIDE * MIN_SOURCE_SHORT_SIDE
MAX_CALIBRATION_PROFILE_BYTES = 256 * 1024
STEP_DEFINITIONS = (
    (0, "Thu thập ảnh", "Import ảnh PCB", "IN"),
    (1, "Tiền xử lý", "Undistort và chuẩn hóa", "FX"),
    (2, "Căn chỉnh PCB", "Golden image + homography", "AL"),
    (3, "Khoanh vùng PCB", "Xác định board ROI", "ROI"),
    (4, "Phát hiện linh kiện", "Detector từ Kaggle", "AI"),
    (5, "Cắt linh kiện", "Crop + normalize + export", "CUT"),
    (6, "6.1 Phân loại linh kiện", "Family + accept/review/unknown", "CLS"),
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
        # Mirrors the step-6.1 training crop recipe: pad = 0.15 * max(w, h),
        # no squaring, letterbox to 224. A fixed pixel padding is not a
        # substitute -- 6 px around an 0603 chip is effectively no padding.
        "crops": {
            "padding": 0,
            "padding_ratio": 0.15,
            "square": False,
            "normalize": True,
            "target_size": 224,
            "image_format": "png",
        },
        # Left empty on purpose: the pipeline runs unchanged with no CAD,
        # and the sidebar fills these in when a board file is uploaded.
        "cad": {
            "path": None,
            "fmt": "auto",
            "units": "mm",
            "side": "top",
            "registration_path": None,
            "auto_register": True,
        },
        "fusion": {
            "enabled": True,
            "local_refine": True,
            "max_shift_mm": 0.5,
            "merge_mode": "union",
            "emit_cad_only_rois": True,
        },
        "solder_grading": {
            "enabled": True,
            "model_path": None,
            "manifest_path": None,
            "rules_only_defect_decision": "review",
        },
        "solder": {
            "enabled": True,
            "split_pins": False,
            "include_body_view": True,
            "target_size": 128,
            "terminal_outer_ratio": 0.45,
            "lead_outer_ratio": 0.26,
        },
        "classification": {
            "batch_size": 32,
            "top_k": 3,
            "device": "cpu",
            "accept_threshold": None,
            "review_threshold": None,
            "temperature": None,
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
        "classifier_model_path": None,
        "classifier_model_name": None,
        "classifier_model_digest": None,
        "classifier_manifest_path": None,
        "classifier_manifest_name": None,
        "classifier_manifest_digest": None,
        "classifier_manifest_quality_warning": None,
        "cad_path": None,
        "cad_name": None,
        "cad_digest": None,
        "cad_summary": None,
        "cad_registration_path": None,
        "cad_registration_name": None,
        "solder_model_path": None,
        "solder_model_name": None,
        "solder_model_digest": None,
        "solder_manifest_path": None,
        "solder_manifest_name": None,
        "solder_manifest_digest": None,
        "pt_model_trusted": False,
        "preprocess_result": None,
        "alignment_result": None,
        "board_result": None,
        "detection_result": None,
        "crops": [],
        "solder_result": None,
        "classification_result": None,
        "statuses": {step: "pending" for step in range(7)},
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
            "cad": None,
            "cad_registration": None,
            "solder_model": None,
            "solder_manifest": None,
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
    }
    for candidate in range(step + 1, 7):
        st.session_state[result_keys[candidate]] = [] if candidate == 5 else None
        if candidate == 5:
            # Step 5.5 shares step 5's inputs, so it expires with the crops.
            st.session_state.solder_result = None
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
    elif kind == "component":
        st.session_state.pt_model_trusted = Path(upload.name).suffix.lower() != ".pt"
        _invalidate_after(3)
    else:
        _invalidate_after(5)
    st.session_state.messages.append(f"Đã nạp {kind} model: {upload.name}")


def _remove_model(kind: str) -> None:
    st.session_state.ignored_uploads[kind] = st.session_state[f"{kind}_model_digest"]
    st.session_state[f"{kind}_model_path"] = None
    st.session_state[f"{kind}_model_name"] = None
    st.session_state[f"{kind}_model_digest"] = None
    if kind == "board":
        _invalidate_after(2)
    elif kind == "component":
        st.session_state.pt_model_trusted = False
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
        or manifest.get("schema_version") != "pcb-component-classifier/1.0"
    ):
        raise ValueError(
            "Manifest không đúng schema pcb-component-classifier/1.0 của bước 6.1."
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
    _invalidate_after(5)
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
    _invalidate_after(5)
    st.session_state.messages.append(f"Đã nạp manifest 6.2: {upload.name}")


def _remove_solder_model() -> None:
    for key in (
        "solder_model_path", "solder_model_name", "solder_model_digest",
        "solder_manifest_path", "solder_manifest_name", "solder_manifest_digest",
    ):
        st.session_state[key] = None
    st.session_state.config["solder_grading"]["model_path"] = None
    st.session_state.config["solder_grading"]["manifest_path"] = None
    _invalidate_after(5)
    st.session_state.messages.append(
        "Đã gỡ model 6.2; bước này quay về chấm bằng luật đo."
    )


def _set_cad(upload: Any) -> None:
    """Accept a board CAD file and report what was parsed out of it.

    Parsing happens here rather than at run time so a wrong or unreadable file
    is caught while the operator is still looking at the sidebar.
    """

    if upload is None:
        return
    data = upload.getvalue()
    if not data or len(data) > 32 * 1024 * 1024:
        raise ValueError("File CAD rỗng hoặc vượt quá 32 MB.")
    digest = _digest(data)
    if digest == st.session_state.ignored_uploads.get("cad"):
        return
    if digest == st.session_state.cad_digest:
        return
    path = _materialize_upload(upload.name, data)
    side = st.session_state.config["cad"].get("side") or None
    try:
        board = load_cad(
            path,
            fmt=st.session_state.config["cad"].get("fmt", "auto"),
            units=st.session_state.config["cad"].get("units", "mm"),
            side=side,
        )
    except CadError as exc:
        st.session_state.ignored_uploads["cad"] = digest
        raise ValueError(str(exc)) from exc

    st.session_state.cad_path = path
    st.session_state.cad_name = upload.name
    st.session_state.cad_digest = digest
    st.session_state.cad_summary = {
        "format": board.source_format,
        "components": len(board.components),
        "pads": board.pad_count,
        "with_pads": sum(1 for item in board.components if item.has_pads),
        "side": side or "cả hai mặt",
    }
    st.session_state.config["cad"]["path"] = path
    st.session_state.ignored_uploads["cad"] = None
    _invalidate_after(4)
    st.session_state.messages.append(
        f"Đã nạp CAD {upload.name}: {len(board.components)} linh kiện, "
        f"{board.pad_count} pad."
    )


def _remove_cad() -> None:
    for key in ("cad_path", "cad_name", "cad_digest", "cad_summary"):
        st.session_state[key] = None
    st.session_state.config["cad"]["path"] = None
    _invalidate_after(4)
    st.session_state.messages.append("Đã gỡ file CAD; bước 5.5 quay lại ROI suy ra.")


def _set_cad_registration(upload: Any) -> None:
    """Reuse a registration measured earlier for this SKU and fixture."""

    if upload is None:
        return
    data = upload.getvalue()
    if not data or len(data) > 256 * 1024:
        raise ValueError("File registration rỗng hoặc quá lớn.")
    digest = _digest(data)
    if digest == st.session_state.ignored_uploads.get("cad_registration"):
        return
    try:
        CadRegistration.from_dict(json.loads(data.decode("utf-8")))
    except (UnicodeError, json.JSONDecodeError, CadError, KeyError, TypeError, ValueError) as exc:
        st.session_state.ignored_uploads["cad_registration"] = digest
        raise ValueError(f"registration.json không hợp lệ: {exc}") from exc
    path = _materialize_upload(upload.name, data)
    st.session_state.cad_registration_path = path
    st.session_state.cad_registration_name = upload.name
    st.session_state.config["cad"]["registration_path"] = path
    st.session_state.ignored_uploads["cad_registration"] = None
    _invalidate_after(4)
    st.session_state.messages.append(f"Đã nạp CAD registration: {upload.name}")


def _remove_cad_registration() -> None:
    st.session_state.cad_registration_path = None
    st.session_state.cad_registration_name = None
    st.session_state.config["cad"]["registration_path"] = None
    _invalidate_after(4)


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


def _get_bridge() -> PipelineBridge:
    config_json = json.dumps(st.session_state.config, sort_keys=True, ensure_ascii=False)
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
    _execute_solder(bridge)
    return crops


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
        return None
    try:
        result = bridge.make_solder_crops(source, detection_result.detections)
    except Exception as exc:
        st.session_state.solder_result = None
        st.session_state.messages.append(
            f"Bước 5.5 lỗi: {type(exc).__name__}: {exc}"
        )
        return None
    st.session_state.solder_result = result
    st.session_state.messages.append(f"Bước 5.5: {result.message}")
    return result


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
    for index, (step, label, callback) in enumerate(stages, start=1):
        progress.progress((index - 1) / len(stages), text=f"Bước {step}/6.1 · {label}")
        try:
            callback(bridge)
        except Exception as exc:
            st.session_state.statuses[step] = "error"
            st.session_state.messages.append(f"Bước {step} lỗi: {type(exc).__name__}: {exc}")
            progress.empty()
            st.error(f"Dừng ở bước {step}: {exc}")
            return
    final_step = 6 if classifier_ready else 5
    progress.progress(1.0, text=f"Hoàn tất workflow 0–{final_step}")
    st.session_state.active_step = final_step
    st.session_state.pending_navigation = final_step
    if classifier_ready:
        st.toast("Đã chạy xong workflow đến bước 6.1.", icon="✅")
    else:
        st.info(
            "Đã chạy xong bước 5. Bước 6.1 đang chờ best.onnx và "
            "model_manifest.json từ notebook train."
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
        st.markdown('<div class="sidebar-section-label">WORKFLOW · 0–6.1</div>', unsafe_allow_html=True)
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
        if "sidebar_step_navigation" not in st.session_state:
            st.session_state.sidebar_step_navigation = st.session_state.active_step
        elif st.session_state.sidebar_step_navigation not in step_options:
            st.session_state.sidebar_step_navigation = st.session_state.active_step
        selected = st.radio(
            "Mở bước",
            options=step_options,
            # Session State is the single source of truth. Passing an ``index``
            # as well would make Streamlit warn that the widget has two defaults.
            index=None,
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

        with st.expander("Model phân loại 6.1", expanded=True):
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

        with st.expander("Model kiểm tra mối hàn 6.2 (tuỳ chọn)", expanded=False):
            st.caption(
                "Chưa có model thì bước 6.2 vẫn chấm bằng luật đo hình học. Nạp "
                "cả hai file do training/train_solder_classifier.py xuất để bật "
                "thêm tầng phân loại."
            )
            solder_model_upload = st.file_uploader(
                "Model (.onnx)",
                type=["onnx"],
                key="solder_model_uploader",
            )
            if solder_model_upload is not None:
                try:
                    _set_solder_model(solder_model_upload)
                except ValueError as exc:
                    st.error(str(exc))
            solder_manifest_upload = st.file_uploader(
                "Contract (model_manifest.json)",
                type=["json"],
                key="solder_manifest_uploader",
                help="Bắt buộc: quy định thứ tự class, tiền xử lý và ngưỡng quyết định.",
            )
            if solder_manifest_upload is not None:
                try:
                    _set_solder_manifest(solder_manifest_upload)
                except ValueError as exc:
                    st.error(str(exc))

            has_model = bool(st.session_state.solder_model_name)
            has_manifest = bool(st.session_state.solder_manifest_name)
            if has_model and has_manifest:
                st.success(f"6.2: {st.session_state.solder_model_name}")
                if st.button("Gỡ model 6.2", key="remove_solder_model", width="stretch"):
                    _remove_solder_model()
                    st.rerun()
            elif has_model or has_manifest:
                st.warning(
                    "Cần đủ cả .onnx và model_manifest.json; thiếu một trong hai thì "
                    "bước 6.2 vẫn chạy bằng luật."
                )
            else:
                st.caption("Chưa có model · bước 6.2 chấm bằng luật đo")

        with st.expander("Sơ đồ CAD (tuỳ chọn)", expanded=False):
            st.caption(
                "Chưa có CAD thì bước 5.5 vẫn chạy bằng ROI suy ra. Nạp file vào "
                "đây để hợp nhất toạ độ land thật với hình học đó."
            )
            cad_config = st.session_state.config["cad"]
            side_options = ["top", "bottom", "both"]
            current_side = cad_config.get("side") or "both"
            chosen_side = st.selectbox(
                "Mặt board đang soi",
                side_options,
                index=side_options.index(current_side) if current_side in side_options else 0,
                key="cad_side_select",
            )
            cad_config["side"] = None if chosen_side == "both" else chosen_side
            cad_upload = st.file_uploader(
                "Board CAD",
                type=["csv", "txt", "ipc", "d356", "json"],
                key="cad_uploader",
                help=(
                    "Bảng pad, file pick-and-place (centroid), IPC-D-356A, hoặc "
                    "cad_json đã lưu. Định dạng được nhận dạng tự động."
                ),
            )
            if cad_upload is not None:
                try:
                    _set_cad(cad_upload)
                except ValueError as exc:
                    st.error(str(exc))
            summary = st.session_state.cad_summary
            if summary:
                st.success(f"CAD: {st.session_state.cad_name}")
                st.caption(
                    f"{summary['format']} · {summary['components']} linh kiện · "
                    f"{summary['pads']} pad · {summary['with_pads']} linh kiện có land · "
                    f"mặt {summary['side']}"
                )
                if summary["pads"] == 0:
                    st.info(
                        "File chỉ có vị trí đặt, không có land. Bước 5.5 sẽ dựng lại "
                        "ROI suy ra trên tâm và góc xoay của CAD."
                    )
                if st.button("Gỡ CAD", key="remove_cad", width="stretch"):
                    _remove_cad()
                    st.rerun()
            else:
                st.caption("Chưa có CAD · bước 5.5 chỉ dùng ROI suy ra")

            registration_upload = st.file_uploader(
                "Registration đã lưu (JSON)",
                type=["json"],
                key="cad_registration_uploader",
                help=(
                    "Ma trận CAD→ảnh đo một lần cho mỗi SKU/đồ gá. Không có thì app "
                    "tự căn theo detection của từng ảnh."
                ),
            )
            if registration_upload is not None:
                try:
                    _set_cad_registration(registration_upload)
                except ValueError as exc:
                    st.error(str(exc))
            if st.session_state.cad_registration_name:
                st.success(f"Registration: {st.session_state.cad_registration_name}")
                if st.button("Gỡ registration", key="remove_cad_reg", width="stretch"):
                    _remove_cad_registration()
                    st.rerun()
            else:
                cad_config["auto_register"] = st.checkbox(
                    "Tự căn CAD theo detection",
                    value=bool(cad_config.get("auto_register", True)),
                )

        st.markdown('<div class="security-note"><b>Lưu ý model</b><br>.pt có thể chứa pickle. Chỉ mở weight do bạn tự train hoặc nguồn tin cậy; ưu tiên ONNX khi trao đổi.</div>', unsafe_allow_html=True)
        quick_run = st.button(
            "▶  Chạy pipeline 0–6.1",
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
    _, name, description, code = STEP_DEFINITIONS[step]
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
                    st.error(resolution_issue)
                elif st.button("Nạp ảnh này vào pipeline", type="primary", width="stretch"):
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
                st.error(active_issue)
            else:
                st.success("Ảnh đã sẵn sàng cho bước 1.")
            if not active_issue and st.button("Đi đến bước 1 →", width="stretch"):
                st.session_state.active_step = 1
                st.session_state.pending_navigation = 1
                st.rerun()


def _render_step_one() -> None:
    _render_step_heading(1)
    if st.session_state.input_image is None:
        _render_empty("Thiếu ảnh đầu vào", "Quay lại bước 0 và import một ảnh PCB.")
        return
    config = st.session_state.config["preprocess"]
    has_calibration = isinstance(config.get("calibration_profile"), dict)
    if not has_calibration:
        # A disabled checkbox with only a tooltip reads like a broken feature.
        # It is not: undistort needs a profile measured from the real camera,
        # and there is no honest default for that.
        st.info(
            "**Sửa méo ống kính đang tắt vì chưa có profile hiệu chỉnh** — đây là "
            "trạng thái bình thường, không phải lỗi. Mọi tuỳ chọn còn lại của bước 1 "
            "vẫn dùng được.\n\n"
            "Để bật: chụp 15–25 ảnh bàn cờ bằng **đúng** camera/lens/tiêu cự sẽ dùng "
            "cho AOI, chạy `scripts/calibrate_camera.py` để tạo file `.json`, rồi tải "
            "file đó ở sidebar **Camera calibration**. Xem mục *Hiệu chỉnh méo ống "
            "kính camera* trong README."
        )
    control_col, preview_col = st.columns([0.85, 2.15], gap="large")
    with control_col:
        st.markdown("#### Recipe tiền xử lý")
        with st.form("preprocess_form"):
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
        st.info(
            "**Bước 2 cần một Golden Image thì nút căn chỉnh mới bật** — đây là "
            "trạng thái bình thường, không phải lỗi, và **không cần camera**: bất kỳ "
            "ảnh nào của một board đạt chuẩn cùng loại đều dùng được.\n\n"
            "Tải ảnh đó ở sidebar **Golden Image / Reference**. Ảnh chuẩn phải chụp "
            "cùng camera, lens và recipe ánh sáng với ảnh kiểm.\n\n"
            "Không có Golden Image thì bấm **Bỏ qua căn chỉnh**: pipeline vẫn chạy "
            "hết bước 3–6.1, chỉ là toạ độ không được đưa về hệ của board chuẩn."
        )
    controls, preview = st.columns([0.85, 2.15], gap="large")
    with controls:
        st.markdown("#### Feature matching")
        st.caption(
            "Phương pháp: **ORB + homography**, ECC affine làm fallback. Đây là "
            "phương pháp duy nhất được nối vào core nên không có gì để chọn; SIFT "
            "chưa được nối."
        )
        with st.form("alignment_form"):
            features = st.slider("Số feature tối đa", 500, 8000, int(config["features"]), 500)
            match_ratio = st.slider("Lowe ratio", 0.50, 0.95, float(config["match_ratio"]), 0.01)
            ransac = st.slider("RANSAC threshold", 1.0, 10.0, float(config["ransac_threshold"]), 0.5)
            has_reference = st.session_state.reference_image is not None
            submitted = st.form_submit_button(
                "Căn chỉnh với reference",
                type="primary",
                width="stretch",
                disabled=not has_reference,
                help=(
                    None
                    if has_reference
                    else "Tải Golden Image ở sidebar để bật nút này."
                ),
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
            gallery_tab, solder_tab, export_tab = st.tabs(
                ["Crop gallery", "ROI mối hàn (6.2)", "Export package"]
            )
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
            with solder_tab:
                _render_solder_rois()
            with export_tab:
                _render_exports()


SOLDER_ROI_COLORS = {
    "joint": (0, 200, 255),  # BGR amber
    "body": (255, 170, 0),   # BGR blue
}

# Provenance colours, so a glance at the overlay says which ROIs rest on CAD
# land coordinates and which were inferred from the detector box alone.
SOLDER_SOURCE_COLORS = {
    "cad+derived": (80, 220, 80),   # BGR green: both sources agreed
    "cad": (255, 120, 255),         # BGR magenta: CAD only
    "derived": (0, 200, 255),       # BGR amber: detector geometry only
}

CAD_SEVERITY_ICONS = {"defect": "🔴", "review": "🟠", "info": "🔵"}

SOLDER_MIN_READABLE_PX = 24


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


def _solder_frame(crops: list[SolderCropRecord]) -> pd.DataFrame:
    return pd.DataFrame(
        [
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
                "designator": crop.designator or "",
                "pin": crop.pin or "",
                "net": crop.net or "",
                "defect_class": "",
            }
            for crop in crops
        ]
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
                "split_pins": split_pins,
                "include_body_view": include_body,
                "terminal_outer_ratio": terminal_outer,
                "lead_outer_ratio": lead_outer,
                "target_size": joint_size,
            }
        )
        _execute_solder(_get_bridge())
        st.rerun()


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


VERDICT_DECISION_COLORS = {
    "accept": (80, 200, 80),   # BGR green
    "review": (0, 170, 255),   # BGR orange
    "reject": (40, 40, 230),   # BGR red
}

VERDICT_DECISION_ICONS = {"accept": "🟢", "review": "🟠", "reject": "🔴"}


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
        _render_empty(
            "Chưa chấm được mối hàn",
            "Bước 6.2 chạy cùng bước 5.5. Nếu trống, hãy chạy lại bước 5.",
        )
        return

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

        overlay_tab, gallery_tab, grade_tab, table_tab, cad_tab = st.tabs(
            [
                "ROI overlay",
                "Joint gallery",
                "Chấm mối hàn (6.2)",
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
                        _show_image(crop.image)
                        tag = f" · {crop.designator}" if crop.designator else ""
                        st.caption(f"**{crop.label}**{tag}\n\n{crop.position}")
        with table_tab:
            frame = _solder_frame(result.crops)
            st.dataframe(frame, width="stretch", height=320)
            st.caption(
                "Cột defect_class để trống chính là chỗ gán nhãn cho bước 6.2. "
                "Hình học đã giải quyết xong nên gán nhãn chỉ còn là phán quyết "
                "theo từng dòng."
            )
            st.download_button(
                "Tải solder_joints.csv",
                frame.to_csv(index=False).encode("utf-8-sig"),
                file_name="solder_joints.csv",
                mime="text/csv",
            )
        with grade_tab:
            _render_solder_grading(result)
        with cad_tab:
            _render_cad_panel(result)


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


DECISION_OVERLAY_COLORS = {
    "accept": (60, 180, 75),    # BGR green
    "review": (0, 165, 255),    # BGR orange
    "unknown": (32, 32, 220),   # BGR red
}
DECISION_OVERLAY_DEFAULT_COLOR = (200, 200, 200)


def _draw_classification_overlay(
    image: np.ndarray,
    detections: list[DetectionRecord],
    classifications: list[ClassificationRecord],
) -> np.ndarray:
    """Draw bbox + family/decision label per classified crop, same visual
    language (rectangle + filled label strip) as the step-4 detector overlay."""

    overlay = image.copy()
    detection_by_id = {item.detection_id: item for item in detections}
    for item in classifications:
        detection = detection_by_id.get(item.detection_id)
        if detection is None:
            continue
        x1, y1, x2, y2 = detection.bbox
        color = DECISION_OVERLAY_COLORS.get(item.decision, DECISION_OVERLAY_DEFAULT_COLOR)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        label = f"{item.family} {item.probability:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
        )
        label_top = max(0, y1 - text_height - baseline - 4)
        cv2.rectangle(
            overlay,
            (x1, label_top),
            (x1 + text_width + 6, label_top + text_height + baseline + 4),
            color,
            -1,
        )
        cv2.putText(
            overlay,
            label,
            (x1 + 3, label_top + text_height + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return overlay


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
        result = st.session_state.classification_result
        if not isinstance(result, ClassificationResult):
            _render_empty(
                "Chưa có kết quả phân loại",
                "Nạp model và manifest rồi chạy bước 6.1. Nhãn detector không được dùng thay thế.",
            )
            return
        items = result.classifications
        detection_result = st.session_state.detection_result
        source_image = _analysis_image()
        overlay_image = None
        if source_image is not None and isinstance(detection_result, DetectionResult) and items:
            overlay_image = _draw_classification_overlay(
                source_image, detection_result.detections, items
            )
        overlay_tab, table_tab, stats_tab, review_tab = st.tabs(
            ["Classification overlay", "Classification table", "Family stats", "Review queue"]
        )
        with overlay_tab:
            if overlay_image is not None:
                _show_image(overlay_image, "Family + decision theo từng detection")
            else:
                _render_empty(
                    "Chưa có overlay",
                    "Cần cả bbox từ bước 4 và kết quả phân loại để vẽ overlay.",
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
        if overlay_image is not None:
            st.download_button(
                "Tải ảnh classified PNG",
                _encode_png(overlay_image),
                file_name=f"{_safe_name(st.session_state.input_name or 'pcb')}_classified.png",
                mime="image/png",
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
          <span>AOI PCB WORKBENCH · STEPS 0–6.1</span>
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
        6: _render_step_six,
    }
    renderers[st.session_state.active_step]()
    _render_activity_log()
    _render_footer()


if __name__ == "__main__":
    main()