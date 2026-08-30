"""Bản ghi đánh giá model: lưu toạ độ, và từ chối khi không chắc.

Giá trị của tính năng này nằm ở chỗ nó **tích luỹ** — vài chục lần người vận
hành nói "model sai ở đây" thì mới thành bằng chứng. Nên thứ phải đúng trước
tiên không phải giao diện mà là tầng lưu trữ: một dòng hỏng không được làm mất
cả lịch sử, hai phiên ghi cùng lúc không được đè nhau, và một bản ghi phải còn
đọc được sau khi model nó nói tới đã bị thay.
"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pytest

from aoi_pipeline.reporting.evidence import EvidenceMismatch, EvidenceViewer
from aoi_pipeline.modelops.model_feedback import (
    ERROR_KINDS,
    MAX_COMMENT_CHARS,
    SCHEMA_VERSION,
    FeedbackEntry,
    FeedbackError,
    FeedbackTargetRef,
    ModelIdentity,
    append_feedback,
    entries_for_source,
    error_label,
    evidence_bundle_for,
    feedback_root,
    group_by_model,
    load_feedback,
    preprocess_identity,
)

DIGEST = hashlib.sha256(b"board").hexdigest()
OTHER_DIGEST = hashlib.sha256(b"khac").hexdigest()


def _identity(**kwargs) -> ModelIdentity:
    base = dict(slot="component", kind="detector", loaded=True,
                name="detector/best.onnx", origin="active",
                sha256="a" * 64, version="v1", architecture="yolo26s",
                created="2026-08-17")
    base.update(kwargs)
    return ModelIdentity(**base)


def _entry(**kwargs) -> FeedbackEntry:
    base = dict(
        stage="detection", step=4, bbox=(10, 20, 50, 60), error_kind="missed",
        model=_identity(), source_name="board.png", source_sha256=DIGEST,
        analysis_width=160, analysis_height=120,
    )
    base.update(kwargs)
    return FeedbackEntry(**base)


# --------------------------------------------------------------------------
# Vòng đời một bản ghi
# --------------------------------------------------------------------------


def test_an_entry_survives_a_round_trip(tmp_path) -> None:
    original = _entry(comment="con trở ở góc dưới", expected_label="resistor",
                      target=FeedbackTargetRef(record_type="detection",
                                               record_id="d007",
                                               model_label="capacitor",
                                               model_probability=0.61))
    append_feedback(original, path=tmp_path / "log.jsonl")
    entries, problems = load_feedback(root=tmp_path)

    assert problems == []
    assert len(entries) == 1
    back = entries[0]
    assert back.bbox == (10, 20, 50, 60)
    assert all(isinstance(value, int) for value in back.bbox)
    assert back.comment == original.comment
    assert back.target.model_probability == pytest.approx(0.61)
    assert back.model.sha256 == original.model.sha256


def test_a_multiline_comment_stays_on_one_physical_line(tmp_path) -> None:
    """Bất biến của JSON Lines. Đây là thứ hỏng đầu tiên nếu ai đó đổi sang
    `json.dumps(..., indent=2)` cho dễ đọc."""

    comment = 'dòng một\ndòng hai có "nháy" và dấu tiếng Việt: ệ ữ ỹ'
    target = tmp_path / "log.jsonl"
    append_feedback(_entry(comment=comment), path=target)

    assert len(target.read_text(encoding="utf-8").splitlines()) == 1
    entries, _ = load_feedback(root=tmp_path)
    assert entries[0].comment == comment


def test_a_truncated_last_line_does_not_destroy_the_history(tmp_path) -> None:
    """Tiến trình bị giết giữa lúc ghi. Hai bản ghi trước đó phải còn nguyên."""

    target = tmp_path / "log.jsonl"
    append_feedback(_entry(comment="một"), path=target)
    append_feedback(_entry(comment="hai"), path=target)
    with target.open("a", encoding="utf-8") as handle:
        handle.write('{"schema_version": "aoi-model-fee')

    entries, problems = load_feedback(root=tmp_path)
    assert [entry.comment for entry in entries] == ["một", "hai"]
    assert len(problems) == 1 and "log.jsonl:3" in problems[0]


def test_a_duplicated_line_from_a_merge_is_counted_once(tmp_path) -> None:
    """Git merge kiểu union có thể nhân đôi dòng; uuid4 làm điều đó vô hại."""

    target = tmp_path / "log.jsonl"
    entry = _entry()
    append_feedback(entry, path=target)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(entry.to_json_line() + "\n")

    entries, problems = load_feedback(root=tmp_path)
    assert len(entries) == 1
    assert problems == []


def test_a_different_schema_version_is_reported_not_coerced(tmp_path) -> None:
    payload = _entry().to_dict()
    payload["schema_version"] = "aoi-model-feedback/0.9"
    (tmp_path / "log.jsonl").write_text(
        json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")

    entries, problems = load_feedback(root=tmp_path)
    assert entries == []
    assert problems and "0.9" in problems[0]


def test_concurrent_appends_do_not_lose_or_corrupt_entries(tmp_path) -> None:
    """Lý do chọn JSON Lines thay vì một mảng JSON.

    Tám phiên ghi cùng lúc. Với một mảng JSON đây là đọc-sửa-ghi và sẽ mất bản
    ghi trong im lặng; với chế độ nối thêm thì hệ điều hành đặt con trỏ ở cuối
    file ngay lúc ghi.
    """

    target = tmp_path / "log.jsonl"

    def write(index: int) -> None:
        for number in range(25):
            append_feedback(_entry(comment=f"{index}-{number}"), path=target)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(write, range(8)))

    entries, problems = load_feedback(root=tmp_path)
    assert problems == [], problems[:3]
    assert len(entries) == 200


# --------------------------------------------------------------------------
# Từ chối những gì không chắc
# --------------------------------------------------------------------------


def test_an_error_kind_from_another_stage_is_refused() -> None:
    """Một lỗi gõ ở giao diện không được đẻ ra loại lỗi không đếm được."""

    with pytest.raises(FeedbackError, match="không thuộc bước"):
        _entry(stage="detection", error_kind="escape")


def test_an_empty_or_flipped_box_is_refused() -> None:
    with pytest.raises(FeedbackError, match="rỗng hoặc lật"):
        _entry(bbox=(50, 20, 10, 60))
    with pytest.raises(FeedbackError, match="rỗng hoặc lật"):
        _entry(bbox=(10, 20, 10, 60))


def test_an_oversized_comment_is_refused_at_the_storage_boundary() -> None:
    """Giới hạn của widget chỉ là gợi ý; đây mới là bất biến của file."""

    with pytest.raises(FeedbackError, match="quá"):
        _entry(comment="x" * (MAX_COMMENT_CHARS + 1))


def test_a_source_digest_that_is_not_a_digest_is_refused() -> None:
    with pytest.raises(FeedbackError, match="sha256"):
        _entry(source_sha256="khong-phai-digest")


def test_the_record_leaks_no_absolute_workstation_path() -> None:
    """`AGENTS.md` cấm export đường dẫn tuyệt đối, và file này được commit.

    App cũng không hề ghi ảnh gốc ra đĩa, nên một trường `path` sẽ là bịa đặt
    trên mọi máy — kể cả máy vừa ghi nó.
    """

    line = _entry(comment="C:/thu/muc/nao/do trong ghi chú thì tuỳ người dùng").to_json_line()
    payload = json.loads(line)
    payload.pop("comment")
    serialised = json.dumps(payload)

    assert "\\\\" not in serialised
    for pattern in (":\\", ":/", "/home/", "/tmp/", "AppData"):
        assert pattern not in serialised, f"rò rỉ đường dẫn: {pattern}"
    assert "path" not in payload["source"]


# --------------------------------------------------------------------------
# Danh tính model
# --------------------------------------------------------------------------


def test_the_shipped_manifests_all_yield_a_digest() -> None:
    """`compare_key` chỉ có giá trị khi manifest thật sự khai sha256."""

    from aoi_pipeline.modelops.model_registry import discover_models

    for entry in discover_models():
        assert entry.summary().sha256, f"{entry.name} không khai sha256"


def test_two_models_judged_on_the_same_record_stay_apart(tmp_path) -> None:
    """Đây là điều tính năng này tồn tại để làm được: so hai model trên cùng
    những lỗi đã báo."""

    target = tmp_path / "log.jsonl"
    ref = FeedbackTargetRef(record_type="detection", record_id="d007")
    append_feedback(_entry(target=ref, model=_identity(sha256="a" * 64)), path=target)
    append_feedback(_entry(target=ref, model=_identity(sha256="b" * 64)), path=target)

    entries, _ = load_feedback(root=tmp_path)
    grouped = group_by_model(entries)
    assert set(grouped) == {"a" * 64, "b" * 64}
    assert all(len(group) == 1 for group in grouped.values())


def test_an_entry_recorded_with_no_model_is_still_valid(tmp_path) -> None:
    """Bước 4 chạy CV demo vẫn đáng ghi nhận; `runtime_mode` giữ ngữ cảnh để
    sau này lọc ra."""

    identity = ModelIdentity(slot="component", kind="detector", loaded=False)
    assert identity.compare_key == "no-model"
    assert identity.display == "chưa nạp model"

    append_feedback(_entry(model=identity, runtime_mode="CV DEMO"),
                    path=tmp_path / "log.jsonl")
    entries, problems = load_feedback(root=tmp_path)
    assert problems == []
    assert entries[0].model.loaded is False
    assert entries[0].runtime_mode == "CV DEMO"


# --------------------------------------------------------------------------
# Cắt lại pixel — dùng lại nguyên bộ máy của evidence.py
# --------------------------------------------------------------------------


@pytest.fixture
def board(tmp_path):
    rng = np.random.default_rng(3)
    frame = rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)
    path = tmp_path / "board.png"
    cv2.imwrite(str(path), frame)
    stored = cv2.imread(str(path))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return path, stored, digest


def test_the_crop_is_the_pixels_the_error_was_reported_on(board) -> None:
    path, frame, digest = board
    entry = _entry(source_sha256=digest, bbox=(10, 12, 40, 38),
                   analysis_width=160, analysis_height=120)
    bundle = evidence_bundle_for([entry], path)

    viewer = EvidenceViewer(lambda p: cv2.imread(str(p)))
    patch = viewer.crop(bundle, entry.entry_id)
    assert np.array_equal(patch, frame[12:38, 10:40])


def test_a_changed_source_is_refused_through_the_same_path(board) -> None:
    """Không viết lại logic từ chối — thừa hưởng nguyên từ `evidence.py`, nên
    không có bản sao thứ hai để mà lệch nhau."""

    path, frame, digest = board
    entry = _entry(source_sha256=digest, analysis_width=160, analysis_height=120)
    bundle = evidence_bundle_for([entry], path)
    cv2.imwrite(str(path), np.zeros_like(frame))

    viewer = EvidenceViewer(lambda p: cv2.imread(str(p)))
    with pytest.raises(EvidenceMismatch, match="digest"):
        viewer.crop(bundle, entry.entry_id)


def test_entries_from_two_different_images_cannot_share_a_bundle(board) -> None:
    path, _, digest = board
    with pytest.raises(FeedbackError, match="nhiều ảnh"):
        evidence_bundle_for(
            [_entry(source_sha256=digest), _entry(source_sha256=OTHER_DIGEST)], path)


# --------------------------------------------------------------------------
# Phần còn lại
# --------------------------------------------------------------------------


def test_every_stage_has_a_vocabulary_with_unique_codes() -> None:
    assert set(ERROR_KINDS) == {"detection", "classification", "solder"}
    for stage, kinds in ERROR_KINDS.items():
        codes = [code for code, _ in kinds]
        assert len(codes) == len(set(codes)), f"{stage} có mã trùng"
        assert all(label.strip() for _, label in kinds)


def test_the_escape_kind_is_worded_as_the_severe_one() -> None:
    """Board xấu lọt lưới là lỗi nặng nhất trong AOI; nhãn phải nói ra điều đó
    chứ không nằm lẫn giữa các loại khác."""

    assert "BỎ SÓT LỖI" in error_label("solder", "escape")


def test_preprocess_identity_keeps_the_pixels_not_the_camera(tmp_path) -> None:
    """`calibration_profile` là nội tại của một camera cụ thể — không nên nằm
    trong file được commit. Digest vẫn đổi khi nó đổi."""

    config = {"clahe": True, "max_side": 4096,
              "calibration_profile": {"camera_matrix": [[1, 2], [3, 4]]}}
    kept = preprocess_identity(config)

    assert "calibration_profile" not in kept
    assert kept["clahe"] is True and kept["max_side"] == 4096
    other = preprocess_identity({**config, "calibration_profile": None})
    assert kept["sha256"] != other["sha256"], "digest phải đổi khi cấu hình đổi"


def test_entries_are_filtered_to_the_image_in_front_of_you(tmp_path) -> None:
    target = tmp_path / "log.jsonl"
    append_feedback(_entry(source_sha256=DIGEST), path=target)
    append_feedback(_entry(source_sha256=OTHER_DIGEST), path=target)
    append_feedback(_entry(source_sha256=DIGEST, stage="classification",
                           step=6, error_kind="wrong_family"), path=target)

    entries, _ = load_feedback(root=tmp_path)
    assert len(entries_for_source(entries, DIGEST)) == 2
    assert len(entries_for_source(entries, DIGEST, stage="detection")) == 1


def test_the_log_directory_can_be_moved_out_of_the_repo(monkeypatch, tmp_path) -> None:
    """Đọc biến môi trường lúc GỌI, không lúc import — AppTest chạy trong cùng
    tiến trình nên `monkeypatch.setenv` phải có tác dụng."""

    monkeypatch.setenv("AOI_FEEDBACK_DIR", str(tmp_path / "o-noi-khac"))
    assert feedback_root() == tmp_path / "o-noi-khac"
    written = append_feedback(_entry())
    assert written.is_absolute()
    assert (tmp_path / "o-noi-khac") in written.parents


def test_a_missing_directory_reads_as_empty_not_as_an_error(tmp_path) -> None:
    entries, problems = load_feedback(root=tmp_path / "chua-ai-ghi-gi")
    assert entries == [] and problems == []
