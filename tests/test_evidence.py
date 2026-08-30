"""Keeping defect coordinates instead of defect pictures.

The saving is real but modest -- 7.1 MB of crops against 1.39 MB of numbers per
board, measured on this project's own board. What makes the trade worth taking
is that the numbers are searchable and the pictures are not, and that cutting a
crop from a frame already in memory costs 0.004 ms against 219 ms to rebuild
that frame.

So the risk is not size, it is *correctness*: coordinates recorded against one
preprocessing recipe point at different pixels under another, and an operator
shown the wrong pixels is worse off than one shown nothing. Most of what is
tested here is refusal.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from aoi_pipeline.reporting.evidence import (
    DefectEvidence,
    EvidenceBundle,
    EvidenceMismatch,
    EvidenceViewer,
    SourceFingerprint,
    build_evidence_bundle,
    file_digest,
)


@pytest.fixture
def board(tmp_path: Path):
    """A source file plus the frame it is expected to rebuild into."""

    rng = np.random.default_rng(5)
    frame = rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)
    path = tmp_path / "board.png"
    cv2.imwrite(str(path), frame)
    stored = cv2.imread(str(path))
    return path, stored


def _bundle(path: Path, frame: np.ndarray, boxes) -> EvidenceBundle:
    height, width = frame.shape[:2]
    return EvidenceBundle(
        board_id="b1",
        source=SourceFingerprint(
            path=str(path), sha256=file_digest(path),
            analysis_width=width, analysis_height=height,
            preprocess={"clahe": True},
        ),
        defects=[
            DefectEvidence(joint_id=f"j{index}", detection_id="d0", label="insufficient",
                           decision="review", bbox=box)
            for index, box in enumerate(boxes)
        ],
    )


def test_a_bundle_survives_a_round_trip_through_disk(board, tmp_path) -> None:
    path, frame = board
    bundle = _bundle(path, frame, [(10, 12, 40, 38)])
    saved = bundle.save(tmp_path / "evidence.json")
    again = EvidenceBundle.load(saved)

    assert again.board_id == bundle.board_id
    assert again.source.sha256 == bundle.source.sha256
    assert [d.bbox for d in again.defects] == [(10, 12, 40, 38)]


def test_the_crop_is_the_same_pixels_the_verdict_was_measured_on(board) -> None:
    """The whole point: coordinates plus the source must reproduce the picture."""

    path, frame = board
    bundle = _bundle(path, frame, [(10, 12, 40, 38)])
    viewer = EvidenceViewer(lambda p: cv2.imread(str(p)))

    patch = viewer.crop(bundle, "j0")
    assert np.array_equal(patch, frame[12:38, 10:40])


def test_one_frame_is_held_and_reused_across_crops(board) -> None:
    """An operator opens a board and looks through its defects. Rebuilding per
    defect would cost about 59,000 times more than cutting from a held frame."""

    path, frame = board
    bundle = _bundle(path, frame, [(5, 5, 25, 25), (40, 30, 70, 60), (80, 70, 110, 100)])

    rebuilds = []

    def rebuild(source):
        rebuilds.append(source)
        return cv2.imread(str(source))

    viewer = EvidenceViewer(rebuild)
    for index in range(3):
        assert viewer.crop(bundle, f"j{index}").size > 0
    assert len(rebuilds) == 1, f"dựng lại {len(rebuilds)} lần, phải là 1"


def test_releasing_lets_the_memory_go(board) -> None:
    path, frame = board
    bundle = _bundle(path, frame, [(5, 5, 25, 25)])
    viewer = EvidenceViewer(lambda p: cv2.imread(str(p)))

    viewer.crop(bundle, "j0")
    assert viewer.loaded_board is not None
    viewer.release()
    assert viewer.loaded_board is None


def test_a_changed_source_file_is_refused_not_guessed(board) -> None:
    """The dangerous case. The path still resolves and the crop would still
    produce *a* picture -- of something else."""

    path, frame = board
    bundle = _bundle(path, frame, [(10, 12, 40, 38)])
    cv2.imwrite(str(path), np.zeros_like(frame))       # cùng tên, khác nội dung

    viewer = EvidenceViewer(lambda p: cv2.imread(str(p)))
    with pytest.raises(EvidenceMismatch, match="digest"):
        viewer.crop(bundle, "j0")


def test_a_frame_of_a_different_size_is_refused(board) -> None:
    """Preprocessing changed between inspection and viewing: same file, same
    digest, different pixels under the same coordinates."""

    path, frame = board
    bundle = _bundle(path, frame, [(10, 12, 40, 38)])
    viewer = EvidenceViewer(lambda p: cv2.resize(cv2.imread(str(p)), (80, 60)))

    with pytest.raises(EvidenceMismatch, match="tiền xử lý"):
        viewer.crop(bundle, "j0")


def test_a_missing_source_says_so_plainly(board, tmp_path) -> None:
    path, frame = board
    bundle = _bundle(path, frame, [(10, 12, 40, 38)])
    path.unlink()

    viewer = EvidenceViewer(lambda p: cv2.imread(str(p)))
    with pytest.raises(EvidenceMismatch, match="ảnh gốc"):
        viewer.crop(bundle, "j0")


def test_an_unknown_joint_id_is_a_key_error_not_a_blank_image(board) -> None:
    path, frame = board
    bundle = _bundle(path, frame, [(10, 12, 40, 38)])
    viewer = EvidenceViewer(lambda p: cv2.imread(str(p)))
    with pytest.raises(KeyError):
        viewer.crop(bundle, "khong-ton-tai")


def test_only_the_calls_worth_opening_are_kept(board) -> None:
    """``accept`` rows would grow the record by roughly a third for lines nobody
    opens."""

    from types import SimpleNamespace

    path, frame = board
    verdicts = [
        SimpleNamespace(joint_id="a", detection_id="d", label="good", decision="accept",
                        bbox=(1, 1, 5, 5), component_label="r", reasons=[], features={}),
        SimpleNamespace(joint_id="b", detection_id="d", label="insufficient",
                        decision="review", bbox=(6, 6, 12, 12), component_label="r",
                        reasons=["solder_ratio 0.11"], features={"solder_ratio": 0.11}),
        SimpleNamespace(joint_id="c", detection_id="d", label="bridge", decision="reject",
                        bbox=(20, 20, 30, 30), component_label="r", reasons=[], features={}),
    ]
    bundle = build_evidence_bundle("b1", path, frame.shape[:2], verdicts,
                                   preprocess={"clahe": True})
    assert sorted(d.joint_id for d in bundle.defects) == ["b", "c"]
    assert bundle.defects[0].reasons == ["solder_ratio 0.11"]
    assert bundle.source.analysis_width == frame.shape[1]


def test_an_old_schema_is_refused_rather_than_read_loosely(board, tmp_path) -> None:
    path, frame = board
    payload = _bundle(path, frame, [(1, 1, 5, 5)]).to_dict()
    payload["schema_version"] = "aoi-defect-evidence/0.9"
    target = tmp_path / "old.json"
    target.write_text(__import__("json").dumps(payload), encoding="utf-8")

    with pytest.raises(EvidenceMismatch, match="schema"):
        EvidenceBundle.load(target)
